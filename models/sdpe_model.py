import math
from transformers import AutoModelForPreTraining, AutoModelForMaskedLM
import torch
import torch.nn as nn
import torch.nn.functional as F 


class SDPE(nn.Module):

    def __init__(self, init_model, max_len, max_step, k=1) -> None:
        super().__init__()
        if "bert-base" in init_model:
            self.model = AutoModelForMaskedLM.from_pretrained(init_model)
            freezed_w = [self.model.bert.embeddings.token_type_embeddings.weight,self.model.bert.embeddings.word_embeddings.weight] #self.model.bert.embeddings.LayerNorm.weight, self.model.bert.embeddings.LayerNorm.bias
        else:
            self.model = AutoModelForPreTraining.from_pretrained(init_model)
            freezed_w = [self.model.cls.seq_relationship.bias, self.model.cls.seq_relationship.weight, self.model.bert.pooler.dense.bias, self.model.bert.pooler.dense.weight, self.model.bert.embeddings.token_type_embeddings.weight,self.model.bert.embeddings.word_embeddings.weight] #self.model.bert.embeddings.LayerNorm.weight, self.model.bert.embeddings.LayerNorm.bias
        self.max_len = max_len
        self.max_step = max_step
        self.k=k
        self.time_embed = nn.Embedding(max_step,self.model.config.hidden_size)
        #self.layernorm = nn.LayerNorm(self.model.config.hidden_size, eps=self.model.config.layer_norm_eps)
        for p in  freezed_w:
            p.requires_grad = False
        nn.init.constant_(self.time_embed.weight, 0)

    def forward(self,input_ids,token_type_ids,attention_mask, labels, labels_token_type_ids, labels_attention_mask):
        t = self.max_step
        input_shape = input_ids.size()
        seq_length = input_shape[1]
        device = input_ids.device
        
        position_ids = self.model.bert.embeddings.position_ids[:, 0 : seq_length]
        position_embeddings = self.model.bert.embeddings.position_embeddings(position_ids)

        # Trial 31:
        output_shape = labels.size()
        out_seq_length = output_shape[1]
        N = input_shape[0]
        # outpos_ids = self.model.bert.embeddings.position_ids[:, 0 : out_seq_length]
        # out_pos_embeddings = self.model.bert.embeddings.position_embeddings(outpos_ids)

        with torch.no_grad():
            target_emb = self.model.bert.embeddings.word_embeddings(labels)
            inp_emb = self.model.bert.embeddings.word_embeddings(input_ids)
        #print(word_emb.shape)
            token_type_embeddings = self.model.bert.embeddings.token_type_embeddings(token_type_ids)
        # labels_token_type_embeddings = self.model.bert.embeddings.token_type_embeddings(labels_token_type_ids)
        
            xt = torch.normal(0,1,(N,self.max_len,self.model.config.hidden_size)).to(device) #/ math.sqrt(self.model.config.hidden_size)
            xt_token_type_ids = torch.zeros(N,self.max_len).long().to(device)
            attention_mask = torch.ones(N,self.max_len).long().to(device)
            extended_attention_mask = self.model.bert.get_extended_attention_mask(attention_mask, attention_mask.shape)
            xt_token_type_embeddings = self.model.bert.embeddings.token_type_embeddings(token_type_ids)
        xt_position_ids = self.model.bert.embeddings.position_ids[:, 0 : self.max_len]
        xt_position_embeddings = self.model.bert.embeddings.position_embeddings(position_ids)
        
        loss_x0 = None
        for t in range(self.max_step-1,0,-1):
            # print("Step", t)
            diffusion_steps = torch.ones(size = (output_shape[0],),device=input_ids.device).long()*t
            time_embedding = self.time_embed(diffusion_steps).unsqueeze(1)

            model_input = inp_emb+xt+position_embeddings+xt_position_embeddings+time_embedding
            model_input = self.model.bert.embeddings.LayerNorm(model_input)
            #denoise
            encoder_outputs = self.model.bert.encoder(
                model_input,
                attention_mask=extended_attention_mask,
                head_mask=[None] * self.model.config.num_hidden_layers
            )
            sequence_output = encoder_outputs[0]
            prediction_scores = self.model.cls.predictions(sequence_output)

            #clamp
            # pred = torch.argmax(prediction_scores,-1).long()
            # denoised_word = self.model.bert.embeddings.word_embeddings(pred)
            denoised_word = prediction_scores.softmax(-1) @ self.model.bert.embeddings.word_embeddings.weight.unsqueeze(0)

            if loss_x0 == None:
            # loss_x0 = F.cross_entropy(prediction_scores.view(-1, self.model.config.vocab_size),labels.flatten(),ignore_index=0)
                loss_x0 = F.mse_loss(denoised_word, target_emb)
            else:
            #     loss_x0 += F.cross_entropy(prediction_scores.view(-1, self.model.config.vocab_size),labels.flatten(),ignore_index=0)
                loss_x0 += F.mse_loss(denoised_word, target_emb)
            #DDIM
            alpha_tk = 1 - math.sqrt((t)/self.max_step)#+1e-5
            alpha_t = 1 - math.sqrt((t+1)/self.max_step)+1e-5
            noise = (xt - math.sqrt(alpha_t)*denoised_word)/math.sqrt(1-alpha_t)
            xt = math.sqrt(alpha_tk)*(xt/math.sqrt(alpha_t) + (math.sqrt((1-alpha_tk)/alpha_tk) - math.sqrt((1-alpha_t)/alpha_t))*noise)
            #noisy_word = math.sqrt(alpha_tk)*denoised_word + math.sqrt(1-alpha_tk)*noise
        
        prediction_scores = self.model.cls.predictions(xt)
        loss_emb = F.mse_loss(xt, target_emb)
        loss_round = F.cross_entropy(prediction_scores.view(-1, self.model.config.vocab_size),labels.flatten(),ignore_index=0)
        
        loss = loss_x0 + loss_emb + loss_round
        #loss = F.smooth_l1_loss(sequence_output,word_emb)
        return loss, prediction_scores, labels
