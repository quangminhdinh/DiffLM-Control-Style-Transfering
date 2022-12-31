from torch.utils.data.dataset import Dataset
from transformers import AutoTokenizer


class e2e(Dataset): 
    def __init__(self,csv_dir,init_model,max_len):
        self.tokenizer = AutoTokenizer.from_pretrained(init_model)
        with open(csv_dir,'r') as f:
            story_teller = f.readlines()
            self.story = list(story_teller)
        self.max_len = max_len
        self.tokenizer.model_max_length = max_len
    def __getitem__(self, index):
        #index = 0
        #index = random.randint(0,9)
        story = self.story[index].split("||")[-1].strip()
        from_tokenizer = self.tokenizer(story,padding="max_length",truncation = True,return_tensors="pt")
        input_ids = from_tokenizer["input_ids"].squeeze_().long()
        token_type_ids = from_tokenizer["token_type_ids"].squeeze_().long()
        attention_mask = from_tokenizer["attention_mask"].squeeze_().long()
        return input_ids,token_type_ids,attention_mask
    def __len__(self):
        return len(self.story)
