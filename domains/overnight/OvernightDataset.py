from torch.utils.data.dataset import Dataset
from transformers import AutoTokenizer
from .config import DATA_CONFIG
import json
import numpy as np


DOMAINS = DATA_CONFIG['DOMAINS']


simplifier = lambda txt: txt.replace("edu.stanford.nlp.sempre.overnight.SimpleWorld.", "")


def get_data(domain, dataset="train_with_dev"):
    data = []
    with open(DATA_CONFIG['DIR'] + domain + "." + dataset + ".jsonl") as f:
        for line in f:
            record = json.loads(line)
            record["formula"] = simplifier(record["formula"])
            data.append(record)
    return data


def split_train_dev(domain, domains_data, train_size=200, remain_dev=0.2, shuffle=True):
  data = domains_data[domain]
  if shuffle:
    np.random.shuffle(data)
  size = len(data)
  dev_size = np.ceil((size - train_size) * 0.2).astype(int) + train_size
  return data[:train_size], data[train_size:dev_size]


def prepare_data(data, shuffle=True):
  inputs = []
  outputs = []
  for domain in DOMAINS:
    domain_data = data[domain]
    if shuffle:
      np.random.shuffle(domain_data)
    for record in domain_data:
      inputs.append(record['natural'])
      outputs.append(record['canonical'])
  return inputs, outputs


class OvernightDataset(Dataset): 

    def __init__(self, data, init_model, max_len, func=prepare_data):
        self.tokenizer = AutoTokenizer.from_pretrained(init_model)
        self.inputs, self.labels = func(data)
        self.max_len = max_len
        self.tokenizer.model_max_length = max_len

    def __getitem__(self, index):
        from_tokenizer = self.tokenizer(self.inputs[index],padding="max_length",truncation = True,return_tensors="pt")
        label_tokens = self.tokenizer(self.labels[index],padding="max_length",truncation = True,return_tensors="pt")
        input_ids = from_tokenizer["input_ids"].squeeze_().long()
        ret_labels = label_tokens["input_ids"].squeeze_().long()
        token_type_ids = from_tokenizer["token_type_ids"].squeeze_().long()
        attention_mask = from_tokenizer["attention_mask"].squeeze_().long()
        labels_token_type_ids = label_tokens["token_type_ids"].squeeze_().long()
        labels_attention_mask = label_tokens["attention_mask"].squeeze_().long()
        # return input_ids,token_type_ids,attention_mask
        return {
            "input_ids": input_ids, 
            "token_type_ids" : token_type_ids, 
            "attention_mask" : attention_mask, 
            "labels" : ret_labels, 
            "labels_token_type_ids" : labels_token_type_ids, 
            "labels_attention_mask" : labels_attention_mask
        }

    def __len__(self):
        return len(self.labels)


train_all = {}
test_all = {}
for domain in DOMAINS:
    train_all[domain] = get_data(domain)
    test_all[domain] = get_data(domain, dataset="test")

train_dict = {}
dev_dict = {}
for domain in DOMAINS:
    train_dict[domain], dev_dict[domain] = split_train_dev(domain, train_all)
