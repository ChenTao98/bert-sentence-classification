from torch.utils.data import Dataset
import numpy as np
import torch
class SentenceClassificationDataset(Dataset):
    def __init__(self,data_path,tokenizer,args):
        label_list,sentence_list=self.read_data(data_path)
        self.labels=torch.tensor(label_list,dtype=torch.long)
        tokenizer_out=tokenizer(sentence_list,max_length=args.max_length,padding=True,truncation=True,return_tensors="pt")
        self.input_ids=tokenizer_out["input_ids"]
        self.attention_mask=tokenizer_out["attention_mask"]
        if("token_type_ids" in tokenizer_out):
            self.token_type_ids=tokenizer_out["token_type_ids"]
        else:
            self.token_type_ids=torch.zeros(len(self.input_ids))
    def read_data(self,data_path):
        label_list,sentence_list=[],[]
        with open(data_path) as in_fp:
            for line in in_fp:
                label,sentence=line.strip().split("\t")
                label_list.append(int(label))
                sentence_list.append(sentence)
        return label_list,sentence_list
    
    def __getitem__(self, index):
        return self.input_ids[index],self.token_type_ids[index],self.attention_mask[index],self.labels[index]
    
    def __len__(self):
        return len(self.input_ids)
