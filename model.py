from transformers import BertTokenizer, BertModel
import torch.nn as nn
import torch

class SentenceClassification(nn.Module):
    def __init__(self,config,args):
        super(SentenceClassification,self).__init__()
        self.model=config["model"].from_pretrained(config["model_path"])
        self.loss_fct=nn.CrossEntropyLoss()
        self.classifier=nn.Linear(config["hidden_dim"],args.num_classes)
        self.model_type=args.model
        self.use_token_type_id=args.use_token_type_id
        self.pad_id=config["pad_token_id"]
    
    def forward(self,input_ids,token_type_ids,attention_mask,labels=None):
        if(self.use_token_type_id):
            outputs=self.model(input_ids,token_type_ids=token_type_ids,attention_mask=attention_mask)
        else:
            outputs = self.model(input_ids,attention_mask=attention_mask,return_dict=True)
        if(self.model_type not in ["gpt2","gpt-neo"]):
            pooled_output = outputs[1]
        else:
            batch_size, sequence_length = input_ids.shape[:2]
            pooled_output=outputs[0]
            sequence_lengths = torch.ne(input_ids, self.pad_id).sum(-1) - 1
            pooled_output = pooled_output[range(batch_size), sequence_lengths]
        logits=self.classifier(pooled_output)
        if(labels is not None):
            loss=self.loss_fct(logits,labels.view(-1))
            logits=torch.softmax(logits,dim=-1)
            return logits,loss
        else:
            logits=torch.softmax(logits,dim=-1)
            return logits