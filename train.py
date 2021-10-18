import json

from model import SentenceClassification
from data_reader import SentenceClassificationDataset
import torch
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import Dataset,DataLoader
import torch.nn as nn
from transformers import BertTokenizer,BertModel,AlbertTokenizer,AlbertModel,RobertaTokenizer,RobertaModel,GPT2Tokenizer,GPT2Model,GPTNeoModel
import numpy as np
from tqdm import tqdm
import os,random,time,logging,sys
import argparse
import itertools

THC_CACHING_ALLOCATOR=0
logging.basicConfig(level=logging.INFO)

model_map_all={
    "albert":{
        "model":AlbertModel,
        "tokenizer":AlbertTokenizer
    },
    "bert":{
        "model":BertModel,
        "tokenizer":BertTokenizer
    },
    "roberta":{
        "model":RobertaModel,
        "tokenizer":RobertaTokenizer
    },
    "gpt2":{
        "model":GPT2Model,
        "tokenizer":GPT2Tokenizer
    },
    "gpt-neo":{
        "model":GPTNeoModel,
        "tokenizer":GPT2Tokenizer
    }
}

def setup_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def init_model_config(args):
    with open("model_config.json") as in_fp:
        model_config=json.load(in_fp)
    config=model_config[args.model]
    config["model"]=model_map_all[args.model]["model"]
    config["tokenizer"]=model_map_all[args.model]["tokenizer"]
    return config

class Trainer():
    def __init__(self,args,config,total_step):
        self.config=config
        self.model=SentenceClassification(config,args)
        self.model.to(args.device)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
        {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
        self.bert_optimizer = AdamW(optimizer_grouped_parameters,lr=args.lr)
        self.bert_scheduler = get_linear_schedule_with_warmup(self.bert_optimizer, num_warmup_steps=0.1*total_step,
                                                    num_training_steps=total_step)
        self.best_accuracy=0
    
    def train_on_epoch(self,train_dataloader,cur_epoch,args):
        self.model.train()
        total_loss=0
        count=0
        loop=tqdm(train_dataloader,desc="train {}".format(cur_epoch))
        for input_ids,token_type_ids,attention_mask,labels in loop:
            logits,loss=self.model(input_ids.to(args.device),token_type_ids.to(args.device),attention_mask.to(args.device),labels.to(args.device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.max_grad_norm)
            self.bert_optimizer.step()
            self.bert_scheduler.step()
            self.bert_optimizer.zero_grad()
            total_loss+=loss.item()
            count+=1
            loop.set_postfix(loss=total_loss/count)
    
    def evaluate(self,dev_dataloader,cur_epoch,args):
        self.model.eval()
        with torch.no_grad():
            all_logits=[]
            all_labels=[]
            for input_ids,token_type_ids,attention_mask,labels in tqdm(dev_dataloader,desc="eval"):
                logits=self.model(input_ids.to(args.device),token_type_ids.to(args.device),attention_mask.to(args.device))
                all_logits.extend(logits.cpu().numpy().tolist())
                all_labels.extend(labels.numpy().tolist())
            all_logits=np.asarray(all_logits)
            all_labels=np.asarray(all_labels).reshape(-1)
            predict_label=np.argmax(all_logits,axis=1).reshape(-1)
            cur_acc=np.sum(all_labels==predict_label)/len(all_labels)
            if(cur_acc>self.best_accuracy and args.train):
                torch.save({'state_dict': self.model.state_dict(), 'epoch': cur_epoch}, args.save_model_path+".pth")
                self.best_accuracy=cur_acc
            if(not args.train):
                np.save(args.predict_save+"_logits.npy",all_logits)
                np.save(args.predict_save+"_predict_labels.npy",predict_label)
                np.save(args.predict_save+"_oringin_labels.npy",all_labels)
            logging.info("epoch:{} | accuracy {} | best accuracy {} ".format(cur_epoch,cur_acc,self.best_accuracy))
    
    def train(self,train_dataloader,dev_dataloader,args):
        for epoch in range(args.num_epochs):
            self.train_on_epoch(train_dataloader,epoch,args)
            self.evaluate(dev_dataloader,epoch,args)
    
def main(args):
    config=init_model_config(args)
    print(config)
    tokenizer=config["tokenizer"].from_pretrained(config["model_path"])
    if(args.model in ["gpt2","gpt-neo"]):
        tokenizer.pad_token = tokenizer.eos_token
    total_step=0
    if(args.train):
        train_data=SentenceClassificationDataset(args.train_file,tokenizer,args)
        total_step=len(train_data)//args.batch_size*args.num_epochs
        dev_dataloader=DataLoader(SentenceClassificationDataset(args.dev_file,tokenizer,args),batch_size=args.batch_size)
        train_dataloader=DataLoader(train_data,shuffle=True,batch_size=args.batch_size)
    trainer=Trainer(args,config,total_step)
    test_dataloader=DataLoader(SentenceClassificationDataset(args.test_file,tokenizer,args),batch_size=args.batch_size)
    # sys.exit()
    if(args.train):
        trainer.train(train_dataloader,dev_dataloader,args)
    args.train=False
    checkpoint=torch.load(os.path.join(args.save_model_path+".pth"))
    trainer.model.load_state_dict(checkpoint["state_dict"])
    trainer.evaluate(test_dataloader,0,args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data",help="the data dir of train/valid/test data")
    parser.add_argument("--model_dir", type=str, default="./model",help="the model checkpoint will save in this dir")
    parser.add_argument("--predict_dir", type=str, default="./predict_result",help="the predict result will save in this dir")
    parser.add_argument("--train_file", type=str, default="train.txt",help="the train data file")
    parser.add_argument("--dev_file", type=str, default="dev.txt")
    parser.add_argument("--test_file", type=str, default="test.txt")
    parser.add_argument("--device", type=str,default="cpu",help="the device, can be cpu/cuda")
    parser.add_argument("--model", type=str,required=True,help="the lm use to train, can be bert/roberta/albert/gpt2")
    parser.add_argument("--save_model_path",type=str,required=True,help="the name of save model checkpoint, model will save in model dir")
    parser.add_argument("--train",action='store_true',help="the train flag, if set True, will train model, if not set, will evaluate in the test data")
    parser.add_argument("--num_classes",type=int,default=4,help="the num of the classes in the dataset")
    
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--weight_decay", default=0.01, type=float)
    parser.add_argument("--batch_size", type=int, default=16,
            help="number of hidden units")
    parser.add_argument("--lr", type=float, default=1e-5,
            help="learning rate")
    parser.add_argument("--max_length",type=int,default=256)
    args = parser.parse_args()
    args.train_file=os.path.join(args.data_dir,args.train_file)
    args.dev_file=os.path.join(args.data_dir,args.dev_file)
    args.test_file=os.path.join(args.data_dir,args.test_file)
    args.save_model_path="{}_{}_lr_{}_batch_{}_maxlen_{}".format(args.save_model_path,args.model,args.lr,args.batch_size,args.max_length)
    args.predict_save=os.path.join(args.predict_dir,args.save_model_path)
    args.save_model_path=os.path.join(args.model_dir,args.save_model_path)
    if(args.model in ["bert","albert"]):
        args.use_token_type_id=True
    else:
        args.use_token_type_id=False
    setup_seed(111)
    print(args)
    main(args)