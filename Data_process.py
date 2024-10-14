# First step is to import the needed libraries
from tqdm.cli import main
import torch,os
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
import pickle
from tqdm import tqdm
from sklearn.metrics import f1_score,accuracy_score
import math
import re
from torch.utils.data import Dataset, DataLoader
import logging
from tqdm.contrib.logging import logging_redirect_tqdm
from config import *

#this function converts tokens to ids and then to a tensor
def prepare_sequence(seq, to_ix):
    idxs = list(map(lambda w: to_ix[w] if w in to_ix.keys() else to_ix["<UNK>"], seq))
    tensor = Variable(torch.LongTensor(idxs)).cuda() if USE_CUDA else Variable(torch.LongTensor(idxs))
    return tensor
# this function turns class text to id
def prepare_intent(intent, to_ix):
    idxs = to_ix[intent] if intent in to_ix.keys() else to_ix["UNKNOWN"]
    return idxs
# converts numbers to <NUM> TAG
def number_to_tag(txt):
    return "<NUM>" if txt.isdecimal() else txt

# Here we remove multiple spaces and punctuation which cause errors in tokenization for bert & elmo.
def remove_punc(mlist):
    mlist = [re.sub(" +"," ",t.split("\t")[0][4:-4]) for t in mlist] # remove spaces down to 1
    temp_train_tokens = []
    # punct remove example:  play samuel-el jackson from 2009 - 2010 > play samuelel jackson from 2009 - 2010
    for row in mlist:
        tokens = row.split(" ")
        newtokens = []
        for token in tokens:
            newtoken = re.sub(r"[.,'\"\\/\-:&’—=–官方杂志¡…“”~%]",r"",token) # remove punc
            newtoken = re.sub(r"[楽園追放�]",r"A",newtoken)
            newtokens.append(newtoken if len(token)>1 else token)
        if newtokens[-1]=="":
            newtokens.pop(-1)
        if newtokens[0]=="":
            newtokens.pop(0)
        temp_train_tokens.append(" ".join(newtokens))
    return temp_train_tokens
# this function returns the main tokens so that we can apply tagging on them. see original paper.
def get_subtoken_mask(current_tokens,bert_tokenizer):
    temp_mask = []
    for i in current_tokens:
        temp_row_mask = []
        temp_row_mask.append(False) # for cls token
        temp = bert_tokenizer.tokenize(i)
        for j in temp:
            temp_row_mask.append(j[:2]!="##")
        while len(temp_row_mask)<LENGTH:
            temp_row_mask.append(False)
        temp_mask.append(temp_row_mask)
        if sum(temp_row_mask)!=len(i.split(" ")):
            print(f"inconsistent:{temp}")
            print(i)
            print(sum(temp_row_mask))
            print(len(i.split(" ")))
    return torch.tensor(temp_mask).cuda()

flatten = lambda l: [number_to_tag(item) for sublist in l for item in sublist]

def tokenize_dataset(dataset_address):
    # added tokenizer and tokens for
    from transformers import AutoTokenizer
    bert_tokenizer = AutoTokenizer.from_pretrained(ENV_BERT_ADDR)
    # bert_tokenizer = torch.hub.load(ENV_BERT_ADDR, 'tokenizer', ENV_BERT_ADDR,verbose=False,source="local")#38toks snips,52Atis
    ##open database and read line by line
    dataset = open(dataset_address,"r").readlines()
    print("example input:"+dataset[0])
    ##remove last character of lines -\n- in train file
    dataset = [t[:-1] for t in dataset]
    #converts string to array of tokens + array of tags + target intent [array with x=3 and y dynamic]
    dataset_tokens = remove_punc(dataset)
    dataset_subtoken_mask = get_subtoken_mask(dataset_tokens,bert_tokenizer)
    dataset_toks = bert_tokenizer.batch_encode_plus(dataset_tokens,max_length=LENGTH,add_special_tokens=True,return_tensors='pt'
                                                  ,return_attention_mask=True , padding='max_length',truncation=True)
    dataset = [[re.sub(" +"," ",t.split("\t")[0]).split(" "),t.split("\t")[1].split(" ")[:-1],t.split("\t")[1].split(" ")[-1]] for t in dataset]
    #removes BOS, EOS from array of tokens and tags
    dataset = [[t[0][1:-1],t[1][1:],t[2]] for t in dataset]
    return dataset, dataset_subtoken_mask,dataset_toks

#defining datasets.
def remove_values_from_list(the_list, val):
   return [value for value in the_list if value != val]

class NLUDataset(Dataset):
    def __init__(self, sin,sout,intent,input_ids,attention_mask,token_type_ids,subtoken_mask):
        self.sin = [prepare_sequence(temp,word2index) for temp in sin]
        self.sout = [prepare_sequence(temp,tag2index) for temp in sout]
        self.intent = Variable(torch.LongTensor([prepare_intent(temp,intent2index) for temp in intent])).cuda()
        self.input_ids=input_ids.cuda()
        self.attention_mask=attention_mask.cuda()
        self.token_type_ids=token_type_ids.cuda()
        self.subtoken_mask=subtoken_mask.cuda()
        self.x_mask = [Variable(torch.BoolTensor(tuple(map(lambda s: s ==0, t )))).cuda() for t in self.sin]
    def __len__(self):
        return len(self.intent)
    def __getitem__(self, idx):
        sample = self.sin[idx],self.sout[idx],self.intent[idx],self.input_ids[idx],self.attention_mask[idx],self.token_type_ids[idx],self.subtoken_mask[idx],self.x_mask[idx]
        return sample


# we put all tags inside of the batch in a flat array for F1 measure.
# we use masking so that we only non PAD tokens are counted in f1 measurement



def add_paddings(seq_in,seq_out):
    sin=[]
    sout=[]
    for i in range(len(seq_in)):
        temp = seq_in[i]
        if len(temp)<LENGTH:
            while len(temp)<LENGTH:
                temp.append('<PAD>')
        else:
            temp = temp[:LENGTH]
        temp = ['[cls]'] + temp[:LENGTH-1]
        sin.append(temp)
        # add padding inside output tokens
        temp = seq_out[i]
        if len(temp)<LENGTH:
            while len(temp)<LENGTH:
                temp.append('<PAD>')
        else:
            temp = temp[:LENGTH]
        temp = ['O'] + temp[:LENGTH-1]
        sout.append(temp)
    return sin,sout

def process(datasets):
    if datasets == "atis":
        dataset_train = atis_train_dev
        dataset_test = atis_test
    else:
        dataset_train = snips_train_dev
        dataset_test = snips_test
    train,train_subtoken_mask,train_toks = tokenize_dataset(dataset_train)
    test, test_subtoken_mask, test_toks = tokenize_dataset(dataset_test)
    #convert above array to separate lists
    seq_in,seq_out, intent = list(zip(*train))
    seq_in_test,seq_out_test, intent_test = list(zip(*test.copy()))
    # Create Sets of unique tokens
    vocab = set(flatten(seq_in))           #len = 722
    slot_tag = set(flatten(seq_out))       # len = 120
    intent_tag = set(intent)
    # adds paddings
    sin=[] #padded input tokens
    sout=[] # padded output translated tags
    sin_test=[] #padded input tokens
    sout_test=[] # padded output translated tags
    ## adds padding inside input tokens
    sin,sout=add_paddings(seq_in,seq_out)
    sin_test,sout_test=add_paddings(seq_in_test,seq_out_test)

    # making dictionary (token:id), initial value
    global word2index, tag2index, intent2index
    word2index = {'<PAD>': 0, '<UNK>':1,'<BOS>':2,'<EOS>':3,'<NUM>':4,'[cls]':5}
    # add rest of token list to dictionary
    for token in vocab:
        if token not in word2index.keys():
            word2index[token]=len(word2index)
    #make id to token list ( reverse )
    index2word = {v:k for k,v in word2index.items()}

    # initial tag2index dictionary
    tag2index = {'<PAD>' : 0,'<BOS>':2,'<UNK>':1,'<EOS>':3,'[cls]':4}
    # add rest of tag tokens to list
    for tag in slot_tag:
        if tag not in tag2index.keys():
            tag2index[tag] = len(tag2index)
    # making index to tag
    index2tag = {v:k for k,v in tag2index.items()}

    #initialize intent to index
    intent2index={'UNKNOWN':0}
    for ii in intent_tag:
        if ii not in intent2index.keys():
            intent2index[ii] = len(intent2index)
    index2intent = {v:k for k,v in intent2index.items()}

    train_data=NLUDataset(sin,sout,intent,train_toks['input_ids'],train_toks['attention_mask'],train_toks['token_type_ids'],train_subtoken_mask)
    test_data=NLUDataset(sin_test,sout_test,intent_test,test_toks['input_ids'],test_toks['attention_mask'],test_toks['token_type_ids'],test_subtoken_mask)
    train_data = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_data = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

    data_dict ={
        "train_data": train_data,
        "test_data": test_data,
        "tag2index": tag2index,
        "intent2index": intent2index,
        "test": test,
        "test_subtoken_mask": test_subtoken_mask,
        "test_toks": test_toks,
        "word2index": word2index,
        "index2word": index2word,
        "index2tag": index2tag,
        "index2intent": index2intent        
    }

    return data_dict