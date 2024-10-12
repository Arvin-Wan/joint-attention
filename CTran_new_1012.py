
# First step is to import the needed libraries
import torch
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


# in this section we define static values and variables for ease of access and testing
_fn="final" # file unique id for saving and loading models
bert_base='./pretrain/bert-base-uncased/'
bert_large='./pretrain/bert-large-uncased/'
deberta_base = "./pretrain/deberta-v3-large"

snips_train="./dataset/snips_train.iob"
snips_test="./dataset/snips_test.iob"
snips_train_dev = "./dataset/snips_train_dev.iob"
atis_train="./dataset/atis-2.train.w-intent.iob"
atis_test="./dataset/atis-2.test.w-intent.iob"
atis_train_dev = "./dataset/atis-2.w-intent_train_dev.iob"
#ENV variables directly affect the model's behaviour
ENV_DATASET_TRAIN=atis_train_dev
ENV_DATASET_TEST=atis_test

ENV_EMBEDDING_SIZE=768  # dimention of embbeding, bertbase=768,bertlarge&elmo=1024
ENV_BERT_ADDR=bert_base
ENV_SEED=1331
ENV_HIDDEN_SIZE=ENV_EMBEDDING_SIZE
ENV_CNN_FILTERS=ENV_HIDDEN_SIZE // 4  #128
ENV_CNN_KERNELS=4
DEPTH = 4

#these are related to training
BATCH_SIZE=32
LENGTH=60
STEP_SIZE=50

# you must use cuda to run this code. if this returns false, you can not proceed.
USE_CUDA = torch.cuda.is_available()
torch.manual_seed(ENV_SEED)
if USE_CUDA:
    print("You are using cuda. Good!")
    torch.cuda.manual_seed(ENV_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
else:
    print('You are NOT using cuda! Some problems may occur.')
np.random.seed(ENV_SEED)
random.seed(ENV_SEED)

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
train,train_subtoken_mask,train_toks = tokenize_dataset(ENV_DATASET_TRAIN)
test, test_subtoken_mask, test_toks = tokenize_dataset(ENV_DATASET_TEST)

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
        # temp = ['[cls]'] + temp[:LENGTH-1]
        sin.append(temp)
        # add padding inside output tokens
        temp = seq_out[i]
        if len(temp)<LENGTH:
            while len(temp)<LENGTH:
                temp.append('<PAD>')
        else:
            temp = temp[:LENGTH]
        # temp = ['O'] + temp[:LENGTH-1]
        sout.append(temp)
    return sin,sout
sin,sout=add_paddings(seq_in,seq_out)
sin_test,sout_test=add_paddings(seq_in_test,seq_out_test)


# making dictionary (token:id), initial value
word2index = {'<PAD>': 0, '<UNK>':1,'<BOS>':2,'<EOS>':3,'<NUM>':4}
# add rest of token list to dictionary
for token in vocab:
    if token not in word2index.keys():
        word2index[token]=len(word2index)
#make id to token list ( reverse )
index2word = {v:k for k,v in word2index.items()}

# initial tag2index dictionary
tag2index = {'<PAD>' : 0,'<BOS>':2,'<UNK>':1,'<EOS>':3}
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
#making single list
train_data=NLUDataset(sin,sout,intent,train_toks['input_ids'],train_toks['attention_mask'],train_toks['token_type_ids'],train_subtoken_mask)
test_data=NLUDataset(sin_test,sout_test,intent_test,test_toks['input_ids'],test_toks['attention_mask'],test_toks['token_type_ids'],test_subtoken_mask)
train_data = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_data = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)


# we put all tags inside of the batch in a flat array for F1 measure.
# we use masking so that we only non PAD tokens are counted in f1 measurement
def mask_important_tags(predictions,tags,masks):
    result_tags=[]
    result_preds=[]
    for pred,tag,mask in zip(predictions.tolist(),tags.tolist(),masks.tolist()):
        #index [0] is to get the data
        for p,t,m in zip(pred,tag,mask):
            if not m:
                result_tags.append(p)
                result_preds.append(t)
        #result_tags.pop()
        #result_preds.pop()
    return result_preds,result_tags


# generates transformer mask
def generate_square_subsequent_mask(sz: int) :
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
def generate_square_diagonal_mask(sz: int) :
    """Generates a matrix which there are zeros on diag and other indexes are -inf."""
    return torch.triu(torch.ones(sz,sz)-float('inf'), diagonal=1)+torch.tril(torch.ones(sz,sz)-float('inf'), diagonal=-1)
# positional embedding used in transformers
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"

#start of the shared encoder
class BertLayer(nn.Module):
    def __init__(self):
        super(BertLayer, self).__init__()
        from transformers import BertModel,AutoModel
        # self.bert_model = torch.hub.load(ENV_BERT_ADDR, 'model', ENV_BERT_ADDR,source="local")
        self.bert_model = BertModel.from_pretrained(ENV_BERT_ADDR)

    def forward(self, bert_info=None):
        (bert_tokens, bert_mask, bert_tok_typeid) = bert_info
        bert_encodings = self.bert_model(bert_tokens, bert_mask, bert_tok_typeid)
        bert_last_hidden = bert_encodings['last_hidden_state']
        bert_pooler_output = bert_encodings['pooler_output']
        return bert_last_hidden, bert_pooler_output
    
class SENET(nn.Module):
    def __init__(self, input_dim, ratio=4,p_dropout = 0.5):
        super(SENET,self).__init__()
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.linear1 = nn.Linear(input_dim, input_dim // ratio)
        self.linear2 = nn.Linear(input_dim // ratio, input_dim)
        self.activation = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p_dropout)

    def forward(self,x):
        batch_size, seq_len, dim = x.size()
        
        # x = torch.transpose(x,dim0=1,dim1=2)
        x1 = x
        x1 = self.avgpool(x1).view(batch_size, seq_len)
        x1 = self.activation(self.linear1(x1))
        x1 = self.dropout(x1)
        x1 = self.sigmoid(self.linear2(x1)).view(batch_size, seq_len, 1)

        return x * x1

class Encoder(nn.Module):
    def __init__(self, p_dropout=0.5):
        super(Encoder, self).__init__()
        self.filter_number = ENV_CNN_FILTERS
        self.kernel_number = ENV_CNN_KERNELS  # tedad size haye filter : 2,3,5 = 3
        self.embedding_size = ENV_EMBEDDING_SIZE
        # self.activation = nn.ReLU()
        self.activation = RMSNorm(ENV_CNN_FILTERS)
        self.softmax = nn.Softmax(dim=1)
        # self.linear1 = nn.Linear(self.embedding_size, self.filter_number)
        self.conv1 = nn.Conv1d(in_channels=self.embedding_size, out_channels=self.filter_number, kernel_size=(1,),
                               padding="same", padding_mode="zeros")
        self.conv2 = nn.Conv1d(in_channels=self.embedding_size, out_channels=self.filter_number, kernel_size=(3,),
                               padding="same", padding_mode="zeros")
        self.conv3 = nn.Conv1d(in_channels=self.embedding_size, out_channels=self.filter_number, kernel_size=(3,),
                               padding="same", padding_mode="zeros",dilation=3)

        self.conv4 = nn.Conv1d(in_channels=self.embedding_size, out_channels=self.filter_number, kernel_size=(5,),
                               padding="same", padding_mode="zeros")

        self.dropout = nn.Dropout(p_dropout)
        self.layer_norm = nn.LayerNorm(ENV_EMBEDDING_SIZE)
        self.senet = SENET(LENGTH)

    def forward(self, bert_last_hidden):
        trans_embedded = torch.transpose(bert_last_hidden, dim0=1, dim1=2)
        # linear1 = self.activation(self.linear1(bert_last_hidden))
        convolve1 = self.activation(torch.transpose(self.conv1(trans_embedded), dim0=1, dim1=2))
        convolve2 = self.activation(torch.transpose(self.conv2(trans_embedded), dim0=1, dim1=2))
        convolve3 = self.activation(torch.transpose(self.conv3(trans_embedded), dim0=1, dim1=2))
        convolve4 = self.activation(torch.transpose(self.conv4(trans_embedded), dim0=1, dim1=2))

        output = torch.cat((convolve1, convolve2,convolve3,convolve4), dim=2)

        output = self.layer_norm(self.dropout(output)) + bert_last_hidden
        output = self.senet(output)
        return output




from torch import nn

#start of the decoder
class Decoder(nn.Module):

    def __init__(self,slot_size,intent_size,dropout_p=0.7,ratio_squeeze= 4):
        super(Decoder, self).__init__()
        self.slot_size = slot_size
        self.intent_size = intent_size
        self.dropout_p = dropout_p
        self.softmax= nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        # Define the layers
        self.embedding = nn.Embedding(self.slot_size, ENV_HIDDEN_SIZE)
        self.activation = nn.ReLU()
        self.dropout1 = nn.Dropout(self.dropout_p)
        self.dropout2 = nn.Dropout(self.dropout_p)
        self.dropout3 = nn.Dropout(self.dropout_p)
        self.slot_trans = nn.Linear(ENV_HIDDEN_SIZE, self.slot_size)
        self.intent_out = nn.Linear(ENV_HIDDEN_SIZE,self.intent_size)
        self.intent_out_cls = nn.Linear(ENV_EMBEDDING_SIZE,self.intent_size) # dim of bert
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=ENV_HIDDEN_SIZE, nhead=4,batch_first=True,dim_feedforward=512 ,activation="relu")
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=4)
        self.transformer_mask = generate_square_subsequent_mask(LENGTH).cuda()
        self.transformer_diagonal_mask = generate_square_diagonal_mask(LENGTH).cuda()
        self.pos_encoder = PositionalEncoding(ENV_HIDDEN_SIZE, dropout=0.1)
        self.self_attention = nn.MultiheadAttention(embed_dim=ENV_HIDDEN_SIZE
                                                    ,num_heads=8,dropout=0.1
                                                    ,batch_first=True)
        self.layer_norm = nn.LayerNorm(ENV_HIDDEN_SIZE)

        self.dropout4 = nn.Dropout(self.dropout_p)
        self.squeeze_linear = nn.Linear(LENGTH, LENGTH //ratio_squeeze)
        self.activation2 = nn.ReLU()
        # self.activation2 = RMSNorm(LENGTH // ratio_squeeze)



    def forward(self, input,encoder_outputs,encoder_maskings,bert_subtoken_maskings=None,infer=False):
        # encoder outputs: BATCH,LENGTH,Dims (16,60,1024)
        batch_size = encoder_outputs.shape[0]
        length = encoder_outputs.size(1) #for every token in batches
        embedded = self.embedding(input)


        encoder_outputs2=torch.transpose(encoder_outputs,dim0=1,dim1=2)
        encoder_outputs2 = self.activation2(self.dropout4(self.squeeze_linear(encoder_outputs2)))
        encoder_outputs2 = torch.transpose(encoder_outputs2,dim0=1,dim1=2).mean(1)
        intent_score = self.intent_out(self.dropout1(encoder_outputs2))
        intent_score = F.log_softmax(intent_score,dim=1)   

        newtensor = torch.cuda.FloatTensor(batch_size, length,ENV_HIDDEN_SIZE).fill_(0.) # size of newtensor same as original
        for i in range(batch_size): # per batch
            newtensor_index=0
            for j in range(length): # for each token
                if bert_subtoken_maskings[i][j].item()==1:
                    newtensor[i][newtensor_index] = encoder_outputs[i][j]
                    newtensor_index+=1

        if infer==False:
            embedded=embedded*math.sqrt(ENV_HIDDEN_SIZE)
            embedded = self.pos_encoder(embedded)
            zol = self.transformer_decoder(tgt=embedded,memory=newtensor
                                           ,memory_mask=self.transformer_diagonal_mask
                                           ,tgt_mask=self.transformer_mask)

            scores = self.slot_trans(self.dropout3(zol))    # B, S, 124
            slot_scores = F.log_softmax(scores,dim=2)
        else:
            bos = Variable(torch.LongTensor([[tag2index['<BOS>']]*batch_size])).cuda().transpose(1,0)
            bos = self.embedding(bos)
            tokens=bos
            for i in range(length):
                temp_embedded=tokens*math.sqrt(ENV_HIDDEN_SIZE)
                temp_embedded = self.pos_encoder(temp_embedded)
                zol = self.transformer_decoder(tgt=temp_embedded,
                                               memory=newtensor,
                                               tgt_mask=self.transformer_mask[:i+1,:i+1],
                                               memory_mask=self.transformer_diagonal_mask[:i+1,:]
                                               )
                scores = self.slot_trans(self.dropout3(zol))
                softmaxed = F.log_softmax(scores,dim=2)
                #the last token is apended to vectors
                _,input = torch.max(softmaxed,2)
                newtok = self.embedding(input)
                tokens=torch.cat((bos,newtok),dim=1)
            slot_scores = softmaxed

        return slot_scores.view(input.size(0)*length,-1), intent_score


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.bert_layer = BertLayer()
        self.encoder = Encoder()
        self.decoder = Decoder(len(tag2index),len(intent2index))

    def forward(self,bert_tokens,bert_mask,bert_toktype,subtoken_mask,tag_target=None,infer=False):
        batch_size=bert_tokens.size(0)
        bert_hidden,_ = self.bert_layer(bert_info=(bert_tokens,bert_mask,bert_toktype))

        encoder_output = self.encoder(bert_last_hidden=bert_hidden)

        start_decode = Variable(torch.LongTensor([[tag2index['<BOS>']]*batch_size])).cuda().transpose(1,0)
        
        if not infer:
            start_decode = torch.cat((start_decode,tag_target[:,:-1]),dim=1)

        # tag score shape 960,124,  tag target 16,60
        # intent scoer shape 16,22,  intent target shape 16
        # tag_score, intent_score = self.decoder(start_decode,output,bert_mask==0,bert_subtoken_maskings=subtoken_mask,infer=infer)
        tag_score, intent_score = self.decoder(start_decode,encoder_output,bert_mask==0,bert_subtoken_maskings=subtoken_mask,infer=infer)

        return tag_score, intent_score



from crf import CRF
crf_log_likelihood = CRF(num_tags=len(tag2index),batch_first=True).cuda()
loss_function_1 = nn.CrossEntropyLoss(ignore_index=0)
loss_function_2 = nn.CrossEntropyLoss()
loss_function_3 = nn.L1Loss()

model = Model()
if USE_CUDA:
    model.cuda()
optimizer = optim.AdamW(model.parameters(), lr=0.0001,weight_decay=0.8)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.96)



def train_loop(model, train_data, postfix_dict):
    losses=[]
    loss_1es = []
    loss_2es = []
    id_precision=[]
    sf_f1=[]
    model.train()
    for i,(x,tag_target,intent_target,bert_tokens,bert_mask,bert_toktype,subtoken_mask,x_mask) in enumerate(train_data):
        batch_size=tag_target.size(0)
        tag_score, intent_score = model(bert_tokens,bert_mask,bert_toktype,subtoken_mask,tag_target)

        perd_intent = torch.argmax(intent_score,dim=1)
        # loss_1 = loss_function_1(tag_score,tag_target.view(-1))
        tag_scores = tag_score.view(batch_size, LENGTH, -1)
        loss_1 = - crf_log_likelihood.log_likelihood(tag_scores, tag_target, reduction="mean")
        loss_2 = loss_function_2(intent_score,intent_target)
        # loss_3 = loss_function_3(perd_intent.float(),intent_target)
        loss = loss_1 + loss_2
        # loss = loss_2 + loss_3

        loss_1es.append(loss_1.data.cpu().numpy() if USE_CUDA else loss_1.data.numpy()[0])
        loss_2es.append(loss_2.data.cpu().numpy() if USE_CUDA else loss_2.data.numpy()[0])
        losses.append(loss.data.cpu().numpy() if USE_CUDA else loss.data.numpy()[0])
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

        optimizer.step()
        id_precision.append(accuracy_score(intent_target.detach().cpu(),torch.argmax(intent_score,dim=1).detach().cpu()))
        pred_list,target_list=mask_important_tags(torch.argmax(tag_score,dim=1).view(batch_size,LENGTH),tag_target,x_mask)
        sf_f1.append(f1_score(pred_list,target_list,average="micro",zero_division=0))

        postfix_dict["T_L"] = round(float(np.mean(losses)),4)
        postfix_dict["SF_L"] = round(float(np.mean(loss_1es)),4)
        postfix_dict["ID_L"] = round(float(np.mean(loss_2es)),4)
        postfix_dict["SF_F1"] = round(float(np.mean(sf_f1)),3) * 100
        postfix_dict["ID_P"] = round(float(np.mean(id_precision)),3) * 100

    return postfix_dict


def eval_loop(model, test_data):
    postfix_dict = {}
    losses=[]
    loss_1es = []
    loss_2es = []
    id_precision=[]
    sf_f1=[]
    model.eval()
    with torch.no_grad(): # to turn off gradients computation
        for i,(x,tag_target,intent_target,bert_tokens,bert_mask,bert_toktype,subtoken_mask,x_mask) in enumerate(test_data):
            batch_size=tag_target.size(0)
            model.zero_grad()
            
            tag_score, intent_score = model(bert_tokens,bert_mask,bert_toktype,subtoken_mask,infer=True)
            tag_scores = tag_score.view(batch_size, LENGTH, -1)
            # loss_1 = loss_function_1(tag_score,tag_target.view(-1))
            loss_1 = - crf_log_likelihood.log_likelihood(tag_scores, tag_target, reduction="mean")
            loss_2 = loss_function_2(intent_score,intent_target)
            loss = loss_1 + loss_2
            loss_1es.append(loss_1.data.cpu().numpy() if USE_CUDA else loss_1.data.numpy()[0])
            loss_2es.append(loss_2.data.cpu().numpy() if USE_CUDA else loss_2.data.numpy()[0])
            losses.append(loss.data.cpu().numpy() if USE_CUDA else loss.data.numpy()[0])
            id_precision.append(accuracy_score(intent_target.detach().cpu(),torch.argmax(intent_score,dim=1).detach().cpu()))
            pred_list,target_list=mask_important_tags(torch.argmax(tag_score,dim=1).view(batch_size,LENGTH),tag_target,x_mask)
            sf_f1.append(f1_score(pred_list,target_list,average="micro",zero_division=0))

        postfix_dict["Te_L"] = round(float(np.mean(losses)),4)
        postfix_dict["Te_SF_L"] = round(float(np.mean(loss_1es)),4)
        postfix_dict["Te_ID_L"] = round(float(np.mean(loss_2es)),4)
        postfix_dict["Te_SF_F1"] = round(float(np.mean(sf_f1)),4) * 100
        postfix_dict["Te_ID_P"] = round(float(np.mean(id_precision)),4) * 100

    return postfix_dict, round(float(np.mean(sf_f1)),4), round(float(np.mean(id_precision)),4)


max_id_prec=0.
max_sf_f1=0.
max_id_prec_both=0.
max_sf_f1_both=0.

with tqdm(total=STEP_SIZE,desc="Epoch") as epoiter:
    for step in range(STEP_SIZE):
        postfix_dict = {}

        postfix_dict = train_loop(model, train_data, postfix_dict)
        eval_dict,sf_f1,id_precision = eval_loop(model, test_data)

        # Save Best
        max_sf_f1 = max_sf_f1 if sf_f1<=max_sf_f1 else sf_f1
        max_id_prec = max_id_prec if id_precision<=max_id_prec else id_precision
        if max_sf_f1_both<=sf_f1 and max_id_prec_both<=id_precision:
            max_sf_f1_both=sf_f1
            max_id_prec_both=id_precision
            torch.save(model,f"models/model.pkl")
        
        postfix_dict.update(eval_dict)
        postfix_dict["Best_ID_P"] = max_id_prec_both * 100
        postfix_dict["Best_SF_F1"] = max_sf_f1_both * 100
        
        scheduler.step()
        epoiter.set_postfix(postfix_dict)
        epoiter.update(1)
    
    print(f"max single ID PR: {max_id_prec}")
    print(f"max single SF F1: {max_sf_f1}")
    print(f"max mutual PR: {max_id_prec_both}   SF:{max_sf_f1_both}")



# max_id_prec=0.
# max_sf_f1=0.
# max_id_prec_both=0.
# max_sf_f1_both=0.


# with tqdm(total=STEP_SIZE,desc="Epoch") as epoiter:
#     for step in range(STEP_SIZE):
#         postfix_dict = {}
#         losses=[]
#         loss_1es = []
#         loss_2es = []
#         id_precision=[]
#         sf_f1=[]

#         ### TRAIN
#         encoder.train() # set to train mode
#         middle.train()
#         decoder.train()
#         bert_layer.train()
#         for i,(x,tag_target,intent_target,bert_tokens,bert_mask,bert_toktype,subtoken_mask,x_mask) in enumerate(train_data):
#             batch_size=tag_target.size(0)
#             bert_layer.zero_grad()
#             encoder.zero_grad()
#             middle.zero_grad()
#             decoder.zero_grad()
#             # bert_hidden shape 16,60,768
#             # bert pooler shape 16, 768
#             bert_hidden,bert_pooler = bert_layer(bert_info=(bert_tokens,bert_mask,bert_toktype))

#             # encoder output shape 16,60,512
#             encoder_output = encoder(bert_last_hidden=bert_hidden)

#             # output shape same as encoder output
#             output = middle(encoder_output,bert_mask==0,training=True)

#             # start decode shape 16,60
#             start_decode = Variable(torch.LongTensor([[tag2index['<BOS>']]*batch_size])).cuda().transpose(1,0)
#             start_decode = torch.cat((start_decode,tag_target[:,:-1]),dim=1)

#             # tag score shape 960,124,  tag target 16,60
#             # intent scoer shape 16,22,  intent target shape 16
#             tag_score, intent_score = decoder(start_decode,output,bert_mask==0,bert_subtoken_maskings=subtoken_mask)
#             loss_1 = loss_function_1(tag_target.view(-1),tag_score)

            
#             loss_2 = loss_function_2(intent_target,intent_score)
#             loss_1 = loss_function_1(tag_score,tag_target.view(-1))
#             # loss_2 = loss_function_2(intent_score,intent_target)
#             loss = loss_1+ loss_2
#             loss_1es.append(loss_1.data.cpu().numpy() if USE_CUDA else loss_1.data.numpy()[0])
#             loss_2es.append(loss_2.data.cpu().numpy() if USE_CUDA else loss_2.data.numpy()[0])
#             losses.append(loss.data.cpu().numpy() if USE_CUDA else loss.data.numpy()[0])
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(encoder.parameters(), 0.5)
#             torch.nn.utils.clip_grad_norm_(middle.parameters(), 0.5)
#             torch.nn.utils.clip_grad_norm_(decoder.parameters(), 0.5)
#             torch.nn.utils.clip_grad_norm_(bert_layer.parameters(), 0.5)
#             enc_optim.step()
#             mid_optim.step()
#             dec_optim.step()
#             ber_optim.step()
#             #print(bert_tokens[0])
#             #print(tag_target[0])
#             id_precision.append(accuracy_score(intent_target.detach().cpu(),torch.argmax(intent_score,dim=1).detach().cpu()))
#             pred_list,target_list=mask_important_tags(torch.argmax(tag_score,dim=1).view(batch_size,LENGTH),tag_target,x_mask)
#             sf_f1.append(f1_score(pred_list,target_list,average="micro",zero_division=0))

#         postfix_dict["T_L"] = round(float(np.mean(losses)),4)
#         postfix_dict["SF_L"] = round(float(np.mean(loss_1es)),4)
#         postfix_dict["ID_L"] = round(float(np.mean(loss_2es)),4)
#         postfix_dict["SF_F1"] = round(float(np.mean(sf_f1)),3)
#         postfix_dict["ID_P"] = round(float(np.mean(id_precision)),3)

#         losses=[]
#         loss_1es=[]
#         loss_2es=[]
#         sf_f1=[]
#         id_precision=[]
#         #scheduler.step()

#         #### TEST
#         encoder.eval() # set to test mode
#         middle.eval()
#         decoder.eval()
#         bert_layer.eval()
#         with torch.no_grad(): # to turn off gradients computation
#             for i,(x,tag_target,intent_target,bert_tokens,bert_mask,bert_toktype,subtoken_mask,x_mask) in enumerate(test_data):
#                 batch_size=tag_target.size(0)
#                 encoder.zero_grad()
#                 middle.zero_grad()
#                 decoder.zero_grad()
#                 bert_layer.zero_grad()
#                 bert_hidden,bert_pooler = bert_layer(bert_info=(bert_tokens,bert_mask,bert_toktype))
#                 encoder_output = encoder(bert_last_hidden=bert_hidden)
#                 output = middle(encoder_output,bert_mask==0,training=True)
#                 start_decode = Variable(torch.LongTensor([[tag2index['<BOS>']]*batch_size])).cuda().transpose(1,0)
#                 tag_score, intent_score = decoder(start_decode,output,bert_mask==0,bert_subtoken_maskings=subtoken_mask,infer=True)
#                 loss_1 = loss_function_1(tag_score,tag_target.view(-1))
#                 loss_2 = loss_function_2(intent_score,intent_target)
#                 loss = loss_1 + loss_2
#                 loss_1es.append(loss_1.data.cpu().numpy() if USE_CUDA else loss_1.data.numpy()[0])
#                 loss_2es.append(loss_2.data.cpu().numpy() if USE_CUDA else loss_2.data.numpy()[0])
#                 losses.append(loss.data.cpu().numpy() if USE_CUDA else loss.data.numpy()[0])
#                 id_precision.append(accuracy_score(intent_target.detach().cpu(),torch.argmax(intent_score,dim=1).detach().cpu()))
#                 pred_list,target_list=mask_important_tags(torch.argmax(tag_score,dim=1).view(batch_size,LENGTH),tag_target,x_mask)
#                 sf_f1.append(f1_score(pred_list,target_list,average="micro",zero_division=0))

#         postfix_dict["Te_L"] = round(float(np.mean(losses)),4)
#         postfix_dict["Te_SF_L"] = round(float(np.mean(loss_1es)),4)
#         postfix_dict["Te_ID_L"] = round(float(np.mean(loss_2es)),4)
#         postfix_dict["SF_F1"] = round(float(np.mean(sf_f1)),4)
#         postfix_dict["ID_P"] = round(float(np.mean(id_precision)),4)

#         #Save Best
#         max_sf_f1 = max_sf_f1 if round(float(np.mean(sf_f1)),4)<=max_sf_f1 else round(float(np.mean(sf_f1)),4)
#         max_id_prec = max_id_prec if round(float(np.mean(id_precision)),4)<=max_id_prec else round(float(np.mean(id_precision)),4)
#         if max_sf_f1_both<=round(float(np.mean(sf_f1)),4) and max_id_prec_both<=round(float(np.mean(id_precision)),4):
#             max_sf_f1_both=round(float(np.mean(sf_f1)),4)
#             max_id_prec_both=round(float(np.mean(id_precision)),4)
#             torch.save(bert_layer,f"models/ctran{_fn}-bertlayer.pkl")
#             torch.save(encoder,f"models/ctran{_fn}-encoder.pkl")
#             torch.save(middle,f"models/ctran{_fn}-middle.pkl")
#             torch.save(decoder,f"models/ctran{_fn}-decoder.pkl")
#         postfix_dict["Best_ID_P"] = max_id_prec_both
#         postfix_dict["Best_SF_F1"] = max_sf_f1_both
#         enc_scheduler.step()
#         dec_scheduler.step()
#         mid_scheduler.step()
#         ber_scheduler.step()

#         epoiter.set_postfix(postfix_dict)
#         epoiter.update(1)
#     print(f"max single SF F1: {max_sf_f1}")
#     print(f"max single ID PR: {max_id_prec}")
#     print(f"max mutual SF:{max_sf_f1_both}  PR: {max_id_prec_both}")




# This cell reloads the best model during training from hard-drive.
model.load_state_dict(torch.load(f'models/model.pkl').state_dict())

if USE_CUDA:
    model = model.cuda()







global clipindex
clipindex=0
def removepads(toks,clip=False):
    global clipindex
    result = toks.copy()
    for i,t in enumerate(toks):
        if t=="<PAD>":
            result.remove(t)
        elif t=="<EOS>":
            result.remove(t)
            if not clip:
                clipindex=i
    if clip:
        result=result[:clipindex]
    return result


print("Example of model prediction on test dataset")

model.eval()
with torch.no_grad():
    index = random.choice(range(len(test)))
    test_raw = test[index][0]
    bert_tokens = test_toks['input_ids'][index].unsqueeze(0).cuda()
    bert_mask = test_toks['attention_mask'][index].unsqueeze(0).cuda()
    bert_toktype = test_toks['token_type_ids'][index].unsqueeze(0).cuda()
    subtoken_mask = test_subtoken_mask[index].unsqueeze(0).cuda()
    test_in = prepare_sequence(test_raw,word2index)
    test_mask = Variable(torch.BoolTensor(tuple(map(lambda s: s ==0, test_in.data)))).cuda() if USE_CUDA else Variable(torch.ByteTensor(tuple(map(lambda s: s ==0, test_in.data)))).view(1,-1)
    start_decode = Variable(torch.LongTensor([[word2index['<BOS>']]*1])).cuda().transpose(1,0) if USE_CUDA else Variable(torch.LongTensor([[word2index['<BOS>']]*1])).transpose(1,0)
    test_raw = [removepads(test_raw)]

    tag_score, intent_score = model(bert_tokens,bert_mask,bert_toktype,subtoken_mask,infer=True)

    v,i = torch.max(tag_score,1)
    print("Sentence           : ",*test_raw[0])
    print("Tag Truth          : ", *test[index][1][:len(test_raw[0])])
    print("Tag Prediction     : ",*(list(map(lambda ii:index2tag[ii],i.data.tolist()))[:len(test_raw[0])]))
    v,i = torch.max(intent_score,1)
    print("Intent Truth       : ", test[index][2])
    print("Intent Prediction  : ",index2intent[i.data.tolist()[0]])


print("Instances where model predicted intent wrong")
model.eval()
total_wrong_predicted_intents = 0
with torch.no_grad():
    for i in range(len(test)):
        index = i
        test_raw = test[index][0]
        bert_tokens = test_toks['input_ids'][index].unsqueeze(0).cuda()
        bert_mask = test_toks['attention_mask'][index].unsqueeze(0).cuda()
        bert_toktype = test_toks['token_type_ids'][index].unsqueeze(0).cuda()
        subtoken_mask = test_subtoken_mask[index].unsqueeze(0).cuda()
        test_in = prepare_sequence(test_raw,word2index)
        test_mask = Variable(torch.BoolTensor(tuple(map(lambda s: s ==0, test_in.data)))).cuda() if USE_CUDA else Variable(torch.ByteTensor(tuple(map(lambda s: s ==0, test_in.data)))).view(1,-1)
        # print(removepads(test_raw))
        start_decode = Variable(torch.LongTensor([[word2index['<BOS>']]*1])).cuda().transpose(1,0) if USE_CUDA else Variable(torch.LongTensor([[word2index['<BOS>']]*1])).transpose(1,0)
        test_raw = [removepads(test_raw)]
        
        tag_score, intent_score = model(bert_tokens,bert_mask,bert_toktype,subtoken_mask,infer=True)

        v,i = torch.max(intent_score,1)
        if test[index][2]!=index2intent[i.data.tolist()[0]]:
            v,i = torch.max(tag_score,1)
            print("Sentence           : ",*test_raw[0])
            print("Tag Truth          : ", *test[index][1][:len(test_raw[0])])
            print("Tag Prediction     : ",*list(map(lambda ii:index2tag[ii],i.data.tolist()))[:len(test_raw[0])])
            v,i = torch.max(intent_score,1)
            print("Intent Truth       : ", test[index][2])
            print("Intent Prediction  : ",index2intent[i.data.tolist()[0]])
            print("--------------------------------------")
            total_wrong_predicted_intents+=1

print("Total instances of wrong intent prediction is ",total_wrong_predicted_intents)





