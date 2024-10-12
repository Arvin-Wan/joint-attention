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
import logging
from tqdm.contrib.logging import logging_redirect_tqdm
from config import *

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
from functools import partial

pair = lambda x: x if isinstance(x, tuple) else (x, x)

class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x)) + x

def FeedForward(dim, expansion_factor = 4, dropout = 0., dense = nn.Linear):
    inner_dim = int(dim * expansion_factor)
    return nn.Sequential(
        dense(dim, inner_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        dense(inner_dim, dim),
        nn.Dropout(dropout)
    )

class IDClassifier(nn.Module):
    def __init__(self, depth, dim, num_classes, expansion_factor = 4, expansion_factor_token = 0.5, dropout = 0.):
        super(IDClassifier,self).__init__()
        self.num_tokens = LENGTH
        self.chan_first, self.chan_last = partial(nn.Conv1d, kernel_size = 1), nn.Linear
        
        self.class_mlp = nn.Sequential(
            *[nn.Sequential(
                PreNormResidual(dim, FeedForward(self.num_tokens, expansion_factor, dropout, self.chan_first)),
                PreNormResidual(dim, FeedForward(dim, expansion_factor_token, dropout, self.chan_last))
            ) for _ in range(depth)],
            nn.LayerNorm(dim)  
        )
        self.activation2 = nn.ReLU()
        self.classifier = nn.Linear(dim, num_classes)
        self.dropout1 = nn.Dropout(dropout)
        self.squeeze_linear = nn.Linear(LENGTH, LENGTH //4)
        self.activation2 = nn.ReLU()
    
    def forward(self,x):
        x = self.class_mlp(x)
        x=torch.transpose(x,dim0=1,dim1=2)
        x = self.activation2(self.dropout1(self.squeeze_linear(x)))
        x = torch.transpose(x,dim0=1,dim1=2).mean(1)
        x = self.classifier(self.dropout1(x))

        x = F.log_softmax(x,dim=1)      
        return x
    



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



    def forward(self, input, encoder_outputs, encoder_maskings, bert_subtoken_maskings=None, infer=False, tag2index=None):
        # encoder outputs: BATCH,LENGTH,Dims (16,60,1024)
        batch_size = encoder_outputs.shape[0]
        length = encoder_outputs.size(1) #for every token in batches
        embedded = self.embedding(input)


        encoder_outputs2 = torch.transpose(encoder_outputs, dim0=1, dim1=2)
        encoder_outputs2 = self.activation2(self.dropout4(self.squeeze_linear(encoder_outputs2)))
        encoder_outputs2 = torch.transpose(encoder_outputs2, dim0=1, dim1=2).mean(1)
        # encoder_outputs2 = torch.argmax(encoder_outputs,dim=1)
        intent_score = self.intent_out(self.dropout1(encoder_outputs2))
        intent_score = F.log_softmax(intent_score, dim=1)   
        # intent_score = self.IDclassifier(encoder_outputs)                     

        newtensor = torch.cuda.FloatTensor(batch_size, length, ENV_HIDDEN_SIZE).fill_(0.) # size of newtensor same as original
        for i in range(batch_size): # per batch
            newtensor_index=0
            for j in range(length): # for each token
                if bert_subtoken_maskings[i][j].item()==1:
                    newtensor[i][newtensor_index] = encoder_outputs[i][j]
                    newtensor_index+=1

        if infer==False:
            embedded=embedded*math.sqrt(ENV_HIDDEN_SIZE)
            embedded = self.pos_encoder(embedded)
            zol = self.transformer_decoder(tgt=embedded, memory=newtensor
                                           , memory_mask=self.transformer_diagonal_mask
                                           , tgt_mask=self.transformer_mask)

            scores = self.slot_trans(self.dropout3(zol))    # B, S, 124
            slot_scores = F.log_softmax(scores, dim=2)
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
    def __init__(self, tag2index, intent2index, **kwargs):
        super(Model, self).__init__()
        self.bert_layer = BertLayer()
        self.encoder = Encoder()
        self.decoder = Decoder(len(tag2index), len(intent2index), ratio_squeeze=kwargs.get("ratio_squeeze", 4))
        self.intent2index = intent2index
        self.tag2index = tag2index

    def forward(self,bert_tokens,bert_mask,bert_toktype,subtoken_mask,tag_target=None,infer=False):
        batch_size=bert_tokens.size(0)
        bert_hidden,_ = self.bert_layer(bert_info=(bert_tokens,bert_mask,bert_toktype))

        encoder_output = self.encoder(bert_last_hidden=bert_hidden)

        start_decode = Variable(torch.LongTensor([[self.tag2index['<BOS>']]*batch_size])).cuda().transpose(1,0)
        
        if not infer:
            start_decode = torch.cat((start_decode,tag_target[:,:-1]),dim=1)

        # tag score shape 960,124,  tag target 16,60
        # intent scoer shape 16,22,  intent target shape 16
        # tag_score, intent_score = self.decoder(start_decode,output,bert_mask==0,bert_subtoken_maskings=subtoken_mask,infer=infer)
        tag_score, intent_score = self.decoder(start_decode,encoder_output,bert_mask==0,bert_subtoken_maskings=subtoken_mask,infer=infer,tag2index=self.tag2index)

        return tag_score, intent_score
