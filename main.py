import torch
import argparse
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
from model import Model
from Data_process import process
from crf import CRF


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

def train_loop(postfix_dict):
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
        tag_scores = tag_score.view(batch_size, LENGTH, -1)
        loss_1 = - crf_log_likelihood.log_likelihood(tag_scores, tag_target, reduction="mean") + loss_function_1(tag_score,tag_target.view(-1))
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


def eval_loop():
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
            loss_1 = - crf_log_likelihood.log_likelihood(tag_scores, tag_target, reduction="mean") + loss_function_1(tag_score,tag_target.view(-1))
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

def train_evaluate(args):
    max_id_prec=0.
    max_sf_f1=0.
    max_id_prec_both=0.
    max_sf_f1_both=0.

    with tqdm(total=STEP_SIZE,desc="Epoch") as epoiter:
        for step in range(STEP_SIZE):
            postfix_dict = {}

            postfix_dict = train_loop(postfix_dict)
            eval_dict,sf_f1,id_precision = eval_loop()

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
        
        print(f"{args.data} max single ID PR: {max_id_prec}")
        print(f"{args.data} max single SF F1: {max_sf_f1}")
        print(f"{args.data} max mutual PR: {max_id_prec_both}   SF:{max_sf_f1_both}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-data",
                        default="atis", type=str, required=False,
                        help="Datasets")
    args = parser.parse_args()
    train_data, test_data, tag2index, intent2index = process(args.data)

    model = Model(tag2index, intent2index)
    if USE_CUDA:
        model.cuda()

    # global optimizer, scheduler, loss_function_1, loss_function_2, loss_function_3
    optimizer = optim.AdamW(model.parameters(), lr=0.0001,weight_decay=0.8)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.96)
    crf_log_likelihood = CRF(num_tags=len(tag2index),batch_first=True).cuda()
    loss_function_1 = nn.CrossEntropyLoss(ignore_index=0)
    loss_function_2 = nn.CrossEntropyLoss()
    loss_function_3 = nn.L1Loss()


    train_evaluate(args)
    

