bert_base='./pretrain/bert-base-uncased/'
bert_large='./pretrain/bert-large-uncased/'
deberta_base = "./pretrain/deberta-v3-large"

snips_train="./dataset/snips_train.iob"
snips_test="./dataset/snips_test.iob"
snips_train_dev = "./dataset/snips_train_dev.iob"
atis_train="./dataset/atis-2.train.w-intent.iob"
atis_test="./dataset/atis-2.test.w-intent.iob"
atis_train_dev = "./dataset/atis-2.w-intent_train_dev.iob"

MixATIS_clean_train = "./dataset/MixATIS_clean_train.txt"
MixATIS_clean_test = "./dataset/MixATIS_clean_test.txt"
MixATIS_clean_dev = "./dataset/MixATIS_clean_dev.txt"

MixSNIPS_clean_train  = "./dataset/MixSNIPS_clean_train.txt"
MixSNIPS_clean_test  = "./dataset/MixSNIPS_clean_test.txt"
MixSNIPS_clean_dev  = "./dataset/MixSNIPS_clean_dev.txt"

#ENV variables directly affect the model's behaviour
# ENV_DATASET_TRAIN=atis_train_dev
# ENV_DATASET_TEST=atis_test

ENV_EMBEDDING_SIZE=768  # dimention of embbeding, bertbase=768,bertlarge&elmo=1024
ENV_BERT_ADDR=bert_base
ENV_SEED=1331
ENV_HIDDEN_SIZE=ENV_EMBEDDING_SIZE
ENV_CNN_FILTERS=ENV_HIDDEN_SIZE // 4  #128
ENV_CNN_KERNELS=4
DEPTH = 4

#these are related to training
BATCH_SIZE=64
LENGTH=90
STEP_SIZE=50
ENV_SEED=1331

import torch
USE_CUDA = torch.cuda.is_available()