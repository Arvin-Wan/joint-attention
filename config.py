_fn="final" # file unique id for saving and loading models
bert_base='./pretrain/bert-base-uncased/'
bert_large='./pretrain/bert-large-uncased/'
deberta_base = "./pretrain/deberta-v3-large"

snips_train="./dataset/snips_train.iob"
snips_test="./dataset/snips_test.iob"
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
ENV_SEED=1331

import torch
import numpy as np
# you must use cuda to run this code. if this returns false, you can not proceed.
USE_CUDA = torch.cuda.is_available()
if USE_CUDA:
    print("You are using cuda. Good!")
    torch.cuda.manual_seed(ENV_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
else:
    torch.manual_seed(ENV_SEED)
    print('You are NOT using cuda! Some problems may occur.')
np.random.seed(ENV_SEED)
# random.seed(ENV_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False