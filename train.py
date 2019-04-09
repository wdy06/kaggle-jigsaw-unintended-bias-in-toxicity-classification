import numpy as np 
import os
import random as rn

#fix random seed
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
rn.seed(12345)
import tensorflow as tf
tf.set_random_seed(1234)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1,
                              gpu_options=tf.GPUOptions(allow_growth=True))
# session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
from keras import backend as K
K.set_session(sess)
import torch
torch.manual_seed(2019)
torch.cuda.manual_seed(2019)
torch.cuda.manual_seed_all(2019)
torch.backends.cudnn.deterministic = True

import argparse
import pandas as pd 
import gc
import logging
import datetime
import warnings
from tqdm import tqdm
tqdm.pandas()

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints, optimizers, layers
import keras.layers as L
from keras.models import Model
from keras.optimizers import Adam
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

import utils
from dataset import ToxicDataset

parser = argparse.ArgumentParser(description='jigsaw unintended bias in  toxicity classification on kaggle')
parser.add_argument("--debug", help="run debug mode",
                    action="store_true")
parser.add_argument("--no_cache", help="extract feature without cache",
                    action="store_true")
args = parser.parse_args()

def main():
    COMMENT_TEXT_COL = 'comment_text'
    EMB_MAX_FEAT = 300
    MAX_LEN = 220
    MAX_FEATURES = 100000
    BATCH_SIZE = 1024
    NUM_EPOCHS = 10
    LSTM_UNITS = 64
    
    if args.debug:
        print('running in debug mode')
    
    train_data = ToxicDataset(mode='train', debug=args.debug)
    test_data = ToxicDataset(mode='test')
    train, test = train_data.data, test_data.data
    train, test = utils.perform_preprocessing(train, test)
    X_train, X_test, y_train, word_index = utils.run_tokenizer(train, test, 
                                                               num_words=MAX_FEATURES,
                                                               seq_len=MAX_LEN)
    embedding_matrix = utils.build_embeddings(word_index, emb_max_feat=EMB_MAX_FEAT)
    sub_preds = utils.run_model(X_train, X_test, y_train, embedding_matrix, word_index,
                                batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, 
                                max_len=MAX_LEN, lstm_units=LSTM_UNITS)
    utils.submit(sub_preds)
    
if __name__ == "__main__":
    main()
    
    
    
    
    