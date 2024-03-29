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
from datetime import datetime
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

from pytorch_pretrained_bert import convert_tf_checkpoint_to_pytorch
from pytorch_pretrained_bert import BertTokenizer, BertForSequenceClassification,BertAdam

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
    #BATCH_SIZE = 1024
    BATCH_SIZE = 256
    
    #BATCH_SIZE = 2048
    NUM_EPOCHS = 1
    LSTM_UNITS = 64
    if args.debug:
        print('running in debug mode')
    if args.debug:
        result_dir = os.path.join(utils.RESULT_DIR, 'debug-'+datetime.strftime(datetime.now(), '%Y%m%d%H%M%S'))
    else:
        result_dir = os.path.join(utils.RESULT_DIR, datetime.strftime(datetime.now(), '%Y%m%d%H%M%S'))
    os.mkdir(result_dir)
    print(f'created: {result_dir}')

#     convert_tf_checkpoint_to_pytorch.convert_tf_checkpoint_to_pytorch(
#         os.path.join(utils.BERT_MODEL_PATH, 'bert_model.ckpt'),
#         os.path.join(utils.BERT_MODEL_PATH, 'bert_config.json'),
#         utils.PYTORCH_BERT_MODEL_PATH)
    train_data = ToxicDataset(mode='train', debug=args.debug)
    test_data = ToxicDataset(mode='test')
    train, test = train_data.data, test_data.data
    train = utils.preprocess_data(train, mode='train')
    test = utils.preprocess_data(test)
    #tokenizer = Tokenizer(num_words=MAX_FEATURES, lower=True)
    tokenizer = BertTokenizer.from_pretrained(utils.BERT_MODEL_PATH, 
                                              do_lower_case=True)
    X_train, X_test, y_train = utils.run_bert_tokenizer(tokenizer, train, test, 
                                                               seq_len=MAX_LEN)
    #word_index = tokenizer.word_index
    word_index = None
    #print(word_index)
#    print(f'vocab size: {len(word_index)}')
#     embedding_matrix = utils.build_embeddings(word_index, emb_max_feat=EMB_MAX_FEAT)
#     print(embedding_matrix.shape)
    embedding_matrix = None
    sub_preds, oof_df = utils.run_model_pytorch(result_dir, X_train, X_test, y_train, embedding_matrix, 
                                        word_index, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, 
                                        max_len=MAX_LEN, lstm_units=LSTM_UNITS, oof_df=train)
    bias_metrics_df = utils.compute_bias_metrics_for_model(dataset=oof_df, 
                                                           subgroups=utils.IDENTITY_COLS,
                                                           model=utils.PREDICT_COL, 
                                                           label_col=utils.TOXICITY_COLUMN)
    validation_final_socre = utils.get_final_metric(bias_metrics_df, 
                                                    utils.calculate_overall_auc(oof_df, 
                                                                          utils.TOXICITY_COLUMN)
                                                   )
    print(f'validation final score: {validation_final_socre}')
    utils.submit(result_dir, sub_preds)
    print('finish!!!')
    
if __name__ == "__main__":
    main()
    
    
    
    
    