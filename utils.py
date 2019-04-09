import io
import os
import numpy as np
import pandas as pd
import logging
import gc
from tqdm import tqdm

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints, optimizers, layers
import keras.layers as L
from keras.models import Model
from keras.optimizers import Adam
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from models import modelutils


DIR_PATH = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH = os.path.join(DIR_PATH, 'data/train.csv')
TEST_PATH = os.path.join(DIR_PATH, 'data/test.csv')
SAMPLE_SUB_PATH = os.path.join(DIR_PATH, 'data/sample_submission.csv')

EMB_PATHS = [
    os.path.join(DIR_PATH, 'data/crawl-300d-2M.vec'),
    os.path.join(DIR_PATH, 'data/glove.840B.300d.txt')
]

COMMENT_TEXT_COL = 'comment_text'

def get_logger():
    FORMAT = '[%(levelname)s]%(asctime)s:%(name)s:%(message)s'
    logging.basicConfig(format=FORMAT)
    logger = logging.getLogger('main')
    logger.setLevel(logging.DEBUG)
    return logger

logger = get_logger()

def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')


def load_embeddings(path):
    with open(path) as f:
        #return dict(get_coefs(*line.strip().split(' ')) for line in f)
        return dict(get_coefs(*o.strip().split(" ")) for o in tqdm(io.open(path, 'r', encoding='utf-8', newline='\n', errors='ignore')))


def build_embedding_matrix(word_index, path, emb_max_feat):
    embedding_index = load_embeddings(path)
    embedding_matrix = np.zeros((len(word_index) + 1, emb_max_feat))
    for word, i in word_index.items():
        try:
            embedding_matrix[i] = embedding_index[word]
        except KeyError:
            pass
        except:
            embedding_matrix[i] = embeddings_index["unknown"]
            
    del embedding_index
    gc.collect()
    return embedding_matrix


def load_data():
    logger.info('Load train and test data')
    train = pd.read_csv(os.path.join(JIGSAW_PATH,'train.csv'), index_col='id')
    test = pd.read_csv(os.path.join(JIGSAW_PATH,'test.csv'), index_col='id')
    return train, test

def preprocess_text(df):
    logger.info('data preprocessing')
    
    # adding preprocessing from this kernel: https://www.kaggle.com/taindow/simple-cudnngru-python-keras
    punct_mapping = {"_":" ", "`":" "}
    punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'
    def clean_special_chars(text, punct, mapping):
        for p in mapping:
            text = text.replace(p, mapping[p])    
        for p in punct:
            text = text.replace(p, f' {p} ')     
        return text

    df[COMMENT_TEXT_COL] = df[COMMENT_TEXT_COL].astype(str)
    df[COMMENT_TEXT_COL] = df[COMMENT_TEXT_COL].apply(lambda x: clean_special_chars(x, punct, punct_mapping))
    
    return df


def run_tokenizer(tokenizer, train, test, seq_len):
    logger.info('Fitting tokenizer')
    tokenizer.fit_on_texts(list(train[COMMENT_TEXT_COL]) + list(test[COMMENT_TEXT_COL]))
    X_train = tokenizer.texts_to_sequences(list(train[COMMENT_TEXT_COL]))
    y_train = np.where(train['target'] >= 0.5, 1, 0)
    X_test = tokenizer.texts_to_sequences(list(test[COMMENT_TEXT_COL]))
    
    X_train = pad_sequences(X_train, maxlen=seq_len)
    X_test = pad_sequences(X_test, maxlen=seq_len)
    
    return X_train, X_test, y_train

def build_embeddings(word_index, emb_max_feat):
    logger.info('Load and build embeddings')
    embedding_matrix = np.concatenate(
        [build_embedding_matrix(word_index, f, emb_max_feat) for f in EMB_PATHS], axis=-1) 
    return embedding_matrix


def run_model(X_train, X_test, y_train, embedding_matrix, word_index, 
              batch_size, epochs, max_len, lstm_units):
    logger.info('Prepare folds')
    folds = KFold(n_splits=5, random_state=42)
    oof_preds = np.zeros((X_train.shape[0]))
    sub_preds = np.zeros((X_test.shape[0]))
    
    logger.info('Run model')
    for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train, y_train)):
        
        #K.clear_session()
        check_point = ModelCheckpoint(f'mod_{fold_}.hdf5', save_best_only = True)
        early_stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)
        model = modelutils.build_model(embedding_matrix, word_index, max_len, lstm_units)
        model.fit(X_train[trn_idx],
            y_train[trn_idx],
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_train[val_idx], y_train[val_idx]),
            callbacks = [early_stopping,check_point])
    
        oof_preds[val_idx] += model.predict(X_train[val_idx])[:,0]
        sub_preds += model.predict(X_test)[:,0]
    sub_preds /= folds.n_splits
    print(roc_auc_score(y_train,oof_preds))
    logger.info('Complete run model')
    return sub_preds


def submit(sub_preds):
    logger.info('Prepare submission')
    submission = pd.read_csv(SAMPLE_SUB_PATH, index_col='id')
    submission['prediction'] = sub_preds
    submission.reset_index(drop=False, inplace=True)
    submission.to_csv('submission.csv', index=False)
    
