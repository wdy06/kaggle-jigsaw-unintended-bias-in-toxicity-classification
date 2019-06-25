import io
import os
import numpy as np
import pandas as pd
import pickle
import logging
import gc
from tqdm import tqdm
from collections import OrderedDict

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints, optimizers, layers
import keras.layers as L
from keras.models import Model
from keras.optimizers import Adam
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import torch
from torch import nn
from torch.utils import data
from torch.nn import functional as F

from pytorch_pretrained_bert import convert_tf_checkpoint_to_pytorch
from pytorch_pretrained_bert import BertTokenizer, BertForSequenceClassification,BertAdam
from apex import amp

from models import modelutils
from models.model_pytorch import SimpleLSTM


DIR_PATH = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH = os.path.join(DIR_PATH, 'data/train.csv')
TEST_PATH = os.path.join(DIR_PATH, 'data/test.csv')
RESULT_DIR = os.path.join(DIR_PATH, 'results')
SAMPLE_SUB_PATH = os.path.join(DIR_PATH, 'data/sample_submission.csv')
BERT_MODEL_PATH = os.path.join(DIR_PATH, 
                               'data/uncased_L-12_H-768_A-12/')
BERT_MODEL_CONFIG = os.path.join(BERT_MODEL_PATH, 'bert_config.json')

EMB_PATHS = [
    os.path.join(DIR_PATH, 'data/crawl-300d-2M.vec.pkl'),
    os.path.join(DIR_PATH, 'data/glove.840B.300d.txt.pkl')
]

# List all identities
IDENTITY_COLS = [
    'male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',
    'muslim', 'black', 'white', 'psychiatric_or_mental_illness']

TOXICITY_COLUMN = 'target'
COMMENT_TEXT_COL = 'comment_text'
PREDICT_COL = 'oof_predict'

SUBGROUP_AUC = 'subgroup_auc'
BPSN_AUC = 'bpsn_auc'  # stands for background positive, subgroup negative
BNSP_AUC = 'bnsp_auc'  # stands for background negative, subgroup positive

def get_logger():
    FORMAT = '[%(levelname)s]%(asctime)s:%(name)s:%(message)s'
    logging.basicConfig(format=FORMAT)
    logger = logging.getLogger('main')
    logger.setLevel(logging.DEBUG)
    return logger

logger = get_logger()

def load_pickle(path):
    with open(path, 'rb') as f:
        pickle_data = pickle.load(f)
    return pickle_data

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')


def load_embeddings(path):
    with open(path) as f:
        #return dict(get_coefs(*line.strip().split(' ')) for line in f)
        return dict(get_coefs(*o.strip().split(" ")) for o in tqdm(io.open(path, 'r', encoding='utf-8', newline='\n', errors='ignore')))


def build_embedding_matrix(word_index, path, emb_max_feat):
#     embedding_index = load_embeddings(path)
    embedding_index = load_pickle(path)
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

def build_embeddings(word_index, emb_max_feat):
    logger.info('Load and build embeddings')
    embedding_matrix = np.concatenate(
        [build_embedding_matrix(word_index, f, emb_max_feat) for f in EMB_PATHS], axis=-1) 
    return embedding_matrix

def load_data():
    logger.info('Load train and test data')
    train = pd.read_csv(os.path.join(JIGSAW_PATH,'train.csv'), index_col='id')
    test = pd.read_csv(os.path.join(JIGSAW_PATH,'test.csv'), index_col='id')
    return train, test

def preprocess_data(df, mode=None):
    logger.info('data preprocessing')
    
    # adding preprocessing from this kernel: https://www.kaggle.com/taindow/simple-cudnngru-python-keras
    punct_mapping = {"_":" ", "`":" "}
    punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'
    def _clean_special_chars(text, punct, mapping):
        for p in mapping:
            text = text.replace(p, mapping[p])    
        for p in punct:
            text = text.replace(p, f' {p} ')     
        return text
    # List all identities

    # Convert taget and identity columns to booleans
    def _convert_to_bool(df, col_name):
        df[col_name] = np.where(df[col_name] >= 0.5, True, False)

    def _convert_dataframe_to_bool(df):
        bool_df = df.copy()
        for col in ['target'] + IDENTITY_COLS:
            print(col)
            _convert_to_bool(bool_df, col)
        return bool_df

    df[COMMENT_TEXT_COL] = df[COMMENT_TEXT_COL].astype(str)
    df[COMMENT_TEXT_COL] = df[COMMENT_TEXT_COL].apply(lambda x: _clean_special_chars(x, punct, punct_mapping))
    if mode == 'train':
        df = _convert_dataframe_to_bool(df)
    
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

def convert_lines(example, max_seq_length,tokenizer):
    # Converting the lines to BERT format
    max_seq_length -=2
    all_tokens = []
    longer = 0
    for text in tqdm(example):
        tokens_a = tokenizer.tokenize(text)
        if len(tokens_a)>max_seq_length:
            tokens_a = tokens_a[:max_seq_length]
            longer += 1
        one_token = tokenizer.convert_tokens_to_ids(["[CLS]"]+tokens_a+["[SEP]"])+[0] * (max_seq_length - len(tokens_a))
        all_tokens.append(one_token)
    print(longer)
    return np.array(all_tokens)

def run_bert_tokenizer(tokenizer, train, test, seq_len):
    logger.info('use bert tokenizer')
#     X_train = convert_lines(train["comment_text"] , seq_len, tokenizer)
    X_train = load_pickle('./data/X_train_tokenized.pkl')
    X_test = convert_lines(test["comment_text"] , seq_len, tokenizer)
    
    y_train = np.where(train['target'] >= 0.5, 1, 0)
    
    return X_train, X_test, y_train

def run_model(result_dir, X_train, X_test, y_train, embedding_matrix, word_index, 
              batch_size, epochs, max_len, lstm_units, oof_df):
    logger.info('Prepare folds')
    folds = StratifiedKFold(n_splits=5, random_state=42)
    oof_preds = np.zeros((X_train.shape[0]))
    sub_preds = np.zeros((X_test.shape[0]))
    
    logger.info('Run model')
    for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train, y_train)):
        
        check_point = ModelCheckpoint(os.path.join(result_dir, f'model_{fold_}.hdf5'), save_best_only = True, 
                                      verbose=1, monitor='val_loss', mode='min')
        early_stopping = EarlyStopping(monitor='val_loss', mode='min', patience=5)
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
    oof_df[PREDICT_COL] = oof_preds
    print(roc_auc_score(y_train,oof_preds))
    logger.info('Complete run model')
    return sub_preds, oof_df

def run_model_pytorch(result_dir, X_train, X_test, y_train, embedding_matrix, word_index, 
              batch_size, epochs, max_len, lstm_units, oof_df):
    logger.info('Prepare folds')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
    folds = StratifiedKFold(n_splits=5, random_state=42)
    oof_preds = np.zeros((X_train.shape[0]))
    sub_preds = np.zeros((X_test.shape[0]))
    
    logger.info('Run model')
    for fold_, (trn_idx, val_idx) in tqdm(enumerate(folds.split(X_train, y_train))):
        #model = SimpleLSTM(embedding_matrix)
        model = BertForSequenceClassification.from_pretrained(BERT_MODEL_PATH, 
                                                              num_labels=1)
        model = torch.nn.DataParallel(model)
        model.to(device)
        #optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 
             'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 
             'weight_decay': 0.0}
            ]
        optimizer = BertAdam(optimizer_grouped_parameters, lr=0.001)
        
        print(f'fold: {fold_}')
        x_train_torch = torch.tensor(X_train[trn_idx], dtype=torch.long).to(device)
        y_train_torch = torch.tensor(y_train[trn_idx, np.newaxis], dtype=torch.float32).to(device)
        x_val_torch = torch.tensor(X_train[val_idx], dtype=torch.long).to(device)
        y_val_torch = torch.tensor(y_train[val_idx, np.newaxis], dtype=torch.float32).to(device)
        
        train_dataset = torch.utils.data.TensorDataset(x_train_torch, y_train_torch)
        val_dataset = torch.utils.data.TensorDataset(x_val_torch, y_val_torch)
        
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=batch_size,
                                                 shuffle=False)
        best_val_score = 1000000
        save_path = os.path.join(result_dir, f'model_fold{fold_}')
        for epoch in tqdm(range(epochs)):
            model.train()
            avg_loss = 0.
            # train
            for data, target in tqdm(train_loader):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                y_pred = model(data, attention_mask=(data>0).to(device), labels=None)
                loss_func = nn.BCEWithLogitsLoss(reduction='mean')
                loss = loss_func(y_pred, target)
                loss.backward()
                optimizer.step()
                avg_loss += loss.item() / len(train_loader)

            # validation
            model.eval()
            avg_val_loss = 0.
            valid_preds_fold = np.zeros(len(val_dataset))
            for i, (data, target) in enumerate(val_loader):
                with torch.no_grad():
                    y_pred = model(data, attention_mask=(data>0).to(device), labels=None).detach()

                avg_val_loss += loss_func(y_pred, target).item() / len(val_loader)
                valid_preds_fold[i * batch_size:(i + 1) * batch_size] = sigmoid(y_pred.cpu().numpy())[:, 0]
            print(f'epoch {epoch+1} / {epochs}, loss: {avg_loss}, val_loss: {avg_val_loss}')
            is_best = bool(avg_val_loss < best_val_score)
            if is_best:
                best_val_score = avg_val_loss
                print(f'update best score !! current best score: {best_val_score} !!')
            save_checkpoint(model, is_best, save_path)
        
        oof_preds[val_idx] = valid_preds_fold
        
    
    oof_df[PREDICT_COL] = oof_preds
    # inference test data with fold averaging
    print(f'predicting test data ...')
    test_batch_size = int(batch_size/2)
    x_test_torch = torch.tensor(X_test, dtype=torch.long).to(device)
    test_dataset = torch.utils.data.TensorDataset(x_test_torch)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=int(test_batch_size),
                                              shuffle=False)
    test_preds = np.zeros(len(X_test))
    for fold_ in range(folds.n_splits):
        model = load_pytorch_model(os.path.join(result_dir, f'model_fold{fold_}'), embedding_matrix)
        model.eval()
        model = torch.nn.DataParallel(model)
        model.to(device)
        for i, (data,) in enumerate(test_loader):
            with torch.no_grad():
                y_pred = model(data, attention_mask=(data>0).to(device), labels=None).detach()
            test_preds[i * test_batch_size:(i + 1) * test_batch_size] += sigmoid(y_pred.cpu().numpy()[:,0])
    test_preds /= folds.n_splits
    logger.info('Complete run model')
    return test_preds, oof_df

def submit(result_dir, sub_preds):
    logger.info('Prepare submission')
    submission = pd.read_csv(SAMPLE_SUB_PATH, index_col='id')
    submission['prediction'] = sub_preds
    submission.reset_index(drop=False, inplace=True)
    submission.to_csv(os.path.join(result_dir,'submission.csv'), index=False)
    

def compute_auc(y_true, y_pred):
    try:
        return roc_auc_score(y_true, y_pred)
    except ValueError as e:
        print(e)
        return np.nan

def compute_subgroup_auc(df, subgroup, label, model_name):
    subgroup_examples = df[df[subgroup]]
    return compute_auc(subgroup_examples[label], subgroup_examples[model_name])

def compute_bpsn_auc(df, subgroup, label, model_name):
    """Computes the AUC of the within-subgroup negative examples and the background positive examples."""
    subgroup_negative_examples = df[df[subgroup] & ~df[label]]
    non_subgroup_positive_examples = df[~df[subgroup] & df[label]]
    examples = subgroup_negative_examples.append(non_subgroup_positive_examples)
    return compute_auc(examples[label], examples[model_name])

def compute_bnsp_auc(df, subgroup, label, model_name):
    """Computes the AUC of the within-subgroup positive examples and the background negative examples."""
    subgroup_positive_examples = df[df[subgroup] & df[label]]
    non_subgroup_negative_examples = df[~df[subgroup] & ~df[label]]
    examples = subgroup_positive_examples.append(non_subgroup_negative_examples)
    return compute_auc(examples[label], examples[model_name])

def compute_bias_metrics_for_model(dataset,
                                   subgroups,
                                   model,
                                   label_col,
                                   include_asegs=False):
    """Computes per-subgroup metrics for all subgroups and one model."""
    records = []
    for subgroup in subgroups:
        record = {
            'subgroup': subgroup,
            'subgroup_size': len(dataset[dataset[subgroup]])
        }
        record[SUBGROUP_AUC] = compute_subgroup_auc(dataset, subgroup, label_col, model)
        record[BPSN_AUC] = compute_bpsn_auc(dataset, subgroup, label_col, model)
        record[BNSP_AUC] = compute_bnsp_auc(dataset, subgroup, label_col, model)
        records.append(record)
    return pd.DataFrame(records).sort_values('subgroup_auc', ascending=True)

def calculate_overall_auc(df, model_name):
    true_labels = df[TOXICITY_COLUMN]
    predicted_labels = df[model_name]
    return roc_auc_score(true_labels, predicted_labels)

def power_mean(series, p):
    total = sum(np.power(series, p))
    return np.power(total / len(series), 1 / p)

def get_final_metric(bias_df, overall_auc, POWER=-5, OVERALL_MODEL_WEIGHT=0.25):
    bias_score = np.average([
        power_mean(bias_df[SUBGROUP_AUC], POWER),
        power_mean(bias_df[BPSN_AUC], POWER),
        power_mean(bias_df[BNSP_AUC], POWER)
    ])
    return (OVERALL_MODEL_WEIGHT * overall_auc) + ((1 - OVERALL_MODEL_WEIGHT) * bias_score)


def save_pytorch_model(model, path):
    torch.save(model.state_dict(), path)
    
def load_pytorch_model(path, *args, **kwargs):
    state_dict = torch.load(path)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    #model = SimpleLSTM(*args, **kwargs)
    model = BertForSequenceClassification.from_pretrained(BERT_MODEL_PATH, 
                                                          num_labels=1)
    model.load_state_dict(new_state_dict)
    return model


def save_checkpoint(model, is_best, path):
    """Save checkpoint if a new best is achieved"""
    if is_best:
        print (f"=> Saving a new best to {path}")
        save_pytorch_model(model, path)  # save checkpoint
    else:
        print ("=> Validation Accuracy did not improve")










