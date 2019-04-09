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

from .attention import Attention


def build_model(embedding_matrix, word_index, max_len, lstm_units, 
                verbose = False, compile = True):
    #logger.info('Build model')
    sequence_input = L.Input(shape=(max_len,), dtype='int32')
    embedding_layer = L.Embedding(*embedding_matrix.shape,
                                weights=[embedding_matrix],
                                trainable=False)
    x = embedding_layer(sequence_input)
    x = L.SpatialDropout1D(0.3)(x)
    x = L.Bidirectional(L.CuDNNLSTM(lstm_units, return_sequences=True))(x)
    x = L.Bidirectional(L.CuDNNLSTM(lstm_units, return_sequences=True))(x)
    att = Attention(max_len)(x)
    avg_pool1 = L.GlobalAveragePooling1D()(x)
    max_pool1 = L.GlobalMaxPooling1D()(x)
    x = L.concatenate([att,avg_pool1, max_pool1])
    preds = L.Dense(1, activation='sigmoid')(x)
    model = Model(sequence_input, preds)
    if verbose:
        model.summary()
    if compile:
        model.compile(loss='binary_crossentropy',optimizer=Adam(0.005),metrics=['acc'])
    return model