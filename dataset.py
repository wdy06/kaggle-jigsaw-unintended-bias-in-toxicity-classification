import numpy as np
import pandas as pd

import utils

class ToxicDataset():
    def __init__(self, mode='train', debug=False):
        self.mode = mode
        self.debug = debug
        if mode == 'train':
            self.data = pd.read_csv(utils.TRAIN_PATH)
        elif mode == 'test':
            self.data = pd.read_csv(utils.TEST_PATH)
        else:
            raise ValueError('unknown mode')
        if debug:
            self.data = self.data[:100]
            
    def __getitem__(self, index):
        return self.data.iloc[index]
    
    def __len__(self):
        return self.data.shape[0]






