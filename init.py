import cv2
import os
import torch
import numpy as np
from torch.utils.data import Dataset
from config import dataPath
from utils.dataProceeder import proceed
import pandas as pd


class DataReader(Dataset):

    def __init__(self, st, data_name):
        self.st = st
        self.data_name = data_name
        filepath = None
        if data_name in dataPath:
            filepath = dataPath[data_name]
        else:
            print('Data Type 404')
            exit(-1)
        print('data path:', filepath)
        self.rng = np.random
        dt = pd.read_csv(filepath, header=None, index_col=None).to_numpy()
        if st != 'ftest':
            self.dataset = dt[:, :-1]
            self.label = dt[:, -1]
        else:
            self.dataset = dt
            self.label = np.zeros(self.dataset.shape[0])
        # self.dataset, self.label = proceed(filepath, 0 if st == 'train' else 1)
        self.len = self.dataset.shape[0]
        self.type = 2
        print('Data mode: '+self.st)
        self.dataset = np.array(self.dataset, dtype=float)
        self.label = np.array(self.label, dtype=int)
        print('Types:', self.type)
        print('Label:', self.label.shape)
        print('Data:', self.dataset.shape)
        self.x = torch.from_numpy(self.dataset).float()
        self.y = torch.from_numpy(self.label).long()

    def __getitem__(self, index):
        if self.st == 'train':
            return self.x[index], self.y[index]
        elif self.st == 'test' or self.st == 'ftest':
            return self.x[index], self.y[index], index
        else:
            exit(-1)

    def __len__(self):
        return self.len


if __name__ == '__main__':
    DataReader('train', 'trainData')
