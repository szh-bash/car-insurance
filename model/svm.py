import matplotlib.pyplot as plt
import numpy as np
import scipy.io as scio
from sklearn import svm
import pandas as pd
import sys
sys.path.append('..')
from config import dataPath
import json


def worker(train_path, test_path):
    print("SVM training started!")
    train = pd.read_csv(train_path, header=None, index_col=None).to_numpy()
#     idx = np.arange(train.shape[0])
#     np.random.shuffle(idx)
#     train = train[idx]
#     print(train.shape)
    train_X = train[:, :-1]
    train_y = train[:, -1]
    # print(train_X.shape)
    # print(train_y.shape)
    # print('Training Linear SVM (Spam Classification)')
    # print('(this may take 1 to 2 minutes)')
    c = 0.1
    clf = svm.SVC(c, kernel='linear')
    clf.fit(train_X, train_y)
    p = clf.predict(train_X)
    print('  Training Accuracy: {}'.format(np.mean(p == train_y) * 100))
    test = pd.read_csv(test_path, header=None, index_col=None)
    test = test.to_numpy()
    print(test.shape)
    test_X = test[:, :-1]
    test_y = test[:, -1]
    # test_X = test[:, :]
    # test_y = np.array([0]*test_X.shape[0])
    print(test_X.shape)
    print(test_y.shape)
    p = clf.predict(test_X)
    # p = np.array(p)
    tp = (p*test_y).sum()
    fn = ((1-p)*test_y).sum()
    fp = (p*(1-test_y)).sum()
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    fscore = 2/(1/precision+1/recall)
    print('  Test Accuracy: {}'.format(np.mean(p == test_y) * 100))
    print('  Precision: {}'.format(precision))
    print('  Recall: {}'.format(recall))
    print('  f1-score: {}'.format(fscore))
    js = {}
    for i in range(len(p)):
        js[str(i)] = int(p[i])
    with open('/home/shenzhonghai/car-insurance/model/submission.json', 'w') as file_obj:
        json.dump(js, file_obj)
    return np.array([precision, recall, fscore])


def DaGongRen(rd):
    print('Mission %d Started!' % round)
    f = np.array([0]*3, dtype=float)
    for j in range(5):
        res = worker(rd, j)
        f += res
    f /= 5
    print('Average_Precision: {}'.format(f[0]))
    print('Average_Recall: {}'.format(f[1]))
    print('Average_f1-score: {}'.format(f[2]))
    print('Mission %d Achieved!' % rd)


if __name__ == "__main__":
    worker(dataPath["trainDataMini"], dataPath['trainAll'])
