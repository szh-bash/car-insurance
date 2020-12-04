import os
import re
import sys
import torch
import numpy as np
import progressbar as pb

sys.path.append("..")
from model.nn.net import Net
from torch.utils.data import DataLoader
from config import test_batch_size as batch_size
from loss import ArcMarginProduct as ArcFace


def get(filepath, data):
    _store = {}
    _feats = []
    # load data
    data_loader = DataLoader(dataset=data, batch_size=batch_size, shuffle=False, pin_memory=True)

    # load model
    device = torch.device('cuda:0')
    checkpoint = torch.load(filepath)
    model = Net().cuda()
    model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['net'].items()})
    model.eval()
    # arc = ArcFace(40, data.type).to(device)
    # arc.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['arc'].items()})
    # arc.eval()  # DropOut/BN

    # get feat
    print('Calculating Feature Map...')
    acc = 0
    res = np.zeros(data.len)
    lbs = np.zeros(data.len)
    for i, (inputs, labels, names) in enumerate(data_loader):
        feat = model(inputs.to(device))
        feat = feat.cpu().detach().numpy()
        res[names] = feat
        lbs[names] = labels.numpy()
    p = 0
    threshold = 0.5
    for i in range(1000):
        x = i/1000
        tp = ((res > x) * lbs).sum()
        fp = (res > x).sum()-tp
        if tp+fp == 0:
            pre = 999999999
        else:
            pre = tp/(tp+fp)
        recall = tp/lbs.sum()
        if pre>0 and recall>0 and 2/(1/pre+1/recall) > p:
            p = 2/(1/pre+1/recall)
            threshold = x
    acc = ((res > 0.5) == lbs).sum()/lbs.shape[0]*100
    print('epoch: %d\niters: %d\nloss: %.3lf\ntrain_acc: %.3lf' %
          (checkpoint['epoch'],
           checkpoint['iter'],
           checkpoint['loss'],
           checkpoint['acc']
           ))
    return acc, res, lbs, threshold, p
