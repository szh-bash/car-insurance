import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math


class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
          in_features: size of each input sample
          out_features: size of each output sample
          s: norm of input feature
          m: margin
          cos(theta + m)
    """

    def __init__(self, in_features, out_features, s=4.0, m=0.45, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, inputs, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(inputs), F.normalize(self.weight))
        sine = torch.sqrt(- torch.pow(cosine, 2) + 1.0)
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        # # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device='cuda:0')
        # two_hot = torch.zeros(cosine.size(), device='cuda:0')
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # two_hot = label
        # two_hot = torch.where(label > 0, label/label, two_hot)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + (- one_hot + 1.0) * cosine  # torch.__version__ == 0.4
        # output = (two_hot * phi) + (- two_hot + 1.0) * cosine
        output *= self.s

        return output
