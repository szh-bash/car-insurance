import os
import time
import socket
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model.nn.net import Net
from loss import ArcMarginProduct as ArcFace

from config import learning_rate, batch_size, weight_decay, Total, modelSavePath, server
from init import DataReader


def get_label(output):
    # print(output.shape)
    # return torch.argmax(output, dim=1)
    return output > 0.5


def get_loss(ft, target):
    logsoftmax = nn.LogSoftmax(dim=1).to(device)
    return torch.mean(torch.sum(-target * logsoftmax(ft), dim=1))


def save_grad(name):
    def hook(grad):
        grads[name] = grad
    return hook


def get_max_gradient(g):
    # print('Gradient:', g)
    pm = torch.max(g)
    nm = torch.min(g)
    if -nm > pm:
        return nm
    else:
        return pm


def save_test(status, filepath):
    if status is not None:
        torch.save(status, filepath)
        print('Model saved to '+filepath)
    ip_port = ('127.0.0.1', server)
    s = socket.socket()
    try:
        s.connect(ip_port)
    except:
        print('Test Server Down...')
    else:
        s.sendall(filepath.encode())
        # print('Test request sent!')
        s.close()


if __name__ == '__main__':
    # set config
    data = DataReader('train', 'trainData')
    slides = (data.len - 1) // batch_size + 1
    grads = {}

    # Some Args setting
    net = Net()

    device = torch.device("cuda:0")
    if torch.cuda.device_count() > 1:
        devices_ids = [0]
        net = nn.DataParallel(net, device_ids=devices_ids)
        print("Let's use %d/%d GPUs!" % (len(devices_ids), torch.cuda.device_count()))
    net.to(device)
    data_loader = DataLoader(dataset=data, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    arcFace = ArcFace(640, data.type).to(device)
    # criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([1, 7])).float()).to(device)
    criterion = nn.BCEWithLogitsLoss().to(device)
    optimizer = optim.Adam([{'params': net.parameters()}],
                            # {'params': arcFace.parameters()}],
                           lr=learning_rate, weight_decay=weight_decay)
    # optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1000, 2000], gamma=0.1, last_epoch=-1)
    print(net.parameters())
    print(arcFace.parameters())
    if os.path.exists(modelSavePath+'.tar'):
        checkpoint = torch.load(modelSavePath+'.tar')
        net.load_state_dict(checkpoint['net'])
        arcFace.load_state_dict(checkpoint['arc'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        epoch_start = checkpoint['epoch']
        iter_start = checkpoint['iter']
        print('Load checkpoint Successfully!')
        print('epoch: %d\niter: %d' % (epoch_start, iter_start))
        print(scheduler.state_dict())
    else:
        epoch_start = 0
        iter_start = 0
        print('Model saved to %s' % (modelSavePath + '.tar'))

    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    for param in arcFace.parameters():
        num_params += param.numel()
    print("Parameter Number: %d M" % (num_params / 1e6))

    print("Training Started!")
    iterations = iter_start
    state = {}
    for epoch in range(epoch_start, Total):
        data_time, train_time = 0, 0
        pred, train_x, train_y, loss = None, None, None, None

        batch_data_time = time.time()
        acc_bc, loss_bc = 0, 0
        for i, (inputs, labels) in enumerate(data_loader):
            train_x, train_y = inputs.to(device), labels.to(device)
            dt = time.time() - batch_data_time
            data_time = data_time + dt
            # exit(0)
            batch_train_time = time.time()
            feat = net(train_x)
            # feat = arcFace(feat, train_y)
            feat.register_hook(save_grad('feat_grad'))
            loss = criterion(feat, train_y.float())
            # loss = get_loss(feat, train_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            tt = time.time() - batch_train_time
            train_time = train_time + tt

            iterations += 1
            # if iterations % 200 == 0:
            #     print('Output Max pred: %s' % torch.max(feat.gather(1, train_y.view(-1, 1))))
            #     print('Abs Max Gradient of net_output:',
            #           get_max_gradient(grads['feat_grad'].gather(1, train_y.view(-1, 1))))

            pred = get_label(feat)
            # acc = (pred == train_y.argmax(dim=1)).sum().float() / train_y.size(0) * 100
            acc = (pred == train_y).sum().float() / train_y.size(0) * 100
            print('epoch: %d/%d, iters: %d, lr: %.5f, '
                  'loss: %.5f, acc: %.5f, train_time: %.5f, data_time: %.5f' %
                  (epoch, Total, iterations, scheduler.get_lr()[0],
                   float(loss), float(acc), tt, dt))
            loss_bc += float(loss)
            acc_bc += float(acc)
            batch_data_time = time.time()

        epoch += 1
        loss_bc /= slides
        acc_bc /= slides
        print('epoch: %d/%d, loss: %.5f, acc: %.5f, train_time: %.5f, data_time: %.5f' %
              (epoch, Total, loss_bc, acc_bc, train_time, data_time))
        state = {'net': net.state_dict(),
                 'arc': arcFace.state_dict(),
                 'optimizer': optimizer.state_dict(),
                 'scheduler': scheduler.state_dict(),
                 'epoch': epoch,
                 'iter': iterations,
                 'loss': loss_bc,
                 'acc': acc_bc}
        save_test(state, modelSavePath+'_'+str(epoch)+'.tar')

    torch.save(state, modelSavePath+'.tar')
    save_test(None, 'exit')
    print('fydnb!')
