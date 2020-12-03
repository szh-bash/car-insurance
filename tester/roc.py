import re
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
sys.path.append("..")
from tester.getFeat import get
from init import DataReader
from config import modelPath, server, modelSavePath
import socket
import time


def calc(filepath):
    _acc, _res = get(filepath, data)
    print("Test ACC: %.3f" % (_acc*100))
    return _acc


def link_handler(link):
    filepath = link.recv(1024).decode()
    if filepath == 'exit':
        print("Train End....")
        return True
    print(time.strftime("%Y-%m-%d %H:%M:%S Test server activated....", time.localtime()))
    print("Model path: " + filepath)
    calc(filepath)
    link.close()
    return False


def test_server():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('127.0.0.1', server))
    s.listen(5)
    print(time.strftime("%Y-%m-%d %H:%M:%S Test server Online....", time.localtime()))
    while True:
        cnn, addr = s.accept()
        if link_handler(cnn):
            exit(0)


if __name__ == '__main__':
    # data = DataReader("test", "testData")
    # data = DataReader("test", "trainAll")
    data = DataReader("ftest", "testAll")
    # test_server()
    # acc, res = get(modelPath, data)
    acc, res = get(modelSavePath+"_5.tar", data)
    # res = np.zeros(res.shape[0])
    print(res.sum())
    js = {}
    for i in range(res.shape[0]):
        js[str(i+1)] = int(res[i])
    with open('submission.json', 'w') as file_obj:
        json.dump(js, file_obj)
