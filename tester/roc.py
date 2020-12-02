import re
import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append("..")
from tester.getFeat import get
from init import DataReader
from config import modelPath, server
import socket
import time


def calc(filepath):
    acc = get(filepath, data)
    print("Test ACC: %.3f" % (acc*100))
    return acc


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
    data = DataReader("test", "testData")
    length = 10000
    thresholds_left, thresholds_right = -0.0, 1.0
    thresholds = np.linspace(thresholds_left, thresholds_right, length)
    test_server()