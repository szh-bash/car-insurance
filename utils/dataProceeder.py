import numpy as np
import pandas as pd
from config import dataPath

# data-info: https://shimo.im/docs/VtgxWKkv36r3QTyr/read


def proceed(path, md, rg, mini=True):
    print(path)
    _df = pd.read_csv(path, header=[0], sep=',')
    print(_df.shape)
    a = _df.to_numpy()
    print(a[0])
    a = a[:, 2-md:13-md*2]
    if md == 0:
        lb = a[:, -1].copy()
        a = np.delete(a, [a.shape[1]-1], axis=1)

    # print(a.shape)
    a[:, 0] = (a[:, 0] == 'Male') * 1.0
    # a[:, 1] = (a[:, 1] - 50.0) / 100
    a[:, 1] /= 100
    # sc = np.array([int(x) for x in a[:, 1]])
    # age = np.eye(a.shape[0], 100)[sc]
    # a = np.concatenate((a, age), axis=1)

    # print(max(a[:, 3]))
    # a[:, 3] = a[:, 3] / 52.0
    sc = np.array(a[:, 3]-1, dtype=int)
    reg = np.eye(a.shape[0], 52)[sc]
    print(reg.shape, a.shape)
    a = np.concatenate((a, reg), axis=1)

    g = {'> 2 Years': 0, '1-2 Year': 1, '< 1 Year': 2}
    sc = np.array([g[x] for x in a[:, 5]])
    vage = np.eye(a.shape[0], 3)[sc]
    a = np.concatenate((a, vage), axis=1)

    a[:, 6] = (a[:, 6] == 'Yes') * 1.0
    # print(max(a[:, 7]))
    a[:, 7] /= 540165.0
    # print(max(a[:, 8]))
    a[:, 8] /= 163.0
    # print(max(a[:, 9]))
    # print(np.sort(a[:, 9])[-250000])
    a[:, 9] /= 299
    # a[:, :-1] = (a[:, :-1] - 0.5) * 2
    b = np.delete(a, [3, 5, 2, 7, 9], axis=1)
    b = np.array(b, dtype=float)
    if md == 0:
        print(b.shape, lb.shape)
        b = np.concatenate((b, lb.reshape(lb.shape[0], -1)), axis=1)
    print(b.shape)
    print(b[0])
    if rg:
        return b
    b = b[b[:, -1].argsort()]
    tot = int(np.sum(b[:, [-1]]))
    indx = np.arange(b.shape[0]-tot)
    np.random.shuffle(indx)
    b[:-tot] = b[indx]
    if mini:
        print("hello")
        indx = np.arange(tot)
        np.random.shuffle(indx)
        b[-tot:] = b[indx+b.shape[0]-tot]
        b = b[-tot*2:]
        b = b[int(b.shape[0]*0.4):int(b.shape[0]*0.6)]
    else:
        sc = np.ones(b.shape[0])
        sc[-tot:] = 7
        b = np.repeat(b, sc.tolist(), axis=0)
    print(b.shape)
    # 53, 66, 50, 58, 76, 66, 76.6, 50, 64.5, 50
    #  0,  1,  2,  3,  4,  5,    6,  7,    8,  9
    return b


def build_test():
    filepath = dataPath['trainAll']
    dts = proceed(dataPath['train-origin'], 0, 1)
    dff = pd.DataFrame(dts)
    dff.to_csv(filepath, header=False, index=False)
    filepath = dataPath['testAll']
    dts = proceed(dataPath['test-origin'], 1, 1)
    dff = pd.DataFrame(dts)
    dff.to_csv(filepath, header=False, index=False)


if __name__ == '__main__':
    build_test()
    exit(0)

    trainData = dataPath['trainDataMini']
    testData = dataPath['testDataMini']
    dt = proceed(dataPath['train-origin'], 0, 0)
    index = np.arange(dt.shape[0])
    np.random.shuffle(index)
    dt = dt[index]
    print(dt.shape)
    # sz = dt.shape[0] // 5 * 4
    sz = dt.shape[0]
    train = dt[:sz]
    test = dt[sz:]
    print(train.shape, test.shape)
    df = pd.DataFrame(train)
    df.to_csv(trainData, header=False, index=False)
    df = pd.DataFrame(test)
    df.to_csv(testData, header=False, index=False)
    # proceed(dataPath['test-origin'], 1)
    df = pd.read_csv(trainData, header=None, index_col=None)
    print(df.shape)
    trainX = df.to_numpy()
    print(trainX[0])
    pass

'''
,id,Gender,Age,Driving_License,Region_Code,Previously_Insured,Vehicle_Age,Vehicle_Damage,Annual_Premium,Policy_Sales_Channel,Vintage,Response
0,1,Male,44,1,28.0,0,> 2 Years,Yes,40454.0,26.0,217,1
1,2,Male,76,1,3.0,0,1-2 Year,No,33536.0,26.0,183,0
2,3,Male,47,1,28.0,0,> 2 Years,Yes,38294.0,26.0,27,1
'''