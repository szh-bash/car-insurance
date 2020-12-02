import numpy as np
import pandas as pd
from config import dataPath

# data-info: https://shimo.im/docs/VtgxWKkv36r3QTyr/read


def proceed(path, md):
    print(path)
    df = pd.read_csv(path, header=[0], sep=',')
    print(df.shape)
    a = df.to_numpy()
    print(a[0])
    a = a[:, 2-md:13-md*2]
    # print(a.shape)
    a[:, 0] = (a[:, 0] == 'Male') * 1.0
    # a[:, 1] = (a[:, 1] - 50.0) / 100
    a[:, 1] /= 100
    # print(max(a[:, 3]))
    a[:, 3] = a[:, 3] / 52.0
    g = {'> 2 Years': 1.0, '1-2 Year': 2/3, '< 1 Year': 0.0}
    a[:, 5] = [g[x] for x in a[:, 5]]
    a[:, 6] = (a[:, 6] == 'Yes') * 1.0
    # print(max(a[:, 7]))
    a[:, 7] /= 540165.0
    # print(max(a[:, 8]))
    a[:, 8] /= 163.0
    # print(max(a[:, 9]))
    a[:, 9] /= 299
    a = np.array(a, dtype=float)
    index = np.arange(a.shape[0])
    np.random.shuffle(index)
    a = a[index]
    print(a.shape)
    return a
    data = a[:, :10]
    # data = (data - 0.5)*2
    # print(data.shape)
    # print(data[:10])
    if not md:
        label = a[:, -1].reshape(-1)
        # print(label.shape)
        # print(label[:10])
    else:
        label = np.zeros(data.shape[0])
    return data, label


if __name__ == '__main__':
    trainData = '/data/shenzhonghai/car-insurance/data/trainData.csv'
    testData = '/data/shenzhonghai/car-insurance/data/testData.csv'
    dt = proceed(dataPath['train-origin'], 0)
    sz = dt.shape[0] // 5 * 4
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
    print(trainX[:5])
    pass

'''
,id,Gender,Age,Driving_License,Region_Code,Previously_Insured,Vehicle_Age,Vehicle_Damage,Annual_Premium,Policy_Sales_Channel,Vintage,Response
0,1,Male,44,1,28.0,0,> 2 Years,Yes,40454.0,26.0,217,1
1,2,Male,76,1,3.0,0,1-2 Year,No,33536.0,26.0,183,0
2,3,Male,47,1,28.0,0,> 2 Years,Yes,38294.0,26.0,27,1
'''