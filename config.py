# Configurations

# init
dataPath = {
'train-origin': '/home/shenzhonghai/car-insurance/data/VI_train.csv',
'test-origin': '/home/shenzhonghai/car-insurance/data/VI_test.csv',
'trainData': '/home/shenzhonghai/car-insurance/data/trainData.csv',
'testData': '/home/shenzhonghai/car-insurance/data/testData.csv',
'trainAll': '/home/shenzhonghai/car-insurance/data/trainAll.csv',
'testAll': '/home/shenzhonghai/car-insurance/data/testAll.csv',
}

# train/test
Total = 20
batch_size = 512  # 1:1 - 40 7:7 - 512
test_batch_size = 512
learning_rate = 0.001
weight_decay = 0.0005
modelName = 'nn3_oh_vage_dup'
modelSavePath = '/data/shenzhonghai/car-insurance/models/'+modelName
modelPath = '/data/shenzhonghai/car-insurance/models/'+modelName+'.tar'
server = 2333

