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
Total = 30
batch_size = 40
test_batch_size = 512
learning_rate = 0.001
weight_decay = 0.0005
modelName = 'cnn3_AF'
modelSavePath = '/data/shenzhonghai/car-insurance/models/'+modelName
modelPath = '/data/shenzhonghai/car-insurance/models/'+modelName+'.tar'
server = 2333

