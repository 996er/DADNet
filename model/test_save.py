
import os
import os.path
import shutil
import sys

# if __name__ == '__main__':

#     root='/home/ADFuzz/aflnet/tutorials/lightftp/out_dir/'

#     collectfile = root+'collection/1/'

#     trainPath = root+'train/1/'

#     testPath2 = root+'test/2/'

#     testPath1 = root+'test/1/'

#     fileNames=os.listdir(collectfile)

#     names=[]

#     for name in fileNames:
#         fileName=name.split(',')[-1]  #hash值/状态值
#         names.append(fileName)

#     #统计了执行次数，和路径/状态
#     fileDict=dict((i, names.count(i)) for i in names) 

#     #对结果进行排序
#     sorted_dict = {k: v for k, v in sorted(fileDict.items(), key=lambda item: item[1])} 
    
#     #排序完，应该选择状态次数较多（高频）的种子作为训练数据，低频种子作为测试数据进行验证
#     # keys=fileDict.keys()
    
#     trainFile = {} #存放训练集
#     testFile  = {} #存放测试集
#     # testFile1 = {} #存放测试集2


#     for k,v in sorted_dict.items():

#         if v > 20 : #选择频率为多少作为训练集 如果阈值的值是定死的，很容易出bug，有可能所有的路径都大于阈值
#             trainFile[k]=v
#         else:       #反之作为低频测试集
#             testFile[k]=v


#     #选择种子copy到test文件中
#     # for i in range(0,5):

#     #     temTuple=fileDict.popitem()
#     #     testFile2[temTuple[0]]=temTuple[1]

#     #     temTuple = fileDict.popitem()
#     #     testFile1[temTuple[0]] = temTuple[1]

#     count = 0
#     for name in fileNames:
#         fileName = name.split(',')[-1]
#         if fileName in trainFile:
#             shutil.copy(collectfile + name, trainPath + name)
#             #不应该只取一个类别的十个，应该每个类别的取一些，这样更丰富
#             if count < 10: #训练集中正常样本数量
#                 shutil.copy(collectfile + name, testPath1 + name)
#                 count=count+1
#         elif fileName in testFile:
#             shutil.copy(collectfile + name, testPath2 + name)



# import torch
# print('CUDA version:',torch.version.cuda)
# print('Pytorch version:',torch.__version__)
# print('usable:','yes' if(torch.cuda.is_available()) else 'no')
# print('numbers:',torch.cuda.device_count())
# print('BF16:','yes' if (torch.cuda.is_bf16_supported()) else 'no')
# print('name:',torch.cuda.get_device_name())
# print('capability:',torch.cuda.get_device_capability())
# print('total_memory:',torch.cuda.get_device_properties(0).total_memory/1024/1024/1024,'GB')
# print('TensorCore:','yes' if (torch.cuda.get_device_properties(0).major >= 7) else 'no')
# print('use rate:',torch.cuda.memory_allocated(0)/torch.cuda.get_device_properties(0).total_memory*100,'%')


if __name__ == '__main__':

    root='/home/ADFuzz/aflnet/tutorials/lightftp/out_dir/'

    collectfile = root+'collection/1/'

    trainPath = root+'train/1/'  #normal

    testPath2 = root+'test_debug/2/'   #abnormal

    testPath1 = root+'test_debug/1/'   #normal

    fileNames=os.listdir(collectfile)

    names=[]

    for name in fileNames:
        fileName=name.split(',')[-1]  #hash值/状态值
        names.append(fileName)

    #统计了执行次数，和路径/状态
    fileDict=dict((i, names.count(i)) for i in names) 

    #对结果进行排序
    sorted_dict = {k: v for k, v in sorted(fileDict.items(), key=lambda item: item[1])} 
    
    #排序完，应该选择状态次数较多（高频）的种子作为训练数据，低频种子作为测试数据进行验证
    # keys=fileDict.keys()
    
    trainFile = {} #存放训练集
    testFile  = {} #存放测试集



    for k,v in sorted_dict.items():

        if v > 10 : #选择频率为多少作为训练集 如果阈值的值是定死的，很容易出bug，有可能所有的路径都大于阈值
            trainFile[k]=v

        elif v < 10 :       #频率低于某值反之作为低频测试集
            testFile[k]=v

    # for name in trainFile.keys():
    #     shutil.copy(collectfile + name, testPath1 + name) #正常路径

    count = 0
    for name in fileNames:
        fileName = name.split(',')[-1]
        if fileName in trainFile:

            if count < len(testFile):
                shutil.copy(collectfile + name, testPath1 + name)
                count=count+1
            else:
                shutil.copy(collectfile + name, trainPath + name)
            #不应该只取一个类别的十个，应该每个类别的取一些，这样更丰富
        


        elif fileName in testFile:
            shutil.copy(collectfile + name, testPath2 + name)  #异常路径

    

