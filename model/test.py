# """
# TRAIN GANOMALY
#
# . Example: Run the following command from the terminal.
#     run train.py                             \
#         --model ganomaly                        \
#         --dataset UCSD_Anomaly_Dataset/UCSDped1 \
#         --batchsize 32                          \
#         --isize 256                         \
#         --nz 512                                \
#         --ngf 64                               \
#         --ndf 64
# """
#
#
# ##
# # LIBRARIES
# import time
#
# import os
# import os.path
# import shutil
# from lib.checkTest import loadData
# from lib.checkTest import loadTestData
# from options import Options
# from lib.data import load_data
#
# from lib.model import Ganomaly
# from lib.model import BaseModel
# from ctypes import *
# import sys
# import threading
# import signal
#
#
# mutex = threading.Lock()
# ##
# # SHM_SIZE = 1
# # SHM_ID = 123456789
# #
# #
# # opt = Options().parse()
# #     ##
# #     # LOAD DATA
# #     # dataloader = load_data(opt)
# # dataloader = loadTestData(opt)
# #     # dataloader = loadData(opt)
# #
# #
# #     ##
# #     # LOAD MODEL
# # model = Ganomaly(opt, dataloader)
# #     ##
# #
# #     # TRAIN MODEL
# # model.train()
# #
# #
# # try:
# #     rt = CDLL('librt.so')
# #
# # except:
# #      rt = CDLL('librt.so.1')
# #
# #
# # shmget = rt.shmget
# # shmget.argtypes = [c_int, c_size_t, c_int]
# # shmget.restype = c_int
# #
# # shmat = rt.shmat
# # shmat.argtypes = [c_int, POINTER(c_void_p), c_int]
# # shmat.restype = POINTER(c_char)
# #
# # shmid = shmget(SHM_ID, SHM_SIZE, 0O644)
# #
# # if shmid<0 :
# #
# #     sys.exit()
# #
# # else:
# #     addr = shmat(shmid,None,0)
#
# # def handler():
# #
# #     addr[0] = model.validate()
# #
# #     file_name = '/home/yu/testC/info.txt'
# #     with open(file_name, 'r') as f:
# #         pid = f.readline().strip('\n')
# #     f.close()
# #     pid = int(pid)
# #     os.kill(pid, 45)
#
#
#
#
#
#
# def train(addr,mutex,shmdata):
#
#
#     # SHM_SIZE = 2
#     # SHM_ID = 123456789
#
#     """ Training
#     """
#     ##
#     # ARGUMENTS
#     opt = Options().parse()
#     ##
#     # LOAD DATA
#     # dataloader = load_data(opt)
#     dataloader = loadTestData(opt)
#     # dataloader = loadData(opt)
#
#
#     ##
#     # LOAD MODEL
#     model = Ganomaly(opt, dataloader)
#     ##
#
#     # TRAIN MODEL
#     # model.train()
#
#     # print(model.validate())
#     # try:
#     #     rt = CDLL('librt.so')
#     #
#     # except:
#     #     rt = CDLL('librt.so.1')
#     #
#     #
#     # shmget = rt.shmget
#     # shmget.argtypes = [c_int, c_size_t, c_int]
#     # shmget.restype = c_int
#     #
#     # shmat = rt.shmat
#     # shmat.argtypes = [c_int, POINTER(c_void_p), c_int]
#     # shmat.restype = POINTER(c_int)
#     #
#     # shmid = shmget(SHM_ID, SHM_SIZE, 0O644)
#     #
#     # if shmid<0 :
#     #
#     #     sys.exit()
#     #
#     # else:
#     #     addr = shmat(shmid,None,0)
#     model.load_pretrain()
#     model.validate_pretrain(addr,mutex,shmdata)
#
#
#     # while True:
#     # #     signal.signal(signal.SIGRTMIN + 10, handler)
#     #
#     #     # signal.signal(signal.SIGINT, bye)
#     #
#     #
#     #     if mutex.acquire(True):
#     #
#     #         if addr[0] == 1 : #说明afl调用infer，共享内存设置为 1
#     #
#     #             addr[1] = model.validate()  #把推理结果重新写入共享内存
#     #
#     #             addr[0] = 0  # 重置为0，告知afl，调用val完成，结果在addr[1]
#     #
#     #
#     #         mutex.release()
#
#
#
# def copy2file():
#     root='/home/yu/公共的/FoRTE-FuzzBench/cjson/cjson-1.7.7/fuzzing/out_dir/'
#
#     trainfile = root+'train/1/'
#
#     testPath2 = root+'test/2/'
#
#     testPath1 = root+'test/1/'
#
#     fileNames=os.listdir(trainfile)
#
#     names=[]
#
#     for name in fileNames:
#         fileName=name.split(',')[-1]  #hash值
#         names.append(fileName)
#
#
#     fileDict=dict((i, names.count(i)) for i in names)
#
#
#
#     # keys=fileDict.keys()
#
#     testFile2 = {}
#     testFile1 = {}
#
#     for i in range(0,10):
#
#         temTuple=fileDict.popitem()
#         testFile2[temTuple[0]]=temTuple[1]
#
#         temTuple = fileDict.popitem()
#         testFile1[temTuple[0]] = temTuple[1]
#
#
#     for name in fileNames:
#         fileName = name.split(',')[-1]
#         if fileName in testFile2:
#             shutil.move(trainfile+name,testPath2+name)
#         elif fileName in testFile1:
#             shutil.copy(trainfile + name, testPath1 + name)
#
#
#
# if __name__ == '__main__':
#
#     SHM_SIZE = 33
#     SHM_ID = 123456789
#     SHM_DATA_ID = 12345671
#     SHM_DATA_SIZE = 1024
#     SHM_DATA_NUM = 32
#
#     try:
#         rt = CDLL('librt.so')
#
#     except:
#         rt = CDLL('librt.so.1')
#
#     shmget = rt.shmget
#     shmget.argtypes = [c_int, c_size_t, c_int]
#     shmget.restype = c_int
#
#     shmat = rt.shmat
#     shmat.argtypes = [c_int, POINTER(c_void_p), c_int]
#     shmat.restype = POINTER(c_int)
#
#     shmid = shmget(SHM_ID, SHM_SIZE, 0O644)
#
#
#     if shmid < 0:
#
#         sys.exit()
#
#     else:
#         addr = shmat(shmid, None, 0)
#
#     shmdataid=[]
#     shmdata=[]
#     for i in range(SHM_DATA_NUM):
#         shmdataid.append(shmget(SHM_DATA_ID+i, SHM_DATA_SIZE, 0O644))
#         if shmdataid[i] < 0:
#             sys.exit()
#         else:
#             temp_addr=shmat(shmdataid[i], None, 0)
#             shmdata.append(temp_addr)
#
#
#
#     # copy2file()
#
#     train(addr,mutex,shmdata)
#

# from PIL import Image
# import numpy as np
#
# data=np.array([[255,255,255,255,255,255,255,255],
#                [0,0,0,0,0,0,0,0],
#                [255,255,255,255,255,255,255,255],
#                [0,0,0,0,0,0,0,0],
#                [0,0,0,0,0,0,0,0],
#                [0,0,0,0,0,0,0,0],
#                [0,0,0,0,0,0,0,0],
#                [0,0,0,0,0,0,0,0]],dtype=np.uint8)
# # data=np.zeros((32,32),dtype='uint8')
# im=Image.fromarray(data)
# str="/home/yu/图片/2.jpg"
# im.save(str,quality=100)
# import os

# root="/home/yu/公共的/FoRTE-FuzzBench/cjson/out_dir_AD1/train/1"
# filenames=os.listdir(root)

# names=[]

# for name in filenames:
#     fileName=name.split(',')[-1]  #hash值
#     names.append(fileName)

# fredict={}
# for x in names:
#     fredict[x]=fredict.get(x,0)+1

# # print(fredict)

# fd=open("/home/yu/图片/dict.txt","w")
# for i,k in enumerate(fredict):
#     if int(fredict[k]) > 10:
#         fd.write(str(k)+","+str(fredict[k])+"\n")
# fd.close()

import os


root='/home/ADFuzz/aflnet/tutorials/lightftp/out_dir/'

collectfile = root+'collection/1/'

trainPath = root+'train/1/'  #normal

testPath2 = root+'test/2/'   #abnormal

testPath1 = root+'test/1/'

def is_folder_empty(folder_path):
    # 检查文件夹是否存在
    if not os.path.exists(folder_path):
        return False

    # 检查文件夹是否为空
    if len(os.listdir(folder_path)) == 0:
        return True
    else:
        return False

def delete_files_in_folder(folder_path):
    # 检查文件夹是否存在
    if not os.path.exists(folder_path):
        return

    # 删除文件夹下的所有文件
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

def wait_fuzzer():

    #再次训练时，清空训练集和测试集

    # 判断文件夹是否为空
    if not is_folder_empty(trainPath):
        print("trainpath not none ")
        # 删除文件夹下的文件
        # delete_files_in_folder(trainPath)
    # 判断文件夹是否为空
    if not is_folder_empty(testPath2):
        print("testPath2 not none ")
        # 删除文件夹下的文件
        # delete_files_in_folder(testPath2)
    # 判断文件夹是否为空
    if not is_folder_empty(testPath1):
        print("testPath1 not none ")
        # 删除文件夹下的文件
        # delete_files_in_folder(testPath1)
    # 判断文件夹是否为空
    if not is_folder_empty(collectfile):
        print("collectfile not none ")
        # 删除文件夹下的文件
        # delete_files_in_folder(collectfile)


    #删除domain.txt确保第二次训练成功
    infoDir = "/home/ADFuzz/aflnet/tutorials/lightftp/output/ganomaly/testMain"
    if is_folder_empty(infoDir):
        file_name = os.path.join(infoDir, 'domain.txt')
        os.remove(file_name)


wait_fuzzer()