from ctypes import *
import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch.utils.data
import sys
from binascii import hexlify
from PIL import Image
import threading
# def info(array):
#
#     print(array)
#
#     tempData = np.frombuffer(array, dtype='uint8')
#
#     print(tempData)
#
#
# TenIntegers = c_int * 10
# # addr=c_int(123)
# # addr=123
# ii = TenIntegers(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
#
# ptr = POINTER(c_int)
#
# # addr=[123]
#
# ptr = ii
#
#
#
# info(ptr)
# mutex = threading.Lock()

# SHM_SIZE = 2
# SHM_ID = 123456789
#
#
# try:
#     rt = CDLL('librt.so')
#
# except:
#     rt = CDLL('librt.so.1')
#
# shmget = rt.shmget
# shmget.argtypes = [c_int, c_size_t, c_int]
# shmget.restype = c_int
#
# shmat = rt.shmat
# shmat.argtypes = [c_int, POINTER(c_void_p), c_int]
# shmat.restype = POINTER(c_int)
#
# shmid = shmget(SHM_ID, 1024, 0O644)
#
# if shmid < 0:
#
#     sys.exit()
#
# else:
#     addr = shmat(shmid, None, 0)
#
# # for i in range(100):
# #     print(addr[i])
#
# print(id(addr))
# # while True:
# data=np.ctypeslib.as_array(addr,shape=(32,32))
# print(data)
# data = Image.fromarray(data,'L')
# data.save("/home/yu/公共的/FoRTE-FuzzBench/libjpeg/test.jpg")
# # tempData = np.frombuffer(addr, dtype='uint8',count=-1, offset=0)
#     # if mutex.acquire(True):
# print(hexlify(string_at(addr,15)))

# print(tempData)
        # mutex.release()
# data=[]
# for i in range(8):
#     temp = [[1, 2, 3], [12, 3, 1], [1, 1, 1]]
#     temp = torch.tensor(temp)
#     data.append(temp)


# data2=[[1,2,3],[12,3,1],[1,1,1]]
# data3=[[1,2,3],[12,3,1],[1,1,1]]
# data4=[[1,2,3],[12,3,1],[1,1,1]]
# temp1=torch.tensor(data1)
# temp2=torch.tensor(data2)
# temp3=torch.tensor(data3)
# temp4=torch.tensor(data4)
#
# testdata=torch.stack([temp1,temp2], dim=0)
# testdata=torch.stack([data[0],data[1],data[2],data[3],data[4],data[5],data[6],data[7]], dim=0)
# test=[[1],[2],[3],[4],[5]]
# scores = torch.zeros(size=(1,8), dtype=torch.float32 )
# test=torch.tensor(test)
# print(test)
# print(test.size(0))
# test=test.reshape(test.size(0))
#
# scores = (test-torch.min(test))/(torch.max(test)-torch.min(test))
# # print(test)
#
# addr=[]
# for i in range(5):
#     if scores[i] > 0 :
#         print(1)
#     else:
#         print(0)
# print(addr)
import torch
import os
import os.path
import sys
import math

path="/home/yu/桌面/testfd/0"

with open(path, 'rb') as f:
    filelenth = os.path.getsize(path)
    arraylenth = math.ceil(math.sqrt(filelenth))
    dataArr = np.zeros((arraylenth, arraylenth), dtype='uint8')

    tempData = np.frombuffer(f.read(), dtype='uint8')

    for i in range(filelenth):
        line = i // arraylenth
        row = i % arraylenth
        dataArr[line][row] = tempData[i]

    data = Image.fromarray(dataArr)
