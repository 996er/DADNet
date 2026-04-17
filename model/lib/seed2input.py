# import os
# import torch
# import numpy as np
#
# from torch.autograd import Variable
# import torchvision.datasets as datasets
# from picTest import data2array
# from torchvision.datasets import ImageFolder
# import torchvision.transforms as transforms
#
#
#
# def load_data():
#     manualseed = -1
#     dataset='jpeg'
#     dataroot = './data/{}'.format(dataset)
#     splits = ['train', 'test']
#     drop_last_batch = {'train': True, 'test': False}
#     shuffle = {'train': True, 'test': True}
#
#     transform = transforms.Compose([
#                                     transforms.Resize(32),
#                                     transforms.CenterCrop(32),
#                                     transforms.ToTensor(),
#                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ])
#
#
#
#     # dataset = {x: ImageFolder(os.path.join(dataroot, x), transform) for x in splits}
#     dataset = {x: ImageFolder(os.path.join(dataroot, x), transform) for x in splits}
#
#     dataloader = {x: torch.utils.data.DataLoader(dataset=dataset[x],
#                                                  batch_size=3,
#                                                  shuffle=shuffle[x],
#                                                  num_workers=int(3),
#                                                  drop_last=drop_last_batch[x])
#                   for x in splits}
#
#
#     # Jpegdataset = datasets.ImageFolder('./data/jpeg/train',
#     #                                            transform=transform)
#     # dataloader = torch.utils.data.DataLoader(Jpegdataset,
#     #                                              batch_size=1, shuffle=True,
#     #                                              num_workers=1)
#
#
#     #for epoch in range(2):
#     for data in dataloader:
#         print(data)
#             # labels, inputs = data
#             # print(type(inputs))
#             # print(type(labels))
#             #inputs, labels = Variable(inputs), Variable(labels)
#         #data =Variable(data)
#         #labels=data.target
#         #   inputs=data.data
#         #print(labels)
#
#             # print(type(inputs))
#             # print(type(labels))
#        # print(inputs)
#
#             # print("!!!")
#             # print(labels)
#
#
#     return dataloader
#
# if __name__ == '__main__':
#     load_data()


# import math
#
# import os
# lenth=os.path.getsize('/home/yu/公共的/ganomaly/data/testMain/test/-1166893707/id:000003,src:000001,op:flip2,pos:1,+cov,-1166893707')
#
# print(lenth)
# x=1800
# y=math.sqrt(x)
# print(y)
#
# print(math.ceil(y))
import os
import time


def new_report(test):
    i=0
    for root,dir,files in os.walk(test):
        with open(os.path.join(root, "time"), 'w') as f:
            for file in files:
                full_path=os.path.join(root,file)
                mtime=os.stat(full_path).st_mtime
                file_modify_time=time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(mtime))
                i=i+1
                f.write('%s\n' % (file_modify_time))
    print(i)
test=r"/home/yu/公共的/FoRTE-FuzzBench/binutils/out_dir/queue"
new_report(test)