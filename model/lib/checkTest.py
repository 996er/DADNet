import numpy as np

from torch.utils.data import Dataset
import torch
import os
import os.path
import sys
import math
from tqdm import tqdm
import torchvision.transforms as transforms
from PIL import Image

class GetData(Dataset):
    def __init__(self, filePath,transform=None):  # 得到名字list
        super(GetData, self).__init__()
        self.root = filePath                       #路径
        self.fileName = os.listdir(self.root)      #文件名
        self.transform = transform

        classes, class_to_idx = self._find_classes(self.root)  #类和id



        samples = self.make_dataset(self.root, class_to_idx)   #返回  (path,class_to_idx[target])

        if len(samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.root + "\n"))

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples=samples     #path
        self.targets = [s[1] for s in samples]  #idx

    def _find_classes(self, dir):
        """
        Finds the class folders in a dataset.

        Args:
            dir (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """
        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def make_dataset(self, dir, class_to_idx):
        images = []
        for target in sorted(class_to_idx.keys()):
            d = os.path.join(dir, target)
            if not os.path.isdir(d):
                continue
            for root, _, fnames in sorted(os.walk(d)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)

        return images

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):  # 按名取图,index对应批次
        path, target = self.samples[index]

        with open(path, 'rb') as f:

            filelenth = os.path.getsize(path)
            arraylenth = math.ceil(math.sqrt(filelenth))
            dataArr=np.zeros((arraylenth, arraylenth), dtype='uint8')

            tempData = np.frombuffer(f.read(), dtype='uint8')

            for i in range(filelenth):

                line = i // arraylenth
                row = i % arraylenth
                dataArr[line][row] = tempData[i]


            data = Image.fromarray(dataArr)
            data = self.transform(data)

        sample = data


        return sample, target


def loadData(opt):
    dataroot = '/home/ADFuzz/aflnet/tutorials/lightftp/{}'.format('testMain')

    splits = ['train', 'test']
    drop_last_batch = {'train': True, 'test': False}
    shuffle = {'train': True, 'test': True}

    transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                    transforms.Resize(opt.isize),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))
                                    ])

    dataset = {x: GetData(os.path.join(dataroot, x), transform) for x in splits}



    dataloader = {x: torch.utils.data.DataLoader(dataset=dataset[x],
                                                 batch_size=opt.batchsize,
                                                 shuffle=shuffle[x],
                                                 num_workers=8,
                                                 drop_last=drop_last_batch[x],
                                                 )
                  for x in splits}


    return dataloader



def loadTestData(opt):
    # dataroot = '/home/yu/公共的/FoRTE-FuzzBench/{}/out_dir/'.format('libjpeg')
    dataroot = opt.dataroot

    # splits = ['train', 'test', 'val']
    # drop_last_batch = {'train': True, 'test': False, 'val': False}
    # shuffle = {'train': True, 'test': True, 'val': False}

    splits = ['train', 'test']
    drop_last_batch = {'train': True, 'test': False}
    shuffle = {'train': True, 'test': True}

    # splits = ['train']
    # drop_last_batch = {'train': True}
    # shuffle = {'train': True}

    transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                    transforms.Resize(opt.isize),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))
                                    ])

    # testset = GetData(os.path.join(dataroot, 'val'), transform)
    dataset = {x: GetData(os.path.join(dataroot, x), transform) for x in splits}

    # dataloader =  torch.utils.data.DataLoader(dataset=testset,
    #                                              batch_size=opt.batchsize,
    #                                              shuffle=True,
    #                                              num_workers=0,
    #                                              drop_last=False,
    #                                              )

    dataloader = {x: torch.utils.data.DataLoader(dataset=dataset[x],
                                                 batch_size=opt.batchsize,
                                                 shuffle=shuffle[x],
                                                 num_workers=0,   #先试试0
                                                 drop_last=drop_last_batch[x],
                                                 )
                  for x in splits}



    return dataloader


