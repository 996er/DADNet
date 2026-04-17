import sys

import torch

from lib.data import  load_data

from lib.model import Ganomaly
from options import Options
from ctypes import *
import time


def validate():

    opt = Options().parse()

    dataloader = load_data(opt)

    # time_start = time.time()

    model = Ganomaly(opt, dataloader)
    # time_end = time.time()

    # print(' cost', time_end - time_start)

    # time_start = time.time()

    # model.validate(addr,mutex,shmdata)
    model.train()
    # time_end = time.time()

    # print(' inference cost', time_end - time_start)

    # return tem




if __name__ == '__main__':
    validate()



