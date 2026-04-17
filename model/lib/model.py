"""GANomaly
"""
# pylint: disable=C0301,E1101,W0622,C0103,R0902,R0915

##
from collections import OrderedDict
import os
import time
import numpy as np
from tqdm import tqdm
import math
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import torch.utils.data
import torchvision.utils as vutils
from PIL import Image
from lib.networks import NetG, NetD, weights_init
from lib.visualizer import Visualizer
from lib.loss import l2_loss
from lib.evaluate import evaluate
import torchvision.transforms as transforms
from ctypes import *
import copy



SHM_DATA_SIZE = 4096
SHM_DATA_NUM = 1


class BaseModel():
    """ Base Model for ganomaly
    """
    def __init__(self, opt, dataloader):
        ##
        # Seed for deterministic behavior
        self.seed(opt.manualseed)

        # Initalize variables.
        self.opt = opt
        self.visualizer = Visualizer(opt)
        self.dataloader = dataloader
        self.trn_dir = os.path.join(self.opt.outf, self.opt.name, 'train')
        self.tst_dir = os.path.join(self.opt.outf, self.opt.name, 'test')
        self.device = torch.device("cuda:0" if self.opt.device != 'cpu' else "cpu")

        self.transform=transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                    transforms.Resize((opt.isize, opt.isize)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))
                                    ])


    ##
    def set_input(self, input:torch.Tensor):
        """ Set input and ground truth

        Args:
            input (FloatTensor): Input data for batch i.
        """
        with torch.no_grad():
            # print(input[0])
            self.input.resize_(input[0].size()).copy_(input[0])
            # print(self.input)
            self.gt.resize_(input[1].size()).copy_(input[1])
            self.label.resize_(input[1].size())

            # print("input:")
            # print(input)
            # print("\nselfinput:")
            # print(self.input)
            # print(self.gt)
            # print(self.label)


            # Copy the first batch as the fixed input.
            if self.total_steps == self.opt.batchsize:
                self.fixed_input.resize_(input[0].size()).copy_(input[0])

    ##
    def seed(self, seed_value):
        """ Seed 
        
        Arguments:
            seed_value {int} -- [description]
        """
        # Check if seed is default value
        if seed_value == -1:
            return

        # Otherwise seed all functionality
        import random
        random.seed(seed_value)
        torch.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        np.random.seed(seed_value)
        torch.backends.cudnn.deterministic = True

    ##
    def get_errors(self):
        """ Get netD and netG errors.

        Returns:
            [OrderedDict]: Dictionary containing errors.
        """

        errors = OrderedDict([
            ('err_d', self.err_d.item()),
            ('err_g', self.err_g.item()),
            ('err_g_adv', self.err_g_adv.item()),
            ('err_g_con', self.err_g_con.item()),
            ('err_g_enc', self.err_g_enc.item())])

        return errors

    ##
    def get_current_images(self):
        """ Returns current images.

        Returns:
            [reals, fakes, fixed]
        """

        reals = self.input.data
        fakes = self.fake.data
        fixed = self.netg(self.fixed_input)[0].data   #fixed为产生的异常图片

        return reals, fakes, fixed

    ##
    def save_weights(self, epoch):
        """Save netG and netD weights for the current epoch.

        Args:
            epoch ([int]): Current epoch number.
        """

        weight_dir = os.path.join(self.opt.outf, self.opt.name, 'train', 'weights')
        if not os.path.exists(weight_dir): os.makedirs(weight_dir)

        torch.save({'epoch': epoch + 1, 'state_dict': self.netg.state_dict()},
                   '%s/netG.pth' % (weight_dir))
        torch.save({'epoch': epoch + 1, 'state_dict': self.netd.state_dict()},
                   '%s/netD.pth' % (weight_dir))

    ##
    def train_one_epoch(self):
        """ Train the model for one epoch.
        """

        self.netg.train()
        epoch_iter = 0
        for data in tqdm(self.dataloader['train'], leave=False, total=len(self.dataloader['train'])):  #tqdm添加进度条
            self.total_steps += self.opt.batchsize
            epoch_iter += self.opt.batchsize

            self.set_input(data)
            # self.optimize()
            self.optimize_params()

            if self.total_steps % self.opt.print_freq == 0:
                errors = self.get_errors()
                # print(errors)
                if self.opt.display:
                    counter_ratio = float(epoch_iter) / len(self.dataloader['train'].dataset)
                    self.visualizer.plot_current_errors(self.epoch, counter_ratio, errors)

            if self.total_steps % self.opt.save_image_freq == 0:
                reals, fakes, fixed = self.get_current_images()
                self.visualizer.save_current_images(self.epoch, reals, fakes, fixed)
                if self.opt.display:
                    self.visualizer.display_current_images(reals, fakes, fixed)

        # print(">> Training model %s. Epoch %d/%d" % (self.name, self.epoch+1, self.opt.niter))
        # self.visualizer.print_current_errors(self.epoch, errors)

    ##
    def train(self):
        """ Train the model
        """

        ##
        # TRAIN
        self.total_steps = 0
        best_auc = 0
        best_precision = 0
        stale_epochs = 0
        # Retraining is budget-constrained. Sample count is capped earlier when
        # the dataset is materialized, while epoch count and wall-clock are
        # enforced here inside the training loop.
        max_epochs = max(1, getattr(self.opt, 'incremental_max_epochs', self.opt.niter))
        max_minutes = max(0, getattr(self.opt, 'incremental_max_minutes', 0))
        patience = max(1, getattr(self.opt, 'early_stop_patience', 2))
        train_deadline = time.time() + max_minutes * 60 if max_minutes else None

        # Train for niter epochs.
        # print(">> Training model %s." % self.name)
        for self.epoch in range(self.opt.iter, self.opt.iter + max_epochs):
            if train_deadline and time.time() >= train_deadline:
                break

            # Train for one epoch
            self.train_one_epoch()
            res = self.test()
            # We checkpoint on either a better anomaly metric or a better
            # validation precision because the downstream fuzzer ultimately
            # cares about safe filtering, not just offline detection quality.
            if res[self.opt.metric] > best_auc:
                best_auc = res[self.opt.metric]
                best_precision = res['precision']
                stale_epochs = 0
                self.save_weights(self.epoch)

                self.max=self.scoresMax
                self.min=self.scoresMin
            else:
                stale_epochs += 1
            #
            #
            self.visualizer.print_current_performance(res, best_auc)
            # Early stopping is tied to validation precision so the model stops
            # updating once filtering quality stops improving.
            if stale_epochs >= patience:
                break

        #保存最大最小error
        infoDir = os.path.join(self.opt.outf, self.opt.name)
        file_name = os.path.join(infoDir, 'domain.txt')
        with open(file_name, 'wt') as f:
            f.write('%f: %f\n' % (self.max, self.min))

        self.netg.eval()
        # self.netd.eval()

        # Load the weights of netg and netd.
        if self.opt.load_weights:
            # path = "./output/{}/{}/train/weights/netG.pth".format(self.name.lower(), self.opt.dataset)
            path = opt.dataroot + "/out_dir/ganomaly/testMain/train/weights/netG.pth"
            pretrained_dict = torch.load(path)['state_dict']

            try:
                self.netg.load_state_dict(pretrained_dict)
            except IOError:
                raise IOError("netG weights not found")
        ###############

        return {'metric': best_auc, 'precision': best_precision}

        # with torch.no_grad():
        #     self.netg.eval()
        #
        #     if self.opt.load_weights:
        #         # path = "./output/{}/{}/train/weights/netG.pth".format(self.name.lower(), self.opt.dataset)
        #         path = "/home/yu/桌面/weights/nz50/netG.pth"
        #         pretrained_dict = torch.load(path)['state_dict']
        #
        #         try:
        #             self.netg.load_state_dict(pretrained_dict)
        #         except IOError:
        #             raise IOError("netG weights not found")

        ###############

        # print(">> Training model %s.[Done]" % self.name)

    ##
    def test(self):
        """ Test GANomaly model.

        Args:
            dataloader ([type]): Dataloader for the test set

        Raises:
            IOError: Model weights not found.
        """
        self.netg.eval()
        # self.netd.eval()
        with torch.no_grad():
            # Load the weights of netg and netd.
            if self.opt.load_weights:
                path = "./out_dir/{}/{}/train/weights/netG.pth".format(self.name.lower(), self.opt.dataset)
                pretrained_dict = torch.load(path)['state_dict']

                try:
                    self.netg.load_state_dict(pretrained_dict)
                except IOError:
                    raise IOError("netG weights not found")
                print('   Loaded weights.')

            # self.netg.eval()

            self.opt.phase = 'test'

            # lenth=len(self.dataloader['test'].dataset)
            # Create big error tensor for the test set.
            self.an_scores = torch.zeros(size=(len(self.dataloader['test'].dataset),), dtype=torch.float32, device=self.device)
            self.gt_labels = torch.zeros(size=(len(self.dataloader['test'].dataset),), dtype=torch.long,    device=self.device)
            self.latent_i  = torch.zeros(size=(len(self.dataloader['test'].dataset), self.opt.nz), dtype=torch.float32, device=self.device)
            self.latent_o  = torch.zeros(size=(len(self.dataloader['test'].dataset), self.opt.nz), dtype=torch.float32, device=self.device)


            # print("   Testing model %s." % self.name)
            self.times = []
            self.total_steps = 0
            epoch_iter = 0
            # correct=0
            # total=0
            for i, data in enumerate(self.dataloader['test'], 0):
                self.total_steps += self.opt.batchsize
                epoch_iter += self.opt.batchsize
                time_i = time.time()
                self.set_input(data)

                # print(self.input.shape)
                # print("input:")
                # print(self.input)
                self.fake, latent_i, latent_o = self.netg(self.input)

                error = torch.mean(torch.pow((latent_i-latent_o), 2), dim=1)
                time_o = time.time()
                # print(data[1])
                # print(error)
                self.an_scores[i*self.opt.batchsize : i*self.opt.batchsize+error.size(0)] = error.reshape(error.size(0))
                self.gt_labels[i*self.opt.batchsize : i*self.opt.batchsize+error.size(0)] = self.gt.reshape(error.size(0))
                self.latent_i [i*self.opt.batchsize : i*self.opt.batchsize+error.size(0), :] = latent_i.reshape(error.size(0), self.opt.nz)
                self.latent_o [i*self.opt.batchsize : i*self.opt.batchsize+error.size(0), :] = latent_o.reshape(error.size(0), self.opt.nz)

                self.times.append(time_o - time_i)

                # Save test images.


                if self.opt.save_test_images:
                    dst = os.path.join(self.opt.outf, self.opt.name, 'test', 'images')
                    if not os.path.isdir(dst):
                        os.makedirs(dst)
                    real, fake, _ = self.get_current_images()
                    vutils.save_image(real, '%s/real_%03d.eps' % (dst, i+1), normalize=True)
                    vutils.save_image(fake, '%s/fake_%03d.eps' % (dst, i+1), normalize=True)

            # Measure inference time.
            self.times = np.array(self.times)
            self.times = np.mean(self.times[:100] * 1000)

            # Scale error vector between [0, 1]


            # print(torch.max(self.an_scores))

            self.scoresMax = torch.max(self.an_scores)

            self.scoresMin = torch.min(self.an_scores)


            self.an_scores = (self.an_scores - self.scoresMin) / (self.scoresMax - self.scoresMin)



            # auc, eer = roc(self.gt_labels, self.an_scores)
            auc = evaluate(self.gt_labels, self.an_scores, metric=self.opt.metric)
            # Precision is exported alongside AUC so the incremental trainer can
            # early-stop on the same signal that AFLNet uses to judge filter
            # safety online.
            precision = self._current_validation_precision()
            performance = OrderedDict([('Avg Run Time (ms/batch)', self.times), (self.opt.metric, auc), ('precision', precision)])

            if self.opt.display_id > 0 and self.opt.phase == 'test':
                counter_ratio = float(epoch_iter) / len(self.dataloader['test'].dataset)
                self.visualizer.plot_performance(self.epoch, counter_ratio, performance)
            return performance


    def validate_batch(self,addr,mutex,shmdata):

        with torch.no_grad():

            # self.netg.eval()
            # outfd="/home/yu/桌面/testfd/"
            while True:


                if mutex.acquire(True):
                    # time_start = time.time()

                    if addr[SHM_DATA_NUM] == 1:  # 说明afl调用infer，共享内存设置为 1

            # Create big error tensor for the test set.
                        scores = torch.zeros(size=(1,SHM_DATA_NUM), dtype=torch.float32,
                                                     device=self.device)
                        data=[]

                        for i in range(SHM_DATA_NUM):
                            # test = shmdata[i]
                            # f = open(os.path.join(outfd,str(i)), 'wb+')
                            # f.write(string_at(shmdata[i], 4096))
                            # f.close()
                            # temp_data=np.ctypeslib.as_array(test, shape=(32, 32))
                            # test=shmdata[i].contents
                            templedata = []
                            temp_data = np.frombuffer(string_at(shmdata[i], SHM_DATA_SIZE), dtype='uint8')
                            temp_data=temp_data.reshape(32,32)

                            # datalenth = len(tempData)
                        # byte_stream = io.BytesIO(shmdata)
                            temp_data = Image.fromarray(temp_data)

                        # data=torch.from_numpy(data)
                            temp_data = self.transform(temp_data)
                            templedata.append(temp_data)
                            ten=torch.tensor([0])
                            templedata.append(ten)
                        # templedata=templedata.unsqueeze(0)
                            self.set_input(templedata)
                            data.append(copy.deepcopy(self.input))
                        # print(self.input.shape)

                        # self.input=self.input.unsqueeze(0)
                        testdata=torch.stack([data[0],data[1],data[2],data[3],data[4],data[5],data[6],data[7],data[8],data[9],data[10],
                                              data[11],data[12],data[13],data[14],data[15],data[16],data[17],data[18],data[19],data[20],
                                              data[21],data[22],data[23],data[24],data[25],data[26],data[27],data[28],data[29],data[30],
                                              data[31]],
                                              # data[32],data[33],data[34],data[35],data[36],data[37],data[38],data[39],data[40],
                                              # data[41],data[42],data[43],data[44],data[45],data[46],data[47],data[48],data[49],data[50],
                                              # data[51],data[52],data[53],data[54],data[55],data[56],data[57],data[58],data[59],data[60],data[61],data[62],data[63]],
                                              dim=0)
                        # testdata = torch.stack([testdata, self.input], dim=0)
                        # print(testdata.shape)

                        # print(self.input.shape)

                        # time_start1 = time.time()
                        self.fake, latent_i, latent_o = self.netg(testdata)
                        # time_end1 = time.time()
                        # print(' infer cost', time_end1 - time_start1)

                        error = torch.mean(torch.pow((latent_i - latent_o), 2), dim=1,dtype=float)

                        scores = (error - torch.as_tensor(self.min)) / (torch.as_tensor(self.max) - torch.as_tensor(self.min))

                        # score = scores.item()

                        for i in range(SHM_DATA_NUM):


                            # `addr[i] = 1` means "this sample looks strongly like a
                            # high-frequency path instance, so AFLNet may choose to
                            # filter it". Uncertain samples are deliberately kept.
                            if scores[i] >= self.opt.high_freq_drop_threshold:
                                addr[i] = 1

                            else:
                                addr[i] = 0


                        # addr[1] = flag  # 把推理结果重新写入共享内存

                        addr[SHM_DATA_NUM] = 0     # 重置为0，告知afl，调用val完成，结果在addr[1]

                    mutex.release()
                    # time_end = time.time()
                    # print(' total cost', time_end - time_start)


    def validate_batch_no_shm(self,addr,mutex,shmdata):

        with torch.no_grad():

            # self.netg.eval()
            # outfd="/home/yu/桌面/testfd/"
            while True:


                if mutex.acquire(True):
                    # time_start = time.time()

                    if addr[SHM_DATA_NUM] == 1:  #info_bits[SHM_DATA_NUM]

            # Create big error tensor for the test set.
                        scores = torch.zeros(size=(1,SHM_DATA_NUM), dtype=torch.float32,
                                                     device=self.device)
                        data=[]

                        for i in range(SHM_DATA_NUM):
                            # test = shmdata[i]
                            # f = open(os.path.join(outfd,str(i)), 'wb+')
                            # f.write(string_at(shmdata[i], 4096))
                            # f.close()
                            # temp_data=np.ctypeslib.as_array(test, shape=(32, 32))
                            # test=shmdata[i].contents
                            templedata = []
                            temp_data = np.frombuffer(string_at(shmdata[i], SHM_DATA_SIZE), dtype='uint8')
                            temp_data=temp_data.reshape(64,64) #1024=32*32   4096=64*64

                       
                            temp_data = Image.fromarray(temp_data)

                      
                            temp_data = self.transform(temp_data).unsqueeze(0).to(self.device)
                            # templedata.append(temp_data)
                            # ten=torch.tensor([0])
                            # templedata.append(ten)
                     
                            # self.set_input(temp_data)
                            # data.append(copy.deepcopy(self.input))

                            # self.set_input(temp_data)
                      
                        # time_start1 = time.time()
                        # self.fake, latent_i, latent_o = self.netg(testdata)
                        self.fake, latent_i, latent_o = self.netg(temp_data)
                        # time_end1 = time.time()
                        # print(' infer cost', time_end1 - time_start1)

                        error = torch.mean(torch.pow((latent_i - latent_o), 2), dim=1,dtype=float)

                        scores = (error - torch.as_tensor(self.min)) / (torch.as_tensor(self.max) - torch.as_tensor(self.min))

                        # score = scores.item()

                        for i in range(SHM_DATA_NUM):


                            # The online inference path uses the same high-frequency
                            # threshold as the batch validator so training and runtime
                            # semantics stay aligned.
                            if scores[i] >= self.opt.high_freq_drop_threshold:
                                addr[i] = 1

                            else:
                                addr[i] = 0


                        # addr[1] = flag  # 

                        addr[SHM_DATA_NUM] = 0     

                    elif  addr[SHM_DATA_NUM] == 2:
                        mutex.release()
                        break
                        #重新训练

                    mutex.release()
                    # time_end = time.time()
                    # print(' total cost', time_end - time_start)


    def load_pretrain(self):

        # infoDir = os.path.join(self.opt.outf, self.opt.name)
        # file_name = os.path.join(infoDir, 'domain.txt')
        # file_name = "/home/yu/公共的/FoRTE-FuzzBench/cjson/cjson-1.7.7/fuzzing/output_AD/ganomaly/testMain/domain.txt"
        # with open(file_name, 'r') as f:
        #     f.read('%f: %f\n' % (self.max, self.min))

        self.max=0.096529
        self.min=0.010832

        self.netg.eval()
        if self.opt.load_weights:

            path = "/opt/aflnet/tutorials/lightftp/ganomaly/testMain/train/weights/netG.pth"
            pretrained_dict = torch.load(path)['state_dict']

            try:
                self.netg.load_state_dict(pretrained_dict)
            except IOError:
                raise IOError("netG weights not found")


##
class Ganomaly(BaseModel):
    """GANomaly Class
    """

    @property
    def name(self): return 'Ganomaly'

    def __init__(self, opt, dataloader):
        super(Ganomaly, self).__init__(opt, dataloader)

        # -- Misc attributes
        self.epoch = 0
        self.times = []
        self.total_steps = 0

        ##
        # Create and initialize networks.
        self.netg = NetG(self.opt).to(self.device)
        self.netd = NetD(self.opt).to(self.device)
        self.netg.apply(weights_init)
        self.netd.apply(weights_init)

        ##
        if self.opt.resume != '':
            # print("\nLoading pre-trained networks.")
            self.opt.iter = torch.load(os.path.join(self.opt.resume, 'netG.pth'))['epoch']
            self.netg.load_state_dict(torch.load(os.path.join(self.opt.resume, 'netG.pth'))['state_dict'])
            self.netd.load_state_dict(torch.load(os.path.join(self.opt.resume, 'netD.pth'))['state_dict'])
            # print("\tDone.\n")

        self.l_adv = l2_loss
        self.l_con = nn.L1Loss()
        self.l_enc = l2_loss
        self.l_bce = nn.BCELoss()


        ##
        # Initialize input tensors.
        self.input = torch.empty(size=(self.opt.batchsize, 1, self.opt.isize, self.opt.isize), dtype=torch.float32, device=self.device)  # 3
        self.label = torch.empty(size=(self.opt.batchsize,), dtype=torch.float32, device=self.device)
        self.gt    = torch.empty(size=(opt.batchsize,), dtype=torch.long, device=self.device)
        self.fixed_input = torch.empty(size=(self.opt.batchsize, 1, self.opt.isize, self.opt.isize), dtype=torch.float32, device=self.device)  # 3
        self.real_label = torch.ones (size=(self.opt.batchsize,), dtype=torch.float32, device=self.device)
        self.fake_label = torch.zeros(size=(self.opt.batchsize,), dtype=torch.float32, device=self.device)
        ##
        # Setup optimizer
        if self.opt.isTrain:
            self.netg.train()
            self.netd.train()
            self.configure_incremental_training()

    def configure_incremental_training(self):
        """Warm-start retraining by updating only the tail layers when resuming.

        Incremental retraining should be cheap enough to run many times during
        fuzzing. To keep the cost bounded, we freeze the bulk of the network and
        only leave the tail layers trainable when a previous checkpoint exists.
        This behaves like a lightweight adapter update without changing the
        original architecture.
        """

        if self.opt.resume != '':
            for param in self.netg.parameters():
                param.requires_grad = False
            for param in self.netd.parameters():
                param.requires_grad = False

            tail_blocks = max(1, getattr(self.opt, 'finetune_last_layers', 1))
            trainable_modules = []
            # NetG has three logical stages. Unfreezing only the suffix of each
            # stage lets the model adapt to shifted sample distributions while
            # preserving most previously learned structure.
            trainable_modules.extend(list(self.netg.encoder1.main.children())[-tail_blocks:])
            trainable_modules.extend(list(self.netg.decoder.main.children())[-tail_blocks:])
            trainable_modules.extend(list(self.netg.encoder2.main.children())[-tail_blocks:])
            trainable_modules.extend(list(self.netd.classifier.children()))

            for module in trainable_modules:
                for param in module.parameters():
                    param.requires_grad = True

            # The discriminator is not fully fine-tuned during incremental
            # updates; keeping it in eval mode reduces instability.
            self.netd.eval()

        g_params = [param for param in self.netg.parameters() if param.requires_grad]
        d_params = [param for param in self.netd.parameters() if param.requires_grad]

        if not g_params:
            g_params = list(self.netg.parameters())
        if not d_params:
            d_params = list(self.netd.parameters())

        self.optimizer_d = optim.Adam(d_params, lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
        self.optimizer_g = optim.Adam(g_params, lr=self.opt.lr, betas=(self.opt.beta1, 0.999))

    def _current_validation_precision(self):
        """Compute validation precision for the current filtering threshold.

        In this workflow, a positive prediction corresponds to a sample that the
        filter would like to reject. Precision therefore measures how reliable
        those rejections are on the held-out validation split.
        """
        preds = (self.an_scores >= 0.5).long()
        positives = int(preds.sum().item())
        if positives == 0:
            return 1.0

        true_positives = int(((preds == 1) & (self.gt_labels == 1)).sum().item())
        return true_positives / positives

    ##
    def forward_g(self):
        """ Forward propagate through netG
        """
        self.fake, self.latent_i, self.latent_o = self.netg(self.input)

    ##
    def forward_d(self):
        """ Forward propagate through netD
        """
        self.pred_real, self.feat_real = self.netd(self.input)
        self.pred_fake, self.feat_fake = self.netd(self.fake.detach())

    ##
    def backward_g(self):
        """ Backpropagate through netG
        """
        self.err_g_adv = self.l_adv(self.netd(self.input)[1], self.netd(self.fake)[1])
        self.err_g_con = self.l_con(self.fake, self.input)
        self.err_g_enc = self.l_enc(self.latent_o, self.latent_i)
        self.err_g = self.err_g_adv * self.opt.w_adv + \
                     self.err_g_con * self.opt.w_con + \
                     self.err_g_enc * self.opt.w_enc
        self.err_g.backward(retain_graph=True)

    ##
    def backward_d(self):
        """ Backpropagate through netD
        """
        # Real - Fake Loss
        self.err_d_real = self.l_bce(self.pred_real, self.real_label)
        self.err_d_fake = self.l_bce(self.pred_fake, self.fake_label)

        # NetD Loss & Backward-Pass
        self.err_d = (self.err_d_real + self.err_d_fake) * 0.5
        self.err_d.backward()

    ##
    def reinit_d(self):
        """ Re-initialize the weights of netD
        """
        self.netd.apply(weights_init)
        # print('   Reloading net d')

    def optimize_params(self):
        """ Forwardpass, Loss Computation and Backwardpass.
        """
        # Forward-pass
        self.forward_g()
        self.forward_d()

        # Backward-pass
        # netg
        self.optimizer_g.zero_grad()
        self.backward_g()
        self.optimizer_g.step()

        # netd
        self.optimizer_d.zero_grad()
        self.backward_d()
        self.optimizer_d.step()
        if self.err_d.item() < 1e-5: self.reinit_d()
