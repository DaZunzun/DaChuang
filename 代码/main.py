# -*- coding:utf-8 -*-
import random
import argparse
import os
import warnings
from datetime import datetime
import torch
from torch.backends import cudnn
from network.unet_cm import *
from network.ProbCM import *
from network.unet import *
from trainer import *
from dataset import *
from test import test_model

warnings.filterwarnings("ignore")


def main(configs):
    # load dataset
    centers = ['op_er', 'cl_di']  # 开放、随机侵蚀random erode and open   关闭、随机膨胀random dilate and close
    # centers = ['c1']
    train_loaders = {}
    vali_loaders = {}
    with open('/data/micca2018/test/loss.txt', 'a') as f:
        print('data loading', file=f)
    for center in centers:
        train, vali = train_val_split(configs.dataroot + center)
        train_loaders[center] = getTrainLoader(train, configs)
        vali_loaders[center] = getValiLoader(vali, configs)

    imgs = np.load(configs.dataroot + 'test/imgs.npy', allow_pickle=True)
    labs = np.load(configs.dataroot + 'test/labs.npy', allow_pickle=True)
    test_loaders = getTestLoader({'imgs': imgs, 'labs': labs})
    with open('/data/micca2018/test/loss.txt', 'a') as f:
        print('model choosing', file=f)
    if configs.mode == 'feddan':
        net = ProbCMNet(latent_dim=8,  ###
                        in_channels=1,
                        num_classes=2,
                        low_rank=False,
                        num_1x1_convs=3,
                        init_features=32,
                        lq=configs.lq)
    elif configs.mode == 'fedcm':
        net = UNet_CMs(1, 32, 4, 2, low_rank=False)  ###
    else:
        net = ResUnet(configs, num_cls=2)  ###

    cur_path = '/data/micca2018'
    date_time = current_time()
    SAVE_DIR = cur_path + '/result/' + configs.mode + '_' + configs.loss + date_time + '/'
    with open('/data/micca2018/test/loss.txt', 'a') as f:
        print('training', file=f)
    # training
    if configs.mode == 'fedavg':
        trainer = Trainer_FedAvg(net, train_loaders, vali_loaders, configs, SAVE_DIR)
    if configs.mode == 'feddan':
        trainer = Trainer_FedDAN(net, train_loaders, vali_loaders, configs, SAVE_DIR)
    if configs.mode == 'fedcm':
        trainer = Trainer_FedCM(net, train_loaders, vali_loaders, configs, SAVE_DIR)
    if configs.mode == 'single':
        trainer = Trainer_Single(net, train_loaders, vali_loaders, configs, SAVE_DIR, 'cl_di')
    elif configs.mode == 'ditto':
        trainer = Trainer_Ditto(net, train_loaders, vali_loaders, configs, SAVE_DIR)
    elif configs.mode == 'fedrep':
        trainer = Trainer_FedRep(net, train_loaders, vali_loaders, configs, SAVE_DIR)
    elif configs.mode == 'fedprox':
        trainer = Trainer_FedProx(net, train_loaders, vali_loaders, configs, SAVE_DIR)
    elif configs.mode == 'fedcurv':
        trainer = Trainer_FedCurv(net, train_loaders, vali_loaders, configs, SAVE_DIR)

    model = trainer.train()
    with open('/data/micca2018/test/loss.txt', 'a') as f:
        print('model saving', file=f)
    # save dir
    with open('/data/micca2018/test/loss.txt', 'a') as f:
        print(trainer.save_epoch, file=f)
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    torch.save(model, SAVE_DIR + 'model.pth')
    for client, local_model in trainer.models.items():
        torch.save(local_model, SAVE_DIR + 'model_{}.pth'.format(client))
    with open('/data/micca2018/test/loss.txt', 'a') as f:
        print('testing and test result saving', file=f)
    # testing
    test_model(model, test_loaders, SAVE_DIR)
    with open('/data/micca2018/test/loss.txt', 'a') as f:
        print('Train finished: ', current_time(), file=f)


def current_time():
    now = datetime.now()
    date_time = now.strftime("%m_%d_%Y, %H:%M:%S")
    return date_time


if __name__ == '__main__':
    # set parameters
    parser = argparse.ArgumentParser()

    # dataset param
    parser.add_argument('--dataroot', type=str, default='/data/micca2018/')
    parser.add_argument('--num_data', type=int, default=1000)
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=44)
    parser.add_argument('--num_workers', type=int, default=8)

    # network param
    parser.add_argument('--name', type=str, default='resnet18')
    parser.add_argument('--skip', type=bool, default=True)
    parser.add_argument('--hidden_dim', type=int, default=256, help='backbone feature')

    # train param
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lq', type=float, default=0.7)
    parser.add_argument('--weight', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--fine_tune', type=bool, default=False)
    parser.add_argument('--fine_tune_epoch', type=int, default=20)
    parser.add_argument('--local_epoch', type=int, default=10)
    parser.add_argument('--mode', type=str, default='feddan')
    parser.add_argument('--loss', type=str, default='CE')
    parser.add_argument('--cuda', type=int, default=0, )

    CONFIGs = parser.parse_args()

    # os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'
    torch.cuda.set_device(CONFIGs.cuda)
    # torch.cuda.set_device('PCI\VEN_10DE&DEV_1F9D&SUBSYS_15361025&REV_A1')

    cudnn.benchmark = True
    with open('/data/micca2018/test/loss.txt', 'a') as f:
        print(CONFIGs, file=f)
    main(CONFIGs)
