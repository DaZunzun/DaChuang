# -*- coding:utf-8 -*-


import torch
import logging, os
from itertools import cycle
from torch.optim.lr_scheduler import LRScheduler  #规划学习率变化的模块
from collections import defaultdict
from evaluation import *
import torch.nn.functional as F
import numpy as np
from losses import *
import copy

class Poly(LRScheduler):  #制定学习率的变化规则
    def __init__(self, optimizer, num_epochs, iters_per_epoch, warmup_epochs=10, last_epoch=-1):

        self.iters_per_epoch = iters_per_epoch   #每轮训练迭代的次数

        self.cur_iter = 0   #当前训练轮次内是第几次迭代

        self.N = num_epochs * iters_per_epoch   #总的迭代次数

        self.warmup_iters = warmup_epochs * iters_per_epoch   #热身的迭代次数

        super(Poly, self).__init__(optimizer, last_epoch)   #optimizer优化器  last_epoch 上一轮是第几轮

    def get_lr(self):

        T = self.last_epoch * self.iters_per_epoch + self.cur_iter    #总共进行了几次迭代

        factor =  pow((1 - 1.0 * T / self.N), 0.9)   #非热身阶段 的学习率变化因子

        if self.warmup_iters > 0 and T < self.warmup_iters:  #处于热身阶段 的学习率变化因子

            factor = 1.0 * T / self.warmup_iters

        self.cur_iter %= self.iters_per_epoch
        self.cur_iter += 1     #cur_iter自增

        assert factor >= 0, 'error in lr_scheduler' # 检查：factor 必须 >=0

        return [base_lr * factor for base_lr in self.base_lrs]    #所有学习率都乘factor


class Trainer_FedAvg(object):    #经典的联邦学习算法
    def __init__(self,model,data_loader_train,data_loader_vali,config,save_dir):

        # models
        # models:字典类型 --> 键:(client)各个中心的名字   值:(model)模型,其中不同中心的模型有各自的数据加载器(data)
        # #ps:每个中心的模型都相同，但是数据各不相同，为属于各自中心的数据
        self.models = {client:copy.deepcopy(model).cuda() \
                       for client, data in data_loader_train.items()}

        self.model_cache = copy.deepcopy(model).cuda()   #全局模型

        self.best_model = None

        self.save_epoch = 0

        self.client_weights = {c: torch.tensor(1/len(data_loader_train)).cuda() \
                               for c, data in data_loader_train.items()}

        # train
        self.lr = config.lr           #学习率
        self.epochs = config.epochs   #训练轮数
        self.local_epoch = config.local_epoch   #局部训练轮数
        self.num_data = config.num_data         #样本的数量
        self.iters_per_epoch = self.num_data // config.batch_size   #每一轮迭代的次数

        self.criterion = get_loss(config.loss).cuda()  #损失函数

        self.train_loaders = data_loader_train    #训练数据的加载器
        self.vali_loaders = data_loader_vali      #验证数据的加载器
        #optimizers:字典类型 --> 键:(client)各个中心的名字   值:优化器,用于参数更新
        self.optimizers = {client:torch.optim.Adam([{'params': model.parameters()}],lr = self.lr,weight_decay=1e-4) \
                           for client, model in self.models.items()}

        # evaluate 验证
        self.epochs_per_vali = 2  #每训练epochs_per_vali进行一次验证
        self.evaluator = Evaluator(data_loader_vali)
        # save result
        self.save_dir = save_dir   #模型保存的路径

    #训练
    def train(self):

        max_dice = 0   #最佳的dice系数初始化为0
        #lr_scheduler各个中心学习率变化规则 --> 键:(c)各个中心的名字  值:(学习率变化规则)
        lr_scheduler = {c:Poly(optimizer=op, num_epochs=self.epochs,iters_per_epoch=self.iters_per_epoch) \
                        for c, op in self.optimizers.items()}
        #开始训练(局部训练+聚合)-->这样共进行self.epochs次
        for epoch in range(self.epochs):
            #局部训练(各个中心都各自进行一次训练)-->这样共进行local_epoch次 -->这样完成之后进行聚合
            for l_epoch in range(self.local_epoch):
                #遍历各个中心
                for client in self.train_loaders.keys():

                    self.train_local(client,epoch)   #各个中心都进行1次训练 + 反向传播参数更新

                    lr_scheduler[client].step(epoch=epoch)   #进行学习率更新

            self.communication() #参数聚合

            # 评估 --> 每训练self.epochs_per_vali次,就用验证集进行一次模型评估,以选出最佳模型
            if epoch % self.epochs_per_vali==self.epochs_per_vali-1:

                meanDice = 0.    #每次验证,平均dice系数初始化为0
                for client in self.models.keys():   #遍历每个中心,计算平均dice系数
                    dice, std = self.evaluator.eval(self.model_cache,client)
                    with open('/data/micca2018/test/loss.txt', 'a') as f:
                        print(epoch, client, dice, std,file=f)
                    meanDice += dice

                if meanDice >= max_dice:  #平均dice系数最优的模型为最优的模型
                    max_dice = meanDice
                    self.save_epoch = epoch
                    self.best_model = copy.deepcopy(self.model_cache)  #最好的模型

        return self.best_model


    def communication(self):    #聚合参数

        with torch.no_grad():   #在该模块下,所有计算得出的tensor的requires_grad都自动设置为False,即取消自动求导
                # FedAvg
            for key in self.model_cache.state_dict().keys():  #遍历每一层
            #state_dict:字典对象,将每一层的参数映射成tensor张量 --> 键:每一层  值:张量形式的参数
                if 'num_batches_tracked' not in key:
                    temp = torch.zeros_like(self.model_cache.state_dict()[key])
                    # aggregate parameters 聚合参数
                    for client, model in self.models.items():   #加权参数总和得到新参数
                        temp += self.client_weights[client] * model.state_dict()[key]
                    self.model_cache.state_dict()[key].data.copy_(temp)  #全局模型参数更新
                    for client, model in self.models.items():
                        self.models[client].state_dict()[key].data.copy_(temp)  #每个局部模型参数更新


    def train_local(self,client,epoch):  #局部模型在局部进行一次训练 + 误差逆传播更新局部参数

        self.models[client].train()  #模型在局部进行一次训练
        # 参数更新
        for train_batch in self.train_loaders[client]:   #分批处理

            #self.visualization(client, epoch, step)
            self.optimizers[client].zero_grad()      #之前计算的梯度清零
            #train_batch = next(self.train_loaders[client])
            imgs = torch.from_numpy(train_batch['data']).cuda(non_blocking=True)    #图像
            labs = torch.from_numpy(train_batch['seg']).type(torch.LongTensor).cuda(non_blocking=True)  #标签
            output = self.models[client](imgs)  #预测结构
            if len(labs.shape) == len(output.shape):
                labs = labs[:, 0]
            loss = self.criterion(output, labs)   #计算损失
            loss.backward()    #误差逆传播
            self.optimizers[client].step()  #更新参数


class Trainer_Naive(Trainer_FedAvg):

    def __init__(self,model,data_loader_train,data_loader_vali,config,save_dir):
        super().__init__(model,data_loader_train,data_loader_vali,config,save_dir)

        self.best_weights = {}    # 创建一个字典,记录训练过程中各个中心模型最好的参数

    def train(self):  #训练

        max_dice = defaultdict(lambda: 0.)   # 创建一个字典,记录训练过程中各个中心模型的最佳dice系数,其初始值都是0
        #为每个中心制定学习率变化规则
        lr_scheduler = {c:Poly(optimizer=op, num_epochs=self.epochs,iters_per_epoch=self.iters_per_epoch) \
                        for c, op in self.optimizers.items()}
        # 开始训练(局部训练+聚合) --> 这样共进行self.epochs次
        for epoch in range(self.epochs):
            # 局部训练(各个中心都各自进行一次训练) --> 这样共进行local_epoch次 --> 这样完成之后进行聚合
            for l_epoch in range(self.local_epoch):
                # 遍历各个中心
                for client in self.train_loaders.keys():

                    self.train_local(client,epoch)   #各个中心都进行1次训练 + 反向传播参数更新

                    lr_scheduler[client].step(epoch=epoch)   #学习率更新

            self.communication() #参数聚合


            # 评估 --> 每训练self.epochs_per_vali次,就用验证集进行一次模型评估,以选出最佳模型
            if epoch % self.epochs_per_vali==self.epochs_per_vali-1:

                for client, model in self.models.items():  #遍历各个中心,为每个中心挑出训练开始以来最好的模型(参数)
                    meanDice,std = self.evaluator.eval(model,client)  #对各个中心的模型进行评估并输出平均Dice系数和标准差
                    with open('/data/micca2018/test/loss.txt', 'a') as f:
                        print(epoch, client, meanDice, std,file=f)
                    if meanDice >= max_dice[client]:
                        max_dice[client] = meanDice
                        self.best_weights[client] = copy.deepcopy(model.state_dict())  #储存各个中心最好的参数


        for client, model in self.models.items():
            model.load_state_dict(self.best_weights[client])
        with open('/data/micca2018/test/loss.txt', 'a') as f:
            print(max_dice,file=f)
        return self.model_cache


class Trainer_Single(Trainer_FedAvg):

    def __init__(self,model,data_loader_train,data_loader_vali,config,save_dir,center):
        super().__init__(model,data_loader_train,data_loader_vali,config,save_dir)

        self.center = center

    def train(self):

        max_dice = 0

        lr_scheduler = {c:Poly(optimizer=op, num_epochs=self.epochs,iters_per_epoch=self.iters_per_epoch) \
                        for c, op in self.optimizers.items()}

        for epoch in range(self.epochs):

            self.train_local(self.center,epoch)

            lr_scheduler[self.center].step(epoch=epoch)

            # evaluate
            if epoch % self.epochs_per_vali==self.epochs_per_vali-1:

                dice,std = self.evaluator.eval(self.models[self.center],self.center)
                with open('/data/micca2018/test/loss.txt', 'a') as f:
                    print(epoch, self.center, dice,std,file=f)

                if dice >= max_dice:
                    max_dice = dice
                    self.save_epoch = epoch
                    self.best_model = copy.deepcopy(self.models[self.center])

        return self.best_model


class Trainer_FedDAN(Trainer_Naive):

    def __init__(self,model,data_loader_train,data_loader_vali,config,save_dir):
        super().__init__(model,data_loader_train,data_loader_vali,config,save_dir)

        self.alpha = config.weight    #正则化项(L_TR)的系数
        self.beta = 0.01              #KL散度(L_KL)的系数
        self.evaluator = Evaluator_feddan(data_loader_vali)   #评估模型的函数

    # 输入中心client以及训练轮数epoch(用于判断是否为热身阶段),进行局部训练
    def train_local(self,client,epoch):

        self.models[client].train()   #中心client的模型进行一次训练

        #误差逆传播算法更新参数
        with open('/data/micca2018/test/loss.txt', 'a') as f:
            print("client:",client,"  epoch:",epoch,file=f)
        for train_batch in self.train_loaders[client]:  #对该中心的数据分批处理

            #self.visualization(client, epoch, step)
            self.optimizers[client].zero_grad()   #之前计算的梯度清零
            #train_batch = next(self.train_loaders[client])
            imgs = torch.from_numpy(train_batch['data']).cuda(non_blocking=True)  #图像
            labs = torch.from_numpy(train_batch['seg']).type(torch.LongTensor).cuda(non_blocking=True)  #金标准
            # 热身阶段
            if epoch < 10:
                pred,cms = self.models[client](imgs,labs)    #pred是全局模型输出的结构   cms是cm_layer输出的混淆矩阵
                # 输入pred cms 金标准labs[:,0] alpha=1 以及 beta
                # 得到热身阶段损失L_warm = beta * L_KL + L_CE + L_TR
                loss = self.models[client].elbo1(pred, cms, labs[:,0],1,self.beta)
            # 正式训练阶段
            else:
                pred,cms = self.models[client](imgs,labs)
                # 得到总损失 L = beta * L_KL + L_CE + alpha * L_TR + L_NR
                loss = self.models[client].elbo(pred, cms, labs[:,0],self.alpha,self.beta)

            loss.backward()   #误差逆传播
            self.optimizers[client].step()   #用优化器更新参数
        with open('/data/micca2018/test/loss.txt', 'a') as f:
            print("loss",loss,file=f)

    # 全局聚合
    def communication(self):

        with torch.no_grad():  # 取消自动求导
                # FedAvg
            for key in self.model_cache.state_dict().keys():  #遍历每一层的参数
                #if 'num_batches_tracked' not in key:
                if 'num_batches_tracked' not in key and 'decoders_noisy_layers' not in key:  #判断该层是否进行全局参数聚合
                    temp = torch.zeros_like(self.model_cache.state_dict()[key])
                    # 参数聚合
                    for client, model in self.models.items():  #参数求加权平均，并更新该层的参数
                        temp += self.client_weights[client] * model.state_dict()[key]
                    self.model_cache.state_dict()[key].data.copy_(temp)
                    for client, model in self.models.items():
                        self.models[client].state_dict()[key].data.copy_(temp)


############################# Compared Methods #############################################


class Trainer_FedRep(Trainer_Naive):

    def communication(self):

        with torch.no_grad():
            for key in self.model_cache.state_dict().keys():
                if 'num_batches_tracked' not in key and 'decoder.seg_head' not in key:
                    temp = torch.zeros_like(self.model_cache.state_dict()[key])
                    # aggregate parameters
                    for client, model in self.models.items():
                        temp += self.client_weights[client] * model.state_dict()[key]
                    self.model_cache.state_dict()[key].data.copy_(temp)
                    for client, model in self.models.items():
                        self.models[client].state_dict()[key].data.copy_(temp)



class Trainer_Ditto(Trainer_Naive):

    def __init__(self,model,data_loader_train,data_loader_vali,config,save_dir):
        super().__init__(model,data_loader_train,data_loader_vali,config,save_dir)

        self.weight = config.weight

        self.local_model = {client:copy.deepcopy(model).cuda() \
                            for client, data in data_loader_train.items()}

        self.local_optimizers = {client:torch.optim.Adam([{'params': model.parameters()}],lr = self.lr,weight_decay=1e-4) \
                           for client, model in self.local_model.items()}


    def train(self):

        max_dice = defaultdict(lambda: 0.)

        lr_scheduler = {c:Poly(optimizer=op, num_epochs=self.epochs,iters_per_epoch=self.iters_per_epoch) \
                        for c, op in self.optimizers.items()}
        local_scheduler = {c:Poly(optimizer=op, num_epochs=self.epochs,iters_per_epoch=self.iters_per_epoch) \
                        for c, op in self.local_optimizers.items()}

        for epoch in range(self.epochs):

            for l_epoch in range(self.local_epoch):

                for client in self.train_loaders.keys():

                    self.train_local(client,epoch)

                    lr_scheduler[client].step(epoch=epoch)

            self.communication()

            for client in self.train_loaders.keys():
                self.local_distillation(client)
                local_scheduler[client].step(epoch=epoch)

            # evaluate
            if epoch % self.epochs_per_vali==self.epochs_per_vali-1:

                for client, model in self.local_model.items():
                    meanDice,std = self.evaluator.eval(model,client)
                    with open('/data/micca2018/test/loss.txt', 'a') as f:
                        print(epoch, client, meanDice, std,file=f)
                    if meanDice >= max_dice[client]:
                        max_dice[client] = meanDice
                        self.best_weights[client] = copy.deepcopy(model.state_dict())

        for client, model in self.models.items():
            model.load_state_dict(self.best_weights[client])

        return self.best_model

    def local_distillation(self, client):

        self.local_model[client].train()

        for train_batch in self.train_loaders[client]:
            self.local_optimizers[client].zero_grad()
            imgs = torch.from_numpy(train_batch['data']).cuda(non_blocking=True)
            labs = torch.from_numpy(train_batch['seg']).type(torch.LongTensor).cuda(non_blocking=True)
            output = self.local_model[client](imgs)
            if len(labs.shape) == len(output.shape):
                labs = labs[:, 0]
            loss = self.criterion(output, labs)
            loss.backward()
            for p_local, p_global in zip(self.local_model[client].parameters(),self.model_cache.parameters()):
                if p_local.grad is not None:
                    diff = p_local.data-p_global.data
                    p_local.grad += self.weight * diff
            self.local_optimizers[client].step()


class Trainer_FedProx(Trainer_FedAvg):

    def __init__(self,model,data_loader_train,data_loader_vali,config,save_dir):
        super().__init__(model,data_loader_train,data_loader_vali,config,save_dir)

        self.weight = config.weight
        self.omega = defaultdict(lambda: 1)


    def train_local(self,client,epoch):

        self.models[client].train()
        for train_batch in self.train_loaders[client]:

            self.optimizers[client].zero_grad()
            imgs = torch.from_numpy(train_batch['data']).cuda(non_blocking=True)
            labs = torch.from_numpy(train_batch['seg']).type(torch.LongTensor).cuda(non_blocking=True)
            output = self.models[client](imgs)
            if len(labs.shape) == len(output.shape):
                labs = labs[:, 0]

            loss = self.criterion(output, labs)

            distill_loss = self.weight * L2_penalty(self.models[client], self.model_cache, self.omega)

            loss += distill_loss

            loss.backward()
            self.optimizers[client].step()


class Trainer_FedCurv(Trainer_FedProx):

    def train(self):

        max_dice = 0

        lr_scheduler = {c:Poly(optimizer=op, num_epochs=self.epochs,iters_per_epoch=self.iters_per_epoch) \
                        for c, op in self.optimizers.items()}

        for epoch in range(self.epochs):

            for l_epoch in range(self.local_epoch):

                for client in self.train_loaders.keys():

                    self.train_local(client,epoch)
                    self.diag_fisher(client)
                    lr_scheduler[client].step(epoch=epoch)

            self.communication()


            # evaluate
            if epoch % self.epochs_per_vali==self.epochs_per_vali-1:

                meanDice = 0.
                for client in self.models.keys():
                    dice, std = self.evaluator.eval(self.model_cache,client)
                    with open('/data/micca2018/test/loss.txt', 'a') as f:
                        print(epoch, client, dice, std,file=f)
                    meanDice += dice

                if meanDice >= max_dice:
                    max_dice = meanDice
                    self.save_epoch = epoch
                    self.best_model = copy.deepcopy(self.model_cache)

        return self.best_model

    def diag_fisher(self, client):

        precision_matrices = {n: torch.zeros_like(p, dtype=torch.float32).cuda() \
                              for n, p in self.models[client].named_parameters() if p.requires_grad}


        for train_batch in self.train_loaders[client]:
            self.model_cache.train()
            self.model_cache.zero_grad()

            imgs = torch.from_numpy(train_batch['data']).cuda(non_blocking=True)
            labs = torch.from_numpy(train_batch['seg']).type(torch.LongTensor).cuda(non_blocking=True)
            output = self.model_cache(imgs)
            if len(labs.shape) == len(output.shape):
                labs = labs[:, 0]

            loss = self.criterion(output, labs)

            loss.backward()

            for n, p in self.models[client].named_parameters():
                if p.grad is not None:
                    precision_matrices[n] += p.grad.data ** 2 / (self.num_data)

        self.omega = {n: p for n, p in precision_matrices.items()}
        self.model_cache.zero_grad()


class Trainer_FedCM(Trainer_FedAvg):   #第二个消融实验的模型训练
# Input image x instead of feature z in adaptation network
    def __init__(self,model,data_loader_train,data_loader_vali,config,save_dir):
        super().__init__(model,data_loader_train,data_loader_vali,config,save_dir)

        self.criterion = get_loss('cm_loss',low_rank=False)
        self.seg_loss = get_loss('CE')
        self.lq_loss = get_loss('lq')
        self.alpha = config.weight

    def train_local(self,client,epoch):

        self.models[client].train()

        for train_batch in self.train_loaders[client]:

            #self.visualization(client, epoch, step)
            self.optimizers[client].zero_grad()
            #train_batch = next(self.train_loaders[client])
            imgs = torch.from_numpy(train_batch['data']).cuda(non_blocking=True)
            labs = torch.from_numpy(train_batch['seg']).type(torch.LongTensor).cuda(non_blocking=True)
            pred,cms = self.models[client](imgs)
            if len(labs.shape) == 4:
                labs = labs[:,0]

            if epoch < 100:
                _,_,TR = self.criterion(pred, cms, [labs], 1)
                loss = self.seg_loss(pred,labs) - TR
            else:
                loss,_,_ = self.criterion(pred, cms, [labs], self.alpha)
                loss = loss + self.lq_loss(pred,labs)

            loss.backward()
            self.optimizers[client].step()

        #print(loss.item(),CE.item(),TR.item())
        with open('/data/micca2018/test/loss.txt', 'a') as f:
            print(loss.item(),file=f)

    def communication(self):

        with torch.no_grad():
                # FedAvg
            for key in self.model_cache.state_dict().keys():
                #if 'num_batches_tracked' not in key:
                if 'num_batches_tracked' not in key and 'decoders_noisy_layers' not in key:
                    temp = torch.zeros_like(self.model_cache.state_dict()[key])
                    # aggregate parameters
                    for client, model in self.models.items():
                        temp += self.client_weights[client] * model.state_dict()[key]
                    self.model_cache.state_dict()[key].data.copy_(temp)
                    for client, model in self.models.items():
                        self.models[client].state_dict()[key].data.copy_(temp)
