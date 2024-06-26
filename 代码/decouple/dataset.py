# -*- coding:utf-8 -*-
import time
import random, glob
import numpy as np
import numpy.linalg as npl
import cv2

import nibabel as nib
from nibabel.affines import apply_affine

from batchgenerators.dataloading.dataset import Dataset
from batchgenerators.dataloading.data_loader import DataLoader       #batchgenerator用于数据增强的python包
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter

from augmentations import *

def train_val_split(dataroot):
    
    imgs = np.load(dataroot+'/imgs.npy', allow_pickle=True)
    labs = np.load(dataroot+'/labs.npy', allow_pickle=True)
    

#     rand_seed = random.randint(0,100)
#     random.seed(rand_seed)
#     random.shuffle(imgs)
#     random.seed(rand_seed)
#     random.shuffle(labs)

#     test_split = len(imgs) // 3
#     train_split = len(imgs) // 2
    train_split = len(imgs) // 4 * 3

    
    return ({'imgs':imgs[:train_split],'labs':labs[:train_split]}, \
            {'imgs':imgs[train_split:],'labs':labs[train_split:]})
    
#     return ({'imgs':imgs[:train_split],'labs':labs[:train_split]}, \
#             {'imgs':imgs[train_split:-test_split],'labs':labs[train_split:-test_split]}, \
#             {'imgs':imgs[-test_split:],'labs':labs[-test_split:]})



class CMRDataLoader(DataLoader):
    def __init__(self, data, batch_size, num_threads_in_multithreaded, seed_for_shuffle=1,
                 return_incomplete=False, shuffle=True, infinite=False):
        """
                :param data: will be stored in self._data for use in generate_train_batch
                :param batch_size: will be used by get_indices to return the correct number of indices
                :param num_threads_in_multithreaded: num_threads_in_multithreaded necessary for synchronization同时 of dataloaders
                when using multithreaded augmenter
                :param seed_for_shuffle: for reproducibility再现
                :param return_incomplete: whether or not to return batches that are incomplete. Only applies is infinite=False.
                If your data has len of 34 and your batch size is 32 then there return_incomplete=False will make this loader
                return only one batch of shape 32 (omitting 2 of your training examples). If return_incomplete=True a second
                batch with batch size 2 will be returned.
                :param shuffle: if True, the order of the indices will be shuffled between epochs. Only applies if infinite=False
                :param infinite: if True, each batch contains randomly (uniformly) sampled indices. An unlimited number of
                batches is returned. If False, DataLoader will iterate over the data only once
                :param sampling_probabilities: only applies if infinite=True. If sampling_probabilities is not None, the
                probabilities will be used by np.random.choice to sample the indexes for each batch. Important:
                sampling_probabilities must have as many entries as there are samples in your dataset AND
                sampling_probabilitiesneeds to sum to 1
                """
        super(CMRDataLoader, self).__init__(data, batch_size, num_threads_in_multithreaded, seed_for_shuffle,
                                                    return_incomplete=return_incomplete, shuffle=shuffle,
                                                    infinite=infinite)
        
        self.imgs = np.concatenate(data['imgs'],axis=0)    #数组拼接
        self.labs = np.concatenate(data['labs'],axis=0)
                
        self.indices = np.arange(len(self.imgs))  #索引
        self.patch_size = self.imgs.shape[-2:]    #每张图的尺寸
        
    def __len__(self):
        return self.imgs.shape[0]

    def generate_train_batch(self):
        indices = self.get_indices()
        data = np.zeros((self.batch_size, 1, *self.patch_size), dtype=np.float32)
        seg = np.zeros((self.batch_size, 1, *self.patch_size), dtype=np.float32)
        for i, idx in enumerate(indices):    #i索引   idx值
            data[i][0] = self.imgs[idx]
            seg[i][0]  = self.labs[idx]
            
        return {'data': data, 'seg':seg, 'idx':indices}   #字典


def getTrainLoader(data,configs):
    dataloader_train = CMRDataLoader(data, configs.batch_size, configs.num_workers, infinite=False)
    transforms = get_DA((configs.size,configs.size), spatial_DA=True, intensity_DA=False)
    tr_gen = MultiThreadedAugmenter(dataloader_train, transforms, num_processes=configs.num_workers,
                                    num_cached_per_queue=3,
                                     pin_memory=False)
    return tr_gen

def getValiLoader(data, configs):

    vali_loader = []
    
    for imgs, seg in zip(data['imgs'],data['labs']):
        vali_loader.append({'data': np.expand_dims(imgs,1).astype(np.float32), 'seg':seg.astype(np.float32)})
        
    return vali_loader

def getTestLoader(data):
    
    test_loader = []
    
    for imgs, seg in zip(data['imgs'],data['labs']):
        test_loader.append({'data': np.expand_dims(imgs,1).astype(np.float32), 'seg':seg.astype(np.float32)})

    return test_loader