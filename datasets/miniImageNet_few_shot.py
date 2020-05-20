# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate

import torch
from PIL import Image
import numpy as np
import pandas as pd
import torchvision.transforms as transforms
import datasets.additional_transforms as add_transforms
from torch.utils.data import Dataset, DataLoader
from abc import abstractmethod
from torchvision.datasets import ImageFolder

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

identity = lambda x:x

from datasets.configs import DATA_PATH

class SimpleDataset(Dataset):
    def __init__(self, transform=None, target_transform=identity, data_path=DATA_PATH+'/processed'):
        dataset = 'miniImagenet_train'
        self.dataset = dataset
        print('load dataset: %s'%(dataset))
        if dataset not in ['miniImagenet_test', 'miniImagenet_val']:
            dataset_path = data_path + '/' + self.dataset + '.npy'
            self.data = np.load(dataset_path, encoding='latin1', allow_pickle=True).item()
        else:
            dataset_path = data_path + '/miniImagenet_test.npy'
            data_ = np.load(dataset_path)
            self.data = {}
            r = range(16) if dataset is 'miniImagenet_val' else range(16,36)
            for i in r:
                self.data[i] = data_[i]
        print('OK')

        self.imdb = self._make_imdb()
        self.transform = transform
        self.target_transform = target_transform

    def _make_imdb(self):
        imdb = []
        ks = list(self.data.keys())
        ks.sort()
        for c_i, c in enumerate(ks):
            imdb_ = [(c, c_i, i) for i in range(self.data[c].shape[0])]
            imdb.extend(imdb_)
        return imdb

    def __getitem__(self, item):
        c, c_i, i = self.imdb[item]
        im = Image.fromarray(self.data[c][i], mode='RGB')
        if self.transform is not None:
            im = self.transform(im)
        return im, c_i

    def __len__(self):
        return len(self.imdb)


class SetDataset:
    def __init__(self, batch_size, transform, dataset='miniImagenet_test', data_path=DATA_PATH+'/processed'):

        self.dataset = dataset
        if dataset not in ['miniImagenet_test', 'miniImagenet_val']:
            dataset_path = data_path + '/' + self.dataset + '.npy'
            self.data = np.load(dataset_path, encoding='latin1', allow_pickle=True).item()
        else:
            dataset_path = data_path + '/miniImagenet_test.npy'
            data_ = np.load(dataset_path)
            self.data = {}
            r = range(16) if dataset is 'miniImagenet_val' else range(16, 36)
            for i in r:
                self.data[i] = data_[i]
        #self.data = np.load(dataset_path, encoding='latin1', allow_pickle=True).item()
        self.num_class = len(self.data.keys())

        self.sub_dataloader = []
        sub_data_loader_params = dict(batch_size = batch_size,
                                  shuffle = True,
                                  num_workers = 0, #use main thread only or may receive multiple batches
                                  pin_memory = False)

        ks = list(self.data.keys())
        ks.sort()
        for c_i, c in enumerate(ks):
            sub_dataset = SubDataset(self.data[c], c_i, transform=transform)
            self.sub_dataloader.append(torch.utils.data.DataLoader(sub_dataset, **sub_data_loader_params))

    def __getitem__(self, i):
        return next(iter(self.sub_dataloader[i]))

    def __len__(self):
        return len(self.sub_dataloader)


class SubDataset:
    def __init__(self, sub_meta, cl, transform=None, target_transform=identity):
        self.sub_meta = sub_meta
        self.cl = cl
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, i):
        img = Image.fromarray(self.sub_meta[i], mode='RGB')
        if self.transform:
            img = self.transform(img)
        target = self.target_transform(self.cl)
        return img, target

    def __len__(self):
        return len(self.sub_meta)


class EpisodicBatchSampler(object):
    def __init__(self, n_classes, n_way, n_episodes):
        self.n_classes = n_classes
        self.n_way = n_way
        self.n_episodes = n_episodes

    def __len__(self):
        return self.n_episodes

    def __iter__(self):
        for i in range(self.n_episodes):
            yield torch.randperm(self.n_classes)[:self.n_way]


class TransformLoader:
    def __init__(self, image_size,
                 imagejitter=0.4,
                 normalize_param    = dict(mean= [0.485, 0.456, 0.406] , std=[0.229, 0.224, 0.225]),
                 grayscale = 0.5):
        jitter_param = dict(Brightness=imagejitter, Contrast=imagejitter, Color=imagejitter)
        self.image_size      = image_size
        self.normalize_param = normalize_param
        self.jitter_param    = jitter_param
        self.grayscale       = grayscale
    
    def parse_transform(self, transform_type):
        if transform_type=='ImageJitter':
            method = add_transforms.ImageJitter( self.jitter_param )
            return method
        method = getattr(transforms, transform_type)
        if transform_type=='RandomSizedCrop':
            return method(self.image_size) 
        elif transform_type=='CenterCrop':
            return method(self.image_size) 
        elif transform_type=='Scale':
            return method([int(self.image_size*1.15), int(self.image_size*1.15)])
        elif transform_type=='Normalize':
            return method(**self.normalize_param )
        elif transform_type == 'RandomGrayscale':
            return method(p=self.grayscale)
        else:
            return method()

    def get_composed_transform(self, aug = False, grayscale=False):
        if aug and grayscale:
            transform_list = ['RandomSizedCrop', 'ImageJitter', 'RandomGrayscale',
                              'RandomHorizontalFlip', 'ToTensor', 'Normalize']
        elif aug:
            transform_list = ['RandomSizedCrop', 'ImageJitter',
                              'RandomHorizontalFlip', 'ToTensor', 'Normalize']
        else:
            transform_list = ['CenterCrop', 'ToTensor', 'Normalize']

        transform_funcs = [ self.parse_transform(x) for x in transform_list]
        transform = transforms.Compose(transform_funcs)
        return transform

class DataManager(object):
    @abstractmethod
    def get_data_loader(self, data_file, aug):
        pass 

class SimpleDataManager(DataManager):
    def __init__(self, image_size, batch_size, imagejitter=0.4):
        super(SimpleDataManager, self).__init__()
        self.batch_size = batch_size
        self.trans_loader = TransformLoader(image_size, imagejitter)

    def get_data_loader(self, aug, grayscale=False): #parameters that would change on train/val set
        transform = self.trans_loader.get_composed_transform(aug, grayscale=grayscale)
        print(transform)
        dataset = SimpleDataset(transform)

        data_loader_params = dict(batch_size = self.batch_size, shuffle = True, num_workers = 12, pin_memory = True)       
        data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)

        return data_loader

class SetDataManager(DataManager):
    def __init__(self, image_size, n_way=5, n_support=5, n_query=16, n_eposide = 100):
        super(SetDataManager, self).__init__()
        self.image_size = image_size
        self.n_way = n_way
        self.batch_size = n_support + n_query
        self.n_eposide = n_eposide

        self.trans_loader = TransformLoader(image_size)

    def get_data_loader(self, aug): #parameters that would change on train/val set
        transform = self.trans_loader.get_composed_transform(aug)
        dataset = SetDataset(self.batch_size, transform)
        sampler = EpisodicBatchSampler(len(dataset), self.n_way, self.n_eposide )  
        data_loader_params = dict(batch_sampler = sampler,  num_workers = 12, pin_memory = True)       
        data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)
        return data_loader