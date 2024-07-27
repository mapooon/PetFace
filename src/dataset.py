import numbers
import os
import queue as Queue
import threading
from typing import Iterable

import mxnet as mx
import numpy as np
import torch
from functools import partial
from torch import distributed
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from utils.utils_distributed_sampler import DistributedSampler
from utils.utils_distributed_sampler import get_dist_info, worker_init_fn
from tqdm import tqdm
import pickle
import cv2
from PIL import Image
from glob import glob
import random
import pandas as pd

def get_dataloader(
    train_csv,
    basedir,
    local_rank,
    batch_size,
    dali = False,
    seed = 2048,
    num_workers = 2
    ) -> Iterable:


    train_set = Train(train_csv=train_csv, basedir=basedir, local_rank=local_rank)

    
    rank, world_size = get_dist_info()
    train_sampler = DistributedSampler(
        train_set, num_replicas=world_size, rank=rank, shuffle=True, seed=seed)

    if seed is None:
        init_fn = None
    else:
        init_fn = partial(worker_init_fn, num_workers=num_workers, rank=rank, seed=seed)

    train_loader = DataLoaderX(
        local_rank=local_rank,
        dataset=train_set,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=init_fn,
    )

    return train_loader,train_set.num_classes,len(train_set)


def get_tripletdataloader(
    train_csv,
    basedir,
    local_rank,
    batch_size,
    dali = False,
    seed = 2048,
    num_workers = 2,
    ) -> Iterable:


    train_set = Triplet(train_csv=train_csv, basedir=basedir, local_rank=local_rank)

    
    rank, world_size = get_dist_info()
    train_sampler = DistributedSampler(
        train_set, num_replicas=world_size, rank=rank, shuffle=True, seed=seed)

    if seed is None:
        init_fn = None
    else:
        init_fn = partial(worker_init_fn, num_workers=num_workers, rank=rank, seed=seed)

    train_loader = DataLoaderX(
        local_rank=local_rank,
        dataset=train_set,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=init_fn,
    )

    return train_loader,train_set.num_classes,len(train_set)

def get_unifieddataloader(
    train_csv_list,
    basedir,
    local_rank,
    batch_size,
    dali = False,
    seed = 2048,
    num_workers = 2
    ) -> Iterable:


    train_set = UnifiedTrain(train_csv_list=train_csv_list, basedir=basedir, local_rank=local_rank)

    
    rank, world_size = get_dist_info()
    train_sampler = DistributedSampler(
        train_set, num_replicas=world_size, rank=rank, shuffle=True, seed=seed)

    if seed is None:
        init_fn = None
    else:
        init_fn = partial(worker_init_fn, num_workers=num_workers, rank=rank, seed=seed)

    train_loader = DataLoaderX(
        local_rank=local_rank,
        dataset=train_set,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=init_fn,
    )

    return train_loader,train_set.num_classes,len(train_set)

class BackgroundGenerator(threading.Thread):
    def __init__(self, generator, local_rank, max_prefetch=6):
        super(BackgroundGenerator, self).__init__()
        self.queue = Queue.Queue(max_prefetch)
        self.generator = generator
        self.local_rank = local_rank
        self.daemon = True
        self.start()

    def run(self):
        torch.cuda.set_device(self.local_rank)
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def next(self):
        next_item = self.queue.get()
        if next_item is None:
            raise StopIteration
        return next_item

    def __next__(self):
        return self.next()

    def __iter__(self):
        return self


class DataLoaderX(DataLoader):

    def __init__(self, local_rank, **kwargs):
        super(DataLoaderX, self).__init__(**kwargs)
        self.stream = torch.cuda.Stream(local_rank)
        self.local_rank = local_rank

    def __iter__(self):
        self.iter = super(DataLoaderX, self).__iter__()
        self.iter = BackgroundGenerator(self.iter, self.local_rank)
        self.preload()
        return self

    def preload(self):
        self.batch = next(self.iter, None)
        if self.batch is None:
            return None
        with torch.cuda.stream(self.stream):
            for k in range(len(self.batch)):
                self.batch[k] = self.batch[k].to(device=self.local_rank, non_blocking=True)

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        if batch is None:
            raise StopIteration
        self.preload()
        return batch

class Train(Dataset):
    def __init__(self, train_csv,basedir, local_rank):
        super(Train, self).__init__()
        
        self.transform = transforms.Compose(
            [
                # transforms.Resize(224),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
             ])
        self.local_rank=local_rank
        image_list=[]
        label_list=[]
        
        df=pd.read_csv(train_csv)
        image_list=df['filename'].tolist()
        image_list=[os.path.join(basedir,p) for p in image_list]
        label_list=df['label'].tolist()
       
        assert len(image_list)>0
        
        self.image_list=image_list
        self.label_list=label_list
        self.num_classes=len(list(set(label_list)))
        print(f'num_classes: {self.num_classes}, num_images: {len(image_list)}')
            

    def __getitem__(self, index):
        flag=True
        while flag:
            try:
                path_img = self.image_list[index]
                label = self.label_list[index]
                img = Image.open(path_img).convert('RGB')
                
                sample = self.transform(img)
                label = torch.tensor(label, dtype=torch.long)
                flag=False
            except Exception as e:
                print(label,path_img,e)
                index=torch.randint(low=0,high=len(self),size=(1,)).item()
        return sample, label

    def __len__(self):
        return len(self.image_list)



class Triplet(Dataset):
    def __init__(self, train_csv,basedir, local_rank):
        super(Triplet, self).__init__()
        
        self.transform = transforms.Compose(
            [
            #  transforms.Resize(224),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
             ])
        self.local_rank=local_rank
        image_list=[]
        label_list=[]
        # label=0
        df=pd.read_csv(train_csv)
        image_list=df['filename'].tolist()
        image_list=[os.path.join(basedir,p) for p in image_list]
        label_list=df['label'].tolist()
        label2pos={}
        for i in range(len(image_list)):
            label=label_list[i]
            image=image_list[i]
            if label not in label2pos:
                label2pos[label]=[]
            label2pos[label].append(image)
        
        assert len(image_list)>0

        self.label2pos=label2pos
        
        self.image_list=image_list
        self.label_list=label_list
        self.num_classes=len(list(set(label_list)))
        print(f'num_classes: {self.num_classes}, num_images: {len(image_list)}')
            

    def __getitem__(self, index):
        flag=True
        while flag:
            try:
                path_img = self.image_list[index]
                label = self.label_list[index]
                img = Image.open(path_img).convert('RGB')

                path_pos=random.choice(list(set(self.label2pos[label])-set([path_img])))
                path_neg=random.choice(self.label2pos[random.choice(list(set(self.label_list)-set([label])))])
                img_pos=Image.open(path_pos).convert('RGB')
                img_neg=Image.open(path_neg).convert('RGB')
                
                sample = self.transform(img)
                sample_pos = self.transform(img_pos)
                sample_neg = self.transform(img_neg)

                label = torch.tensor(label, dtype=torch.long)
                flag=False
            except Exception as e:
                print(label,path_img,e)
                index=torch.randint(low=0,high=len(self),size=(1,)).item()
        return sample, label, sample_pos, sample_neg

    def __len__(self):
        return len(self.image_list)







class UnifiedTrain(Dataset):
    def __init__(self, train_csv_list,basedir, local_rank):
        super(UnifiedTrain, self).__init__()
        
        self.transform = transforms.Compose(
            [
                # transforms.Resize(224),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
             ])
        self.local_rank=local_rank
        image_list=[]
        label_list=[]
        cnt=0
        for train_csv in train_csv_list:
            df=pd.read_csv(train_csv)
            image_list_temp=df['filename'].tolist()
            image_list+=[os.path.join(basedir,p) for p in image_list_temp]
            label_list_temp=df['label'].tolist()
            label_list+=[lab+cnt for lab in label_list_temp]
            cnt+=max(label_list_temp)+1

        
        assert len(image_list)>0
        
        self.image_list=image_list
        self.label_list=label_list
        self.num_classes=len(list(set(label_list)))
        print(f'num_classes: {self.num_classes}, num_images: {len(image_list)}')
            

    def __getitem__(self, index):
        flag=True
        while flag:
            try:
                path_img = self.image_list[index]
                label = self.label_list[index]
                img = Image.open(path_img).convert('RGB')
                
                sample = self.transform(img)
                label = torch.tensor(label, dtype=torch.long)
                flag=False
            except Exception as e:
                print(label,path_img,e)
                index=torch.randint(low=0,high=len(self),size=(1,)).item()
        return sample, label

    def __len__(self):
        return len(self.image_list)



####---- Test datasets ----

class Reidentification(Dataset):
    def __init__(self, csv_path,basedir):
        super(Reidentification, self).__init__()
        self.transform = transforms.Compose([
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
             ])
        
        
        df=pd.read_csv(csv_path)
        img_list=df['filename'].tolist()
        self.label_list=df['label'].tolist()
        self.img_list=[os.path.join(basedir,p) for p in img_list]
    

    def __getitem__(self, index):
        
        path_img = self.img_list[index]
        label = self.label_list[index]
        
        img = self.transform(Image.open(path_img).convert('RGB'))
        
        label = torch.tensor(label, dtype=torch.long)
        
        return img, label

    def __len__(self):
        return len(self.img_list)

class Verification(Dataset):
    def __init__(self, csv_path,basedir,transform=None):
        super(Verification, self).__init__()
        
        if transform is not None:
            self.transform=transform
        else:
            self.transform = transforms.Compose([
                    # transforms.RandomHorizontalFlip(),
                    # transforms.Resize(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ])
        
        
        df=pd.read_csv(csv_path)
        img1_list=df['filename1'].tolist()
        img2_list=df['filename2'].tolist()
        self.label_list=df['label'].tolist()
        self.img1_list=[os.path.join(basedir,p) for p in img1_list]
        self.img2_list=[os.path.join(basedir,p) for p in img2_list]
        

            

    def __getitem__(self, index):
        
        path_img1 = self.img1_list[index]
        path_img2 = self.img2_list[index]
        label = self.label_list[index]
        
        img1 = self.transform(Image.open(path_img1).convert('RGB'))
        img2 = self.transform(Image.open(path_img2).convert('RGB'))
        
        
        label = torch.tensor(label, dtype=torch.long)
        
        return img1,img2, label

    def __len__(self):
        return len(self.img1_list)

class UnifiedReidentification(Dataset):
    def __init__(self, csv_path_list, basedir):
        super(UnifiedReidentification, self).__init__()
        self.transform = transforms.Compose([
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
             ])
        
        cnt=0
        img_list=[]
        label_list=[]
        idx2idxs={}
        for idx_class,csv_path in enumerate(csv_path_list):
            df=pd.read_csv(csv_path)
            img_list+=df['filename'].tolist()
            label_list_temp=df['label'].tolist()
            label_list_temp=[lab+cnt for lab in label_list_temp]
            label_list+=label_list_temp
            cnt=max(label_list_temp)+1
            idx2idxs.update({i:label_list_temp for i in label_list_temp})
            
        
        self.label_list=label_list
        self.img_list=[os.path.join(basedir,p) for p in img_list]
        self.idx2idxs=idx2idxs
    

    def __getitem__(self, index):
        
        path_img = self.img_list[index]
        label = self.label_list[index]
        
        img = self.transform(Image.open(path_img).convert('RGB'))
        
        label = torch.tensor(label, dtype=torch.long)
        
        return img, label

    def __len__(self):
        return len(self.img_list)



