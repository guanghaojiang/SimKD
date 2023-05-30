import os
from PIL import Image
import torch
import random
import numpy as np
import cv2
import pandas as pd
import albumentations as A
from albumentations import (RandomBrightnessContrast,HueSaturationValue,Normalize,HorizontalFlip,VerticalFlip,Blur,
                            MotionBlur,OneOf,MedianBlur,IAAAdditiveGaussianNoise,GaussNoise,OpticalDistortion,RGBShift,RandomCrop,
                            Cutout,Resize,RandomResizedCrop,GaussianBlur,RandomSizedCrop)
from albumentations.pytorch import ToTensorV2

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from .BaseDataset import BaseDataset

__all__ = ['Wheatchannel3Dataset']

def getFiles(dir, suffix='.png'): # 查找根目录，文件后缀 
    res = []
    labels = []
    dic = {}
    for root, directory, files in os.walk(dir):  # =>当前根,根下目录,目录下的文件
        for filename in files:
            name, suf = os.path.splitext(filename) # =>文件名,文件后缀
            if suf == suffix:
                if len(root.split('/')[-1])>2:
                    label = int(root.split('/')[-1][1])
                else:
                    label = int(root.split('/')[-1])
                res.append(os.path.join(root, filename)) # =>吧一串字符串组合成路径
                labels.append(label)
                
                if label in dic.keys():
                    dic[label]+=1
                else:
                    dic[label]=1

    print('---'*10,dic)
    return res, labels

def get_imglists(root, mode='train', spilt='train'):
    '''
    get all images path
    @param: 
        root : root path to dataset
        mode : method to read images
        spilt: sub path to specific dataset folder
    '''
    if mode == 'test':
        files = []
        files = list( map(lambda x: os.path.join(root, x), os.listdir(root)))
        files = pd.DataFrame({"filename": files})
        return files
    
    elif mode == 'inference':
        imgs, labels = [], []
        current_folder = os.path.join(root, 'test')
        imgs, labels = getFiles(current_folder)

        files = pd.DataFrame({'filename': imgs, 'label': labels})
        return files
    
    else:
        imgs, labels = [], []
        current_folder = os.path.join(root, spilt)  # train,val, or test
        imgs, labels = getFiles(current_folder)
        files = pd.DataFrame({'filename': imgs, 'label': labels})
        return files
def ua_ub_h_3c(img):
    W_SINGLE_IMG = int(img.shape[1]/6)
    ua_img = img[:,:W_SINGLE_IMG, :]
    ub_img = img[:,W_SINGLE_IMG:2*W_SINGLE_IMG, :]

    if random.random()<0.5:
        ub_img = img[:,:W_SINGLE_IMG, :]
        ua_img = img[:,W_SINGLE_IMG:2*W_SINGLE_IMG, :]

    combine_uimg = np.concatenate((ua_img,ub_img),axis=1)
    return combine_uimg 
  
def ua_ub_da_da_2x2_3c(img):

    W_SINGLE_IMG = int(img.shape[1]/6)
    ua_img = img[:,:W_SINGLE_IMG, :]
    ub_img = img[:,W_SINGLE_IMG:2*W_SINGLE_IMG, :]
    da_img = img[:,2*W_SINGLE_IMG:3*W_SINGLE_IMG, :]
    db_img = img[:,3*W_SINGLE_IMG:4*W_SINGLE_IMG, :]

    if random.random()<0.5:
        ub_img = img[:,:W_SINGLE_IMG, :]
        ua_img = img[:,W_SINGLE_IMG:2*W_SINGLE_IMG, :]

    combine_uimg = np.concatenate((ua_img,ub_img),axis=1)
    combine_dimg = np.concatenate((da_img,db_img),axis=1)
    combine_udimg = np.concatenate((combine_uimg,combine_dimg),axis=0)

    return combine_udimg

class Wheatchannel3Dataset(BaseDataset):
  def __init__(self, imglist, mode='train',img_preprocess='ua_ub_da_db_2x2_3c', transforms=None):
    super(Wheatchannel3Dataset,self).__init__(mode, transforms)
    self.mode = mode
    self.transforms = transforms
    self.imglist = imglist
    self.img_preprocess =img_preprocess
    imgs = []
    if self.mode == 'test':
      for index,row in imglist.iterrows():
        imgs.append((row['filename'],row['label']))
      self.imgs = imgs
    
    else:
      for index, row in imglist.iterrows():
        imgs.append((row['filename'],row['label']))
      self.imgs = imgs



  def __len__(self):
    return len(self.imgs)


  def __getitem__(self,index):

    filename, label = self.imgs[index]
    img = cv2.imread(filename)
    if img is None:
      print(filename)
    if self.img_preprocess=='ua_ub_h_3c':
      img = ua_ub_h_3c(img)
    elif self.img_preprocess=='ua_ub_da_db_2x2_3c':
      img = ua_ub_da_da_2x2_3c(img)
 
    img = self.transforms(image=img)['image']

    return img, label
  
def get_grain_dataloader(opt):

    train_imgs = get_imglists(opt.train_data_folder, mode='train', spilt="train")
    val_imgs = get_imglists(opt.val_data_folder, mode='train', spilt='val')

    MEANS = (0.308562, 0.251994, 0.187898) # RGB
    STDS  = (0.240441, 0.197289, 0.149387) # RGB

    MEANS = MEANS[::-1]
    STDS  = STDS[::-1]

    train_transform =A.Compose([
        
        # OneOf([
        #     Resize(248,248),
        #     Resize(192,192),
        #     Resize(256,200),
        #     Resize(200,256),
        # ],p=1),

        Resize(192,192),
        # RandomCrop(224,224),
        # Cutout(num_holes=4,max_h_size=6,max_w_size=6,p=0.3),
        HorizontalFlip(),
        VerticalFlip(),
        RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.3),
        # HueSaturationValue(hue_shift_limit=20,sat_shift_limit=25,val_shift_limit=20,p=0.1),
        RGBShift(10,10,10,p=0.3),
        OneOf([
            # 模糊相关操作
            MedianBlur(blur_limit=5, p=0.3),
            Blur(blur_limit=5, p=0.3),
            GaussianBlur(),
        ], p=0.3),
        Normalize(mean=MEANS, std=STDS),
        ToTensorV2()
    ])

    val_transform = A.Compose([
        Resize(192,192),
        Normalize(mean=MEANS, std=STDS),
        ToTensorV2()
    ])

    train_dataset = Wheatchannel3Dataset(train_imgs, mode='train', img_preprocess=opt.img_preprocess, transforms=train_transform)
    val_dataset = Wheatchannel3Dataset(val_imgs, mode='train', img_preprocess=opt.img_preprocess, transforms=val_transform)
    print('Total train:',len(train_dataset),'  total val:', len(val_dataset))
    if opt.mul_dist:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        test_sampler = DistributedSampler(val_dataset, shuffle=False)
    else:
        train_sampler = None
        test_sampler = None
    train_loader = DataLoader(train_dataset,
                            batch_size=opt.batch_size,
                            shuffle=(train_sampler is None),
                            num_workers=opt.num_workers,
                            pin_memory=True,
                            drop_last=True,
                            sampler=train_sampler)

    test_loader = DataLoader(val_dataset,
                             batch_size=opt.batch_size,
                             shuffle=False,
                             num_workers=opt.num_workers,
                             pin_memory=True,
                             drop_last=True,
                             sampler=test_sampler)

    return train_loader, test_loader, train_sampler
