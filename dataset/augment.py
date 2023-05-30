import albumentations as A
from albumentations import (RandomBrightnessContrast,HueSaturationValue,Normalize,HorizontalFlip,VerticalFlip,Blur,
                            MotionBlur,OneOf,MedianBlur,IAAAdditiveGaussianNoise,GaussNoise,OpticalDistortion,RGBShift,RandomCrop,
                            Cutout,Resize,RandomResizedCrop,GaussianBlur,RandomSizedCrop)
from albumentations.pytorch import ToTensorV2

import numpy as np
import torch
import imgaug as ia
import imgaug.augmenters as iaa

def get_augmentor():
    seq = iaa.Sequential([
        iaa.Crop(px=(0, 16)),
        iaa.Fliplr(0.5),
        iaa.GaussianBlur(sigma=(0, 3.0))
    ])
    def augment(images):
        return seq.augment(images.transpose(0, 2, 3, 1)).transpose(0, 2, 3, 1)
    return augment
    
    
def sharpen(x, T):
    temp = x**(1/T)
    return temp / temp.sum(axis=1, keepdims=True)

def mixup(x1, x2, y1, y2, alpha):
    beta = np.random.beta(alpha, -alpha)
    x = beta * x1 + (1 - beta) * x2
    y = beta * y1 + (1 - beta) * y2
    return x, y


def mixmatch(x, y, u, model, augment_fn, T=0.5, K=2, alpha=0.75):
    xb = augment_fn(x)
    ub = [augment_fn(u) for _ in range(K)]
    qb = sharpen(sum(map(lambda i: model(i), ub)) / K)
    Ux = np.concatenate(ub, axis=0)
    Uy = np.concatenate([qb for _ in range(K)], axis=0)
    indices = np.random.shuffle(np.arange(len(xb) + len(Ux)))
    Wx = np.concatenate([Ux, xb], axis=0)[indices]
    Wy = np.concatenate([qb, y], axis=0)[indices]
    X, p = mixup(xb, Wx[:len(xb)], y, Wy[:len(xb)], alpha)
    U, q = mixup(Ux, Wx[len(xb):], Uy, Wy[len(xb):], alpha)
    return X, p, U, q
 
 
class MixMatchLoss(torch.nn.Module):
    def __init__(self, lambda_u=100):
        self.lambda_u = lambda_u
        self.xent = torch.nn.CrossEntropyLoss()
        self.mse = torch.nn.MSELoss()
        super(MixMatchLoss, self).__init__()
    
    def forward(X, U, p, q):
        X_ = np.concatenate([X, U], axis=1)
        y_ = np.concatenate([p, q], axis=1)
        return self.xent(preds[:len(p)], p) + self.mse(preds[len(p):], q)


def get_transforms(RandBright_limit=0.15, RandBright_ratio=0.3,RandContra_limit=0.15,RandContra_ratio=0.3):

    MEANS = (0.308562, 0.251994, 0.187898) # RGB
    STDS  = (0.240441, 0.197289, 0.149387) # RGB

    MEANS = MEANS[::-1]
    STDS  = STDS[::-1]

    train_transform = A.Compose([
        
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

    return train_transform, val_transform



def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

