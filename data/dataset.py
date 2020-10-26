import random

import cv2
import torch
from torchvision import transforms as T
from torch.utils import data
from data.data_utils import *

import numpy as np
from PIL import Image

def rotate_bound(image,angle):
    if angle==90:
        out = Image.fromarray(image).transpose(Image.ROTATE_90)
    elif angle==180:
        out = Image.fromarray(image).transpose(Image.ROTATE_180)
    elif angle==270:
        out = Image.fromarray(image).transpose(Image.ROTATE_270)
    else:
        pass
    return np.array(out)


def data_augment(x,y):
    flag = random.choice([1,2,3,4,5,6])
    if flag ==1:
        x, y = cv2.flip(x,0), cv2.flip(y,0)
    elif flag ==2:
        x, y = cv2.flip(x,1), cv2.flip(y,1)
    elif flag ==3:
        x, y = rotate_bound(x,90) ,rotate_bound(y,90)
    elif flag ==4:
        x, y = rotate_bound(x,180) ,rotate_bound(y,180)
    elif flag ==5:
        x, y = rotate_bound(x,270) ,rotate_bound(y,270)
    else:
        pass
    return x,y


class PredictData(data.Dataset):
    def __init__(self,images_path, n_classes = 8, transform=None):
        self.imgs =get_predict_img_paths(images_path)
        self.n_classes =n_classes
        if transform == None:
            normalize = T.Normalize(mean = [0.355, 0.384, 0.359],
                                     std = [0.137, 0.136, 0.138])
            self.transform = T.Compose([
                                        T.ToTensor(),
                                        normalize])

    def __getitem__(self, item):
        img_path = self.imgs[item]
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)  # 256,256,3     uint8

        img = self.transform(img)
        index = img_path.split(".")[-2].split("/")[-1]
        return img, index

    def __len__(self):
        return len(self.imgs)


class Dataset(data.Dataset):
    def __init__(self,images_path, labels_path,mode = "train", train_val_scale = 0.9, transform = None,use_augment = True):
        assert mode =="train" or mode == "val"
        self.imgs = get_img_label_paths(images_path,labels_path)
        self.train_val_scale =train_val_scale
        #self.imgs.sort()
        random.shuffle(self.imgs)
        self.train_num = int(self.train_val_scale*len(self.imgs))
        self.train_imgs = self.imgs[:self.train_num]
        self.val_imgs = self.imgs[self.train_num:]
        print("[-- {} --][all images num: {}],   [train image num: {}],   [val image num: {}]".format(mode,len(self.imgs),len(self.train_imgs),len(self.val_imgs)))

        self.n_classes = len(matches)
        self.use_augment =use_augment
        self.mode = mode
        self.transform = transform
        if self.transform == None:
            normalize = T.Normalize(mean = [0.355, 0.384, 0.359],
                                     std = [0.137, 0.136, 0.138])
            self.transform = T.Compose([T.ToTensor(),
                            normalize])
            self.transform_mask =T.Compose([
                T.ToTensor()
            ])

    def __getitem__(self,item):
        if self.mode == "train":
            img_path, mask_path = self.train_imgs[item]
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)  #256,256,3     uint8
            mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)     #256,256    uint16

            if self.use_augment:
                img, mask = data_augment(img, mask)
                #img = gaussion_data_augment(img)
            '''if random.random()<0.5:
                img, mask = train_sync_transform(img,mask)'''

            img = self.transform(img)
            mask_img = get_segmentation_array(mask)
            return img, mask_img
        else:
            img_path, mask_path = self.val_imgs[item]
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

            img = self.transform(img)
            mask_img = get_segmentation_array(mask)
            return img,mask_img

    def __len__(self):
        if self.mode =="train":
            return len(self.train_imgs)
        elif self.mode =="val":
            return len(self.val_imgs)


if __name__=="__main__":
    print()