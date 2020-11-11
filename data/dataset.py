import random
import cv2
from torchvision import transforms as T
from torch.utils import data
from data.data_utils import *
from data.data_augment import all_augment

import numpy as np

img_mean = [0.400,0.417,0.417]
img_std = [0.172,0.168,0.181]


class PredictData(data.Dataset):
    def __init__(self,images_path, n_classes = 17, transform=None):
        self.imgs =get_predict_img_paths(images_path)
        self.n_classes =n_classes
        if transform == None:
            normalize = T.Normalize(mean = img_mean,
                                     std = img_std)
            self.transform = T.Compose([T.ToTensor(),
                                        normalize])

    def __getitem__(self, item):
        img_path = self.imgs[item]
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)     # 256,256,3     uint8

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
            normalize = T.Normalize(mean = img_mean,
                                     std = img_std)
            self.transform = T.Compose([T.ToTensor(),
                            normalize])

    def __getitem__(self,item):
        if self.mode == "train":
            img_path, mask_path = self.train_imgs[item]
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)  #256,256,3     uint8
            mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)     #256,256    uint16

            if self.use_augment:
                img, mask = all_augment(img, mask)

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
