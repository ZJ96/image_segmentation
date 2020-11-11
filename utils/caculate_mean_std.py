import os
import cv2
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torchvision
import time
from time import time
from tqdm import tqdm

np.std()
def compute_mean_and_std(dataset):
    # 输入PyTorch的dataset，输出均值和标准差
    mean_r = 0
    mean_g = 0
    mean_b = 0
    print("计算均值>>>")
    for img_path, _ in tqdm(dataset,ncols=80):
      img=Image.open(img_path)
      img = np.asarray(img) # change PIL Image to numpy array
      mean_b += np.mean(img[:, :, 0])
      mean_g += np.mean(img[:, :, 1])
      mean_r += np.mean(img[:, :, 2])

    mean_b /= len(dataset)
    mean_g /= len(dataset)
    mean_r /= len(dataset)

    diff_r = 0
    diff_g = 0
    diff_b = 0

    N = 0
    print("计算方差>>>")
    for img_path, _ in tqdm(dataset,ncols=80):
      img=Image.open(img_path)
      img = np.asarray(img)
      diff_b += np.sum(np.power(img[:, :, 0] - mean_b, 2))
      diff_g += np.sum(np.power(img[:, :, 1] - mean_g, 2))
      diff_r += np.sum(np.power(img[:, :, 2] - mean_r, 2))

      N += np.prod(img[:, :, 0].shape)

    std_b = np.sqrt(diff_b / N)
    std_g = np.sqrt(diff_g / N)
    std_r = np.sqrt(diff_r / N)

    mean = (mean_b.item() / 255.0, mean_g.item() / 255.0, mean_r.item() / 255.0)
    std = (std_b.item() / 255.0, std_g.item() / 255.0, std_r.item() / 255.0)
    return mean, std


#文件路径，必须隔一个文件夹
path = ""
train_data = torchvision.datasets.ImageFolder(path)

train_mean,train_std=compute_mean_and_std(train_data.imgs)
time_start =time()
time_end=time()
print("消耗时间：", round(time_end - time_start, 4), "s")

print("平均值：{}".format(train_mean))
print("方差：{}".format(train_std))


# 平均值：(0.417267969136556, 0.4177038959506166, 0.4003281150823855)
# 方差：(0.18120442467448047, 0.16814594406089894, 0.17162799434448614)
