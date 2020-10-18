import os
import numpy as np
import random
from skimage.filters import gaussian
from PIL import Image
from PIL import Image, ImageOps, ImageFilter


# 类别对应
matches = [100, 200, 300, 400, 500, 600, 700, 800]

def get_predict_img_paths(images_path):
    res = []
    for dir_entry in os.listdir(images_path):
        res.append(os.path.join(images_path, dir_entry))
    return res

def get_img_label_paths(images_path, labels_path):
    res = []
    for dir_entry in os.listdir(images_path):
        if os.path.isfile(os.path.join(images_path, dir_entry)):
            file_name, _ = os.path.splitext(dir_entry)
            res.append((os.path.join(images_path, file_name+".tif"),
                        os.path.join(labels_path, file_name+".png")))
    return res


def get_segmentation_array(img):
    img = np.array(img,dtype=np.long)
    for m in matches:
        img[img == m] = matches.index(m)
    return img


def gaussion_data_augment(img):
    if random.random() < 0.5:
        sigma = random.random()*0.2
        blurred_img = gaussian(np.array(img), sigma=sigma, multichannel=True)
        blurred_img *= 255
        return blurred_img.astype(np.uint8)
    else:
        return img

def train_sync_transform(img, mask):
    base_size = 256
    crop_size = 256
    img,mask = Image.fromarray(img), Image.fromarray(mask)

    # random scale (short edge)
    short_size = random.randint(int(base_size * 1.0), int(base_size * 1.5))
    w, h = img.size
    if h > w:
        ow = short_size
        oh = int(1.0 * h * ow / w)
    else:
        oh = short_size
        ow = int(1.0 * w * oh / h)
    img = img.resize((ow, oh), Image.BILINEAR)
    mask = mask.resize((ow, oh), Image.NEAREST)
    # pad crop
    if short_size < crop_size:
        padh = crop_size - oh if oh < crop_size else 0
        padw = crop_size - ow if ow < crop_size else 0
        img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
        mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=800)
    # random crop crop_size
    w, h = img.size
    x1 = random.randint(0, w - crop_size)
    y1 = random.randint(0, h - crop_size)
    img = img.crop((x1, y1, x1 + crop_size, y1 + crop_size))
    mask = mask.crop((x1, y1, x1 + crop_size, y1 + crop_size))

    img, mask = np.array(img) , np.array(mask)
    return img, mask