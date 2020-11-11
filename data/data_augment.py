import numpy as np
import random
from skimage.filters import gaussian
from PIL import Image, ImageOps
import cv2

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

def flip_and_rotate(x,y):
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


def random_brightness(img):
    if random.random() < 0.5:
        return img
    shift_value = 10
    img = img.astype(np.float32)
    shift = random.randint(-shift_value, shift_value)
    img[:, :, :] += shift
    img = np.around(img)
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def gaussion_data_augment(img):
    if random.random() < 0.5:
        sigma = random.random()*0.2
        blurred_img = gaussian(np.array(img), sigma=sigma, multichannel=True)
        blurred_img *= 255
        return blurred_img.astype(np.uint8)
    else:
        return img

def mulit_scale_augment(img, mask):
    if random.random() < 0.5:
        return img,mask
    base_size = 256
    crop_size = 256
    img,mask = Image.fromarray(img), Image.fromarray(mask)

    # random scale (short edge)
    short_size = random.randint(int(base_size * 1.0), int(base_size * 1.2))
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

def all_augment(img,mask):
    img,mask = flip_and_rotate(img,mask)
    img,mask = mulit_scale_augment(img,mask)

    return img,mask
    # img = gaussion_data_augment(img)