import numpy as np
import random
from PIL import Image
import cv2

def clip(img, dtype, maxval):
    return np.clip(img, 0, maxval).astype(dtype)

#随机翻转，和随机旋转90度等等
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

#随机亮度，对比度增强
def random_brightness(img, alpha=0.2):
    return alpha * img
def random_contrast(img, alpha = 0.2):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = (3.0 * (1.0 - alpha) / gray.size) * np.sum(gray)
    return alpha * img + gray

#模糊处理
def blur(img, ksize=3):
    return cv2.blur(img, (ksize, ksize))
def median_blur(img, ksize=3):
    return cv2.medianBlur(img, ksize)
def motion_blur(img):
    kernel = np.zeros((9, 9))
    xs, ys = np.random.randint(0, kernel.shape[1]), np.random.randint(0, kernel.shape[0])
    xe, ye = np.random.randint(0, kernel.shape[1]), np.random.randint(0, kernel.shape[0])
    cv2.line(kernel, (xs, ys), (xe, ye), 1, thickness=1)
    return cv2.filter2D(img, -1, kernel / np.sum(kernel))


#各种   噪声  增强
def gauss_noise(image, var=20):
    row, col, ch = image.shape
    mean = var
    # var = 30
    sigma = var**0.5
    gauss = np.random.normal(mean,sigma,(row,col,ch))
    gauss = gauss.reshape(row,col,ch)
    gauss = (gauss - np.min(gauss)).astype(np.uint8)
    return image.astype(np.int32) + gauss

def salt_pepper_noise(image):
    #todo
    s_vs_p = 0.5
    amount = 0.004
    noisy = image
    # Salt mode
    num_salt = np.ceil(amount * image.size * s_vs_p)
    coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in image.shape]
    noisy[coords] = 255

    # Pepper mode
    num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
    coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in image.shape]
    noisy[coords] = 0
    return noisy

def poisson_noise(image):
    #todo
    vals = len(np.unique(image))
    vals = 2 ** np.ceil(np.log2(vals))
    noisy = np.random.poisson(image * vals) / float(vals)
    return noisy

def speckle_noise(image):
    #斑点噪声
    row, col, ch = image.shape
    gauss = np.random.randn(row,col,ch)
    gauss = gauss.reshape(row,col,ch)
    noisy = image + image * gauss
    return noisy

#通道 shuffle
def channel_shuffle(img):
    ch_arr = [0, 1, 2]
    np.random.shuffle(ch_arr)
    img = img[..., ch_arr]
    return img

def shift_hsv(img, hue_shift =20, sat_shift =20, val_shift =20):
    dtype = img.dtype
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.int32)
    h, s, v = cv2.split(img)
    h = cv2.add(h, hue_shift)
    h = np.where(h < 0, 255 - h, h)
    h = np.where(h > 255, h - 255, h)
    h = h.astype(dtype)
    s = clip(cv2.add(s, sat_shift), dtype, 255 if dtype == np.uint8 else 1.)
    v = clip(cv2.add(v, val_shift), dtype, 255 if dtype == np.uint8 else 1.)
    img = cv2.merge((h, s, v)).astype(dtype)
    img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
    return img

def do_brightness_shift(image, alpha=0.125):
    image = image + alpha
    image = np.clip(image, 0, 1)
    return image


def do_brightness_multiply(image, alpha=1):
    image = alpha*image
    image = np.clip(image, 0, 1)
    return image

def do_gamma(image, gamma=1.0):
    image = image ** (1.0 / gamma)
    image = np.clip(image, 0, 1)
    return image


def all_augment(img,mask):
    img,mask = flip_and_rotate(img,mask)

    # if random.random() < 0.2:
    #     gauss_noise(img,var=10)

    # if random.random() <0.3:
    #     median_blur(img)
    #
    # if random.random() < 0.3:
    #     shift_hsv(img)

    if random.random() < 0.3:
        channel_shuffle(img)

    if np.random.rand() < 0.5:
        c = np.random.choice(3)
        if c==0:
            img = do_brightness_shift(img,np.random.uniform(-0.1,+0.1))
        if c==1:
            img = do_brightness_multiply(img,np.random.uniform(1-0.08,1+0.08))
        if c==2:
            img = do_gamma(img,np.random.uniform(1-0.08,1+0.08))
        img = np.array(img,dtype=np.float32)

    return img,mask
