from torch.utils import data
from torch.utils.data import DataLoader
from torchvision import transforms as T
import numpy as np
import cv2

def get_batch_img(img,overlap_pix = 60):
    w = img.shape[0]
    h = img.shape[1]
    if w ==256 and h ==256:
        return np.array([img])
    a, b = 256, 256
    res = []
    gap = 256 - overlap_pix

    while a < w:
        while b < h:
            res.append(img[a - 256:a, b - 256:b, :])
            b += gap
        res.append(img[a - 256:a, h - 256:h, :])
        a += gap
        b = 256
    while b < h:
        res.append(img[w - 256:w, b - 256:b, :])
        b += gap
    res.append(img[w - 256:w, h - 256:h, :])
    return np.array(res)

class PredictData(data.Dataset):
    def __init__(self,origin_img):
        #origin_img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
        self.imgs = get_batch_img(origin_img)
        normalize = T.Normalize(mean = [0.355, 0.384, 0.359],
                                     std = [0.137, 0.136, 0.138])
        self.transform = T.Compose([T.ToTensor(),
                                        normalize])

    def __getitem__(self, item):
        img = self.imgs[item]
        img = self.transform(img)
        return img

    def __len__(self):
        return len(self.imgs)

def get_DataLoader(img):
    data = PredictData(img)
    data_loader = DataLoader(data,batch_size=40,shuffle=False, num_workers=0)
    return data_loader
