import torch
import cv2
import numpy as np
import os
from .dataloader import get_DataLoader

def merge_label(label,w,h,overlap_pix = 60):
    if w==256 and h==256:
        return label[0]

    pix = int(overlap_pix / 2)
    gap = 256 - overlap_pix

    #the first col
    a, b = 256, 256
    res = label[0][0:256-pix,0:256-pix]
    i = 1
    while b + gap < h:
        res = np.hstack([res, label[i][0:256-pix,pix:256-pix]])
        i += 1
        b += gap
    if b!=h:
        res = np.hstack([res, label[i][0:256-pix,256 - (h-b+pix):256]])
        i += 1

    #center col
    while a + gap <w:
        tmp = label[i][pix:256-pix,0:256-pix]
        i += 1
        b = 256
        while b + gap < h:
            tmp = np.hstack([tmp, label[i][pix:256-pix,pix:256-pix]])
            i += 1
            b += gap
        if b!=h:
            tmp = np.hstack([tmp, label[i][pix:256-pix, 256 - (h-b+pix):256 ]])
            i += 1
        res = np.vstack([res, tmp])
        a += gap

    #the last col
    if a!=w:
        tmp = label[i][256 - (w-a+pix):256, 0:256-pix]
        i += 1
        b = 256
        while b + gap < h:
            tmp = np.hstack([tmp, label[i][256 - (w-a+pix):256, pix:256-pix]])
            i += 1
            b += gap
        if b!=h:
            tmp = np.hstack([tmp, label[i][256 - (w-a+pix):256, 256 - (h-b+pix):256]])
            i += 1
        res = np.vstack([res, tmp])

    return res


def predict(model, input_path, output_dir):
    name, ext = os.path.splitext(input_path)
    name = os.path.split(name)[-1] + ".png"
    img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    w = img.shape[0]
    h = img.shape[1]

    dataloader = get_DataLoader(img)

    model.eval()
    with torch.no_grad():
        for batch_id, (data) in enumerate(dataloader):
            data = data.cuda()
            every_label = model(data)

            every_label = torch.argmax(every_label,dim=1)
            if batch_id == 0:
                label = every_label
            else:
                label = torch.cat((label, every_label), dim=0)
    label = label.cpu().numpy().astype(np.uint16) + 1
    label = merge_label(label, w, h)
    cv2.imwrite(os.path.join(output_dir, name), label)

