from models.U_Nets import *
from models import DeepLab

from opt import opt
from data.dataset import *
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch
import cv2
import numpy as np
import os
import shutil
import ttach as tta


def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    else:
        shutil.rmtree(dir_name)
        os.mkdir(dir_name)

def predict(model,image_file,  output_path, n_class, weights_path=None):
    print("*******  begin test  *******")
    print("final results are going to the dir:  {}".format(output_path))
    check_mkdir(output_path)
    test_data = PredictData(image_file,n_classes=opt.n_classes)
    test_dataLoader = DataLoader(test_data, opt.batch, shuffle=False, num_workers=opt.num_workers)

    model.eval()
    with torch.no_grad():
        with tqdm(total=len(test_dataLoader), unit='batch') as pbar:
            for batch_id, (data,index) in enumerate(test_dataLoader):
                data= data.to(opt.device)
                out = model(data)
                out  = out.data.cpu()

                for ii in range(len(out)):
                    every_out = out[ii]
                    every_out = np.transpose(every_out.numpy(), (1, 2, 0))   #chage to 256,256,8

                    #every_out = every_out.reshape((256, 256, n_class))
                    predict = every_out.argmax(axis=2)
                    seg_img = np.zeros((256, 256), dtype=np.uint16)
                    for c in range(opt.n_classes):
                        seg_img[predict[:,:] == c] = c
                    seg_img = cv2.resize(seg_img, (256, 256), interpolation=cv2.INTER_NEAREST)
                    save_img = np.zeros((256, 256), dtype=np.uint16)
                    for i in range(256):
                        for j in range(256):
                            save_img[i][j] = matches[int(seg_img[i][j])]
                    img_name = index[ii] + ".png"
                    cv2.imwrite(os.path.join(output_path, img_name), save_img)
                pbar.update(1)

if __name__ == "__main__":
    weights_path = "./checkpoints/u_net_1.pth"
    input_path = "./data/test/images/"
    output_path = "./data/test/labels/"

    model = DeepLab(output_stride=16,class_num=opt.n_classes,pretrained=True,bn_momentum=0.1,freeze_bn=False).to(opt.device)
    model.load_state_dict(torch.load(weights_path))
    tta_model = tta.SegmentationTTAWrapper(model, tta.aliases.d4_transform(), merge_mode='mean')
    predict(model, input_path,output_path, opt.n_classes, weights_path)
