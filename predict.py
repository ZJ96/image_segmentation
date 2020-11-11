from models import DeepLab
from data.dataset import *
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch
import cv2
import numpy as np
import os
import ttach as tta
from utils import check_new_mkdir
from config import get_args


#use this method to load weights in single gpu
from collections import OrderedDict
def my_load_state_dict(model,weights_path):
    weights = torch.load(weights_path)
    new_weights =OrderedDict()
    for k,v in weights.items():
        k=k[7:]
        new_weights[k] =v
    model.load_state_dict(new_weights)
    return model


def predict(model, args):
    print("*******  begin test  *******")
    print("final results are going to the dir:  {}".format(args.test_output_path))

    check_new_mkdir(args.test_output_path)
    test_data = PredictData(args.test_input_path,n_classes=args.n_classes)
    test_dataLoader = DataLoader(test_data, args.batch_size, shuffle=False, num_workers=8)

    model.eval()
    with torch.no_grad():
        with tqdm(total=len(test_dataLoader), unit='batch') as pbar:
            for batch_id, (data,index) in enumerate(test_dataLoader):
                data= data.cuda()
                out = model(data)
                out  = out.data.cpu()

                for ii in range(len(out)):
                    every_out = out[ii]
                    every_out = np.transpose(every_out.numpy(), (1, 2, 0))   #chage to 256,256,8

                    #every_out = every_out.reshape((256, 256, n_class))
                    predict = every_out.argmax(axis=2)
                    seg_img = np.zeros((256, 256), dtype=np.uint16)
                    for c in range(args.n_classes):
                        seg_img[predict[:,:] == c] = c
                    seg_img = cv2.resize(seg_img, (256, 256), interpolation=cv2.INTER_NEAREST)
                    save_img = np.zeros((256, 256), dtype=np.uint16)
                    for i in range(256):
                        for j in range(256):
                            save_img[i][j] = matches[int(seg_img[i][j])]
                    img_name = index[ii] + ".png"
                    cv2.imwrite(os.path.join(args.test_output_path, img_name), save_img)
                pbar.update(1)



if __name__ == "__main__":
    args = get_args()

    model = DeepLab(output_stride=16,class_num=args.n_classes,pretrained=True,bn_momentum=0.1,freeze_bn=False).cuda()
    model.load_state_dict(torch.load(args.test_weights_path))
    #model = my_load_state_dict(model,args.test_weights_path)

    model = tta.SegmentationTTAWrapper(model, tta.aliases.d4_transform(), merge_mode='mean')
    predict(model, args)
