import os
from model_define import init_model
#from model_predict import
# predict
from model_predict_3model import predict
import segmentation_models_pytorch as smp
import numpy as np
from time import time

# input_path = "/Users/zj/Downloads/mini-train/images"
# output_path = "/Users/zj/Downloads/mini-train/results"
#
# model  = init_model()
#
# for i in os.listdir(input_path):
#     print(i)
#     img_path = os.path.join(input_path,i)
#     predict(model,img_path,output_path)

t = time()
model  = init_model()
predict(model,"./5.tif","./")
print(time()-t)