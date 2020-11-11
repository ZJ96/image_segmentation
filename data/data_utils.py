import os
import numpy as np

# 类别对应
matches = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]
#matches = [100, 200, 300, 400, 500, 600, 700, 800]

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

