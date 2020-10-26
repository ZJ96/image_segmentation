import os
from PIL import Image
import numpy as np
class FWIOU(object):
    def __init__(self,num_classes=8):
        super().__init__()
        self.num_classes = num_classes
        self.confusion_matrix = np.zeros([num_classes, num_classes],dtype=np.float64)
        self.num_points = 0

    def update(self,predict,gt):
        gt = np.array(gt)
        gt[gt > (self.num_classes - 1)] = 0
        pred = np.array(predict)
        # get shape
        height = np.shape(gt)[0]
        width = np.shape(pred)[1]
        # read into matrix
        for h in range(height):
            for w in range(width):
                self.confusion_matrix[gt[h][w]][pred[h][w]] += 1
        self.num_points += height * width


    def calculate_fwiou(self):

        fwIoU = 0
        for i in range(self.num_classes):  # for every class
            IoU = 0
            pii = self.confusion_matrix[i][i]
            sum_pij = 0
            for j in range(self.num_classes):
                sum_pij += self.confusion_matrix[i][j]
            sum_pji = 0
            for j in range(self.num_classes):
                sum_pji += self.confusion_matrix[j][i]
            if (sum_pij + sum_pji - pii)==0:
                #print('The IoU of', i, 'th class is nan!!!!!!!!!!!!', )
                continue
            IoU = pii / (sum_pij + sum_pji - pii)
            fwIoU += IoU * sum_pij / self.num_points
            #print('The IoU of', i, 'th class is', IoU)
        print('fwIoU of all classes:', fwIoU)
        return fwIoU

