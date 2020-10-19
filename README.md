## Requirements
PyTorch and Torchvision needs to be installed before running the scripts, together with `PIL` ,`scikit-image`and `opencv` for data-preprocessing and `tqdm` for showing the training progress. 
what's more,tensorboard are used to analyze the loss decline.
Finally,we use `ttach` for post processing.
Python version 3.6
```bash
pip install -r requirements.txt
```
or for a local installation
```bash
pip install --user -r requirements.txt
```

## Model Weights
Download the weights from Baidu Yun
链接:https://pan.baidu.com/s/159BZTja3AsS2X1JOOxLcrQ  密码:h60a
(Because of the limited time, the weight was only trained to 90epoch. And got 66th in B List)

## Models and papers 
- (**Deeplab V3+**) Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation [[Paper]](https://arxiv.org/abs/1802.02611)
- (**U-Net**) Convolutional Networks for Biomedical Image Segmentation (2015): [[Paper]](https://arxiv.org/abs/1505.04597)
- (**scSE**) Concurrent Spatial and Channel ‘Squeeze &Excitation’ in Fully Convolutional Networks [[Paper]](https://arxiv.org/pdf/1803.02579)

##  Data augmentation
- Random rotation : [0,90,180,270]
- Random h-flip or v-flip
- (optional)  Add Gaussian noise
- (optional)  Random scale and random crop

## Losses
- Cross-Entorpy loss

## Optimizer and lr_scheduler
- We use the Nadam(Incorporating Nesterov Momentum into Adam [[Paper]](http://cs229.stanford.edu/proj2015/054_report.pdf)) as the optimizer.
- We use Warm Up and Cos for the lr_schedule

## Train
- Before train, you should change the paramters in the file `opt.py`.
- We train our model in 3 GPUs.
```bash
python train.py
```

## Predict
- Before predict, you should change the paramters in the file `predict.py`.
```bash
python predict.py
```