## Requirements
Project use pytorch,torchvision,opencv,tqdm,ttach and so on.
```bash
pip install -r requirements.txt
```
```bash
pip install --user -r requirements.txt
```

## Related Papers 
- (**Deeplab V3+**) Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation [[Paper]](https://arxiv.org/abs/1802.02611)
- (**U-Net**) Convolutional Networks for Biomedical Image Segmentation : [[Paper]](https://arxiv.org/abs/1505.04597)
- (**scSE**) Concurrent Spatial and Channel ‘Squeeze &Excitation’ in Fully Convolutional Networks [[Paper]](https://arxiv.org/pdf/1803.02579)

## Data augmentation
- Random rotation : [0,90,180,270]
- Random h-flip ,  v-flip
- (optional) Random_brightness , random_contrast
- (optional) Blur , median_blur , motion_blur
- (optional) Gauss_noise , salt_pepper_noise , poisson_noise , speckle_noise
- (optional) Channel_shuffle , shift_hsv

## Losses
- Cross-Entorpy loss
- (optional) CrossEntropyLoss_FocalLoss

## Optimizer and lr_scheduler
- Nadam (Incorporating Nesterov Momentum into Adam [[Paper]](http://cs229.stanford.edu/proj2015/054_report.pdf) ) as the optimizer.
- Warm Up
- (optional) Radam , SGD
- LR_schedule:<br>
``(1)Step : lr = baselr * 0.1 ^ {floor(epoch-1 / lr_step)}``<br>
``(2)Cosine : lr = baselr * 0.5 * (1 + cos(iter/maxiter))``<br>
``(3)Poly : lr = baselr * (1 - iter/maxiter) ^ 0.9``

## Train
- Before train, you should change the parameters in the file `config.py`.
```bash
python main.py
```

## Predict
- Before predict, you should change the parameters in the file `predict.py`.
```bash
python predict.py
```

## Submit
- The directory `submit` contains the code of remote sensing large image slice and model fusion.

## Thanks
> https://github.com/mrgloom/awesome-semantic-segmentation <br>
> https://github.com/qubvel/segmentation_models.pytorch <br>
> https://github.com/qubvel/ttach <br>