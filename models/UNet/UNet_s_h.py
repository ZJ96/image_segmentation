import math
import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models
import torch.optim as optim
import numpy as np
from collections import OrderedDict
from torch.utils import model_zoo
from time import time

from torch.autograd.variable import Variable
class SEModule(nn.Module):
    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x

class SCSEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SCSEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.channel_excitation = nn.Sequential(nn.Linear(channel, int(channel // reduction)),
                                                nn.ReLU(inplace=True),
                                                nn.Linear(int(channel // reduction), channel),
                                                nn.Sigmoid())

        self.spatial_se = nn.Sequential(nn.Conv2d(channel, 1, kernel_size=1,
                                                  stride=1, padding=0, bias=False),
                                        nn.Sigmoid())

    def forward(self, x):
        bahs, chs, _, _ = x.size()

        # Returns a new tensor with the same data as the self tensor but of a different size.
        chn_se = self.avg_pool(x).view(bahs, chs)
        chn_se = self.channel_excitation(chn_se).view(bahs, chs, 1, 1)
        chn_se = torch.mul(x, chn_se)

        spa_se = self.spatial_se(x)
        spa_se = torch.mul(x, spa_se)
        return torch.add(chn_se, 1, spa_se)

class ConvBn2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=1):
        super(ConvBn2d, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              bias=False, dilation=dilation)
        self.bn = nn.BatchNorm2d(out_channels)
        #self.bn = SynchronizedBatchNorm2d(out_channels)

    def forward(self, z):
        x = self.conv(z)
        x = self.bn(x)
        return x
class _SelfAttentionBlock(nn.Module):
    '''
    The basic implementation for self-attention block/non-local block
    Input:
        N X C X H X W
    Parameters:
        in_channels       : the dimension of the input feature map
        key_channels      : the dimension after the key/query transform
        value_channels    : the dimension after the value transform
        scale             : choose the scale to downsample the input feature maps (save memory cost)
    Return:
        N X C X H X W
        position-aware context features.(w/o concate or add with the input)
    '''

    def __init__(self, in_channels, key_channels, value_channels, out_channels=None, scale=1):
        super(_SelfAttentionBlock, self).__init__()
        self.scale = scale
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.key_channels = key_channels
        self.value_channels = value_channels
        if out_channels == None:
            self.out_channels = in_channels
        self.pool = nn.MaxPool2d(kernel_size=(scale, scale))
        self.f_key = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.key_channels),
            nn.ReLU()
        )
        self.f_query = self.f_key
        self.f_value = nn.Conv2d(in_channels=self.in_channels, out_channels=self.value_channels,
                                 kernel_size=1, stride=1, padding=0)
        self.W = nn.Conv2d(in_channels=self.value_channels, out_channels=self.out_channels,
                           kernel_size=1, stride=1, padding=0)
        nn.init.constant_(self.W.weight, 0)
        nn.init.constant_(self.W.bias, 0)

    def forward(self, x):
        batch_size, h, w = x.size(0), x.size(2), x.size(3)
        if self.scale > 1:
            x = self.pool(x)

        value = self.f_value(x).view(batch_size, self.value_channels, -1)
        value = value.permute(0, 2, 1)
        query = self.f_query(x).view(batch_size, self.key_channels, -1)
        query = query.permute(0, 2, 1)
        key = self.f_key(x).view(batch_size, self.key_channels, -1)

        sim_map = torch.matmul(query, key)
        sim_map = (self.key_channels ** -.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)

        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(batch_size, self.value_channels, *x.size()[2:])
        context = self.W(context)
        if self.scale > 1:
            context = F.interpolate(input=context, size=(h, w), mode='bilinear', align_corners=True)
        return context

class SelfAttentionBlock2D(_SelfAttentionBlock):
    def __init__(self, in_channels, key_channels, value_channels, out_channels=None, scale=1):
        super(SelfAttentionBlock2D, self).__init__(in_channels,
                                                   key_channels,
                                                   value_channels,
                                                   out_channels,
                                                   scale)
class BaseOC_Context_Module(nn.Module):
    """
    Output only the context features.
    Parameters:
        in_features / out_features: the channels of the input / output feature maps.
        dropout: specify the dropout ratio
        fusion: We provide two different fusion method, "concat" or "add"
        size: we find that directly learn the attention weights on even 1/8 feature maps is hard.
    Return:
        features after "concat" or "add"
    """

    def __init__(self, in_channels, out_channels, key_channels, value_channels, dropout, sizes=([1])):
        super(BaseOC_Context_Module, self).__init__()
        self.stages = []
        self.stages = nn.ModuleList(
            [self._make_stage(in_channels, out_channels, key_channels, value_channels, size) for size in sizes])
        self.conv_bn_dropout = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def _make_stage(self, in_channels, output_channels, key_channels, value_channels, size):
        return SelfAttentionBlock2D(in_channels,
                                    key_channels,
                                    value_channels,
                                    output_channels,
                                    size)

    def forward(self, feats):
        priors = [stage(feats) for stage in self.stages]
        context = priors[0]
        for i in range(1, len(priors)):
            context += priors[i]
        output = self.conv_bn_dropout(context)
        return output
class ASP_OC_Module(nn.Module):
    def __init__(self, features, out_features=512, dilations=(12, 24, 36)):
        super(ASP_OC_Module, self).__init__()
        self.context = nn.Sequential(nn.Conv2d(features, out_features, kernel_size=3, padding=1, dilation=1, bias=True),
                                     nn.BatchNorm2d(out_features),
                                     nn.ReLU(),
                                     BaseOC_Context_Module(in_channels=out_features, out_channels=out_features,
                                                           key_channels=out_features // 2, value_channels=out_features,
                                                           dropout=0, sizes=([2])))
        self.conv2 = nn.Sequential(nn.Conv2d(features, out_features, kernel_size=1, padding=0, dilation=1, bias=False),
                                   nn.BatchNorm2d(out_features),
                                   nn.ReLU(),)
        self.conv3 = nn.Sequential(
            nn.Conv2d(features, out_features, kernel_size=3, padding=dilations[0], dilation=dilations[0], bias=False),
            nn.BatchNorm2d(out_features),
            nn.ReLU(),)
        self.conv4 = nn.Sequential(
            nn.Conv2d(features, out_features, kernel_size=3, padding=dilations[1], dilation=dilations[1], bias=False),
            nn.BatchNorm2d(out_features),
            nn.ReLU(),)
        self.conv5 = nn.Sequential(
            nn.Conv2d(features, out_features, kernel_size=3, padding=dilations[2], dilation=dilations[2], bias=False),
            nn.BatchNorm2d(out_features),
            nn.ReLU(),)

        self.conv_bn_dropout = nn.Sequential(
            nn.Conv2d(out_features * 5, out_features, kernel_size=1, padding=0, dilation=1, bias=False),
            nn.BatchNorm2d(out_features),
            nn.ReLU(),
            nn.Dropout2d(0.1)
        )

    def _cat_each(self, feat1, feat2, feat3, feat4, feat5):
        assert (len(feat1) == len(feat2))
        z = []
        for i in range(len(feat1)):
            z.append(torch.cat((feat1[i], feat2[i], feat3[i], feat4[i], feat5[i]), 1))
        return z

    def forward(self, x):
        if isinstance(x, Variable):
            _, _, h, w = x.size()
        elif isinstance(x, tuple) or isinstance(x, list):
            _, _, h, w = x[0].size()
        else:
            raise RuntimeError('unknown input type')

        feat1 = self.context(x)
        feat2 = self.conv2(x)
        feat3 = self.conv3(x)
        feat4 = self.conv4(x)
        feat5 = self.conv5(x)

        if isinstance(x, Variable):
            out = torch.cat((feat1, feat2, feat3, feat4, feat5), 1)
        elif isinstance(x, tuple) or isinstance(x, list):
            out = self._cat_each(feat1, feat2, feat3, feat4, feat5)
        else:
            raise RuntimeError('unknown input type')

        output = self.conv_bn_dropout(out)
        return output

class SENet(nn.Module):
    def __init__(self, block, layers, groups, reduction, dropout_p=0.2,
                 inplanes=128, input_3x3=True, downsample_kernel_size=3,
                 downsample_padding=1, num_classes=None):

        super(SENet, self).__init__()
        self.inplanes = inplanes
        if input_3x3:
            layer0_modules = [
                ('conv1', nn.Conv2d(3, 64, 3, stride=2, padding=1,
                                    bias=False)),
                ('bn1', nn.BatchNorm2d(64)),
                ('relu1', nn.ReLU(inplace=True)),
                ('conv2', nn.Conv2d(64, 64, 3, stride=1, padding=1,
                                    bias=False)),
                ('bn2', nn.BatchNorm2d(64)),
                ('relu2', nn.ReLU(inplace=True)),
                ('conv3', nn.Conv2d(64, inplanes, 3, stride=1, padding=1,
                                    bias=False)),
                ('bn3', nn.BatchNorm2d(inplanes)),
                ('relu3', nn.ReLU(inplace=True)),
            ]
        else:
            layer0_modules = [
                ('conv1', nn.Conv2d(3, inplanes, kernel_size=7, stride=2,
                                    padding=3, bias=False)),
                ('bn1', nn.BatchNorm2d(inplanes)),
                ('relu1', nn.ReLU(inplace=True)),
            ]
        # To preserve compatibility with Caffe weights `ceil_mode=True`
        # is used instead of `padding=1`.
        # layer0_modules.append(('pool', nn.MaxPool2d(3, stride=2,
        #                                             ceil_mode=True)))
        self.layer0 = nn.Sequential(OrderedDict(layer0_modules))
        self.layer1 = self._make_layer(
            block,
            planes=64,
            blocks=layers[0],
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=1,
            downsample_padding=0
        )
        self.layer2 = self._make_layer(
            block,
            planes=128,
            blocks=layers[1],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )
        self.layer3 = self._make_layer(
            block,
            planes=256,
            blocks=layers[2],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )
        self.layer4 = self._make_layer(
            block,
            planes=512,
            blocks=layers[3],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )
        self.avg_pool = nn.AvgPool2d(7, stride=1)
        self.dropout = nn.Dropout(dropout_p) if dropout_p is not None else None
        # if num_classes:
        #     self.last_linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, groups, reduction, stride=1,
                    downsample_kernel_size=1, downsample_padding=0):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=downsample_kernel_size, stride=stride,
                          padding=downsample_padding, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, groups, reduction, stride,
                            downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups, reduction))

        return nn.Sequential(*layers)

class Bottleneck(nn.Module):
    """
    Base class for bottlenecks that implements `forward()` method.
    """

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)

        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.se_module(out) + residual
        out = self.relu(out)

        return out
class SEResNeXtBottleneck(Bottleneck):
    """
    ResNeXt bottleneck type C with a Squeeze-and-Excitation module.
    """
    expansion = 4

    def __init__(self, inplanes, planes, groups, reduction, stride=1,
                 downsample=None, base_width=4):
        super(SEResNeXtBottleneck, self).__init__()
        width = int(math.floor(planes * (base_width / 64.)) * groups)
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, bias=False,
                               stride=1)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride,
                               padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(planes * 4, reduction=reduction)
        self.downsample = downsample
        self.stride = stride

def se_resnext101_32x4d(pretrained=True):
    model = SENet(SEResNeXtBottleneck, [3, 4, 23, 3], groups=32, reduction=16,
                  dropout_p=None, inplanes=64, input_3x3=False,
                  downsample_kernel_size=1, downsample_padding=0)
    if pretrained:
        url = 'http://data.lip6.fr/cadene/pretrainedmodels/se_resnext101_32x4d-3b2fe3d8.pth'
        pretrained_dict = model_zoo.load_url(url)
        model_dict = model.state_dict()
        #pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        for key in pretrained_dict.keys():
            if 'last_linear' in key:
                print(key)
                continue
            model_dict[key] = pretrained_dict[key]
        model.load_state_dict(model_dict)
        print("-----pretrain success_____")
    return model

class Decoder(nn.Module):
    def __init__(self, in_channels, channels, out_channels ,OC=False,dilation=(4,8,12)):
        super(Decoder, self).__init__()
        self.oc=OC
        self.conv1 =  ConvBn2d(in_channels,  channels, kernel_size=3, padding=1)
        self.conv2 =  ConvBn2d(channels, out_channels, kernel_size=3, padding=1)
        # self.conv1 = ConvBn2dV2(in_channels, channels, kernel_size=3, padding=1)
        # self.conv2 = ConvBn2dV2(channels, out_channels, kernel_size=3, padding=1)
        self.context = nn.Sequential(
                ASP_OC_Module(out_channels,out_channels,dilations=dilation)
                )
        # self.context = nn.Sequential(
        #     nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(out_channels),
        #     nn.ReLU(),
        #     BaseOC_Module(in_channels=out_channels, out_channels=out_channels, key_channels=out_channels//2, value_channels=out_channels-out_channels//2,
        #                   dropout=0.05, sizes=([1]))
        # )
        self.scse_gate = SCSEBlock(out_channels)


    def forward(self, x):

        x = F.relu(self.conv1(x), inplace=True)
        x = F.relu(self.conv2(x), inplace=True)
        if self.oc:
            y = self.context(x)
        # x= self.conv1(F.elu(x,inplace=True))
        # x = self.conv2(F.elu(x, inplace=True))
        x = self.scse_gate(x)
        if self.oc:
            return torch.cat((x,y),1)
        else:
            return x

class Unet_scSE_hyper(nn.Module):
    # def load_pretrain(self, pretrain_file):
    #     pretrain_dict = torch.load(pretrain_file)
    #     state_dict = {}
    #     keys = list(pretrain_dict.keys())
    #     for key in keys:
    #         if 'last_layer' in key:
    #             continue
    #         state_dict[(key)] = pretrain_dict[key]
    #     self.encoder.load_state_dict(state_dict)

    def __init__(self,class_num = 17,dilation=False):
        super().__init__()
        self.dilation = dilation
        self.encoder =se_resnext101_32x4d()
        self.encoder1 = self.encoder.layer0
        #self.encoder1 = self.encoder.conv1
        self.encoder2 = self.encoder.layer1  # 256
        self.encoder3 = self.encoder.layer2  # 512
        self.encoder4 = self.encoder.layer3  # 1024
        self.encoder5 = self.encoder.layer4  # 2048
        self.center = nn.Sequential(
            ConvBn2d(2048, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            ConvBn2d(1024, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        if self.dilation:
            self.center1 = nn.Sequential(
                ConvBn2d(512, 512, kernel_size=3, padding=2, dilation=2),
                nn.ReLU(inplace=True)
            )
            self.center2 = nn.Sequential(
                ConvBn2d(512, 512, kernel_size=3, padding=4, dilation=4),
                nn.ReLU(inplace=True)
            )
        self.decoder5 = Decoder(2048 + 512, 512, 64,OC=True,dilation=(2,4,6))
        self.decoder4 = Decoder(128 + 1024, 256, 64,OC=True,dilation=(4,8,12))
        self.decoder3 = Decoder(128 + 512, 128, 64)
        self.decoder2 = Decoder(64 + 256, 64, 64)
        self.decoder1 = Decoder(64, 32, 64)

        self.logit = nn.Sequential(
            nn.Conv2d(64*7, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,class_num , kernel_size=1, padding=0),
        )

    def forward(self, x):
        #print('x',x.size())
        e1 = self.encoder1(x)  #; print('e1',e1.size())
        e2 = self.encoder2(e1)   #; print('e2',e2.size())
        e3 = self.encoder3(e2)   #; print('e3',e3.size())
        e4 = self.encoder4(e3)   #; print('e4',e4.size())
        e5 = self.encoder5(e4)   #; print('e5',e5.size())
        #e1,e2,e3,e4,e5 = self.encoder(x)
        f = self.center(e5)  #; print('center',f.size())
        if self.dilation:
            f1= self.center1(f)#; print('center',f1.size())
            f2=self.center2(f1)#; print('center',f2.size())
            # f3=self.center3(f2); print('center',f3.size())
            # f4=self.center4(f3); print('center',f4.size())
            #f5=self.center5(f4)
            f = torch.add(f,1,f1)
            f = torch.add(f,1,f2)
        # f=torch.cat((
        #     f,
        #     f1,
            # f2,
            # f3,
            # f4,
        # ),1)
        f = F.interpolate(f, scale_factor=2, mode='bilinear', align_corners=True)
        d5 = self.decoder5(torch.cat([f, e5], 1))   #; print('d5',d5.size())
        d5 = F.interpolate(d5, scale_factor=2, mode='bilinear', align_corners=True)
        d4 = self.decoder4(torch.cat([d5, e4], 1))   #; print('d4',d4.size())
        d4 = F.interpolate(d4, scale_factor=2, mode='bilinear', align_corners=True)
        d3 = self.decoder3(torch.cat([d4, e3], 1))   #; print('d3',d3.size())
        d3 = F.interpolate(d3, scale_factor=2, mode='bilinear', align_corners=True)
        d2 = self.decoder2(torch.cat([d3, e2], 1))   #; print('d2',d2.size())
        d2 = F.interpolate(d2, scale_factor=2, mode='bilinear', align_corners=True)
        d1 = self.decoder1(d2)   #; print('d1',d1.size())
        f = torch.cat((
            d1,
            F.interpolate(d2, scale_factor=1, mode='bilinear', align_corners=False),
            F.interpolate(d3, scale_factor=2, mode='bilinear', align_corners=False),
            F.interpolate(d4, scale_factor=4, mode='bilinear', align_corners=False),
            F.interpolate(d5, scale_factor=8, mode='bilinear', align_corners=False)
        ), 1)
        f = F.dropout2d(f, p=0.20)
        logit = self.logit(f)   #; print('logit',logit.size())
        return logit


