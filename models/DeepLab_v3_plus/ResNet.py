import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from .sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from ..scSE import scSE,cSE

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnet_se101':'http://data.lip6.fr/cadene/pretrainedmodels/se_resnet101-7e38fcc6.pth',
}

class SEModule(nn.Module):

    def __init__(self, channels, reduction=4):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1,
                             padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1,
                             padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x

class SE_Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, bn_momentum=0.1):
        super(SE_Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = SynchronizedBatchNorm2d(planes, momentum=bn_momentum)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation, dilation=dilation, bias=False)
        self.bn2 = SynchronizedBatchNorm2d(planes, momentum=bn_momentum)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = SynchronizedBatchNorm2d(planes * self.expansion, momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.se = SEModule(planes * self.expansion)
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
        out = self.se(out)
        out += residual
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, bn_momentum=0.1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = SynchronizedBatchNorm2d(planes, momentum=bn_momentum)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation, dilation=dilation, bias=False)
        self.bn2 = SynchronizedBatchNorm2d(planes, momentum=bn_momentum)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = SynchronizedBatchNorm2d(planes * self.expansion, momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        #self.se = SeModule(planes * self.expansion)
        #self.triplet_attention = TripletAttention()


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

        #out = self.se(out)
        #out = self.triplet_attention(out)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, bn_momentum=0.1, pretrained=False, output_stride=16,model_name = None):
        if output_stride ==16:
            dilations = [1, 1, 1, 2]
            strides = [1, 2, 2, 1]
        elif output_stride == 8:
            dilations = [1, 1, 2, 4]
            strides = [1, 2, 1, 1]
        else:
            raise Warning("output_stride must be 8 or 16!")
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.model_name = model_name
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = SynchronizedBatchNorm2d(64, momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=strides[0], dilation=dilations[0],
                                       bn_momentum=bn_momentum)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=strides[1], dilation=dilations[1],
                                       bn_momentum=bn_momentum)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=strides[2], dilation=dilations[2],
                                       bn_momentum=bn_momentum)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=strides[3], dilation=dilations[3],
                                       bn_momentum=bn_momentum)

        self._init_weight()
        if pretrained:
            self._load_pretrained_model()


    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, bn_momentum=0.1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                SynchronizedBatchNorm2d(planes * block.expansion, momentum=bn_momentum),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation, downsample, bn_momentum=bn_momentum))
        self.inplanes = planes * block.expansion
        #layers.append(SeModule(planes * block.expansion))
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation, bn_momentum=bn_momentum))
        #layers.append(SeModule(planes * block.expansion))
        return nn.Sequential(*layers)

    def _load_pretrained_model(self):
        pretrain_dict = model_zoo.load_url(model_urls[self.model_name])
        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrain_dict.items():
            if k in state_dict:
                model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)
        print("Having loaded imagenet-pretrained successfully!")

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        low_level_feat = x
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)
        return x, low_level_feat

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()



def resnet101(bn_momentum=0.1, pretrained=False, output_stride=16):
    model = ResNet(Bottleneck, [3, 4, 23, 3], bn_momentum, pretrained, output_stride,model_name="resnet101")
    return model

def resnet50(bn_momentum=0.1, pretrained=False, output_stride=16):
    model = ResNet(Bottleneck, [3, 4, 6, 3], bn_momentum, pretrained, output_stride,model_name="resnet50")
    return model

def resnet152(bn_momentum=0.1, pretrained=False, output_stride=16):
    model = ResNet(Bottleneck, [3, 8, 36, 3], bn_momentum, pretrained, output_stride,model_name="resnet152")
    return model

def resnet_se101(bn_momentum=0.1, pretrained=False, output_stride=16):
    model = ResNet(SE_Bottleneck, [3, 4, 23, 3], bn_momentum, pretrained, output_stride,model_name="resnet_se101")
    return model

if __name__ == "__main__":
    model = resnet50(pretrained=False)
    model.eval()
    print(model)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)