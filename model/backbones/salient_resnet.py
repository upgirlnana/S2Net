# encoding: utf-8


import math

import torch
from torch import nn


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class SalieBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(SalieBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class SalieBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(SalieBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

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

        out += residual
        out = self.relu(out)

        return out


class SalieResNet(nn.Module):
    def __init__(self, last_stride=2, block=SalieBottleneck, layers=[3, 4, 6, 3]):
        self.inplanes = 64
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)   # add missed relu
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer2_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.layer3_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.layer4_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.whole1_channel = nn.Sequential(nn.Linear(256, int(256 / 8)),
                                            # nn.ReLU(inplace=True),
                                            nn.Linear(int(256 / 8), 256),
                                            nn.Sigmoid())
        self.whole1_spatial = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3,padding=1),
                                            # nn.AdaptiveAvgPool2d((16, 8)),
                                            nn.ReLU(inplace=True),
                                            nn.Conv2d(256, 256, kernel_size=3,padding=1),
                                            nn.ReLU(inplace=True),
                                            nn.Conv2d(256, 1, kernel_size=1),
                                            nn.Sigmoid())
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=last_stride)

        self.layer2_channel = nn.Sequential(nn.Linear(512, int(512 / 8)),
                                            nn.Linear(int(512 / 8), 512),
                                            nn.Sigmoid())
        self.layer3_channel = nn.Sequential(nn.Linear(1024, int(1024 / 8)),
                                            nn.Linear(int(1024 / 8), 1024),
                                            nn.Sigmoid())
        self.layer4_channel = nn.Sequential(nn.Linear(2048, int(2048 / 8)),
                                            nn.Linear(int(2048 / 8), 2048),
                                            nn.Sigmoid())

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)    # add missed relu
        x = self.maxpool(x)
        x = self.layer1(x)
        x_att1_spatial = self.whole1_spatial(x)

        # x_channel=self.whole1_channel()
        x_att1_channel = self.layer2_pool(x)
        x_att1_channel = x_att1_channel.view(x_att1_channel.size(0), -1)
        x_att1_channel = self.whole1_channel(x_att1_channel)
        x_att1_channel = torch.unsqueeze(torch.unsqueeze(x_att1_channel, (2)), 3)
        x=x*x_att1_spatial

        x=x*x_att1_channel
        # import pdb
        # pdb.set_trace()
        x = self.layer2(x)
        # layer2_ch = self.layer2_pool(x)
        # layer2_ch = layer2_ch.view(layer2_ch.size(0), -1)
        # layer2_ch = self.layer2_channel(layer2_ch)
        # layer2_ch = torch.unsqueeze(torch.unsqueeze(layer2_ch, (2)), 3)
        # x = x * layer2_ch

        # x_att1_channel = self.layer2_pool(x)
        # x_att1_channel = x_att1_channel.view(x_att1_channel.size(0), -1)
        # x_att1_channel = self.whole1_channel(x_att1_channel)
        # x_att1_channel = torch.unsqueeze(torch.unsqueeze(x_att1_channel, (2)), 3)
        #

        # # print('x_att1_spatial', x_att1_spatial.size())
        # # print('x_att1_channel', x_att1_channel.size())
        # att1_mask = x_att1_spatial * x_att1_channel


        x = self.layer3(x)
        # layer3_ch = self.layer3_pool(x)
        # layer3_ch = layer3_ch.view(layer3_ch.size(0), -1)
        #
        # layer3_ch = self.layer3_channel(layer3_ch)
        # layer3_channel = torch.unsqueeze(torch.unsqueeze(layer3_ch, (2)), 3)
        # x = x * layer3_channel
        # x=att1_mask*x
        x = self.layer4(x)
        # layer4_ch = self.layer4_pool(x)
        # layer4_ch = layer4_ch.view(layer4_ch.size(0), -1)
        # layer4_ch = self.layer4_channel(layer4_ch)
        # layer4_ch = torch.unsqueeze(torch.unsqueeze(layer4_ch, (2)), 3)
        # x = x * layer4_ch


        return x,x_att1_spatial

    def load_param(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            if 'fc' in i:
                continue
            self.state_dict()[i].copy_(param_dict[i])

    def random_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

