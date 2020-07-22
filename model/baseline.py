# encoding: utf-8

import time
import numpy as np

import torch
from torch import nn
import matplotlib.pyplot as plt
from .backbones.body_part_atten import BodyBasicBlock,BodyResNet,BodyBottleneck
from .backbones.salient_resnet import SalieBasicBlock,SalieBottleneck,SalieResNet
savepath='/home/rxn/myproject/MGAN/pth/fig/map/market/'
def draw_features(width,height,x,savename):
    tic=time.time()
    fig = plt.figure(figsize=(16, 16))
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.05, hspace=0.05)
    for i in range(width*height):
        plt.subplot(height,width, i + 1)
        plt.axis('off')
        # plt.tight_layout()
        img = x[0, i, :, :]

        # img = x
        # print(img)
        pmin = np.min(img)
        pmax = np.max(img)
        img = (img - pmin) / (pmax - pmin + 0.000001)
        plt.imshow(img, cmap='hot')
        # plt.imshow(img)
        print("{}/{}".format(i,width*height))
    fig.savefig(savename, dpi=100)
    print('fig saved')
    fig.clf()
    plt.close()
    print("time:{}".format(time.time()-tic))
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)
class Baseline(nn.Module):
    in_planes = 2048
    def __init__(self, num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice):
        super(Baseline, self).__init__()
        self.base = SalieResNet(last_stride=last_stride,
                                    block=SalieBottleneck,
                                    layers=[3, 4, 6, 3])

        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......')

        self.gap = nn.AdaptiveAvgPool2d(1)
        # self.gap = nn.AdaptiveMaxPool2d(1)
        self.num_classes = num_classes
        self.neck = neck
        self.neck_feat = neck_feat

        if self.neck == 'no':
            self.classifier = nn.Linear(self.in_planes, self.num_classes)
            # self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)     # new add by luo
            # self.classifier.apply(weights_init_classifier)  # new add by luo
        elif self.neck == 'bnneck':
            self.bottleneck = nn.BatchNorm1d(self.in_planes)
            self.bottleneck.bias.requires_grad_(False)  # no shift
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)

            self.bottleneck.apply(weights_init_kaiming)
            self.classifier.apply(weights_init_classifier)

    def forward(self, x):

        feature, mask = self.base(x)
        global_feat = self.gap(feature)  # (b, 2048, 1, 1)

        global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)
        if self.neck == 'no':
            feat = global_feat
        elif self.neck == 'bnneck':
            feat = self.bottleneck(global_feat)  # normalize for angular softmax

        if self.training:
            cls_score = self.classifier(feat)

            return cls_score, global_feat, mask  # global feature for triplet loss
        else:
            if self.neck_feat == 'after':
                # print("Test with feature after BN")
                return feat
            else:
                # print("Test with feature before BN")
                return global_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        # print (param_dict)
        for i in param_dict:
            if 'classifier' in i:
                continue
            # print (self.state_dict()[i])
            self.state_dict()[i].copy_(param_dict[i])

class Baseline1(nn.Module):
    in_planes = 512
    in_planes1 = 2048
    in_planes2 = 2560
    # #
    def __init__(self, num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice):
        super(Baseline1, self).__init__()
        if model_name == 'bodyresnet50':
            print('512 bodyresnet50')
            self.base = BodyResNet(last_stride=last_stride,
                               block=BodyBottleneck,
                               layers=[3, 4, 6, 3])
        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......')

        self.gap = nn.AdaptiveAvgPool2d(1)
        # self.gap = nn.AdaptiveMaxPool2d(1)
        self.num_classes = num_classes
        self.neck = neck
        self.neck_feat = neck_feat

        if self.neck == 'no':
            self.classifier = nn.Linear(self.in_planes, self.num_classes)
        elif self.neck == 'bnneck':
            self.bottleneck = nn.BatchNorm1d(self.in_planes)
            self.bottleneck.bias.requires_grad_(False)  # no shift
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)

            self.bottleneck.apply(weights_init_kaiming)
            self.classifier.apply(weights_init_classifier)
            self.bottleneck1 = nn.BatchNorm1d(self.in_planes1)
            self.bottleneck1.bias.requires_grad_(False)  # no shift
            self.classifier1 = nn.Linear(self.in_planes1, self.num_classes, bias=False)

            self.bottleneck1.apply(weights_init_kaiming)
            self.classifier1.apply(weights_init_classifier)
            # #
            self.bottleneck2 = nn.BatchNorm1d(self.in_planes2)
            self.bottleneck2.bias.requires_grad_(False)  # no shift
            self.classifier2 = nn.Linear(self.in_planes2, self.num_classes, bias=False)

            self.bottleneck2.apply(weights_init_kaiming)
            self.classifier2.apply(weights_init_classifier)

    def forward(self, x):

        feature,mask=self.base(x)
        global_feat = self.gap(feature[0])  # (b, 2048, 1, 1)
        head_feat = self.gap(feature[1])  # (b, 2048, 1, 1)
        upper_feat = self.gap(feature[2])  # (b, 2048, 1, 1)
        lower_feat = self.gap(feature[3])  # (b, 2048, 1, 1)
        shoes_feat = self.gap(feature[4])  # (b, 2048, 1, 1)


        global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)
        head_feat = head_feat.view(head_feat.shape[0], -1)  # flatten to (bs, 2048)
        upper_feat = upper_feat.view(upper_feat.shape[0], -1)  # flatten to (bs, 2048)
        lower_feat = lower_feat.view(lower_feat.shape[0], -1)  # flatten to (bs, 2048)
        shoes_feat = shoes_feat.view(shoes_feat.shape[0], -1)  # flatten to (bs, 2048)
        combine_local = torch.cat(( head_feat, upper_feat, lower_feat,shoes_feat), 1)
        combine_global=torch.cat((global_feat,head_feat,upper_feat,lower_feat,shoes_feat),1)
        # import pdb
        # pdb.set_trace()
        if self.neck == 'no':
            feat = global_feat
        elif self.neck == 'bnneck':
            # print("NONE")
            feat = self.bottleneck(global_feat)  # normalize for angular softmax
            #
            # combine_global=self.bottleneck(combine_global)
            combine_local=self.bottleneck1(combine_local)
            # combine_global=self.bottleneck2(combine_global)

        if self.training:
            cls_score1 = self.classifier(global_feat)
            cls_score=self.classifier1(combine_local)
            return [cls_score,cls_score1],combine_global,mask  # global feature for triplet loss
        else:
            if self.neck_feat == 'after':
                # print("Test with feature after BN")
                return combine_global
            else:
                # print("Test with feature before BN")
                return global_feat



