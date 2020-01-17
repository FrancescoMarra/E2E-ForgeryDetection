# -*- coding: utf-8 -*-
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
# Copyright (c) 2019 Image Processing Research Group of University Federico II of Naples ('GRIP-UNINA').
# All rights reserved.
# This work should only be used for nonprofit purposes.
#
# By downloading and/or using any of these files, you implicitly agree to all the
# terms of the license, as specified in the document LICENSE.md
# (included in this package) and online at
# http://www.grip.unina.it/download/LICENSE_OPEN.txt
#

from E2E.networks.xception import *
from E2E.networks.FCnet_pytorch import *

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd
    def forward(self, x):
        return self.lambd(x)

class MLPNet(nn.Module):
    def __init__(self ,feat_in ,classes):
        super(MLPNet, self).__init__()
        self.fc1 = nn.Linear(feat_in, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def name(self):
        return "MLP"

class StaTnet(nn.Module):
    def __init__(self ,feat_net ,channels ,feat_out ,tile_pooling_size=None ,tile_pooling_stride=None ,preconv=None):
        super(StaTnet, self).__init__()
        self.channels = channels
        self.feat_out = feat_out

        if isinstance(feat_net, nn.Module):
            self.prefeat_net = None
            self.feat_net = feat_net
        else:
            assert len(feat_net) == 2
            self.prefeat_net = feat_net[0]
            self.feat_net = feat_net[1]

        if isinstance(preconv, nn.Module): self.preconv = preconv
        elif preconv is None or preconv.lower() =='none' : self.preconv = LambdaLayer(lambda x: x)
        elif preconv.lower() == 'tanh':  self.preconv = nn.Tanh()
        #elif preconv.lower() == 'conv': self.preconv =  get_preconv(self.channels)
        else: raise ValueError(preconv)


        self.MinPool2d = LambdaLayer(lambda x: -nn.AdaptiveMaxPool2d((1, 1))(-x) )
        self.SquaredAvgPool2d = LambdaLayer(lambda x: nn.AdaptiveAvgPool2d((1, 1))(torch.pow(x ,2)) )

        self.initValues = [torch.tensor(-float("Inf")) ,torch.tensor(float("Inf")) ,torch.tensor(0.) ,torch.tensor(0.)]

        self.tile_aggregation_ops = [
            nn.AdaptiveMaxPool2d((1, 1)),
            self.MinPool2d,
            nn.AdaptiveAvgPool2d((1, 1)),
            self.SquaredAvgPool2d
        ]

        self.tile_accumulate_ops  = [
            LambdaLayer(lambda x: torch.max(x, 0, keepdim=True)[0].squeeze()),
            LambdaLayer(lambda x: torch.min(x, 0, keepdim=True)[0].squeeze()),
            LambdaLayer(lambda x: torch.sum(x, 0, keepdim=True)[0].squeeze()),
            LambdaLayer(lambda x: torch.sum(x, 0, keepdim=True)[0].squeeze())
        ]

        self.accumulate_ops = [
            lambda h ,H: torch.max(h ,H),
            lambda h ,H: torch.min(h ,H),
            lambda h ,H: h+ H,
            lambda h, H: h + H
        ]
        self.post_ops = [
            lambda H, X_shape: H,
            lambda H, X_shape: H,
            lambda H, X_shape: torch.div(H, X_shape[0]),
            lambda H, X_shape: torch.div(H, X_shape[0])
        ]

        self.apply_ops = lambda ops_list, hist, H: [ops(h) for h, ops in zip(hist, ops_list)] if H is None else [
            ops(h, H) for h, H, ops in zip(hist, H, ops_list)]

    def forward(self, x, H, tile_accumulate=True):
        if self.prefeat_net is not None:
            with torch.no_grad():
                x = self.prefeat_net(x)

        x = self.preconv(x)
        hist = self.feat_net(x)
        hist = self.apply_ops(self.tile_aggregation_ops, [hist for i in range(len(self.tile_aggregation_ops))], None)
        if tile_accumulate:
            hist = self.apply_ops(self.tile_accumulate_ops, hist, None)
            if H is not None:
                hist = self.apply_ops(self.accumulate_ops, hist, H)

        hist = torch.stack(hist, 0)

        return hist

    def tile_accumulate(self,hist,H=None):
        hist = self.apply_ops(self.tile_accumulate_ops, hist, None)
        if H is not None:
            hist = self.apply_ops(self.accumulate_ops, hist, H)

        hist = torch.stack(hist, 0)
        return hist

    def post(self, H, X_shape, accumulate=True):
        if accumulate:
            H = torch.stack(self.apply_ops(self.post_ops, H, [X_shape for i in range(len(self.post_ops))]), 0)

        H = H.view((1, -1))

        return H

    def initFeat(self):
        return torch.stack([torch.ones((self.feat_out)) * self.initValues[i] for i in range(len(self.initValues))], 0)

    def name(self):
        return "StaTnet"

def FCnet_preconv(preconv_old, out_channels=1):
    np_extractor = FullConvNet()
    c = np_extractor.features[-1]

    if out_channels == c.out_channels:
        last_conv = c
    elif out_channels > c.out_channels:
        last_conv = nn.Conv2d(c.in_channels, out_channels, c.kernel_size, c.stride, c.padding, c.dilation, c.groups, c.bias is not None)
        last_conv.weight.data = c.weight.data.repeat(3, 1, 1, 1)  # todo: parametrizzare
        if c.bias is not None:
            last_conv.bias.data = c.bias.data.repeat(3)   # todo: parametrizzare
    else:
        last_conv = nn.Conv2d(c.in_channels, out_channels, c.kernel_size, c.stride, c.padding, c.dilation, c.groups, c.bias is not None)
        last_conv.weight.data = c.weight.data[:out_channels]
        if c.bias is not None:
            last_conv.bias.data = c.bias.data[:out_channels]
    np_extractor.features = torch.nn.Sequential(*np_extractor.features[:-1])
    np_extractor.eval()

    preconv = nn.Sequential()
    preconv.add_module('conv', last_conv)
    if preconv_old is None or preconv_old.lower() == 'none':
        preconv.add_module('identity', LambdaLayer(lambda x: x))
    #elif preconv_old.lower() == 'conv':
    #    preconv.add_module(preconv_old.lower(), get_preconv(out_channels))
    elif preconv_old.lower() == 'tanh':
        preconv.add_module(preconv_old.lower(), nn.Tanh())
    else:
        raise ValueError(preconv_old)
    preconv.train()
    return np_extractor, preconv
# -------------------- NET DEFINITION -----------------------------------------
preconv = None

def get_model(mode,aggregate_net_name, feat_net_name,classes, channel, tile_pooling_size, tile_pooling_stride, preconv):
    prefeat_net = None
    if mode == 'FULL': prefeat_net, preconv = FCnet_preconv(preconv, channel)

    if feat_net_name == 'xception':
        feats = 2048
        feat_net = xception(classes)
        feat_net.forward = feat_net.features
        feat_net.last_linear = None
    else:
        raise Exception('No Network knowns with the name: ' + feat_net_name)

    if aggregate_net_name.lower() == 'statnet':
        model = StaTnet((prefeat_net, feat_net), channel, feats, tile_pooling_size, tile_pooling_stride, preconv)
        feats = feats * len(model.tile_aggregation_ops)
    else:
        raise Exception('No Aggregate Network knowns with the name: ' + aggregate_net_name)

    classifier = MLPNet(feats, classes)

    return model,classifier, feats

