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


import torch
import E2E.parameters as parameteres
torch.backends.cudnn.benchmark=True
from torchvision import transforms
import numpy as np
from PIL import Image
from E2E.networks.FCnet_pytorch import get_FCnet,extractFC_stride
from E2E.utility.utilityRead import imread2f,imread2f_pil

global FC_net
FC_net = None

def get_tranform(mode):

    normalize_N = lambda x : np.clip(x, np.percentile(x,1),np.percentile(x,99))
    normalize_RGN = lambda x :  np.stack([ x[:,:,0] *2 - 1,x[:,:,1] *2 - 1 , np.clip( x[:,:,2], np.percentile(x[:,:,2],1),np.percentile(x[:,:,2],99))],2)
    normalize_FULL = lambda x:  x * 255.0 /256.0

    if mode == 'RGB':
        transform = transforms.Compose([transforms.ToTensor()])
    elif mode == 'FULL':
        transform = transforms.Compose([normalize_FULL, transforms.ToTensor()])
    elif mode == 'N':
        transform = transforms.Compose([normalize_N,transforms.ToTensor()])
    elif mode == 'RGN':
        transform = transforms.Compose([normalize_RGN,transforms.ToTensor()])

    return transform

def getNP(img):
    global FC_net

    if FC_net is None:
        FC_net = get_FCnet(parameteres.FCnet_weights)

    if parameteres.use_cuda:
        FC_net = FC_net.cuda()

    NP = extractFC_stride(img*255/256,FC_net,parameteres.use_cuda)
    if parameteres.use_cuda:
        FC_net.cpu()

    return NP

def loader(path,mode):
    X, RGB, NP, RGN, im_mode = None,None,None,None,None

    if mode == 'RGB' or mode == 'FULL':
        RGB,im_mode = RGB_loader(path)
        X = RGB
    elif mode == 'N':
        NP,RGB,im_mode = NP_loader(path)
        X = NP
    elif (mode == 'RGN') or (mode == 'FUSION'):
        RGN, RGB, NP,im_mode = RGN_loader(path)
        X = RGN

    return X,RGB,NP,RGN,im_mode

def RGB_loader(path):
    try:
        RGB, mode = imread2f(path, channel=3)

    except Exception as e:
        raise Exception("Error in opening image file")
    return RGB,mode

def NP_loader(path,channel=3):
    RGB,mode = RGB_loader(path)
    NP = getNP(RGB)

    if channel == 1:
        if len(NP.shape)>2: NP = NP[:,:,0]
        NP = np.expand_dims(NP, 2)
    elif channel==3:
        if len(NP.shape) > 2: NP = NP[:, :, 0]
        NP = np.stack([NP,NP,NP],2)

    return NP,RGB,mode

def RGN_loader(path):
    NP, RGB,mode = NP_loader(path,channel=3)
    RGN = np.stack( [RGB[:,:,0],RGB[:,:,1] ,NP[:,:,0]],2 )
    return RGN,RGB,NP,mode

