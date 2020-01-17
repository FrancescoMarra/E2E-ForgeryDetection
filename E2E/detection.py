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


from E2E.networks.networks import *
import torch
import E2E.parameters as parameters
torch.backends.cudnn.benchmark=True
from E2E.utility.utility_np import np_im2patch
import numpy as np
from E2E.dataloaders.data_loader import get_tranform


# EVALUATION ------------------------------------------
global models
global classifiers
models = dict()
classifiers = dict()

def load_model(mode):
    global models
    global classifiers
    if (mode in models.keys()) and (mode in classifiers.keys()):
        model = models[mode]
        classifier = classifiers[mode]
    else:
        model, classifier, _ = get_model(mode,parameters.aggregate_net_name, parameters.feat_net_name, parameters.classes, parameters.channel, parameters.tile_pooling_size, parameters.tile_pooling_stride, parameters.preconv)
        model_file, class_file = parameters.weights_name(parameters.ds, mode)
        model.load_state_dict(torch.load(model_file, map_location='cpu'), strict=True)
        classifier.load_state_dict(torch.load(class_file, map_location='cpu'), strict=True)
        models[mode] = model
        classifiers[mode] = classifier

    return model,classifier

def get_score(X,mode,use_cuda=False):
    model,classifier = load_model(mode)

    #it has effect only the first time
    if use_cuda:
        model.cuda()
        classifier.cuda()
    else:
        model.cpu()
        classifier.cpu()

    score = classify_whole_img(X, model, classifier, get_tranform(mode), parameters.tile_size, parameters.tile_stride,mode)

    return score

def sigmoid(x , a, c):
   '''
   Returns sigmoid function
   output between 0 and 1
   Function parameters c = center; b = width
   '''
   s= 1/(1+np.exp((-a)*(x-c)))
   return s


def score_fusion(scores,confidences):
    #Very simple fusion of the [0,1] scores
    #does not use confidences on scores
    return np.mean(scores)

def preload(mode='RGB'):

    if mode == 'FUSION':
        _, _ = load_model('RGB')
        _, _ = load_model('N')
        _, _ = load_model('RGN')
    elif mode in ['RGB','N','RGN']:
        _, _ = load_model(mode)
    return


def detection(X, RGB=None, NP=None, RGN=None,mode='RGB'):

    if mode == 'FUSION':
        score_RGB = get_score(RGB,'RGB',parameters.use_cuda)
        score_NP =  get_score(NP, 'N',parameters.use_cuda)
        score_RGN = get_score(RGN, 'RGN',parameters.use_cuda)
        score = score_fusion([score_RGB,score_NP,score_RGN],[1,1,1])
    elif mode in ['RGB','N','RGN','FULL']:
        score = get_score(X,mode,parameters.use_cuda)

    return score

def classify_whole_img(img,model,classifier,transform,tile_size,tile_stride,mode):
    model = model.eval()
    classifier = classifier.eval()

    img = transform(img).cpu().numpy().transpose(1,2,0)
    X = np_im2patch(img,tile_size,tile_stride)
    X = torch.FloatTensor(X)
    X = X.permute(0,1,4,2,3).contiguous().view((-1,3,tile_size,tile_size)) #TODO: number of channels
    H = model.initFeat()
    if parameters.use_cuda: H = H.cuda()
    with torch.no_grad():
        chunk = torch.chunk(X, int(np.ceil(X.shape[0] / parameters.TILE_BATCH_SIZE)), 0)
        #chunk = [c.cuda(non_blocking=True) for c in chunk]
        for x in chunk:
            if parameters.use_cuda: x = x.cuda() #To be sure
            H = model(x, H)

        H = model.post(H,X.shape)
        out = classifier(H) 
        
    score = np.diff(out.cpu().data.numpy())[0]
    score = sigmoid(score,parameters.sig_a[mode],parameters.sig_c[mode])

    return score 



