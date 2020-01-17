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

import numpy as np
from skimage.util import view_as_blocks,view_as_windows
from math import floor
from scipy.interpolate import interp2d
from scipy.io import savemat


################################################
import matplotlib.pyplot as plt


def np_im2patch(im, pShape,pStride=None):
    if np.isscalar(pShape):
        pShape = (pShape, pShape)

    if pStride is None:
        pStride = (pShape, pShape)

    if np.isscalar(pStride):
        pStride = (pStride, pStride)

    imShape = im.shape
    if im.ndim == 3:
        if imShape[2] == 1:
            im = im.squeeze()
        elif len(pShape) == 2:
            pShape = (pShape[0], pShape[1], imShape[2])
            pStride = (pStride[0], pStride[1], imShape[2])

    pad = np.array( np.add( np.multiply ( np.ceil( np.divide( np.subtract( imShape, np.subtract(pShape, pStride) )  , pStride, dtype=np.float) ),pStride),    np.subtract(pShape, pStride)) - imShape,dtype=np.int)
    pad = pad[0:2]
    assert((pad >= 0).all())


    if pad.sum() == 0:
        p = view_as_windows(im, pShape,pStride)
        if im.ndim == 3 and p.shape[2] == 1:
            p = np.squeeze(p, axis=2)
        return p

    if im.ndim == 2:
        im_post = np.pad(im, ((0, pad[0]), (0, pad[1])), 'constant')
        im_pre = np.pad(im, ((pad[0], 0), (pad[1], 0)), 'constant')
        im_pre_c = np.pad(im, ((pad[0], 0), (0, pad[1])), 'constant')
        im_pre_r = np.pad(im, ((0, pad[0]), (pad[1], 0)), 'constant')
    elif im.ndim == 3:
        im_post = np.pad(im, ((0, pad[0]), (0, pad[1]), (0,0)), 'constant')
        im_pre = np.pad(im, ((pad[0], 0), (pad[1], 0), (0,0)), 'constant')
        im_pre_c = np.pad(im, ((pad[0], 0), (0, pad[1]), (0,0)), 'constant')
        im_pre_r = np.pad(im, ((0, pad[0]), (pad[1], 0), (0,0)), 'constant')
    else:
        raise NotImplementedError('2D or 3D input images are accepted.')

    p = view_as_windows(im_post, pShape,pStride).copy()

    if im.ndim == 3 and p.shape[2] == 1:
        p = np.squeeze(p, axis=2)

    for i in range(p.shape[0]-1):
        p[i, -1] = im[i * pStride[0]:i * pStride[0] + pShape[0],-pShape[1]:, :]
    for j in range(p.shape[1]-1):
        p[-1, j] = im[-pShape[0]:,j * pStride[1]:j * pStride[1] + pShape[1], :]

    p[-1, -1] = im[-pShape[0]:,-pShape[1]:, :]

    return p


def np_patch2im(p, imShape,pStride=None,aggregation_mean=True):
    pShape = p.shape[2:]

    if pStride is None:
        pStride = pShape
    if np.isscalar(pStride):
        pStride = (pStride, pStride,imShape[2])


    img = np.zeros(imShape)
    #obj = plotimg(img)
    for i in range(0,p.shape[0]-1):
        for j in range(0,p.shape[1]-1):
            #print(i,j,i * pStride, i * pStride + pShape[0],(j * pStride),(j * pStride + pShape[1]))
            img[(i * pStride[0]):(i * pStride[0] + pShape[0]), (j * pStride[1]):(j * pStride[1] + pShape[1]), :] += p[i,j]
            #plotimg(img,obj)

    for i in range(p.shape[0]-1):
        img[i * pStride[0]:i * pStride[0] + pShape[0],-pShape[1]:, :] += p[i, -1]
        #plotimg(img, obj)
    for j in range(p.shape[1]-1):
        img[-pShape[0]:,j * pStride[1]:j * pStride[1] + pShape[1], :] += p[-1, j]
        #plotimg(img, obj)

    img[-pShape[0]:,-pShape[1]:, :] += p[-1, -1]


    if pStride != pShape and aggregation_mean:
        p_1 = np_im2patch(np.ones(imShape), pShape, pStride)
        div,_ = np_patch2im(p_1, imShape, pStride,False)
        #plotimg(div / np.max(div))
        # div = np_patch2im(np.ones_like(p), imShape,pStride,aggregation_mean=False)
        # div = np_patch2im(np.ones_like(p), imShape,pStride,aggregation_mean=False)
        #div = np_patch2im(np.ones_like(p), imShape,pStride,aggregation_mean=False)
        img /= div
        return img,div
    return img,None


