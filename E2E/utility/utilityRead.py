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


from PIL.JpegImagePlugin import convert_dict_qtables
from PIL import Image
import numpy as np
import rawpy

def imread2f_pil(stream, channel = 1, dtype = np.float32):
    img = Image.open(stream)
    mode = img.mode
    
    if channel == 3:
        img = img.convert('RGB')
        img = np.asarray(img).astype(dtype) / 255.0
    elif channel == 1:
        if img.mode == 'L':
            img = np.asarray(img).astype(dtype) / 255.0
        else:
            img = img.convert('RGB')
            img = np.asarray(img).astype(dtype)
            img = (0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2])/255.0
    else:
        img = np.asarray(img).astype(dtype) / 255.0
    return img, mode

def imread2f_raw(stream, channel = 1, dtype = np.float32):
    raw = rawpy.imread(stream)
    img = raw.postprocess()
    raw.close()
    ori_dtype = img.dtype
    img = np.asarray(img).astype(dtype)
    if channel == 1:
        img = (0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2])
    
    if ori_dtype == np.uint8:
        img = img / 255.0
    elif ori_dtype == np.uint16:
        img = img / ((2.0 ** 16)-1)
    elif ori_dtype == np.uint32:
        img = img / ((2.0 ** 32)-1)
    elif ori_dtype == np.uint64:
        img = img / ((2.0 ** 64)-1)
    elif ori_dtype == np.uint128:
        img = img / ((2.0 ** 128)-1)
    return img, 'RAW'

def imread2f(stream, channel = 1, dtype = np.float32):
    try:
        return imread2f_raw(stream, channel=channel, dtype=dtype)
    except:
        return imread2f_pil(stream, channel=channel, dtype=dtype)

