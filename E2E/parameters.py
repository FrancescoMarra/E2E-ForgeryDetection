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

mode = 'RGB'
ds = 'E2E'
aggregate_net_name = 'statnet'
feat_net_name = 'xception'

classes = 2
channel = 3
tile_size = 256
tile_stride = 192

use_cuda = False

tile_pooling_size = 1
tile_pooling_stride = 1
TILE_BATCH_SIZE = 32

FCnet_weights = './models/FCnet/model_20.npz.pth'
preconv ='tanh'

sig_a ={'RGB': 0.8,
        'N': 0.4,
        'RGN': 0.3}
sig_c ={'RGB': 0.,
        'N': 0.1,
        'RGN': 0.}

def weights_name(ds,mode):
    return './models/{}/{}/last_model.th'.format(ds,mode), \
           './models/{}/{}/last_class.th'.format(ds,mode)
