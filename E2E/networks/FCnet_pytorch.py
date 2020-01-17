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

from numpy import sqrt, maximum
import torch
from torch.nn import Conv2d, BatchNorm2d, ReLU, Sequential
import numpy as np
import E2E.parameters as parameters

class FullConvNet(torch.nn.Module):
    def __init__(self, n_channels=3, num_levels=17, padding="same", momentum=0.1):
        super(FullConvNet, self).__init__()

        self._num_levels = num_levels
        self._actfun = [ReLU(), ] * (self._num_levels - 1) + [None, ]
        self._f_size = [3, ] * self._num_levels
        padding = padding.lower()
        if padding == 'valid':
            self._f_pad = [0, ] * self._num_levels
        elif padding == 'same':
            self._f_pad = [1, ] * self._num_levels
        else:
            raise ValueError('Padding must be either "valid" or "same", instead "%s" is given.' % padding)
        self._f_num = [64, ] * (self._num_levels - 1) + [1, ]
        self._f_in = [n_channels, ] + [64, ] * (self._num_levels - 1)
        self._f_stride = [1, ] * self._num_levels
        self._bnorm = [False, ] + [True, ] * (self._num_levels - 2) + [False, ]
        self._bnorm_epsilon = 1e-5
        self._bnorm_momentum = momentum
        self.decay_list = []

        self.features = Sequential()
        for i in range(self._num_levels):

            # convolution (with bias if batch normalization is not executed in this level)
            self.features.add_module(module=Conv2d(self._f_in[i], self._f_num[i], self._f_size[i], self._f_stride[i], self._f_pad[i], bias=not self._bnorm[i]),
                                     name='level_%d/conv' % i)
            torch.nn.init.normal_(self.features[-1].weight, std=sqrt(2.0/self._f_size[i]*self._f_size[i]*maximum(self._f_in[i], self._f_num[i])))
            self.decay_list.append(self.features[-1].weight)

            # eventual batch normalization
            if self._bnorm[i]:
                self.features.add_module(module=BatchNorm2d(self._f_num[i], eps=self._bnorm_epsilon, momentum=self._bnorm_momentum, affine=True),
                                         name='level_%d/bn' % i)

            # eventual activation
            if self._actfun[i] is not None:
                self.features.add_module(module=self._actfun[i],
                                         name='level_%d/activation' % i)

    def load_pretrained_weights(self, filename):
        try:
            self.load_state_dict(torch.load(filename))
        except:
            print('Trying to convert file %s to ./temp.pth' % filename)
            convert_numpy_weights(filename, './temp.pth')
            print('Conversion compleated!\nLoading ./temp.pth')
            self.load_state_dict(torch.load('./temp.pth'))
            print('Loading compleated!\nRemoving ./temp.pth')
            try:
                from os import remove
                remove(filename + '.pth')
            except:
                print('Cannot remove ./temp.pth')

    def forward(self, images):
        return self.features(images)

def convert_numpy_weights(input_filename, output_filename=None):
    import numpy as np
    in_file = np.load(input_filename)
    num_levels = max([int(name.split('/')[0].split('_')[1]) for name in in_file['list']]) + 1
    net = FullConvNet(num_levels)
    for name, mod in net.features.named_modules():
        print('Module %s:' % name)
        if isinstance(mod, Conv2d):
            mod.weight.data = torch.from_numpy(np.transpose(in_file[name + '/weights:0'], (3, 2, 0, 1)))
            print('+ weight loaded')
            if 'level_0' in name or 'level_%d' % (num_levels-1) in name:
                mod.bias.data = torch.from_numpy(in_file[name.split('/')[0] + '/bias/beta:0'])
                print('+ bias loaded')
            else:
                print('- no bias')
        elif isinstance(mod, BatchNorm2d):
            mod.running_mean = torch.from_numpy(in_file[name + '/moving_mean:0'])
            print('+ moving_mean loaded')
            mod.running_var = torch.from_numpy(in_file[name + '/moving_variance:0'])
            print('+ moving_variance loaded')
            mod.weight.data = torch.from_numpy(in_file[name + '/gamma:0'])
            print('+ weight loaded')
            if not('level_0' in name or 'level_%d' % (num_levels-1) in name):
                mod.bias.data = torch.from_numpy(in_file[name.split('/')[0] + '/bias/beta:0'])
                print('+ bias loaded')
            else:
                print('- no bias')
        print('* DONE\n')
    if output_filename is None:
        output_filename = input_filename + '.pth'
    print('Conversion completed: saving weights in %s' % output_filename)
    torch.save(net.state_dict(), output_filename)

def get_FCnet(weights_filename,padding='same'):
    FCnet = FullConvNet(padding=padding)
    FCnet.load_state_dict(torch.load(weights_filename, map_location='cpu'))
    FCnet.eval()
    return FCnet 

def extractFC(FCnet, img, use_cuda = False):
    image = np.transpose(img , (2, 0, 1)) # * 255. / 256.
    image = torch.autograd.Variable(torch.from_numpy(image.astype(np.float32))[:3].unsqueeze(0), requires_grad=False)
    FCnet = FCnet.eval()
    if use_cuda:
        FCnet = FCnet.cuda()
        image = image.cuda()

    with torch.no_grad():
        res = FCnet(image)

    res = res.cpu().data.numpy().squeeze()
    return res

def extractFC_stride(img,FCnet,use_cuda=False):
    slice_dim = 512

    if use_cuda:
        FCnet = FCnet.cuda()

    if img.shape[1] > 2700:
        posSplitend = img.shape[1] - slice_dim - 34
        res = extractFC(FCnet,  img[:, :(slice_dim + 34),:], use_cuda)
        res = res[:, :-34]
        posSplit = slice_dim
        while posSplit < posSplitend:
            resA = extractFC(FCnet, img[:, (posSplit - 34): (posSplit + slice_dim + 34), :] ,use_cuda)
            posSplit = posSplit + slice_dim
            res = np.concatenate((res, resA[:, 34:-34]), 1)
        resC = extractFC(FCnet, img[:,  (posSplit - 34):, :], use_cuda)
        res = np.concatenate((res, resC[:, 34:]), 1)
    elif img.shape[1] > 1024:
        posSplit = (int(img.shape[1] // 3), int(img.shape[1] // 3 * 2))
        resA = extractFC(FCnet, img[:, :posSplit[0] + 34, :],use_cuda)
        resB = extractFC(FCnet, img[:,  posSplit[0] - 34: posSplit[1] + 34, :],use_cuda)
        resC = extractFC(FCnet, img[:,  posSplit[1] - 34:, :],use_cuda)
        res = np.concatenate((resA[:, :-34], resB[:, 34:-34], resC[:, 34:]), 1)
    elif img.shape[1] > 512:
        posSplit = img.shape[1] // 2
        resA = extractFC(FCnet, img[:, :posSplit + 34, :],use_cuda)
        resB = extractFC(FCnet, img[:, posSplit - 34:, :],use_cuda)
        res = np.concatenate((resA[:, :-34], resB[:, 34:]), 1)
    else:
        res = extractFC(FCnet, img,use_cuda)

    if use_cuda:
        FCnet = FCnet.cpu()

    res = np.squeeze(res)
    return res


