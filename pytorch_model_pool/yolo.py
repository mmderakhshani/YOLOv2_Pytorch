from .model import Darknet19
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import pickle as pickle
from collections import OrderedDict
import math

class YoloV2(nn.Module):
    """Yolo version 2; It is an extented version of 
    yolo v1 capable of detecting 9000 object"""
    def __init__(self, modelUrl, extra_arg):
        super(YoloV2, self).__init__()
        
        self.modelUrl = modelUrl
        self.extra_arg = extra_arg

        self.darknet19 = Darknet19()
        with open(modelUrl, 'rb') as fp:
            self.weights = pickle.load(fp)
        fp.close()
        arch = self.darknet19.arch

        self.path1 = self.makeSequence(arch[0]) # path1
        self.path2 = self.makeSequence(arch[1]) # path2
        self.path3 = self.makeSequence(arch[2]) # path3
        self.path4 = self.makeSequence(arch[3]) # path4
        self.path5 = self.makeSequence(arch[4]) # path5
        self.parallel1 = self.makeSequence(arch[5]) # paralell1
        self.parallel2 = self.makeSequence(arch[6]) # paralell2
        self.path6 = self.makeSequence(arch[7]) # path6
        self.final = self.makeSequence(arch[8]) # final

    def makeSequence(self, arch):
        layers = []
        for id, name in lenumerate(arch):
            layers.append(self.weights[name])
        return nn.Sequnetial(*layers)

    def forward(self, 
        input, 
        mask, 
        epoch = None,
        train = False):
        
        #### --------------------
        # Here we should curriculemed the mask first of all
        if train:
            epoch epoch % self.extra_arg['T_period']
            alpha = self.extra_arg['eta_min'] + 0.5 * \
                    (self.extra_arg['eta_max'] - self.extra_arg['eta_min']) * \
                    (1 + math.cos(math.pi * epoch / self.extra_arg['T_period']))

            if epoch == (self.extra_arg['T_period'] - 1):
                self.extra_arg['T_period'] = 2 * self.extra_arg['T_period']

            mask['mask1'][mask['mask1'] == 1] = alpha
            mask['mask1'][mask['mask1'] == 0] = 1

            mask['mask2'][mask['mask2'] == 1] = alpha
            mask['mask2'][mask['mask2'] == 0] = 1

            mask['mask3'][mask['mask3'] == 1] = alpha
            mask['mask3'][mask['mask3'] == 0] = 1

            mask['mask4'][mask['mask4'] == 1] = alpha
            mask['mask4'][mask['mask4'] == 0] = 1

            mask['mask5'][mask['mask5'] == 1] = alpha
            mask['mask5'][mask['mask5'] == 0] = 1

        else:
            mask['mask1'][mask['mask1'] == 1] = 1
            mask['mask1'][mask['mask1'] == 0] = 1

            mask['mask2'][mask['mask2'] == 1] = 1
            mask['mask2'][mask['mask2'] == 0] = 1

            mask['mask3'][mask['mask3'] == 1] = 1
            mask['mask3'][mask['mask3'] == 0] = 1

            mask['mask4'][mask['mask4'] == 1] = 1
            mask['mask4'][mask['mask4'] == 0] = 1

            mask['mask5'][mask['mask5'] == 1] = 1
            mask['mask5'][mask['mask5'] == 0] = 1
        #### --------------------

        out = input
        out = self.path1(out)
        out = self.specialized_max_pooling(out, mask['mask1'])
        out = self.path2(out)
        out = self.specialized_max_pooling(out, mask['mask2'])
        out = self.path3(out)
        out = self.specialized_max_pooling(out, mask['mask3'])
        out = self.path4(out)
        out = self.specialized_max_pooling(out, mask['mask4'])
        out = self.path5(out)

        out1 = out.clone()
        
        out1 = self.specialized_max_pooling(out1, mask['mask5'])
        out1 = self.parallel1(out1)

        out2 = out.clone()
        out2 = self.parallel2(out2)

        out = torch.cat([out2, out1], dim=1)
        out = self.path6(out)

        # Regression Head
        out = self.final(out)

        return out

    def specialized_max_pooling(self, inp, amp):
        amp = amp.expand_as(inp) # bs * 1 * 30 * 30 -> bs * nc * 30 * 30 
        inp = inp * amp
        output, indices = F.max_pool2d(inp, 2, return_indices = True)
        amp_after_pooling = amp.view(-1)[indices.data.view(-1)].view_as(indices)
        output = output / amp_after_pooling
        return output