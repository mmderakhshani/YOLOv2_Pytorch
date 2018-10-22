from .model import Darknet19
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import pickle as pickle
from collections import OrderedDict

class YoloV2(nn.Module):
    """Yolo version 2; It is an extented version of 
    yolo v1 capable of detecting 9000 object"""
    def __init__(self, modelUrl):
        super(YoloV2, self).__init__()
        self.modelUrl = modelUrl

        self.darknet19 = Darknet19()
        with open(modelUrl, 'rb') as fp:
            self.weights = pickle.load(fp)
        fp.close()
        arch = self.darknet19.arch

        self.path1 = self.makeSequence(arch[0]) # path1z
        self.parallel1 = self.makeSequence(arch[1]) # paralell1
        self.parallel2 = self.makeSequence(arch[2]) # paralell2
        self.path2 = self.makeSequence(arch[3]) # path2
        self.final = self.makeSequence(arch[4]) # final

    def makeSequence(self, arch):
        layers = []
        for id, name in enumerate(arch):
            layers.append(self.weights[name])
        return nn.ModuleList(layers)

    def forward(self, input):
        
        out = input
        for layer in self.path1:
            out = layer(out)


        out1 = out.clone()
        for layer in self.parallel1:
            out1 = layer(out1)
        out2 = out.clone()
        for layer in self.parallel2:
            out2 = layer(out2)
        out = torch.cat([out2, out1], dim=1)
        for layer in self.path2:
            out = layer(out)
        # Regression Head
        finalOut = out.clone()
        for layer in self.final:
            finalOut = layer(finalOut)

        return finalOut # 1 * 425 * 15 * 15