import torch
import torch.nn as nn

class space_to_depth(nn.Module):
    def __init__(self, block_size=1):
        super(space_to_depth, self).__init__()
        self.block_size = block_size        
    def forward(self, input):
        x = input.permute(0,2,3,1)
        batch, height, width, depth = x.size()
        reduced_height = height // self.block_size
        reduced_width = width // self.block_size
        x = x.contiguous().view(batch, reduced_height, self.block_size,
                             reduced_width, self.block_size, depth)
        x = x.permute(0,1,3,2,4,5).contiguous().view(batch, 
            reduced_height, reduced_width, -1)
        x = x.permute(0,3,1,2)
        return x


class Darknet19:
    """This is the model to create the pretrained darknet19"""
    def __init__(self):
        super(Darknet19, self).__init__()
        self.lid = {}
        self.lod = {}
        self.lin = {}
        self.arch = self.makeArch()

    def makeArch(self):
        path1 = ["conv2d_1", 
                "batch_normalization_1",
                "leaky_re_lu_1"]
        path2 = ["conv2d_2", 
                "batch_normalization_2",
                "leaky_re_lu_2"]
        path3 = ["conv2d_3", 
                "batch_normalization_3",
                "leaky_re_lu_3",
                "conv2d_4", 
                "batch_normalization_4",
                "leaky_re_lu_4",
                "conv2d_5", 
                "batch_normalization_5",
                "leaky_re_lu_5"]
        path4 = ["conv2d_6", 
                "batch_normalization_6",
                "leaky_re_lu_6",
                "conv2d_7", 
                "batch_normalization_7",
                "leaky_re_lu_7",
                "conv2d_8", 
                "batch_normalization_8",
                "leaky_re_lu_8"]
        path5 = ["conv2d_9", 
                "batch_normalization_9",
                "leaky_re_lu_9",
                "conv2d_10", 
                "batch_normalization_10",
                "leaky_re_lu_10",
                "conv2d_11", 
                "batch_normalization_11",
                "leaky_re_lu_11",
                "conv2d_12", 
                "batch_normalization_12",
                "leaky_re_lu_12",
                "conv2d_13", 
                "batch_normalization_13",
                "leaky_re_lu_13"
            ]
        paralle1 = ["conv2d_14", 
                "batch_normalization_14",
                "leaky_re_lu_14",
                "conv2d_15", 
                "batch_normalization_15",
                "leaky_re_lu_15",
                "conv2d_16", 
                "batch_normalization_16",
                "leaky_re_lu_16",
                "conv2d_17", 
                "batch_normalization_17",
                "leaky_re_lu_17",
                "conv2d_18", 
                "batch_normalization_18",
                "leaky_re_lu_18",
                "conv2d_19", 
                "batch_normalization_19",
                "leaky_re_lu_19",
                "conv2d_20", 
                "batch_normalization_20",
                "leaky_re_lu_20"
            ]
        paralle2 = ["conv2d_21", 
                "batch_normalization_21",
                "leaky_re_lu_21",
                "space_to_depth_x2"
            ]
        path6 = ["conv2d_22",
                "batch_normalization_22",
                "leaky_re_lu_22"]
        final = ["conv2d_23"]
        
        return path1, path2, path3, path4, path5 paralle1, paralle2, path6, final