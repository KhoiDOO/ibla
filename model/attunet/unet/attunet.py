from lzma import FILTER_X86
import torch
from torch import nn 

from .attunet_core import *

class attunet(nn.Module):
    """https://arxiv.org/abs/1804.03999"""
    def __init__(self, args):
        super(attunet, self).__init__()

        self.seg_n_classes = args.seg_n_classes
        self.init_filter = args.init_filter
        self.t = args.t 

        self.Maxpool = nn.MaxPool2d(kernel_size = 2, stride = 2)

        self.encoder = nn.ModuleList([            
            conv_block(3, self.init_filter),
            conv_block(self.init_filter, self.init_filter*2),
            conv_block(self.init_filter*2, self.init_filter*4),
            conv_block(self.init_filter*4, self.init_filter*8),
            conv_block(self.init_filter*8, self.init_filter*16)
        ])

        self.decoder = nn.ModuleList([
            up_conv(self.init_filter*16, self.init_filter*8),
            conv_block(self.init_filter*16, self.init_filter*8),

            up_conv(self.init_filter*8, self.init_filter*4),
            conv_block(self.init_filter*8, self.init_filter*4),

            up_conv(self.init_filter*4, self.init_filter*2),
            conv_block(self.init_filter*4, self.init_filter*2),

            up_conv(self.init_filter*2, self.init_filter),
            conv_block(self.init_filter*2, self.init_filter),            
        ])

        self.Att1 = Attention_block(F_g=self.init_filter*8, F_l=self.init_filter*8, F_int=self.init_filter*4)
        self.Att2 = Attention_block(F_g=self.init_filter*4, F_l=self.init_filter*4, F_int=self.init_filter*2)
        self.Att3 = Attention_block(F_g=self.init_filter*2, F_l=self.init_filter*2, F_int=self.init_filter)
        self.Att4 = Attention_block(F_g=self.init_filter, F_l=self.init_filter, F_int=32)

        self.Conv = nn.Conv2d(self.init_filter, self.seg_n_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):

        e1 = self.encoder[0]

        e2 = self.Maxpool(e1)
        e2 = self.encoder[1]

        e3 = self.Maxpool(e2)
        e3 = self.encoder[2]

        e4 = self.Maxpool(e3)
        e4 = self.encoder[3]

        e5 = self.Maxpool(e4)
        e5 = self.encoder[4]

        d5 = self.decoder[0]        
        x4 = self.Att1(g=d5, x=e4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.decoder[1]

        d4 = self.decoder[2]
        x3 = self.Att2(g=d4, x=e3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.decoder[3]

        d3 = self.decoder[4]
        x2 = self.Att3(g=d3, x=e2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.decoder[5]

        d2 = self.decoder[6]
        x1 = self.Att3(g=d2, x=e1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.decoder[7]

        out = self.Conv(d2)

        return out