import torch
from torch import nn 

from .runet_core import *

class runet(nn.Module):
    """https://arxiv.org/abs/1802.06955"""
    def __init__(self, args):
        super(runet, self).__init__()

        self.seg_n_classes = args.seg_n_classes
        self.init_filter = args.init_filter
        self.t = args.t 

        self.Maxpool = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.Upsample = nn.Upsample(scale_factor = 2)
  
        self.encoder = nn.ModuleList([            
            RRCNN_block(3, self.init_filter, t = self.t),
            RRCNN_block(self.init_filter, self.init_filter * 2, t = self.t),
            RRCNN_block(self.init_filter * 2, self.init_filter * 4, t = self.t),
            RRCNN_block(self.init_filter * 4 , self.init_filter * 8, t = self.t),
            RRCNN_block(self.init_filter * 8 , self.init_filter * 16, t = self.t)
        ])

        self.decoder = nn.ModuleList([
            up_conv(self.init_filter*16, self.init_filter*8),
            RRCNN_block(self.init_filter *16 , self.init_filter * 8, t = self.t),

            up_conv(self.init_filter*8, self.init_filter*4),
            RRCNN_block(self.init_filter *8 , self.init_filter * 4, t = self.t),

            up_conv(self.init_filter*4, self.init_filter*2),
            RRCNN_block(self.init_filter *4 , self.init_filter * 2, t = self.t),
                
            up_conv(self.init_filter*2, self.init_filter),
            RRCNN_block(self.init_filter *2 , self.init_filter, t = self.t),
        ])

        self.Conv = nn.Conv2d(self.init_filter, self.seg_n_classes, kernel_size=1, stride=1, padding=0)
    
    def forward(self, x):
        x1 = self.encoder[0](x)

        x2 = self.Maxpool(x1)
        x2 = self.encoder[1](x2)

        x3 = self.Maxpool(x2)
        x3 = self.encoder[2](x3)
        
        x4 = self.Maxpool(x3)
        x4 = self.encoder[3](x4)

        x5 = self.Maxpool(x4)
        x5 = self.encoder[4](x5)
        
        d5 = self.decoder[0](x5)
        d5 = torch.cat((x4, d5), dim = 1)
        d5 = self.decoder[1](d5)

        d4 = self.decoder[2](d5)
        d4 = torch.cat((x3, d4), dim = 1)
        d4 = self.decoder[3](d4)   

        d3 = self.decoder[4](d4)
        d3 = torch.cat((x2, d3), dim = 1)
        d3 = self.decoder[5](d3)   

        d2 = self.decoder[6](d3)
        d2 = torch.cat((x1, d2), dim = 1)
        d2 = self.decoder[7](d2)  

        out = self.Conv(d2)   

        return out