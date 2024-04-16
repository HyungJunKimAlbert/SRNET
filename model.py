import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.utils import save_image

# Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network
# https://arxiv.org/abs/1609.04802

class SRNet(nn.Module):
    def __init__(self, output_size=1, num_filters=128):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=num_filters, kernel_size=9, padding=4)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(num_filters, num_filters//2, kernel_size=1, padding=0, padding_mode='replicate')
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(num_filters//2, output_size, kernel_size=5, padding=2, padding_mode='replicate')

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)

        return x
    


# ResNet + SRCNN
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out += residual  
        out = self.relu(out)

        return out

class SRCNN(nn.Module):
    def __init__(self, output_size=1, num_filters=64):
        super(SRCNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=num_filters, kernel_size=9, padding=4)
        self.relu1 = nn.ReLU()

        self.resblocks = nn.Sequential(
            ResidualBlock(num_filters, num_filters),
            ResidualBlock(num_filters, num_filters),
            ResidualBlock(num_filters, num_filters),
            ResidualBlock(num_filters, num_filters),
        )

        self.conv3 = nn.Conv2d(num_filters, output_size, kernel_size=5, padding=2, padding_mode='replicate')

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.resblocks(x)
        x = self.conv3(x)

        return x



# 02. ResUNET
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        output = self.up(x)
        return output

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    def forward(self, x):
        return self.maxpool_conv(x)
    
class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels//2, in_channels//2, kernel_size=2, stride=2)
        
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX//2, diffX - diffX//2,
                        diffY//2, diffY-diffY//2
                        ])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class ConvUNet(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(ConvUNet, self).__init__()

        self.inc = DoubleConv(in_channels, out_channels=64*2)
        self.down1 = Down(64*2, 128*2)
        self.down2 = Down(128*2, 256*2)
        self.down3 = Down(256*2, 512*2)
        self.down4 = Down(512*2, 512*2)

        self.up1 = Up(1024*2, 256*2, bilinear)
        self.up2 = Up(512*2, 128*2, bilinear)
        self.up3 = Up(256*2, 64*2, bilinear)
        self.up4 = Up(128*2, 64*2, bilinear)
        self.out = nn.Conv2d(64*2, out_channels, kernel_size=1)
        

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        # Dec        
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        logits = self.out(x)
        return logits







