#!user/bin/env python3

# Component modules of the CNNBTR model.

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """
    Made some minor changes to the ResNet18 module
    """
    def __init__(self, in_channels, out_channels, kernel_size=2, stride=1):
        super(ResBlock, self).__init__()
        # Two consecutive convolutional layers are defined here
        self.left = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding='same', bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding='same', bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            # Residual connection
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        self.pool = nn.AvgPool2d(2,2)

    def forward(self, x):
        out = self.left(x)
        out = out + self.shortcut(x)
        out = F.relu(out)
        out = self.pool(out)

        return out


class CNNBTR_dream(nn.Module):
    def __init__(self, drop_out=0.2):
        """
        CNNBTR_dream model using only gene expression data
        """
        super().__init__()

        self.layer_1 = nn.Sequential(
            ResBlock(3, 32, 3)
        )

        self.layer_2= nn.Sequential(
            ResBlock(32, 64, 3)
        )

        self.layer_3 = nn.Sequential(
            ResBlock(64, 128, 3)
        )

        self.layer_4 = nn.Sequential(
            ResBlock(128, 256, 3)
        )

        self.fc = nn.Sequential(
            nn.Linear(in_features=1024, out_features=128),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_out),

            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_out),

            nn.Linear(in_features=64, out_features=1)
        )
        self.out = nn.Sigmoid()

    def forward(self, x):

        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = torch.flatten(x,1)
        
        x = self.fc(x)
        x = self.out(x)
      
        return x


class CNNBTR(nn.Module):
    def __init__(self, drop_out=0.2):
        """
        CNNBTR model using gene expression data and genomic distance
        """
        super().__init__()        
     
        self.layer_1 = nn.Sequential(
            ResBlock(3, 32, 3)
        )

        self.layer_2= nn.Sequential(
            ResBlock(32, 64, 3)
        )

        self.layer_3 = nn.Sequential(
            ResBlock(64, 128, 3)
        )

        self.layer_4 = nn.Sequential(
            ResBlock(128, 256, 3)
        )
        # Additional connectivity layers to integrate genomic data
        self.dense = nn.Sequential(
            nn.Linear(in_features=1, out_features=32),
            nn.ReLU(inplace=True)
        )

        self.fc = nn.Sequential(
            nn.Linear(in_features=1056, out_features=128),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_out),

            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_out),

            nn.Linear(in_features=64, out_features=1)
        )

        self.out = nn.Sigmoid()

    def forward(self, x1, x2):

        x1 = self.layer_1(x1)
        x1 = self.layer_2(x1)
        x1 = self.layer_3(x1)
        x1 = self.layer_4(x1)        
        x1 = torch.flatten(x1, 1)
        x = torch.cat((x1, self.dense(x2)), dim=1)      
        x = self.fc(x)
        x = self.out(x)
        
        return x
