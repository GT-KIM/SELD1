#
# The SELDnet architecture
#
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Squeeze_Excitation(nn.Module) :
    def __init__(self, c, r=16) :
        super(Squeeze_Excitation, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(c, c // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c // r, c, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x) :
        bs, c, _, _ = x.shape
        y = self.squeeze(x).view(bs, c)
        y = self.excitation(y).view(bs, c, 1, 1)
        return x * y.expand_as(x)

class SE_ResLayer(nn.Module) :
    def __init__(self, in_channels, out_channels, downsample=1) :
        super(SE_ResLayer, self).__init__()
        self.downsample = downsample
        if self.downsample > 1 :
            self.AvgPool = nn.AvgPool2d((downsample, downsample), stride=(downsample, downsample), padding=(0,0))
        self.conv_res = nn.Conv2d(in_channels, out_channels, (1,1), (1,1), (0,0))
        self.bn_res = nn.BatchNorm2d(out_channels)

        self.conv1 = nn.Conv2d(in_channels, out_channels, (3,3), (1,1), (1,1))
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, (3,3), (1,1), (1,1))
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.SE = Squeeze_Excitation(out_channels, r=16)


    def forward(self, x) :
        if self.downsample > 1 :
            x = self.AvgPool(x)
        x_res = self.bn_res(self.conv_res(x))
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = self.SE(x)
        return F.relu(x + x_res, inplace=True)

class SE_ResBlock(nn.Module) :
    def __init__(self, in_channels, out_channels, L, downsample=1) :
        super(SE_ResBlock, self).__init__()
        self.downsample = downsample
        self.first_SE = SE_ResLayer(in_channels, out_channels, downsample)

        self.SE = nn.ModuleList()
        for _ in range(L-1) :
            self.SE.append(SE_ResLayer(out_channels, out_channels, downsample=1))

    def forward(self, x) :
        x = self.first_SE(x)
        for layer in self.SE :
            x = layer(x)
        return x


class Network(nn.Module) :
    def __init__(self, out_shape) :
        super(Network, self).__init__()
        self.first_conv = nn.Conv2d(7, 32, (3,3), (1,1), (1,1))
        self.first_bn = nn.BatchNorm2d(32)

        self.SE1 = SE_ResBlock(32, 32, 3, downsample=1)
        self.SE2 = SE_ResBlock(32, 64, 4, downsample=2)
        self.SE3 = SE_ResBlock(64, 128, 6, downsample=2)
        self.SE4 = SE_ResBlock(128, 256, 3, downsample=1)

        outmap_size = 32

        self.attention = nn.Sequential(
            nn.Conv1d(256 * outmap_size, 128, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 256 * outmap_size, kernel_size=1),
            nn.Softmax(dim=2),
            )

        self.in_gru_size = 256 * outmap_size
        self.gru = torch.nn.GRU(input_size=self.in_gru_size, hidden_size=128,
                                num_layers=2, batch_first=True,
                                dropout=0, bidirectional=True)
        self.fnn_list = torch.nn.ModuleList()
        for fc_cnt in range(1):
            self.fnn_list.append(
                torch.nn.Linear(128, 128, bias=True)
            )
        self.fnn_list.append(
            torch.nn.Linear(128, out_shape[-1], bias=True)
        )

    def forward(self, x) :
        x = self.first_bn(F.relu(self.first_conv(x)))
        x = self.SE4(self.SE3(self.SE2(self.SE1(x)))) # [B, 256, 20, 32]
        x = x.transpose(2, 3).contiguous() # [B, 256, 32, 20]
        x = x.view(x.shape[0], -1, x.shape[-1]).contiguous() # [B, 256 * 32, 20]
        w = self.attention(x)
        x = x * w
        x = x.transpose(1, 2).contiguous() # [B, 20, 256*32]

        x, _ = self.gru(x)
        x = torch.tanh(x)
        x = x[:, :, x.shape[-1] // 2:] * x[:, :, :x.shape[-1] // 2]

        for fnn_cnt in range(len(self.fnn_list)-1):
            x = self.fnn_list[fnn_cnt](x)
        doa = torch.tanh(self.fnn_list[-1](x))
        return doa