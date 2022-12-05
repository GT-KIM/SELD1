#
# The SELDnet architecture
#
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from proposed_model.Conformer import *


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

class SpatialAttention3d(nn.Module) :
    def __init__(self, c, fs, t, f, r=16) :
        super(SpatialAttention3d, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation1 = nn.Sequential(
            nn.Linear(c*fs, (c*fs) // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear((c*fs) // r, c*fs, bias=False),
            nn.Sigmoid()
        )
        self.excitation2 = nn.Sequential(
            nn.Linear(t*c, (t*c) // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear((t*c) // r, t*c, bias=False),
            nn.Sigmoid()
        )
        self.excitation3 = nn.Sequential(
            nn.Linear(f*c, (f*c) // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear((f*c) // r, f*c, bias=False),
            nn.Sigmoid()
        )


    def forward(self, x) :
        bs, c, fs, t, f = x.shape
        x1 = x.reshape(bs, c*fs, t, f)
        y1 = self.squeeze(x1).reshape(bs, c*fs)
        y1 = self.excitation1(y1).reshape(bs, c, fs, 1, 1)

        x2 = x.permute(0, 1, 3, 2, 4).reshape(bs, c * t, fs, f)
        y2 = self.squeeze(x2).reshape(bs, c * t)
        y2 = self.excitation2(y2).reshape(bs, c, t, 1, 1).permute(0, 1, 3, 2, 4)

        x3 = x.permute(0, 1, 4, 3, 2).reshape(bs, c * f, fs, t)
        y3 = self.squeeze(x3).reshape(bs, c * f)
        y3 = self.excitation3(y3).reshape(bs, c, f, 1, 1).permute(0, 1, 4, 3, 2)

        return x * y1.expand_as(x) + x * y2.expand_as(x) + x * y3.expand_as(x)

class SE_ResLayer3d(nn.Module) :
    def __init__(self, in_channels, out_channels, fs, t, f, downsample_fs=1, downsample_t =1, downsample_f=1) :
        super(SE_ResLayer3d, self).__init__()
        self.downsample_fs = downsample_fs
        self.downsample_t = downsample_t
        self.downsample_f = downsample_f
        if self.downsample_fs > 1 or self.downsample_t > 1 or self.downsample_f > 1 :
            self.AvgPool = nn.AvgPool3d((downsample_fs, downsample_t, downsample_f), stride=(downsample_fs, downsample_t, downsample_f),
                                        padding=((downsample_fs+1) // 2 - 1, (downsample_t+1) // 2 - 1, (downsample_f+1) // 2 - 1))
        self.conv_res = nn.Conv3d(in_channels, out_channels, (1,1,1), (1,1,1), (0,0,0))
        self.bn_res = nn.BatchNorm3d(out_channels)

        self.conv1 = nn.Conv3d(in_channels, out_channels, (1,7,7), (1,1,1), (0,3,3))
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, (fs//downsample_fs,1,1), (1,1,1), (0,0,0))
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.conv3 = nn.Conv3d(out_channels, out_channels, (1,7,7), (1,1,1), (0,3,3))
        self.bn3 = nn.BatchNorm3d(out_channels)
        self.SE = SpatialAttention3d(out_channels, fs=fs //downsample_fs, t=t//downsample_t, f=f//downsample_f, r=16)

    def forward(self, x) :
        if self.downsample_fs > 1 or self.downsample_t > 1 or self.downsample_f > 1 :
            x = self.AvgPool(x)
        x_res = self.bn_res(self.conv_res(x))
        x = self.bn1(F.relu(self.conv1(x)))
        x_res2 = x
        x = torch.sigmoid(self.conv2(x))
        x = self.bn2(x.expand_as(x) * x_res2 + x_res2)
        x = self.bn3(self.conv3(x))
        x = self.SE(x)
        return F.relu(x + x_res, inplace=True)

class SE_ResBlock3d(nn.Module) :
    def __init__(self, in_channels, out_channels, L, fs, t, f, downsample_fs=1, downsample_t =1, downsample_f = 1 ) :
        super(SE_ResBlock3d, self).__init__()
        self.first_SE = SE_ResLayer3d(in_channels, out_channels, fs, t, f, downsample_fs, downsample_t, downsample_f)

        self.SE = nn.ModuleList()
        for _ in range(L-1) :
            self.SE.append(SE_ResLayer3d(out_channels, out_channels, fs // downsample_fs, t // downsample_t, f // downsample_f,
                                         downsample_fs=1, downsample_t=1, downsample_f=1))

    def forward(self, x) :
        x = self.first_SE(x)
        for layer in self.SE :
            x = layer(x)
        return x


class Networks(nn.Module) :
    def __init__(self, out_shape) :
        super(Networks, self).__init__()
        self.first_conv = nn.Conv3d(1, 32, (1, 3, 3), (1, 1, 1), (0, 1, 1))
        self.first_bn = nn.BatchNorm3d(32)

        self.SE1 = SE_ResBlock3d(32, 32, 3, fs=7, t=250, f=128, downsample_fs=2, downsample_t=1, downsample_f=2)
        self.SE2 = SE_ResBlock3d(32, 32, 4, fs=3, t=250, f=64, downsample_fs=2, downsample_t=1, downsample_f=2)
        self.SE3 = SE_ResBlock3d(32, 128, 6, fs=1, t=250, f=32, downsample_fs=1, downsample_t=5, downsample_f=2)
        self.SE4 = SE_ResBlock3d(128, 128, 3, fs=1, t=50, f=16, downsample_fs=1, downsample_t=1, downsample_f=2)

        outmap_size = 8 # last fs * last f

        """
        self.attention = nn.Sequential(
            nn.Conv1d(256 * outmap_size, 128, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 256 * outmap_size, kernel_size=1),
            nn.Softmax(dim=2),
            )

        num_attention_head = 1
        for i in range(1, 16) :
            if (128 * outmap_size) % i == 0 :
                num_attention_head = i

        self.conformer = nn.ModuleList([SelfAttentionBlock(
            encoder_dim=128 * outmap_size,
            num_attention_heads=num_attention_head,
            feed_forward_expansion_factor=4,
            conv_expansion_factor=2,
            feed_forward_dropout_p=0.05,
            attention_dropout_p=0.05,
            conv_dropout_p=0.05,
            conv_kernel_size=1,
            half_step_residual=True,
        ) for _ in range(2)])
        """
        self.in_gru_size = 128 * outmap_size
        self.gru = torch.nn.GRU(input_size=self.in_gru_size, hidden_size=self.in_gru_size,
                                num_layers=2, batch_first=True,
                                dropout=0, bidirectional=True)
        self.fnn_list = torch.nn.ModuleList()
        for fc_cnt in range(1):
            self.fnn_list.append(
                torch.nn.Linear(self.in_gru_size, 128, bias=True)
            )
        self.fnn_list.append(
            torch.nn.Linear(128, out_shape[-1], bias=True)
        )

    def forward(self, x) :
        x = x.unsqueeze(1)
        x = self.first_bn(F.relu(self.first_conv(x)))
        x = self.SE4(self.SE3(self.SE2(self.SE1(x)))).squeeze(2) # [B, 256, 20, 16]
        x = x.transpose(2, 3).contiguous() # [B, 256, 16, 20]
        x = x.view(x.shape[0], -1, x.shape[-1]).contiguous() # [B, 256 * 32, 20]
        #w = self.attention(x)
        #x = x * w
        x = x.transpose(1, 2).contiguous() # [B, 20, 256*32]

        #for layer in self.conformer :
        #    x = layer(x)

        x, _ = self.gru(x)
        x = torch.tanh(x)
        x = x[:, :, x.shape[-1] // 2:] * x[:, :, :x.shape[-1] // 2]

        #x = x.permute(0, 2, 1)
        #x = self.time_pooling(x)
        #x = x.permute(0, 2, 1)

        for fnn_cnt in range(len(self.fnn_list)-1):
            x = self.fnn_list[fnn_cnt](x)
        doa = torch.tanh(self.fnn_list[-1](x))
        return doa