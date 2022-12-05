#
# The SELDnet architecture
#
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from proposed_model.Conformer import *

class SpatialAttention2d(nn.Module) :
    def __init__(self, c, t, f, r=16) :
        super(SpatialAttention2d, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation1 = nn.Sequential(
            nn.Linear(c, c // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c // r, c, bias=False),
            nn.Sigmoid()
        )
        self.excitation2 = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=(r+1,r+1), stride=(1,1), padding=(r//2, r//2)),
            nn.BatchNorm2d(c),
            nn.Conv2d(c, 1, kernel_size=(1,1), stride=(1,1), padding=(0,0)),
            nn.Sigmoid()
        )

    def forward(self, x) :
        bs, c, t, f = x.shape

        y1 = self.squeeze(x).squeeze(-1).squeeze(-1)
        y1 = self.excitation1(y1).reshape(bs, c, 1, 1)

        y2 = self.excitation2(x)

        return x * y1.expand_as(x) + x * y2.expand_as(x)

class SE_ResLayer2d(nn.Module) :
    def __init__(self, in_channels, out_channels, t, f, downsample_t =1, downsample_f=1) :
        super(SE_ResLayer2d, self).__init__()
        self.downsample_t = downsample_t
        self.downsample_f = downsample_f
        if self.downsample_t > 1 or self.downsample_f > 1 :
            self.AvgPool = nn.AvgPool2d((downsample_t, downsample_f), stride=(downsample_t, downsample_f),
                                        padding=((downsample_t+1) // 2 - 1, (downsample_f+1) // 2 - 1))
        self.conv_res = nn.Conv2d(in_channels, out_channels, (1,1), (1,1), (0,0))
        self.bn_res = nn.BatchNorm2d(out_channels)

        self.conv1 = nn.Conv2d(in_channels, out_channels*2, (7,7), (1,1), (3,3))
        self.bn1 = nn.BatchNorm2d(out_channels*2)
        self.conv2 = nn.Sequential(nn.Conv2d(out_channels*2, out_channels*2, (7,7), (1,1), (3,3)), nn.GELU())
        self.pool2 = nn.AdaptiveAvgPool2d(1)
        self.fc2 = nn.Linear(out_channels*2, out_channels*2)
        self.bn2 = nn.BatchNorm2d(out_channels*2)
        self.conv3 = nn.Conv2d(out_channels*2, out_channels, (7,7), (1,1), (3,3))
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.SE = SpatialAttention2d(out_channels, t=t//downsample_t, f=f//downsample_f, r=16)

    def forward(self, x) :
        if self.downsample_t > 1 or self.downsample_f > 1 :
            x = self.AvgPool(x)
        x_res = self.bn_res(self.conv_res(x))
        x = self.bn1(F.relu(self.conv1(x)))
        b, c, t, f = x.shape
        x1 = self.conv2(x)
        x2 = self.fc2(self.pool2(x).squeeze(-1).squeeze(-1)).reshape(b, c, 1, 1).expand_as(x)
        x = self.bn2(x + torch.tanh(x2) * x1)
        x = self.bn3(self.conv3(x))
        x = self.SE(x)
        return F.relu(x + x_res, inplace=True)

class SE_ResBlock2d(nn.Module) :
    def __init__(self, in_channels, out_channels, L, t, f, downsample_fs=1, downsample_t =1, downsample_f = 1 ) :
        super(SE_ResBlock2d, self).__init__()
        self.first_SE = SE_ResLayer2d(in_channels, out_channels, t, f, downsample_t, downsample_f)

        self.SE = nn.ModuleList()
        for _ in range(L-1) :
            self.SE.append(SE_ResLayer2d(out_channels, out_channels, t // downsample_t, f // downsample_f,
                                         downsample_t=1, downsample_f=1))

    def forward(self, x) :
        x = self.first_SE(x)
        for layer in self.SE :
            x = layer(x)
        return x


class Networks(nn.Module) :
    def __init__(self, out_shape) :
        super(Networks, self).__init__()
        self.first_conv = nn.Conv2d(7, 32, (3, 3), (1, 1), (1, 1))
        self.first_bn = nn.BatchNorm2d(32)

        self.SE1 = SE_ResBlock2d(32, 64, 3, t=250, f=128,downsample_t=1, downsample_f=2)
        self.SE2 = SE_ResBlock2d(64, 64, 4, t=250, f=64, downsample_t=5, downsample_f=2)
        self.SE3 = SE_ResBlock2d(64, 128, 6, t=250, f=32, downsample_t=1, downsample_f=2)
        self.SE4 = SE_ResBlock2d(128, 256, 3, t=50, f=16,  downsample_t=1, downsample_f=2)

        outmap_size = 8 # last fs * last f

        self.attention = nn.Sequential(
            nn.Conv1d(256 * outmap_size, 128, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 256 * outmap_size, kernel_size=1),
            nn.Softmax(dim=2),
            )
        """
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
        self.in_gru_size = 256 * outmap_size
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
        x = self.first_bn(F.relu(self.first_conv(x)))
        x = self.SE4(self.SE3(self.SE2(self.SE1(x)))) # [B, 256, 20, 16]
        x = x.transpose(2, 3).contiguous() # [B, 256, 16, 20]
        x = x.view(x.shape[0], -1, x.shape[-1]).contiguous() # [B, 256 * 32, 20]
        w = self.attention(x)
        x = x * w
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