import numpy as np
from torch import nn


from proposed_model.ConvNeXt import *
from proposed_model.Conformer import *

class ProposedNetwork(nn.Module) :
    def __init__(self, in_feat_shape, out_shape, params) :
        super(ProposedNetwork, self).__init__()
        self.nb_classes = params['unique_classes']

        self.convNeXt = ConvNeXt(in_chans=7, depths=[1, 2, 3, 4], dims = [32, 64, 128, 256], drop_path_rate=0.1, layer_scale_init_value=1e-6)
        self.feature_size = 256 * int(np.floor(in_feat_shape[-1] / 2 ** 3) / 4)
        self.conformer = nn.ModuleList([ConformerBlock(
            encoder_dim=self.feature_size,
            num_attention_heads=16,
            feed_forward_expansion_factor=4,
            conv_expansion_factor=2,
            feed_forward_dropout_p=0.1,
            attention_dropout_p=0.1,
            conv_dropout_p=0.1,
            conv_kernel_size=5,
            half_step_residual=True,
        ) for _ in range(2
                         )])
        """
        self.gru = torch.nn.GRU(input_size=self.feature_size, hidden_size=1024,
                                num_layers=2, batch_first=True,
                                dropout=0.1, bidirectional=True)
        """
        self.time_pooling = nn.Conv1d(1024, 1024, 21, 10, 10)

        self.fnn_list = torch.nn.ModuleList()
        for fc_cnt in range(2):
            self.fnn_list.append(
                torch.nn.Linear(self.feature_size, self.feature_size, bias=True)
            )
            self.fnn_list.append(
                nn.Tanh()
            )

        self.fnn_list.append(
            torch.nn.Linear(self.feature_size, out_shape[-1], bias=True)
        )
    def forward(self, x):
        '''input: (batch_size, mic_channels, time_steps, mel_bins)'''
        x = self.convNeXt(x)

        x = x.transpose(1, 2).contiguous()
        x = x.view(x.shape[0], x.shape[1], -1).contiguous()
        ''' (batch_size, time_steps, feature_maps):'''

        for layer in self.conformer:
            x = layer(x)

        """
        (x, _) = self.gru(x)
        x = torch.tanh(x)
        x = x[:, :, x.shape[-1]//2:] * x[:, :, :x.shape[-1]//2]
        """

        x = x.permute(0, 2, 1)
        x = self.time_pooling(x)
        x = x.permute(0, 2, 1)

        for fnn_cnt in range(len(self.fnn_list)-1):
            x = self.fnn_list[fnn_cnt](x)
        doa = torch.tanh(self.fnn_list[-1](x))
        '''(batch_size, time_steps, label_dim)'''

        return doa
