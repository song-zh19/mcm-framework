import torch
import torch.nn as nn
from clinical_ts.mac import MACReasonLayer
from .basic_conv1d import listify, bn_drop_lin
from .cpc import AdaptiveConcatPoolRNN



class BaseBlock(nn.Module):
    def __init__(self, in_planes):
        super(BaseBlock, self).__init__()

        self.bottleneck = nn.Conv1d(in_planes, 128, kernel_size=1, stride=1, bias=False)
        self.conv4 = nn.Conv1d(128, 128, kernel_size=39, stride=1, padding=19, bias=False)
        self.conv3 = nn.Conv1d(128, 128, kernel_size=19, stride=1, padding=9, bias=False)
        self.conv2 = nn.Conv1d(128, 128, kernel_size=9, stride=1, padding=4, bias=False)

        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=False)
        self.conv1 = nn.Conv1d(in_planes, 128, kernel_size=1, stride=1, bias=False)

        self.bn = nn.BatchNorm1d(128 * 4, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        output = self.bottleneck(x)
        output4 = self.conv4(output)
        output3 = self.conv3(output)
        output2 = self.conv2(output)

        output1 = self.maxpool(x)
        output1 = self.conv1(output1)

        x_out = self.relu(self.bn(torch.cat((output1, output2, output3, output4), dim=1)))
        return x_out


class InceptionTime(nn.Module):
    def __init__(self, in_channel=12, num_classes=10, multi_modal_dim=None, n_hidden=512, lin_ftrs_head=[512], ps_head=0.5):
        super(InceptionTime, self).__init__()

        self.BaseBlock1 = BaseBlock(in_channel)
        self.BaseBlock2 = BaseBlock(n_hidden)
        self.BaseBlock3 = BaseBlock(n_hidden)

        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv1d(in_channel, n_hidden, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm1d(n_hidden)

        self.BaseBlock4 = BaseBlock(n_hidden)
        self.BaseBlock5 = BaseBlock(n_hidden)
        self.BaseBlock6 = BaseBlock(n_hidden)

        self.conv2 = nn.Conv1d(n_hidden, n_hidden, kernel_size=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm1d(n_hidden)

        # self.Avgpool = nn.AdaptiveAvgPool1d(1)
        # self.fc = nn.Linear(n_hidden * 5, num_classes)
        self.pool = AdaptiveConcatPoolRNN()

        if multi_modal_dim is not None:
            multi_modal_proj_dim = 2 * n_hidden
            self.linear_proj = nn.Linear(multi_modal_dim, multi_modal_proj_dim)
            self.modal_proj = nn.Transformer(multi_modal_proj_dim, nhead=8, num_encoder_layers=1, num_decoder_layers=1)

            self.fusion = MACReasonLayer(dim_v=n_hidden, dim_q=multi_modal_proj_dim, max_step=6, self_attention=False, memory_gate=False)
        else:
            self.modal_proj = None


        layers_head =[]
        nf = 3*n_hidden
        nf = nf + 3*multi_modal_proj_dim + 2*n_hidden if multi_modal_dim is not None else nf
        lin_ftrs_head = [nf, num_classes] if lin_ftrs_head is None else [nf] + lin_ftrs_head + [num_classes]
        ps_head = listify(ps_head)
        if len(ps_head)==1:
            ps_head = [ps_head[0]/2] * (len(lin_ftrs_head)-2) + ps_head
        actns = [nn.ReLU(inplace=True)] * (len(lin_ftrs_head)-2) + [None]

        for ni,no,p,actn in zip(lin_ftrs_head[:-1],lin_ftrs_head[1:],ps_head,actns):
            layers_head+=bn_drop_lin(ni,no,True,p,actn)
        self.head=nn.Sequential(*layers_head)

    def forward(self, x):
        x, modaldata, _ = x
        shortcut1 = self.bn1(self.conv1(x))

        output1 = self.BaseBlock1(x)
        output1 = self.BaseBlock2(output1)
        output1 = self.BaseBlock3(output1)
        output1 = self.relu(output1 + shortcut1)
        # print(output1.shape)
        shortcut2 = self.bn2(self.conv2(output1))
        # print(shortcut2.shape)
        output2 = self.BaseBlock4(output1)
        output2 = self.BaseBlock5(output2)
        output2 = self.BaseBlock6(output2)
        output2 = self.relu(output2 + shortcut2)
        # # print(output2.shape)
        # output = self.Avgpool(output2)
        # # print(output.shape)
        # output = output.view(output.size(0), -1)
        output = self.pool(output2)
        # print(output.shape)
        if self.modal_proj is not None:
            modaldata = self.linear_proj(modaldata)
            attention_mask = (modaldata.sum(dim=-1) != 0).float()
            modaldata = modaldata.permute(1, 0, 2)
            modal_output_all = self.modal_proj(modaldata, modaldata, src_key_padding_mask=~attention_mask.bool())
            modal_output_all = modal_output_all.permute(1, 2, 0)
            # modal_output = self.Avgpool(modal_output_all)
            # modal_output = modal_output.view(modal_output.size(0), -1)
            modal_output = self.pool(modal_output_all)
            fusion_output = self.fusion(output2.permute(0, 2, 1), modal_output_all.mean(2), modal_output_all.permute(0, 2, 1))
            output = torch.cat([output, modal_output, fusion_output], axis=1)

        # output = self.fc(output)
        # print(output.shape)
        # exit(-1)
        output = self.head(output)
        return output


def inceptiontime(**kwargs):
    return InceptionTime(**kwargs)


