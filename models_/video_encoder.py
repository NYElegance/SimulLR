import torch
import torch.nn as nn
import torch.nn.functional as F


class VisualFront(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.drop3d = nn.Dropout3d(0.2)
        self.conv1 = nn.Conv3d(3, 16, (5, 3, 3), padding=(2, 1, 1), dilation=1)
        self.bn1 = nn.BatchNorm3d(16)
        self.conv1_activation = nn.PReLU(init=0)
        self.pool1 = nn.MaxPool3d((1, 3, 3), (1, 2, 2))

        self.conv2 = nn.Conv3d(16, 32, (3, 3, 3), padding=(1, 1, 1))
        self.bn2 = nn.BatchNorm3d(32)
        self.conv2_activation = nn.PReLU(init=0)
        self.pool2 = nn.MaxPool3d((1, 3, 3), (1, 2, 2))

        self.conv31 = nn.Conv3d(32, 64, (3, 3, 3), padding=(1, 1, 1))
        self.bn31 = nn.BatchNorm3d(64)
        self.conv31_activation = nn.PReLU(init=0)
        self.conv32 = nn.Conv3d(64, 64, (3, 3, 3), padding=(1, 1, 1))
        self.bn32 = nn.BatchNorm3d(64)
        self.conv32_activation = nn.PReLU(init=0)
        self.pool3 = nn.MaxPool3d((1, 3, 3), (1, 2, 2))

        self.conv4 = nn.Conv3d(64, 128, (1, 3, 3), padding=(0, 1, 1))
        self.bn4 = nn.BatchNorm3d(128)
        self.conv4_activation = nn.PReLU(init=0)
        self.pool4 = nn.MaxPool3d((1, 3, 3), (1, 2, 2))

        self.fc5 = nn.Conv3d(128, 256, (1, 2, 5))
        self.bn5 = nn.BatchNorm3d(256)
        self.fc5_activation = nn.PReLU(init=0)

        self.fc6 = nn.Conv3d(256, args.d_model, (1, 1, 1))
        self.fc6_activation = nn.PReLU(init=0)
        self.init_params()

    def init_params(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, x_len=None):
        # x:[bs, channels, d, h, w]
        if x_len is not None:
            C = self.args.chunk_size
            P = C - max(x_len) % C
            x = F.pad(x, pad=(0, 0, 0, 0, 0, P), value=0)
            b, c, s, h, w = x.size()
            x = x.transpose(1, 2)
            x = x.contiguous().view(s // C * b, C, c, h, w)
            x = x.transpose(1, 2)

        x = self.conv1_activation(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.conv2_activation(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = self.conv31_activation(self.bn31(self.conv31(x)))
        x = self.conv32_activation(self.bn32(self.conv32(x))) + x
        x = self.pool3(x)
        x = self.conv4_activation(self.bn4(self.conv4(x)))
        x = self.pool4(x)
        x = self.fc5_activation(self.bn5(self.fc5(x)))
        x = self.fc6_activation(self.fc6(x))
        x = x.squeeze(-1).squeeze(-1).transpose(1, 2)
        if x_len is not None:
            x = x.contiguous().view(b, s, -1)[:, :max(x_len)]
        return x