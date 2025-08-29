import torch.nn as nn
import torch
import torch.nn.functional as F


class CAM(nn.Module):
    def __init__(self, channels, r):
        super(CAM, self).__init__()
        self.channels = channels
        self.r = r
        self.linear = nn.Sequential(
            nn.Linear(
                in_features=self.channels,
                out_features=self.channels // self.r,
                bias=True,
            ),
            nn.ReLU(inplace=True),
            nn.Linear(
                in_features=self.channels // self.r,
                out_features=self.channels,
                bias=True,
            ),
        )

    def forward(self, x):
        max = F.adaptive_max_pool2d(x, output_size=1)
        avg = F.adaptive_avg_pool2d(x, output_size=1)
        b, c, _, _ = x.size()
        linear_max = self.linear(max.view(b, c)).view(b, c, 1, 1)
        linear_avg = self.linear(avg.view(b, c)).view(b, c, 1, 1)
        output = linear_max + linear_avg
        output = F.sigmoid(output) * x
        return output


# so SAM in put is Output of CAM
class SAM(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv = nn.Conv2d(
            in_channels=2,
            out_channels=1,
            kernel_size=7,
            stride=1,
            padding=3,
            dilation=1,
        )

    def forward(self, x):
        max = torch.max(x, 1)[0].unsqueeze(1)
        mean = torch.mean(x, 1).unsqueeze(1)
        concat = torch.cat((max, mean), dim=1)
        f = self.conv(concat)
        f = F.sigmoid(f) * x  # F'' = F' * whatever the fuck just happen
        return f


class CBAM(nn.Module):
    def __init__(self, channels, r):
        super(CBAM, self).__init__()
        self.c = channels
        self.r = r
        self.SAM = SAM()
        self.CAM = CAM(channels=self.c, r=self.r)

    def forward(self, x):
        output = self.CAM(x)
        output = self.SAM(output)
        output = output + x
