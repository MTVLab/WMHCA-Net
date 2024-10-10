import torch
import torch.nn as nn

BatchNorm2d = nn.BatchNorm2d
BN_MOMENTUM = 0.01

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.gn1 = nn.GroupNorm(num_groups=planes // 16, num_channels=planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(num_groups=planes // 16, num_channels=planes)

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.gn3 = nn.GroupNorm(num_groups=(planes * self.expansion) // 16, num_channels=planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):

        residual = x

        out = self.conv1(x)
        out = self.gn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.gn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.gn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class RefinementMagNet(nn.Module):
    def __init__(self, n_classes, use_bn=False):
        super().__init__()
        self.use_bn = use_bn
        if use_bn:
            self.bn0 = BatchNorm2d(n_classes * 2, momentum=BN_MOMENTUM)
        # 2 conv layers
        self.conv1 = nn.Conv2d(n_classes * 2, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = BatchNorm2d(64, momentum=BN_MOMENTUM)
        # self.bn1 = nn.GroupNorm(num_groups=64 // 16, num_channels=64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = BatchNorm2d(64, momentum=BN_MOMENTUM)
        # self.bn2 = nn.GroupNorm(num_groups=64 // 16, num_channels=64)
        self.relu = nn.ReLU(inplace=True)

        # 2 residual blocks
        self.residual = self._make_layer(Bottleneck, 64, 32, 2)

        # Prediction head
        self.seg_conv = nn.Conv2d(128, n_classes, kernel_size=1, stride=1, padding=0, bias=False)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        """Make residual block"""
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                # BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
                nn.GroupNorm(num_groups=planes * block.expansion // 16, num_channels=planes * block.expansion)
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, fine_segmentation, coarse_segmentation):

        x = torch.cat([fine_segmentation, coarse_segmentation], dim=1)
        if self.use_bn:
            x = self.bn0(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.residual(x)

        return torch.sigmoid(self.seg_conv(x))

if __name__ == '__main__':
    modle = RefinementMagNet(1)
    x1 = torch.randn([2, 1, 256, 256])
    x2 = torch.randn([2, 1, 256, 256])

    modle(x1, x2)
