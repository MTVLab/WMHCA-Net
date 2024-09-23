import torch
from torch import nn

from UBlock import UNet, UNet_winmul, UNet_swin, Unet_MSAC
from torch.utils.checkpoint import checkpoint
from PointRend import PointHead, PointRendLoss
from ACUblock import ACmixUnet
from UXconv import Unet_X
import torch.nn.functional as F
import optuna
from UResconv import ResNet
from UCTans import UNet_CTans

def init_rate_half(tensor):
    if tensor is not None:
        tensor.data.fill_(0.25)

class C2MANet2(nn.Module):
    #申明通道数和类别数
    def __init__(self):
    # def __init__(self, trial):
        super(C2MANet2, self).__init__()
        # self.Unet1 = UNet_winmul(nchannels=3, N=16)
        # self.Unet2 = UNet_winmul(nchannels=3, N=16)

        self.Unet1 = UNet_CTans(nchannels=3, N=16)
        self.Unet2 = UNet_CTans(nchannels=3, N=16)


        #
        # self.Unet1 = UNet(nchannels=3, N=128)
        # self.Unet2 = UNet(nchannels=3, N=128)

        # self.Unet1 = ACmixUnet(nchannels=3, N=128)
        # self.Unet2 = ACmixUnet(nchannels=3, N=128)

        # self.Unet1 = Unet_MSAC(nchannels=3, N=128)
        # self.Unet2 = Unet_MSAC(nchannels=3, N=128)

        # self.Unet1 = Unet_X(nchannels=3, N=128)
        # self.Unet2 = Unet_X(nchannels=3, N=128)

        blk = []
        blk.append(nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1, stride=1, padding=0, bias=False))
        # blk.append(nn.Conv2d(in_channels=32, out_channels=4, kernel_size=1, stride=1, padding=0, bias=False))
        # blk.append(nn.Conv2d(in_channels=4, out_channels=1, kernel_size=1, stride=1, padding=0, bias=False))

        self.conv = nn.Sequential(*blk)


    def forward(self, CBF, CBV, MTT, Tmax):

        out1 = self.Unet1(CBF, CBV)
        out2 = self.Unet2(MTT, Tmax)

        out = torch.cat([out1, out2], dim=1)
        out = checkpoint(self.conv, out)
        out = torch.sigmoid(out)
        # print(out.shape)
        # print('rate:', rate1.cpu().detach(), 'rate2:', rate2.cpu().detach())

        return out#[b,,]   [b,1,64,64]

class C2MANet(nn.Module):
    #申明通道数和类别数
    def __init__(self):
        super(C2MANet, self).__init__()
        self.Unet1 = UNet(nchannels=3)
        self.Unet2 = UNet(nchannels=3)

        # blk = []
        # blk.append(nn.Conv2d(in_channels=512, out_channels=128, kernel_size=1, stride=1, padding=0, bias=False))
        # blk.append(nn.Conv2d(in_channels=128, out_channels=32, kernel_size=1, stride=1, padding=0, bias=False))
        # blk.append(nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1, stride=1, padding=0, bias=False))
        #
        # self.conv = nn.Sequential(*blk)
        self.conv = nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1, stride=1, padding=0, bias=False)
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, CBF, CBV, MTT, Tmax):

        out1, fine1 = self.Unet1(CBF, CBV)
        out2, fine2 = self.Unet2(MTT, Tmax)

        out = torch.cat([out1, out2], dim=1)
        out = checkpoint(self.conv, out)
        # out = torch.sigmoid(out)
        # print('back', out[0])

        fine = torch.cat([fine1, fine2], dim=1)

        return {"res2": fine, "coarse": out}#[b,,]   [b,1,64,64]

class PointRend(nn.Module):
    def __init__(self):
        super(PointRend, self).__init__()
        self.backbone = C2MANet()
        self.head = PointHead(in_c=128*2+1, num_classes=1, k=3, beta=0.75)

    def forward(self, CBF, CBV, MTT, Tmax):
        result = self.backbone(CBF, CBV, MTT, Tmax)
        result.update(self.head(CBF, result["res2"], result["coarse"]))

        return result

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = C2MANet2()
    # print(list(model.named_parameters()))
    # model = PointRend()
    loss = PointRendLoss()
    CBF = torch.randn(2, 3, 256, 256)
    # print(torch.cuda.memory_allocated(0) / 1024 / 1024)

    CBV = torch.randn(2, 3, 256, 256)
    MTT = torch.randn(2, 3, 256, 256)
    TTP= torch.randn(2, 3, 256, 256)
    label = torch.zeros(2, 3, 256, 256)

    out = model(CBF, CBV, MTT, TTP)#dict_keys(['res2', 'coarse', 'rend', 'points'])

    # loss_sum = loss(out, label)
    # print(model)
    # print(out.keys())
    # print(out['res2'].shape)
    # print('paramters', sum(param.numel() for param in model.parameters()))



