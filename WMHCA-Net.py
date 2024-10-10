import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import ml_collections
##########################################################################
# WMHCA_Block
##########################################################################
class SeBlock(nn.Module):
    def __init__(self, channel, reduction=2):
        super(SeBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, n, _, _ = x.size()
        # [B, N, H * W, H/2 * W/2]-->[B, N]
        y = self.avg_pool(x).view(b, n)
        # [B, N, C]-->[B, N, C, 1 , 1]
        y = checkpoint(self.fc, y).view(b, n, 1, 1)
        return x * y.expand_as(x)   #改变大小和你矩阵乘法

class WMHCA_Block(nn.Module):
    def __init__(self, channel, r=2, N=16):#缩放因子r,分组N
        super(WMHCA_Block, self).__init__()#继承
        self.inter_channel = channel // r
        self.WindowSize = N

        ###########来吧，用分组卷积############################
        self.conv_phi = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1, padding=0, bias=False)
        #用来初始化权重或者偏置
        self.conv_theta = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_g = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1, padding=0, bias=False)
        ##########################################################

        self.softmax = nn.Softmax(dim=1)
        self.conv_mask = nn.Conv2d(in_channels=self.inter_channel, out_channels=channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.maxpool = nn.MaxPool2d(2, 2)
        #SE模块
        self.SE = SeBlock(channel=int(32 // self.WindowSize) ** 2, reduction=2)
        self.unfold_patch = nn.Unfold(kernel_size=(self.WindowSize, self.WindowSize), dilation=1, padding=0, stride=self.WindowSize)
        self.fold_delpatch = nn.Fold(output_size=(32, 32), kernel_size=(self.WindowSize, self.WindowSize), stride=self.WindowSize)

    def forward(self, xa_phi, xb_theta):
        xc = xa_phi + xb_theta
        #[B, C, H , W]-->[B, C//r, H , W]
        xa_phi = self.conv_phi(xa_phi)
        xb_theta = self.conv_theta(xb_theta)
        xab_g = self.conv_g(xc)

        b, c_r, h, w = xab_g.size()#[b,c//r,h,w]

        # [B, C//r, H , W]-->[B, (H//self.WindowSize)**2, C//r, self.WindowSize , self.WindowSize]-->[B, N, C//rN, H W]or[B, N, HW, C//rN]
        # [b,c*self.Head*self.Head,l]l是分块个数
        xa_phi = self.unfold_patch(xa_phi).transpose(2, 1).contiguous().view(b, int(h // self.WindowSize) ** 2, c_r, -1)
        xb_theta = self.unfold_patch(xb_theta).transpose(2, 1).contiguous().view(b, int(h // self.WindowSize) ** 2, c_r, -1).permute(0, 1, 3, 2).contiguous()
        xab_g = self.unfold_patch(xab_g).transpose(2, 1).contiguous().view(b, int(h // self.WindowSize) ** 2, c_r, -1).permute(0, 1, 3, 2).contiguous()#[b, l, self.WindowSize*self.WindowSize, c]

        #矩阵乘法#
        mul_theta_phi = torch.matmul(xb_theta, xa_phi)#[b, l, self.WindowSize*self.WindowSize,  self.WindowSize*self.WindowSize]

        # 类似SE的通道注意力模块
        mul_theta_phi = self.SE(mul_theta_phi)
        mul_theta_phi = self.softmax(mul_theta_phi)#[b, l,  self.WindowSize*self.WindowSize, self.WindowSize*self.WindowSize]
        # 第二次矩阵乘法
        mul_theta_phi = torch.matmul(mul_theta_phi, xab_g)
        #################开始还原了
        mul_theta_phi = mul_theta_phi.permute(0, 1, 3, 2).contiguous().view(b, (int(h // self.WindowSize) ** 2), -1).transpose(2, 1).contiguous()
        mul_theta_phi = self.fold_delpatch(mul_theta_phi)
        mul_theta_phi = checkpoint(self.conv_mask, mul_theta_phi)
        mul_theta_phi = mul_theta_phi + xc
        return mul_theta_phi

##########################################################################
# U-Net
##########################################################################
def get_CTranS_config():
    config = ml_collections.ConfigDict()
    config.transformer = ml_collections.ConfigDict()
    config.KV_size = 448  # KV_size = Q1 + Q2 + Q3
    config.transformer.num_heads  = 4
    config.transformer.num_layers = 4
    config.expand_ratio           = 4  # MLP channel dimension expand ratio
    config.transformer.embeddings_dropout_rate = 0.1
    config.transformer.attention_dropout_rate = 0.1
    config.transformer.dropout_rate = 0
    config.patch_sizes = [16, 8, 4, 2]
    config.base_channel = 64 # base channel of U-Net
    config.n_classes = 1
    return config

config = get_CTranS_config()

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class CCA(nn.Module):
    """
    CCA Block
    """
    def __init__(self, F_g, F_x):
        super().__init__()
        self.mlp_x = nn.Sequential(
            Flatten(),
            nn.Linear(F_x, F_x))
        self.mlp_g = nn.Sequential(
            Flatten(),
            nn.Linear(F_g, F_x))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # channel-wise attention
        avg_pool_x = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
        channel_att_x = self.mlp_x(avg_pool_x)
        avg_pool_g = F.avg_pool2d( g, (g.size(2), g.size(3)), stride=(g.size(2), g.size(3)))
        channel_att_g = self.mlp_g(avg_pool_g)

        channel_att_sum = (channel_att_x + channel_att_g)/2.0
        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        x_after_channel = x * scale
        out = self.relu(x_after_channel)
        return out

class DownBlock(nn.Module):#num_conv是内部卷积层
    def __init__(self, num_convs, inchannels, outchannels, pool=True):
        super(DownBlock, self).__init__()
        self.inchannel = inchannels
        self.outchannel = outchannels
        self.pool = pool

        blk = []

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        for i in range(num_convs):
            if i == 0:
                blk.append(nn.Conv2d(self.inchannel, self.outchannel, kernel_size=1, stride=1))
                blk.append(nn.Conv2d(self.outchannel, self.outchannel, kernel_size=3, stride=1, padding=1))
            else:
                blk.append(nn.Conv2d(self.outchannel, self.outchannel, kernel_size=3, stride=1, padding=1))
            blk.append(nn.GroupNorm(num_groups=self.outchannel//16, num_channels=self.outchannel))
            blk.append(nn.ReLU(inplace=True))
        self.layer = nn.Sequential(*blk)


    def forward(self, x):
        if self.pool:
            out = self.maxpool(x)
            x = self.layer(out)
            return x
        else:
            x = self.layer(x)
            return x
###########################上采样#########################
class UpBlock(nn.Module):
    def __init__(self, inchannel1, inchannel2, outchannels):
        super(UpBlock, self).__init__()
        self.inchannel1 = inchannel1
        self.inchannel2 = inchannel2
        self.outchannel = outchannels

        #反卷积多用于分割等像素级任务
        #图片H,W变大,通道数压缩
        self.convt = nn.ConvTranspose2d(self.inchannel1, self.outchannel, kernel_size=2, stride=2)

        self.coatt = CCA(F_g=self.inchannel1 // 2, F_x=self.inchannel1 // 2)  # 前面是上采样，后面是跳跃连接

        self.conv = nn.Sequential(
            nn.Conv2d(self.inchannel2, self.outchannel, kernel_size=1, stride=1),
            nn.Conv2d(self.outchannel, self.outchannel, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=self.outchannel//16, num_channels=self.outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.outchannel, self.outchannel, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=self.outchannel//16, num_channels=self.outchannel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x1 = checkpoint(self.convt, x1)
        x2 = self.coatt(g=x1, x=x2)
        _, _, h, _ = x1.size()

        x1 = torch.cat([x2, x1], dim=1)

        x1 = checkpoint(self.conv, x1)
        return x1

class UShape(nn.Module):
    #申明通道数和类别数
    def __init__(self, nchannels=1, N=64, depths=[2, 2, 2, 2], dims=[64, 128, 256, 512]):
        super(UShape, self).__init__()

        self.downa1 = DownBlock(num_convs=depths[0], inchannels=nchannels, outchannels=dims[0], pool=False)
        self.downa2 = DownBlock(num_convs=depths[1], inchannels=dims[0], outchannels=dims[1])
        self.downa3 = DownBlock(depths[2], dims[1], dims[2])
        self.downa4 = DownBlock(depths[3], dims[2], dims[3])

        self.downb1 = DownBlock(num_convs=depths[0], inchannels=nchannels, outchannels=dims[0], pool=False)
        self.downb2 = DownBlock(depths[1], dims[0], dims[1])
        self.downb3 = DownBlock(depths[2], dims[1], dims[2])
        self.downb4 = DownBlock(depths[3], dims[2], dims[3])

        self.WMHCA = WMHCA_Block(channel=dims[3], r=2, N=N)

        self.up1 = UpBlock(dims[3], dims[3], dims[2])
        self.up2 = UpBlock(dims[2], dims[2], dims[1])
        self.up3 = UpBlock(dims[1], dims[1], dims[0])

    def forward(self, xa, xb):
        #两模态开始初输入
        #[B, C, H , W]-->[B, 64, H , W]
        xa = self.downa1(xa)
        xb = self.downb1(xb)

        # [B, 64, H , W]-->[B, 128, H/2 , W/2]
        xa2 = self.downa2(xa)
        xb2 = self.downb2(xb)
        # [B, 128, H/2 , W/2]-->[B, 256, H/4 , W/4]
        xa3 = checkpoint(self.downa3, xa2)
        xb3 = checkpoint(self.downb3, xb2)
        #[B, 256, H/4 , W/4]-->[B, 512, H/8 , W/8]
        xa4 = checkpoint(self.downa4, xa3)
        xb4 = checkpoint(self.downb4, xb3)

        y,_ = self.WMHCA(xa4, xb4)
        # print('NL之后', y.shape)
        xa = xa + xb#[B, C, H, W]
        xa2 = xa2 + xb2
        xa3 = xa3 + xb3
        # [B, 512, H/8 , W/8]-->[B, 256, H/4 , W/4]
        y = self.up1(y, xa3)
        # [B, 256, H/4 , W/4]-->[B, 128, H/2 , W/2]
        y = self.up2(y, xa2)
        # [B, 128, H/2 , W/2]-->[B, 64, H , W]
        y = self.up3(y, xa)


        return y

class WMHCA_Net(nn.Module):
    #申明通道数和类别数
    def __init__(self):
        super(WMHCA_Net, self).__init__()

        self.Unet1 = UShape(nchannels=3, N=16)
        self.Unet2 = UShape(nchannels=3, N=16)

        blk = []
        blk.append(nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1, stride=1, padding=0, bias=False))

        self.conv = nn.Sequential(*blk)

    def forward(self, CBF, CBV, MTT, Tmax):

        out1 = self.Unet1(CBF, CBV)
        out2 = self.Unet2(MTT, Tmax)

        out = torch.cat([out1, out2], dim=1)
        out = checkpoint(self.conv, out)
        out = torch.sigmoid(out)

        return out

if __name__=='__main__':

    n = 4;
    r = 2;

    input1 = torch.randn(4, 512, 32, 32)
    input2 = torch.randn(4, 512, 32, 32)

    b, c , _, _ = input1.size()
    model = WMHCA_Block(channel=c, r=2, N=16)
    out = model(input1, input2)
