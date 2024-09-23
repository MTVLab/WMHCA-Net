import torch
import torch.nn as nn
#非典型NON-LOCAL
from SeBlock import SeBlock
#目前是仅打乱通道
from ShuffleBlock import ShuffleBlock
from torch.utils.checkpoint import checkpoint

#目前的逻辑，双模态输入，然后打乱通道，然后分组卷积，顺便池化，然后再实质性的分组，增加一个维度
class NLBlock1(nn.Module):
    def __init__(self, channel, r=4, N=16):#缩放因子r,分组N
        super(NLBlock1, self).__init__()#继承
        self.inter_channel = channel // r
        self.Head = N

        ###########来吧，用分组卷积############################
        self.conv_phi = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1, padding=0, bias=False)
        #用来初始化权重或者偏置
        # nn.init.constant_(self.conv_phi.weight, 0)
        self.conv_theta = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_g = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1, padding=0, bias=False)
        ##########################################################

        self.softmax = nn.Softmax(dim=1)
        self.conv_mask = nn.Conv2d(in_channels=self.inter_channel, out_channels=channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.maxpool = nn.MaxPool2d(2, 2)
        #SENET模块
        self.SE = SeBlock(channel=int(32//self.Head)**2, reduction=2)
        self.unfold_patch = nn.Unfold(kernel_size=(self.Head, self.Head), dilation=1, padding=0, stride=self.Head)
        self.fold_delpatch = nn.Fold(output_size=(32, 32), kernel_size=(self.Head, self.Head), stride=self.Head)
        # self.Shuffle = ShuffleBlock(groups=N)

    def forward(self, xa_phi, xb_theta):
        xc = xa_phi + xb_theta

        # print('打乱后', xa_phi.shape)
        #[B, C, H , W]-->[B, C//r, H , W]
        xa_phi = self.conv_phi(xa_phi)
        xb_theta = self.conv_theta(xb_theta)
        xab_g = self.conv_g(xc)

        b, c_r, h, w = xab_g.size()#[b,c_r,h,w]
        # print('卷积后', xab_g.shape)

        # #分头 [B, C//r, H , W]-->[B, (H//self.Head)**2, C//r, self.Head , self.Head]-->[B, N, C//rN, H W]or[B, N, HW, C//rN]
        # [b,c*self.Head*self.Head,l]l是分块个数
        # print(xab_g)
        # print( self.unfold_patch(xa_phi).shape)
        xa_phi = self.unfold_patch(xa_phi).transpose(2, 1).contiguous().view(b, int(h//self.Head)**2, c_r, -1)#[1,4,4,256]
        xb_theta = self.unfold_patch(xb_theta).transpose(2, 1).contiguous().view(b, int(h//self.Head)**2, c_r, -1).permute(0, 1, 3, 2).contiguous()#[1,4,256,4]
        xab_g = self.unfold_patch(xab_g).transpose(2, 1).contiguous().view(b, int(h//self.Head)**2, c_r, -1).permute(0, 1, 3, 2).contiguous()#[b, l, self.Head*self.Head, c]
        # print(xab_g)
        # print(xb_theta.shape)

        # print('x_phi完成分组和hw合并后', xa_phi.shape,'x_theta完成分组和hw合并后', xb_theta.shape,'x_g完成分组和hw合并后', xab_g.shape)

        #矩阵乘法#
        mul_theta_phi = torch.matmul(xb_theta, xa_phi)#[b, l, self.Head*self.Head,  self.Head*self.Head]
        # print('第一次矩阵乘法', mul_theta_phi.shape)

        # [B, H//self.Head, Head * Head, Head * Head]
        # 类似SE的通道注意力模块
        mul_theta_phi, attention = self.SE(mul_theta_phi)
        # print('经过SE模块', mul_theta_phi.shape)
        mul_theta_phi = self.softmax(mul_theta_phi)#[b, l,  self.Head*self.Head, self.Head*self.Head]
        # [b, head_number, self.Head*self.Head, c]    //  mul[b, head_number, self.Head*self.Head, self.Head*self.Head]  //  xab_g[b, l, self.Head*self.Head, c]
        # 第二次矩阵乘法
        mul_theta_phi = torch.matmul(mul_theta_phi, xab_g)
        # print('第二次矩阵乘法', mul_theta_phi.shape)
        #################开始还原了
        #[B,l,块的长宽积,通道]
        # print(mul_theta_phi)
        #[B, l, self.Head * self.Head, C/r] -->[B, 2*H//self.Head , C/r*self.Head*self.Head ]
        mul_theta_phi = mul_theta_phi.permute(0, 1, 3, 2).contiguous().view(b, (int(h//self.Head)**2), -1).transpose(2,1).contiguous()
        # print(mul_theta_phi)
        # [B, 2*H//self.Head , C/r*self.Head*self.Head ] -->[B, C / r, H, W]
        # print(mul_theta_phi.shape)
        mul_theta_phi = self.fold_delpatch(mul_theta_phi)
        # print('第一阶段恢复', mul_theta_phi)
        # print(mul_theta_phi.shape)
        #[B, C/r, H , W]-->[B, C, H , W]
        mul_theta_phi = checkpoint(self.conv_mask, mul_theta_phi)
        # print('第二阶段恢复', mask.shape)
        mul_theta_phi = mul_theta_phi + xc
        return mul_theta_phi, attention

if __name__=='__main__':

    n = 4;
    r = 2;

    input1 = torch.randn(4, 512, 32, 32)
    input2 = torch.randn(4, 512, 32, 32)
    # for i in range(2):
    #     input1[0][i] = i+1
    #     input2[0][i] = i + 1.5

    b, c , _, _ = input1.size()
    model = NLBlock1(channel=c, r=2, N=16)
    # print('paramters', sum(param.numel() for param in model.parameters()))

    # Shuflle = ShuffleBlock(2)
    # x = Shuflle(input)
    out = model(input1, input2)
    # print(out)