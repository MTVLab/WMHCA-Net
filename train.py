from torch.optim.lr_scheduler import CosineAnnealingLR,CosineAnnealingWarmRestarts,StepLR
from DataSet import DataSet
from torch.utils.data import DataLoader
from WMHCA-Net import WMHCA_Net
from refinement import RefinementMagNet

import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
# from visdom import Visdom#python -m visdom.server

#init
def weight_init(m):
    # 也可以判断是否为conv2d，使用相应的初始化方式
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
     # 是否为批归一化层
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


#loss
class BCE_WITH_WEIGHT(torch.nn.Module):
    def __init__(self, alpha=0.2, reduction='mean'):
        super(BCE_WITH_WEIGHT, self).__init__()
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, predict, target):
        pt = predict
        loss = -((1-self.alpha) * target * torch.log(pt+1e-5) + self.alpha * (1 - target) * torch.log(1 - pt+1e-5))
        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)

        return loss

class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()
        self.Wbce = BCE_WITH_WEIGHT(alpha=0.25, reduction='mean')

    def forward(self, inputs, targets, smooth=0.0001):
        # inputs = torch.sigmoid(inputs)

        BCE = self.Wbce(inputs, targets)
        # BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        N = targets.size()[0]
        # 平滑变量
        # smooth = 1
        # 将宽高 reshape 到同一纬度
        input_flat = inputs.view(N, -1)
        targets_flat = targets.view(N, -1)
        # 计算交集
        intersection = input_flat * targets_flat
        N_dice_eff = (2 * intersection.sum(1) + smooth) / (input_flat.sum(1) + targets_flat.sum(1) + smooth)
        # 计算一个批次中平均每张图的损失
        dice_loss =1 - N_dice_eff.sum() / N
        Dice_BCE = BCE + dice_loss

        return Dice_BCE

def TrainWMHCA(Device, Epochs=40, lr=0.001, BatchSize=2, accumulation = 2):
    # region setting
    # viz = Visdom()
    #load dataset
    dataset = DataSet(mod='train')
    TrainLoader = DataLoader(dataset=dataset, batch_size=BatchSize, shuffle=True, num_workers=0)

    datasettest = DataSet(mod='test')
    TestLoader = DataLoader(dataset=datasettest, batch_size=4, shuffle=True, num_workers=0)

    per_epoch_num = len(dataset)/BatchSize
    test_len_ = len(datasettest) / 4

    # optimizer
    optimizer = torch.optim.Adam(refine.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    scheduler = CosineAnnealingLR(optimizer, T_max=Epochs, eta_min=0, last_epoch=-1, verbose=False)
    

    #Loss
    criterion = DiceBCELoss()#使用sigmod

    best_loss = float('inf')
    best_test_loss = 0

    loss_prin = torch.zeros([1])
    loss_mean = torch.zeros([1])
    # 训练epochs次
    lr_change = []
    #tqdm不影响代码运行效率的情况下，会显示一个程序的执行进度
    # endregion
    with tqdm(total=Epochs * per_epoch_num) as pbar:#就是表示总训练的轮次上限

        for epoch in range(Epochs):
            optimizer.zero_grad()
            # 训练模式
            Net.train()

            # 按照batch_size开始训练
            for index, (x, label) in enumerate(TrainLoader):#x为[b,四大参数图,c,h,w]
                # 将数据拷贝到device中
                x = x.transpose(0, 1).contiguous()#[b,四大参数图,c,h,w]-->[四大参数图,b,c,h,w]
                cbv = x[0].to(device=Device)
                cbf = x[1].to(device=Device, dtype=torch.float32)
                mtt = x[2].to(device=Device, dtype=torch.float32)
                ttp = x[3].to(device=Device, dtype=torch.float32)
                label = label.to(device=Device, dtype=torch.float32)

                # 使用网络参数，输出预测结果
                pred = Net(cbv, cbf, mtt, ttp)
                # 计算loss,设备显存受限，采用累计梯度算法，可根据自身情况修改
                loss = criterion(pred, label) / accumulation

                loss_prin = loss.item() + loss_prin  # 梯度累计
                # print('{}/{}：Loss/train'.format(epoch + 1, Epochs), loss.item())

                # 更新参数
                loss.backward()
                if ((index + 1) % accumulation) == 0:
                    print('{}/{}：Loss/train'.format(epoch + 1, Epochs), loss_prin)
                    loss_mean = loss_mean + loss_prin
                    # 保存loss值最小的网络参数
                    if loss_prin < best_loss:
                        best_loss = loss_prin
                        torch.save(Net.state_dict(), 'best_model.pth')
                    loss_prin = torch.zeros([1])

                    optimizer.step()
                    optimizer.zero_grad()
                    # print(index)
                    if (((index + 1))) == (len(dataset) / BatchSize):  # 1120，index最大为1120/batchsize
                        #visom观察训练过程
                        # if epoch == 0:
                        #     viz.line([(loss_mean * accumulation) / (len(dataset) / BatchSize)], [epoch + 1],
                        #              win='train_loss', opts=dict(title='train loss'))
                        # else:
                        #     viz.line([(loss_mean * accumulation) / (len(dataset) / BatchSize)], [epoch + 1],
                        #              win='train_loss', update='append')  # 140
                        loss_mean = torch.zeros([1])
                pbar.update(1)
            # 获取当前学习率
            lr_change.append(scheduler.get_last_lr()[0])
            scheduler.step()  # 注意 每个epoch 结束， 更新learning rate

#切分高分辨率区域
def roi_section(x, index_):

    if torch.is_tensor(x):
        b, c, h, w = x.shape
        # print(x.shape)
        if h==512:
            unfold_ = nn.Unfold(kernel_size=(256, 256), dilation=1, padding=0, stride=256)

            x = unfold_(x).permute(0, 2, 1).contiguous().view(b, 4, c, 256, 256)
            out = torch.zeros([b, c, 256, 256])

            for i in range(b):
                out[i] = x[i, index_[i]]

        elif h==256:
            unfold_ = nn.Unfold(kernel_size=(128, 128), dilation=1, padding=0, stride=128)

            x = unfold_(x).permute(0, 2, 1).contiguous().view(b, 4, c, 128, 128)
            out = torch.zeros([b, c, 128, 128])

            for i in range(b):
                out[i] = x[i, index_[i]]

    return out#[b,c,h,w]

def TrainRefine(Device, Epochs=40, lr=0.001, BatchSize=2, accumulation=2):
    # region setting
    # viz = Visdom()
    # load dataset
    dataset = DataSet(mod='train')
    TrainLoader = DataLoader(dataset=dataset, batch_size=BatchSize, shuffle=True, num_workers=0)

    datasettest = DataSet(mod='test')
    TestLoader = DataLoader(dataset=datasettest, batch_size=4, shuffle=True, num_workers=0)

    per_epoch_num = len(dataset) / BatchSize
    test_len_ = len(datasettest) / 4

    # optimizer
    optimizer = torch.optim.Adam(refine.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0,
                                 amsgrad=False)
    scheduler = CosineAnnealingLR(optimizer, T_max=Epochs, eta_min=0, last_epoch=-1, verbose=False)

    # Loss
    criterion = DiceBCELoss()  # 使用sigmod

    best_loss = float('inf')
    best_test_loss = 0

    loss_prin = torch.zeros([1])
    loss_mean = torch.zeros([1])
    # 训练epochs次
    lr_change = []
    # tqdm不影响代码运行效率的情况下，会显示一个程序的执行进度
    # endregion
    with tqdm(total=Epochs * per_epoch_num) as pbar:  # 就是表示总训练的轮次上限

        for epoch in range(Epochs):
            optimizer.zero_grad()
            # 训练模式
            Net.eval()
            refine.train()
            # 按照batch_size开始训练
            for index, (x, label, Y_Uncer_index_) in enumerate(TrainLoader):#Y_Uncer_index_为不确定度最高的区域索引，提前离线计算减少训练时间
                x512 = x[0]#x512为不确定度最高的区域，提前切分出来减少训练时间
                x256 = x[1]
                label256 = label
                del x, label

                x512 = x512.to(device=Device, dtype=torch.float32)

                x256 = roi_section(x256, Y_Uncer_index_).to(device=Device, dtype=torch.float32)
                label256 = roi_section(label256, Y_Uncer_index_).to(device=Device, dtype=torch.float32)

                logits = refine(x512, x256)  # [b,c,128,128]

                loss = criterion(logits, label256) / accumulation
                loss_prin = loss.item() + loss_prin  # 梯度累计
                # 更新参数
                loss.backward()
                if ((index + 1) % accumulation) == 0:
                    print('{}/{}：Loss/train'.format(epoch + 1, Epochs), loss_prin)
                    loss_mean = loss_mean + loss_prin
                    # 保存loss值最小的网络参数
                    if loss_prin < best_loss:
                        best_loss = loss_prin
                        torch.save(refine.state_dict(), 'best_refine_model.pth')
                    loss_prin = torch.zeros([1])

                    optimizer.step()
                    optimizer.zero_grad()
                    # if (((index + 1))) == (len(dataset) / BatchSize):  # 1120，index最大为1120/batchsize
                    #     if epoch == 0:
                    #         viz.line([(loss_mean * accumulation) / (len(dataset) / BatchSize)], [epoch + 1],
                    #                  win='train_loss', opts=dict(title='train loss'))
                    #     else:
                    #         viz.line([(loss_mean * accumulation) / (len(dataset) / BatchSize)], [epoch + 1],
                    #                  win='train_loss', update='append')  # 140
                    #     loss_mean = torch.zeros([1])
                pbar.update(1)
                # 获取当前学习率
            lr_change.append(scheduler.get_last_lr()[0])
            scheduler.step()  # 注意 每个epoch 结束， 更新learning rate


if __name__ == '__main__':
    # 选择设备，有cuda用cuda，没有就用cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    Net = WMHCA_Net().to(device=device)
    Net.apply(weight_init)

    refine = RefinementMagNet(n_classes=1, use_bn=True).to(device=device)
    refine.apply(weight_init)

    # TrainWMHCA(Device=device, Epochs=50, BatchSize=32, accumulation=1, lr=0.001)
    print('train——over')
