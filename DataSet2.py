import numpy as np
import torch
from torch.utils.data import Dataset
import random
import cv2

#读取数据集
class DataSet2(Dataset):

    def __init__(self,  CT_PATH = 'D:/LLF/GPU/IMG/CT/',  CTP_path = 'D:/LLF/GPU/IMG/CTP/', mod ='train'):
        path, list_index_ = DataPath(CT_PATH, CTP_path, mod)
        x, y = CTtoLab(path)#返回图片numpy
        self.y_data = np.array(y)#[b,w,h,c]
        self.mod = mod
        OUT = []
        #CBVOut, CBFOut, MTTOut, TTPOut
        OUT.append(x[0]),OUT.append(x[1]),OUT.append(x[2]),OUT.append(x[3])
        self.x_data = np.array(OUT).transpose(1, 0, 2, 3, 4)#[b,参数,w,h,c]

    def __getitem__(self, index):

        #输入[参数,w,h,c],标签[w,h,c]---> (CBVOut, CBFOut, MTTOut, TTPOut, LabOut)
        AUG, yaug = DataAugment(self.x_data[index][0],self.x_data[index][1],self.x_data[index][2],self.x_data[index][3],self.y_data[index],self.mod)

        c,h,w = AUG[0].size()

        ##########此处没有关于b的通道，但是本函数在传给main文件之后，会自动组合一个b出来，香的很
        # print(AUG[0].shape)#[3,256,256]=[c,h,w]
        Temp = torch.zeros([4, c, h, w])
        Temp[0] = AUG[0]
        Temp[1] = AUG[1]
        Temp[2] = AUG[2]
        Temp[3] = AUG[3]

        self.x_data_aug = Temp
        # print(self.x_data_aug.shape)
        self.y_data_aug = yaug

        return self.x_data_aug, self.y_data_aug

    def __len__(self):
        return len(self.y_data)

class DataSetmag(Dataset):

    def __init__(self,  CT_PATH = 'D:/LLF/GPU/IMG/CT/',  CTP_path = 'D:/LLF/GPU/IMG/CTP/', mod ='train'):
        path, self.list_index_ = DataPath(CT_PATH, CTP_path, mod)
        x, y = CTtoLab(path, mod='mag')#返回图片numpy
        # x, y = CTtoLab(path)  # 返回图片numpy
        self.y_data = np.array(y)#[b,w,h,c]
        self.mod = mod
        OUT = []
        #CBVOut, CBFOut, MTTOut, TTPOut
        OUT.append(x[0]), OUT.append(x[1]), OUT.append(x[2]), OUT.append(x[3])
        self.x_data = np.array(OUT).transpose(1, 0, 2, 3, 4)#[b,参数,w,h,c]

    def __getitem__(self, index):

        #输入[参数,w,h,c],标签[w,h,c]---> (CBVOut, CBFOut, MTTOut, TTPOut, LabOut)
        AUG, yaug = DataAugment(self.x_data[index][0], self.x_data[index][1], self.x_data[index][2], self.x_data[index][3], self.y_data[index], mod=self.mod)
        # AUG256 = DataAugment(self.x256_data[index][0], self.x256_data[index][1], self.x256_data[index][2], self.x256_data[index][3],
        #                  self.y256_data[index], mod=self.mod)

        c, h, w = AUG[0].size()
        # c256, h256, w256 = AUG256[0].size()

        ##########此处没有关于b的通道，但是本函数在传给main文件之后，会自动组合一个b出来，香的很
        # print(AUG[0].shape)#[3,256,256]=[c,h,w]
        Temp = torch.zeros([4, c, h, w])
        Temp[0] = AUG[0]
        Temp[1] = AUG[1]
        Temp[2] = AUG[2]
        Temp[3] = AUG[3]

        # Temp256 = torch.zeros([4, c256, h256, w256])
        # Temp256[0] = AUG256[0]
        # Temp256[1] = AUG256[1]
        # Temp256[2] = AUG256[2]
        # Temp256[3] = AUG256[3]

        # temp_out_ = torch.concat([Temp, Temp256], dim=0)
        # AUG_out_ = torch.concat([AUG[4], AUG256[4]], dim=0)

        # self.x_data_aug = (Temp, Temp256)
        # self.y_data_aug = (AUG[4], AUG256[4])

        self.x_data_aug = Temp
        self.y_data_aug = yaug

        self.xy_list_index_ = torch.tensor(self.list_index_[index])
        # self.xy_list_index_ = 0

        return self.x_data_aug, self.y_data_aug, self.xy_list_index_

    def __len__(self):
        return len(self.y_data)

#读取数据路径
def DataPath(mod = 'train'):
    if mod == 'train':
        a = np.load('aug+train.npy')  # dwi+ct,未分组
        a = a.tolist()

        c = np.load('Train_list_.npy')#提前计算出不确定度Y最高的区域，以节省计算时间。当然也可以在线运算。
        c = c.tolist()
        return a, c
    elif mod == 'test':
        b = np.load('test_path.npy')  # dwi+ct，未分组
        b = b.tolist()

        c = np.load('Test_list_.npy')
        c = c.tolist()
        return b, c
    else:
        print('exit!!!')
        return 0

#根据路径读取数据
def CTtoLab(XYPath, mod = 'no'):
    CBVOut = []
    CBFOut = []
    MTTOut = []
    TTPOut = []
    LabOut = []

    for i, Index in enumerate(XYPath):
        CT_pic_dir = Index[4]
        CBV_dir = Index[0]
        TTP_dir = Index[3]
        CBF_dir = Index[1]
        MTT_dir = Index[2]

#############读取
        CT_pic = cv2.imread(CT_pic_dir)
        # print(CT_pic_dir)
        CT_pic[CT_pic <= 254] = 0


        CBV = cv2.imread(CBV_dir)
        # print(CBV_dir)
        TTP = cv2.imread(TTP_dir)
        CBF = cv2.imread(CBF_dir)
        MTT = cv2.imread(MTT_dir)

        if mod == 'mag':
            CBV = cv2.resize(CBV, [512, 512])
            TTP = cv2.resize(TTP, [512, 512])
            CBF = cv2.resize(CBF, [512, 512])
            MTT = cv2.resize(MTT, [512, 512])
            CT_pic = cv2.resize(CT_pic, [512, 512])
        elif mod == 'no':
            CBV = cv2.resize(CBV, [256, 256])
            TTP = cv2.resize(TTP, [256, 256])
            CBF = cv2.resize(CBF, [256, 256])
            MTT = cv2.resize(MTT, [256, 256])
            CT_pic = cv2.resize(CT_pic, [256, 256])

        CBVOut.append(CBV)
        CBFOut.append(CBF)
        MTTOut.append(MTT)
        TTPOut.append(TTP)
        LabOut.append(CT_pic)

    return (CBVOut, CBFOut, MTTOut, TTPOut), LabOut

if __name__ == '__main__':
    E = DataPath(mod='test')
