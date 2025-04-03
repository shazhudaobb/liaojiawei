import torch
import torch.nn as nn
from pytorch_wavelets import DWTForward


class FSFDown(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(FSFDown, self).__init__()
        self.wt = DWTForward(J=1, mode='zero', wave='haar')

        self.dconv1 = nn.Sequential(
                                  nn.Sequential(nn.Conv2d(in_ch * 2, in_ch * 2, 3, 1, 1, groups=in_ch * 2),
                                  nn.GELU(),
                                  nn.BatchNorm2d(in_ch * 2)
                        ),
            nn.Conv2d(in_ch * 2, out_ch, 1),
        )
        self.dconv2 = nn.Sequential(
                                  nn.Sequential(nn.Conv2d(in_ch * 2, in_ch * 2, 3, 1, 1, groups=in_ch * 2),
                                  nn.GELU(),
                                  nn.BatchNorm2d(in_ch * 2)
                        ),
            nn.Conv2d(in_ch * 2, out_ch, 1),
        )
        self.conv1 = nn.Conv2d(out_ch * 2, out_ch, kernel_size=1, stride=1, padding=0)
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        x1 = torch.cat([x1[..., ::2, ::2], x1[..., 1::2, ::2], x1[..., ::2, 1::2], x1[..., 1::2, 1::2]], 1)
        x1 = self.dconv1(x1)
        yL, yH = self.wt(x2)
        y_HL = yH[0][:,:,0,::]
        y_LH = yH[0][:,:,1,::]
        y_HH = yH[0][:,:,2,::]
        x2 = torch.cat([yL, y_HL, y_LH, y_HH], dim=1)
        x2 = self.dconv2(x2)
        x = torch.cat([x1, x2], 1)
        return self.conv1(x)



if __name__ =='__main__':

    FSFDown = FSFDown(256,256)
    #创建一个输入张量
    batch_size = 8
    input_tensor=torch.randn(batch_size, 256, 64, 64 )
    #运行模型并打印输入和输出的形状
    output_tensor =FSFDown(input_tensor)
    print("Input shape:",input_tensor.shape)
    print("0utput shape:",output_tensor.shape)



