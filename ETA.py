import torch
import torch.nn as nn

class Mix(nn.Module):
    def __init__(self, m=-0.80):
        super(Mix, self).__init__()
        w = torch.nn.Parameter(torch.FloatTensor([m]), requires_grad=True)
        w = torch.nn.Parameter(w, requires_grad=True)
        self.w = w
        self.mix_block = nn.Sigmoid()

    def forward(self, fea1, fea2):
        mix_factor = self.mix_block(self.w)
        out = fea1 * mix_factor.expand_as(fea1) + fea2 * (1 - mix_factor.expand_as(fea2))
        return out


class ZPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class AttentionGate(nn.Module):
    def __init__(self, k):
        super(AttentionGate, self).__init__()
        self.compress = ZPool()
        self.conv1 = nn.Conv2d(2, 1, kernel_size=k, stride=1, padding=int(k / 2))

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.conv1(x_compress)
        return x_out
class SimAM(torch.nn.Module):
    def __init__(self, e_lambda=1e-4):
        super(SimAM, self).__init__()
        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    @staticmethod
    def get_module_name():
        return "simam"

    def forward(self, x):
        b, c, h, w = x.size()

        n = w * h - 1

        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5

        return x * self.activaton(y)
class ETA(nn.Module):
    def __init__(self):
        super(ETA, self).__init__()
        self.cw = AttentionGate(7)
        self.hc = AttentionGate(7)
        self.hw = AttentionGate(7)
        self.simam=SimAM()
    def forward(self, x):
        x_perm1 = x.permute(0, 2, 1, 3).contiguous()
        x_out1 = self.simam((self.cw(x_perm1))).sigmoid()
        x_out11 = x_out1.permute(0, 2, 1, 3).contiguous()
        x_perm2 = x.permute(0, 3, 2, 1).contiguous()
        x_out2 = self.simam((self.hc(x_perm2))).sigmoid()
        x_out21 = x_out2.permute(0, 3, 2, 1).contiguous()
        x_out =  self.simam(self.hw(x)).sigmoid()
        x_out = 1 / 3 * (x_out + x_out11 + x_out21)
        return x_out
if __name__ =='__main__':
    ETA1 = ETA()
    #创建一个输入张量
    batch_size = 8
    input_tensor=torch.randn(batch_size, 256, 64, 64 )
    #运行模型并打印输入和输出的形状
    output_tensor =ETA1(input_tensor)
    print("Input shape:",input_tensor.shape)
    print("0utput shape:",output_tensor.shape)
