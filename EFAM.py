import torch
import torch.nn as nn

class Residual(torch.nn.Module):
    def __init__(self, m, drop=0.):
        super().__init__()
        self.m = m
        self.drop = drop

    def forward(self, x):
        if self.training and self.drop > 0:
            return x + self.m(x) * torch.rand(x.size(0), 1, 1, 1,
                                              device=x.device).ge_(self.drop).div(1 - self.drop).detach()
        else:
            return x + self.m(x)

    @torch.no_grad()
    def fuse(self):
        if isinstance(self.m, nn.Conv2d):
            m = self.m.fuse()
            assert (m.groups == m.in_channels)
            identity = torch.ones(m.weight.shape[0], m.weight.shape[1], 1, 1)
            identity = torch.nn.functional.pad(identity, [1, 1, 1, 1])
            m.weight += identity.to(m.weight.device)
            return m
        else:
            return self

class PSSA(nn.Module):
    def __init__(self, dim, r=0.25):
        super(PSSA, self).__init__()
        self.dim = dim
        self.pdim = int(dim * r)  # 处理通道数
        self.qk_dim = self.pdim // 2  # Q和K的维度
        self.scale = self.pdim ** -0.5
        # 使用 Conv2d 替换 Linear
        self.qkv = nn.Conv2d(self.pdim, 2 * self.qk_dim + self.pdim, kernel_size=1)
        # 特征处理
        self.norm = nn.BatchNorm2d(self.pdim)
        self.act = nn.GELU()
        # 输出投影
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        # 通道分离
        x1, x2 = torch.split(x, [self.pdim, self.dim - self.pdim], dim=1)
        # Q 生成
        x1 = self.norm(x1)
        qkv = self.qkv(x1)  # B, pdim, H, W
        q, k, v = qkv.split([self.qk_dim, self.qk_dim, self.pdim], dim=1)
        q, k, v = q.flatten(2), k.flatten(2), v.flatten(2)
        # 注意力计算
        attn = (q.transpose(1, 2) @ k) * self.scale  # B, HW, HW/p^2
        attn = attn.softmax(dim=-1)
        # 应用注意力
        x1_out = (attn @ v.transpose(1, 2))  # B, HW, pdim
        x1_out = x1_out.reshape(B, H, W, self.pdim).permute(0, 3, 1, 2)  # B, pdim, H, W
        # 合并特征
        x_cat = torch.cat([x1_out, x2], dim=1)  # B, C, H, W
        # 投影输出
        x_cat = self.act(x_cat)
        out = self.proj(x_cat)  # B, C, H, W
        return out

class EFEB(torch.nn.Module):
    def __init__(self, dim):
        super(EFEB, self).__init__()
        self.dim = dim
        self.dconv3 = nn.Sequential(
            Residual(nn.Sequential(
                nn.Conv2d(dim, dim, 3, 1, 1, groups=dim),
                nn.GELU(),
                nn.BatchNorm2d(dim)
            )),
            nn.Conv2d(dim, dim * 2, 1, 1, 0),
            torch.nn.ReLU(),
            nn.Conv2d(dim * 2, dim, 1),
            nn.BatchNorm2d(dim)
        )
        self.shortcut = nn.Conv2d(dim, dim, kernel_size=1, stride=1)

    def forward(self, x):
        identy = x
        x = self.dconv3(x)
        x = x + self.shortcut(identy)
        return x

class EFAM(torch.nn.Module):
    def __init__(self, c1, c2, n=1, r=0.5, g=1, e=0.5):
        super().__init__()
        self.r = r
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = nn.Conv2d(c1, 2 * self.c, 1, 1)
        self.cv2 = nn.Conv2d((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList([EFEB(self.c) for _ in range(n)])
        self.mixer = Residual(PSSA(self.c, r))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        y[-1] = self.mixer(y[-1])
        return self.cv2(torch.cat(y, 1))

if __name__ == '__main__':
    efam = EFAM(256, 256)
    batch_size = 8
    input_tensor = torch.randn(batch_size, 256, 64, 64)
    output_tensor = efam(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)