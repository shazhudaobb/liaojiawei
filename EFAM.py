import torch
import torch.nn as nn


class Residual(torch.nn.Module):
    """
    Residual connection with stochastic depth regularization
    Implements random residual dropout during training to prevent overfitting
    """

    def __init__(self, m, drop=0.):
        super().__init__()
        self.m = m  # Main branch module
        self.drop = drop  # Drop probability (0 for no dropout)

    def forward(self, x):
        # Apply residual connection with dropout during training
        if self.training and self.drop > 0:
            # Stochastic depth implementation
            return x + self.m(x) * torch.rand(x.size(0), 1, 1, 1,
                                              device=x.device).ge_(self.drop).div(1 - self.drop).detach()
        else:
            return x + self.m(x)

    @torch.no_grad()
    def fuse(self):
        # Model fusion optimization for deployment
        if isinstance(self.m, nn.Conv2d):
            m = self.m.fuse()
            assert (m.groups == m.in_channels)
            # Add identity mapping to convolutional weights
            identity = torch.ones(m.weight.shape[0], m.weight.shape[1], 1, 1)
            identity = torch.nn.functional.pad(identity, [1, 1, 1, 1])
            m.weight += identity.to(m.weight.device)
            return m
        else:
            return self


class PSSA(nn.Module):
    """
    Partial Single-Head Self-Attention Module
    Implements channel-separated self-attention mechanism
    """

    def __init__(self, dim, r=0.25):
        super(PSSA, self).__init__()
        self.dim = dim  # Input dimension
        self.pdim = int(dim * r)  # Processed channel dimension (r=0.25 for P4 layer)
        self.qk_dim = self.pdim // 2  # Q/K dimension
        self.scale = self.pdim ** -0.5  # Scaling factor

        # QKV projection using 1x1 convolution
        self.qkv = nn.Conv2d(self.pdim, 2 * self.qk_dim + self.pdim, kernel_size=1)
        self.norm = nn.BatchNorm2d(self.pdim)  # Batch normalization
        self.act = nn.GELU()  # Activation function
        self.proj = nn.Conv2d(dim, dim, 1)  # Output projection

    def forward(self, x):
        B, C, H, W = x.shape
        # Channel separation (PSSA part and direct pass part)
        x1, x2 = torch.split(x, [self.pdim, self.dim - self.pdim], dim=1)

        # Process PSSA part
        x1 = self.norm(x1)
        qkv = self.qkv(x1)  # Generate Q/K/V matrices
        q, k, v = qkv.split([self.qk_dim, self.qk_dim, self.pdim], dim=1)

        # Flatten for attention calculation
        q, k, v = q.flatten(2), k.flatten(2), v.flatten(2)

        # Compute attention weights
        attn = (q.transpose(1, 2) @ k) * self.scale  # Scaled dot-product attention
        attn = attn.softmax(dim=-1)

        # Apply attention
        x1_out = (attn @ v.transpose(1, 2))  # B, HW, pdim
        x1_out = x1_out.reshape(B, H, W, self.pdim).permute(0, 3, 1, 2)  # B, pdim, H, W

        # Concatenate and project
        x_cat = torch.cat([x1_out, x2], dim=1)  # Combine processed and direct features
        x_cat = self.act(x_cat)
        return self.proj(x_cat)  # Final projection


class EFEB(torch.nn.Module):
    """
    Efficient Feature Extraction Block
    Implements depthwise separable convolution with residual connection
    """

    def __init__(self, dim):
        super(EFEB, self).__init__()
        self.dim = dim

        # Depthwise separable convolution branch
        self.dconv3 = nn.Sequential(
            Residual(nn.Sequential(
                nn.Conv2d(dim, dim, 3, 1, 1, groups=dim),  # Depthwise convolution
                nn.GELU(),
                nn.BatchNorm2d(dim)
            )),
            nn.Conv2d(dim, dim * 2, 1, 1, 0),  # Expand channels
            torch.nn.ReLU(),
            nn.Conv2d(dim * 2, dim, 1),  # Compress channels
            nn.BatchNorm2d(dim)
        )
        self.shortcut = nn.Conv2d(dim, dim, kernel_size=1, stride=1)  # Shortcut connection

    def forward(self, x):
        identity = x
        x = self.dconv3(x)
        return x + self.shortcut(identity)  # Residual connection


class EFAM(torch.nn.Module):
    """
    Efficient Feature Aggregation Module
    Combines EFEB and PSSA for enhanced feature extraction
    """

    def __init__(self, c1, c2, n=1, r=0.5, g=1, e=0.5):
        super().__init__()
        self.r = r  # Channel ratio for PSSA
        self.c = int(c2 * e)  # Hidden channels

        # Initial projection
        self.cv1 = nn.Conv2d(c1, 2 * self.c, 1, 1)
        self.cv2 = nn.Conv2d((2 + n) * self.c, c2, 1)  # Final projection

        # Multiple EFEB blocks
        self.m = nn.ModuleList([EFEB(self.c) for _ in range(n)])

        # PSSA module
        self.mixer = Residual(PSSA(self.c, r))

    def forward(self, x):
        # Initial split and projection
        y = list(self.cv1(x).chunk(2, 1))

        # Process through EFEB blocks
        y.extend(m(y[-1]) for m in self.m)

        # Apply PSSA
        y[-1] = self.mixer(y[-1])

        # Concatenate and project
        return self.cv2(torch.cat(y, 1))


if __name__ == '__main__':
    # Example usage
    efam = EFAM(256, 256)
    batch_size = 8
    input_tensor = torch.randn(batch_size, 256, 64, 64)
    output_tensor = efam(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)
