import torch
import torch.nn as nn

class Mix(nn.Module):
    """
    Mixing module that combines two feature maps with a learnable mixing factor.
    The mixing factor is calculated using a sigmoid function applied to a learnable parameter.
    """
    def __init__(self, m=-0.80):
        super(Mix, self).__init__()
        # Learnable parameter for mixing factor
        w = torch.nn.Parameter(torch.FloatTensor([m]), requires_grad=True)
        self.w = w
        # Sigmoid activation for generating the mixing factor
        self.mix_block = nn.Sigmoid()

    def forward(self, fea1, fea2):
        # Calculate the mixing factor
        mix_factor = self.mix_block(self.w)
        # Combine the two feature maps using the mixing factor
        out = fea1 * mix_factor.expand_as(fea1) + fea2 * (1 - mix_factor.expand_as(fea2))
        return out

class ZPool(nn.Module):
    """
    ZPool module that concatenates the max and mean values along the channel dimension.
    This operation helps in capturing both the maximum and average responses in the feature map.
    """
    def forward(self, x):
        # Concatenate the max and mean values along the channel dimension
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)

class AttentionGate(nn.Module):
    """
    Attention gate module that compresses the input feature map using ZPool and then applies a 1x1 convolution.
    This module is used to generate attention maps.
    """
    def __init__(self, k):
        super(AttentionGate, self).__init__()
        # Compression operation using ZPool
        self.compress = ZPool()
        # 1x1 convolution to generate the attention map
        self.conv1 = nn.Conv2d(2, 1, kernel_size=k, stride=1, padding=int(k / 2))

    def forward(self, x):
        # Compress the input feature map
        x_compress = self.compress(x)
        # Generate the attention map
        x_out = self.conv1(x_compress)
        return x_out

class SimAM(torch.nn.Module):
    """
    SimAM module that applies a self-attention mechanism to the input feature map.
    It computes an attention weight for each element in the feature map based on its deviation from the mean.
    """
    def __init__(self, e_lambda=1e-4):
        super(SimAM, self).__init__()
        # Sigmoid activation for generating attention weights
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
        # Calculate the squared deviation from the mean
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        # Compute the attention weight
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5
        # Apply the attention weight to the input feature map
        return x * self.activaton(y)

class ETA(nn.Module):
    """
    ETA (Enhanced Triple Attention) module that combines multiple attention mechanisms.
    It applies attention in different permutations of the input feature map and then averages the results.
    """
    def __init__(self):
        super(ETA, self).__init__()
        # Attention gates for different permutations
        self.cw = AttentionGate(7)
        self.hc = AttentionGate(7)
        self.hw = AttentionGate(7)
        # SimAM self-attention module
        self.simam = SimAM()

    def forward(self, x):
        # First permutation: swap channel and height dimensions
        x_perm1 = x.permute(0, 2, 1, 3).contiguous()
        # Apply attention and sigmoid activation
        x_out1 = self.simam((self.cw(x_perm1))).sigmoid()
        # Permute back to the original dimension order
        x_out11 = x_out1.permute(0, 2, 1, 3).contiguous()

        # Second permutation: swap channel and width dimensions
        x_perm2 = x.permute(0, 3, 2, 1).contiguous()
        # Apply attention and sigmoid activation
        x_out2 = self.simam((self.hc(x_perm2))).sigmoid()
        # Permute back to the original dimension order
        x_out21 = x_out2.permute(0, 3, 2, 1).contiguous()

        # Apply attention and sigmoid activation without permutation
        x_out = self.simam(self.hw(x)).sigmoid()

        # Average the attention maps
        x_out = 1 / 3 * (x_out + x_out11 + x_out21)
        return x_out

if __name__ == '__main__':
    # Initialize the ETA module
    ETA1 = ETA()
    # Create a random input tensor
    batch_size = 8
    input_tensor = torch.randn(batch_size, 256, 64, 64)
    # Forward pass through the ETA module
    output_tensor = ETA1(input_tensor)
    # Print the input and output shapes
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)
