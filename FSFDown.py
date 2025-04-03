import torch
import torch.nn as nn
from pytorch_wavelets import DWTForward
"""
   FSFDown module is designed for downsampling and feature extraction in a neural network.
   It takes an input tensor, splits it into two parts, and processes each part differently.
   One part is downsampled using space to depth and the other part is processed using wavelet transform.
   Finally, the processed features are concatenated and further compressed.
   """
class FSFDown(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(FSFDown, self).__init__()
        # Initialize the wavelet transform with J=1 level, zero-padding mode, and Haar wavelet
        self.wt = DWTForward(J=1, mode='zero', wave='haar')

        # First depthwise separable convolution block for the first part of the input
        self.dconv1 = nn.Sequential(
            nn.Sequential(
                # Depthwise convolution
                nn.Conv2d(in_ch * 2, in_ch * 2, 3, 1, 1, groups=in_ch * 2),
                # GELU activation function
                nn.GELU(),
                # Batch normalization
                nn.BatchNorm2d(in_ch * 2)
            ),
            # Pointwise convolution to reduce the number of channels to out_ch
            nn.Conv2d(in_ch * 2, out_ch, 1)
        )

        # Second depthwise separable convolution block for the second part of the input
        self.dconv2 = nn.Sequential(
            nn.Sequential(
                # Depthwise convolution
                nn.Conv2d(in_ch * 2, in_ch * 2, 3, 1, 1, groups=in_ch * 2),
                # GELU activation function
                nn.GELU(),
                # Batch normalization
                nn.BatchNorm2d(in_ch * 2)
            ),
            # Pointwise convolution to reduce the number of channels to out_ch
            nn.Conv2d(in_ch * 2, out_ch, 1)
        )

        # Final 1x1 convolution to further reduce the number of channels after concatenation
        self.conv1 = nn.Conv2d(out_ch * 2, out_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # Split the input tensor into two parts along the channel dimension
        x1, x2 = x.chunk(2, dim=1)

        # Downsample the first part by slicing the spatial dimensions
        x1 = torch.cat([x1[..., ::2, ::2], x1[..., 1::2, ::2], x1[..., ::2, 1::2], x1[..., 1::2, 1::2]], 1)
        # Pass the downsampled first part through the first depthwise separable convolution block
        x1 = self.dconv1(x1)

        # Apply wavelet transform to the second part
        yL, yH = self.wt(x2)
        # Extract the high-frequency components
        y_HL = yH[0][:, :, 0, :, :]
        y_LH = yH[0][:, :, 1, :, :]
        y_HH = yH[0][:, :, 2, :, :]
        # Concatenate the low-frequency component and high-frequency components
        x2 = torch.cat([yL, y_HL, y_LH, y_HH], dim=1)
        # Pass the concatenated components through the second depthwise separable convolution block
        x2 = self.dconv2(x2)

        # Concatenate the processed first and second parts along the channel dimension
        x = torch.cat([x1, x2], 1)
        # Pass the concatenated tensor through the final 1x1 convolution
        return self.conv1(x)

if __name__ == '__main__':
    # Initialize the FSFDown module with input and output channels set to 256
    FSFDown_module = FSFDown(256, 256)
    # Create a random input tensor with batch size 8, 256 channels, and spatial dimensions 64x64
    batch_size = 8
    input_tensor = torch.randn(batch_size, 256, 64, 64)
    # Pass the input tensor through the FSFDown module
    output_tensor = FSFDown_module(input_tensor)
    # Print the input and output shapes
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)
