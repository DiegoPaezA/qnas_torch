""" Copyright (c) 2023, Diego Páez
* Licensed under the MIT license

- CNN model

"""
import torch
import torch.nn as nn
import torch.nn.init as init

class ConvBlock(nn.Module):
    """ Convolutional Block with Conv -> BatchNorm -> ReLU """

    def __init__(self,kernel, filters, strides, mu, epsilon, in_channel=3, channels_last=True):
        """ Initialize ConvBlock.

        Args:
            in_channel : int
                Represents the number of channels in the input image (default 3 for RGB)
            kernel : int
                Represents the size of the convolutional window (3 means [3,3])
            filters : int
                Number of filters
            strides : int
                Represents the stride of the convolutional window (3 means [3,3])
            mu : float
                Mean for the batch normalization
            epsilon : float
                Epsilon for the batch normalization
        """
        super().__init__()
        self.in_channels = in_channel
        self.kernel_size = kernel
        self.filters = filters
        self.strides = strides
        self.batch_norm_mu = mu
        self.batch_norm_epsilon = epsilon
        self.padding = (self.kernel_size - 1) // 2 # Calculate "same" padding
        self.activation = nn.ReLU()
        self.channels_last = channels_last

        # PyTorch does not require specifying the activation function and initializer separately.
        self.conv = nn.Conv2d(in_channels=self.in_channels, out_channels=self.filters, 
                              kernel_size=self.kernel_size, 
                              stride=self.strides, 
                              padding=self.padding)
        init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')  # He Normal initialization
        self.batch_norm = nn.BatchNorm2d(num_features=self.filters,
                                         momentum=self.batch_norm_mu, 
                                         eps=self.batch_norm_epsilon)

    def forward(self, inputs):
        """ Convolutional block with convolution op + batch normalization op.

        Args:
            inputs: input tensor to the block.

        Returns:
            output tensor.
        """
        if self.channels_last:
            inputs = inputs.permute(0, 3, 1, 2) # Convert NHWC to NCHW format

        
        tensor = self.conv(inputs)
        tensor = self.batch_norm(tensor)
        tensor = self.activation(tensor)
        
        if self.channels_last:
            tensor = tensor.permute(0, 2, 3, 1) # Convert NCHW to NHWC format
            
        return tensor
