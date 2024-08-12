""" Copyright (c) 2023, Diego Páez
* Licensed under the MIT license

- CNN model
- CBAM Reference: https://arxiv.org/abs/1807.06521
    - Code Reference: https://tinyurl.com/25wyxnb8
    - Code Resnet CBAM: https://tinyurl.com/2b8zkaol

"""
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torchvision.ops import DeformConv2d

class ChannelAttention(nn.Module):
    """
    Channel Attention Module (CAM) for capturing channel-wise dependencies.

    Args:
        in_channels (int): Number of input channels.
        reduction_ratio (int, optional): Reduction ratio for the channel attention block. Default is 16.
    """

    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1, bias=False)
        init.kaiming_normal_(self.fc1.weight,nonlinearity='relu')
        init.kaiming_normal_(self.fc2.weight,nonlinearity='relu')

    def forward(self, x):
        """
        Forward pass through the Channel Attention Module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying channel attention.
        """
        avg_pool = self.avg_pool(x)
        max_pool = self.max_pool(x)
        avg_out = self.fc2(self.relu1(self.fc1(avg_pool)))
        max_out = self.fc2(self.relu1(self.fc1(max_pool)))
        channel_attention = torch.sigmoid(avg_out + max_out)

        return channel_attention

class SpatialAttention(nn.Module):
    """
    Spatial Attention Module (SAM) for capturing spatial dependencies.

    Args:
        kernel_size (int, optional): Size of the convolutional kernel for spatial attention. Default is 7.
    """

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()
        init.kaiming_normal_(self.conv.weight,nonlinearity='relu')

    def forward(self, x):
        """
        Forward pass through the Spatial Attention Module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying spatial attention.
        """
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        pooled_tensor = torch.cat([avg_pool, max_pool], dim=1)
        spatial_attention = self.sigmoid(self.conv(pooled_tensor))

        return spatial_attention

class CBAMBlock(nn.Module):
    """
    Convolutional Block Attention Module (CBAM) combining Channel Attention Module (CAM)
    and Spatial Attention Module (SAM).

    Args:
        in_channels (int): Number of input channels.
        reduction_ratio (int, optional): Reduction ratio for the channel attention block. Default is 16.
        kernel_size (int, optional): Size of the convolutional kernel for spatial attention. Default is 7.
    """

    def __init__(self, in_channels=1, reduction_ratio=16, kernel=7):
        super(CBAMBlock, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel)

    def forward(self, x):
        """
        Forward pass through the CBAM block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying CBAM.
        """
        residual = x
        out = self.channel_attention(x) * x     # Channel attention output
        out = self.spatial_attention(out) * out # Spatial attention - cbam output
        return out + residual

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation (SE) Block.

    Args:
        in_channels (int): Number of input channels.
        reduction_ratio (int, optional): Reduction ratio for the SE block. Default is 16.
    """
    def __init__(self, in_channels, reduction_ratio=16):
        """
        Initializes the Squeeze-and-Excitation block.

        Args:
            in_channels (int): Number of input channels.
            reduction_ratio (int, optional): Reduction ratio for the SE block. Default is 16.
        """
        super(SEBlock, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1)
        self.fc2 = nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1)
        init.kaiming_normal_(self.fc1.weight,nonlinearity='relu')
        init.kaiming_normal_(self.fc2.weight,nonlinearity='relu')

    def forward(self, x):
        """
        Forward pass through the Squeeze-and-Excitation block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying the SE block.
        """
        x_se = self.pool(x)
        x_se = F.relu(self.fc1(x_se))
        x_se = torch.sigmoid(self.fc2(x_se))
        return x * x_se
   
class ConvBlock(nn.Module):
    """ Convolutional Block with Conv -> BatchNorm -> ReLU """

    def __init__(self, kernel=1, in_channels=1, filters=1, strides=1, mu=1, epsilon=1, channels_last=False):
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
        self.kernel_size = kernel
        self.filters = filters
        self.strides = strides
        self.batch_norm_mu = mu
        self.batch_norm_epsilon = epsilon
        self.padding = (self.kernel_size - 1) // 2 # Calculate "same" padding
        self.activation = nn.ReLU()
        self.channels_last = channels_last
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=self.filters, 
                            kernel_size=self.kernel_size, 
                            stride=self.strides, 
                            padding= self.padding)
        init.kaiming_normal_(self.conv.weight,nonlinearity='relu')
        self.batch_norm = nn.BatchNorm2d(num_features=self.filters)
            
    def forward(self, inputs):
        """ Convolutional block with convolution op + batch normalization op.

        Args:
            inputs: input tensor to the block.

        Returns:
            output tensor.
        """
        if self.channels_last:
            inputs = inputs.permute(0, 3, 1, 2) # Convert NHWC to NCHW format
        #print(f'ConvBlock input.shape: {inputs.shape}')
        tensor = self.conv(inputs)
        #print(f'layer1 output.shape: {tensor.shape}')
        tensor = self.batch_norm(tensor)
        tensor = self.activation(tensor)
        
        if self.channels_last:
            tensor = tensor.permute(0, 2, 3, 1) # Convert NCHW to NHWC format
            
        return tensor

class DefConvBlock(nn.Module):
    """ Deformable Convolutional Block with DeformConv -> BatchNorm -> ReLU """

    def __init__(self, kernel=1, in_channels=1, filters=1, strides=1, mu=1, epsilon=1, channels_last=False):
        """ Initialize DeformableConvBlock.

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
        self.kernel_size = kernel
        self.filters = filters
        self.strides = strides
        self.batch_norm_mu = mu
        self.batch_norm_epsilon = epsilon
        self.padding = (self.kernel_size - 1) // 2 # Calculate "same" padding
        self.activation = nn.ReLU()
        self.channels_last = channels_last
        
        # Deformable convolution
        self.offsets = nn.Conv2d(in_channels=in_channels, out_channels=2*kernel*kernel, 
                                 kernel_size=kernel, stride=strides, padding=self.padding)
        self.deform_conv = DeformConv2d(in_channels=in_channels, out_channels=self.filters, 
                                        kernel_size=self.kernel_size, stride=self.strides, 
                                        padding=self.padding)
        init.kaiming_normal_(self.deform_conv.weight, nonlinearity='relu')
        
        self.batch_norm = nn.BatchNorm2d(num_features=self.filters)
            
    def forward(self, inputs):
        """ Deformable convolutional block with deformable convolution op + batch normalization op.

        Args:
            inputs: input tensor to the block.

        Returns:
            output tensor.
        """
        if self.channels_last:
            inputs = inputs.permute(0, 3, 1, 2) # Convert NHWC to NCHW format
        
        # Calculate offsets
        offsets = self.offsets(inputs)
        
        # Apply deformable convolution
        tensor = self.deform_conv(inputs, offsets)
        
        tensor = self.batch_norm(tensor)
        tensor = self.activation(tensor)
        
        if self.channels_last:
            tensor = tensor.permute(0, 2, 3, 1) # Convert NCHW to NHWC format
            
        return tensor

class SEConvBlock(nn.Module):
    """
    Squeeze-and-Excitation Convolution Block.
    """
    def __init__(self, kernel=1, in_channels=1, filters=1, strides=1, mu=1, epsilon=1, channels_last=False, reduction_ratio=16):
        """
        Initializes the Squeeze-and-Excitation Convolution block.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int, optional): Size of the convolutional kernel. Default is 3.
            stride (int, optional): Stride for the convolution. Default is 1.
            padding (int, optional): Padding for the convolution. Default is 1.
            reduction_ratio (int, optional): Reduction ratio for the SE block. Default is 16.
        """
        super(SEConvBlock, self).__init__()
        self.conv_block = ConvBlock(kernel, in_channels, filters, strides, mu, epsilon, channels_last)
        self.se_block = SEBlock(filters, reduction_ratio)

    def forward(self, inputs):
        """
        Forward pass through the Squeeze-and-Excitation Convolution block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying the Squeeze-and-Excitation Convolution block.
        """
        conv_output = self.conv_block(inputs)
        se_output = self.se_block(conv_output)
        return se_output

class CBAMConvBlock(nn.Module):
    """
    Convolutional Block with Convolutional Block Attention Module (CBAM).

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel (int, optional): Size of the convolutional kernel. Default is 3.
        stride (int, optional): Stride of the convolutional operation. Default is 1.
        padding (int, optional): Padding for the convolutional operation. Default is 1.
        reduction_ratio (int, optional): Reduction ratio for the channel attention block. Default is 16.
    """

    def __init__(self, kernel=1, in_channels=1, filters=1, strides=1, mu=1, epsilon=1, channels_last=False, reduction_ratio=16):
        super(CBAMConvBlock, self).__init__()

        self.conv_block = ConvBlock(kernel, in_channels, filters, strides,mu, epsilon, channels_last)
        self.cbam_block = CBAMBlock(filters, reduction_ratio)

    def forward(self, x):
        """
        Forward pass through the CBAM Convolutional Block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying CBAM Convolutional Block.
        """
        # check if the channel is equal to 3 
        
        ##print(f'CBAMConvBlock input.shape: {x.shape}')
        x = self.conv_block(x)
        ##print(f'layer1 output.shape: {x.shape}')
        x = self.cbam_block(x)
        ##print(f'layer2 output.shape: {x.shape}')
        return x

class DWConvBlock(nn.Module):
    """ Depth Wise Separable Convolutional Block with Conv -> DepthwiseConv -> BatchNorm -> ReLU """

    def __init__(self, kernel=1, in_channels=1, filters=1, strides=1, mu=1, epsilon=1, channels_last=False):
        """ Initialize DepthwiseSeparableConvBlock.

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
        self.kernel_size = kernel
        self.filters = filters
        self.strides = strides
        self.batch_norm_mu = mu
        self.batch_norm_epsilon = epsilon
        self.padding = (self.kernel_size - 1) // 2 # Calculate "same" padding
        self.activation = nn.ReLU()
        self.channels_last = channels_last

        # Depthwise Separable Convolution: Depthwise Convolution + Pointwise Convolution
        self.depthwise_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=self.kernel_size,
                                        stride=self.strides, padding=self.padding, groups=in_channels)
        self.pointwise_conv = nn.Conv2d(in_channels=in_channels,out_channels= self.filters, kernel_size=1)

        init.kaiming_normal_(self.depthwise_conv.weight,nonlinearity='relu')
        init.kaiming_normal_(self.pointwise_conv.weight,nonlinearity='relu')

        self.batch_norm = nn.BatchNorm2d(num_features=self.filters)
            
    def forward(self, inputs):
        """ Depthwise Separable Convolutional block with depthwise convolution + pointwise convolution
            + batch normalization + ReLU activation.

        Args:
            inputs: input tensor to the block.

        Returns:
            output tensor.
        """
        if self.channels_last:
            inputs = inputs.permute(0, 3, 1, 2) # Convert NHWC to NCHW format
        
        tensor = self.depthwise_conv(inputs)
        tensor = self.pointwise_conv(tensor)
        tensor = self.batch_norm(tensor)
        tensor = self.activation(tensor)
        
        if self.channels_last:
            tensor = tensor.permute(0, 2, 3, 1) # Convert NCHW to NHWC format
            
        return tensor

class MBConv(nn.Module):
    """
    MobileNetV3 Bottleneck Block with Squeeze-and-Excitation (SE) Block.
    """
    def __init__(self,kernel=1, in_channels=1, filters=1, strides=1, mu=1, epsilon=1, expand_ratio=6, reduction_ratio=16):
        super(MBConv, self).__init__()
        mid_channels = in_channels * expand_ratio

        # Expand
        self.expand_conv = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.expand_bn = nn.BatchNorm2d(mid_channels)
        self.expand_relu = nn.ReLU6(inplace=True)

        # Depthwise
        self.depthwise_conv = nn.Conv2d(mid_channels, mid_channels, kernel_size=kernel, stride=strides, padding=(kernel - 1) // 2, groups=mid_channels, bias=False)
        self.depthwise_bn = nn.BatchNorm2d(mid_channels)
        self.depthwise_relu = nn.ReLU6(inplace=True)

        # Squeeze-and-Excitation
        self.se_block = SEBlock(mid_channels, reduction_ratio)

        # Project - Pointwise
        self.project_conv = nn.Conv2d(mid_channels, filters, kernel_size=1, bias=False)
        self.project_bn = nn.BatchNorm2d(filters)
        
        self.use_residual = (in_channels == filters and strides == 1)

    def forward(self, x):
        identity = x

        # Expand
        out = self.expand_conv(x)
        out = self.expand_bn(out)
        out = self.expand_relu(out)

        # Depthwise
        out = self.depthwise_conv(out)
        out = self.depthwise_bn(out)
        out = self.depthwise_relu(out)

        # Squeeze-and-Excitation
        out = self.se_block(out) # skip is inside the block

        # Project - project the features back to the original number of channels
        out = self.project_conv(out)
        out = self.project_bn(out)

        if self.use_residual:
            out = out + identity

        return out
    
class MBConv_V2(nn.Module):
    """
    MobileNetV2 Bottleneck Block
    Ref 1: Effective Data Augmentation and Training Techniques for Improving Deep Learning in Plant Leaf Disease Recognition
    link: https://tinyurl.com/2axqr4cl
    """
    def __init__(self,kernel=1, in_channels=1, filters=1, strides=1, mu=1, epsilon=1, expand_ratio=6, reduction_ratio=16):
        super(MBConv_V2, self).__init__()
        mid_channels = in_channels * expand_ratio

        # Expand
        self.expand_conv = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.expand_bn = nn.BatchNorm2d(mid_channels)
        self.expand_relu = nn.ReLU6(inplace=True)

        # Depthwise
        self.depthwise_conv = nn.Conv2d(mid_channels, mid_channels, kernel_size=kernel, stride=strides, padding=(kernel - 1) // 2, groups=mid_channels, bias=False)
        self.depthwise_bn = nn.BatchNorm2d(mid_channels)
        self.depthwise_relu = nn.ReLU6(inplace=True)

        # Project
        self.project_conv = nn.Conv2d(mid_channels, filters, kernel_size=1, bias=False)
        self.project_bn = nn.BatchNorm2d(filters)
        
        self.use_residual = (in_channels == filters and strides == 1)

    def forward(self, x):
        identity = x

        # Expand
        out = self.expand_conv(x)
        out = self.expand_bn(out)
        out = self.expand_relu(out)

        # Depthwise
        out = self.depthwise_conv(out)
        out = self.depthwise_bn(out)
        out = self.depthwise_relu(out)

        # Project
        out = self.project_conv(out)
        out = self.project_bn(out)

        if self.use_residual:
            out = out + identity

        return out
    
class MBConv_EPPGA(nn.Module):
    """
    EPPGA block structure: MobileNetV3 style Bottleneck Block with Squeeze-and-Excitation (SE) Block.
    Ref: An evolutionary neural architecture search method based on performance prediction and weight inheritance
    link: https://www.sciencedirect.com/science/article/pii/S0020025524003797
    """
    def __init__(self,kernel=1, in_channels=1, filters=1, strides=1, mu=1, epsilon=1, expand_ratio=6, reduction_ratio=16):
        super(MBConv_EPPGA, self).__init__()
        mid_channels = in_channels * expand_ratio

        # Expand
        self.expand_conv = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.expand_bn = nn.BatchNorm2d(mid_channels)
        self.expand_relu = nn.ReLU6(inplace=True)

        # Depthwise
        self.depthwise_conv = nn.Conv2d(mid_channels, mid_channels, kernel_size=kernel, stride=strides, padding=(kernel - 1) // 2, groups=mid_channels, bias=False)
        self.depthwise_bn = nn.BatchNorm2d(mid_channels)
        self.depthwise_relu = nn.ReLU6(inplace=True)
        
        # Pointwise - Project
        self.pointwise_conv = nn.Conv2d(mid_channels, filters, kernel_size=1, bias=False)
        self.pointwise_bn = nn.BatchNorm2d(filters)

        # Squeeze-and-Excitation
        self.se_block = SEBlock(filters, reduction_ratio)
        
        self.use_residual = (in_channels == filters and strides == 1)

    def forward(self, x):
        identity = x

        # Expand
        out = self.expand_conv(x)
        out = self.expand_bn(out)
        out = self.expand_relu(out)

        # Depthwise
        out = self.depthwise_conv(out)
        out = self.depthwise_bn(out)
        out = self.depthwise_relu(out)
        
        # Pointwise - to project the features back to the original number of channels
        out = self.pointwise_conv(out)
        out = self.pointwise_bn(out)

        # Squeeze-and-Excitation
        out = self.se_block(out) # skip is inside the block

        if self.use_residual:
            out = out + identity

        return out
    
class ResidualV1(nn.Module):
    def __init__(self, in_channels=1, kernel=1, filters=1, strides=1, mu=1, epsilon=1, channels_last=False):
        super().__init__()
        self.kernel_size = kernel
        self.filters = filters
        self.strides = strides
        self.batch_norm_mu = mu
        self.batch_norm_epsilon = epsilon
        self.channels_last = channels_last
        self.padding = (self.kernel_size - 1) // 2 # Calculate "same" padding
        
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=filters, 
                               kernel_size=self.kernel_size,stride=strides, 
                               padding=self.padding, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=self.filters)
        self.conv2 = nn.Conv2d(in_channels=filters, out_channels=filters, 
                               kernel_size=self.kernel_size, stride=1, 
                               padding=self.padding, bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=self.filters)
        
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, filters, kernel_size=1, padding=0, stride=strides, bias=False),
            nn.BatchNorm2d(num_features=self.filters)
        ) if strides != 1 or in_channels != filters else nn.Identity()
        
        init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.conv2.weight, mode='fan_out', nonlinearity='relu')
        if not isinstance(self.projection, nn.Identity):
            init.kaiming_normal_(self.projection[0].weight, mode='fan_out', nonlinearity='relu')

    def forward(self, inputs):
        if self.channels_last:
            inputs = inputs.permute(0, 3, 1, 2) # Convert NHWC to NCHW format
        
        #print(f'inputs.shape V1: {inputs.shape}')            
        tensor = self.conv1(inputs)
        tensor = self.bn1(tensor)
        tensor = F.relu(tensor)
        #print(f'tensor.shape Layer 1: {tensor.shape}')
            
        tensor = self.conv2(tensor)
        tensor = self.bn2(tensor)
              
        #print(f'tensor.shape Layer 2: {tensor.shape}')
              
        tensor += self.projection(inputs)
        tensor = F.relu(tensor)
        
        if self.channels_last:
            tensor = tensor.permute(0, 2, 3, 1) # Convert NCHW to NHWC format
        
        #print(f'output.shape: {tensor.shape}')
        return tensor

class ResidualV1CBAM(nn.Module):
    """ Residual Block with Conv -> BatchNorm -> ReLU -> Conv -> BatchNorm -> CBAM -> Add -> ReLU """
    def __init__(self, in_channels=1, kernel=1, filters=1, strides=1, mu=1, epsilon=1, channels_last=False):
        """ Initialize ResidualV1.

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
        self.kernel_size = kernel
        self.filters = filters
        self.strides = strides
        self.batch_norm_mu = mu
        self.batch_norm_epsilon = epsilon
        self.channels_last = channels_last
        self.padding = (self.kernel_size - 1) // 2 # Calculate "same" padding
        
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=filters, 
                              kernel_size=self.kernel_size,stride=strides, 
                              padding= self.padding ,bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=self.filters)
        self.conv2 = nn.Conv2d(in_channels=filters, out_channels=filters, 
                              kernel_size=self.kernel_size,stride=strides, 
                              padding= self.padding ,bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=self.filters)
        
        init.kaiming_normal_(self.conv1.weight,nonlinearity='relu')  # He Normal initialization
        init.kaiming_normal_(self.conv2.weight,nonlinearity='relu')  # He Normal initialization
        
        self.channel_attention = ChannelAttention(filters)
        self.spatial_attention = SpatialAttention()
                # Shortcut connection
        if strides != 1 or in_channels != filters:
            #print("Shortcut connection")
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, filters, kernel_size=1, padding=0,stride=strides, bias=False),
                nn.BatchNorm2d(num_features=self.filters)
            )
        else:
            self.shortcut = nn.Identity()
        

    def forward(self, inputs):
        """ Residual block with convolution op + batch normalization op + add op.

        Args:
            inputs: input tensor to the
        Returns:
            output tensor.
        """
        if self.channels_last:
            inputs = inputs.permute(0, 3, 1, 2) # Convert NHWC to NCHW format
        
        tensor = self.conv1(inputs)
        tensor = self.bn1(tensor)
        tensor = F.relu(tensor)
            
        tensor = self.conv2(tensor)
        tensor = self.bn2(tensor)
        
        # Apply CBAM
        tensor = self.channel_attention(tensor) * tensor
        tensor = self.spatial_attention(tensor) * tensor

        
        tensor = tensor + self.shortcut(inputs)
        tensor = F.relu(tensor)
        if self.channels_last:
            tensor = tensor.permute(0, 2, 3, 1) # Convert NCHW to NHWC format

        return tensor
    
class ResidualV1Pr(nn.Module):
    """ Residual V1 block with projection shortcut """
    def __init__(self, in_channels=1, kernel=1, filters=1, strides=1, mu=1, epsilon=1, channels_last=False):
        """ Initialize ResidualV1.

        Args:
            in_channels : int
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
        self.kernel_size = kernel
        self.filters = filters
        self.strides = strides
        self.batch_norm_mu = mu
        self.batch_norm_epsilon = epsilon
        self.channels_last = channels_last
        self.padding = (self.kernel_size - 1) // 2 # Calculate "same" padding
        
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=filters, 
                              kernel_size=self.kernel_size,stride=strides, 
                              padding= self.padding ,bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=self.filters)
        self.conv2 = nn.Conv2d(in_channels=filters, out_channels=filters, 
                              kernel_size=self.kernel_size,stride=strides, 
                              padding= self.padding ,bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=self.filters)
        
        init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')  # He Normal initialization
        init.kaiming_normal_(self.conv2.weight, mode='fan_out', nonlinearity='relu')  # He Normal initialization
        
        # Shortcut connection
        if strides != 1 or in_channels != filters:
            #print("Shortcut connection")
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, filters, kernel_size=1, padding=0,stride=strides, bias=False),
                nn.BatchNorm2d(num_features=self.filters)
            )
        else:
            self.shortcut = nn.Identity()


    def forward(self, inputs):
        """ Residual block with convolution op + batch normalization op + add op.

        Args:
            inputs: input tensor to the
        Returns:
            output tensor.
        """
        if self.channels_last:
            inputs = inputs.permute(0, 3, 1, 2) # Convert NHWC to NCHW format
        
        #print(f'inputs.shape: {inputs.shape}')            
        tensor = self.conv1(inputs)
        tensor = self.bn1(tensor)
        tensor = F.relu(tensor)
        #print(f'tensor.shape Layer 1: {tensor.shape}')
            
        tensor = self.conv2(tensor)
        tensor = self.bn2(tensor)
              
        #print(f'tensor.shape Layer 2: {tensor.shape}')
              
        tensor = tensor + self.shortcut(inputs)
        tensor = F.relu(tensor)
        #print(f'output.shape: {tensor.shape}')
        if self.channels_last:
            tensor = tensor.permute(0, 2, 3, 1) # Convert NCHW to NHWC format

        return tensor
        
class MaxPooling(nn.Module):
    """ Max Pooling layer """

    def __init__(self, kernel=1, strides=1, channels_last=False):
        """ Initialize MaxPooling.

        Args:
            kernel : int
                Represents the size of the pooling window (3 means [3,3])
            strides : int
                Represents the stride of the pooling window (3 means [3,3])
        """
        super().__init__()
        self.kernel = kernel
        self.strides = strides
        self.padding = 0 # 'valid' no padding
        self.channels_last = channels_last

        self.max_pool = nn.MaxPool2d(kernel_size=self.kernel, 
                                     stride=self.strides, 
                                     padding=self.padding)

    def forward(self, inputs):
        """ Max Pooling layer.

        Args:
            inputs: input tensor to the block.

        Returns:
            output tensor.
        """
        if self.channels_last:
            inputs = inputs.permute(0, 3, 1, 2) # Convert NHWC to NCHW format
        
        # check of the image size    
        if inputs.shape[2] >= self.kernel and inputs.shape[3] >= self.kernel:
            tensor = self.max_pool(inputs)
        else:
            ##print("Warning: MaxPooling layer not applied because the image size is smaller than the kernel size")
            return inputs
        
        if self.channels_last:
            tensor = tensor.permute(0, 2, 3, 1) # Convert NCHW to NHWC format

        return tensor

class AvgPooling(nn.Module):
    """ Average Pooling layer """

    def __init__(self, kernel=1, strides=1, channels_last=False):
        """ Initialize AvgPooling.

        Args:
            kernel : int
                Represents the size of the pooling window (3 means [3,3])
            strides : int
                Represents the stride of the pooling window (3 means [3,3])
        """
        super().__init__()
        self.kernel = kernel
        self.strides = strides
        self.padding = 0 # 'valid' no padding
        self.channels_last = channels_last

        self.avg_pool = nn.AvgPool2d(kernel_size=self.kernel, 
                                     stride=self.strides, 
                                     padding=self.padding)

    def forward(self, inputs):
        """ Average Pooling layer.

        Args:
            inputs: input tensor to the block.

        Returns:
            output tensor.
        """
        ##print(f'inputs.shape avg: {inputs.shape}')
        if self.channels_last:
            inputs = inputs.permute(0, 3, 1, 2) # Convert NHWC to NCHW format
        
        # check of the image size    
        if inputs.shape[2] >= self.kernel and inputs.shape[3] >= self.kernel:
            tensor = self.avg_pool(inputs)
        else:
            #print("Warning: AvgPooling layer not applied because the image size is smaller than the kernel size")
            return inputs
            #return inputs.permute(0, 2, 3, 1) # Convert NCHW to NHWC format
        
        if self.channels_last:
            tensor = tensor.permute(0, 2, 3, 1) # Convert NCHW to NHWC format

        return tensor
    
class FullyConnected(nn.Module):
    def __init__(self,input_features=1, units=1):
        """ Initialize FullyConnected.

        Args:
            inputs_features : int
                Represents the number of inputs features of the layer
            units : int
                Represents the number of neurons in the layer

        """
        super().__init__()
        self.inputs__features = input_features
        self.units = units                
        self.fc = nn.Linear(in_features=self.inputs__features,
                            out_features=self.units)
        init.kaiming_normal_(self.fc.weight,nonlinearity='relu')                   
        
    def forward(self, inputs):
        """ FullyConnected layer.

        Args:
            inputs: input tensor to the block.

        Returns:
            output tensor.
        """
        tensor = self.fc(inputs)
        return tensor
    
class NoOp(nn.Module):
    """ NoOp layer.
    """
    pass

functions_dict = {'ConvBlock': ConvBlock,
                  'DWConvBlock': DWConvBlock,
                  'SEConvBlock': SEConvBlock,
                  'MBConv': MBConv,
                  'MBConvV2': MBConv_V2,
                  'MBEPPGA': MBConv_EPPGA,
                  'ResidualV1': ResidualV1,
                  'ResidualV1Pr': ResidualV1Pr,
                  'CBAMConvBlock': CBAMConvBlock,
                  'ResidualV1CBAM': ResidualV1CBAM,
                  'CBAMBlock' : CBAMBlock,
                  'MaxPooling': MaxPooling,
                  'AvgPooling': AvgPooling,
                  'FullyConnected': FullyConnected,
                  'no_op': NoOp}
 

class NetworkGraph(nn.Module):
    def __init__(self, num_classes, mu=0.9, epsilon=2e-5, in_channels=3):
        """ Initialize NetworkGraph.

        Args:
            num_classes: int 
                number of classes for classification model.
            mu: float
                batch normalization decay; default = 0.9
            epsilon: float 
             
                   batch normalization epsilon; default = 2e-5.
        Returns:
            output logits tensor.
        """
        super().__init__()

        self.num_classes = num_classes
        self.mu = mu
        self.epsilon = epsilon
        self.in_channels = in_channels
        #self.layer_dict = nn.ModuleDict()
        
    def create_functions(self, net_list, fn_dict, cbam=False):
        """ Generate all possible functions from functions descriptions in *self.fn_dict*.

        Args:
            fn_dict: dict with definitions of the functions (name and parameters);
                format --> {'fn_name': ['FNClass', {'param1': value1, 'param2': value2}]}.
        """
        in_channels = self.in_channels
        self.layers = []
        if cbam:
            # Fix the first layer with a 1x1 convolution 
            net_list.insert(0, 'conv_1_1_32')
            conv_1_1_info = {'conv_1_1_32': {'function': 'ConvBlock', 'params': {'kernel': 1, 'strides': 1, 'filters': 32}}}
            fn_dict.update(conv_1_1_info)

        for name in net_list:
            parameters = fn_dict[name]
            if parameters['function'] == 'NoOp':
                continue
            if parameters['function'] in ['ConvBlock', 'DWConvBlock', 'SEConvBlock', 'ResidualV1CBAM','MBConv','MBConvV2','MBEPPGA','ResidualV1', 'ResidualV1Pr']:
                parameters['params']['mu'] = self.mu
                parameters['params']['epsilon'] = self.epsilon
                parameters['params']['in_channels'] = in_channels
                in_channels = parameters['params']['filters']
            
            if parameters['function'] in ['CBAMBlock']:
                parameters['params']['in_channels'] = in_channels
   
            self.layers.append(functions_dict[parameters['function']](**parameters['params']))
        self.model = nn.Sequential(*self.layers)
        self.fc = None


    def forward(self, inputs, debug=False):
        """ Create a PyTorch network from a list of layer names.

        Args:
            net_list: list of layer names, representing the network layers.
            inputs: input tensor to the network.

        Returns:
            logits tensor.
        """
        #print(f'inputs.shape: {inputs.shape}')
        if debug:
            for f in self.layers:
                print(f'f: {f}')
                inputs = f(inputs)
                print(f'layer output.shape: {inputs.shape}')
        else:
            inputs = self.model(inputs) # BUG: the model is not being updated when the dataset is not downloaded before running the training
            #print(f'layer output.shape: {inputs.shape}')

        if self.fc is None:
            batch_size, num_features, height, width = inputs.size()
            num_flat_features = num_features * height * width
            self.fc = FullyConnected(input_features=num_flat_features, units=self.num_classes)

        batch_size = inputs.size(0)
        inputs = inputs.reshape(batch_size, -1)
        #print('FullyConnected')
        #print(f'layer output.shape: {inputs.shape}')
        logits = self.fc(inputs)

        return logits