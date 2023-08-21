""" Copyright (c) 2023, Diego Páez
* Licensed under the MIT license

- CNN model

"""
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

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

class ResidualV1(nn.Module):
    """ Residual Block with Conv -> BatchNorm -> ReLU -> Conv -> BatchNorm -> Add -> ReLU """
    def __init__(self, kernel, filters, strides, mu, epsilon, channels_last=True):
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
        
        self.batch_norm = nn.BatchNorm2d(num_features=self.filters,
                                         momentum=self.batch_norm_mu, 
                                         eps=self.batch_norm_epsilon)

    def forward(self, inputs):
        """ Residual block with convolution op + batch normalization op + add op.

        Args:
            inputs: input tensor to the
        Returns:
            output tensor.
        """
        if self.channels_last:
            inputs = inputs.permute(0, 3, 1, 2) # Convert NHWC to NCHW format
        
        print(f'inputs.shape: {inputs.shape}')            
        tensor = self._conv_fixed_pad(inputs=inputs, kernel_size=self.kernel_size, 
                                      filters=self.filters, strides=self.strides)
        tensor = self.batch_norm(tensor)
        tensor = F.relu(tensor)
        print(f'tensor.shape Layer 1: {tensor.shape}')
            
        tensor = self._conv_fixed_pad(inputs=tensor, kernel_size=self.kernel_size, 
                                      filters=self.filters, strides=1)
        tensor = self.batch_norm(tensor)
        
      
        print(f'tensor.shape Layer 2: {tensor.shape}')
        inputs, tensor = pad_features([inputs, tensor], channels_last=False)
        
        print(f'inputs.shape Layer 2: {inputs.shape}')
        print(f'tensor.shape Pad: {tensor.shape}')
        
        
        tensor = tensor + inputs
        
        tensor = F.relu(tensor)
        
        if self.channels_last:
            tensor = tensor.permute(0, 2, 3, 1) # Convert NCHW to NHWC format

        return tensor
    
    def _conv_fixed_pad(self, inputs, kernel_size, filters, strides):
        """ Convolution operation for residual unit wrapper. There is no bias and padding is
            determined by *strides*. When *strides* = 1, SAME padding is applied. Otherwise,
            the input is explicitly padded in the spatial dimensions before convolution, based
            only on kernel size.

        Args:
            inputs: input tensor.
            kernel_size: int representing the size of the convolution window (3 means [3, 3]).
            filters: int representing the number of filters in the convolution.
            strides: (int) specifies the strides of the convolution operation (1 means [1, 1]).

        Returns:
            output tensor.
        """
        padding = (kernel_size - 1) // 2  # Calculate "same" padding
        if strides > 1:
            print(f'inputs.shape 1 stride: {inputs.shape}')
            pad = kernel_size - 1
            pad_beg = pad // 2
            pad_end = pad - pad_beg
            inputs = F.pad(inputs, (pad_beg, pad_end, pad_beg, pad_end))
            print(f'inputs.shape 2 stride: {inputs.shape}')
            # Set padding to "valid" mode
            padding = 0

        conv = nn.Conv2d(in_channels=inputs.shape[1], out_channels=filters, 
                              kernel_size=kernel_size, 
                              stride=strides, 
                              padding= padding,
                              bias=False)
        init.kaiming_normal_(conv.weight, mode='fan_out', nonlinearity='relu')  # He Normal initialization

        return conv(inputs)

class ResidualV1Pr(ResidualV1):
    """ Residual V1 block with projection shortcut """
    def _projection(self, inputs, filters):
        conv2d = nn.Conv2d(in_channels=inputs.size(1), out_channels=filters, 
                           kernel_size=1, stride=1, padding=0, bias=False)
        init.kaiming_normal_(conv2d.weight, mode='fan_out', nonlinearity='relu')  # He Normal initialization
        
        return conv2d(inputs)

    def forward(self, inputs):
        """ Residual unit with 2 sub layers, using Plan A for shortcut connection.

        Args:
            inputs: input tensor to the block.

        Returns:
            output tensor.
        """
        if self.channels_last:
            inputs = inputs.permute(0, 3, 1, 2) # Convert NHWC to NCHW format
        
        shortcut = self._projection(inputs, filters=self.filters)
        shortcut = self.batch_norm(shortcut)

        tensor = self._conv_fixed_pad(inputs=inputs, kernel_size=self.kernel_size,
                                      filters=self.filters, strides=self.strides)
        tensor = self.batch_norm(tensor)
        tensor = F.relu(tensor)

        tensor = self._conv_fixed_pad(inputs=tensor, kernel_size=self.kernel_size,
                                      filters=self.filters, strides=1)
        tensor = self.batch_norm(tensor)

        tensor = tensor + shortcut
        
        tensor = F.relu(tensor)
        
        if self.channels_last:
            tensor = tensor.permute(0, 2, 3, 1) # Convert NCHW to NHWC format

        return tensor

class MaxPooling(nn.Module):
    """ Max Pooling layer """

    def __init__(self, kernel, strides, channels_last=True):
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
            #print("Warning: MaxPooling layer not applied because the image size is smaller than the kernel size")
            return inputs.permute(0, 2, 3, 1) # Convert NCHW to NHWC format
        
        if self.channels_last:
            tensor = tensor.permute(0, 2, 3, 1) # Convert NCHW to NHWC format

        return tensor

class AvgPooling(nn.Module):
    """ Average Pooling layer """

    def __init__(self, kernel, strides, channels_last=True):
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
        if self.channels_last:
            inputs = inputs.permute(0, 3, 1, 2) # Convert NHWC to NCHW format
        
        # check of the image size    
        if inputs.shape[2] >= self.kernel and inputs.shape[3] >= self.kernel:
            tensor = self.avg_pool(inputs)
        else:
            #print("Warning: AvgPooling layer not applied because the image size is smaller than the kernel size")
            return inputs.permute(0, 2, 3, 1) # Convert NCHW to NHWC format
        
        if self.channels_last:
            tensor = tensor.permute(0, 2, 3, 1) # Convert NCHW to NHWC format

        return tensor
    
class FullyConnected(nn.Module):
    def __init__(self,inputs_features, units):
        """ Initialize FullyConnected.

        Args:
            inputs_features : int
                Represents the number of inputs features of the layer
            units : int
                Represents the number of neurons in the layer

        """
        super().__init__()
        self.inputs__features = inputs_features
        self.units = units                
        self.fc = nn.Linear(in_features=self.inputs__features,
                            out_features=self.units)
        init.kaiming_normal_(self.fc.weight, mode='fan_out', nonlinearity='relu')
        
    def forward(self, inputs):
        """ FullyConnected layer.

        Args:
            inputs: input tensor to the block.

        Returns:
            output tensor.
        """
        tensor = self.fc(inputs)
                   
        return tensor
    
class NoOp(object):
    """ NoOp layer.
    """
    pass
    
def pad_features(tensors, channels_last=True):
    """ Pad with zeros the channels of the tensor in *tensors* list 
    that have the smaller number of feature maps.
    Args:
        tensors: list of 2 tensors to compare sizes.
    Returns:
        tensors with matching number of channels.
    """
    shapes = [list(t.shape) for t in tensors]
    
    channel_axis = - 1 if channels_last else 1
    
    if shapes[0][channel_axis] < shapes[1][channel_axis]:
        small_ch_id, large_ch_id = (0, 1)
    else:
        small_ch_id, large_ch_id = (1, 0)
    
    pad = shapes[large_ch_id][channel_axis] - shapes[small_ch_id][channel_axis]
    pad_beg = pad // 2
    pad_end = pad - pad_beg
    if channels_last:
        tensors[small_ch_id] = F.pad(tensors[small_ch_id],(pad_beg, pad_end, 0, 0, 0, 0, 0, 0))
    else:
        tensors[small_ch_id] = F.pad(tensors[small_ch_id], (0, 0, 0, 0,pad_beg,pad_end, 0, 0))
        
    return tensors

class NetworkGraph(nn.Module):
    def __init__(self, num_classes, mu=0.9, epsilon=2e-5):
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
        self.layer_dict = nn.ModuleDict()

    def create_functions(self, fn_dict):
        """ Generate all possible functions from functions descriptions in *self.fn_dict*.

        Args:
            fn_dict: dict with definitions of the functions (name and parameters);
                format --> {'fn_name': ['FNClass', {'param1': value1, 'param2': value2}]}.
        """
        for name, definition in fn_dict.items():
            if definition['function'] in ['ConvBlock', 'ResidualV1', 'ResidualV1Pr']:
                definition['params']['mu'] = self.mu
                definition['params']['epsilon'] = self.epsilon
            self.layer_dict[name] = globals()[definition['function']](**definition['params'])

    def forward(self, net_list, inputs):
        """ Create a PyTorch network from a list of layer names.

        Args:
            net_list: list of layer names, representing the network layers.
            inputs: input tensor to the network.

        Returns:
            logits tensor.
        """
        for f in net_list:
            if f == 'no_op':
                continue
            inputs = self.layer_dict[f](inputs)

        num_features = inputs.shape[1] * inputs.shape[2] * inputs.shape[3]
        tensor = torch.reshape(inputs, [-1, num_features])

        logits = FullyConnected(inputs_features=num_features,units=self.num_classes)(inputs=tensor)

        return logits