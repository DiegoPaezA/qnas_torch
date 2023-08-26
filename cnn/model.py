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

    def __init__(self, kernel=1, in_channels=1, filters=1, strides=1, mu=1, epsilon=1, channels_last=True):
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

        self.batch_norm = nn.BatchNorm2d(num_features=self.filters,
                                         momentum=self.batch_norm_mu, 
                                         eps=self.batch_norm_epsilon)
            
    def _conv_2d(self, inputs):
        """ Convolution operation wrapper.
        Args:
            inputs: input tensor.
        Returns:
            output tensor.
        """
        init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')  # He Normal initialization
        return self.conv(inputs)
    
    def forward(self, inputs):
        """ Convolutional block with convolution op + batch normalization op.

        Args:
            inputs: input tensor to the block.

        Returns:
            output tensor.
        """
        if self.channels_last:
            inputs = inputs.permute(0, 3, 1, 2) # Convert NHWC to NCHW format
        
        tensor = self._conv_2d(inputs)
        tensor = self.batch_norm(tensor)
        tensor = self.activation(tensor)
        
        if self.channels_last:
            tensor = tensor.permute(0, 2, 3, 1) # Convert NCHW to NHWC format
            
        return tensor

class ResidualV1(nn.Module):
    """ Residual Block with Conv -> BatchNorm -> ReLU -> Conv -> BatchNorm -> Add -> ReLU """
    def __init__(self, in_channel=1, kernel=1, filters=1, strides=1, mu=1, epsilon=1, channels_last=True):
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
        
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=filters, 
                              kernel_size=self.kernel_size,stride=strides, 
                              padding= self.padding ,bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=self.filters, momentum=self.batch_norm_mu, 
                                         eps=self.batch_norm_epsilon)
        self.conv2 = nn.Conv2d(in_channels=filters, out_channels=filters, 
                              kernel_size=self.kernel_size,stride=strides, 
                              padding= self.padding ,bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=self.filters,momentum=self.batch_norm_mu, 
                                         eps=self.batch_norm_epsilon)
        
        init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')  # He Normal initialization
        init.kaiming_normal_(self.conv2.weight, mode='fan_out', nonlinearity='relu')  # He Normal initialization
        

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
        tensor = self.conv1(inputs)
        tensor = self.bn1(tensor)
        tensor = F.relu(tensor)
        print(f'tensor.shape Layer 1: {tensor.shape}')
            
        tensor = self.conv2(tensor)
        tensor = self.bn2(tensor)
              
        print(f'tensor.shape Layer 2: {tensor.shape}')
        inputs, tensor = pad_features([inputs, tensor], channels_last=False)
        print(f'inputs.shape Layer 2: {inputs.shape}')
        print(f'tensor.shape Pad: {tensor.shape}')
        
        
        tensor = tensor + inputs
        
        tensor = F.relu(tensor)
        
        if self.channels_last:
            tensor = tensor.permute(0, 2, 3, 1) # Convert NCHW to NHWC format

        return tensor

class ResidualV1Pr(nn.Module):
    """ Residual V1 block with projection shortcut """
    def __init__(self, in_channel=1, kernel=1, filters=1, strides=1, mu=1, epsilon=1, channels_last=True):
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
        
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=filters, 
                              kernel_size=self.kernel_size,stride=strides, 
                              padding= self.padding ,bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=self.filters, momentum=self.batch_norm_mu, 
                                         eps=self.batch_norm_epsilon)
        self.conv2 = nn.Conv2d(in_channels=filters, out_channels=filters, 
                              kernel_size=self.kernel_size,stride=strides, 
                              padding= self.padding ,bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=self.filters,momentum=self.batch_norm_mu, 
                                         eps=self.batch_norm_epsilon)
        
        init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')  # He Normal initialization
        init.kaiming_normal_(self.conv2.weight, mode='fan_out', nonlinearity='relu')  # He Normal initialization
        
        # Shortcut connection
        if strides != 1 or in_channel != filters:
            print("Shortcut connection")
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channel, filters, kernel_size=1, padding=0,stride=strides, bias=False),
                nn.BatchNorm2d(num_features=self.filters,momentum=self.batch_norm_mu, 
                                         eps=self.batch_norm_epsilon)
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
        
        print(f'inputs.shape: {inputs.shape}')            
        tensor = self.conv1(inputs)
        tensor = self.bn1(tensor)
        tensor = F.relu(tensor)
        print(f'tensor.shape Layer 1: {tensor.shape}')
            
        tensor = self.conv2(tensor)
        tensor = self.bn2(tensor)
              
        print(f'tensor.shape Layer 2: {tensor.shape}')
              
        tensor = tensor + self.shortcut(inputs)
        tensor = F.relu(tensor)
        print(f'output.shape: {tensor.shape}')
        if self.channels_last:
            tensor = tensor.permute(0, 2, 3, 1) # Convert NCHW to NHWC format

        return tensor

class MaxPooling(nn.Module):
    """ Max Pooling layer """

    def __init__(self, kernel=1, strides=1, channels_last=True):
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

    def __init__(self, kernel=1, strides=1, channels_last=True):
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
        #print(f'inputs.shape avg: {inputs.shape}')
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
    
class NoOp(nn.Module):
    """ NoOp layer.
    """
    pass

functions_dict = {'ConvBlock': ConvBlock,
                  'ResidualV1': ResidualV1,
                  'ResidualV1Pr': ResidualV1Pr,
                  'MaxPooling': MaxPooling,
                  'AvgPooling': AvgPooling,
                  'FullyConnected': FullyConnected,
                  'no_op': NoOp}
 
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
        
    def create_functions(self, net_list, fn_dict):
        """ Generate all possible functions from functions descriptions in *self.fn_dict*.

        Args:
            fn_dict: dict with definitions of the functions (name and parameters);
                format --> {'fn_name': ['FNClass', {'param1': value1, 'param2': value2}]}.
        """
        in_channels = self.in_channels
        self.layers = []

        for name in net_list:
            parameters = fn_dict[name]
            if parameters['function'] == 'no_op':
                continue
            if parameters['function'] in ['ConvBlock', 'ResidualV1', 'ResidualV1Pr']:
                parameters['params']['mu'] = self.mu
                parameters['params']['epsilon'] = self.epsilon
                parameters['params']['in_channels'] = in_channels
                in_channels = parameters['params']['filters']
            
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
        print(f'inputs.shape: {inputs.shape}')
        if debug:
            for f in self.layers:
                print(f'f: {f}')
                inputs = f(inputs)
                print(f'layer output.shape: {inputs.shape}')
        else:
            inputs = self.model(inputs)
            print(f'layer output.shape: {inputs.shape}')

        if self.fc is None:
            batch_size, num_features, height, width = inputs.size()
            num_flat_features = num_features * height * width
            self.fc = FullyConnected(input_features=num_flat_features, units=self.num_classes)  # Replace the placeholder

        batch_size = inputs.size(0)
        inputs = inputs.reshape(batch_size, -1)
        print('FullyConnected')
        print(f'layer output.shape: {inputs.shape}')
        logits = self.fc(inputs)

        return logits