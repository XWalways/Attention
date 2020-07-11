#Cnnot Use net.hybridize()
__all__ = ['srm_resnet18', 'srm_resnet34', 'srm_resnet50', 'srm_resnet101', 'srm_resnet152']

import mxnet as mx
from mxnet.gluon import nn
from mxnet.gluon.nn import HybridBlock, HybridSequential

class SRMLayer(HybridBlock):
    def __init__(self, channel, reduction=None):
        # Reduction for compatibility with layer_block interface
        super(SRMLayer, self).__init__()

        # CFC: channel-wise fully connected layer
        self.cfc = nn.Conv1D(channel, in_channels=channel, kernel_size=2, use_bias=False, groups=channel)
        self.bn = nn.BatchNorm(in_channels=channel, momentum=0.1)

    def hybrid_forward(self, F, x):
        b, c, h, w = x.shape

        # Style pooling
        mean, std = F.moments(x.reshape(b,c,h*w), axes=-1)
        mean = F.expand_dims(mean, axis=-1)
        std = F.expand_dims(std, axis=-1)
        
        u = F.concat(mean, std, dim=-1)# (b, c, 2)
         
        # Style integration
        z = self.cfc(u)  # (b, c, 1)
        z = self.bn(z)
        g = F.sigmoid(z)
        g = g.reshape(b, c, 1, 1)

        return F.broadcast_mul(g, x)

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2D(out_planes, in_channels=in_planes, kernel_size=3, strides=stride,
                     padding=1, use_bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2D(out_planes, in_channels=in_planes, kernel_size=1, strides=stride,
                     use_bias=False)

class BasicBlock(HybridBlock):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm(in_channels=planes, momentum=0.1)
        self.relu = nn.Activation('relu')
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm(in_channels=planes, momentum=0.1)
        self.layer_block = layer_block
        self.stride = stride
        self.downsample = downsample
        
        self.layer_block = SRMLayer(planes, reduction=16)

    def hybrid_forward(self, F, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.layer_block(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Bottleneck(HybridBlock):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm(in_channels=planes, momentum=0.1)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm(in_channels=planes, momentum=0.1)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm(in_channels=planes * self.expansion, momentum=0.1)
        self.relu = nn.Activation('relu')
        self.downsample = downsample
        self.stride = stride
        self.layer_block = SRMLayer(planes, reduction=16)

    def hybrid_forward(self, F, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.layer_block(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out




class ResNet(HybridBlock):

    def __init__(self, block, layers, num_classes=1000):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2D(64, in_channels=3, kernel_size=7, strides=2, padding=3,
                               use_bias=False)
        self.bn1 = nn.BatchNorm(in_channels=64, momentum=0.1)
        self.relu = nn.Activation('relu')
        self.maxpool = nn.MaxPool2D(pool_size=3, strides=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.GlobalAvgPool2D()
        self.fc = nn.Dense(num_classes, in_units=512 * block.expansion)
        


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        with self.name_scope():
            layers = nn.HybridSequential('')
        if stride != 1 or self.inplanes != planes * block.expansion:
            with self.name_scope():
                downsample = nn.HybridSequential('') 
                downsample.add(conv1x1(self.inplanes, planes * block.expansion, stride))
                downsample.add(nn.BatchNorm(in_channels=planes * block.expansion))

       
        layers.add(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.add(block(self.inplanes, planes))
        
        
        return layers

    def hybrid_forward(self, F, x):
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        
        x = self.fc(x)

        return x


def srm_resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def srm_resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def srm_resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def srm_resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def srm_resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model
