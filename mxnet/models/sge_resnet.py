#Cnnot Use net.hybridize()
__all__ = ['sge_resnet18', 'sge_resnet34', 'sge_resnet50', 'sge_resnet101', 'sge_resnet152']

import os
import mxnet as mx
from mxnet import cpu
from mxnet.gluon import nn, HybridBlock, Parameter


class SpatialGroupEnhance(HybridBlock):
    def __init__(self, groups = 64):
        super(SpatialGroupEnhance, self).__init__()
        self.groups   = groups
        
        self.weight = self.params.get('weight', shape=(1, groups, 1, 1), init='zero')
        self.bias = self.params.get('bias', shape=(1, groups, 1, 1), init='ones')
        self.sig = nn.Activation('sigmoid')
        self.layer_shape = None
    
    def forward(self, x):
        self.layer_shape = x.shape
        return HybridBlock.forward(self,x) 

    def hybrid_forward(self, F, x, weight, bias): # (b, c, h, w)
        
        b, c, h, w = self.layer_shape
        x = F.Reshape(x, shape=(b*self.groups, -1, h, w))
        xn = F.broadcast_mul(x, F.contrib.AdaptiveAvgPooling2D(x, output_size=1))
        xn = xn.sum(axis=1, keepdims=True)
        t = F.Reshape(xn, shape=(b * self.groups, -1))
        mean, std = F.op.moments(t, axes=1, keepdims=True)
        t = F.broadcast_minus(t, mean)
        #t = F.broadcast_minus(t, t.mean(axis=1, keepdims=True))
        #_, std = F.op.moments(t, axes=1, keepdims=True)
        t = F.broadcast_div(t, F.sqrt(std)+1e-5)
        t = F.Reshape(t, shape=(b, self.groups, h, w))
        t = F.broadcast_add(F.broadcast_mul(t, weight), bias)
        t = F.Reshape(t, shape=(b * self.groups, 1, h, w))
        x = F.broadcast_mul(x, self.sig(t))
        x = F.Reshape(x, shape=(b, c, h, w))
        return x


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2D(out_planes, in_channels=in_planes, kernel_size=3, strides=stride,
                     padding=1, use_bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2D(out_planes, in_channels=in_planes, kernel_size=1, strides=stride, use_bias=False)


class BasicBlock(HybridBlock):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm(in_channels=planes, momentum=0.1)
        self.relu = nn.Activation('relu')
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm(in_channels=planes, momentum=0.1)
        self.downsample = downsample
        self.stride = stride
        self.sge = SpatialGroupEnhance(64)
        self.layer_shape = None

    def forward(self, x):
        self.layer_shape = x.shape
        return HybridBlock.forward(self,x)

    def hybrid_forward(self, F, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.sge(out)

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
        self.sge = SpatialGroupEnhance(64)
        self.layer_shape = None

    def forward(self, x):
        self.layer_shape = x.shape
        return HybridBlock.forward(self,x)

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
        out = self.sge(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet(HybridBlock):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False):
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
        self.layer_shape = None

    def forward(self, x):
        self.layer_shape = x.shape
        return HybridBlock.forward(self,x)


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


def sge_resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def sge_resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def sge_resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def sge_resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def sge_resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model

