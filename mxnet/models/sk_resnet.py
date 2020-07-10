import os
import mxnet as mx
from mxnet import cpu
from mxnet.gluon import nn, HybridBlock

from .common import conv1x1, conv1x1_block, conv3x3_block, Concurrent, ResInitBlock

__all__ = ['sk_resnet18', 'sk_resnet34', 'sk_resnet50', 'sk_resnet101',
           'sk_resnet152']


class SKConvBlock(HybridBlock):
    def __init__(self,
                 in_channels,
                 out_channels,
                 strides,
                 groups=32,
                 bn_use_global_stats=False,
                 num_branches=2,
                 reduction=16,
                 min_channels=32,
                 **kwargs):
        super(SKConvBlock, self).__init__(**kwargs)
        self.num_branches = num_branches
        self.out_channels = out_channels
        mid_channels = max(in_channels // reduction, min_channels)

        with self.name_scope():
            self.branches = Concurrent(stack=True, prefix="")
            for i in range(num_branches):
                dilation = 1 + i
                self.branches.add(conv3x3_block(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    strides=strides,
                    padding=dilation,
                    dilation=dilation,
                    groups=groups,
                    bn_use_global_stats=bn_use_global_stats))
            self.fc1 = conv1x1_block(
                in_channels=out_channels,
                out_channels=mid_channels,
                bn_use_global_stats=bn_use_global_stats)
            self.fc2 = conv1x1(
                in_channels=mid_channels,
                out_channels=(out_channels * num_branches))

    def hybrid_forward(self, F, x):
        y = self.branches(x)

        u = y.sum(axis=1)
        s = F.contrib.AdaptiveAvgPooling2D(u, output_size=1)
        z = self.fc1(s)
        w = self.fc2(z)

        w = w.reshape((0, self.num_branches, self.out_channels))
        w = F.softmax(w, axis=1)
        w = w.expand_dims(3).expand_dims(4)

        y = F.broadcast_mul(y, w)
        y = y.sum(axis=1)
        return y


class SKNetBasicBlock(HybridBlock):
    def __init__(self,
                 in_channels,
                 out_channels,
                 strides,
                 bn_use_global_stats=False,
                 **kwargs):
        super(SKNetBasicBlock, self).__init__(**kwargs)
        with self.name_scope():
            self.conv1 = conv3x3_block(
                in_channels=in_channels,
                out_channels=out_channels,
                bn_use_global_stats=bn_use_global_stats)
            self.conv2 = SKConvBlock(
                in_channels=mid_channels,
                out_channels=mid_channels,
                strides=strides,
                bn_use_global_stats=bn_use_global_stats)
    def hybrid_forward(self, F, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

         

class SKNetBottleneck(HybridBlock):
    def __init__(self,
                 in_channels,
                 out_channels,
                 strides,
                 bn_use_global_stats=False,
                 bottleneck_factor=2,
                 **kwargs):
        super(SKNetBottleneck, self).__init__(**kwargs)
        mid_channels = out_channels // bottleneck_factor

        with self.name_scope():
            self.conv1 = conv1x1_block(
                in_channels=in_channels,
                out_channels=mid_channels,
                bn_use_global_stats=bn_use_global_stats)
            self.conv2 = SKConvBlock(
                in_channels=mid_channels,
                out_channels=mid_channels,
                strides=strides,
                bn_use_global_stats=bn_use_global_stats)
            self.conv3 = conv1x1_block(
                in_channels=mid_channels,
                out_channels=out_channels,
                bn_use_global_stats=bn_use_global_stats,
                activation=None)

    def hybrid_forward(self, F, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x

class SKNetUnit(HybridBlock):
    def __init__(self,
                 in_channels,
                 out_channels,
                 strides,
                 bn_use_global_stats=False,
                 bottleneck = True,
                 **kwargs):
        super(SKNetUnit, self).__init__(**kwargs)
        self.resize_identity = (in_channels != out_channels) or (strides != 1)

        with self.name_scope():
            if bottleneck:
                self.body = SKNetBottleneck(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    strides=strides,
                    bn_use_global_stats=bn_use_global_stats)
            else:
                self.body = SKNetBasicBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    strides=strides,
                    bn_use_global_stats=bn_use_global_stats)
            if self.resize_identity:
                self.identity_conv = conv1x1_block(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    strides=strides,
                    bn_use_global_stats=bn_use_global_stats,
                    activation=None)
            self.activ = nn.Activation("relu")

    def hybrid_forward(self, F, x):
        if self.resize_identity:
            identity = self.identity_conv(x)
        else:
            identity = x
        x = self.body(x)
        x = x + identity
        x = self.activ(x)
        return x


class SKNet(HybridBlock):
    def __init__(self,
                 channels,
                 init_block_channels,
                 bn_use_global_stats=False,
                 in_channels=3,
                 in_size=(224, 224),
                 classes=1000,
                 bottleneck = True,
                 **kwargs):
        super(SKNet, self).__init__(**kwargs)
        self.in_size = in_size
        self.classes = classes

        with self.name_scope():
            self.features = nn.HybridSequential(prefix="")
            self.features.add(ResInitBlock(
                in_channels=in_channels,
                out_channels=init_block_channels,
                bn_use_global_stats=bn_use_global_stats))
            in_channels = init_block_channels
            for i, channels_per_stage in enumerate(channels):
                stage = nn.HybridSequential(prefix="stage{}_".format(i + 1))
                with stage.name_scope():
                    for j, out_channels in enumerate(channels_per_stage):
                        strides = 2 if (j == 0) and (i != 0) else 1
                        stage.add(SKNetUnit(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            strides=strides,
                            bn_use_global_stats=bn_use_global_stats,
                            bottleneck = True))
                        in_channels = out_channels
                self.features.add(stage)
            self.features.add(nn.AvgPool2D(
                pool_size=7,
                strides=1))

            self.output = nn.HybridSequential(prefix="")
            self.output.add(nn.Flatten())
            self.output.add(nn.Dense(
                units=classes,
                in_units=in_channels))

    def hybrid_forward(self, F, x):
        x = self.features(x)
        x = self.output(x)
        return x

def get_sknet(blocks,
              model_name=None,
              pretrained=False,
              ctx=cpu(),
              root=os.path.join("~", ".mxnet", "models"),
              **kwargs):
    if blocks == 18:
        layers = [2, 2, 2, 2]
        channels_per_layers = [64, 128, 256, 512]
        bottleneck = False
    elif blocks == 34:
        layers = [3, 4, 6, 3]
        channels_per_layers = [64, 128, 256, 512]
        bottleneck = False
    elif blocks == 50:
        layers = [3, 4, 6, 3]
        channels_per_layers = [256, 512, 1024, 2048]
        bottleneck = True
    elif blocks == 101:
        layers = [3, 4, 23, 3]
        channels_per_layers = [256, 512, 1024, 2048]
        bottleneck = True
    elif blocks == 152:
        layers = [3, 8, 36, 3]
        channels_per_layers = [256, 512, 1024, 2048]
        bottleneck = True
    else:
        raise ValueError("Unsupported SKNet with number of blocks: {}".format(blocks))

    init_block_channels = 64
    #channels_per_layers = [256, 512, 1024, 2048]
    channels = [[ci] * li for (ci, li) in zip(channels_per_layers, layers)]

    net = SKNet(
        channels=channels,
        init_block_channels=init_block_channels,
        bottleneck=bottleneck,
        **kwargs)

    if pretrained:
        if (model_name is None) or (not model_name):
            raise ValueError("Parameter `model_name` should be properly initialized for loading pretrained model.")
        from .model_store import get_model_file
        net.load_parameters(
            filename=get_model_file(
                model_name=model_name,
                local_model_store_dir_path=root),
            ctx=ctx)

    return net

def sk_resnet18(**kwargs):
    return get_sknet(blocks=18, **kwargs)

def sk_resnet34(**kwargs):
    return get_sknet(blocks=34, **kwargs)

def sk_resnet50(**kwargs):
    return get_sknet(blocks=50, model_name="sknet50", **kwargs)

def sk_resnet101(**kwargs):
    return get_sknet(blocks=101, model_name="sknet101", **kwargs)

def sk_resnet152(**kwargs):
    return get_sknet(blocks=152, model_name="sknet152", **kwargs)

def _test():
    import numpy as np
    import mxnet as mx

    pretrained = False
    models = [
        sk_resnet18,
        sk_resnet34,
        #sk_resnet50,
        #sk_resnet101,
        #sk_resnet152,
    ]

    for model in models:

        net = model(pretrained=pretrained)

        ctx = mx.cpu()
        if not pretrained:
            net.initialize(ctx=ctx)

        # net.hybridize()
        net_params = net.collect_params()
        weight_count = 0
        for param in net_params.values():
            if (param.shape is None) or (not param._differentiable):
                continue
            weight_count += np.prod(param.shape)
        print("m={}, {}".format(model.__name__, weight_count))
        #assert (model != sk_resnet50 or weight_count == 27479784)
        #assert (model != sk_resnet101 or weight_count == 48736040)
        #assert (model != sk_resnet152 or weight_count == 66295656)

        x = mx.nd.zeros((14, 3, 224, 224), ctx=ctx)
        y = net(x)
        assert (y.shape == (14, 1000))

if __name__ == "__main__":
    _test()
