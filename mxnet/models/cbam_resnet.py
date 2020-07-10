__all__ = ['cbam_resnet18', 'cbam_resnet34', 'cbam_resnet50', 'cbam_resnet101', 'cbam_resnet152']

import os
from mxnet import cpu
from mxnet.gluon import nn, HybridBlock
from .common import conv1x1_block, conv7x7_block, ResInitBlock, ResBlock, ResBottleneck

class MLP(HybridBlock):
    def __init__(self,
                 channels,
                 reduction_ratio=16,
                 **kwargs):
        super(MLP, self).__init__(**kwargs)
        mid_channels = channels // reduction_ratio

        with self.name_scope():
            self.flatten = nn.Flatten()
            self.fc1 = nn.Dense(
                units=mid_channels,
                in_units=channels)
            self.activ = nn.Activation("relu")
            self.fc2 = nn.Dense(
                units=channels,
                in_units=mid_channels)

    def hybrid_forward(self, F, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.activ(x)
        x = self.fc2(x)
        return x

class ChannelGate(HybridBlock):
    def __init__(self,
                 channels,
                 reduction_ratio=16,
                 **kwargs):
        super(ChannelGate, self).__init__(**kwargs)
        with self.name_scope():
            self.avg_pool = nn.GlobalAvgPool2D()
            self.max_pool = nn.GlobalMaxPool2D()
            self.mlp = MLP(
                channels=channels,
                reduction_ratio=reduction_ratio)
            self.sigmoid = nn.Activation("sigmoid")

    def hybrid_forward(self, F, x):
        att1 = self.avg_pool(x)
        att1 = self.mlp(att1)
        att2 = self.max_pool(x)
        att2 = self.mlp(att2)
        att = att1 + att2
        att = self.sigmoid(att)
        att = att.expand_dims(2).expand_dims(3).broadcast_like(x)
        x = x * att
        return x

class SpatialGate(HybridBlock):
    def __init__(self,
                 bn_use_global_stats,
                 **kwargs):
        super(SpatialGate, self).__init__(**kwargs)
        with self.name_scope():
            self.conv = conv7x7_block(
                in_channels=2,
                out_channels=1,
                bn_use_global_stats=bn_use_global_stats,
                activation=None)
            self.sigmoid = nn.Activation("sigmoid")

    def hybrid_forward(self, F, x):
        att1 = x.max(axis=1).expand_dims(1)
        att2 = x.mean(axis=1).expand_dims(1)
        att = F.concat(att1, att2, dim=1)
        att = self.conv(att)
        att = self.sigmoid(att).broadcast_like(x)
        x = x * att
        return x

class CbamBlock(HybridBlock):
    def __init__(self,
                 channels,
                 reduction_ratio=16,
                 bn_use_global_stats=False,
                 **kwargs):
        super(CbamBlock, self).__init__(**kwargs)
        with self.name_scope():
            self.ch_gate = ChannelGate(
                channels=channels,
                reduction_ratio=reduction_ratio)
            self.sp_gate = SpatialGate(bn_use_global_stats=bn_use_global_stats)

    def hybrid_forward(self, F, x):
        x = self.ch_gate(x)
        x = self.sp_gate(x)
        return x

class CbamResUnit(HybridBlock):
    def __init__(self,
                 in_channels,
                 out_channels,
                 strides,
                 bn_use_global_stats,
                 bottleneck,
                 **kwargs):
        super(CbamResUnit, self).__init__(**kwargs)
        self.resize_identity = (in_channels != out_channels) or (strides != 1)

        with self.name_scope():
            if bottleneck:
                self.body = ResBottleneck(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    strides=strides,
                    bn_use_global_stats=bn_use_global_stats,
                    conv1_stride=False)
            else:
                self.body = ResBlock(
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
            self.cbam = CbamBlock(channels=out_channels)
            self.activ = nn.Activation("relu")

    def hybrid_forward(self, F, x):
        if self.resize_identity:
            identity = self.identity_conv(x)
        else:
            identity = x
        x = self.body(x)
        x = self.cbam(x)
        x = x + identity
        x = self.activ(x)
        return x

class CbamResNet(HybridBlock):
    def __init__(self,
                 channels,
                 init_block_channels,
                 bottleneck,
                 bn_use_global_stats=False,
                 in_channels=3,
                 in_size=(224, 224),
                 classes=1000,
                 **kwargs):
        super(CbamResNet, self).__init__(**kwargs)
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
                        stage.add(CbamResUnit(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            strides=strides,
                            bn_use_global_stats=bn_use_global_stats,
                            bottleneck=bottleneck))
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


def get_resnet(blocks,
               model_name=None,
               pretrained=False,
               ctx=cpu(),
               root=os.path.join("~", ".mxnet", "models"),
               **kwargs):
    if blocks == 18:
        layers = [2, 2, 2, 2]
    elif blocks == 34:
        layers = [3, 4, 6, 3]
    elif blocks == 50:
        layers = [3, 4, 6, 3]
    elif blocks == 101:
        layers = [3, 4, 23, 3]
    elif blocks == 152:
        layers = [3, 8, 36, 3]
    else:
        raise ValueError("Unsupported CBAM-ResNet with number of blocks: {}".format(blocks))

    init_block_channels = 64

    if blocks < 50:
        channels_per_layers = [64, 128, 256, 512]
        bottleneck = False
    else:
        channels_per_layers = [256, 512, 1024, 2048]
        bottleneck = True

    channels = [[ci] * li for (ci, li) in zip(channels_per_layers, layers)]

    net = CbamResNet(
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


def cbam_resnet18(**kwargs):
    return get_resnet(blocks=18, model_name="cbam_resnet18", **kwargs)

def cbam_resnet34(**kwargs):
    return get_resnet(blocks=34, model_name="cbam_resnet34", **kwargs)

def cbam_resnet50(**kwargs):
    return get_resnet(blocks=50, model_name="cbam_resnet50", **kwargs)

def cbam_resnet101(**kwargs):
    return get_resnet(blocks=101, model_name="cbam_resnet101", **kwargs)

def cbam_resnet152(**kwargs):
    return get_resnet(blocks=152, model_name="cbam_resnet152", **kwargs)

def _test():
    import numpy as np
    import mxnet as mx

    pretrained = False

    models = [
        cbam_resnet18,
        cbam_resnet34,
        cbam_resnet50,
        cbam_resnet101,
        cbam_resnet152,
    ]

    for model in models:

        net = model(pretrained=pretrained)

        ctx = mx.cpu()
        if not pretrained:
            net.initialize(ctx=ctx)

        net.hybridize()
        net_params = net.collect_params()
        weight_count = 0
        for param in net_params.values():
            if (param.shape is None) or (not param._differentiable):
                continue
            weight_count += np.prod(param.shape)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != cbam_resnet18 or weight_count == 11779392)
        assert (model != cbam_resnet34 or weight_count == 21960468)
        assert (model != cbam_resnet50 or weight_count == 28089624)
        assert (model != cbam_resnet101 or weight_count == 49330172)
        assert (model != cbam_resnet152 or weight_count == 66826848)

        x = mx.nd.zeros((2, 3, 224, 224), ctx=ctx)
        y = net(x)
        assert (y.shape == (2, 1000))


if __name__ == "__main__":
    _test()
