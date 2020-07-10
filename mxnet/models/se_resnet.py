import os
from mxnet import cpu
from mxnet.gluon import nn, HybridBlock
from .common import conv1x1_block, SEBlock, ResBlock, ResBottleneck, ResInitBlock

__all__ = ['se_resnet10', 'se_resnet12', 'se_resnet14', 'se_resnet16', 'se_resnet18', 'se_resnet26',
           'se_resnetbc26b', 'se_resnet34', 'se_resnetbc38b', 'se_resnet50', 'se_resnet50b', 'se_resnet101', 'se_resnet101b',
           'se_resnet152', 'se_resnet152b', 'se_resnet200', 'se_resnet200b']


class SEResUnit(HybridBlock):
    def __init__(self,
                 in_channels,
                 out_channels,
                 strides,
                 bn_use_global_stats,
                 bottleneck,
                 conv1_stride,
                 **kwargs):
        super(SEResUnit, self).__init__(**kwargs)
        self.resize_identity = (in_channels != out_channels) or (strides != 1)

        with self.name_scope():
            if bottleneck:
                self.body = ResBottleneck(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    strides=strides,
                    bn_use_global_stats=bn_use_global_stats,
                    conv1_stride=conv1_stride)
            else:
                self.body = ResBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    strides=strides,
                    bn_use_global_stats=bn_use_global_stats)
            self.se = SEBlock(channels=out_channels)
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
        x = self.se(x)
        x = x + identity
        x = self.activ(x)
        return x


class SEResNet(HybridBlock):
    def __init__(self,
                 channels,
                 init_block_channels,
                 bottleneck,
                 conv1_stride,
                 bn_use_global_stats=False,
                 in_channels=3,
                 in_size=(224, 224),
                 classes=1000,
                 **kwargs):
        super(SEResNet, self).__init__(**kwargs)
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
                        stage.add(SEResUnit(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            strides=strides,
                            bn_use_global_stats=bn_use_global_stats,
                            bottleneck=bottleneck,
                            conv1_stride=conv1_stride))
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


def get_seresnet(blocks,
                 bottleneck=None,
                 conv1_stride=True,
                 model_name=None,
                 pretrained=False,
                 ctx=cpu(),
                 root=os.path.join("~", ".mxnet", "models"),
                 **kwargs):
    if bottleneck is None:
        bottleneck = (blocks >= 50)

    if blocks == 10:
        layers = [1, 1, 1, 1]
    elif blocks == 12:
        layers = [2, 1, 1, 1]
    elif blocks == 14 and not bottleneck:
        layers = [2, 2, 1, 1]
    elif (blocks == 14) and bottleneck:
        layers = [1, 1, 1, 1]
    elif blocks == 16:
        layers = [2, 2, 2, 1]
    elif blocks == 18:
        layers = [2, 2, 2, 2]
    elif (blocks == 26) and not bottleneck:
        layers = [3, 3, 3, 3]
    elif (blocks == 26) and bottleneck:
        layers = [2, 2, 2, 2]
    elif blocks == 34:
        layers = [3, 4, 6, 3]
    elif (blocks == 38) and bottleneck:
        layers = [3, 3, 3, 3]
    elif blocks == 50:
        layers = [3, 4, 6, 3]
    elif blocks == 101:
        layers = [3, 4, 23, 3]
    elif blocks == 152:
        layers = [3, 8, 36, 3]
    elif blocks == 200:
        layers = [3, 24, 36, 3]
    else:
        raise ValueError("Unsupported SE-ResNet with number of blocks: {}".format(blocks))

    if bottleneck:
        assert (sum(layers) * 3 + 2 == blocks)
    else:
        assert (sum(layers) * 2 + 2 == blocks)

    init_block_channels = 64
    channels_per_layers = [64, 128, 256, 512]

    if bottleneck:
        bottleneck_factor = 4
        channels_per_layers = [ci * bottleneck_factor for ci in channels_per_layers]

    channels = [[ci] * li for (ci, li) in zip(channels_per_layers, layers)]

    net = SEResNet(
        channels=channels,
        init_block_channels=init_block_channels,
        bottleneck=bottleneck,
        conv1_stride=conv1_stride,
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

def se_resnet10(**kwargs):
    return get_seresnet(blocks=10, model_name="seresnet10", **kwargs)

def se_resnet12(**kwargs):
    return get_seresnet(blocks=12, model_name="seresnet12", **kwargs)

def se_resnet14(**kwargs):
    return get_seresnet(blocks=14, model_name="seresnet14", **kwargs)

def se_resnet16(**kwargs):
    return get_seresnet(blocks=16, model_name="seresnet16", **kwargs)

def se_resnet18(**kwargs):
    return get_seresnet(blocks=18, model_name="seresnet18", **kwargs)

def se_resnet26(**kwargs):
    return get_seresnet(blocks=26, bottleneck=False, model_name="seresnet26", **kwargs)

def se_resnetbc26b(**kwargs):
    return get_seresnet(blocks=26, bottleneck=True, conv1_stride=False, model_name="seresnetbc26b", **kwargs)

def se_resnet34(**kwargs):
    return get_seresnet(blocks=34, model_name="seresnet34", **kwargs)

def se_resnetbc38b(**kwargs):
    return get_seresnet(blocks=38, bottleneck=True, conv1_stride=False, model_name="seresnetbc38b", **kwargs)

def se_resnet50(**kwargs):
    return get_seresnet(blocks=50, model_name="seresnet50", **kwargs)

def se_resnet50b(**kwargs):
    return get_seresnet(blocks=50, conv1_stride=False, model_name="seresnet50b", **kwargs)

def se_resnet101(**kwargs):
    return get_seresnet(blocks=101, model_name="seresnet101", **kwargs)

def se_resnet101b(**kwargs):
    return get_seresnet(blocks=101, conv1_stride=False, model_name="seresnet101b", **kwargs)

def se_resnet152(**kwargs):
    return get_seresnet(blocks=152, model_name="seresnet152", **kwargs)

def se_resnet152b(**kwargs):
    return get_seresnet(blocks=152, conv1_stride=False, model_name="seresnet152b", **kwargs)

def se_resnet200(**kwargs):
    return get_seresnet(blocks=200, model_name="seresnet200", **kwargs)

def se_resnet200b(**kwargs):
    return get_seresnet(blocks=200, conv1_stride=False, model_name="seresnet200b", **kwargs)

def _test():
    import numpy as np
    import mxnet as mx

    pretrained = False

    models = [
        se_resnet10,
        se_resnet12,
        se_resnet14,
        se_resnet16,
        se_resnet18,
        se_resnet26,
        se_resnetbc26b,
        se_resnet34,
        se_resnetbc38b,
        se_resnet50,
        se_resnet50b,
        se_resnet101,
        se_resnet101b,
        se_resnet152,
        se_resnet152b,
        se_resnet200,
        se_resnet200b,
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
        assert (model != se_resnet10 or weight_count == 5463332)
        assert (model != se_resnet12 or weight_count == 5537896)
        assert (model != se_resnet14 or weight_count == 5835504)
        assert (model != se_resnet16 or weight_count == 7024640)
        assert (model != se_resnet18 or weight_count == 11778592)
        assert (model != se_resnet26 or weight_count == 18093852)
        assert (model != se_resnetbc26b or weight_count == 17395976)
        assert (model != se_resnet34 or weight_count == 21958868)
        assert (model != se_resnetbc38b or weight_count == 24026616)
        assert (model != se_resnet50 or weight_count == 28088024)
        assert (model != se_resnet50b or weight_count == 28088024)
        assert (model != se_resnet101 or weight_count == 49326872)
        assert (model != se_resnet101b or weight_count == 49326872)
        assert (model != se_resnet152 or weight_count == 66821848)
        assert (model != se_resnet152b or weight_count == 66821848)
        assert (model != se_resnet200 or weight_count == 71835864)
        assert (model != se_resnet200b or weight_count == 71835864)

        x = mx.nd.zeros((1, 3, 224, 224), ctx=ctx)
        y = net(x)
        assert (y.shape == (1, 1000))


if __name__ == "__main__":
    _test()
