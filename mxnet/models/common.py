import os
from pathlib import Path
import requests
import errno
import shutil
import hashlib
import zipfile
import logging
from tqdm import tqdm
from inspect import isfunction
import mxnet as mx
from mxnet.gluon import nn, HybridBlock

logger = logging.getLogger(__name__)

_model_sha1 = {name: checksum for checksum, name in [
    ('bcfefe1dd1dd1ef5cfed5563123c1490ea37b42e', 'resnest50'),
    ('5da943b3230f071525a98639945a6b3b3a45ac95', 'resnest101'),
    ('0c5d117df664ace220aa6fc2922c094bb079d381', 'resnest200'),
    ('11ae7f5da2bcdbad05ba7e84f9b74383e717f3e3', 'resnest269'),]}

encoding_repo_url = 'https://s3.us-west-1.wasabisys.com/resnest/'
_url_format = '{repo_url}gluon/{file_name}.zip'

def short_hash(name):
    if name not in _model_sha1:
        raise ValueError('Pretrained model for {name} is not available.'.format(name=name))
    return _model_sha1[name][:8]

def get_model_file(name, root=os.path.join('~', '.encoding', 'models')):
    if name not in _model_sha1:
        import gluoncv as gcv
        return gcv.model_zoo.model_store.get_model_file(name, root=root)
    file_name = '{name}-{short_hash}'.format(name=name, short_hash=short_hash(name))
    root = os.path.expanduser(root)
    file_path = os.path.join(root, file_name+'.params')
    sha1_hash = _model_sha1[name]
    if os.path.exists(file_path):
        if check_sha1(file_path, sha1_hash):
            return file_path
        else:
            print('Mismatch in the content of model file {} detected.' +
                  ' Downloading again.'.format(file_path))
    else:
        print('Model file {} is not found. Downloading.'.format(file_path))

    if not os.path.exists(root):
        os.makedirs(root)

    zip_file_path = os.path.join(root, file_name+'.zip')
    repo_url = os.environ.get('ENCODING_REPO', encoding_repo_url)
    if repo_url[-1] != '/':
        repo_url = repo_url + '/'
    download(_url_format.format(repo_url=repo_url, file_name=file_name),
             path=zip_file_path,
             overwrite=True)
    with zipfile.ZipFile(zip_file_path) as zf:
        zf.extractall(root)
    os.remove(zip_file_path)

    if check_sha1(file_path, sha1_hash):
        return file_path
    else:
        raise ValueError('Downloaded file has different hash. Please try again.')


def download(url, path=None, overwrite=False, sha1_hash=None):
    if path is None:
        fname = url.split('/')[-1]
    else:
        path = os.path.expanduser(path)
        if os.path.isdir(path):
            fname = os.path.join(path, url.split('/')[-1])
        else:
            fname = path

    if overwrite or not os.path.exists(fname) or (sha1_hash and not check_sha1(fname, sha1_hash)):
        dirname = os.path.dirname(os.path.abspath(os.path.expanduser(fname)))
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        logger.info('Downloading %s from %s...'%(fname, url))
        r = requests.get(url, stream=True)
        if r.status_code != 200:
            raise RuntimeError("Failed downloading url %s"%url)
        total_length = r.headers.get('content-length')
        with open(fname, 'wb') as f:
            if total_length is None: # no content length header
                for chunk in r.iter_content(chunk_size=1024):
                    if chunk: # filter out keep-alive new chunks
                        f.write(chunk)
            else:
                total_length = int(total_length)
                for chunk in tqdm(r.iter_content(chunk_size=1024),
                                  total=int(total_length / 1024. + 0.5),
                                  unit='KB', unit_scale=False, dynamic_ncols=True):
                    f.write(chunk)

        if sha1_hash and not check_sha1(fname, sha1_hash):
            raise UserWarning('File {} is downloaded but the content hash does not match. ' \
                              'The repo may be outdated or download may be incomplete. ' \
                              'If the "repo_url" is overridden, consider switching to ' \
                              'the default repo.'.format(fname))

    return fname


def check_sha1(filename, sha1_hash):
    sha1 = hashlib.sha1()
    with open(filename, 'rb') as f:
        while True:
            data = f.read(1048576)
            if not data:
                break
            sha1.update(data)

    return sha1.hexdigest() == sha1_hash

def conv1x1(in_channels,
            out_channels,
            strides=1,
            groups=1,
            use_bias=False):
    return nn.Conv2D(
        channels=out_channels,
        kernel_size=1,
        strides=strides,
        groups=groups,
        use_bias=use_bias,
        in_channels=in_channels)

class ReLU6(HybridBlock):
    """
    ReLU6 activation layer.
    """
    def __init__(self, **kwargs):
        super(ReLU6, self).__init__(**kwargs)

    def hybrid_forward(self, F, x):
        return F.clip(x, 0.0, 6.0, name="relu6")


class PReLU2(HybridBlock):
    """
    Parametric leaky version of a Rectified Linear Unit (with wide alpha).
    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    alpha_initializer : Initializer
        Initializer for the `embeddings` matrix.
    """
    def __init__(self,
                 in_channels=1,
                 alpha_initializer=mx.init.Constant(0.25),
                 **kwargs):
        super(PReLU2, self).__init__(**kwargs)
        with self.name_scope():
            self.alpha = self.params.get("alpha", shape=(in_channels,), init=alpha_initializer)

    def hybrid_forward(self, F, x, alpha):
        return F.LeakyReLU(x, gamma=alpha, act_type="prelu", name="fwd")


class HSigmoid(HybridBlock):
    """
    Approximated sigmoid function, so-called hard-version of sigmoid from 'Searching for MobileNetV3,'
    https://arxiv.org/abs/1905.02244.
    """
    def __init__(self, **kwargs):
        super(HSigmoid, self).__init__(**kwargs)

    def hybrid_forward(self, F, x):
        return F.clip(x + 3.0, 0.0, 6.0, name="relu6") / 6.0


class HSwish(HybridBlock):
    """
    H-Swish activation function from 'Searching for MobileNetV3,' https://arxiv.org/abs/1905.02244.
    """
    def __init__(self, **kwargs):
        super(HSwish, self).__init__(**kwargs)

    def hybrid_forward(self, F, x):
        return x * F.clip(x + 3.0, 0.0, 6.0, name="relu6") / 6.0


def get_activation_layer(activation):
    """
    Create activation layer from string/function.
    Parameters:
    ----------
    activation : function, or str, or HybridBlock
        Activation function or name of activation function.
    Returns
    -------
    HybridBlock
        Activation layer.
    """
    assert (activation is not None)
    if isfunction(activation):
        return activation()
    elif isinstance(activation, str):
        if activation == "relu6":
            return ReLU6()
        elif activation == "swish":
            return nn.Swish()
        elif activation == "hswish":
            return HSwish()
        elif activation == "hsigmoid":
            return HSigmoid()
        else:
            return nn.Activation(activation)
    else:
        assert (isinstance(activation, HybridBlock))
        return activation

class ConvBlock(HybridBlock):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 strides,
                 padding,
                 dilation=1,
                 groups=1,
                 use_bias=False,
                 use_bn=True,
                 bn_epsilon=1e-5,
                 bn_use_global_stats=False,
                 activation=(lambda: nn.Activation("relu")),
                 **kwargs):
        super(ConvBlock, self).__init__(**kwargs)
        self.activate = (activation is not None)
        self.use_bn = use_bn

        with self.name_scope():
            self.conv = nn.Conv2D(
                channels=out_channels,
                kernel_size=kernel_size,
                strides=strides,
                padding=padding,
                dilation=dilation,
                groups=groups,
                use_bias=use_bias,
                in_channels=in_channels)
            if self.use_bn:
                self.bn = nn.BatchNorm(
                    in_channels=out_channels,
                    epsilon=bn_epsilon,
                    use_global_stats=bn_use_global_stats)
            if self.activate:
                self.activ = get_activation_layer(activation)

    def hybrid_forward(self, F, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        if self.activate:
            x = self.activ(x)
        return x

def conv1x1_block(in_channels,
                  out_channels,
                  strides=1,
                  groups=1,
                  use_bias=False,
                  use_bn=True,
                  bn_epsilon=1e-5,
                  bn_use_global_stats=False,
                  activation=(lambda: nn.Activation("relu")),
                  **kwargs):
    return ConvBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        strides=strides,
        padding=0,
        groups=groups,
        use_bias=use_bias,
        use_bn=use_bn,
        bn_epsilon=bn_epsilon,
        bn_use_global_stats=bn_use_global_stats,
        activation=activation,
        **kwargs)


def conv3x3_block(in_channels,
                  out_channels,
                  strides=1,
                  padding=1,
                  dilation=1,
                  groups=1,
                  use_bias=False,
                  use_bn=True,
                  bn_epsilon=1e-5,
                  bn_use_global_stats=False,
                  activation=(lambda: nn.Activation("relu")),
                  **kwargs):
    return ConvBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        strides=strides,
        padding=padding,
        dilation=dilation,
        groups=groups,
        use_bias=use_bias,
        use_bn=use_bn,
        bn_epsilon=bn_epsilon,
        bn_use_global_stats=bn_use_global_stats,
        activation=activation,
        **kwargs)

def conv7x7_block(in_channels,
                  out_channels,
                  strides=1,
                  padding=3,
                  use_bias=False,
                  use_bn=True,
                  bn_use_global_stats=False,
                  activation=(lambda: nn.Activation("relu")),
                  **kwargs):
    return ConvBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=7,
        strides=strides,
        padding=padding,
        use_bias=use_bias,
        use_bn=use_bn,
        bn_use_global_stats=bn_use_global_stats,
        activation=activation,
        **kwargs)


class Concurrent(nn.HybridSequential):
    def __init__(self,
                 axis=1,
                 stack=False,
                 **kwargs):
        super(Concurrent, self).__init__(**kwargs)
        self.axis = axis
        self.stack = stack

    def hybrid_forward(self, F, x):
        out = []
        for block in self._children.values():
            out.append(block(x))
        if self.stack:
            out = F.stack(*out, axis=self.axis)
        else:
            out = F.concat(*out, dim=self.axis)
        return out

class ResInitBlock(HybridBlock):
    def __init__(self,
                 in_channels,
                 out_channels,
                 bn_use_global_stats=False,
                 **kwargs):
        super(ResInitBlock, self).__init__(**kwargs)
        with self.name_scope():
            self.conv = conv7x7_block(
                in_channels=in_channels,
                out_channels=out_channels,
                strides=2,
                bn_use_global_stats=bn_use_global_stats)
            self.pool = nn.MaxPool2D(
                pool_size=3,
                strides=2,
                padding=1)

    def hybrid_forward(self, F, x):
        x = self.conv(x)
        x = self.pool(x)
        return x
class SEBlock(HybridBlock):
    def __init__(self,
                 channels,
                 reduction=16,
                 round_mid=False,
                 mid_activation=(lambda: nn.Activation("relu")),
                 out_activation=(lambda: nn.Activation("sigmoid")),
                 **kwargs):
        super(SEBlock, self).__init__(**kwargs)
        mid_channels = channels // reduction if not round_mid else round_channels(float(channels) / reduction)

        with self.name_scope():
            self.conv1 = conv1x1(
                in_channels=channels,
                out_channels=mid_channels,
                use_bias=True)
            self.activ = get_activation_layer(mid_activation)
            self.conv2 = conv1x1(
                in_channels=mid_channels,
                out_channels=channels,
                use_bias=True)
            self.sigmoid = get_activation_layer(out_activation)

    def hybrid_forward(self, F, x):
        w = F.contrib.AdaptiveAvgPooling2D(x, output_size=1)
        w = self.conv1(w)
        w = self.activ(w)
        w = self.conv2(w)
        w = self.sigmoid(w)
        x = F.broadcast_mul(x, w)
        return x

class ResBlock(HybridBlock):
    def __init__(self,
                 in_channels,
                 out_channels,
                 strides,
                 bn_use_global_stats=False,
                 **kwargs):
        super(ResBlock, self).__init__(**kwargs)
        with self.name_scope():
            self.conv1 = conv3x3_block(
                in_channels=in_channels,
                out_channels=out_channels,
                strides=strides,
                bn_use_global_stats=bn_use_global_stats)
            self.conv2 = conv3x3_block(
                in_channels=out_channels,
                out_channels=out_channels,
                bn_use_global_stats=bn_use_global_stats,
                activation=None)

    def hybrid_forward(self, F, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class ResBottleneck(HybridBlock):
    def __init__(self,
                 in_channels,
                 out_channels,
                 strides,
                 padding=1,
                 dilation=1,
                 bn_use_global_stats=False,
                 conv1_stride=False,
                 bottleneck_factor=4,
                 **kwargs):
        super(ResBottleneck, self).__init__(**kwargs)
        mid_channels = out_channels // bottleneck_factor

        with self.name_scope():
            self.conv1 = conv1x1_block(
                in_channels=in_channels,
                out_channels=mid_channels,
                strides=(strides if conv1_stride else 1),
                bn_use_global_stats=bn_use_global_stats)
            self.conv2 = conv3x3_block(
                in_channels=mid_channels,
                out_channels=mid_channels,
                strides=(1 if conv1_stride else strides),
                padding=padding,
                dilation=dilation,
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



class ResUnit(HybridBlock):
    def __init__(self,
                 in_channels,
                 out_channels,
                 strides,
                 padding=1,
                 dilation=1,
                 bn_use_global_stats=False,
                 bottleneck=True,
                 conv1_stride=False,
                 **kwargs):
        super(ResUnit, self).__init__(**kwargs)
        self.resize_identity = (in_channels != out_channels) or (strides != 1)

        with self.name_scope():
            if bottleneck:
                self.body = ResBottleneck(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    strides=strides,
                    padding=padding,
                    dilation=dilation,
                    bn_use_global_stats=bn_use_global_stats,
                    conv1_stride=conv1_stride)
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
