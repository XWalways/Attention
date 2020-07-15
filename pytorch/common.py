import os
import math
import random
import shutil
from PIL import Image
from torchvision.transforms import Resize
from torchvision.transforms import *
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import threading
import numpy as np
from collections import defaultdict
import PIL, PIL.ImageOps, PIL.ImageEnhance, PIL.ImageDraw

RESAMPLE_MODE=PIL.Image.BICUBIC#PIL.Image.BILINEAR#

RANDOM_MIRROR = True

__all__=['accuracy', 'torch_dist_sum', 'get_dataset', 'AverageMeter',
         'MixUpWrapper', 'LR_Scheduler', 'get_transform', 'save_checkpoint',
         'LabelSmoothing', 'NLLMultiLabelSmooth']

class LabelSmoothing(nn.Module):
    """
    NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.1):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

class NLLMultiLabelSmooth(nn.Module):
    def __init__(self, smoothing = 0.1):
        super(NLLMultiLabelSmooth, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        if self.training:
            x = x.float()
            target = target.float()
            logprobs = torch.nn.functional.log_softmax(x, dim = -1)

            nll_loss = -logprobs * target
            nll_loss = nll_loss.sum(-1)

            smooth_loss = -logprobs.mean(dim=-1)

            loss = self.confidence * nll_loss + self.smoothing * smooth_loss

            return loss.mean()



def save_checkpoint(state, args, is_best, filename='checkpoint.pth.tar'):
    """Saves checkpoint to disk"""
    if hasattr(args, 'backbone'):
        directory = "runs/%s/%s/%s/%s/"%(args.dataset, args.model, args.backbone, args.checkname)
    else:
        directory = "runs/%s/%s/%s/"%(args.dataset, args.model, args.checkname)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, directory + 'model_best.pth.tar')

def get_dataset(name, **kwargs):
    if name == 'imagenet':
        return ImageNetDataset(**kwargs)


class ImageNetDataset(datasets.ImageFolder):
    BASE_DIR = "ILSVRC2012"
    def __init__(self, root=os.path.expanduser('~/.encoding/data'), transform=None,
                 target_transform=None, train=True, **kwargs):
        split='train' if train == True else 'val'
        root = os.path.join(root, self.BASE_DIR, split)
        super(ImageNetDataset, self).__init__(
            root, transform, target_transform)

def ShearX(img, v, resample=RESAMPLE_MODE):  # [-0.3, 0.3]
    assert -0.3 <= v <= 0.3
    if RANDOM_MIRROR and random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0),
                         resample=resample)

def ShearY(img, v, resample=RESAMPLE_MODE):  # [-0.3, 0.3]
    assert -0.3 <= v <= 0.3
    if RANDOM_MIRROR and random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0),
                         resample=resample)


def TranslateX(img, v, resample=RESAMPLE_MODE):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert -0.45 <= v <= 0.45
    if RANDOM_MIRROR and random.random() > 0.5:
        v = -v
    v = v * img.size[0]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0),
                         resample=resample)


def TranslateY(img, v, resample=RESAMPLE_MODE):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert -0.45 <= v <= 0.45
    if RANDOM_MIRROR and random.random() > 0.5:
        v = -v
    v = v * img.size[1]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v),
                         resample=resample)


def TranslateXabs(img, v, resample=RESAMPLE_MODE):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert 0 <= v
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0),
                         resample=resample)


def TranslateYabs(img, v, resample=RESAMPLE_MODE):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert 0 <= v
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v),
                         resample=resample)


def Rotate(img, v):  # [-30, 30]
    assert -30 <= v <= 30
    if RANDOM_MIRROR and random.random() > 0.5:
        v = -v
    return img.rotate(v)


def AutoContrast(img, _):
    return PIL.ImageOps.autocontrast(img)


def Invert(img, _):
    return PIL.ImageOps.invert(img)


def Equalize(img, _):
    return PIL.ImageOps.equalize(img)


def Flip(img, _):  # not from the paper
    return PIL.ImageOps.mirror(img)


def Solarize(img, v):  # [0, 256]
    assert 0 <= v <= 256
    return PIL.ImageOps.solarize(img, v)


def SolarizeAdd(img, addition=0, threshold=128):
    img_np = np.array(img).astype(np.int)
    img_np = img_np + addition
    img_np = np.clip(img_np, 0, 255)
    img_np = img_np.astype(np.uint8)
    img = PIL.Image.fromarray(img_np)
    return PIL.ImageOps.solarize(img, threshold)


def Posterize(img, v):  # [4, 8]
    #assert 4 <= v <= 8
    v = int(v)
    return PIL.ImageOps.posterize(img, v)

def Contrast(img, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Contrast(img).enhance(v)


def Color(img, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Color(img).enhance(v)


def Brightness(img, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Brightness(img).enhance(v)


def Sharpness(img, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Sharpness(img).enhance(v)


def CutoutAbs(img, v):  # [0, 60] => percentage: [0, 0.2]
    # assert 0 <= v <= 20
    if v < 0:
        return img
    w, h = img.size
    x0 = np.random.uniform(w)
    y0 = np.random.uniform(h)

    x0 = int(max(0, x0 - v / 2.))
    y0 = int(max(0, y0 - v / 2.))
    x1 = min(w, x0 + v)
    y1 = min(h, y0 + v)

    xy = (x0, y0, x1, y1)
    color = (125, 123, 114)
    # color = (0, 0, 0)
    img = img.copy()
    PIL.ImageDraw.Draw(img).rectangle(xy, color)
    return img


def Cutout(img, v):  # [0, 60] => percentage: [0, 0.2]
    assert 0.0 <= v <= 0.2
    if v <= 0.:
        return img

    v = v * img.size[0]
    return CutoutAbs(img, v)

def rand_augment_list():  # 16 oeprations and their ranges
    l = [
        (AutoContrast, 0, 1),
        (Equalize, 0, 1),
        (Invert, 0, 1),
        (Rotate, 0, 30),
        (Posterize, 0, 4),
        (Solarize, 0, 256),
        (SolarizeAdd, 0, 110),
        (Color, 0.1, 1.9),
        (Contrast, 0.1, 1.9),
        (Brightness, 0.1, 1.9),
        (Sharpness, 0.1, 1.9),
        (ShearX, 0., 0.3),
        (ShearY, 0., 0.3),
        (CutoutAbs, 0, 40),
        (TranslateXabs, 0., 100),
        (TranslateYabs, 0., 100),
    ]

    return l

class RandAugment(object):
    def __init__(self, n, m):
        self.n = n
        self.m = m
        self.augment_list = rand_augment_list()

    def __call__(self, img):
        ops = random.choices(self.augment_list, k=self.n)
        for op, minval, maxval in ops:
            if random.random() > random.uniform(0.2, 0.8):
                continue
            val = (float(self.m) / 30) * float(maxval - minval) + minval
            img = op(img, val)
        return img


def torch_dist_sum(gpu, *args):
    process_group = torch.distributed.group.WORLD
    tensor_args = []
    pending_res = []
    for arg in args:
        if isinstance(arg, torch.Tensor):
            tensor_arg = arg.clone().reshape(-1).detach().cuda(gpu)
        else:
            tensor_arg = torch.tensor(arg).reshape(-1).cuda(gpu)
        tensor_args.append(tensor_arg)
        pending_res.append(torch.distributed.all_reduce(tensor_arg, group=process_group, async_op=True))
    for res in pending_res:
        res.wait()
    return tensor_args

class LR_Scheduler(object):
    def __init__(self, mode, base_lr, num_epochs, iters_per_epoch=0,
                 lr_step=0, warmup_epochs=0, quiet=False):
        self.mode = mode
        self.quiet = quiet
        if not quiet:
            print('Using {} LR scheduler with warm-up epochs of {}!'.format(self.mode, warmup_epochs))
        if mode == 'step':
            assert lr_step
        self.base_lr = base_lr
        self.lr_step = lr_step
        self.iters_per_epoch = iters_per_epoch
        self.epoch = -1
        self.warmup_iters = warmup_epochs * iters_per_epoch
        self.total_iters = (num_epochs - warmup_epochs) * iters_per_epoch

    def __call__(self, optimizer, i, epoch, best_pred):
        T = epoch * self.iters_per_epoch + i
        # warm up lr schedule
        if self.warmup_iters > 0 and T < self.warmup_iters:
            lr = self.base_lr * 1.0 * T / self.warmup_iters
        elif self.mode == 'cos':
            T = T - self.warmup_iters
            lr = 0.5 * self.base_lr * (1 + math.cos(1.0 * T / self.total_iters * math.pi))
        elif self.mode == 'poly':
            T = T - self.warmup_iters
            lr = self.base_lr * pow((1 - 1.0 * T / self.total_iters), 0.9)
        elif self.mode == 'step':
            lr = self.base_lr * (0.1 ** (epoch // self.lr_step))
        else:
            raise NotImplemented
        if epoch > self.epoch and (epoch == 0 or best_pred > 0.0):
            if not self.quiet:
                print('\n=>Epoch %i, learning rate = %.4f, \
                    previous best = %.4f' % (epoch, lr, best_pred))
            self.epoch = epoch
        assert lr >= 0
        self._adjust_learning_rate(optimizer, lr)

    def _adjust_learning_rate(self, optimizer, lr):
        for i in range(len(optimizer.param_groups)):
            optimizer.param_groups[i]['lr'] = lr

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        #self.val = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        #self.val = val
        self.sum += val * n
        self.count += n

    @property
    def avg(self):
        avg = 0 if self.count == 0 else self.sum / self.count
        return avg

class MixUpWrapper(object):
    def __init__(self, alpha, num_classes, dataloader, device):
        self.alpha = alpha
        self.dataloader = dataloader
        self.num_classes = num_classes
        self.device = device

    def mixup_loader(self, loader):
        def mixup(alpha, num_classes, data, target):
            with torch.no_grad():
                bs = data.size(0)
                c = np.random.beta(alpha, alpha)
                perm = torch.randperm(bs).cuda()

                md = c * data + (1-c) * data[perm, :]
                mt = c * target + (1-c) * target[perm, :]
                return md, mt

        for input, target in loader:
            input, target = input.cuda(self.device), target.cuda(self.device)
            target = torch.nn.functional.one_hot(target, self.num_classes)
            i, t = mixup(self.alpha, self.num_classes, input, target)
            yield i, t

    def __len__(self):
        return len(self.dataloader)

    def __iter__(self):
        return self.mixup_loader(self.dataloader)

class Lighting(object):
    """Lighting noise(AlexNet - style PCA - based noise)"""

    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = eigval
        self.eigvec = eigvec

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone()\
            .mul(alpha.view(1, 3).expand(3, 3))\
            .mul(self.eigval.view(1, 3).expand(3, 3))\
            .sum(1).squeeze()

        return img.add(rgb.view(3, 1, 1).expand_as(img))

class ERandomCrop:
    def __init__(self, imgsize, min_covered=0.1, aspect_ratio_range=(3./4, 4./3),
                 area_range=(0.1, 1.0), max_attempts=10):
        assert 0.0 < min_covered
        assert 0 < aspect_ratio_range[0] <= aspect_ratio_range[1]
        assert 0 < area_range[0] <= area_range[1]
        assert 1 <= max_attempts

        self.imgsize = imgsize
        self.min_covered = min_covered
        self.aspect_ratio_range = aspect_ratio_range
        self.area_range = area_range
        self.max_attempts = max_attempts
        self._fallback = ECenterCrop(imgsize)
        self.resize_method = Resize((imgsize, imgsize), interpolation=Image.BICUBIC)

    def __call__(self, img):
        original_width, original_height = img.size
        min_area = self.area_range[0] * (original_width * original_height)
        max_area = self.area_range[1] * (original_width * original_height)

        for _ in range(self.max_attempts):
            aspect_ratio = random.uniform(*self.aspect_ratio_range)
            height = int(round(math.sqrt(min_area / aspect_ratio)))
            max_height = int(round(math.sqrt(max_area / aspect_ratio)))

            if max_height * aspect_ratio > original_width:
                max_height = (original_width + 0.5 - 1e-7) / aspect_ratio
                max_height = int(max_height)
                if max_height * aspect_ratio > original_width:
                    max_height -= 1

            if max_height > original_height:
                max_height = original_height

            if height >= max_height:
                height = max_height

            height = int(round(random.uniform(height, max_height)))
            width = int(round(height * aspect_ratio))
            area = width * height

            if area < min_area or area > max_area:
                continue
            if width > original_width or height > original_height:
                continue
            if area < self.min_covered * (original_width * original_height):
                continue
            if width == original_width and height == original_height:
                return self._fallback(img)

            x = random.randint(0, original_width - width)
            y = random.randint(0, original_height - height)
            img = img.crop((x, y, x + width, y + height))
            return self.resize_method(img)

        return self._fallback(img)


class ECenterCrop:
    """Crop the given PIL Image and resize it to desired size.
    Args:
        img (PIL Image): Image to be cropped. (0,0) denotes the top left corner of the image.
        output_size (sequence or int): (height, width) of the crop box. If int,
            it is used for both directions
    Returns:
        PIL Image: Cropped image.
    """
    def __init__(self, imgsize):
        self.imgsize = imgsize
        self.resize_method = Resize((imgsize, imgsize), interpolation=Image.BICUBIC)

    def __call__(self, img):
        image_width, image_height = img.size
        image_short = min(image_width, image_height)

        crop_size = float(self.imgsize) / (self.imgsize + 32) * image_short

        crop_height, crop_width = crop_size, crop_size
        crop_top = int(round((image_height - crop_height) / 2.))
        crop_left = int(round((image_width - crop_width) / 2.))
        img = img.crop((crop_left, crop_top, crop_left + crop_width, crop_top + crop_height))
        return self.resize_method(img)


def get_transform(dataset, base_size=None, crop_size=224, rand_aug=False, etrans=True, **kwargs):
    normalize = Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])
    base_size = base_size if base_size is not None else int(1.0 * crop_size / 0.875)
    if dataset == 'imagenet':
        train_transforms = []
        val_transforms = []
        if rand_aug:
            
            train_transforms.append(RandAugment(2, 12))
        if etrans:
            train_transforms.extend([
                ERandomCrop(crop_size),
            ])
            val_transforms.extend([
                ECenterCrop(crop_size),
            ])
            
        else:
            train_transforms.extend([
                RandomResizedCrop(crop_size),
            ])
            val_transforms.extend([
                Resize(base_size),
                CenterCrop(crop_size),
            ])
        train_transforms.extend([
            RandomHorizontalFlip(),
            ColorJitter(0.4, 0.4, 0.4),
            ToTensor(),
            Lighting(0.1, _imagenet_pca['eigval'], _imagenet_pca['eigvec']),
            normalize,
        ])
        val_transforms.extend([
            ToTensor(),
            normalize,
        ])
        transform_train = Compose(train_transforms)
        transform_val = Compose(val_transforms)
    elif dataset == 'minc':
        transform_train = Compose([
            Resize(base_size),
            RandomResizedCrop(crop_size),
            RandomHorizontalFlip(),
            ColorJitter(0.4, 0.4, 0.4),
            ToTensor(),
            Lighting(0.1, _imagenet_pca['eigval'], _imagenet_pca['eigvec']),
            normalize,
        ])
        transform_val = Compose([
            Resize(base_size),
            CenterCrop(crop_size),
            ToTensor(),
            normalize,
        ])
    elif dataset == 'cifar10':
        transform_train = Compose([
            RandomCrop(32, padding=4),
            RandomHorizontalFlip(),
            ToTensor(),
            Normalize((0.4914, 0.4822, 0.4465), 
                      (0.2023, 0.1994, 0.2010)),
        ])
        transform_val = Compose([
            ToTensor(),
            Normalize((0.4914, 0.4822, 0.4465), 
                    (0.2023, 0.1994, 0.2010)),
        ])
    return transform_train, transform_val

_imagenet_pca = {
    'eigval': torch.Tensor([0.2175, 0.0188, 0.0045]),
    'eigvec': torch.Tensor([
        [-0.5675,  0.7192,  0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948,  0.4203],
    ])
}

