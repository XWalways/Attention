import argparse
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
#back > 0.03
from back import Bone, utils

from models.srm_resnet import cifar_resnet32, cifar_se_resnet32,\
    cifar_srm_resnet32


import torchvision
import torchvision.transforms as transforms

normalize = transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010))

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    normalize
])


def get_datasets(data_dir):
    return {
        'train': torchvision.datasets.CIFAR10(root=data_dir,
                                              train=True,
                                              download=True,
                                              transform=transform_train),
        'val': torchvision.datasets.CIFAR10(root=data_dir,
                                            train=False,
                                            download=True,
                                            transform=transform_test)
    }

data_dir = 'cifar10'
model_names = ['resnet', 'senet', 'srmnet']
num_classes = 10
batch_size = 128
epochs_count = 100
num_workers = 8

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', required=True, choices=model_names)
args = parser.parse_args()

datasets = get_datasets(data_dir)

if args.model_name == 'resnet':
    model = cifar_resnet32(num_classes=num_classes)
elif args.model_name == 'senet':
    model = cifar_se_resnet32(num_classes=num_classes)
elif args.model_name == 'srmnet':
    model = cifar_srm_resnet32(num_classes=num_classes)

optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9,
                      weight_decay=1e-4)

scheduler = MultiStepLR(optimizer, [70, 80], 0.1)
criterion = nn.CrossEntropyLoss()

backbone = Bone(model,
                datasets,
                criterion,
                optimizer,
                scheduler=scheduler,
                scheduler_after_ep=False,
                metric_fn=utils.accuracy_metric,
                metric_increase=True,
                batch_size=batch_size,
                num_workers=num_workers,
                weights_path=f'weights/cifar_best_{args.model_name}.pth',
                log_dir=f'logs/cifar_{args.model_name}')

backbone.fit(epochs_count)
