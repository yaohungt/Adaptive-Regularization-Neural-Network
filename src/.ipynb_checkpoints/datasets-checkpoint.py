# -*- coding: utf-8 -*-
# Description: Datasets
import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler

def get_Dataset(args):
    if args.dataset == 'MNIST':
        return get_MNIST(args)
    elif args.dataset == 'CIFAR10' or args.dataset == 'CIFAR100':
        return get_CIFAR(args)

def get_MNIST(args):

    
    trains = datasets.MNIST(args.dataset_path, train=True, download=True,
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))
                            ]))
    tests = datasets.MNIST(args.dataset_path, train=False,
                           transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))
                            ]))
    
    kwargs = {"num_workers": 1, "pin_memory": True} if torch.cuda.is_available() else {}
    
    
    if args.trainAll == 1:
        train_loader = torch.utils.data.DataLoader(trains, batch_size=args.batch_size, shuffle=True, **kwargs)
    else:
        #np.random.seed(seed = args.trainSize_seed)
        idx = np.arange(len(trains)) # len(trains) = 60000
        np.random.shuffle(idx)
        train_idx = idx[:args.trainSize]
        train_sampler = SubsetRandomSampler(train_idx)
        train_loader = torch.utils.data.DataLoader(trains, batch_size=args.batch_size, sampler=train_sampler, **kwargs)
    
    test_loader = torch.utils.data.DataLoader(tests, batch_size=args.batch_size, shuffle=True, **kwargs)
    
    return train_loader, test_loader

def get_CIFAR(args):

    tr_transform = transforms.Compose(
        [transforms.RandomCrop(32, padding=4),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]) if args.resnet == 1 else transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    
    te_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    if args.dataset == 'CIFAR10':
        trains = datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=tr_transform)

        tests = datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=te_transform)
    elif args.dataset == 'CIFAR100':
        trains = datasets.CIFAR100(root='./data', train=True,
                                        download=True, transform=tr_transform)

        tests = datasets.CIFAR100(root='./data', train=False,
                                       download=True, transform=te_transform)
    
    
    kwargs = {"num_workers": 1, "pin_memory": True} if torch.cuda.is_available() else {}
    
    
    if args.trainAll == 1:
        train_loader = torch.utils.data.DataLoader(trains, batch_size=args.batch_size, shuffle=True, **kwargs)
    else:
        #np.random.seed(seed = args.trainSize_seed)
        idx = np.arange(len(trains)) # len(trains) = 60000
        np.random.shuffle(idx)
        train_idx = idx[:args.trainSize]
        train_sampler = SubsetRandomSampler(train_idx)
        train_loader = torch.utils.data.DataLoader(trains, batch_size=args.batch_size, sampler=train_sampler, **kwargs)
    
    test_loader = torch.utils.data.DataLoader(tests, batch_size=args.batch_size, shuffle=True, **kwargs)
    
    return train_loader, test_loader
