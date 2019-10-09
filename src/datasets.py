# -*- coding: utf-8 -*-
# Description: Datasets
import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler

def get_Dataset(args):
    if args.dataset == 'MNIST':
        tr_transform = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))
                            ])
        te_transform = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))
                            ])
    elif args.dataset == 'CIFAR10' or args.dataset == 'CIFAR100':
        tr_transform = transforms.Compose(
            [transforms.RandomCrop(32, padding=4),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        
        te_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    
    trains = getattr(datasets, args.dataset)(root='./data', train=True,
                                        download=True, transform=tr_transform)

    tests = getattr(datasets, args.dataset)(root='./data', train=False,
                                       download=True, transform=te_transform)

    kwargs = {"num_workers": 1, "pin_memory": True} if torch.cuda.is_available() else {}
    
    if args.trainPartial:
        #np.random.seed(seed = args.trainSize_seed)
        idx = np.arange(len(trains)) # len(trains) = 60000
        np.random.shuffle(idx)
        train_idx = idx[:args.trainSize]
        train_sampler = SubsetRandomSampler(train_idx)
        train_loader = torch.utils.data.DataLoader(trains, batch_size=args.batch_size, sampler=train_sampler, **kwargs)
    else:
        train_loader = torch.utils.data.DataLoader(trains, batch_size=args.batch_size, shuffle=True, **kwargs)
    
    test_loader = torch.utils.data.DataLoader(tests, batch_size=args.batch_size, shuffle=True, **kwargs)
    
    return train_loader, test_loader
