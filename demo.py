# -*- coding: utf-8 -*-
# Description: Demo code for NeurIPS 2019 paper: Learning Neural Networks with Adaptive Regularization

import argparse
import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

from src import datasets
from src import train_eval
from src import model
import pickle

#### args
parser = argparse.ArgumentParser()
##### dataset
parser.add_argument("-d", "--dataset", help="Dataset for experiment.", type=str, default='CIFAR10')
parser.add_argument("-dp", "--dataset_path", help="Path for dataset.", type=str, default='./data')
parser.add_argument("-b", "--batch_size", help="Batch size.", type=int, default=256)
parser.add_argument("-tA", "--trainPartial", help="Train on entire (1) or partial (0) dataset.", action='store_true')
parser.add_argument("-bn", "--BatchNorm", help="Use batch normalization or not.", action='store_true')
## only works if args.trainPartial is True
parser.add_argument("-s", "--trainSize_seed", help="Random seed for num of training data.", type=int, default=2)
parser.add_argument("-tS", "--trainSize", help="Num of training data.", type=int, default=50000)
##

parser.add_argument("-t", "--tau", help="Coefficient for the L2-regularization", type=float, default=1e-2)
parser.add_argument("-e", "--inner_epoch", help="Number of training epochs.", type=int, default=30)
parser.add_argument("-r", "--lrate", help="Learning rate.", type=float, default=5e-4)
parser.add_argument("-dpR", "--dp_rate", help="Dropout Rate. 0.0 if no dropout rate.", type=float, default=0.0) 

##### Bayes Regularization
parser.add_argument("-rT", "--regu_type", help="Regularization type: ALL, CONV, FC, or LAST", type=str, default='LAST')
parser.add_argument("-p", "--rho", help="Coefficient for the trace-regularization", type=float, default=1e-4)
parser.add_argument("-ol", "--outer_loop", help="Number of outer loops for coordinate descent.", type=int, default=10)
parser.add_argument("-lT", "--lower_threshold", help="Lower bound of the smallest singular value.", type=float, default=1e-3)
parser.add_argument("-uT", "--upper_threshold", help="Upper bound of the largest singular value.", type=float, default=1e3)

# Compile and configure all the model parameters.
args = parser.parse_args()

train_loader, test_loader = datasets.get_Dataset(args)
    
myNet = model.BayesNet(args)

optimizer = optim.Adam(filter(lambda p: p.requires_grad, myNet.parameters()), lr=args.lrate, weight_decay=args.tau)

acc = train_eval.bayes_training_evaluation(myNet, optimizer, train_loader, test_loader, args)
print(acc)
