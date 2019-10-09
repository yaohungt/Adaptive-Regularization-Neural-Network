# -*- coding: utf-8 -*-
# Description: Perform Training

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

def customized_criterion(args):
    if args.dataset == 'MNIST' or args.dataset == 'CIFAR10' or args.dataset == 'CIFAR100':
        return nn.CrossEntropyLoss()

def bayes_training(myNet, optimizer, train_loader, args):
    myNet.train()
    
    criterion = customized_criterion(args)
    
    for k in range(10):
        # optimize the model parameters
        for t in range(30):
            print('in epoch ' + str(k*30 + t) + '.....')
            for xs, ys in train_loader:
                if torch.cuda.is_available():
                    xs, ys = xs.cuda(), ys.cuda()
                xs, ys = Variable(xs), Variable(ys)
                
                optimizer.zero_grad()
                
                y_pred = myNet(xs)
                closs = criterion(y_pred, ys)
                rloss_list = myNet.regularizer()
                rloss = 0
                for i in range(len(rloss_list)):
                    rloss = rloss + rloss_list[i]
                
                loss = closs + args.rho * rloss
                loss.backward()
                optimizer.step()
                
        # Optimize covariance matrices, using SVD closed form solutions.
        myNet.update_covs(args.lower_threshold, args.upper_threshold)

def evaluation(myNet, test_loader, args):
    myNet.eval()
    
    num_correct = 0
    num_total = 0
    with torch.no_grad():
        for xs, ys in test_loader:
            if torch.cuda.is_available():
                xs, ys = xs.cuda(), ys.cuda()
            xs, ys = Variable(xs), Variable(ys)
            ypreds = myNet(xs)
            _, preds = torch.max(ypreds, -1)
            num_correct += float((preds == ys).int().sum().data) # because (preds == ys) is only 8-bit range 
            num_total += len(preds)
        
    accuracy = 100.0 * num_correct / float(num_total)
    return accuracy

def bayes_training_evaluation(myNet, optimizer, train_loader, test_loader, args):
    myNet.train()
    
    criterion = customized_criterion(args)
    
    _inside_acc = np.zeros((300, 1))
    
    for k in range(10):
        # optimize the model parameters
        #print('in outer_loop ' + str(k) + '.....')
        for t in range(30):
            print('in epoch ' + str(k*30 + t) + '.....')
            for xs, ys in train_loader:
                if torch.cuda.is_available():
                    xs, ys = xs.cuda(), ys.cuda()
                xs, ys = Variable(xs), Variable(ys)
                
                optimizer.zero_grad()
                
                y_pred = myNet(xs)
                closs = criterion(y_pred, ys)
                rloss_list = myNet.regularizer()
                rloss = 0
                for i in range(len(rloss_list)):
                    rloss = rloss + rloss_list[i]
                
                loss = closs + args.rho * rloss
                loss.backward()
                optimizer.step()
                
            acc = evaluation(myNet, test_loader, args)
            print(acc)
            _inside_acc[k*30+t] = acc
                
        # Optimize covariance matrices, using SVD closed form solutions.
        myNet.update_covs(args.lower_threshold, args.upper_threshold)

        
    return _inside_acc
