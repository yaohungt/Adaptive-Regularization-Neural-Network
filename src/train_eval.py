# -*- coding: utf-8 -*-
# Description: Perform Training

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

criterion = nn.CrossEntropyLoss()

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
    _inside_acc = np.zeros((args.outer_loop*args.inner_epoch, 1))
    
    for k in range(args.outer_loop):
        # optimize the model parameters
        #print('in outer_loop ' + str(k) + '.....')
        for t in range(args.inner_epoch):
            print('in epoch ' + str(k*args.inner_epoch + t) + '.....')
            myNet.train()
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
            _inside_acc[k*args.inner_epoch+t] = acc
                
        # Optimize covariance matrices, using SVD closed form solutions.
        myNet.update_covs(args.lower_threshold, args.upper_threshold)
    return _inside_acc
