# -*- coding: utf-8 -*-
# Description: A simple ConvNet for CIFAR10

import torch
from torch import nn
import torch.nn.functional as F

class CIFAR_Net(nn.Module):
    def __init__(self, args):
        super(CIFAR_Net, self).__init__()
        
        # every layer 
        self.conv1 = torch.nn.Sequential(
            nn.Conv2d( 3, 10, kernel_size = 5),
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Dropout(p = args.dp_rate),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1))
        ) if args.BatchNorm else torch.nn.Sequential(
            nn.Conv2d( 3, 10, kernel_size = 5),
            nn.ReLU(),
            nn.Dropout(p = args.dp_rate),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1))
        ) # 10 * 14 * 14
        
        self.conv2 = torch.nn.Sequential(
            nn.Conv2d( 10, 20, kernel_size = 5),
            nn.BatchNorm2d(20),
            nn.ReLU(),
            nn.Dropout(p = args.dp_rate),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1))
        ) if args.BatchNorm else torch.nn.Sequential(
            nn.Conv2d( 10, 20, kernel_size = 5),
            nn.ReLU(),
            nn.Dropout(p = args.dp_rate),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1))
        ) # 20 * 5 * 5
        
        self.fc1 = torch.nn.Sequential(
            nn.Linear(500, 500),
            nn.BatchNorm1d(500),
            nn.Dropout(p = args.dp_rate),
            nn.ReLU()
        ) if args.BatchNorm else torch.nn.Sequential(
            nn.Linear(500, 500),
            nn.Dropout(p = args.dp_rate),
            nn.ReLU()
        )
        
        self.fc2 = torch.nn.Sequential(
            nn.Linear(500, 500),
            nn.BatchNorm1d(500),
            nn.Dropout(p = args.dp_rate),
            nn.ReLU()
        ) if args.BatchNorm else torch.nn.Sequential(
            nn.Linear(500, 500),
            nn.Dropout(p = args.dp_rate),
            nn.ReLU()
        )
        
        self.fc3 = torch.nn.Sequential(
            nn.Linear(500, 10),
        ) 
        
    def forward(self, f, if_decov = False):
        f = self.conv1(f)
        f = self.conv2(f)
        f = f.view(-1, 500)
        f = self.fc1(f)
        feat = self.fc2(f)
        return self.fc3(feat)

class MNIST_Net(nn.Module):
    def __init__(self, args):
        super(MNIST_Net, self).__init__()
        
        # every layer 
        self.conv1 = torch.nn.Sequential(
            nn.Conv2d( 1, 10, kernel_size = 5),
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Dropout(p = args.dp_rate),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1))
        ) if args.BatchNorm else torch.nn.Sequential(
            nn.Conv2d( 1, 10, kernel_size = 5),
            nn.ReLU(),
            nn.Dropout(p = args.dp_rate),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1))
        )
        
        self.conv2 = torch.nn.Sequential(
            nn.Conv2d( 10, 20, kernel_size = 5),
            nn.BatchNorm2d(20),
            nn.ReLU(),
            nn.Dropout(p = args.dp_rate),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1))
        ) if args.BatchNorm else torch.nn.Sequential(
            nn.Conv2d( 10, 20, kernel_size = 5),
            nn.ReLU(),
            nn.Dropout(p = args.dp_rate),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1))
        )
        
        self.fc1 = torch.nn.Sequential(
            nn.Linear(320, 50),
            nn.BatchNorm1d(50),
            nn.Dropout(p = args.dp_rate),
            nn.ReLU()
        ) if args.BatchNorm else torch.nn.Sequential(
            nn.Linear(320, 50),
            nn.Dropout(p = args.dp_rate),
            nn.ReLU()
        )
        
        
        self.fc2 = torch.nn.Sequential(
            nn.Linear(50, 10),
        )
        
    def forward(self, f):
        f = self.conv1(f)
        f = self.conv2(f)
        f = f.view(-1, 320)
        f = self.fc1(f)
        return self.fc2(f)
        
def BayesNet(args):
    if args.dataset == 'MNIST':
        net_name = MNIST_Net
    elif args.dataset == 'CIFAR10' or args.dataset == 'CIFAR100':
        net_name = CIFAR_Net
    class OurNet(net_name):
        """
        The network with empirical bayes assumptions on weights.
        We're going to update cov with this class, while the update for weights we simply use gradient descent method.
        """
        def __init__(self, args):
            super(OurNet, self).__init__(args)
        
            if torch.cuda.is_available():
                self.cuda()
                
            self.sqrt_cov_prev = []
            self.sqrt_cov_next = []
            self.cov_weight  = []
    
            if args.regu_type == 'LAST':
                name, module = list(self.named_children())[-1]
                # print(name)
                shape = list(module.parameters())[0].shape
                sqt_cov_pre = nn.Parameter(torch.eye(shape[1]), requires_grad = False)
                sqt_cov_nex = nn.Parameter(torch.eye(shape[0]), requires_grad = False)
                if torch.cuda.is_available():
                    sqt_cov_pre = sqt_cov_pre.cuda()
                    sqt_cov_nex = sqt_cov_nex.cuda()
                self.sqrt_cov_prev.append(sqt_cov_pre)
                self.sqrt_cov_next.append(sqt_cov_nex)
                self.cov_weight.append(list(module.parameters())[0])
            else:
                for name, module in self.named_children():
                    # print(name)
                    shape = list(module.parameters())[0].shape
                    if len(shape) == 4: # conv layer
                        if args.regu_type == 'ALL' or args.regu_type == 'CONV':
                            sqt_cov_pre = nn.Parameter(torch.eye(shape[1] * shape[2] * shape[3]), requires_grad = False)
                            sqt_cov_nex = nn.Parameter(torch.eye(shape[0]), requires_grad = False)
                            if torch.cuda.is_available():
                                sqt_cov_pre = sqt_cov_pre.cuda()
                                sqt_cov_nex = sqt_cov_nex.cuda()
                            # we view conv weights as a 2d matrix (shape[0], shape[1] * shape[2] * shape[3])
                            self.sqrt_cov_prev.append(sqt_cov_pre)
                            self.sqrt_cov_next.append(sqt_cov_nex)
                            self.cov_weight.append(list(module.parameters())[0].view(shape[0], -1))
                    elif len(shape) == 2: # MLP layer
                        if args.regu_type == 'ALL' or args.regu_type == 'FC':
                            sqt_cov_pre = nn.Parameter(torch.eye(shape[1]), requires_grad = False)
                            sqt_cov_nex = nn.Parameter(torch.eye(shape[0]), requires_grad = False)
                            if torch.cuda.is_available():
                                sqt_cov_pre = sqt_cov_pre.cuda()
                                sqt_cov_nex = sqt_cov_nex.cuda()
                            self.sqrt_cov_prev.append(sqt_cov_pre)
                            self.sqrt_cov_next.append(sqt_cov_nex)
                            self.cov_weight.append(list(module.parameters())[0])
                    
        def regularizer(self):
            """
            Compute the weight regularizer in the network (layer by layer).
            """
            r = []
            for i in range(len(self.cov_weight)):
                r_sqrt = torch.mm(torch.mm(self.sqrt_cov_next[i], self.cov_weight[i]), self.sqrt_cov_prev[i])
                r.append(torch.sum(r_sqrt * r_sqrt))
            return r
        
        def _thresholding(self, sv, lower, upper):
            """
            Two-way soft-thresholding of singular values.
            :param sv:  A list of singular values.
            :param lower:   Lower bound for soft-thresholding.
            :param upper:   Upper bound for soft-thresholding.
            :return:    Thresholded singular values.
            """
            uidx = sv > upper
            lidx = sv < lower
            sv[uidx] = upper
            sv[lidx] = lower
            return sv
        def update_covs(self, lower, upper):
            """
            Layer by layer, update both the covariance matrix over tasks and over features, using the closed form solutions. 
            :param lower: Lower bound of the truncation.
            :param upper: Upper bound of the truncation.
            """
            for i in range(len(self.cov_weight)):
                cov_next = torch.mm(self.sqrt_cov_next[i], self.sqrt_cov_next[i].t())
                
                # dim: {(p x n) * (n x n)} * (n x p) = (p x p)
                cov_prev_weight = torch.mm(torch.mm(self.cov_weight[i].t(), cov_next), self.cov_weight[i])
                
                # compute SVD
                # U, S, V = SVD(A): A is the input matrix
                u, s, _ = torch.svd(cov_prev_weight.data)
                
                # inverse and do truncation on inverse singular values
                s = s.shape[0] / s
                s = self._thresholding(s, lower, upper)
                
                # recompute the sqrt_cov
                s = torch.sqrt(s)
                # dim for u and s: (p x p)
                self.sqrt_cov_prev[i].data = torch.mm(torch.mm(u, torch.diag(s)), u.t())
                
            for i in range(len(self.cov_weight)):
                cov_prev = torch.mm(self.sqrt_cov_prev[i], self.sqrt_cov_prev[i].t())
                
                # dim: {(n x p) * (p x p)} * (p x n) = (n x n)
                cov_next_weight = torch.mm(torch.mm(self.cov_weight[i], cov_prev), self.cov_weight[i].t())
                
                # compute SVD
                # U, S, V = SVD(A): A is the input matrix
                u, s, _ = torch.svd(cov_next_weight.data)
                
                # inverse and do truncation on inverse singular values
                s = s.shape[0] / s
                s = self._thresholding(s, lower, upper)
                
                # recompute the sqrt_cov
                s = torch.sqrt(s)
                # dim for u and s: (p x p)
                self.sqrt_cov_next[i].data = torch.mm(torch.mm(u, torch.diag(s)), u.t())

    return OurNet(args)   

