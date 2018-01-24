#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 13:47:03 2018

@author: chinwei
"""


import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import torchvision
import torchvision.transforms

from_numpy = torch.from_numpy


batch_size = 64
num_epochs = 10
cuda = False
store_every = 1000
lr0 = 0.02

mnist_transforms = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor()])
mnist_train = torchvision.datasets.MNIST(
        root='./data', train=True, 
        transform=mnist_transforms, download=True)
mnist_test = torchvision.datasets.MNIST(
        root='./data', train=False, 
        transform=mnist_transforms, download=True)

train_loader = torch.utils.data.DataLoader(
        mnist_train, batch_size=batch_size, shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(
        mnist_test, batch_size=batch_size, shuffle=True, num_workers=2)


# building model
class ResLinear(nn.Module):

    def __init__(self, in_features, out_features, activation=nn.ReLU()):
        super(ResLinear, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        
        self.linear = nn.Linear(in_features, out_features)
        if in_features != out_features:
            self.project_linear = nn.Linear(in_features, out_features)
        
    def forward(self, x):
        inner = self.activation(self.linear(x))
        if self.in_features != self.out_features:
            skip = self.project_linear(x)
        else:
            skip = x
        return inner + skip
        
        
model = nn.Sequential(
    ResLinear(784, 312),
    nn.ReLU(),
    ResLinear(312, 312),
    nn.ReLU(),
    ResLinear(312, 10)
)
if cuda:
    model = model.cuda()


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=lr0)

def adjust_lr(optimizer, epoch, total_epochs):
    lr = lr0 * (0.1 ** (epoch / float(total_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(proba, y):
    correct = torch.eq(proba.max(1)[1], y).sum().type(torch.FloatTensor)
    return correct / y.size(0)
    
    
def evaluate(dataset_loader, criterion):
    LOSSES = 0
    COUNTER = 0
    for batch in dataset_loader:
        optimizer.zero_grad()

        x, y = batch
        x = Variable(x).view(-1,784)
        y = Variable(y).view(-1)
        if cuda:
            x = x.cuda()
            y = y.cuda()
            
        loss = criterion(model(x), y)
        n = y.size(0)
        LOSSES += loss.sum().data.cpu().numpy() * n
        COUNTER += n
    
    return LOSSES / float(COUNTER)

def train_model():
    
    LOSSES = 0
    COUNTER = 0
    ITERATIONS = 0
    learning_curve_nll_train = list()
    learning_curve_nll_test = list()
    learning_curve_acc_train = list()
    learning_curve_acc_test = list()
    for e in range(num_epochs):
        for batch in train_loader:
            optimizer.zero_grad()

            x, y = batch
            x = Variable(x).view(-1,784)
            y = Variable(y).view(-1)
            if cuda:
                x = x.cuda()
                y = y.cuda()
                
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            
            n = y.size(0)
            LOSSES += loss.sum().data.cpu().numpy() * n
            COUNTER += n
            ITERATIONS += 1
            if ITERATIONS%(store_every/5) == 0:
                avg_loss = LOSSES / float(COUNTER)
                LOSSES = 0
                COUNTER = 0
                print(" Iteration {}: TRAIN {}".format(
                    ITERATIONS, avg_loss))
        
            if ITERATIONS%(store_every) == 0:     
                
                train_loss = evaluate(train_loader, criterion)
                learning_curve_nll_train.append(train_loss)
                test_loss = evaluate(test_loader, criterion)
                learning_curve_nll_test.append(test_loss)
                
                train_acc = evaluate(train_loader, accuracy)
                learning_curve_acc_train.append(train_acc)
                test_acc = evaluate(test_loader, accuracy)
                learning_curve_acc_test.append(test_acc)
                        
                print(" [NLL] TRAIN {} / TEST {}".format(
                    train_loss, test_loss))
                print(" [ACC] TRAIN {} / TEST {}".format(
                    train_acc, test_acc))
        
        adjust_lr(optimizer, e+1, num_epochs)
        
    return learning_curve_nll_train, \
           learning_curve_nll_test, \
           learning_curve_acc_train, \
           learning_curve_acc_test, 
           

if __name__ == '__main__':
    
    _ = train_model()


