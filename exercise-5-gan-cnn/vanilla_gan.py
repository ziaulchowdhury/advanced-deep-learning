# -*- coding: utf-8 -*-
"""
Created on 

@author: ziaul
"""

import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision 
from torchvision import transforms
from torch.utils.data import DataLoader, random_split


def load_mnist_dataset(batch_size=64, train_val_split_size=[55000, 5000]):

    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
    train_dataset, val_dataset = random_split(train_dataset, train_val_split_size)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    
    return train_loader, test_loader, val_loader

# X_temp, y_temp = next(iter(train_loader_mnist))

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / np.sqrt(in_dim / 2.)
    return Variable(torch.randn(*size) * xavier_stddev, requires_grad=True)


class Generator:
    
    def __init__(self, Z_dim, h_dim, X_dim):
        self.Z_dim = Z_dim
        self.h_dim = h_dim
        self.X_dim = X_dim
        
        self.Wzh = xavier_init(size=[self.Z_dim, self.h_dim])
        self.bzh = Variable(torch.zeros(h_dim), requires_grad=True)
        
        self.Whx = xavier_init(size=[self.h_dim, self.X_dim])
        self.bhx = Variable(torch.zeros(self.X_dim), requires_grad=True)
        
        self.G_params = [self.Wzh, self.bzh, self.Whx, self.bhx]

    def G(self, z):
        h = F.relu(z @ self.Wzh + self.bzh.repeat(z.size(0), 1))
        X = torch.sigmoid(h @ self.Whx + self.bhx.repeat(h.size(0), 1))
        return X


class Discriminator:
    
    def __init__(self, X_dim, h_dim):
        self.X_dim = X_dim
        self.h_dim = h_dim
        
        self.Wxh = xavier_init(size=[self.X_dim, self.h_dim])
        self.bxh = Variable(torch.zeros(self.h_dim), requires_grad=True)
        
        self.Why = xavier_init(size=[self.h_dim, 1])
        self.bhy = Variable(torch.zeros(1), requires_grad=True)
        
        self.D_params = [self.Wxh, self.bxh, self.Why, self.bhy]

    def D(self, X):
        h = F.relu(X @ self.Wxh + self.bxh.repeat(X.size(0), 1))
        y = torch.sigmoid(h @ self.Why + self.bhy.repeat(h.size(0), 1))
        return y


def plot_image(samples):
    
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    if not os.path.exists('out_mnist/'):
        os.makedirs('out_mnist/')

    plt.savefig('out_mnist/{}.png'.format(str(c).zfill(3)), bbox_inches='tight')    
    plt.close(fig)


def reset_grad(params):
    for p in params:
        if p.grad is not None:
            data = p.grad.data
            p.grad = Variable(data.new().resize_as_(data).zero_())

def train_test_model(iterations, mb_size, Z_dim, train_loader, generator, discriminator, params, use_original_loss_function=True):
    
    for it in range(iterations): 
        
        # Sample data
        z = Variable(torch.randn(mb_size, Z_dim))
        X, _ = next(iter(train_loader))
        X = X.view(-1, 784)
        X = Variable(X)
    
        # Dicriminator forward-loss-backward-update
        G_sample = generator.G(z)
        D_real = discriminator.D(X)
        D_fake = discriminator.D(G_sample)
    
        D_loss = -torch.mean(torch.log(D_real) + torch.log(1. - D_fake))
    
        D_loss.backward()
        D_solver.step()
    
        # Reset gradient
        reset_grad(params)
    
        # Generator forward-loss-backward-update
        z = Variable(torch.randn(mb_size, Z_dim))
        G_sample = generator.G(z)
        D_fake = discriminator.D(G_sample)
    
        if use_original_loss_function:
            G_loss = -torch.mean(torch.log(D_fake))  # Loss from research paper
        else:
            G_loss = F.binary_cross_entropy(D_fake, ones_label)
    
        G_loss.backward()
        G_solver.step()
    
        # Reset gradient
        reset_grad(params)
    
        # Print loss values of discriminator and the generator and plot images
        if it % 10 == 0:
            print(f'Iter-{it}; D_loss: {D_loss.data.numpy()}; G_loss: {G_loss.data.numpy()}')
            samples = generator.G(z).data.numpy()[:16]
            plot_image(samples)
            
            global c
            c += 1


if __name__ == '__main__':
    
    mb_size = 64
    Z_dim = 100
    X_dim = 28*28
    h_dim = 128
    c = 0
    learning_rate = 1e-3
    
    discriminator = Discriminator(X_dim, h_dim)
    generator = Generator(Z_dim, h_dim, X_dim)
    
    G_solver = optim.Adam(generator.G_params, lr=learning_rate)
    D_solver = optim.Adam(discriminator.D_params, lr=learning_rate)
    
    ones_label = Variable(torch.ones(mb_size, 1))
    zeros_label = Variable(torch.zeros(mb_size, 1))
    
    train_loader, test_loader, val_loader = load_mnist_dataset()
    
    params = generator.G_params + discriminator.D_params
        
    # Task 1.1: Run with original loss function
    # train_test_model(1000, mb_size, Z_dim, train_loader, generator, discriminator, params, use_original_loss_function=True)
    
    # Task 1.2: Run with a different loss function
    # train_test_model(1000, mb_size, Z_dim, train_loader, generator, discriminator, params, use_original_loss_function=False)
    
    # Task 1.3: Run for 20K iterations
    # train_test_model(20000, mb_size, Z_dim, train_loader, generator, discriminator, params, use_original_loss_function=False)
    
    # Task 1.4: Run for 100K iterations
    train_test_model(100000, mb_size, Z_dim, train_loader, generator, discriminator, params, use_original_loss_function=True)
    