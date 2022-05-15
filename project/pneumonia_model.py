# -*- coding: utf-8 -*-
"""
Created on Fri May 13 00:31:02 2022

@author: ziaul
"""

# import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader
from torchvision.utils import make_grid
import torch.nn as nn
import torch.nn.functional as F


def load_pneumonia_dataset(data_dir):
    train_transform = transforms.Compose([
        transforms.Resize(size = (128,128)),
        transforms.RandomRotation(degrees = (-20, +20)),
        transforms.ToTensor(),
        transforms.Pad(2),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    ])
    
    validation_transform = transforms.Compose([
        transforms.Resize(size = (128, 128)),
        transforms.ToTensor(),
        transforms.Pad(2),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    ])
    
    train_set = ImageFolder(data_dir + 'train', transform=train_transform)
    val_set = ImageFolder(data_dir + 'val', transform=validation_transform)
    test_set = ImageFolder(data_dir + 'test', transform=validation_transform)
    
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=128, shuffle=False)
    
    return train_set, val_set, test_set, train_loader, val_loader, test_loader
    

def show_img(img, label):
    print('Label: ', label)
    plt.imshow(img.permute(1,2,0))

def show_batch(data_loader):
    for images, labels in data_loader:
        fig, ax = plt.subplots(figsize = (12,12))
        ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(make_grid(images[:25], 5).permute(1,2,0))
        break

def create_cnn_model():
    
    cnn_model = nn.Sequential(
    
        nn.Conv2d(3, 16, kernel_size = 3, stride = 1, padding = 1),
        nn.ReLU(),
        nn.MaxPool2d(2,2),
        nn.BatchNorm2d(16), # Output size : bs * 16 * 66 * 66 
        
        nn.Conv2d(16, 64, kernel_size = 3, stride = 1, padding = 1),
        nn.ReLU(),
        nn.MaxPool2d(2,2),
        nn.BatchNorm2d(64), # Output size : bs * 64 * 33 * 33
        
        nn.Conv2d(64, 128, kernel_size = 3, stride = 1, padding = 1),
        nn.ReLU(),
        nn.MaxPool2d(2,2),
        nn.BatchNorm2d(128), # Output size : bs * 128 * 16 * 16
        
        nn.Conv2d(128, 256, kernel_size = 3, stride = 1, padding = 1),
        nn.ReLU(),
        nn.MaxPool2d(2,2),
        nn.BatchNorm2d(256), # Output size : bs * 256 * 8 * 8
        
        nn.Flatten(),
        
        nn.Linear(256*8*8, 256),
        nn.ReLU(),
        
        nn.Linear(256, 8),
        nn.ReLU(),
        
        nn.Linear(8, 2)    
    )
    
    return cnn_model

def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    
def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)
    
def loss_batch(model, loss_func, x, y, opt = None, metric = None):
    pred = model(x)
    
    loss = loss_func(pred, y)
    
    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()
        
    metric_result = None
    
    if metric is not None:
        metric_result = metric(pred, y)
            
    return loss.item(), len(x), metric_result

def evaluate(model, loss_fn, valid_dl, metric = None):
    with torch.no_grad():
        results = [loss_batch(model, loss_fn, x, y, metric = metric) for x, y in valid_dl]
        
        losses, nums, metrics = zip(*results)
        
        total = np.sum(nums)
        
        avg_loss = np.sum(np.multiply(losses, nums)) / total
        
        avg_metric = None
        
        if metric is not None:
            avg_metric = np.sum(np.multiply(metrics, nums)) / total
            
    return avg_loss, total, avg_metric

def fit(epochs, model, loss_fn, train_dl, valid_dl, opt_fn = None, lr = None, metric = None):
    train_losses, val_losses, val_metrics = [], [], []
    
    if opt_fn is None: opt_fn = torch.optim.SGD
    
    opt = opt_fn(model.parameters(), lr = lr)
    
    for epoch in range(epochs):
        model.train()
        
        for x, y in train_dl:
            train_loss, _, _ = loss_batch(model, loss_fn, x, y, opt)
            
        model.eval()
        result = evaluate(model, loss_fn, valid_dl, metric)
        val_loss, total, val_metric = result
        
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_metrics.append(val_metric)
        
        if metric is None:
            print('Epoch{}/{}, train_loss: {:.4f}, val_loss: {:.4f}' 
                 .format(epoch+1, epochs, train_loss, val_loss))
            
        else:
            print('Epoch {}/{}, train_loss: {:.4f}, val_loss: {:.4f}, val_{}: {:.4f}'
                 .format(epoch+1, epochs, train_loss, val_loss, metric.__name__, val_metric))
            
    return train_losses, val_losses, val_metrics

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim = 1)
    return torch.sum(preds == labels).item() / len(preds)

if __name__ == '__main__':
    
    data_dir = './data/'
    
    # Create datasets
    train_set, val_set, test_set, train_loader, val_loader, test_loader = load_pneumonia_dataset(data_dir)
    
    # Explorative study
    show_img(*train_set[1000])
    show_img(*val_set[10])
    
    show_batch(train_loader)
    show_batch(val_loader)
    
    # Create CNN model
    cnn_model = create_cnn_model()
    
    # Move data loaders and model to GPU
    device = get_default_device()
    train_dl = DeviceDataLoader(train_loader, device)
    test_dl = DeviceDataLoader(test_loader, device)
    to_device(cnn_model, device)
    
    # Evaluate model
    val_loss, _, val_acc = evaluate(cnn_model, F.cross_entropy, test_dl, metric = accuracy)
    print(val_loss, val_acc)
    
    num_epochs = 3 # 10
    opt_fn = torch.optim.Adam
    lr = 0.005
    history = fit(num_epochs, cnn_model, F.cross_entropy, train_dl, test_dl, opt_fn, lr, accuracy)
    
    
    
    