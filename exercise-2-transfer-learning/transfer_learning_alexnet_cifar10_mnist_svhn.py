# -*- coding: utf-8 -*-
"""
Created on Sat Apr 16 23:12:55 2022

@author: ziaul
"""

# -*- coding: utf-8 -*-

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from enum import Enum
import time
import copy
import numpy as np
import matplotlib.pyplot as plt
from torch.optim import lr_scheduler

class DataSet(Enum):
    MNIST = 1
    SVHN = 2
    CIFAR10 = 3

def transform_mnist(img):
    img = img.resize((224, 224)).convert('RGB')
    tensor = torchvision.transforms.ToTensor()(img)
    return tensor

class AlexNetModel:
    
    def __init__(self, dataset, n_epoch=1, pretained=True, learning_rate=0.001, feature_extraction=False):
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.dataset = dataset
        
        if(dataset == DataSet.MNIST):
            self.train_data, self.test_data = self.load_mnist_data()
        elif (dataset == DataSet.SVHN):
            self.train_data, self.test_data = self.load_svhn_data()
        elif (dataset == DataSet.CIFAR10):
            self.train_data, self.test_data = self.load_cifar10_data()
        
        self.image_datasets = dict(train=self.train_data, val=self.test_data)
        self.train_loader, self.test_loader = self.create_dataloaders()
        self.dataloaders = dict(train=self.train_loader, val=self.test_loader)
        self.dataset_sizes = {x: len(self.image_datasets[x]) for x in ['train', 'val']}
        
        if(dataset == DataSet.MNIST): 
            self.class_names = self.image_datasets['train'].classes
        
        self.alexnet = self.create_alexnet_model(pretained, feature_extraction)
        
        criterion = nn.CrossEntropyLoss()
        
        optimizer = optim.SGD(self.alexnet.parameters(), lr=learning_rate, momentum=0.9)
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        self.train_model(self.alexnet, criterion, optimizer, exp_lr_scheduler, n_epoch)
        
        # self.test_model()
    
    def load_mnist_data(self):
        train_data = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_mnist)
        test_data = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_mnist)
        return train_data, test_data
    
    def load_svhn_data(self):
        train_data = torchvision.datasets.SVHN(root='./data', split='train', transform=transform_mnist, target_transform=None, download=True)
        test_data = torchvision.datasets.SVHN(root='./data', split='test', transform=transform_mnist, target_transform=None, download=True)
        return train_data, test_data
    
    
    def load_cifar10_data(self):
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        train_data = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_data = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        return train_data, test_data
    
    def create_dataloaders(self):
        train_loader = torch.utils.data.DataLoader(self.train_data, batch_size=4, shuffle=True, num_workers=2)
        test_loader = torch.utils.data.DataLoader(self.test_data, batch_size=4, shuffle=False, num_workers=2)
        return train_loader, test_loader
    
    def create_alexnet_model(self, pretained, feature_extraction):
        alexnet = torch.hub.load('pytorch/vision:v0.6.0', 'alexnet', pretrained=pretained)
        
        if feature_extraction:
            for param in alexnet.parameters():
                param.requires_grad = False
                
        # alexnet.eval()
        alexnet.classifier[4] = nn.Linear(4096, 1024)
        alexnet.classifier[6] = nn.Linear(1024, 10)
        
        alexnet.to(self.device)
        
        return alexnet
    
    def train_model(self, model, criterion, optimizer, scheduler, num_epochs):
        since = time.time()
    
        best_model_wts = copy.deepcopy(self.alexnet.state_dict())
        best_acc = 0.0
    
        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)
    
            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode
    
                running_loss = 0.0
                running_corrects = 0
    
                # Iterate over data.
                for inputs, labels in self.dataloaders[phase]:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
    
                    # zero the parameter gradients
                    optimizer.zero_grad()
    
                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)
    
                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
    
                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                if phase == 'train':
                    scheduler.step()
    
                epoch_loss = running_loss / self.dataset_sizes[phase]
                epoch_acc = running_corrects.double() / self.dataset_sizes[phase]
    
                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
    
                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
    
            print()
    
        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.2f}m {time_elapsed % 60:.2f}s')
        print(f'Best val Acc: {best_acc:4f}')
    
        # load best model weights
        model.load_state_dict(best_model_wts)
        return model
   
    def imshow(self, inp, title=None):
        """Imshow for Tensor."""
        inp = inp.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
        plt.imshow(inp)
        if title is not None:
            plt.title(title)
        plt.pause(0.001)  # pause a bit so that plots are updated 
   
    def visualize_model(self, model, num_images=6):
        was_training = model.training
        model.eval()
        images_so_far = 0
        fig = plt.figure()
        
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(self.dataloaders['val']):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
        
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
        
                for j in range(inputs.size()[0]):
                    images_so_far += 1
                    ax = plt.subplot(num_images//2, 2, images_so_far)
                    ax.axis('off')
                    
                    if(self.dataset == DataSet.MNIST):
                        ax.set_title(f'predicted: {self.class_names[preds[j]]}')
                    self.imshow(inputs.cpu().data[j])
        
                    if images_so_far == num_images:
                        model.train(mode=was_training)
                        return
            model.train(mode=was_training) 
    
    def test_model(self):
        ''' Testing Accuracy '''
        
        start_ts = time.time()
        
        correct = 0
        total = 0
        with torch.no_grad():
            for data in self.test_loader:
                images, labels = data[0].to(self.device), data[1].to(self.device)
                outputs = self.alexnet(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
        print(f"Testing time: {time.time()-start_ts}s")
    

if __name__=='__main__':
    
    print('Cifar10 dataset (AlexNet, tuning) .....')
    cifar10_001_10e_tun = AlexNetModel(DataSet.CIFAR10, n_epoch=10, pretained=True, learning_rate=0.001, feature_extraction=False)
    print('Cifar10 dataset (AlexNet, feature extraction) .....')
    cifar10_001_10e_feaex = AlexNetModel(DataSet.CIFAR10, n_epoch=10, pretained=True, learning_rate=0.001, feature_extraction=True)
        
    print('MNIST dataset (AlexNet, tuning) .....')
    mnist_001_5e_tun = AlexNetModel(DataSet.MNIST, n_epoch=5, pretained=True, learning_rate=0.001, feature_extraction=False)
    print('MNIST dataset (AlexNet, feature extraction) .....')
    mnist_001_5e_feaex = AlexNetModel(DataSet.MNIST, n_epoch=5, pretained=True, learning_rate=0.001, feature_extraction=True)
    
    print('SVHN dataset (AlexNet, tuning) .....')
    svhn_001_5e_tun = AlexNetModel(DataSet.SVHN, n_epoch=5, pretained=True, learning_rate=0.001, feature_extraction=False)
    print('SVHN dataset (AlexNet, feature extraction) .....')
    svhn_001_5e_feaex = AlexNetModel(DataSet.SVHN, n_epoch=5, pretained=True, learning_rate=0.001, feature_extraction=True)
    