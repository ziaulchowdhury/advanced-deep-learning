# -*- coding: utf-8 -*-

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from enum import Enum
import time

class DataSet(Enum):
    MNIST = 1
    SVHN = 2

def transform_mnist(img):
    img = img.resize((224, 224)).convert('RGB')
    tensor = torchvision.transforms.ToTensor()(img)
    return tensor

class AlexNetModel:
    
    def __init__(self, dataset, n_epoch=1, pretained=True, learning_rate=0.001):
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        if(dataset == DataSet.MNIST):
            self.train_data, self.test_data = self.load_mnist_data()
        elif (dataset == DataSet.SVHN):
            self.train_data, self.test_data = self.load_svhn_data()
            
        self.train_loader, self.test_loader = self.create_dataloaders()
        
        self.alexnet = self.create_alexnet_model(pretained=pretained)
        
        #if not pretained:
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.alexnet.parameters(), lr=learning_rate, momentum=0.9)
        self.train_model(optimizer, criterion, n_epoch=n_epoch)
        
        self.test_model()
    
    def load_mnist_data(self):
        train_data = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_mnist)
        test_data = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_mnist)
        return train_data, test_data
    
    def load_svhn_data(self):
        train_data = torchvision.datasets.SVHN(root='./data', split='train', transform=transform_mnist, target_transform=None, download=True)
        test_data = torchvision.datasets.SVHN(root='./data', split='test', transform=transform_mnist, target_transform=None, download=True)
        return train_data, test_data
    
    def create_dataloaders(self):
        train_loader = torch.utils.data.DataLoader(self.train_data, batch_size=4, shuffle=True, num_workers=2)
        test_loader = torch.utils.data.DataLoader(self.test_data, batch_size=4, shuffle=False, num_workers=2)
        return train_loader, test_loader
    
    def create_alexnet_model(self, pretained):
        alexnet = torch.hub.load('pytorch/vision:v0.6.0', 'alexnet', pretrained=pretained)
        alexnet.eval()
        alexnet.classifier[4] = nn.Linear(4096, 1024)
        alexnet.classifier[6] = nn.Linear(1024, 10)
        
        alexnet.to(self.device)
        
        return alexnet
        
    def train_model(self, optimizer, criterion, n_epoch):
        ''' Training model '''
        
        start_ts = time.time()
        
        for epoch in range(n_epoch):
            running_loss = 0.0
            for i, data in enumerate(self.train_loader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
        
                # zero the parameter gradients
                optimizer.zero_grad()
        
                # forward + backward + optimize
                output = self.alexnet(inputs)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()
        
                # print statistics
                running_loss += loss.item()
                if i % 2000 == 1999:    # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0
        print(f"Training time: {time.time()-start_ts}s")
    
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
    
    # mnist_001_1e = AlexNetModel(DataSet.MNIST, n_epoch=1, pretained=True, learning_rate=0.001)
    
    svhn_001_1e = AlexNetModel(DataSet.SVHN, n_epoch=1, pretained=True, learning_rate=0.001)
    
    