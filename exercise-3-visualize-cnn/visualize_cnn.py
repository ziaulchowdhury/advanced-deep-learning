# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 23:07:11 2022

@author: ziaul
"""

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


class CustomCNN(nn.Module):
    
    def __init__(self):
        
        super(CustomCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 16, 5, 1,2 )
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 5, 1, 2)
        self.fc1 = nn.Linear(32 * 7 * 7, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        
        x = self.extract_features(x)
        x = F.leaky_relu(self.fc1(x))
        
        return x

    def extract_features(self, x):
        
        x = self.pool(F.leaky_relu(self.conv1(x)))
        x = self.pool(F.leaky_relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        
        return x
  
class MnistCnnModel:
    
    def __init__(self, learning_rate=0.001, num_epoch=2, n_components=2):
        self.learning_rate = learning_rate
        self.num_epoch = num_epoch
        self.n_components = n_components
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.load_mnist_data()
        self.create_dataloaders()
        
        self.create_cnn_model()
        self.train_cnn_net()
        self.test_model()
        
        self.extract_plot_features()
        
    def load_mnist_data(self):
        trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
        self.testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
        self.trainset, self.valset = torch.utils.data.random_split(trainset, [55000, 5000])
    
    def create_dataloaders(self, num_workers=2):
        self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=5000, shuffle=True, num_workers=num_workers)
        self.valloader = torch.utils.data.DataLoader(self.valset, batch_size=5000, shuffle=True, num_workers=num_workers)
        self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=5000, shuffle=False, num_workers=num_workers)
        
        dataset = torch.utils.data.ConcatDataset([self.trainset, self.testset])
        self.dataloaders = torch.utils.data.DataLoader(dataset, batch_size=4, num_workers=num_workers, persistent_workers=True, shuffle=True, pin_memory=True)
    
    def create_cnn_model(self):
        self.net = CustomCNN()
        self.net = self.net.to(self.device)
        print(self.net)
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.learning_rate)

    def train_cnn_net(self):
        for epoch in range(self.num_epoch):
            self.net.train()
            running_loss = 0.0
         
            for i, data in enumerate(self.trainloader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                outputs = self.net(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                running_loss += loss.item()
                if i % 2000 == 1999:
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 2000))
                    
                    running_loss = 0.0
            print('Finished Training')

    def test_model(self):
        
        correct = 0
        total = 0
        with torch.no_grad():
            self.net.eval()
            for data in self.testloader:
                images, labels = data
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print('Accuracy of the network on the 10000 test images: %d %%' % (
                (100.0 * correct) / total))
 
    def extract_plot_features(self, create_subplot=False):
       
        image, labels = next(iter(self.testloader))
        image = image.to(self.device)
        cnn_features = self.net.extract_features(image).cpu().detach().numpy()
        print(cnn_features.shape)
        
        pca = PCA(n_components=self.n_components)
        pca_values = pca.fit_transform(cnn_features, labels)

        tsne = TSNE(n_components=self.n_components, perplexity=10)
        tsne_values = tsne.fit_transform(cnn_features, labels)

        fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(15,10))
        
        pca_x,pca_y = np.column_stack(pca_values)
        ax1.scatter(pca_x, pca_y, c=labels, cmap='tab10')
        ax1.set_title('PCA 1')

        tsne_x,tsne_y = np.column_stack(tsne_values)
        ax2.scatter(tsne_x,tsne_y, c=labels, cmap='tab10')
        ax2.set_title('TSNE 1')
        
    
if __name__=='__main__':
    
    mnist_cnn_leakyrelu__lr1e6 = MnistCnnModel(learning_rate=1e-6, num_epoch=1)
    
    mnist_cnn_leakyrelu__lr1e6 = MnistCnnModel(learning_rate=1e-3, num_epoch=20)
    