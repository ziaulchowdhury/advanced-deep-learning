# -*- coding: utf-8 -*-

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from enum import Enum

def transform_mnist(img):
    img = img.resize((32, 32)).convert('RGB')
    tensor = torchvision.transforms.ToTensor()(img)
    return tensor

class ActivationFunction(Enum):
    ReLU = 1
    LeakyReLU = 2
    Tanh = 3

class DataSet(Enum):
    MNIST = 1
    SVHN = 2
    CIFAR10 = 3

class CustomCNN(nn.Module):
    
    def __init__(self, activation_function=ActivationFunction):
        
        super(CustomCNN, self).__init__()
        
        self.activation_function = activation_function
        
        self.conv1 = nn.Conv2d(3, 6, 5)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        
        if self.activation_function == ActivationFunction.LeakyReLU:
            x = self.pool(F.leaky_relu(self.conv1(x)))
            x = self.pool(F.leaky_relu(self.conv2(x)))
        elif self.activation_function == ActivationFunction.ReLU:
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
        else:
            x = self.pool(F.tanh(self.conv1(x)))
            x = self.pool(F.tanh(self.conv2(x)))
        
        x = x.view(-1, 16 * 5 * 5)
        
        if self.activation_function == ActivationFunction.LeakyReLU:
            x = F.leaky_relu(self.fc1(x))
            x = F.leaky_relu(self.fc2(x))
        elif self.activation_function == ActivationFunction.ReLU:
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
        else:
            x = F.tanh(self.fc1(x))
            x = F.tanh(self.fc2(x))

        x = self.fc3(x)
        return x


class CifarCNN:
    
    def __init__(self, dataset, activation_function=ActivationFunction, learning_rate=0.001, num_epoch=2):
        
        self.dataset = dataset
        self.activation_function = activation_function
        self.learning_rate = learning_rate
        self.num_epoch = num_epoch
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if dataset == DataSet.CIFAR10:
            self.load_cifar10_data()
        elif dataset == DataSet.MNIST:
            self.load_mnist_data()
        elif dataset == DataSet.SVHN:
            self.load_svhn_data()
        
        self.create_dataloaders()
        self.create_cnn_model()
        self.train_cnn_net()
        self.test_model()
        
    
    def load_cifar10_data(self, batch_size=4, data_dir='./data'):

        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
        self.testset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)
        
        
    def load_mnist_data(self):
        self.trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_mnist)
        self.testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_mnist)
    
    
    def load_svhn_data(self):
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.trainset = torchvision.datasets.SVHN(root='./data', split='train', transform=transform, target_transform=None, download=True)
        self.testset = torchvision.datasets.SVHN(root='./data', split='test', transform=transform, target_transform=None, download=True)
    
        
    def create_dataloaders(self, num_workers=2):
        self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=4, shuffle=True, num_workers=num_workers)
        self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=4, shuffle=False, num_workers=num_workers)
    
    
    def create_cnn_model(self):
        
        self.net = CustomCNN(activation_function=self.activation_function)
        self.net = self.net.to(self.device)
        print(self.net)
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.learning_rate)


    def train_cnn_net(self):
        
        for epoch in range(self.num_epoch):
        
            running_loss = 0.0
         
            for i, data in enumerate(self.trainloader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
        
                outputs = self.net(inputs)
                loss = self.criterion(outputs, labels)
        
                loss.backward()
                self.optimizer.step()
        
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
            for data in self.testloader:
                images, labels = data
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print('Accuracy of the network on the 10000 test images: %d %%' % (
                (100.0 * correct) / total))

    
if __name__=='__main__':
    
    mnist_cnn_leakyrelu__lr001 = CifarCNN(DataSet.MNIST, activation_function=ActivationFunction.LeakyReLU, learning_rate=0.001, num_epoch=5)
    ### 5 epoch ->  accuracy
    
    svhn_cnn_leakyrelu__lr001 = CifarCNN(DataSet.SVHN, activation_function=ActivationFunction.LeakyReLU, learning_rate=0.001, num_epoch=5)
    ### 5 epoch -> 87.00% accuracy
    