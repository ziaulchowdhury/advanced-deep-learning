# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 01:27:17 2022

@author: ziaul
"""
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
from enum import Enum

class ActivationFunction(Enum):
    ReLU = 1
    LeakyReLU = 2
    Tanh = 3

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
    
    def __init__(self, custom_cnn=True, activation_function=ActivationFunction, learning_rate=0.001, use_sgd=False, num_epoch=2, write_summary=False):
        
        self.custom_cnn = custom_cnn
        self.activation_function = activation_function
        self.learning_rate = learning_rate
        self.use_sgd = use_sgd
        self.num_epoch = num_epoch
        self.write_summary = write_summary
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if self.write_summary:
            summary_name = f"runs/cifar_cnn_lr{self.learning_rate}_{'sgd' if self.use_sgd else 'adam'}_epochs{self.num_epoch}_{str(self.activation_function).replace('.', '_')}"
            self.writer = SummaryWriter(summary_name)
        
        self.load_data()
        self.create_cnn_model()
        self.train_cnn_net()
        self.test_random_images()
        self.test_model()
        # self.write_to_tensorboard()
        
        if self.write_summary:
            self.writer.close()
        
    
    def load_data(self, batch_size=4, num_workers=2, data_dir='./data'):

        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        self.trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
        self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        self.testset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)
        self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        
        self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    def create_cnn_model(self):
        
        if self.custom_cnn:
            self.net = CustomCNN(activation_function=self.activation_function)
            self.net = self.net.to(self.device)
            print(self.net)
            
            self.criterion = nn.CrossEntropyLoss()
            if self.use_sgd:
                self.optimizer = optim.SGD(self.net.parameters(), lr=self.learning_rate, momentum=0.9)
            else:
                self.optimizer = optim.Adam(self.net.parameters(), lr=self.learning_rate)

    def train_cnn_net(self):
        
        if self.custom_cnn:
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
                        
                        if self.write_summary:
                            self.writer.add_scalar('training loss',
                                running_loss / 1000,
                                epoch * len(self.trainloader) + i)
    
                            # ...log a Matplotlib Figure showing the model's predictions on a random mini-batch
                            self.writer.add_figure('predictions vs. actuals',
                                            self.plot_classes_preds(inputs, labels),
                                            global_step=epoch * len(self.trainloader) + i)
                        
                        print('[%d, %5d] loss: %.3f' %
                              (epoch + 1, i + 1, running_loss / 2000))
                        
                        running_loss = 0.0
            print('Finished Training')
        
    def test_random_images(self):
        
        dataiter = iter(self.testloader)
        images, labels = dataiter.next()
        images, labels = images.to(self.device), labels.to(self.device)
        
        print('Ground truth: ', ' '.join('%5s' % self.classes[labels[j]] for j in
                                         range(4)))
        
        outputs = self.net(images)
        _, predicted = torch.max(outputs, 1)
        print('Predicted: ', ' '.join('%5s' % self.classes[predicted[j]]
                                      for j in range(4)))

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
                100 * correct / total))
        
        class_correct = list(0. for i in range(10))
        class_total = list(0. for i in range(10))
        with torch.no_grad():
            for data in self.testloader:
                images, labels = data
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.net(images)
                _, predicted = torch.max(outputs, 1)
                c = (predicted == labels).squeeze()
                for i in range(4):
                    label = labels[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1
        
        for i in range(10):
         print('Accuracy of %5s : %2d %%' % (
                self.classes[i], 100 * class_correct[i] / class_total[i]))
    
    def matplotlib_imshow(self, img, one_channel=False):
        if one_channel:
            img = img.mean(dim=0)
        img = img / 2 + 0.5     # unnormalize
        npimg = img.cpu().detach().numpy()
        if one_channel:
            plt.imshow(npimg, cmap="Greys")
        else:
            plt.imshow(np.transpose(npimg, (1, 2, 0)))
            
    def images_to_probs(self, images):
        '''
        Generates predictions and corresponding probabilities from a trained
        network and a list of images
        '''
        output = self.net(images)
        # convert output probabilities to predicted class
        _, preds_tensor = torch.max(output, 1)
        preds = np.squeeze(preds_tensor.cpu().detach().numpy())
        return preds, [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]


    def plot_classes_preds(self, images, labels):
        '''
        Generates matplotlib Figure using a trained network, along with images
        and labels from a batch, that shows the network's top prediction along
        with its probability, alongside the actual label, coloring this
        information based on whether the prediction was correct or not.
        Uses the "images_to_probs" function.
        '''
        preds, probs = self.images_to_probs(images)
        # plot the images in the batch, along with predicted and true labels
        fig = plt.figure(figsize=(12, 48))
        for idx in np.arange(4):
            ax = fig.add_subplot(1, 4, idx+1, xticks=[], yticks=[])
            self.matplotlib_imshow(images[idx], one_channel=True)
            ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
                self.classes[preds[idx]],
                probs[idx] * 100.0,
                self.classes[labels[idx]]),
                        color=("green" if preds[idx]==labels[idx].item() else "red"))
        return fig
    
    def write_to_tensorboard(self):
        #
        # >> tensorboard --logdir=D:\Studies\Lulea-University-of-Technology\Individual-Courses\Advanced-Deep-Learning\Assignments\advanced-deep-learning-github\exercise-1-cnn
        #
        
        if self.write_summary:
        
            dataiter = iter(self.trainloader)
            images, labels = dataiter.next()
            
            # create grid of images
            img_grid = torchvision.utils.make_grid(images)
            
            # show images
            self.matplotlib_imshow(img_grid, one_channel=True)
            
            # write to tensorboard
            self.writer.add_image('four_cifar_images', img_grid)
            
            self.writer.add_graph(self.net, images)
    

if __name__=='__main__':
    
    # SGD for all, LR=0.0001, epochs=3, Activation(ReLU, LeakyReLU, Tanh)
    cifar_cnn_relu_sgd_lr0001 = CifarCNN(activation_function=ActivationFunction.ReLU, learning_rate=0.0001, use_sgd=True, num_epoch=3, write_summary=True)
    cifar_cnn_leakyrelu_sgd_lr0001 = CifarCNN(activation_function=ActivationFunction.LeakyReLU, learning_rate=0.0001, use_sgd=True, num_epoch=3, write_summary=True)
    cifar_cnn_tanh_sgd_lr0001 = CifarCNN(activation_function=ActivationFunction.Tanh, learning_rate=0.0001, use_sgd=True, num_epoch=3, write_summary=True)
    
    # Adam for all, LR=0.0001, epochs=3, Activation(ReLU, LeakyReLU, Tanh)
    cifar_cnn_relu_adam_lr0001 = CifarCNN(activation_function=ActivationFunction.ReLU, learning_rate=0.0001, use_sgd=False, num_epoch=3, write_summary=True)
    cifar_cnn_leakyrelu_adam_lr0001 = CifarCNN(activation_function=ActivationFunction.LeakyReLU, learning_rate=0.0001, use_sgd=False, num_epoch=3, write_summary=True)
    cifar_cnn_leakyrelu_tanh_lr0001 = CifarCNN(activation_function=ActivationFunction.Tanh, learning_rate=0.0001, use_sgd=False, num_epoch=3, write_summary=True)
    
    cifar_cnn_leakyrelu_tanh_lr00001 = CifarCNN(activation_function=ActivationFunction.Tanh, learning_rate=0.00001, use_sgd=False, num_epoch=5, write_summary=True)
    