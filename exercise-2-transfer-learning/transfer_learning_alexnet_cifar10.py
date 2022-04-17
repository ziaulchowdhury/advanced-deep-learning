# -*- coding: utf-8 -*-

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import time


def load_data():
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    train_data = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=4, shuffle=True, num_workers=2)
    test_data = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=4, shuffle=False, num_workers=2)
        
    classes = ('Airplane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck')
    
    return train_data, train_loader, test_data, test_loader, classes


def create_alexnet_model(device, pretained=True):
    alexnet = torch.hub.load('pytorch/vision:v0.6.0', 'alexnet', pretrained=pretained)
    alexnet.eval()
    alexnet.classifier[4] = nn.Linear(4096, 1024)
    alexnet.classifier[6] = nn.Linear(1024, 10)
    
    alexnet.to(device)
    
    return alexnet
    
def train_model(alexnet, optimizer, trainloader, device, n_epoch=10):
    ''' Training model '''
    
    start_ts = time.time()
    
    for epoch in range(n_epoch):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)
    
            # zero the parameter gradients
            optimizer.zero_grad()
    
            # forward + backward + optimize
            output = alexnet(inputs)
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

def test_model(alexnet, test_loader):
    ''' Testing Accuracy '''
    
    start_ts = time.time()
    
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = alexnet(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
    print(f"Testing time: {time.time()-start_ts}s")
    

if __name__=='__main__':
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    train_data, train_loader, test_data, test_loader, classes = load_data()
    
    alexnet = create_alexnet_model(device, pretained=True)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(alexnet.parameters(), lr=0.001, momentum=0.9)
    
    n_epoch = 1
    train_model(alexnet, optimizer, train_loader, device, n_epoch=n_epoch)
    
    test_model(alexnet, test_loader)
    
    