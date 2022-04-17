# Exercise - 2: Practical Tasks

## Task 1.1.1 Transfer Learning from ImageNet (with AlexNet) 

### Accuracies

| Dataset             | Model type       | Learning rate  | Epochs   | Accuracy |
|:----------------:|:-------------------:|:--------------:|:--------:|:--------:|
| Cifar10          | Fine tuning         | 0.001          | 10       | 88.93%   |
| Cifar10          | Feature extraction  | 0.001          | 10       | 84.43%   |


Best result (i.e. 53% accuracy) was achieved by using Adam optimizer with Tanh activation function, learning rate 0.0001 and 3 epochs.

### Difference between two runs

In the above cases, two types of transfer learning (fine tuning of ConvNet and feature extraction) are experimented. It's can be observed that the fine tuning of the AlexNet provided higher accuracies than the feature extraction. The reason behind the better accuracy of fine tuning transfer learning method is that the AlexNet is initialized with the weights of pretrained model trained with ImageNet dataset instead of random initialization but the model is also training with Cifar10 dataset which contains 10 classes. In contrary, weights of the layers of AlexNet is frozen except last fully connected layer in feature learning and thus the final layer is trained with Cifar10 dataset.

## Task 1.1.2 Transfer Learning from MNIST & SVHN (with AlexNet and Custom CNN) 

### Accuracies

| Dataset        | Model type            | Learning rate  | Epochs   | Accuracy |
|:--------------:|:---------------------:|:--------------:|:--------:|:--------:|
| MNIST          | Vanila CNN            | 0.001          | 2       | 97.00%    |
| SVHN           | Vanila CNN            | 0.001          | 5       | 80.00%    |
| MNIST          | Fine tuning (AlexNet) | 0.001          | 5       | 99.57%    |
| SVHN           | Fine tuning (AlexNet) | 0.001          | 5       | 94.30%    |