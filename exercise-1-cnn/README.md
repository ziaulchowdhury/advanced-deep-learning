# Exercise - 1: Practical Tasks

## Task 2.1 Accuracies 

| Activation function | Optimizer | Learning rate | Epochs | Accuracy |
| LeakyReLU | SGD | 0.0001 | 3 | 41% |
| LeakyReLU | Adam | 0.0001 | 3 | 51% |
| Tanh | SGD | 0.0001 | 3 | 44% |
| Tanh | Adam | 0.0001 | 3 | 53% |
| ReLU | SGD | 0.0001 | 3 | 42% |
| ReLU | Adam | 0.0001 | 3 | 51% |

Best result (i.e. 53% accuracy) was achieved by using Adam optimizer with Tanh activation function, learning rate 0.0001 and 3 epochs.

## Task 2.2 2 Visualization in Tensorboard 

Following figure is taken from Tensorboard which shows the loss over the training images. Green line shows the training loss of the best model identified in section 2.1.

![Trtaining loss: Custom Cifar 10 CNN](https://github.com/ziaulchowdhury/advanced-deep-learning/tree/master/exercise-1-cnn/training-loss.png)