
## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## It is suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs

        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
         
        # image shape size = 224
        # output shape size formula: (W-F)/S + 1
        self.conv1 = nn.Conv2d(1, 32, 10, 2)            # in: (1, 224, 224), out: (32, 108, 108)
        self.pool = nn.MaxPool2d(2, 2)                 # out: (32, 54, 54)

        self.batchnorm1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, 5, 2)            # out: (64, 25, 25)
        #pool                                           # out: (64, 12, 12)

        self.batchnorm2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, 5)              # out: (128, 8, 8)
        #pool                                           # out: (128, 4, 4)

        self.batchnorm3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 256, 3)             # out: (256, 2, 2)
        #pool                                           # out: (256, 1, 1)

        self.batchnorm4 = nn.BatchNorm2d(256)

        # self.fc_drop = nn.Dropout(p=0.2)
        self.fc1 = nn.Linear(256*1*1, 512)
        self.fc2 = nn.Linear(512, 136)

        I.xavier_uniform(self.fc1.weight.data)
        I.xavier_uniform(self.fc2.weight.data)
    
    def forward(self, x):

        x = self.pool(F.relu(self.conv1(x)))

        x = self.batchnorm1(x)
        x = self.pool(F.relu(self.conv2(x)))

        x = self.batchnorm2(x)
        x = self.pool(F.relu(self.conv3(x)))

        x = self.batchnorm3(x)
        x = self.pool(F.relu(self.conv4(x)))

        x = self.batchnorm4(x)

        x = x.view(x.size(0), -1) # flatten
        x = F.relu(self.fc1(x))

        x = self.fc2(x)


        # a modified x, having gone through all the layers of your model, should be returned
        return x

# net = Net()
# print(net)
