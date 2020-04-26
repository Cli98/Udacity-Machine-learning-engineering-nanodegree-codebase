# torch imports
import torch.nn.functional as F
import torch.nn as nn


# define the CNN architecture
class Net(nn.Module):
    ### TODO: choose an architecture, and complete the class
    def __init__(self):
        super(Net, self).__init__()
        ## Define layers of a CNN
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=0)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=0)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1, ceil_mode=True)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1, ceil_mode=True)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=2)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=2)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1, ceil_mode=True)
        self.conv_bn1 = nn.BatchNorm2d(32)
        self.conv_bn2 = nn.BatchNorm2d(64)
        self.conv_bn3 = nn.BatchNorm2d(128)
        self.conv_bn4 = nn.BatchNorm2d(256)
        self.dense = nn.Linear(256 * 9 * 9, 133)

    def forward(self, x):
        ## Define forward behavior
        out = F.relu(self.conv1(x))  # --> 224,224,16
        out = F.relu(self.conv2(out))  # --> 224,224,32
        out = self.pool1(out)  # --> 112,112,32
        out = self.conv_bn1(out)
        out = F.relu(self.conv3(out))  # --> 112,112,64
        out = self.pool2(out)  # --> 56,56,64
        out = self.conv_bn2(out)
        out = F.relu(self.conv4(out))  # --> 56,56,128
        out = self.pool2(out)  # --> 28,28,128
        out = self.conv_bn3(out)
        out = F.relu(self.conv5(out))  # --> 28,28,256
        out = self.pool3(out)  # --> 14,14,256
        out = self.conv_bn4(out)
        out = F.relu(self.conv6(out))  # --> 14,14,256
        out = self.pool3(out)  # --> 7,7,256
        out = self.conv_bn4(out)
        # print(out.shape)
        out = out.view(-1, 256 * 9 * 9)
        out = self.dense(out)
        return out