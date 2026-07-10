import torch.nn as nn

class MyModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1=nn.Conv2d(3,32,kernel_size=3,padding=1)
        self.conv2=nn.Conv2d(32,64,kernel_size=3,padding=1)
        self.conv3=nn.Conv2d(64,128,kernel_size=3,padding=1)
        self.conv4=nn.Conv2d(128,256,kernel_size=3,padding=1)
        self.conv5=nn.Conv2d(256,512,kernel_size=3,padding=1)

        self.bn1=nn.BatchNorm2d(32)
        self.bn2=nn.BatchNorm2d(64)
        self.bn3=nn.BatchNorm2d(128)
        self.bn4=nn.BatchNorm2d(256)
        self.bn5=nn.BatchNorm2d(512)

        self.relu=nn.ReLU()
        self.flatten=nn.Flatten()
        self.pooling=nn.MaxPool2d(2,2)
        self.dropout=nn.Dropout(0.5)

        self.linear=nn.Linear((512*4*4),128)
        self.output=nn.Linear(128,10)