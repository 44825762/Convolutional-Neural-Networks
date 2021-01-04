import os
import numpy as np
import math
import argparse
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torchvision
import torch.utils.data as Data

import torch.nn as nn
import torch.nn.functional as F
import torch

import cv2
from PIL import Image


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__();
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.1),
        );
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 64, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.1),
        );

        # 接全连接层的意义是：将神经网络输出的丰富信息加到标准分类器中
        # 此处交叉熵的计算包含了softmax 的计算，所以不需要额外添加 softmax 层
        self.out = nn.Sequential(
            nn.Linear(64 * 7 * 7, 10),
            nn.Linear(10, 10),

        );




    # 定义网络的前向传播,该函数会覆盖 nn.Module 里的forward函数
    # 输入x,经过网络的层层结构，输出为out
    def forward(self, x):
        x = self.conv1(x);
        x = self.conv2(x);

        # Flatten the data (n, 64, 7, 7) --> (n, 7*7*64 = 3136)  =>  (128,3136)
        # 左行右列，-1在哪边哪边固定只有一列
        x = x.view(x.size(0), -1);

        # 以一定概率丢掉一些神经单元，防止过拟合
        # x = self.drop_out(x);

        output = self.out(x);

        return output;


# 加载模型参数
cnn = CNN();
cnn.load_state_dict(torch.load('CNN.pt'));
cnn.cuda();
device = torch.device("cuda:0");
# test function
def test(testLoader, model, device):
  model.to(device);
  with torch.no_grad(): # when in test stage, no grad
    correct = 0;
    total = 0;
    for (imgs, labels) in testLoader:
      imgs = imgs.to(device);
      labels = labels.to(device);
      out = model(imgs);
      _, pre = torch.max(out.data, 1);
      total += labels.size(0);
      correct += (pre == labels).sum().item();
    print('Accuracy: {}'.format(correct / total));


BATCH_SIZE = 128;

test_data = torchvision.datasets.MNIST(root='./mnist/', train=False, transform=torchvision.transforms.ToTensor());

test_loader = Data.DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=True);

test(test_loader, cnn, device)



# 99.17-99.24%