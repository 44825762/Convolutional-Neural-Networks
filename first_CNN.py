import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

########################################################################################################################################
### 使用卷积神经网络 — 两个卷基层+线性修正单元(ReLU)+正则化+拓展数据集+继续插入额外的全连接层+Dropout+组合网络 => 可达到最高正确率
### 组合网络类似于随机森林或者adaboost的集成方法，创建几个神经网络，让他们投票来决定最好的分类。
### 1. 使用卷积层极大地减小了全连接层中的参数的数目，使学习的问题更容易
### 2. 使用更多强有力的规范化技术（尤其是弃权和卷积）来减小过度拟合，
### 3. 使用修正线性单元而不是S型神经元，来加速训练-依据经验，通常是3-5倍，
### 4. 使用GPU来计算
### 5. 利用充分大的数据集，避免过拟合
### 6. 使用正确的代价函数，避免学习减速
### 7. 使用好的权重初始化，避免因为神经元饱和引起的学习减速
###
### 使用卷积神经网络 — 两个卷基层+线性修正单元(ReLU)+拓展数据集+继续插入额外的全连接层+弃权技术
########################################################################################################################################

PATH='./CNN.pt';
EPOCH = 40;
num_classes = 10;
BATCH_SIZE = 128;
learning_rate = 0.001;

train_data = torchvision.datasets.MNIST(root='./mnist/', train=True, transform=torchvision.transforms.ToTensor());

train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True);

test_data = torchvision.datasets.MNIST(root='./mnist/', train=False);


def showimages(img):
    img = img / 2 + 0.5;  # denormalized
    npimage = img.numpy();
    plt.figure();
    plt.imshow(np.transpose(npimage, (1, 2, 0)));
    plt.show();


dataiter = iter(train_loader);
images, labels = dataiter.next();

# show images
# showimages(torchvision.utils.make_grid(images))


test_x = torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor)[:2000].cuda() / 255.;  # Tensor on GPU
test_y = test_data.test_labels[:2000].cuda();


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


# 创建一个CNN实例cnn = CNN();
cnn = CNN();
cnn.cuda();

# 该函数包含了 SoftMax activation 和 cross entorpy，所以在神经网络结构定义的时候不需要定义softmax activation
loss_func = nn.CrossEntropyLoss();

# 在nn.Module类中，方法 nn.parameters()可以让pytorch追踪所有CNN中需要训练的模型参数，让他知道要优化的参数是哪些
optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate);

# 训练模型
total_step = len(train_loader);
loss_list = [];
acc_list = [];
for epoch in range(EPOCH):
    for step, (image, label) in enumerate(train_loader):

        # !!!!!!!! Change in here !!!!!!!!! #
        data = image.cuda();  # Tensor on GPU
        label = label.cuda();  # Tensor on GPU

        # 向网络中输入images，得到output,在这一步的时候模型会自动调用model.forward(images)函数
        output = cnn(data);

        # 计算这损失
        loss = loss_func(output, label);

        # 反向传播，Adam优化训练
        # 先清空所有参数的梯度缓存，否则会在上面累加
        optimizer.zero_grad();

        # 计算反向传播
        loss.backward();
        # 更新梯度
        optimizer.step();

        # 记录精度
        if step % 50 == 0:
            test_output = cnn(test_x);

            # !!!!!!!! Change in here !!!!!!!!! #
            # torch.max(x,1) 按行取最大值
            # output每一行的最大值存在_中，每一行最大值的索引存在predicted中
            # output的每一行的每个元素的值表示是这一类的概率，取最大概率所对应的类作为分类结果
            # 也就是找到最大概率的索引
            pred_y = torch.max(test_output, 1)[1].cuda().data;  # move the computation in GPU

            accuracy = torch.sum(pred_y == test_y).type(torch.FloatTensor) / test_y.size(0);
            acc_list.append(accuracy);
            if accuracy >= max(acc_list):
                torch.save(cnn.state_dict(), PATH); # 保存模型的参数

            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.cpu().numpy(), '| test accuracy: %.4f' % accuracy);


print(max(acc_list));

# 99.17-99.24%

