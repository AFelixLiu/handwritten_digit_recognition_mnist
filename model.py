"""
@author AFelixLiu
@date 2024 9月 02
"""

import torch
from torch import nn


# 定义神经网络
class Network(nn.Module):
    def __init__(self):
        super().__init__()
        # 线性层1：输入层和隐藏层之间的线性层
        self.layer_1 = nn.Linear(784, 256)
        # 线性层2：隐藏层和输出层之间的线性层
        self.layer_2 = nn.Linear(256, 10)

    # 前向传播函数， 输入为图像x
    def forward(self, x):
        x = x.view(-1, 28 * 28)  # 使用view函数将x展平
        x = self.layer_1(x)      # 将x输入至线性层1
        x = torch.relu(x)        # 使用relu激活函数
        return self.layer_2(x)   # 将x输入至线性层2计算结果
