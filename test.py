"""
@author AFelixLiu
@date 2024 9月 02
"""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import Network

if __name__ == '__main__':
    # 图像预处理
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # 转换为单通道灰度图
        transforms.ToTensor()                         # 转换为张量
    ])

    # 读入并构造测试集
    test_dataset = datasets.MNIST(root='./mnist_data/test', train=False, transform=transform)
    print("test_dataset length: ", len(test_dataset))

    # 逐个读取数据
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=True)

    # 设置模型
    model = Network()                                                   # 1、定义模型
    model.load_state_dict(torch.load("./saved_models/mnist_model.pt"))  # 2、加载模型文件

    res = 0  # 保存正确识别数字的数量
    for i, (x, y) in enumerate(test_loader):
        output = model(x)               # 预测
        pred = output.argmax(1).item()  # 选择概率最大的标签作为预测结果
        if pred == y:
            res += 1
        else:
            print(f"wrong case: predict = {pred}, but y = {y.item()}")

    # 计算测试结果
    sample_num = len(test_dataset)
    acc = res * 1.0 / sample_num
    print(f"test_accuracy = {res} / {sample_num} = {acc}")
