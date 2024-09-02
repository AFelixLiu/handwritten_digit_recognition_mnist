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

    # 读入并构造训练集
    train_dataset = datasets.MNIST(root='./mnist_data/train', train=True, transform=transform)
    print("train_dataset length: ", len(train_dataset))

    # 实现小批量数据的读取
    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    print("train_loader length: ", len(train_loader))

    # 设置模型
    model = Network()                                 # 1、模型
    optimizer = torch.optim.Adam(model.parameters())  # 2、优化器
    criterion = torch.nn.CrossEntropyLoss()           # 3、损失函数

    # 训练模型
    # 整个训练集要循环遍历10次
    for epoch in range(10):
        # 小批量的数据读取
        for batch_idx, (data, label) in enumerate(train_loader):
            optimizer.zero_grad()            # 1、梯度清零
            output = model(data)             # 2、计算模型的前向传播结果
            loss = criterion(output, label)  # 3、计算误差
            loss.backward()                  # 4、计算梯度
            optimizer.step()                 # 5、更新参数

            # 每迭代100个小批量，打印模型的误差
            if batch_idx % 100 == 0:
                with open("./train_result.txt", "a+") as f:
                    print(f"train_epoch {epoch + 1}/10"
                          f" | batch {batch_idx}/{len(train_loader)}"
                          f" | loss {loss.item():.4f}", file=f)

    # 保存模型
    torch.save(model.state_dict(), './saved_models/mnist_model.pt')
