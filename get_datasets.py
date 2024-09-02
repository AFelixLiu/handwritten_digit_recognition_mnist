"""
@author AFelixLiu
@date 2024 9月 02
"""

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

if __name__ == '__main__':
    # 实现图像的预处理pipeline
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # 转换为单通道灰度图
        transforms.ToTensor()                         # 转换为张量
    ])

    # 下载训练集
    train_dataset = datasets.MNIST(root='./mnist_data/train', train=True, transform=transform, download=True)
    # 下载测试集
    test_dataset = datasets.MNIST(root='./mnist_data/test', train=False, transform=transform, download=True)

    print("train_dataset length: ", len(train_dataset))
    print("test_dataset length: ", len(test_dataset))

    # # 实现小批量的数据读取
    # train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    #
    # print("train_loader  length: ", len(train_loader))
    #
    # # 循环遍历train_loader
    # for batch_idx, (data, label) in enumerate(train_loader):
    #     if batch_idx != 0:
    #         break
    #
    #     print("batch_idx: ", batch_idx)
    #     print("data.shape : ", data.shape)   # 数据的尺寸
    #     print("label.shape: ", label.shape)  # 标签的尺寸
    #     print("label: ", label)
