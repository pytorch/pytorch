#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
宇宙同构优化示例
===============

这个示例展示了如何使用基于宇宙同构方法的PyTorch优化组件来训练模型。
宇宙同构方法将计算图视为宇宙，通过优化信息熵比和经典化效率来提高性能。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import time

# 导入宇宙同构优化模块
import torch.cosmic as cosmic

# 定义一个简单的CNN模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

def train(model, device, train_loader, optimizer, epoch, log_interval=100):
    """训练函数"""
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        
        if batch_idx % log_interval == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}'
                  f' ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
            
            # 如果使用的是宇宙模块，打印优化指标
            if hasattr(model, 'get_metrics'):
                metrics = model.get_metrics()
                print(f'Cosmic Metrics: {metrics}')

def test(model, device, test_loader):
    """测试函数"""
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)}'
          f' ({accuracy:.0f}%)\n')
    
    return accuracy

def benchmark_model(model_creator, device, train_loader, test_loader, epochs=3, name="Model"):
    """基准测试模型性能"""
    # 创建模型
    model = model_creator().to(device)
    
    # 参数数量
    param_count = sum(p.numel() for p in model.parameters())
    print(f"{name} - 参数数量: {param_count}")
    
    # 为模型创建优化器
    if hasattr(model, 'create_cosmic_optimizer'):
        optimizer = model.create_cosmic_optimizer(optim.Adam, lr=0.001)
    else:
        optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 训练时间测量
    start_time = time.time()
    
    # 训练和测试模型
    accuracies = []
    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, epoch)
        accuracy = test(model, device, test_loader)
        accuracies.append(accuracy)
    
    # 计算训练时间
    train_time = time.time() - start_time
    
    print(f"{name} - 训练时间: {train_time:.2f}秒, 最终准确率: {accuracies[-1]:.2f}%")
    
    return model, accuracies, train_time

def main():
    # 配置 
    batch_size = 64
    test_batch_size = 1000
    epochs = 3
    use_cuda = torch.cuda.is_available()
    
    # 设备选择
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"使用设备: {device}")
    
    # 数据加载参数
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    
    # 加载MNIST数据集
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST('../data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('../data', train=False, transform=transform)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, **kwargs)
    
    # 测试基本模型
    print("\n=== 基准模型 ===")
    standard_model_creator = lambda: Net()
    standard_model, standard_accuracies, standard_time = benchmark_model(
        standard_model_creator, device, train_loader, test_loader, epochs, "标准模型"
    )
    
    # 测试宇宙同构优化模型
    print("\n=== 宇宙同构优化模型 ===")
    cosmic_model_creator = lambda: cosmic.convert_to_cosmic(
        Net(),
        dynamic_graph=True,
        compression=True,
        efficiency=True,
        dynamic_graph_threshold=0.3,  # 稍微降低阈值让更多节点参与
        compression_alpha=0.3,        # 降低压缩强度
        efficiency_gamma=0.2          # 增加经典化效率学习率
    )
    cosmic_model, cosmic_accuracies, cosmic_time = benchmark_model(
        cosmic_model_creator, device, train_loader, test_loader, epochs, "宇宙同构模型"
    )
    
    # 输出比较结果
    print("\n=== 性能比较 ===")
    speedup = standard_time / cosmic_time
    accuracy_change = cosmic_accuracies[-1] - standard_accuracies[-1]
    
    print(f"训练速度提升: {speedup:.2f}x")
    print(f"准确率变化: {accuracy_change:+.2f}%")
    
    # 如果有宇宙模型，输出最终优化指标
    if hasattr(cosmic_model, 'get_metrics'):
        print("\n=== 宇宙同构优化指标 ===")
        final_metrics = cosmic_model.get_metrics()
        for metric_name, value in final_metrics.items():
            print(f"{metric_name}: {value:.4f}")
    
    # 如果有压缩，计算参数压缩率
    if hasattr(cosmic_model, '_count_nonzero_params'):
        standard_params = sum(p.numel() for p in standard_model.parameters())
        cosmic_nonzero_params = cosmic_model._count_nonzero_params()
        compression_rate = 1.0 - (cosmic_nonzero_params / standard_params)
        print(f"参数压缩率: {compression_rate:.2%}")

if __name__ == '__main__':
    main() 