#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
宇宙同构优化演示
===============

这个演示展示了宇宙同构优化如何应用于简单的PyTorch模型。
"""

import sys
import os

# 添加当前目录到Python路径
sys.path.insert(0, os.path.abspath('.'))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

# 导入宇宙同构优化组件
from torch.cosmic.cosmic_dynamic_graph import cosmic_dynamic_graph
from torch.cosmic.cosmic_classical_efficiency import cosmic_classical_efficiency_step
from torch.cosmic.cosmic_state_compression import cosmic_state_compression
from torch.cosmic.cosmic_module import CosmicModule, convert_to_cosmic

# 定义一个简单的神经网络
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 5)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 创建模型和数据
def demo():
    print("=== 宇宙同构优化演示 ===")
    
    # 创建一个简单模型
    model = SimpleNN()
    print("原始模型结构:")
    print(model)
    
    # 创建一些随机输入数据和目标
    inputs = torch.randn(32, 10)
    targets = torch.randn(32, 5)
    
    # 第1步：宇宙同构动态计算图优化（CDG）
    print("\n1. 宇宙同构动态计算图优化（CDG）示例:")
    outputs, efficiency = cosmic_dynamic_graph(model, inputs)
    print(f"   输出形状: {outputs.shape}")
    print(f"   信息熵比效率: {efficiency:.4f}")
    
    # 第2步：宇宙经典化效率最大化（CCE）
    print("\n2. 宇宙经典化效率最大化（CCE）示例:")
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    # 正常的优化步骤
    outputs = model(inputs)
    loss = F.mse_loss(outputs, targets)
    loss.backward()
    
    # 应用宇宙经典化效率优化
    efficiency = cosmic_classical_efficiency_step(optimizer, model)
    print(f"   经典化效率: {efficiency:.4f}")
    
    # 第3步：熵驱动宇宙状态空间压缩（CSC）
    print("\n3. 熵驱动宇宙状态空间压缩（CSC）示例:")
    # 创建一个随机张量
    tensor = torch.randn(5, 5)
    print(f"   原始张量:\n{tensor}")
    
    # 应用压缩
    compressed = cosmic_state_compression(tensor, alpha=0.5)
    print(f"   压缩后张量:\n{compressed}")
    
    # 计算压缩率
    nonzero_before = torch.count_nonzero(tensor).item()
    nonzero_after = torch.count_nonzero(compressed).item()
    compression_ratio = nonzero_after / nonzero_before
    print(f"   压缩率: {compression_ratio:.2%}")
    
    # 第4步：使用宇宙同构优化整合模块
    print("\n4. 使用宇宙同构优化整合模块:")
    cosmic_model = convert_to_cosmic(
        SimpleNN(),
        dynamic_graph=True,
        compression=True,
        efficiency=True,
        dynamic_graph_threshold=0.3,
        compression_alpha=0.3,
        efficiency_gamma=0.2
    )
    print("   宇宙同构优化模型结构:")
    print(f"   {cosmic_model}")
    
    # 创建宇宙优化器
    cosmic_optimizer = cosmic_model.create_cosmic_optimizer(optim.Adam, lr=0.001)
    
    # 模拟训练循环
    print("\n   模拟训练循环:")
    for epoch in range(3):
        # 前向传播
        cosmic_optimizer.zero_grad()
        outputs = cosmic_model(inputs)
        loss = F.mse_loss(outputs, targets)
        # 反向传播
        loss.backward()
        cosmic_optimizer.step()
        
        # 打印指标
        metrics = cosmic_model.get_metrics()
        print(f"   Epoch {epoch+1}, Loss: {loss.item():.4f}, Metrics: {metrics}")

if __name__ == "__main__":
    demo() 