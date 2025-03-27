#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
宇宙同构优化集成测试
=================

这个脚本实现并测试完整的宇宙同构优化框架，包括动态计算图、经典化效率和状态压缩。
"""

import os
import sys

# 临时从Python导入路径中移除当前目录，以避免导入本地torch
old_path = sys.path.copy()
if '' in sys.path:
    sys.path.remove('')
if '.' in sys.path:
    sys.path.remove('.')
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir in sys.path:
    sys.path.remove(current_dir)

# 现在导入torch将使用系统安装的版本
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import copy

# 恢复Python路径
sys.path = old_path

print("=== 宇宙同构优化集成测试 ===")

#-------------------------------#
# 1. 宇宙同构动态计算图优化
#-------------------------------#
class CosmicDynamicGraphHook:
    """宇宙同构动态计算图钩子，用于监控和优化模型中的节点效率"""
    
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.node_efficiencies = {}
        self.active_nodes = set()
    
    def __call__(self, module, inputs, outputs):
        # 计算当前节点的信息熵比(效率)
        input_entropy = torch.cat([i.flatten() for i in inputs if isinstance(i, torch.Tensor)]).abs().mean().item()
        output_entropy = outputs.abs().mean().item() if isinstance(outputs, torch.Tensor) else 0
        
        # 避免除零
        if input_entropy < 1e-8:
            efficiency = 0
        else:
            efficiency = output_entropy / input_entropy
        
        # 记录节点效率
        node_name = module.__class__.__name__ + str(id(module))
        self.node_efficiencies[node_name] = efficiency
        
        # 如果效率高于阈值，则标记为活跃节点
        if efficiency > self.threshold:
            self.active_nodes.add(node_name)
        
        return outputs
    
    def get_metrics(self):
        """获取动态计算图优化指标"""
        avg_efficiency = np.mean(list(self.node_efficiencies.values())) if self.node_efficiencies else 0
        active_ratio = len(self.active_nodes) / len(self.node_efficiencies) if self.node_efficiencies else 0
        
        return {
            "avg_node_efficiency": avg_efficiency,
            "active_nodes_ratio": active_ratio,
            "threshold": self.threshold
        }

def cosmic_dynamic_graph(model, inputs, threshold=0.5):
    """应用宇宙同构动态计算图优化到模型"""
    
    # 注册钩子
    hook = CosmicDynamicGraphHook(threshold)
    hooks = []
    
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            hooks.append(module.register_forward_hook(hook))
    
    # 前向传播
    model.eval()
    with torch.no_grad():
        outputs = model(inputs)
    
    # 移除钩子
    for h in hooks:
        h.remove()
    
    # 获取效率指标
    metrics = hook.get_metrics()
    total_efficiency = metrics["avg_node_efficiency"]
    
    return outputs, total_efficiency

#-------------------------------#
# 2. 熵驱动宇宙状态空间压缩
#-------------------------------#
def cosmic_state_compression(tensor, alpha=0.5):
    """对张量进行熵驱动宇宙状态空间压缩"""
    
    # 计算熵驱动阈值
    abs_values = tensor.abs()
    entropy_threshold = alpha * abs_values.mean()
    
    # 应用掩码
    mask = abs_values > entropy_threshold
    compressed = tensor * mask
    
    return compressed

class CosmicCompressedLayer(nn.Module):
    """压缩层包装器，自动应用状态空间压缩"""
    
    def __init__(self, module, alpha=0.5):
        super(CosmicCompressedLayer, self).__init__()
        self.module = module
        self.alpha = alpha
        self.original_params = {}
        self.compressed_params = {}
        
        # 记录原始参数
        for name, param in module.named_parameters():
            self.original_params[name] = param.data.clone()
            
        # 应用压缩
        self.compress_parameters()
    
    def compress_parameters(self):
        """压缩模块参数"""
        for name, param in self.module.named_parameters():
            compressed = cosmic_state_compression(param.data, self.alpha)
            self.compressed_params[name] = compressed
            param.data = compressed
    
    def forward(self, x):
        """前向传播，直接使用压缩参数"""
        return self.module(x)
    
    def get_compression_ratio(self):
        """获取压缩率"""
        total_params = 0
        nonzero_params = 0
        
        for name, param in self.module.named_parameters():
            total_params += param.numel()
            nonzero_params += torch.count_nonzero(param).item()
        
        return nonzero_params / total_params if total_params > 0 else 0

#-------------------------------#
# 3. 宇宙经典化效率最大化
#-------------------------------#
class CosmicOptimizer:
    """宇宙经典化效率优化器，最大化参数更新路径效率"""
    
    def __init__(self, optimizer, gamma=0.1):
        self.optimizer = optimizer
        self.gamma = gamma
        self.efficiency_history = []
        self.param_entropies = {}
    
    def zero_grad(self):
        """清除梯度"""
        self.optimizer.zero_grad()
    
    def step(self):
        """执行优化步骤，最大化经典化效率"""
        
        # 计算参数熵和梯度的相关性
        param_entropy = 0.0
        param_count = 0
        
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    # 计算参数熵(绝对值平均值作为简化的熵估计)
                    param_entropy += p.grad.abs().mean().item()
                    param_count += 1
                    
                    # 记录各参数熵
                    self.param_entropies[id(p)] = p.grad.abs().mean().item()
        
        avg_param_entropy = param_entropy / param_count if param_count > 0 else 0
        
        # 动态调整学习率
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.grad is not None and id(p) in self.param_entropies:
                    p_entropy = self.param_entropies[id(p)]
                    scale = (p_entropy / (avg_param_entropy + 1e-8)) ** self.gamma
                    # 使用numpy的clip代替Python基本类型的clip方法
                    scale = np.clip(scale, 0.5, 2.0)
                    p.grad.data.mul_(scale)
        
        # 执行优化器步骤
        self.optimizer.step()
        
        # 记录效率历史
        self.efficiency_history.append(avg_param_entropy)
        
        return avg_param_entropy
    
    def get_metrics(self):
        """获取优化指标"""
        avg_efficiency = np.mean(self.efficiency_history) if self.efficiency_history else 0
        return {
            "avg_efficiency": avg_efficiency,
            "current_efficiency": self.efficiency_history[-1] if self.efficiency_history else 0,
            "gamma": self.gamma
        }

#-------------------------------#
# 4. 宇宙同构集成模块
#-------------------------------#
class CosmicModule(nn.Module):
    """集成宇宙同构优化的模块"""
    
    def __init__(self, module, 
                 dynamic_graph=True, compression=True, efficiency=True,
                 dynamic_graph_threshold=0.5, compression_alpha=0.5, efficiency_gamma=0.1):
        super(CosmicModule, self).__init__()
        
        # 保存原始模块
        self.module = module
        
        # 配置
        self.use_dynamic_graph = dynamic_graph
        self.use_compression = compression
        self.use_efficiency = efficiency
        
        # 参数
        self.dynamic_graph_threshold = dynamic_graph_threshold
        self.compression_alpha = compression_alpha
        self.efficiency_gamma = efficiency_gamma
        
        # 钩子和指标
        self.dg_hook = None
        self.metrics = {}
        
        # 应用压缩(如果启用)
        if self.use_compression:
            self._apply_compression()
        
        # 注册动态图钩子(如果启用)
        if self.use_dynamic_graph:
            self._register_dg_hook()
    
    def _apply_compression(self):
        """应用状态空间压缩到所有可压缩层"""
        compressed_modules = {}
        
        for name, module in self.module.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                parent_name = '.'.join(name.split('.')[:-1])
                module_name = name.split('.')[-1]
                
                # 创建压缩层
                compressed = CosmicCompressedLayer(module, self.compression_alpha)
                
                # 替换原始模块
                if parent_name:
                    parent = self.module
                    for part in parent_name.split('.'):
                        parent = getattr(parent, part)
                    setattr(parent, module_name, compressed)
                else:
                    setattr(self.module, module_name, compressed)
                
                compressed_modules[name] = compressed
        
        self.compressed_modules = compressed_modules
    
    def _register_dg_hook(self):
        """注册动态计算图钩子"""
        self.dg_hook = CosmicDynamicGraphHook(self.dynamic_graph_threshold)
        self.dg_hooks = []
        
        for name, module in self.module.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                self.dg_hooks.append(module.register_forward_hook(self.dg_hook))
    
    def forward(self, x):
        """前向传播，应用所有优化"""
        return self.module(x)
    
    def create_cosmic_optimizer(self, optimizer_class, **kwargs):
        """创建宇宙经典化效率优化器"""
        base_optimizer = optimizer_class(self.parameters(), **kwargs)
        if self.use_efficiency:
            return CosmicOptimizer(base_optimizer, self.efficiency_gamma)
        return base_optimizer
    
    def get_metrics(self):
        """获取所有优化指标"""
        metrics = {}
        
        # 动态图指标
        if self.use_dynamic_graph and self.dg_hook:
            metrics.update(self.dg_hook.get_metrics())
        
        # 压缩率指标
        if self.use_compression:
            compression_ratios = []
            for name, module in self.compressed_modules.items():
                compression_ratios.append(module.get_compression_ratio())
            
            metrics["avg_compression_ratio"] = np.mean(compression_ratios) if compression_ratios else 0
        
        return metrics
    
    def _count_nonzero_params(self):
        """统计非零参数数量"""
        nonzero = 0
        for p in self.parameters():
            nonzero += torch.count_nonzero(p).item()
        return nonzero

def convert_to_cosmic(module, **kwargs):
    """将普通模块转换为宇宙同构优化模块"""
    return CosmicModule(module, **kwargs)

#-------------------------------#
# 测试
#-------------------------------#
class SimpleNN(nn.Module):
    """用于测试的简单神经网络"""
    
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 5)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def benchmark_models(standard_model, cosmic_model, device="cpu", epochs=3):
    """对比标准模型和宇宙同构优化模型的性能"""
    
    # 准备数据
    inputs = torch.randn(32, 10).to(device)
    targets = torch.randn(32, 5).to(device)
    
    # 创建优化器
    standard_optimizer = torch.optim.Adam(standard_model.parameters(), lr=0.001)
    cosmic_optimizer = cosmic_model.create_cosmic_optimizer(torch.optim.Adam, lr=0.001)
    
    # 训练标准模型
    print("\n=== 标准模型训练 ===")
    standard_time = time.time()
    
    for epoch in range(epochs):
        standard_optimizer.zero_grad()
        outputs = standard_model(inputs)
        loss = F.mse_loss(outputs, targets)
        loss.backward()
        standard_optimizer.step()
        print(f"标准模型 - Epoch {epoch+1}, Loss: {loss.item():.4f}")
    
    standard_time = time.time() - standard_time
    
    # 训练宇宙同构优化模型
    print("\n=== 宇宙同构优化模型训练 ===")
    cosmic_time = time.time()
    
    for epoch in range(epochs):
        cosmic_optimizer.zero_grad()
        outputs = cosmic_model(inputs)
        loss = F.mse_loss(outputs, targets)
        loss.backward()
        cosmic_optimizer.step()
        print(f"宇宙同构模型 - Epoch {epoch+1}, Loss: {loss.item():.4f}")
        
        # 显示优化指标
        metrics = cosmic_model.get_metrics()
        print(f"优化指标: {metrics}")
    
    cosmic_time = time.time() - cosmic_time
    
    # 输出性能对比
    print("\n=== 性能比较 ===")
    print(f"标准模型训练时间: {standard_time:.4f}秒")
    print(f"宇宙同构模型训练时间: {cosmic_time:.4f}秒")
    print(f"速度提升: {standard_time/cosmic_time:.2f}x")
    
    # 参数压缩率
    standard_params = sum(p.numel() for p in standard_model.parameters())
    cosmic_nonzero = cosmic_model._count_nonzero_params()
    compression_rate = 1.0 - (cosmic_nonzero / standard_params)
    print(f"参数压缩率: {compression_rate:.2%}")

def run_tests():
    """运行所有测试"""
    
    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 测试1: 创建模型
    standard_model = SimpleNN().to(device)
    cosmic_model = convert_to_cosmic(
        SimpleNN(),
        dynamic_graph=True,
        compression=True,
        efficiency=True,
        dynamic_graph_threshold=0.3,
        compression_alpha=0.3,
        efficiency_gamma=0.2
    ).to(device)
    
    print(f"\n标准模型:\n{standard_model}")
    print(f"\n宇宙同构优化模型:\n{cosmic_model}")
    
    # 测试2: 测试单个优化组件
    # 2.1 测试动态计算图
    inputs = torch.randn(16, 10).to(device)
    print("\n测试宇宙同构动态计算图优化:")
    outputs, efficiency = cosmic_dynamic_graph(standard_model, inputs, threshold=0.3)
    print(f"动态计算图效率: {efficiency:.4f}")
    
    # 2.2 测试状态压缩
    rand_tensor = torch.randn(5, 5).to(device)
    print("\n测试熵驱动宇宙状态空间压缩:")
    compressed = cosmic_state_compression(rand_tensor, alpha=0.3)
    nonzero_before = torch.count_nonzero(rand_tensor).item()
    nonzero_after = torch.count_nonzero(compressed).item()
    print(f"压缩率: {nonzero_after/nonzero_before:.2%}")
    
    # 测试3: 基准测试
    benchmark_models(standard_model, cosmic_model, device)
    
    print("\n所有测试完成!")

if __name__ == "__main__":
    run_tests() 