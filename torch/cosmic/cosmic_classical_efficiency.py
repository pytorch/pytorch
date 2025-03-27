"""
宇宙经典化效率最大化 (Cosmic Classical Efficiency)
===============================================
优化参数更新路径，使计算效率最大化。基于信息熵比修正梯度方向，实现最优经典化。

Optimizes parameter update paths to maximize computational efficiency. Corrects gradient 
directions based on information-entropy ratio to achieve optimal classicalization.
"""

import torch
import torch.optim as optim
import math
from .cosmic_dynamic_graph import _calculate_info_entropy

def cosmic_classical_efficiency_step(optimizer, model=None, epsilon=1e-8, gamma=0.1):
    """
    宇宙经典化效率最大化参数更新
    基于公式：θ^{(t+1)} = θ^{(t)} + γ∇_{θ}E_C(θ^{(t)})
    
    其中E_C为经典化效率函数：E_C = (I_C - S_C) / (I_C + S_C + ϵ)
    
    Args:
        optimizer (torch.optim.Optimizer): PyTorch优化器
        model (nn.Module, optional): 模型，用于获取参数组关联
        epsilon (float): 数值稳定性常数
        gamma (float): 经典化效率学习率
        
    Returns:
        float: 平均经典化效率值
        
    Cosmic classical efficiency maximization parameter update
    Based on formula: θ^{(t+1)} = θ^{(t)} + γ∇_{θ}E_C(θ^{(t)})
    
    where E_C is the classical efficiency function: E_C = (I_C - S_C) / (I_C + S_C + ϵ)
    
    Args:
        optimizer (torch.optim.Optimizer): PyTorch optimizer
        model (nn.Module, optional): Model, used to get parameter group associations
        epsilon (float): Numerical stability constant
        gamma (float): Classical efficiency learning rate
        
    Returns:
        float: Average classical efficiency value
    """
    efficiency_values = []
    
    for group in optimizer.param_groups:
        for p in group['params']:
            if p.grad is None:
                continue
            
            # 计算当前参数的信息熵比
            # Calculate information-entropy ratio for current parameter
            info, entropy, _ = _calculate_info_entropy(p.data, epsilon)
            
            # 计算经典化效率
            # Calculate classical efficiency
            efficiency = (info - entropy) / (info + entropy + epsilon)
            efficiency_values.append(efficiency.item())
            
            # 根据经典化效率修正梯度 - 梯度方向与效率最大化相结合，实现宇宙经典化路径更新
            # Modify gradient based on classical efficiency - combines gradient direction 
            # with efficiency maximization to implement cosmic classical pathway update
            efficiency_gradient = p.grad * (efficiency + epsilon)
            
            # 应用修正后的梯度
            # Apply modified gradient
            p.data.add_(efficiency_gradient, alpha=-group['lr'] * gamma)
    
    # 返回平均经典化效率值，作为优化指标
    # Return average classical efficiency value as optimization metric
    mean_efficiency = sum(efficiency_values) / max(len(efficiency_values), 1)
    return mean_efficiency

# 替代标准优化器的包装类，内置宇宙经典化效率优化
# Wrapper class to replace standard optimizers, with built-in cosmic classical efficiency optimization
class CosmicOptimizer(optim.Optimizer):
    """
    宇宙经典化效率优化器包装类
    
    用法:
        optimizer = CosmicOptimizer(
            torch.optim.Adam(model.parameters(), lr=0.001),
            model=model,
            gamma=0.1
        )
        optimizer.step()
        
    Cosmic classical efficiency optimizer wrapper class
    
    Usage:
        optimizer = CosmicOptimizer(
            torch.optim.Adam(model.parameters(), lr=0.001),
            model=model,
            gamma=0.1
        )
        optimizer.step()
    """
    def __init__(self, base_optimizer, model=None, epsilon=1e-8, gamma=0.1):
        """
        Args:
            base_optimizer (torch.optim.Optimizer): 基础优化器
            model (nn.Module, optional): 模型引用
            epsilon (float): 数值稳定性常数
            gamma (float): 经典化效率学习率
            
        Args:
            base_optimizer (torch.optim.Optimizer): Base optimizer
            model (nn.Module, optional): Model reference
            epsilon (float): Numerical stability constant
            gamma (float): Classical efficiency learning rate
        """
        self.base_optimizer = base_optimizer
        self.model = model
        self.epsilon = epsilon
        self.gamma = gamma
        self.param_groups = self.base_optimizer.param_groups
        self.state = self.base_optimizer.state
        
        # 添加钩子到基础优化器
        # Add hooks to base optimizer
        self._register_hooks()
    
    def _register_hooks(self):
        """
        注册宇宙优化钩子
        Register cosmic optimization hooks
        """
        self.base_optimizer._cosmic_post_step_hook = lambda: cosmic_classical_efficiency_step(
            self.base_optimizer, self.model, self.epsilon, self.gamma
        )
    
    def zero_grad(self, set_to_none=False):
        """
        清除梯度
        Clear gradients
        """
        self.base_optimizer.zero_grad(set_to_none)
    
    def step(self, closure=None):
        """
        执行优化步骤
        
        Args:
            closure (callable, optional): 重新评估模型并返回损失的闭包
            
        Returns:
            loss: 如果提供了closure，返回closure的结果
            
        Perform optimization step
        
        Args:
            closure (callable, optional): A closure that reevaluates the model and returns the loss
            
        Returns:
            loss: If closure is provided, returns the result of closure
        """
        loss = self.base_optimizer.step(closure)
        
        # 应用宇宙经典化效率优化
        # Apply cosmic classical efficiency optimization
        efficiency = self.base_optimizer._cosmic_post_step_hook()
        
        return loss
    
    def __getattr__(self, name):
        """
        委托到基础优化器的其他方法
        
        Args:
            name (str): 属性名称
            
        Returns:
            任何基础优化器的属性或方法
            
        Delegate to other methods of base optimizer
        
        Args:
            name (str): Attribute name
            
        Returns:
            Any attribute or method of base optimizer
        """
        if name == 'param_groups' or name == 'state':
            return getattr(self, name)
        return getattr(self.base_optimizer, name)

# 优化器step钩子，便于集成到现有训练流程
# Optimizer step hook, for easy integration into existing training workflows
class CosmicClassicalEfficiencyHook:
    """
    优化器步骤钩子，在每次优化器更新后应用宇宙经典化效率优化
    
    用法:
        hook = CosmicClassicalEfficiencyHook(model, gamma=0.1)
        # 在优化器步骤后调用
        optimizer.step()
        hook(optimizer)
        
    Optimizer step hook, applies cosmic classical efficiency optimization after each optimizer update
    
    Usage:
        hook = CosmicClassicalEfficiencyHook(model, gamma=0.1)
        # Call after optimizer step
        optimizer.step()
        hook(optimizer)
    """
    def __init__(self, model=None, epsilon=1e-8, gamma=0.1):
        """
        Args:
            model (nn.Module, optional): 模型引用
            epsilon (float): 数值稳定性常数
            gamma (float): 经典化效率学习率
            
        Args:
            model (nn.Module, optional): Model reference
            epsilon (float): Numerical stability constant
            gamma (float): Classical efficiency learning rate
        """
        self.model = model
        self.epsilon = epsilon
        self.gamma = gamma
    
    def __call__(self, optimizer):
        """
        在优化器步骤后调用
        
        Args:
            optimizer (torch.optim.Optimizer): PyTorch优化器
            
        Returns:
            float: 平均经典化效率值
            
        Call after optimizer step
        
        Args:
            optimizer (torch.optim.Optimizer): PyTorch optimizer
            
        Returns:
            float: Average classical efficiency value
        """
        return cosmic_classical_efficiency_step(
            optimizer, 
            self.model, 
            self.epsilon, 
            self.gamma
        ) 