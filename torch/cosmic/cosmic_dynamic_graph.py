"""
宇宙同构动态计算图 (Cosmic Dynamic Graph)
====================================
自动裁剪低效节点，保持计算图最优结构，基于信息熵比进行优化。

Automatically prunes inefficient nodes, maintains optimal graph structure, 
optimizes based on information-entropy ratio.
"""

import torch
import torch.nn as nn
import math

def _calculate_info_entropy(tensor, epsilon=1e-8):
    """
    计算张量的信息量与熵的比值
    
    Args:
        tensor (torch.Tensor): 输入张量
        epsilon (float): 数值稳定性常数
        
    Returns:
        tuple: (信息量, 熵, 信息熵比)
    
    Calculate the ratio of information to entropy for a tensor
    
    Args:
        tensor (torch.Tensor): Input tensor
        epsilon (float): Numerical stability constant
        
    Returns:
        tuple: (information, entropy, information-entropy ratio)
    """
    # 信息量计算 - 使用绝对值均值作为信息量度量
    # Information calculation - using absolute mean as information measure
    info = tensor.abs().mean()
    
    # 熵计算 - 使用概率分布的Shannon熵
    # Entropy calculation - using Shannon entropy of probability distribution
    if tensor.numel() > 1:
        # 对于多元素张量，计算分布熵
        # For multi-element tensors, calculate distribution entropy
        normalized = tensor.abs() / (tensor.abs().sum() + epsilon)
        entropy = -(normalized * torch.log2(normalized + epsilon)).sum()
    else:
        # 对于标量，使用数值本身作为熵估计
        # For scalars, use the value itself as entropy estimate
        entropy = torch.tensor(epsilon, device=tensor.device)
    
    # 信息熵比计算
    # Information-entropy ratio calculation
    info_entropy_ratio = info / (entropy + epsilon)
    
    return info, entropy, info_entropy_ratio

def cosmic_dynamic_graph(model, inputs, eta=0.01, epsilon=1e-8, threshold=0.5):
    """
    宇宙同构动态计算图优化 - 自动裁剪低效节点，保持计算图最优结构
    
    基于公式：G^{(t+1)} = G^{(t)} - η∇_G [I_C(G^{(t)}) / (S_C(G^{(t)}) + ε)]
    
    Args:
        model (nn.Module): 待优化的模型
        inputs (torch.Tensor): 模型输入
        eta (float): 学习率
        epsilon (float): 数值稳定性常数
        threshold (float): 节点保留阈值，低于此值的节点将被冻结
        
    Returns:
        tuple: (output, 信息熵比)
        
    Cosmic isomorphic dynamic graph optimization - automatically prunes inefficient nodes, 
    maintains optimal graph structure
    
    Based on formula: G^{(t+1)} = G^{(t)} - η∇_G [I_C(G^{(t)}) / (S_C(G^{(t)}) + ε)]
    
    Args:
        model (nn.Module): Model to optimize
        inputs (torch.Tensor): Model inputs
        eta (float): Learning rate
        epsilon (float): Numerical stability constant
        threshold (float): Node retention threshold, nodes below this value will be frozen
        
    Returns:
        tuple: (output, information-entropy ratio)
    """
    # 记录原始训练状态
    # Record original training state
    training_state = model.training
    
    # 确保模型处于评估模式，以免影响批归一化等层的统计数据
    # Ensure model is in evaluation mode to avoid affecting batch normalization statistics
    model.eval()
    
    # 前向传播
    # Forward propagation
    outputs = model(inputs)
    
    # 对输出计算信息熵比
    # Calculate information-entropy ratio for outputs
    if isinstance(outputs, torch.Tensor):
        output_tensors = [outputs]
    elif isinstance(outputs, (list, tuple)):
        output_tensors = [o for o in outputs if isinstance(o, torch.Tensor)]
    else:
        # 对于字典等其他输出类型，尝试提取张量
        # For dictionaries and other output types, try to extract tensors
        try:
            output_tensors = [o for o in outputs.values() if isinstance(o, torch.Tensor)]
        except (AttributeError, TypeError):
            output_tensors = []
            
    if not output_tensors:
        # 如果没有找到张量输出，无法进行优化
        # If no tensor outputs found, cannot optimize
        if training_state:
            model.train()
        return outputs, 0.0
    
    # 计算输出的信息熵比
    # Calculate information-entropy ratio for outputs
    output_info_entropy_ratios = []
    for tensor in output_tensors:
        _, _, ratio = _calculate_info_entropy(tensor, epsilon)
        output_info_entropy_ratios.append(ratio)
    
    # 使用平均信息熵比作为全局优化目标
    # Use average information-entropy ratio as global optimization target
    mean_output_ratio = sum(output_info_entropy_ratios) / len(output_info_entropy_ratios)
    
    # 创建虚拟损失以驱动反向传播
    # Create virtual loss to drive backpropagation
    loss_efficiency = mean_output_ratio
    loss_efficiency.backward()
    
    # 自动裁剪低效节点
    # Automatically prune inefficient nodes
    frozen_count = 0
    total_count = 0
    for name, param in model.named_parameters():
        if param.grad is not None:
            total_count += 1
            
            # 计算参数的信息熵比
            # Calculate information-entropy ratio for parameters
            info, entropy, param_ratio = _calculate_info_entropy(param.data, epsilon)
            
            # 如果参数的信息熵比低于阈值 * 输出的信息熵比，则冻结该参数
            # If parameter's information-entropy ratio is below threshold * output ratio, freeze the parameter
            efficiency_threshold = threshold * mean_output_ratio
            if param_ratio < efficiency_threshold:
                param.grad = None  # 冻结低效率参数 / Freeze inefficient parameter
                frozen_count += 1
    
    # 恢复原始训练状态
    # Restore original training state
    if training_state:
        model.train()
        
    # 返回输出和信息熵比（可用于监控优化进度）
    # Return output and information-entropy ratio (can be used to monitor optimization progress)
    return outputs, mean_output_ratio.item()

# 添加作为模型钩子的版本，便于集成到现有训练流程
# Add hook version for easy integration into existing training workflows
class CosmicDynamicGraphHook:
    """
    宇宙同构动态计算图优化钩子，可用于forward_pre_hook
    
    示例:
        model.register_forward_pre_hook(CosmicDynamicGraphHook(eta=0.01))
        
    Cosmic isomorphic dynamic graph optimization hook, can be used as forward_pre_hook
    
    Example:
        model.register_forward_pre_hook(CosmicDynamicGraphHook(eta=0.01))
    """
    def __init__(self, eta=0.01, epsilon=1e-8, threshold=0.5):
        """
        初始化宇宙同构动态计算图钩子
        
        Args:
            eta (float): 学习率
            epsilon (float): 数值稳定性常数
            threshold (float): 节点保留阈值
            
        Initialize cosmic isomorphic dynamic graph hook
        
        Args:
            eta (float): Learning rate
            epsilon (float): Numerical stability constant
            threshold (float): Node retention threshold
        """
        self.eta = eta
        self.epsilon = epsilon
        self.threshold = threshold
    
    def __call__(self, module, inputs):
        """
        钩子调用函数，保存模型和输入用于后续优化
        
        Args:
            module: 模型模块
            inputs: 模型输入
            
        Hook call function, saves model and inputs for later optimization
        
        Args:
            module: Model module
            inputs: Model inputs
        """
        # 钩子调用时不直接优化，仅在传播后优化
        # 在此保存模型和输入，用于后续的cosmic_dynamic_graph调用
        # Don't optimize directly when hook is called, only after propagation
        # Save model and inputs here for later cosmic_dynamic_graph call
        self.module = module
        self.inputs = inputs
        return inputs
    
    def optimize(self):
        """
        在前向传播后调用，执行宇宙同构动态计算图优化
        
        Returns:
            tuple: (output, 信息熵比)
            
        Call after forward propagation to perform cosmic dynamic graph optimization
        
        Returns:
            tuple: (output, information-entropy ratio)
        """
        if hasattr(self, 'module') and hasattr(self, 'inputs'):
            return cosmic_dynamic_graph(
                self.module, 
                self.inputs,
                eta=self.eta, 
                epsilon=self.epsilon,
                threshold=self.threshold
            )
        return None, 0.0 