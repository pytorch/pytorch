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
    """
    # 信息量计算 - 使用绝对值均值作为信息量度量
    info = tensor.abs().mean()
    
    # 熵计算 - 使用概率分布的Shannon熵
    if tensor.numel() > 1:
        # 对于多元素张量，计算分布熵
        normalized = tensor.abs() / (tensor.abs().sum() + epsilon)
        entropy = -(normalized * torch.log2(normalized + epsilon)).sum()
    else:
        # 对于标量，使用数值本身作为熵估计
        entropy = torch.tensor(epsilon, device=tensor.device)
    
    # 信息熵比计算
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
    """
    # 记录原始训练状态
    training_state = model.training
    
    # 确保模型处于评估模式，以免影响批归一化等层的统计数据
    model.eval()
    
    # 前向传播
    outputs = model(inputs)
    
    # 对输出计算信息熵比
    if isinstance(outputs, torch.Tensor):
        output_tensors = [outputs]
    elif isinstance(outputs, (list, tuple)):
        output_tensors = [o for o in outputs if isinstance(o, torch.Tensor)]
    else:
        # 对于字典等其他输出类型，尝试提取张量
        try:
            output_tensors = [o for o in outputs.values() if isinstance(o, torch.Tensor)]
        except (AttributeError, TypeError):
            output_tensors = []
            
    if not output_tensors:
        # 如果没有找到张量输出，无法进行优化
        if training_state:
            model.train()
        return outputs, 0.0
    
    # 计算输出的信息熵比
    output_info_entropy_ratios = []
    for tensor in output_tensors:
        _, _, ratio = _calculate_info_entropy(tensor, epsilon)
        output_info_entropy_ratios.append(ratio)
    
    # 使用平均信息熵比作为全局优化目标
    mean_output_ratio = sum(output_info_entropy_ratios) / len(output_info_entropy_ratios)
    
    # 创建虚拟损失以驱动反向传播
    loss_efficiency = mean_output_ratio
    loss_efficiency.backward()
    
    # 自动裁剪低效节点
    frozen_count = 0
    total_count = 0
    for name, param in model.named_parameters():
        if param.grad is not None:
            total_count += 1
            
            # 计算参数的信息熵比
            info, entropy, param_ratio = _calculate_info_entropy(param.data, epsilon)
            
            # 如果参数的信息熵比低于阈值 * 输出的信息熵比，则冻结该参数
            # 这样确保高效参数保持活跃，低效参数被冻结，同时考虑全局目标
            efficiency_threshold = threshold * mean_output_ratio
            if param_ratio < efficiency_threshold:
                param.grad = None  # 冻结低效率参数
                frozen_count += 1
    
    # 恢复原始训练状态
    if training_state:
        model.train()
        
    # 返回输出和信息熵比（可用于监控优化进度）
    return outputs, mean_output_ratio.item()

# 添加作为模型钩子的版本，便于集成到现有训练流程
class CosmicDynamicGraphHook:
    """
    宇宙同构动态计算图优化钩子，可用于forward_pre_hook
    
    示例:
        model.register_forward_pre_hook(CosmicDynamicGraphHook(eta=0.01))
    """
    def __init__(self, eta=0.01, epsilon=1e-8, threshold=0.5):
        self.eta = eta
        self.epsilon = epsilon
        self.threshold = threshold
    
    def __call__(self, module, inputs):
        # 钩子调用时不直接优化，仅在传播后优化
        # 在此保存模型和输入，用于后续的cosmic_dynamic_graph调用
        self.module = module
        self.inputs = inputs
        return inputs
    
    def optimize(self):
        """在前向传播后调用，执行优化"""
        if hasattr(self, 'module') and hasattr(self, 'inputs'):
            return cosmic_dynamic_graph(
                self.module, 
                self.inputs,
                eta=self.eta, 
                epsilon=self.epsilon,
                threshold=self.threshold
            )
        return None, 0.0 