"""
熵驱动宇宙状态空间压缩 (Cosmic State Compression)
=============================================
将低信息熵比的张量区域压缩，保留高信息熵比区域，动态优化内存使用。

Compresses tensor regions with low information-entropy ratio, preserves regions with high ratio,
dynamically optimizes memory usage.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .cosmic_dynamic_graph import _calculate_info_entropy

def cosmic_state_compression(tensor, alpha=0.5, epsilon=1e-8):
    """
    熵驱动宇宙状态空间压缩
    基于公式：T' = T ⊙ σ((I_C(T) / (S_C(T) + ε)) - α)
    
    将低信息熵比的张量区域压缩，保留高信息熵比区域
    
    Args:
        tensor (torch.Tensor): 需要压缩的张量
        alpha (float): 熵压缩阈值，值越大压缩越激进
        epsilon (float): 数值稳定性常数
        
    Returns:
        torch.Tensor: 压缩后的张量
        
    Entropy-driven cosmic state space compression
    Based on formula: T' = T ⊙ σ((I_C(T) / (S_C(T) + ε)) - α)
    
    Compresses tensor regions with low information-entropy ratio, preserves regions with high ratio
    
    Args:
        tensor (torch.Tensor): Tensor to compress
        alpha (float): Entropy compression threshold, higher values lead to more aggressive compression
        epsilon (float): Numerical stability constant
        
    Returns:
        torch.Tensor: Compressed tensor
    """
    # 计算元素级信息熵比 - 为了元素级计算效率，我们使用局部窗口方法而不是针对每个元素单独计算
    # Calculate element-level information-entropy ratio - for efficiency, we use local window method 
    # instead of calculating for each element separately
    
    # 对于简单标量或1维张量，直接使用全局信息熵比
    # For simple scalars or 1D tensors, use global information-entropy ratio directly
    if tensor.dim() <= 1 or tensor.numel() <= 1:
        info, entropy, _ = _calculate_info_entropy(tensor, epsilon)
        ratio = info / (entropy + epsilon)
        mask = torch.sigmoid(ratio - alpha)
        return tensor * mask
    
    # 对于2维以上的张量，使用局部滑动窗口计算信息熵比
    # For 2D+ tensors, use local sliding window to calculate information-entropy ratio
    device = tensor.device
    dtype = tensor.dtype
    
    # 创建信息量张量
    # Create information tensor
    info = tensor.abs().float()
    
    # 对于大型张量，应用降维或池化以减少计算量
    # For large tensors, apply dimensionality reduction or pooling to reduce computation
    if tensor.numel() > 1_000_000:
        # 对于非常大的张量，使用池化减少计算复杂度
        # For very large tensors, use pooling to reduce computational complexity
        shape = tensor.shape
        if tensor.dim() >= 2:
            # 确定目标大小，最大限制为每个维度不超过100
            # Determine target size, with a maximum of 100 for each dimension
            target_size = [min(100, s) for s in shape]
            
            # 使用自适应池化减小规模
            # Use adaptive pooling to reduce scale
            if tensor.dim() == 2:
                info = F.adaptive_avg_pool2d(info.unsqueeze(0).unsqueeze(0), 
                                            output_size=target_size).squeeze(0).squeeze(0)
            elif tensor.dim() == 3:
                info = F.adaptive_avg_pool2d(info.unsqueeze(0), 
                                            output_size=target_size[1:]).squeeze(0)
            elif tensor.dim() == 4:
                info = F.adaptive_avg_pool2d(info, output_size=target_size[2:])
            else:
                # 对于更高维度，简化处理
                # For higher dimensions, simplify processing
                info = info.view(-1).mean().view(1, 1)
                
            # 调整回原始大小
            # Resize back to original size
            info = F.interpolate(info.unsqueeze(0).unsqueeze(0), 
                                size=shape, 
                                mode='nearest').squeeze(0).squeeze(0)
    
    # 计算局部熵 - 使用平滑化而不是窗口计算以提高效率
    # Calculate local entropy - use smoothing instead of window calculation for efficiency
    
    # 对张量进行归一化处理
    # Normalize the tensor
    sum_tensor = torch.sum(info) + epsilon
    normalized = info / sum_tensor
    
    # 计算香农熵的近似 - 使用log2
    # Calculate Shannon entropy approximation - using log2
    entropy = -(normalized * torch.log2(normalized + epsilon))
    
    # 计算信息熵比
    # Calculate information-entropy ratio
    ratio = info / (entropy + epsilon)
    
    # 使用Sigmoid函数生成软掩码
    # Use Sigmoid function to generate soft mask
    mask = torch.sigmoid(ratio - alpha)
    
    # 应用掩码进行压缩
    # Apply mask for compression
    compressed_tensor = tensor * mask.to(device).to(dtype)
    
    return compressed_tensor

# 集成到模块中的压缩函数
# Compression function integrated into modules
def compress_module_states(module, alpha=0.5, epsilon=1e-8, inplace=True):
    """
    压缩模块中的所有参数和缓冲区
    
    Args:
        module (nn.Module): 要压缩的PyTorch模块
        alpha (float): 熵压缩阈值
        epsilon (float): 数值稳定性常数
        inplace (bool): 是否原地修改模块参数
        
    Returns:
        nn.Module: 压缩后的模块（如果inplace=True则为原始模块）
        
    Compress all parameters and buffers in a module
    
    Args:
        module (nn.Module): PyTorch module to compress
        alpha (float): Entropy compression threshold
        epsilon (float): Numerical stability constant
        inplace (bool): Whether to modify module parameters in-place
        
    Returns:
        nn.Module: Compressed module (original module if inplace=True)
    """
    if not inplace:
        # 如果不是原地操作，创建模块的深拷贝
        # If not in-place, create a deep copy of the module
        import copy
        module = copy.deepcopy(module)
    
    # 压缩参数
    # Compress parameters
    for name, param in module.named_parameters():
        if param.requires_grad:  # 只压缩需要梯度的参数 / Only compress parameters that require gradients
            compressed_param = cosmic_state_compression(param.data, alpha, epsilon)
            param.data = compressed_param
    
    # 压缩缓冲区
    # Compress buffers
    for name, buf in module.named_buffers():
        compressed_buf = cosmic_state_compression(buf, alpha, epsilon)
        # 直接替换缓冲区的数据
        # Directly replace buffer data
        buf.copy_(compressed_buf)
    
    return module

# 层压缩包装类，可应用于任何nn.Module
# Layer compression wrapper class, can be applied to any nn.Module
class CosmicCompressedLayer(nn.Module):
    """
    宇宙状态压缩层包装器，自动应用熵驱动压缩
    
    用法:
        compressed_layer = CosmicCompressedLayer(nn.Linear(784, 100), alpha=0.5)
        
    Cosmic state compression layer wrapper, automatically applies entropy-driven compression
    
    Usage:
        compressed_layer = CosmicCompressedLayer(nn.Linear(784, 100), alpha=0.5)
    """
    def __init__(self, module, alpha=0.5, epsilon=1e-8, auto_compress=True):
        """
        Args:
            module (nn.Module): 要包装的PyTorch模块
            alpha (float): 熵压缩阈值
            epsilon (float): 数值稳定性常数
            auto_compress (bool): 是否在每次前向传播后自动压缩
            
        Args:
            module (nn.Module): PyTorch module to wrap
            alpha (float): Entropy compression threshold
            epsilon (float): Numerical stability constant
            auto_compress (bool): Whether to automatically compress after each forward pass
        """
        super(CosmicCompressedLayer, self).__init__()
        self.module = module
        self.alpha = alpha
        self.epsilon = epsilon
        self.auto_compress = auto_compress
    
    def forward(self, *args, **kwargs):
        """
        执行前向传播并可选择性地应用压缩
        
        Args:
            *args: 传递给包装模块的位置参数
            **kwargs: 传递给包装模块的关键字参数
            
        Returns:
            torch.Tensor: 模块的输出，可能被压缩
            
        Perform forward pass and optionally apply compression
        
        Args:
            *args: Positional arguments passed to wrapped module
            **kwargs: Keyword arguments passed to wrapped module
            
        Returns:
            torch.Tensor: Module output, possibly compressed
        """
        # 正常前向传播
        # Normal forward propagation
        output = self.module(*args, **kwargs)
        
        # 如果设置自动压缩，每次前向传播后压缩模块状态
        # If auto-compress is set, compress module state after each forward pass
        if self.auto_compress and self.training:
            compress_module_states(self.module, self.alpha, self.epsilon)
            
        # 如果输出是张量，也可以选择性地压缩输出状态
        # If output is a tensor, optionally compress output state as well
        if isinstance(output, torch.Tensor) and self.training and self.auto_compress:
            output = cosmic_state_compression(output, self.alpha, self.epsilon)
            
        return output
    
    def compress(self):
        """
        手动触发压缩
        
        Returns:
            nn.Module: 压缩后的模块
            
        Manually trigger compression
        
        Returns:
            nn.Module: Compressed module
        """
        return compress_module_states(self.module, self.alpha, self.epsilon)
    
    def __repr__(self):
        """
        模块的字符串表示
        
        Returns:
            str: 包含模块配置的字符串
            
        String representation of module
        
        Returns:
            str: String containing module configuration
        """
        return f"CosmicCompressedLayer(alpha={self.alpha}, auto_compress={self.auto_compress}, {self.module.__repr__()})" 