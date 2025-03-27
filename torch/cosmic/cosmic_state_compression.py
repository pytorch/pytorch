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
    """
    # 计算元素级信息熵比
    # 为了元素级计算效率，我们使用局部窗口方法而不是针对每个元素单独计算
    
    # 对于简单标量或1维张量，直接使用全局信息熵比
    if tensor.dim() <= 1 or tensor.numel() <= 1:
        info, entropy, _ = _calculate_info_entropy(tensor, epsilon)
        ratio = info / (entropy + epsilon)
        mask = torch.sigmoid(ratio - alpha)
        return tensor * mask
    
    # 对于2维以上的张量，使用局部滑动窗口计算信息熵比
    # 这些操作都是在移动到CPU后执行，避免GPU内存问题
    device = tensor.device
    dtype = tensor.dtype
    
    # 创建信息量张量
    info = tensor.abs().float()
    
    # 对于大型张量，应用降维或池化以减少计算量
    if tensor.numel() > 1_000_000:
        # 对于非常大的张量，使用池化减少计算复杂度
        # 这里我们应用自适应平均池化减少维度，同时保持张量的相对结构
        shape = tensor.shape
        if tensor.dim() >= 2:
            # 确定目标大小，最大限制为每个维度不超过100
            target_size = [min(100, s) for s in shape]
            
            # 使用自适应池化减小规模
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
                info = info.view(-1).mean().view(1, 1)
                
            # 调整回原始大小
            info = F.interpolate(info.unsqueeze(0).unsqueeze(0), 
                                size=shape, 
                                mode='nearest').squeeze(0).squeeze(0)
    
    # 计算局部熵 - 使用平滑化而不是窗口计算以提高效率
    # 对张量进行归一化处理
    sum_tensor = torch.sum(info) + epsilon
    normalized = info / sum_tensor
    
    # 计算香农熵的近似 - 使用log2
    entropy = -(normalized * torch.log2(normalized + epsilon))
    
    # 计算信息熵比
    ratio = info / (entropy + epsilon)
    
    # 使用Sigmoid函数生成软掩码
    mask = torch.sigmoid(ratio - alpha)
    
    # 应用掩码进行压缩
    compressed_tensor = tensor * mask.to(device).to(dtype)
    
    return compressed_tensor

# 集成到模块中的压缩函数
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
    """
    if not inplace:
        # 如果不是原地操作，创建模块的深拷贝
        import copy
        module = copy.deepcopy(module)
    
    # 压缩参数
    for name, param in module.named_parameters():
        if param.requires_grad:  # 只压缩需要梯度的参数
            compressed_param = cosmic_state_compression(param.data, alpha, epsilon)
            param.data = compressed_param
    
    # 压缩缓冲区
    for name, buf in module.named_buffers():
        compressed_buf = cosmic_state_compression(buf, alpha, epsilon)
        # 直接替换缓冲区的数据
        buf.copy_(compressed_buf)
    
    return module

# 层压缩包装类，可应用于任何nn.Module
class CosmicCompressedLayer(nn.Module):
    """
    宇宙状态压缩层包装器，自动应用熵驱动压缩
    
    用法:
        compressed_layer = CosmicCompressedLayer(nn.Linear(784, 100), alpha=0.5)
    """
    def __init__(self, module, alpha=0.5, epsilon=1e-8, auto_compress=True):
        """
        Args:
            module (nn.Module): 要包装的PyTorch模块
            alpha (float): 熵压缩阈值
            epsilon (float): 数值稳定性常数
            auto_compress (bool): 是否在每次前向传播后自动压缩
        """
        super(CosmicCompressedLayer, self).__init__()
        self.module = module
        self.alpha = alpha
        self.epsilon = epsilon
        self.auto_compress = auto_compress
    
    def forward(self, *args, **kwargs):
        """执行前向传播并可选择性地应用压缩"""
        # 正常前向传播
        output = self.module(*args, **kwargs)
        
        # 如果设置自动压缩，每次前向传播后压缩模块状态
        if self.auto_compress and self.training:
            compress_module_states(self.module, self.alpha, self.epsilon)
            
        # 如果输出是张量，也可以选择性地压缩输出状态
        if isinstance(output, torch.Tensor) and self.training and self.auto_compress:
            output = cosmic_state_compression(output, self.alpha, self.epsilon)
            
        return output
    
    def compress(self):
        """手动触发压缩"""
        return compress_module_states(self.module, self.alpha, self.epsilon)
    
    def __repr__(self):
        return f"CosmicCompressedLayer(alpha={self.alpha}, auto_compress={self.auto_compress}, {self.module.__repr__()})" 