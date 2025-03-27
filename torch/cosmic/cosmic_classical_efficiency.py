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
    """
    efficiency_values = []
    
    for group in optimizer.param_groups:
        for p in group['params']:
            if p.grad is None:
                continue
            
            # 计算当前参数的信息熵比
            info, entropy, _ = _calculate_info_entropy(p.data, epsilon)
            
            # 计算经典化效率
            efficiency = (info - entropy) / (info + entropy + epsilon)
            efficiency_values.append(efficiency.item())
            
            # 根据经典化效率修正梯度
            # 梯度方向与效率最大化相结合，实现宇宙经典化路径更新
            efficiency_gradient = p.grad * (efficiency + epsilon)
            
            # 应用修正后的梯度
            p.data.add_(efficiency_gradient, alpha=-group['lr'] * gamma)
    
    # 返回平均经典化效率值，作为优化指标
    mean_efficiency = sum(efficiency_values) / max(len(efficiency_values), 1)
    return mean_efficiency

# 替代标准优化器的包装类，内置宇宙经典化效率优化
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
    """
    def __init__(self, base_optimizer, model=None, epsilon=1e-8, gamma=0.1):
        """
        Args:
            base_optimizer (torch.optim.Optimizer): 基础优化器
            model (nn.Module, optional): 模型引用
            epsilon (float): 数值稳定性常数
            gamma (float): 经典化效率学习率
        """
        self.base_optimizer = base_optimizer
        self.model = model
        self.epsilon = epsilon
        self.gamma = gamma
        self.param_groups = self.base_optimizer.param_groups
        self.state = self.base_optimizer.state
        
        # 添加钩子到基础优化器
        self._register_hooks()
    
    def _register_hooks(self):
        """注册宇宙优化钩子"""
        self.base_optimizer._cosmic_post_step_hook = lambda: cosmic_classical_efficiency_step(
            self.base_optimizer, self.model, self.epsilon, self.gamma
        )
    
    def zero_grad(self, set_to_none=False):
        """清除梯度"""
        self.base_optimizer.zero_grad(set_to_none)
    
    def step(self, closure=None):
        """执行优化步骤"""
        loss = self.base_optimizer.step(closure)
        
        # 应用宇宙经典化效率优化
        efficiency = self.base_optimizer._cosmic_post_step_hook()
        
        return loss
    
    def __getattr__(self, name):
        """委托到基础优化器的其他方法"""
        if name == 'param_groups' or name == 'state':
            return getattr(self, name)
        return getattr(self.base_optimizer, name)

# 优化器step钩子，便于集成到现有训练流程
class CosmicClassicalEfficiencyHook:
    """
    优化器步骤钩子，在每次优化器更新后应用宇宙经典化效率优化
    
    用法:
        hook = CosmicClassicalEfficiencyHook(model, gamma=0.1)
        # 在优化器步骤后调用
        optimizer.step()
        hook(optimizer)
    """
    def __init__(self, model=None, epsilon=1e-8, gamma=0.1):
        self.model = model
        self.epsilon = epsilon
        self.gamma = gamma
    
    def __call__(self, optimizer):
        """在优化器步骤后调用"""
        return cosmic_classical_efficiency_step(
            optimizer, 
            self.model, 
            self.epsilon, 
            self.gamma
        ) 