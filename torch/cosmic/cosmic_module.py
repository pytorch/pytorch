"""
宇宙同构优化集成模块 (Cosmic Isomorphism Optimization Integration Module)
======================================================================
集成CDG、CCE和CSC三大优化核心，提供完整宇宙同构优化解决方案。

Integrates CDG, CCE, and CSC optimization cores, providing a complete cosmic isomorphism 
optimization solution.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from .cosmic_dynamic_graph import cosmic_dynamic_graph, CosmicDynamicGraphHook
from .cosmic_classical_efficiency import cosmic_classical_efficiency_step, CosmicOptimizer
from .cosmic_state_compression import cosmic_state_compression, CosmicCompressedLayer, compress_module_states

class CosmicModule(nn.Module):
    """
    宇宙同构优化模块 - 集成CDG、CCE和CSC三大优化核心
    
    基于宇宙同构原理，将PyTorch模型视为宇宙，张量为经典物质，
    梯度与计算流为能量流动，参数更新为宇宙演化路径。
    
    用法:
        model = CosmicModule(
            nn.Sequential(
                nn.Linear(784, 256),
                nn.ReLU(),
                nn.Linear(256, 10)
            ),
            dynamic_graph_threshold=0.5,
            compression_alpha=0.5,
            efficiency_gamma=0.1
        )
        
        # 自动应用所有优化
        optimizer = CosmicOptimizer(
            torch.optim.Adam(model.parameters(), lr=0.001),
            model=model
        )
        
        # 训练循环
        for epoch in range(epochs):
            for data, target in train_loader:
                optimizer.zero_grad()
                output = model(data)
                loss = F.nll_loss(output, target)
                loss.backward()
                optimizer.step()
                
    Cosmic Isomorphism Optimization Module - Integrates CDG, CCE, and CSC optimization cores
    
    Based on cosmic isomorphism principle, views PyTorch model as a universe, tensors as classical matter,
    gradients and computation flow as energy flow, parameter updates as universe evolution paths.
    
    Usage:
        model = CosmicModule(
            nn.Sequential(
                nn.Linear(784, 256),
                nn.ReLU(),
                nn.Linear(256, 10)
            ),
            dynamic_graph_threshold=0.5,
            compression_alpha=0.5,
            efficiency_gamma=0.1
        )
        
        # Automatically apply all optimizations
        optimizer = CosmicOptimizer(
            torch.optim.Adam(model.parameters(), lr=0.001),
            model=model
        )
        
        # Training loop
        for epoch in range(epochs):
            for data, target in train_loader:
                optimizer.zero_grad()
                output = model(data)
                loss = F.nll_loss(output, target)
                loss.backward()
                optimizer.step()
    """
    def __init__(
        self, 
        module,
        dynamic_graph_enabled=True,
        dynamic_graph_threshold=0.5,
        dynamic_graph_eta=0.01,
        compression_enabled=True, 
        compression_alpha=0.5,
        compression_auto=True,
        efficiency_enabled=True,
        efficiency_gamma=0.1,
        epsilon=1e-8
    ):
        """
        Args:
            module (nn.Module): 基础PyTorch模块
            dynamic_graph_enabled (bool): 是否启用宇宙动态图优化
            dynamic_graph_threshold (float): 动态图节点保留阈值
            dynamic_graph_eta (float): 动态图学习率
            compression_enabled (bool): 是否启用状态压缩
            compression_alpha (float): 压缩阈值
            compression_auto (bool): 是否自动压缩
            efficiency_enabled (bool): 是否启用经典化效率优化
            efficiency_gamma (float): 经典化效率学习率
            epsilon (float): 数值稳定性常数
            
        Args:
            module (nn.Module): Base PyTorch module
            dynamic_graph_enabled (bool): Whether to enable cosmic dynamic graph optimization
            dynamic_graph_threshold (float): Dynamic graph node retention threshold
            dynamic_graph_eta (float): Dynamic graph learning rate
            compression_enabled (bool): Whether to enable state compression
            compression_alpha (float): Compression threshold
            compression_auto (bool): Whether to automatically compress
            efficiency_enabled (bool): Whether to enable classical efficiency optimization
            efficiency_gamma (float): Classical efficiency learning rate
            epsilon (float): Numerical stability constant
        """
        super(CosmicModule, self).__init__()
        
        # 保存配置参数
        # Save configuration parameters
        self.dynamic_graph_enabled = dynamic_graph_enabled
        self.dynamic_graph_threshold = dynamic_graph_threshold
        self.dynamic_graph_eta = dynamic_graph_eta
        self.compression_enabled = compression_enabled
        self.compression_alpha = compression_alpha
        self.compression_auto = compression_auto
        self.efficiency_enabled = efficiency_enabled
        self.efficiency_gamma = efficiency_gamma
        self.epsilon = epsilon
        
        # 递归应用层压缩，如果启用
        # Recursively apply layer compression, if enabled
        if compression_enabled and compression_auto:
            # 递归处理，将每个子模块包装成CosmicCompressedLayer
            # Recursively process, wrap each submodule in CosmicCompressedLayer
            self._apply_cosmic_compression(module)
        
        # 保存基础模块
        # Save base module
        self.module = module
        
        # 如果启用动态图优化，注册钩子
        # If dynamic graph optimization is enabled, register hook
        if dynamic_graph_enabled:
            self.cdg_hook = CosmicDynamicGraphHook(
                eta=dynamic_graph_eta,
                epsilon=epsilon,
                threshold=dynamic_graph_threshold
            )
            # 注册钩子，为了正确捕获输入
            # Register hook to correctly capture inputs
            self.register_forward_pre_hook(self.cdg_hook)
        
        # 初始化度量收集器
        # Initialize metrics collector
        self.metrics = {
            'dynamic_graph_efficiency': [],
            'compression_ratio': [],
            'classical_efficiency': []
        }
    
    def _apply_cosmic_compression(self, module):
        """
        递归应用宇宙状态压缩到所有子模块
        
        Args:
            module (nn.Module): 要处理的模块
            
        Recursively apply cosmic state compression to all submodules
        
        Args:
            module (nn.Module): Module to process
        """
        # 获取所有子模块
        # Get all submodules
        for name, child in list(module.named_children()):
            # 对于容器类型的模块，递归处理其子模块
            # For container-type modules, recursively process their submodules
            if isinstance(child, (nn.Sequential, nn.ModuleList, nn.ModuleDict)):
                self._apply_cosmic_compression(child)
            else:
                # 判断是否是简单层，如果是则应用压缩
                # Determine if it's a simple layer, if so apply compression
                is_simple_layer = isinstance(child, (
                    nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d,
                    nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d,
                    nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                    nn.LayerNorm, nn.GroupNorm, nn.InstanceNorm1d,
                    nn.InstanceNorm2d, nn.InstanceNorm3d
                ))
                
                if is_simple_layer and not isinstance(child, CosmicCompressedLayer):
                    # 将该层替换为CosmicCompressedLayer
                    # Replace the layer with CosmicCompressedLayer
                    setattr(module, name, CosmicCompressedLayer(
                        child,
                        alpha=self.compression_alpha,
                        epsilon=self.epsilon,
                        auto_compress=self.compression_auto
                    ))
                else:
                    # 对于复杂层，递归处理其子模块
                    # For complex layers, recursively process their submodules
                    self._apply_cosmic_compression(child)
    
    def forward(self, *args, **kwargs):
        """
        前向传播，应用宇宙同构优化
        
        Args:
            *args: 传递给基础模块的位置参数
            **kwargs: 传递给基础模块的关键字参数
            
        Returns:
            模块输出
            
        Forward propagation, apply cosmic isomorphism optimization
        
        Args:
            *args: Positional arguments passed to base module
            **kwargs: Keyword arguments passed to base module
            
        Returns:
            Module output
        """
        # 正常前向传播
        # Normal forward propagation
        outputs = self.module(*args, **kwargs)
        
        # 如果启用了动态图优化并且在训练模式下，应用钩子优化
        # If dynamic graph optimization is enabled and in training mode, apply hook optimization
        if self.dynamic_graph_enabled and self.training:
            _, efficiency = self.cdg_hook.optimize()
            self.metrics['dynamic_graph_efficiency'].append(efficiency)
        
        # 如果启用压缩且未自动压缩，手动触发压缩
        # If compression is enabled and not auto-compressed, manually trigger compression
        if self.compression_enabled and not self.compression_auto and self.training:
            original_size = self._count_nonzero_params()
            compress_module_states(self.module, self.compression_alpha, self.epsilon)
            compressed_size = self._count_nonzero_params()
            compression_ratio = compressed_size / max(original_size, 1)
            self.metrics['compression_ratio'].append(compression_ratio)
        
        return outputs
    
    def _count_nonzero_params(self):
        """
        计算非零参数数量，用于评估压缩率
        
        Returns:
            int: 非零参数数量
            
        Calculate number of non-zero parameters, used to evaluate compression ratio
        
        Returns:
            int: Number of non-zero parameters
        """
        count = 0
        for p in self.parameters():
            count += torch.count_nonzero(p).item()
        return count
    
    def get_metrics(self):
        """
        获取优化指标
        
        Returns:
            dict: 包含各优化指标的平均值
            
        Get optimization metrics
        
        Returns:
            dict: Contains average values of various optimization metrics
        """
        return {k: sum(v)/max(len(v), 1) for k, v in self.metrics.items() if v}
    
    def reset_metrics(self):
        """
        重置优化指标
        
        Reset optimization metrics
        """
        for k in self.metrics:
            self.metrics[k] = []
            
    def create_cosmic_optimizer(self, optimizer_class, *args, **kwargs):
        """
        创建宇宙优化器，自动应用经典化效率优化
        
        Args:
            optimizer_class: PyTorch优化器类 (如torch.optim.Adam)
            *args, **kwargs: 传递给优化器的参数
            
        Returns:
            CosmicOptimizer: 宇宙优化器实例
            
        Create cosmic optimizer, automatically apply classical efficiency optimization
        
        Args:
            optimizer_class: PyTorch optimizer class (like torch.optim.Adam)
            *args, **kwargs: Parameters passed to the optimizer
            
        Returns:
            CosmicOptimizer: Cosmic optimizer instance
        """
        base_optimizer = optimizer_class(self.parameters(), *args, **kwargs)
        
        if self.efficiency_enabled:
            return CosmicOptimizer(
                base_optimizer, 
                model=self,
                epsilon=self.epsilon,
                gamma=self.efficiency_gamma
            )
        else:
            return base_optimizer

# 方便的工厂函数，将现有PyTorch模型转换为宇宙同构优化模型
# Convenient factory function, converts existing PyTorch model to cosmic isomorphism optimization model
def convert_to_cosmic(
    module, 
    dynamic_graph=True, 
    compression=True, 
    efficiency=True,
    **kwargs
):
    """
    将现有PyTorch模型转换为宇宙同构优化模型
    
    Args:
        module (nn.Module): 要转换的PyTorch模块
        dynamic_graph (bool): 是否启用宇宙动态图优化
        compression (bool): 是否启用状态压缩
        efficiency (bool): 是否启用经典化效率优化
        **kwargs: 传递给CosmicModule的其他参数
    
    Returns:
        CosmicModule: 宇宙同构优化后的模型
    """
    return CosmicModule(
        module,
        dynamic_graph_enabled=dynamic_graph,
        compression_enabled=compression,
        efficiency_enabled=efficiency,
        **kwargs
    ) 