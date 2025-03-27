"""
宇宙同构优化 (Cosmic Isomorphism Optimization)
==========================================
基于量子经典二元论的PyTorch优化模块，提供计算图、参数更新和状态压缩的高效优化。

PyTorch optimization module based on quantum-classical dualism, providing efficient 
optimization for computation graphs, parameter updates, and state compression.
"""

from .cosmic_dynamic_graph import cosmic_dynamic_graph, CosmicDynamicGraphHook
from .cosmic_classical_efficiency import cosmic_classical_efficiency_step, CosmicOptimizer, CosmicClassicalEfficiencyHook
from .cosmic_state_compression import cosmic_state_compression, CosmicCompressedLayer, compress_module_states
from .cosmic_module import CosmicModule, convert_to_cosmic

__all__ = [
    # 宇宙同构动态计算图 (Cosmic Dynamic Graph)
    "cosmic_dynamic_graph",
    "CosmicDynamicGraphHook",
    
    # 宇宙经典化效率最大化 (Cosmic Classical Efficiency)
    "cosmic_classical_efficiency_step",
    "CosmicOptimizer",
    "CosmicClassicalEfficiencyHook",
    
    # 熵驱动宇宙状态空间压缩 (Cosmic State Compression)
    "cosmic_state_compression",
    "CosmicCompressedLayer",
    "compress_module_states",
    
    # 高级集成模块 (Advanced Integration Modules)
    "CosmicModule",
    "convert_to_cosmic"
] 