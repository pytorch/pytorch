# 当前问题总结和修复

## Bug
autotuning后，3个range都选择了相同的implementation，但仍然生成了3个相同的kernel + if/else dispatch。

## 根本原因  
代码按range数量生成kernels，而不是按unique implementation数量。

## 修复
在custom_op.py的range_based_lowering_fn中：
1. 按implementation分组ranges (使用id(impl_fn)作为key)
2. 如果只有1个unique impl → 单个SubgraphBuffer，无dispatch
3. 如果有N个unique impls → N个kernels + N-way dispatch

## 验证
运行test后检查：
- Log应显示 "Found 1 unique implementations" 或 "Found N unique implementations"  
- 只生成对应数量的Triton kernels
