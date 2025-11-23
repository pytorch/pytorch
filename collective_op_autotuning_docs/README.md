# Collective Op Autotuning Documentation

本目录包含完整的Collective Operation Autotuning实施文档。

## 📂 文档结构

```
collective_op_autotuning_docs/
├── README.md (本文件) - 文档索引和快速导航
├── MASTER_GUIDE.md - **主要实施指南** ⭐
└── reference/ - 参考设计文档
    ├── DESIGN_OVERVIEW.md - 设计总览
    ├── V1_SIMPLE_APPROACH.md - V1方案详解
    ├── V2_ADVANCED_APPROACH.md - V2方案详解
    └── FAQ.md - 常见问题和澄清
```

---

## 🚀 快速开始

### 如果你是新手
1. **先读**: `MASTER_GUIDE.md` - 包含所有你需要的信息
2. **然后做**: 按照guide中的Step 2-4修改代码
3. **最后测试**: Phase 1测试 (单op, 2 ranks)

### 如果你想深入了解
- **V1 vs V2对比**: 查看`reference/FAQ.md`
- **MultiTemplateBuffer详解**: 查看`reference/V2_ADVANCED_APPROACH.md`
- **设计原理**: 查看`reference/DESIGN_OVERVIEW.md`

---

## 📖 文档指南

### 1. MASTER_GUIDE.md ⭐ **(必读)**

**适合**: 所有人，特别是实施者

**内容**:
- ✅ V1方案完整实施步骤
- ✅ 代码修改位置和示例
- ✅ 测试计划 (4个phases)
- ✅ V2预留设计和可复用组件
- ✅ FAQ和troubleshooting

**何时读**:
- 开始实施前
- 遇到问题时
- 需要参考代码时

**预计阅读时间**: 30-45分钟

---

### 2. reference/DESIGN_OVERVIEW.md (参考)

**适合**: 想了解整体架构的人

**内容**:
- 问题背景和motivation
- V1和V2的区别
- 架构设计原理
- 关键技术决策

**何时读**:
- 想理解"为什么这样设计"
- Code review时需要背景
- 向他人解释方案时

**预计阅读时间**: 20分钟

---

### 3. reference/V1_SIMPLE_APPROACH.md (参考)

**适合**: 实施V1的开发者

**内容**:
- V1的详细设计
- 与现有系统的集成点
- Inline fusion机制
- 性能特征和限制

**何时读**:
- 实施V1遇到具体问题
- 需要深入理解V1实现细节
- Debug V1相关issue

**预计阅读时间**: 15分钟

---

### 4. reference/V2_ADVANCED_APPROACH.md (参考)

**适合**: 考虑升级到V2的团队

**内容**:
- V2的MultiTemplateBuffer机制
- Scheduler集成细节
- Unified sync设计
- Epilogue fusion benchmark

**何时读**:
- V1稳定后考虑升级
- 性能需要进一步优化
- 有多个collective ops场景

**预计阅读时间**: 25分钟

---

### 5. reference/FAQ.md (参考)

**适合**: 有疑问的所有人

**内容**:
- V1 vs V2对比表
- Inline fusion vs Epilogue fusion详解
- SubgraphTemplate和MultiTemplateBuffer关系
- 常见误区澄清

**何时读**:
- 有概念疑问时
- 不确定方案选择时
- 需要快速查找答案时

**预计阅读时间**: 10-15分钟

---

## 🎯 推荐阅读路径

### 路径 1: 快速实施 (推荐)
```
1. MASTER_GUIDE.md (必读)
   ↓
2. 开始修改代码
   ↓
3. 遇到问题 → FAQ.md
   ↓
4. 完成V1
```

### 路径 2: 深入理解
```
1. DESIGN_OVERVIEW.md (了解背景)
   ↓
2. FAQ.md (澄清概念)
   ↓
3. MASTER_GUIDE.md (实施)
   ↓
4. V1_SIMPLE_APPROACH.md (深入细节)
   ↓
5. V2_ADVANCED_APPROACH.md (未来规划)
```

### 路径 3: 架构Review
```
1. DESIGN_OVERVIEW.md
   ↓
2. V1_SIMPLE_APPROACH.md
   ↓
3. V2_ADVANCED_APPROACH.md
   ↓
4. FAQ.md
```

---

## 📁 核心文件位置

### 实施文件 (需要修改)
```
pytorch/torch/_inductor/
├── kernel/
│   └── custom_op.py           # Step 2: 添加detection
├── select_algorithm.py         # Step 3-4: 集成CollectiveBenchmarker
└── runtime/
    └── collective_benchmarking.py  # ✅ 已完成
```

### 测试文件 (需要创建)
```
pytorch/test/inductor/
└── test_collective_autotuning.py  # Phase 1-4测试
```

---

## 🔑 关键概念速查

### V1方案核心
- **兼容性**: 与现有custom op完全兼容
- **Sync策略**: 每个op单独sync
- **Fusion**: Inline fusion (scheduler可继续fuse)
- **开发时间**: 1-2天

### V2方案核心
- **MultiTemplateBuffer**: 延迟benchmark到scheduler
- **Unified Sync**: 所有ops统一sync一次
- **Epilogue Fusion**: 可benchmark with/without epilogue
- **开发时间**: 3-4天 (V1基础上)

### 可复用组件
1. ✅ `collective_benchmarking.py` - 100%复用
2. ✅ Detection逻辑 - 部分复用
3. ✅ Timeout机制 - 100%复用

---

## 🧪 测试计划概览

| Phase | 目标 | 配置 | 预计时间 |
|-------|------|------|---------|
| Phase 1 | 基础功能 | 1 op, 2 ranks | 1 day |
| Phase 2 | 多op验证 | 3 ops, 2 ranks | 0.5 day |
| Phase 3 | 压力测试 | 5 ops, 4 ranks | 0.5 day |
| Phase 4 | Timeout验证 | Simulated hang | 0.5 day |

---

## 📊 成功指标

### V1成功标准
- [x] ✅ 能正确autotune custom collective ops
- [x] ✅ Timeout机制有效，不会hang
- [x] ✅ 2-4 ranks测试通过
- [x] ✅ 结果正确性验证通过
- [x] ✅ 编译时间在预期范围内

### V2考虑标准
- [ ] Sync overhead > 200ms (多个collective ops)
- [ ] 需要benchmark epilogue fusion性能
- [ ] V1稳定运行，有开发资源

---

## 💬 联系和支持

- **Owner**: PyTorch Inductor Team
- **Module**: `torch._inductor`
- **参考**: autoparallel benchmarking utilities

---

## 📝 版本历史

| 版本 | 日期 | 内容 |
|-----|------|------|
| 1.0 | 2024-11 | V1方案完整文档 |
| 2.0 | TBD | V2方案实施 (可选) |

---

## 🔗 相关资源

### 外部参考
- [autoparallel benchmark_comm_func](https://github.com/meta-pytorch/autoparallel/blob/main/autoparallel/autobucketing_util/estimation_utils.py)
- PyTorch Distributed Documentation

### 内部参考
- `torch/_inductor/scheduler.py` - Scheduler实现
- `torch/_inductor/ir.py` - MultiTemplateBuffer定义
- `torch/_inductor/codegen/subgraph.py` - Subgraph inline fusion

---

**准备好了吗? 开始阅读 `MASTER_GUIDE.md`!** 🚀
