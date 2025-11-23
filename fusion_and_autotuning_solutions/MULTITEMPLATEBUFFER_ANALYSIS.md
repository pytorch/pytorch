# MultiTemplateBuffer Architecture Analysis

**Date**: 2025-11-07
**Purpose**: Understand MultiTemplateBuffer architecture for potential Collective Op Autotuning V2 implementation

---

## Executive Summary

**Key Finding**: `tuned_mm` in `/torch/_inductor/kernel/mm.py` **DOES use MultiTemplateBuffer**, but only when `config.benchmark_epilogue_fusion=True` (default: `False`). This is the **deferred autotuning** path that delays kernel selection until fusion opportunities are known during the scheduling phase.

**Current Collective Op Implementation**: Uses **immediate autotuning** (Path B) without MultiTemplateBuffer support.

**V2 Recommendation**: Implement MultiTemplateBuffer support for collective ops to enable:
1. **Epilogue fusion benchmarking** (e.g., all_reduce + elementwise ops)
2. **Multi-kernel dispatch** based on message size hints
3. **Deferred selection** until distributed context is fully known

---

## Table of Contents

1. [MultiTemplateBuffer Overview](#1-multitemplatebuffer-overview)
2. [tuned_mm Architecture](#2-tuned_mm-architecture)
3. [Two Autotuning Paths](#3-two-autotuning-paths)
4. [Current Collective Op Implementation](#4-current-collective-op-implementation)
5. [V2 Design Proposal](#5-v2-design-proposal)
6. [Implementation Roadmap](#6-implementation-roadmap)

---

## 1. MultiTemplateBuffer Overview

### Location
- **File**: `/torch/_inductor/ir.py`
- **Lines**: 5269-5357
- **Parent Class**: `TritonTemplateBuffer`

### Purpose

MultiTemplateBuffer is a **deferred kernel selection mechanism** that:

1. **Delays benchmarking** until full context is available (fusion opportunities, tensor shapes)
2. **Enables epilogue fusion** by benchmarking choices WITH fused operations
3. **Supports multi-kernel dispatch** based on size hints for different tensor dimensions
4. **Provides lazy evaluation** to avoid redundant benchmarking

### Core Mechanism

```python
class MultiTemplateBuffer(TritonTemplateBuffer):
    """
    Represents a Buffer with multiple backing implementation choices.

    During scheduling, if there is a potential epilogue, we will benchmark
    each choice WITH the epilogue to determine the best implementation.
    Otherwise, the fastest base choice will be chosen.
    """

    def __init__(
        self,
        layout: Layout,
        inputs: Sequence[IRNode],
        choice_timings_fn: Callable[[Optional[int]], dict[ChoiceCaller, float]],
        unfiltered_choices: list[ChoiceCaller],
        allowed_prologue_inps: OrderedSet[str],
    ):
        # KEY: choice_timings_fn is a LAZY callable
        self._choice_timings_fn = choice_timings_fn
        self._choice_timings: dict = {}  # Cache keyed by hint_override
        self._make_kernel_renders: dict = {}  # Final choices per hint
```

### Key Methods

| Method | Purpose | When Called |
|--------|---------|-------------|
| `choice_timings(hint_override=None)` | Lazy evaluation of benchmarking | During scheduling phase |
| `get_min_choice(hint_override=None)` | Returns fastest choice for hint | Finalization without fusion |
| `swap_as_triton_caller(caller)` | **Context manager** for temporary swap | Fusion benchmarking |
| `finalize_as_triton_caller(caller)` | Permanent selection (single choice) | No fusion, direct selection |
| `finalize_as_triton_callers(callers)` | Permanent selection (multi-kernel) | Multi-size-hint dispatch |

### Critical Design Pattern: Lazy Evaluation

```python
# In AlgorithmSelectorCache.__call__() (select_algorithm.py:2927-2971)

def get_timings(hint_override=None):
    """LAZY callable - not executed until scheduler needs it"""
    filtered_choices = [c for c in choices
                       if not hasattr(c, 'hint_override')
                       or c.hint_override == hint_override]

    # Benchmark NOW (deferred until called)
    timings = do_autotuning(filtered_choices, precompile_fn,
                            hint_override=hint_override)

    # Filter slow extern kernels
    return post_filter_timings(timings)

# CREATE MultiTemplateBuffer with lazy benchmarking function
return TensorBox.create(
    MultiTemplateBuffer(
        layout=layout,
        inputs=input_nodes,
        choice_timings_fn=get_timings,  # NOT called yet!
        unfiltered_choices=choices,
        allowed_prologue_inps=OrderedSet(),
    )
)
```

**Key Insight**: Benchmarking is **deferred** until `choice_timings()` is called during scheduling, when fusion context is known.

---

## 2. tuned_mm Architecture

### Entry Point
- **File**: `/torch/_inductor/kernel/mm.py`
- **Function**: `tuned_mm(mat1, mat2, out_dtype=None, *, layout=None)`
- **Lines**: 1100-1329
- **Decorator**: `@register_lowering(aten.mm, type_promotion_kind=None)`

### Complete Flow

```
┌────────────────────────────────────────────────────────┐
│  tuned_mm(mat1, mat2)                                  │
│  ────────────────────────────────────────────────────  │
│                                                         │
│  Phase 1: Extract dimensions (M, N, K)                │
│  Phase 2: Collect choices                             │
│    ├─ aten_mm (cuBLAS)                                │
│    ├─ mm_template (Triton, multiple configs)          │
│    ├─ persistent_tma_mm_template (Triton TMA)         │
│    ├─ CUTLASS3xGemmTemplate                           │
│    ├─ decompose_k_subgraph_template                   │
│    └─ mm_contiguous_subgraph_template                 │
│                                                         │
│  Phase 3: AutoHeuristic filtering (optional)          │
│    └─ ML model ranks choices, keeps top-k             │
│                                                         │
│  Phase 4: Remote cache lookup (async)                 │
│                                                         │
│  Phase 5: Call autotuning                             │
└──────────────┬─────────────────────────────────────────┘
               │
               ▼
┌────────────────────────────────────────────────────────┐
│  autotune_select_algorithm(                            │
│      name="mm",                                        │
│      choices=choices,                                  │
│      input_nodes=[mat1, mat2],                         │
│      layout=layout,                                    │
│      return_multi_template=config.benchmark_epilogue_  │
│                            fusion  # KEY FLAG!         │
│  )                                                      │
└──────────────┬─────────────────────────────────────────┘
               │
               ▼
   ┌───────────────────────────────────────┐
   │  AlgorithmSelectorCache.__call__()    │
   │                                        │
   │  Decision Point:                      │
   │  if return_multi_template=True:       │
   │    → Path A: MultiTemplateBuffer      │
   │  else:                                 │
   │    → Path B: Direct Selection         │
   └───────────────────────────────────────┘
```

### Key Configuration Flag

```python
# torch/_inductor/config.py (line ~500)
benchmark_epilogue_fusion: bool = False  # Default: DISABLED
```

**Impact**:
- `False` (default): Uses **Path B** (immediate autotuning)
- `True`: Uses **Path A** (deferred with MultiTemplateBuffer)

---

## 3. Two Autotuning Paths

### Path A: MultiTemplateBuffer (Deferred Selection)

**Triggered when**: `return_multi_template=True` (i.e., `config.benchmark_epilogue_fusion=True`)

**Location**: `/torch/_inductor/select_algorithm.py`, lines 2927-2971

```python
def __call__(self, name, choices, input_nodes, layout, return_multi_template=False, ...):
    if return_multi_template:
        # DEFERRED PATH

        def get_timings(hint_override=None):
            """Lazy benchmarking callable"""
            filtered_choices = [c for c in choices
                               if not hasattr(c, 'hint_override')
                               or c.hint_override == hint_override]

            # Benchmark NOW (when called, not during creation)
            timings = do_autotuning(filtered_choices, precompile_fn,
                                    hint_override=hint_override)

            # Post-filter: keep tritons faster than slowest extern
            return post_filter_timings(timings)

        # RETURN MultiTemplateBuffer (no benchmarking yet!)
        return TensorBox.create(
            MultiTemplateBuffer(
                layout=layout,
                inputs=input_nodes,
                choice_timings_fn=get_timings,  # LAZY!
                unfiltered_choices=choices,
                allowed_prologue_inps=OrderedSet(),
            )
        )
```

**Characteristics**:
1. ✅ **Lazy benchmarking**: Deferred until scheduling phase
2. ✅ **Fusion-aware**: Can benchmark with epilogue operations
3. ✅ **Multi-kernel support**: Can finalize different choices per size hint
4. ⚠️ **Higher complexity**: Requires scheduler integration

**When finalized**:
- **Location**: `/torch/_inductor/scheduler.py`, lines 3412-3489
- **Function**: `finalize_multi_template_buffers()`
- **Trigger**: During scheduling, when fusion opportunities are identified

---

### Path B: Direct Selection (Immediate)

**Triggered when**: `return_multi_template=False` (default for most ops)

**Location**: `/torch/_inductor/select_algorithm.py`, lines 2973-3007

```python
def __call__(self, name, choices, input_nodes, layout, return_multi_template=False, ...):
    if not return_multi_template:
        # IMMEDIATE PATH

        # Benchmark ALL choices NOW
        timings = do_autotuning(choices, precompile_fn)

        # Select best choice immediately
        choice = min(timings, key=timings.__getitem__)

        # Return output node directly
        return choice.output_node()
```

**Characteristics**:
1. ✅ **Simple**: Immediate benchmarking and selection
2. ✅ **Predictable**: No deferred evaluation
3. ❌ **Fusion-unaware**: Cannot benchmark with epilogue
4. ❌ **No multi-kernel**: Single choice for all sizes

**Used by**: Most operations (matmul with default config, convolution, etc.)

---

### Comparison Table

| Aspect | Path A (MultiTemplate) | Path B (Direct) |
|--------|------------------------|-----------------|
| **Benchmarking** | Deferred (lazy) | Immediate |
| **Fusion Support** | ✅ YES | ❌ NO |
| **Multi-kernel Dispatch** | ✅ YES | ❌ NO |
| **Complexity** | High | Low |
| **Scheduler Integration** | Required | Not required |
| **Used by tuned_mm** | `config.benchmark_epilogue_fusion=True` | Default (`False`) |
| **Used by collective ops** | ❌ **NOT YET** | ✅ **CURRENT** |

---

## 4. Current Collective Op Implementation

### Our Current Implementation (V1)

**Location**: `/torch/_inductor/kernel/custom_op.py`, lines 324-370

```python
def call_function(self, target, args, kwargs):
    # ... (op detection code)

    # Call autotune_select_algorithm WITHOUT return_multi_template
    return autotune_select_algorithm(
        f"custom_op_{op_overload}",
        choices,
        input_nodes,
        layout=layout,
        process_group=process_group,
        is_collective=True,  # NEW FLAG
    )
    # → Uses Path B (immediate selection)
```

**Flow**:
```
custom_op.py:call_function()
    │
    ├─ Detect collective op: is_collective_op(op_overload)
    ├─ Extract process_group from kwargs
    │
    └─► autotune_select_algorithm(
            choices=choices,
            is_collective=True,
            process_group=process_group,
            return_multi_template=False  # DEFAULT → Path B
        )
            │
            └─► AlgorithmSelectorCache.__call__()
                    │
                    └─► do_autotuning()
                            │
                            ├─► make_benchmark_fn()
                            │   └─► CollectiveBenchmarker.benchmark()
                            │       (collective_benchmarking.py)
                            │
                            └─► Return best choice immediately
```

### Limitations of Current V1

1. ❌ **No epilogue fusion**: Cannot benchmark all_reduce + elementwise ops together
2. ❌ **No multi-kernel dispatch**: Single choice for all message sizes
3. ❌ **No deferred selection**: Benchmarking happens immediately without distributed context

**Example missed optimization**:
```python
# User code
result = torch.ops._c10d_functional.all_reduce_(x, "sum", "default")
result = result.relu()  # Epilogue operation

# Current V1: Benchmarks all_reduce ALONE
# Desired V2: Benchmark all_reduce + relu FUSED
```

---

## 5. V2 Design Proposal

### Goal

Enable MultiTemplateBuffer support for collective operations to:
1. Benchmark collective ops WITH epilogue fusion
2. Support multi-kernel dispatch based on message size
3. Defer selection until distributed context is fully known

### Architecture Changes

#### Change 1: Add `return_multi_template` Support

**File**: `/torch/_inductor/kernel/custom_op.py`

```python
def call_function(self, target, args, kwargs):
    # ... (existing detection code)

    if is_collective:
        # NEW: Enable MultiTemplateBuffer for collective ops
        return autotune_select_algorithm(
            f"custom_op_{op_overload}",
            choices,
            input_nodes,
            layout=layout,
            process_group=process_group,
            is_collective=True,
            return_multi_template=config.benchmark_collective_epilogue_fusion,  # NEW!
        )
```

#### Change 2: Add Configuration Flag

**File**: `/torch/_inductor/config.py`

```python
# Enable deferred collective op selection with epilogue fusion
benchmark_collective_epilogue_fusion: bool = False

# Message size hints for multi-kernel collective dispatch
# E.g., [1024, 65536, 1048576] for 1KB, 64KB, 1MB
collective_multi_kernel_hints: list[int] = []
```

#### Change 3: Scheduler Integration

**File**: `/torch/_inductor/scheduler.py`

Extend `finalize_multi_template_buffers()` to handle collective ops:

```python
def finalize_multi_template_buffers(self, nodes):
    """Finalize MultiTemplateBuffer choices with fusion context"""

    for node in nodes:
        multi_node = node.node

        # Check if this is a collective op
        is_collective = getattr(multi_node, 'is_collective', False)

        if is_collective:
            # NEW: Collective-specific finalization logic
            self._finalize_collective_multi_template(node, multi_node)
        else:
            # Existing matmul/conv finalization
            self._finalize_compute_multi_template(node, multi_node)
```

#### Change 4: Collective-Aware Benchmarking in Scheduler

```python
def _finalize_collective_multi_template(self, node, multi_node):
    """
    Finalize collective op MultiTemplateBuffer with epilogue fusion.

    1. Detect epilogue operations (e.g., elementwise ops after all_reduce)
    2. For each choice, benchmark with epilogue fused
    3. Select best choice per message size hint
    """

    # 1. Detect fusion opportunities
    epilogue_ops = self._detect_collective_epilogue(node)

    if not epilogue_ops and not config.collective_multi_kernel_hints:
        # No fusion, no multi-kernel → simple selection
        choice, timing = multi_node.get_min_choice()
        multi_node.finalize_as_triton_caller(choice)
        return

    # 2. Benchmark with fusion (if epilogue exists)
    if epilogue_ops:
        timings = {}
        for choice in multi_node.unfiltered_choices:
            with multi_node.swap_as_triton_caller(choice):
                # Compile with epilogue fused
                mod_fused = self._compile_with_epilogue(node, epilogue_ops)
                # Benchmark (collective-aware)
                ms_fused = self._benchmark_collective_fusion(mod_fused, choice)
                timings[choice] = ms_fused
    else:
        # No epilogue, use base timings
        timings = multi_node.choice_timings()

    # 3. Multi-kernel dispatch based on message size hints
    if config.collective_multi_kernel_hints:
        callers = {}
        callers[None] = min(timings, key=timings.__getitem__)  # Default

        for hint in config.collective_multi_kernel_hints:
            # Benchmark with this message size hint
            hint_timings = multi_node.choice_timings(hint_override=hint)
            best_choice = min(hint_timings, key=hint_timings.__getitem__)
            callers[hint] = best_choice

        multi_node.finalize_as_triton_callers(callers)
    else:
        # Single choice
        best_choice = min(timings, key=timings.__getitem__)
        multi_node.finalize_as_triton_caller(best_choice)
```

#### Change 5: Multi-Kernel Runtime Dispatch

**File**: `/torch/_inductor/codegen/multi_kernel.py`

Extend multi-kernel dispatcher to handle message size:

```python
def generate_multi_kernel_collective(choices_per_hint, input_nodes):
    """
    Generate runtime dispatcher for collective ops based on message size.

    Example generated code:

    if message_size < 65536:
        return collective_kernel_small(tensor, ...)
    elif message_size < 1048576:
        return collective_kernel_medium(tensor, ...)
    else:
        return collective_kernel_large(tensor, ...)
    """

    # Generate dispatch logic based on tensor numel()
    dispatch_code = []
    sorted_hints = sorted(choices_per_hint.keys())

    for hint in sorted_hints:
        kernel = choices_per_hint[hint]
        dispatch_code.append(f"""
if message_size < {hint}:
    return {kernel.render_call()}
""")

    # Default fallback
    dispatch_code.append(f"""
else:
    return {choices_per_hint[None].render_call()}
""")

    return "\n".join(dispatch_code)
```

---

## 6. Implementation Roadmap

### Phase 1: Foundation (Week 1)
1. ✅ **DONE**: Implement `CollectiveBenchmarker` in `collective_benchmarking.py`
2. ✅ **DONE**: Integrate with `select_algorithm.py` (immediate path)
3. ✅ **DONE**: Write tests (`test_collective_autotuning.py`)

### Phase 2: MultiTemplateBuffer Integration (Week 2)
1. Add `benchmark_collective_epilogue_fusion` config flag
2. Modify `custom_op.py` to pass `return_multi_template=True`
3. Test basic MultiTemplateBuffer creation for collective ops

### Phase 3: Scheduler Integration (Week 3)
1. Extend `finalize_multi_template_buffers()` for collective ops
2. Implement `_finalize_collective_multi_template()`
3. Add epilogue detection for collective ops
4. Test fusion benchmarking (all_reduce + relu)

### Phase 4: Multi-Kernel Dispatch (Week 4)
1. Add `collective_multi_kernel_hints` config
2. Implement size-based hint override in benchmarking
3. Extend multi-kernel codegen for collective dispatch
4. Test runtime dispatch with different message sizes

### Phase 5: Testing & Validation (Week 5)
1. Comprehensive test suite for V2
2. Performance benchmarks (fusion speedup, multi-kernel efficiency)
3. Integration tests with real vLLM workloads
4. Documentation and examples

---

## Key Insights & Recommendations

### 1. **tuned_mm DOES use MultiTemplateBuffer**
- But only when `config.benchmark_epilogue_fusion=True` (default: `False`)
- Most production workloads use immediate selection (Path B)

### 2. **Collective ops should follow the same pattern**
- Enable MultiTemplateBuffer for epilogue fusion
- Support multi-kernel dispatch for different message sizes
- Defer selection until distributed context is known

### 3. **Start with Simple Cases**
- Begin with epilogue fusion (all_reduce + elementwise)
- Then add multi-kernel dispatch
- Finally optimize for complex fusion patterns

### 4. **Reuse Existing Infrastructure**
- `MultiTemplateBuffer` class (already exists)
- `finalize_multi_template_buffers()` in scheduler
- `multi_kernel.py` for runtime dispatch
- Just need collective-specific extensions

### 5. **Configuration Flexibility**
- Default to immediate path (backward compatible)
- Opt-in to deferred path via config flags
- Allow incremental adoption

---

## Conclusion

The **MultiTemplateBuffer architecture** provides a powerful framework for deferred kernel selection with fusion support. Our **V1 collective op autotuning** implementation successfully uses the immediate path, but **V2 should leverage MultiTemplateBuffer** to enable:

1. ✅ **Epilogue fusion** (benchmark collective + elementwise ops together)
2. ✅ **Multi-kernel dispatch** (different kernels for different message sizes)
3. ✅ **Deferred selection** (optimize with full distributed context)

The infrastructure already exists in PyTorch Inductor for matmul operations. We just need to extend it for collective operations with distributed-specific considerations.

**Next Steps**:
1. Start Phase 2 implementation
2. Add config flags
3. Modify `custom_op.py` to support `return_multi_template`
4. Test basic MultiTemplateBuffer creation for all_reduce

---

## References

### Key Files
- `/torch/_inductor/ir.py` (MultiTemplateBuffer definition)
- `/torch/_inductor/select_algorithm.py` (autotuning engine)
- `/torch/_inductor/kernel/mm.py` (tuned_mm implementation)
- `/torch/_inductor/scheduler.py` (finalization logic)
- `/torch/_inductor/codegen/multi_kernel.py` (runtime dispatch)
- `/torch/_inductor/runtime/collective_benchmarking.py` (our V1 implementation)

### Related Configurations
- `config.benchmark_epilogue_fusion` (enables MultiTemplateBuffer for matmul)
- `config.multi_kernel_hints` (size-based dispatch for matmul)
- `config.benchmark_collective_epilogue_fusion` (proposed for V2)
- `config.collective_multi_kernel_hints` (proposed for V2)

---

**Document Version**: 1.0
**Last Updated**: 2025-11-07
**Author**: Collective Op Autotuning Team
