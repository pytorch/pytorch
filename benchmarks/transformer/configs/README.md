# Transformer Benchmark Configurations

This directory contains YAML configuration files for running comprehensive attention benchmarks using `score_mod.py`.

## Available Configurations

### `config_basic.yaml`
- **Purpose**: Quick benchmark with basic attention patterns
- **Backends**: SDPA (math, efficient)
- **Patterns**: noop, causal, alibi, sliding_window
- **Use case**: Fast validation and basic performance testing

### `config_comprehensive.yaml`
- **Purpose**: Full benchmark suite comparing all backends
- **Backends**: SDPA + Flash Attention + FlexAttention (efficient, math, cudnn, fav2, fav3)
- **Patterns**: All available patterns including rel, head_bias, prefix_lm, softcap
- **Use case**: Complete performance analysis and comparison

### `config_decoding.yaml`
- **Purpose**: Decoding-specific benchmarks (query length = 1)
- **Backends**: efficient, fav2, fakv (decoding-optimized)
- **Patterns**: causal, alibi, sliding_window, softcap
- **Use case**: Inference performance analysis

### `config_memory_bound.yaml`
- **Purpose**: Memory efficiency testing with large sequences
- **Backends**: efficient, fav2
- **Patterns**: causal, sliding_window, document_mask
- **Use case**: Memory-constrained scenarios

### `config_sdpa_only.yaml`
- **Purpose**: SDPA backend comparison only
- **Backends**: math, efficient, cudnn
- **Patterns**: causal, alibi
- **Use case**: Pure PyTorch SDPA performance analysis

### `config_dynamic.yaml`
- **Purpose**: Dynamic shapes testing for FlexAttention
- **Backends**: efficient, fav2 (dynamic shape compatible)
- **Patterns**: causal, alibi, sliding_window
- **Use case**: Testing dynamic shape performance and compilation

## Compilation Optimization

All configs now include `max_autotune: true` which enables:
- **torch.compile** with `mode="max-autotune-no-cudagraphs"`
- **Optimal kernel selection** for FlexAttention
- **Maximum performance** through aggressive optimization

### Dynamic vs Static Shapes
- **Static shapes** (`dynamic: false`) - Consistent compilation, faster startup
- **Dynamic shapes** (`dynamic: true`) - Flexible input sizes, slower startup but more flexible

## Usage

### Using YAML Configs (Recommended)
```bash
# Run basic benchmark
python run_benchmark.py --config configs/config_basic.yaml

# Run comprehensive benchmark
python run_benchmark.py --config configs/config_comprehensive.yaml

# Run decoding benchmark
python run_benchmark.py --config configs/config_decoding.yaml

# Run memory-bound benchmark
python run_benchmark.py --config configs/config_memory_bound.yaml

# Run SDPA-only benchmark
python run_benchmark.py --config configs/config_sdpa_only.yaml

# Run Flash Attention benchmark
python run_benchmark.py --config configs/config_flash_attention.yaml

# Dry run to see the command that would be executed
python run_benchmark.py --config configs/config_basic.yaml --dry-run
```

### Direct CLI Usage (Simplified)
```bash
# Basic benchmark - all values in single arguments
python score_mod.py -b 2 4 -s 1024 2048 -mods causal alibi --backend efficient math --throughput

# Comprehensive benchmark
python score_mod.py -b 1 2 4 8 -s 512 1024 2048 -mods causal alibi sliding_window --backend efficient math cudnn fav2 --throughput --calculate-bwd

# Decoding benchmark
python score_mod.py --decoding -b 1 4 8 -s 1024 2048 --backend efficient fav2 fakv --throughput

# SDPA-only comparison
python score_mod.py -b 1 8 16 -s 128 256 512 1024 -mods causal alibi --backend math efficient cudnn --throughput --calculate-bwd
```

## Configuration Parameters

- **dynamic**: Enable dynamic shapes for FlexAttention
- **calculate_bwd**: Include backward pass timing
- **dtype**: Data type (bfloat16, float16, float32)
- **b**: Batch sizes to test
- **nh**: Query and key-value head configurations
- **s**: Sequence lengths to test
- **d**: Head dimensions to test
- **mods**: Attention patterns to benchmark
- **backend**: Backends to compare
- **max_autotune**: Enable max-autotune optimization
- **decoding**: Enable decoding mode (query length = 1)
- **kv_size**: KV cache sizes in MiB (overrides batch size)
- **throughput**: Calculate memory bandwidth and TFLOPS
- **save_path**: CSV file to save results

## Backend Options

- **SDPA**: `math`, `efficient`, `cudnn`
- **Flash Attention**: `fav2`, `fav3`, `fakv`
- **FlexAttention**: Always included as "compiled" backend

## Attention Patterns

- **noop**: No masking
- **causal**: Causal attention
- **rel**: Relative position bias
- **head_bias**: Head-specific bias
- **alibi**: ALiBi attention
- **sliding_window**: Sliding window attention
- **document_mask**: Document-level masking
- **prefix_lm**: Prefix language modeling
- **softcap**: Soft capping
