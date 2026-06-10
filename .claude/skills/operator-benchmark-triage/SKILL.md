---
name: operator-benchmark-triage
description: Analyze PR changes to recommend which operator benchmarks should run
---

# Operator Benchmark Triage

Analyze a PyTorch PR to determine which operator benchmarks should be run.

## Your Task

1. **Get the PR diff** using GitHub MCP tools
2. **Check which files changed** - focus on:
   - `aten/src/ATen/**` (operator implementations)
   - `torch/**` (Python frontend)
   - `c10/**` (core library)
3. **Map to benchmarks**:
   - Matmul/linear algebra changes → matmul, mm, bmm, addmm benchmarks
   - Conv changes → conv benchmarks
   - Normalization → batchnorm, layernorm, groupnorm benchmarks
   - Activations → activation, gelu, relu benchmarks
   - Quantization → all q*_test.py benchmarks
   - Dispatcher/autograd → run all "short" benchmarks (broad impact)
4. **Write JSON output** to `/tmp/benchmark_analysis.json`:
```json
{
  "scope": "targeted|short|long|none",
  "benchmarks": ["matmul", "conv"],
  "risk": "high|medium|low",
  "reason": "Brief explanation"
}
```

## Scope Guidelines
- `none`: Docs/tests only
- `targeted`: Specific operators (list 3-5 benchmarks)
- `short`: Broad impact (dispatcher, autograd, core changes)
- `long`: Major refactor

## That's It!
Keep it simple. Claude knows PyTorch operators - trust the analysis.
