# AutoDynamic Mode for PT2 Benchmarks

## Overview

The AutoDynamic benchmark mode is designed to better simulate real-world dynamic shape workloads by actually running models with varying input shapes, rather than using symbolic shape tracking with `mark_dynamic`.

This approach has several advantages:
1. It tests the actual performance with varying shapes, not just the ability to compile with dynamic shapes
2. It measures how well TorchDynamo/Inductor handle recompilations or adaptations to shape changes
3. It identifies areas where dynamic shape support needs improvement
4. It provides a more realistic measurement of speedup in dynamic workloads

## Usage

To run benchmarks with AutoDynamic mode:

```bash
python benchmarks/dynamo/torchbench.py --only resnet50 --backend inductor --performance --inference --auto-dynamic
```

Or for CI use:
```bash
DASHBOARD_TAG="auto-dynamic-true" python .ci/pytorch/test.sh
```

## How It Works

1. Instead of marking tensors as dynamic with `torch._dynamo.mark_dynamic()`, AutoDynamic mode:
   - Takes the example inputs for a model
   - Generates multiple variations of those inputs with slightly different shapes
   - Runs benchmarks with these varied inputs
   - Reports aggregate statistics across all variations

2. Specifically, the implementation:
   - Identifies tensor dimensions that appear to be batch dimensions
   - Creates variations by adjusting these dimensions by a small percentage (±20% by default)
   - Runs both eager and compiled models with the same input variations
   - Reports median speedup as well as min/max/std to show performance variance

3. Results include:
   - Overall speedup (median across variations)
   - Min/max speedup to show the range
   - Standard deviation to show stability
   - Raw latency measurements for each variation

## Compared to `--dynamic-batch-only` and `--dynamic-shapes`

AutoDynamic differs from existing dynamic shape modes:

- `--dynamic-batch-only`: Marks batch dimensions as dynamic but doesn't actually vary shapes
- `--dynamic-shapes`: Disables the assumption that shapes are static by default, but still doesn't vary actual shapes
- `--auto-dynamic`: Actually runs with different input shapes to test real dynamic behavior

## Benefits

1. **More realistic performance**: Measures actual performance with varying shapes, not just compilation capability
2. **Tests recompilation**: Shows how efficiently the system handles shape changes
3. **Identifies real issues**: Problems that only appear with actual shape changes are detected
4. **Variance reporting**: Shows not just average speedup but stability across shape variations

## Limitations

1. Requires the model to handle varying input shapes correctly
2. Currently focuses on varying the batch dimension
3. May trigger more recompilations than strictly necessary in production