# Dynamic Config Generation Example

Shows how different input shapes automatically generate different numbers of configs:

```python
from torch._inductor.utils import get_k_splits
from torch._inductor.kernel.custom_op import CustomOpConfig

def generate_k_split_configs(shapes: dict[str, tuple]) -> list[CustomOpConfig]:
    """Generate k_split configs based on matrix dimensions."""
    m, k = shapes["a"][-2:]
    _, n = shapes["b"][-2:]

    k_splits = get_k_splits(m, n, k)
    return [CustomOpConfig(k_splits=k) for k in k_splits]

# Example: Different shapes → Different number of configs

# Shape 1: Small K dimension
shapes_small = {"a": (256, 4096), "b": (4096, 1024)}
configs_small = generate_k_split_configs(shapes_small)
print(f"Small K=4096: {len(configs_small)} configs")
# Output: Small K=4096: 5 configs
# k_splits = [2, 4, 8, 16, 32]

# Shape 2: Large K dimension
shapes_large = {"a": (256, 65536), "b": (65536, 1024)}
configs_large = generate_k_split_configs(shapes_large)
print(f"Large K=65536: {len(configs_large)} configs")
# Output: Large K=65536: 9 configs
# k_splits = [2, 4, 8, 16, 32, 64, 128, 256, 512]
```

**Key Point:** Larger K dimensions have more valid divisors, so `get_k_splits` generates more candidate configs to explore during autotuning, leading to better optimization for that specific shape.
