# torch.nn.utils.summary

```{eval-rst}
.. automodule:: torch.nn.utils.summary
```

This module provides a utility for inspecting `nn.Module` architectures,
displaying layer names, output shapes, and parameter counts.

## Quick Start

```python
import torch.nn as nn
from torch.nn.utils import summary

model = nn.Sequential(
    nn.Conv2d(3, 64, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.Flatten(),
    nn.Linear(64 * 32 * 32, 10),
)

# Print summary for 32x32 RGB input
summary(model, input_size=(3, 32, 32))
```

## API Reference

```{eval-rst}
.. autofunction:: torch.nn.utils.summary.summary
.. autoclass:: torch.nn.utils.summary.ModelSummary
   :members:
.. autoclass:: torch.nn.utils.summary.LayerInfo
   :members:
```

