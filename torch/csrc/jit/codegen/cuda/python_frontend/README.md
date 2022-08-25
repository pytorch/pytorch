# nvFuser Python Frontend

This frontend allows for a user to describe the set of operations for nvFuser to fuse via 1 or more kernels.  This frontend is intended to be an integration point with Pytorch or standalone applications.

# Usage

## Example

```
import torch
from torch._C._nvfuser import FusionManager, FusionDefinition, DataType

fm = FusionManager.get()
with FusionDefinition(fm) as fd :
    t0 = fd.define_tensor(symbolic_sizes=[-1, 1, -1], contiguous=[True, True, True], dtype=DataType.Float)
    t1 = fd.define_tensor(3)
    c0 = fd.define_constant(3.0)
  
    t2 = fd.ops.add(t0, t1)
    t3 = fd.ops.mul(t2, c0)
    t4 = fd.ops.sum(t3, [-1], False, DataType.Float)
  
    fd.add_output(t4)
    
input1 = torch.ones(2, 1, 8, device='cuda')
input2 = torch.ones(2, 4, 8, device='cuda')

nvf_out1 = fm.execute([input1, input2])[0]
```

## Components

### `FusionManager` - Executes and Caches Defined Fusions

### `FusionDefiniton` Context Manager - Interface for Defining Fusions

# Debug Information
