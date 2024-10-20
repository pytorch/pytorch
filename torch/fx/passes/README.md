## FX Pass Infrastructure
This folder contains the pass infrastructure and passes for transforming fx.Graph.


## Code Structure

* [infra](infra) - Common infrastructure, such as PassManager, PassBase
    * [partitioner.py](infra/partitioner.py) - backend agnostic FX graph partitioner
* [utils](utils) - Utility classes and functions
    * [common.py](utils/common.py) - common utility functions
    * [fuser_utils.py](utils/fuser_utils.py) - utility functions for fusing list of nodes into a single node
* [dialect](dialect) - dialect specific passes
    * [common](dialect/common) - common passes that can be shared by all dialects
        * [cse_pass.py](dialect/common/cse_pass.py) - a CSE pass
    * [aten](dialect/aten) - aten dialect specific passes
    * [prims](dialect/prims) - prim dialect specific passes
* [backends](backends) - Backend specific passes
    * [nvfuser](backends/nvfuser) - passes for nvfuser
        * [operator_support.py](backends/nvfuser/operator_support.py) - nvFuser supported ops
* [conversion](conversion) - Conversion passes between dialects
