## FX Pass Infrastructure
This folder contains the pass infarstructure and passes for transforming FX.graph.


## Code Structure

* [infra](infra) - Common infrastructure, such as PassManager, PassBase
    * [partitioner.py](infra/partitioner.py) - backend agnostic FX graph partitioner
* [utils](utils) - Utility classes and functions
    * [fuser_utis.py](fuser_utils.py) - utility function for fusing node list into a single FX node
* [dialect](dialect) - dialect specific passes
    * [common](dialect/common) - common passes that can be shared by all dialects
    * [aten](dialect/aten) - aten dialect specific passes
    * [prims](dialect/prims) - prim dialect specific passes
    * [exir](dialect/exir) - exir dialect specific passes
    * [acc](dialect/acc) - acc dialect specific passes
* [backends](backends) - Backend specific passes
    * [nvfuser](backends/nvfuser) - passes for nvfuser
        * [operator_support.py](backends/nvfuser/operator_support.py) - nvFuser supported ops
* [conversion](conversion) - Conversion passes between dialects
