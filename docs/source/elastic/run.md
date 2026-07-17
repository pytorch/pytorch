(launcher-api)=

# torchrun (Elastic Launch)

```{eval-rst}
.. automodule:: torch.distributed.run
.. currentmodule:: torch.distributed.run

.. autosummary::
   :toctree: generated
   :nosignatures:

   config_from_args
   determine_local_world_size
   main
   parse_args
   parse_min_max_nnodes
   run
   run_script_path
```

## NUMA Binding

NUMA (Non-Uniform Memory Access) is a memory architecture in which different CPU memory nodes have different access latencies. On NUMA-enabled multi-GPU servers, `--numa-binding` can bind worker processes to CPU cores that are closest to their assigned GPUs to improve locality and potentially improve performance.

This feature is intended for NUMA-enabled multi-GPU servers and may not provide a benefit on typical desktop or laptop systems.

Example:

```bash
torchrun --numa-binding=node --nproc-per-node=8 train.py
```

For more information about the available NUMA binding modes and configuration options, see the {doc}`NUMA Binding <numa>` documentation.