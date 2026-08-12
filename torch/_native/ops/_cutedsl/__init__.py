# Shared CuteDSL machinery for native ops: the trait library (traits), launch glue
# (launch), the hardware-capability struct (hw_caps), and the shape-keyed launch-plan
# memo (plan_cache). Reused by the reduction ops under ../reductions/ and by future
# pointwise ops so a new op family does not re-derive the host-overhead-minimizing
# launch path or the trait protocol. (The DSL-agnostic per-tensor cond primitives --
# is_cow / is_traced / device_ok / on_current_device -- live in torch/_native/utils/
# instead, since they import only torch and run in the cond on every dispatch.)
#
# Importing this package pulls in `cutlass` (traits/launch are @cute.jit code), so it
# must stay off the `import torch` path; the heavy work happens only when a kernel is
# actually compiled/launched.

from . import hw_caps, launch, plan_cache, traits


__all__ = ["hw_caps", "launch", "plan_cache", "traits"]
