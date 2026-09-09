# Shared CuteDSL machinery for native ops: the trait library, launch glue, the
# hardware-capability struct and the shape-keyed launch-plan memo, reused by every op family
# so a new one does not re-derive the host-overhead-minimizing launch path. The DSL-agnostic
# cond primitives live in torch/_native/utils/ instead, since they import only torch.
