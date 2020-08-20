import torch.distributed.algorithms.ddp_comm_hooks.default_hooks as default
import torch.distributed.algorithms.ddp_comm_hooks.quantization_hooks as quantization

# hook_registry wraps the hooks of ``torch.distributed.algorithms.ddp_comm_hooks``
# library inside a dictionary. After importing hook_registry user can simply register
# a hook to a ``ddp_model``` using ``hook_registry(comm_hook)(ddp_model, process_group)``.
hook_registry = {
    "quantize per tensor": lambda model, pg: model._register_comm_hook(
        pg, quantization.quantization_pertensor_hook
    ),
    "quantize per channel": lambda model, pg: model._register_comm_hook(
        pg, quantization.quantization_perchannel_hook
    ),
    "allreduce": lambda model, pg: model._register_comm_hook(
        pg, default.allreduce_hook
    ),
    "allgather then aggregate": lambda model, pg: model._register_comm_hook(
        pg, default.allgather_then_aggregate_hook
    ),
    "fp16 compress": lambda model, pg: model._register_comm_hook(
        pg, default.fp16_compress_hook
    ),
}
