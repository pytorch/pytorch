from .allreduce_hook import allreduce_hook
from .hybrid_hook import hybrid_hook
from .rpc_hook import rpc_hook
from .sparse_rpc_hook import sparse_rpc_hook

ddp_hook_map = {
    "allreduce_hook": allreduce_hook,
    "rpc_hook": rpc_hook,
    "sparse_rpc_hook": sparse_rpc_hook,
    "hybrid_hook": hybrid_hook
}
