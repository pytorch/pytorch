from torch.utils import _pytree as pytree
from torch.utils._python_dispatch import TorchDispatchMode

from transformer_model import *

from helpers import *


args = ModelArgs.from_name("mini_primes")
transformer = Transformer(args)
transformer.setup_caches(4, 16)

ops = log_aten_ops(
    transformer, (torch.ones(4, 2, dtype=torch.int32), torch.ones(2, dtype=torch.int32))
)
for op in ops.keys():
    print(op)
    for arg, out in ops[op]:
        print(f"  {arg} -> {out}")

dot_aten_graph(
    transformer,
    (torch.ones(4, 2, dtype=torch.int32), torch.ones(2, dtype=torch.int32)),
    "transformer_graph.pdf",
    100,
)
