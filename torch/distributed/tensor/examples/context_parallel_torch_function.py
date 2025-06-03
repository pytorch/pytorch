"""
This code serves a minimal repro for torch_function_mode + flex_attention issue:

To run the example, use the following command:
python context_parallel_torch_function.py
"""

from typing import Any, Callable, Optional

import torch

from torch.overrides import TorchFunctionMode


class ContextParallel(TorchFunctionMode):
    def __init__(self):
        print("create torch_function_mode")
        super().__init__()

    def __torch_function__(
        self,
        func: Callable,
        types: Any,
        args: tuple[Any, ...] = (),
        kwargs: Optional[dict[str, Any]] = None,
    ) -> Any:
        kwargs = kwargs or {}

        # print(func)
        return func(*args, **kwargs)


def test():
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention

    def causal_mask(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx

    torch.cuda.manual_seed(10)
    q = torch.rand(1, 1, 16, 16, device="cuda")
    k = torch.rand(1, 1, 16, 16, device="cuda")
    v = torch.rand(1, 1, 16, 16, device="cuda")

    block_mask = create_block_mask(
        causal_mask,
        B=1,
        H=1,
        Q_LEN=16,
        KV_LEN=16,
        device="cuda",
    )
    compiled_flex_attention = torch.compile(
        flex_attention, dynamic=False, fullgraph=True
    )

    # with ContextParallel():
    out = compiled_flex_attention(q, k, v, block_mask=block_mask)
    # print(out)
    # assert torch.equal(out, torch.zeros(2, 2, 2, 2))


if __name__ == "__main__":
    test()
