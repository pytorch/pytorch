import torch
import torch.nn as nn
from einops import einsum, pack, rearrange, reduce, repeat, unpack

class TorchModuleWithOperations(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x_abc, suffix=""):
        a, b, c = x_abc.shape

        def suf(pattern):
            parts = pattern.split()
            return " ".join(
                [p if p[-1] not in "acd" else p + suffix for p in parts]
            )

        # patterns look a bit strange because names a, c, d will be modified on every run
        # by suf function
        x_abcd = repeat(x_abc, suf("a b c -> a b c 4"))
        x_abc = reduce(x_abcd, suf("a b c d -> a b c"), "min")
        x_abdc, ps = pack([x_abc] * (2 + len(suffix)), suf("a b * c"))
        x_array = unpack(
            rearrange(x_abdc, suf("a b d c -> (a b ) 1 c d")), ps, "ab one1 c *"
        )
        x1 = x_array[0] + len(x_array)
        x1 = rearrange(x1, suf("(a b ) 1 c -> a b c"), b=b)
        addition = einsum(x_abc, x_abcd, suf("a b c , a b c d -> d"))[0]
        return x1 + addition

original = TorchModuleWithOperations()
# Einops only interacts with Dynamo but we test backend="inductor" just in case
compiled = torch.compile(original, backend="eager", fullgraph=True)
for size in [10, 20, 40]:
    x = torch.rand([size, size + 1, size + 2])
    for suffix in ["", "suf1", "other_suffix"]:
        result1 = compiled(x, suffix)
