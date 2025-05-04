import torch
from torch.library import register_fake
from torch._inductor.pattern_matcher import register_replacement, fwd_only, PatternMatcherPass

@torch.library.custom_op("mylib::foo_inplace", mutates_args={"x"})
def foo_inplace(x: torch.Tensor) -> None:
    x.add_(1)

# NOTE: only returning None is supported; the custom op cannot return `out`.
@torch.library.custom_op("mylib::bar", mutates_args={"out"})
def bar_out(x: torch.Tensor, out: torch.Tensor) -> None:
    out.copy_(x + 2)

@register_fake("mylib::bar")
def bar_out_fake(x: torch.Tensor, out: torch.Tensor) -> None:
    return None

@torch.library.custom_op("mylib::foobar_out", mutates_args={"out"})
def foobar_out(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    x.add_(1)
    out.copy_(x + 2)
    return out

def pattern(x, out):
    foo_inplace(x)
    bar_out(x, out)
    return out

def replacement(x, out):
    return foobar_out(x, out)

patterns = PatternMatcherPass()
register_replacement(
    search_fn=pattern,
    replace_fn=replacement,
    example_inputs=(torch.randn(3), torch.randn(3)),
    trace_fn=fwd_only,
    pass_dicts=patterns,
)

# user-function
@torch.compile(fullgraph=True)
def f(x):
    x = x.clone()
    out = torch.empty_like(x)
    foo_inplace(x)
    bar_out(x, out)
    return out

x = torch.randn(3, device="cpu")
f(x)