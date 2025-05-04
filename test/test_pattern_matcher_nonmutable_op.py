import torch
from torch.library import register_fake
from torch._inductor.pattern_matcher import register_replacement, fwd_only, PatternMatcherPass

@torch.library.custom_op("mylib::foo", mutates_args={})
def foo(x: torch.Tensor) -> torch.Tensor:
    return x + 1

@register_fake("mylib::foo")
def foo_fake(x: torch.Tensor) -> torch.Tensor:
    return x

@torch.library.custom_op("mylib::bar", mutates_args={})
def bar(x: torch.Tensor) -> torch.Tensor:
    return x + 2

@register_fake("mylib::bar")
def bar_fake(x: torch.Tensor) -> torch.Tensor:
    return x

@torch.library.custom_op("mylib::foobar", mutates_args={})
def foobar(x: torch.Tensor) -> torch.Tensor:
    return x + 3

def pattern(x):
    o1 = foo(x)
    o2 = bar(o1)
    return o2

def replacement(x):
    return foobar(x)

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
    return bar(foo(x))

x = torch.randn(3, device="cpu")
f(x)