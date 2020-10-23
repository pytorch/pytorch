import torch
# from typing import Literal
from typing_extensions import Literal

# @torch.jit._overload
# def foo(x: Literal[True]): ...

# @torch.jit._overload
# def foo(x: Literal[False]): ...

def foo(x: Literal[True]):
    if x:
        return 1, 2
    else:
        return 1

def foo2():
    return foo(True)

print(torch.jit.script(foo2).graph)
