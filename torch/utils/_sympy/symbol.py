"""
This file contains canonical definitions for our symbol naming conventions,
across torch.fx.experimental.symbolic_shapes and torch._inductor.  The
intention is:

1. To make it easily greppable where all the sites we use a prefix are
2. Make it possible to easily tell if we can introduce a new prefix without
   introducing a conflict

You can occasionally test if prefixes have been hardcoded by renaming prefixes
in this file and seeing what breaks.
"""

from enum import auto, Enum
from typing import Sequence, Union

import sympy


class SymT(Enum):
    SIZE = auto()
    UNBACKED_INT = auto()
    UNBACKED_FLOAT = auto()


# Invariant: there must not be a prefix which is a prefix of another string,
# as this introduces ambiguity
prefix_str = {
    SymT.SIZE: "s",  # integer
    SymT.UNBACKED_INT: "u",  # integer
    SymT.UNBACKED_FLOAT: "f",
}


def make_symbol(prefix: SymT, idx: int, **kwargs) -> sympy.Symbol:
    # TODO: maybe put the assumptions here directly
    return sympy.Symbol(f"{prefix_str[prefix]}{idx}", **kwargs)


def symbol_is_type(sym: sympy.Symbol, prefix: Union[SymT, Sequence[SymT]]) -> bool:
    if isinstance(prefix, SymT):
        return sym.name.startswith(prefix_str[prefix])
    else:
        return sym.name.startswith(tuple(prefix_str[p] for p in prefix))
