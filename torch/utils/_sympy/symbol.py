# mypy: allow-untyped-defs
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
from typing import Optional, Tuple, Union

import sympy


class SymT(Enum):
    SIZE = auto()
    FLOAT = auto()
    UNBACKED_INT = auto()
    UNBACKED_FLOAT = auto()
    # Inductor: The intermediates in inner_fn tmp0, one generated per ops call.
    # If one of these shows up in an indexing expression, that means an
    # indirect load is happening.
    TMP = auto()
    # Inductor: Placeholder variable that is later replaced with TMP
    INDIRECT = auto()
    # Inductor: Some size expressions are replaced with a precomputed size ps0
    # which is computed host side, and then directly reused in the kernel, so
    # we don't repeatedly recompute it on device.
    PRECOMPUTED_SIZE = auto()
    # Inductor: An indexing variable i0 in loops IR which ranges over non-reduced
    # dim in the loop
    INDEX = auto()
    # Inductor: A reduction indexing r0 variable in loops IR which ranges over
    # reduced dim in the loop
    RINDEX = auto()
    # Inductor: In templated kernels torch._inductor.kernel, we have a hook to
    # store the final output and append epilogue fusions.  To do this, we must
    # know what the indexes the outputs range over.  NB: These will also
    # advertise as INDEX, this is... probably OK?
    TEMPLATE_INDEX = auto()
    # Inductor: iteration domain for blockIdx.x/blockIdx.y
    XBLOCK = auto()
    YBLOCK = auto()
    # Inductor: this is used solely for dynamic_reshape_indexer
    VIEW = auto()
    # Alternate (non-modular) indexing used in halide kernels
    HALIDE = auto()


# Invariant: there must not be a prefix which is a prefix of another string,
# as this introduces ambiguity
prefix_str = {
    SymT.SIZE: "s",  # integer
    SymT.UNBACKED_INT: "u",  # integer
    # Prefix z here is chosen to avoid false aliasing in symbol_is_type test
    # DO NOT add a "z" type.  You also need to avoid conflicts on these
    # prefixes but this is somewhat easier to manage
    SymT.FLOAT: "zf",
    SymT.UNBACKED_FLOAT: "zuf",
    SymT.TMP: "tmp",
    SymT.PRECOMPUTED_SIZE: "ps",
    SymT.INDEX: "i",
    SymT.RINDEX: "r",
    SymT.TEMPLATE_INDEX: "idx",
    SymT.XBLOCK: "x",
    SymT.YBLOCK: "y",
    SymT.INDIRECT: "indirect",  # false aliasing?
    SymT.VIEW: "view",
    SymT.HALIDE: "h",
}


# A custom class to keep SymT prefix as an attribute.
# The name also keeps the string form of the prefix as an attribute.
class PrefixedSymbol(sympy.Symbol):
    def __new__(
        cls,
        name: str,
        prefix: SymT,
        *,
        integer: Optional[bool] = None,
        nonnegative: Optional[bool] = None,
        positive: Optional[bool] = None,
        real: Optional[bool] = None,
    ):
        obj = sympy.Symbol.__new__(
            cls,
            name,
            integer=integer,
            nonnegative=nonnegative,
            positive=positive,
            real=real,
        )
        obj.prefix = prefix
        return obj

    def _hashable_content(self):
        # since we use only 3 assumptions, we only check on those three instead of the
        # full set of assumptions to avoid an expensive caching operation
        return (
            self.name,
            self.prefix,
            self.is_integer,
            self.is_nonnegative,
            self.is_positive,
            self.is_real,
        )

    def __copy__(self):
        cls = self.__class__
        return cls.__new__(
            cls,
            name=self.name,
            prefix=self.prefix,
            integer=self.is_integer,
            nonnegative=self.is_nonnegative,
            positive=self.is_positive,
            real=self.is_real,
        )

    def __deepcopy__(self, memo):
        result = self.__copy__()
        memo[id(self)] = result
        return result


def make_symbol(
    prefix: SymT,
    idx: int,
    *,
    integer: Optional[bool] = None,
    nonnegative: Optional[bool] = None,
    positive: Optional[bool] = None,
    real: Optional[bool] = None,
) -> sympy.Symbol:
    return PrefixedSymbol(
        f"{prefix_str[prefix]}{idx}",
        prefix=prefix,
        integer=integer,
        nonnegative=nonnegative,
        positive=positive,
        real=real,
    )


# This type is a little wider than it should be, because free_symbols says
# that it contains Basic, rather than Symbol
def symbol_is_type(sym: sympy.Basic, prefix: Union[SymT, Tuple[SymT, ...]]) -> bool:
    if isinstance(sym, PrefixedSymbol):
        # This function is called a *lot* of times, so it needs to be fast
        if type(prefix) is tuple:
            # a list comprehension with any is slow
            for p in prefix:
                if sym.prefix == p:
                    return True
            return False
        elif type(prefix) is SymT:
            return sym.prefix == prefix
        else:
            raise ValueError(f"Unknown type for prefix: {type(prefix)}")
    else:
        assert isinstance(sym, sympy.Symbol)
        # TODO: remove all these Symbols and use PrefixedSymbol
        #       Hint: they are created by sympy_index_symbol
        sym_name = sym.name.lower()
        if type(prefix) is tuple:
            for p in prefix:
                if sym_name.startswith(prefix_str[p]):
                    return True
            return False
        elif type(prefix) is SymT:
            return sym_name.startswith(prefix_str[prefix])
        else:
            raise ValueError(f"Unknown type for prefix: {type(prefix)}")


def free_symbol_is_type(e: sympy.Expr, prefix: SymT) -> bool:
    return any(symbol_is_type(v, prefix) for v in e.free_symbols)
