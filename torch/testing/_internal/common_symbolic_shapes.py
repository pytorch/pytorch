from torch import SymBool, SymFloat, SymInt
from torch.fx.experimental.sym_node import SymNode
from torch.fx.experimental.symbolic_shapes import DimDynamic


def create_symtype(cls, pytype, shape_env, val, duck=True):
    from torch._dynamo.source import ConstantSource

    symbol = shape_env.create_symbol(
        val,
        source=ConstantSource(f"__testing_only{len(shape_env.var_to_val)}"),
        dynamic_dim=DimDynamic.DUCK if duck else DimDynamic.DYNAMIC,
        constraint_dim=None,
    )
    return cls(
        SymNode(
            symbol,
            shape_env,
            pytype,
            hint=val,
        )
    )


# TODO: default duck to False
def create_symint(shape_env, i: int, duck=True) -> SymInt:
    return create_symtype(SymInt, int, shape_env, i, duck=duck)


def create_symbool(shape_env, b: bool) -> SymBool:
    return create_symtype(SymBool, bool, shape_env, b)


def create_symfloat(shape_env, f: float) -> SymFloat:
    return create_symtype(SymFloat, float, shape_env, f)
