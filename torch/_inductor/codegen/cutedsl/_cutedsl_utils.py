# mypy: disable-error-code=import-not-found
# pyrefly: ignore [import-error, missing-import]
import cutlass.cute as cute


@cute.jit  # type: ignore[misc]
def ssa_to_indexable(ssa_value: cute.TensorSSA, dtype: str) -> cute.Numeric:
    """
    Convert SSA form to indexable non-SSA form.

    Workaround for lack of gather support: SSA values cannot be used directly
    as indices in tensor loads. This converts SSA to a register fragment, then
    extracts a scalar for indexing.
    """
    frag = ssa_to_fragment(ssa_value, dtype)
    return frag[0]


@cute.jit  # type: ignore[misc]
def ssa_to_fragment(ssa_value: cute.TensorSSA, dtype: str) -> cute.Tensor:
    """Materialize an SSA vector into a register fragment."""
    frag = cute.make_rmem_tensor(ssa_value.shape, dtype)
    frag.store(ssa_value)
    return frag


@cute.jit  # type: ignore[misc]
def result_to_ssa(value: cute.Numeric, dtype: str) -> cute.TensorSSA:
    """
    Convert non-SSA result back to SSA form.

    After performing operations with non-SSA values (like indexed loads),
    convert the result back to SSA form for further computation.
    """
    frag = cute.make_rmem_tensor(1, dtype)
    frag[0] = value
    return frag.load()
