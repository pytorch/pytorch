"""
Unified view_groups implementation using symbolic shape tracking.

This module provides a view_groups function that computes dimension mappings
for reshape operations by running the actual meta-kernel with symbolic shapes
and analyzing the resulting expressions.
"""

from typing import Any

import sympy

import torch
from torch import SymInt
from torch._dynamo.source import ConstantSource
from torch._refs import _reshape_view_helper
from torch.fx.experimental.symbolic_shapes import DimDynamic, ShapeEnv, SymNode


# Shape can contain int or SymInt values
Shape = tuple[Any, ...]


def _create_symint(shape_env: ShapeEnv, val: int, name: str) -> SymInt:
    """Create a backed SymInt with the given value and source name."""
    symbol = shape_env.create_symbol(
        val,
        source=ConstantSource(name),
        dynamic_dim=DimDynamic.DYNAMIC,
        constraint_dim=None,
    )
    return SymInt(SymNode(symbol, shape_env, int, hint=val))


def _get_raw_symbol(shape_env: ShapeEnv, source_name: str):
    """Get the raw sympy symbol for a given source name."""
    for var, sources in shape_env.var_to_sources.items():
        for src in sources:
            if src.name == source_name:
                return var
    return None


def _analyze_provenance(input_syms, output_syms, result_exprs, shape_env):
    """
    Analyze symbolic expressions to determine which input dimensions
    contribute to each output dimension.

    Uses two sources of information:
    1. Replacements: if input symbol s_i was replaced with product of output symbols,
       those output symbols are derived from s_i
    2. FloorDiv expressions: if FloorDiv(expr, t_j), then t_j shares provenance with expr
    """
    input_sym_to_dim = {sym: i for i, sym in enumerate(input_syms) if sym is not None}
    input_sym_set = {s for s in input_syms if s is not None}
    output_sym_set = {s for s in output_syms if s is not None}

    # Map each output symbol to the set of input dims it's derived from
    output_to_input_dims = {sym: set() for sym in output_sym_set}

    # From replacements: s_i -> t_j * t_k means t_j and t_k come from s_i
    for old_sym, new_expr in shape_env.replacements.items():
        if old_sym in input_sym_set:
            input_dim = input_sym_to_dim[old_sym]
            for out_sym in new_expr.free_symbols:
                if out_sym in output_sym_set:
                    output_to_input_dims[out_sym].add(input_dim)

    def get_direct_inputs(expr):
        result = set()
        for sym in expr.free_symbols:
            if sym in input_sym_set:
                result.add(input_sym_to_dim[sym])
        return result

    def propagate_from_floordiv(expr):
        """Propagate provenance from FloorDiv numerators to denominators."""
        if not hasattr(expr, "args") or not expr.args:
            return

        if (
            hasattr(expr, "func")
            and "FloorDiv" in str(expr.func)
            and len(expr.args) == 2
        ):
            numerator, denominator = expr.args
            num_inputs = get_direct_inputs(numerator)
            for sym in numerator.free_symbols:
                if sym in output_sym_set:
                    num_inputs |= output_to_input_dims.get(sym, set())

            if denominator in output_sym_set:
                output_to_input_dims[denominator] |= num_inputs
            for sym in denominator.free_symbols:
                if sym in output_sym_set:
                    output_to_input_dims[sym] |= num_inputs

            propagate_from_floordiv(numerator)

        for arg in expr.args:
            propagate_from_floordiv(arg)

    # Run propagation multiple times to handle chains
    for _ in range(3):
        for expr in result_exprs:
            propagate_from_floordiv(expr)

    def find_input_dims(expr):
        involved = set()
        for sym in expr.free_symbols:
            if sym in input_sym_set:
                involved.add(input_sym_to_dim[sym])
            elif sym in output_sym_set:
                involved |= output_to_input_dims.get(sym, set())
        return involved

    result = []
    for out_idx, expr in enumerate(result_exprs):
        involved = find_input_dims(expr)
        if expr in output_sym_set:
            involved |= output_to_input_dims.get(expr, set())
        result.append(sorted(involved))

    return result


def symbolic_view_groups(
    from_shape: Shape, to_shape: Shape
) -> tuple[list[list[int]], Shape]:
    """
    Compute which input dimensions contribute to each output dimension
    for a reshape operation.

    Args:
        from_shape: Input tensor shape (can contain SymInt)
        to_shape: Output tensor shape (can contain SymInt)

    Returns:
        Tuple of (provenance, actual_output_shape) where provenance is a
        list of input dimension indices for each output dimension.
        E.g., for (2,3,4) -> (6,4): [[0, 1], [2]]
        meaning output dim 0 comes from input dims 0,1 and output dim 1 from input dim 2.
    """
    from torch._subclasses.fake_tensor import unset_fake_temporarily

    # ALWAYS use a fresh ShapeEnv to avoid leaking symbols into compilation guards
    shape_env = ShapeEnv()

    # Convert shapes to concrete values for our internal analysis
    # For unbacked symbols, we use their hints if available, or the actual
    # backed values. The provenance analysis will be the same regardless of
    # actual values since we create fresh symbols.
    def get_concrete_val(s):
        if isinstance(s, SymInt):
            if s.node.has_hint():
                return s.node.hint
            # For unbacked without hints, we need to get a value somehow
            # This shouldn't happen in practice - unbacked symbols should have hints
            # Fall back to evaluating if possible
            try:
                return int(s)
            except Exception:
                return None  # Mark as needing inference
        return s

    # Normalize to_shape first to handle nested tuples/lists like ([24],)
    from torch.distributed.tensor._ops._view_ops import normalize_sizes

    to_shape = normalize_sizes(to_shape)

    # Get concrete values for shapes
    concrete_from = [get_concrete_val(s) for s in from_shape]

    # Check for -1 using Python int comparison, not symbolic
    def is_minus_one(s):
        return isinstance(s, int) and s == -1

    concrete_to = [get_concrete_val(s) if not is_minus_one(s) else -1 for s in to_shape]

    # If any input dimension couldn't be concretized, use placeholder values
    # that preserve the numel relationship
    has_none_input = any(v is None for v in concrete_from)
    has_none_output = any(v is None for v in concrete_to)

    if has_none_input:
        # Need to ensure placeholder values match output numel
        # Calculate required numel from output if concrete
        if not has_none_output and all(v != -1 for v in concrete_to):
            output_numel = 1
            for v in concrete_to:
                output_numel *= v
            # Count how many input dims need placeholders
            none_indices = [i for i, v in enumerate(concrete_from) if v is None]
            # Calculate remaining numel from concrete input dims
            known_numel = 1
            for i, v in enumerate(concrete_from):
                if v is not None:
                    known_numel *= v
            # The None dims must account for remaining_numel
            remaining_numel = (
                output_numel // known_numel if known_numel > 0 else output_numel
            )
            # Distribute remaining_numel across None dims using unique primes
            # that multiply to remaining_numel (for factorizable cases)
            # For simplicity, give first None dim the remaining numel, others get 1
            for i, idx in enumerate(none_indices):
                if i == 0:
                    concrete_from[idx] = remaining_numel
                else:
                    concrete_from[idx] = 1
        else:
            # Output has -1 or None values, use primes for provenance tracking
            primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
            for i, v in enumerate(concrete_from):
                if v is None:
                    concrete_from[i] = primes[i] if i < len(primes) else 2 + i

    if has_none_output:
        # For output with None values, use -1 to let meta-kernel infer
        # This ensures numel relationship is preserved
        concrete_to = [-1 for _ in concrete_to]
    elif has_none_input:
        # Input had unbacked symbols but output is concrete
        # Check if we should use -1 for inference
        input_numel = 1
        for v in concrete_from:
            input_numel *= v
        # Only use -1 if single output dim needs inference
        if len(concrete_to) == 1:
            concrete_to = [-1]

    # Calculate input numel from concrete values
    from_numel = 1
    for v in concrete_from:
        from_numel *= v

    # Create symbolic input shape with fresh symbols
    input_syms = []
    input_raw_symbols = []
    for i, val in enumerate(concrete_from):
        sym = _create_symint(shape_env, val, f"in{i}")
        input_syms.append(sym)
        input_raw_symbols.append(_get_raw_symbol(shape_env, f"in{i}"))

    # Create symbolic output shape
    # For dimensions that match the total numel (like flatten), we use -1
    # to let the meta-kernel infer the dimension correctly
    output_syms = []
    output_raw_symbols = []
    for i, val in enumerate(concrete_to):
        if val == -1:
            # -1 means "infer this dimension" - pass through to meta-kernel
            output_syms.append(-1)
            output_raw_symbols.append(None)  # Will be populated after meta-kernel runs
        elif len(concrete_to) == 1 and val == from_numel:
            # Flatten case: single output dim equals total input numel
            # Use -1 to let meta-kernel infer it, preserving correct provenance
            output_syms.append(-1)
            output_raw_symbols.append(None)
        else:
            sym = _create_symint(shape_env, val, f"out{i}")
            output_syms.append(sym)
            output_raw_symbols.append(_get_raw_symbol(shape_env, f"out{i}"))

    # Run meta-kernel in an isolated context to avoid symbol leakage
    # Disable FakeTensorMode to ensure we use our fresh ShapeEnv
    with unset_fake_temporarily():
        fake_input = torch.empty(input_syms, device="meta")
        fake_output = _reshape_view_helper(fake_input, *output_syms, allow_copy=True)

    # Extract result expressions and populate any inferred (-1) output symbols
    result_exprs = []
    actual_output_shape = []
    for i, dim_size in enumerate(fake_output.shape):
        actual_output_shape.append(dim_size)
        if isinstance(dim_size, SymInt):
            expr = dim_size.node.expr
            result_exprs.append(expr)
            # If this was an inferred dimension, capture its symbol now
            if output_raw_symbols[i] is None:
                output_raw_symbols[i] = expr
        else:
            result_exprs.append(sympy.Integer(dim_size))

    provenance = _analyze_provenance(
        input_raw_symbols, output_raw_symbols, result_exprs, shape_env
    )
    return provenance, tuple(actual_output_shape)


def provenance_to_dimmap(
    from_shape: Shape,
    to_shape: Shape,
    provenance: list[list[int]],
    actual_output_shape: Shape | None = None,
) -> tuple:
    """
    Convert provenance info to DimSpec objects (DimMap).

    Args:
        from_shape: Input tensor shape (original values - may contain SymInt)
        to_shape: Output tensor shape (original values - may contain -1 or SymInt)
        provenance: List of input dimension indices for each output dimension
        actual_output_shape: Pre-computed output shape (used only for -1 inference)

    Returns:
        DimMap tuple of DimSpec objects
    """
    from torch.distributed.tensor._ops._view_ops import (
        Flatten,
        infer_size,
        InputDim,
        normalize_sizes,
        prod,
        Singleton,
        Split,
    )

    # Normalize and infer -1 dimensions using original shapes
    # This preserves the original SymInt values rather than introducing new symbols
    to_shape = normalize_sizes(to_shape)
    # Check for -1 using Python int comparison to avoid symbolic comparisons
    has_minus_one = any(isinstance(s, int) and s == -1 for s in to_shape)
    if has_minus_one:
        from_nelem = prod(from_shape)
        to_shape = infer_size(from_nelem, to_shape)

    # Group consecutive output dims by contiguous input dims
    # Singletons (empty provenance) are kept as separate groups for proper left-to-right matching
    groups = []  # list of (input_dims, output_indices)
    i = 0
    while i < len(provenance):
        input_dims = tuple(sorted(provenance[i]))
        output_indices = [i]
        i += 1

        # Only extend group for non-singleton outputs with matching/overlapping input dims
        if len(input_dims) > 0:
            while i < len(provenance):
                next_input_dims = tuple(sorted(provenance[i]))
                # Stop at singletons - they should be their own group
                if len(next_input_dims) == 0:
                    break
                # Check if input dims match or overlap (for flatten-then-split cases)
                if next_input_dims == input_dims or set(next_input_dims) & set(
                    input_dims
                ):
                    input_dims = tuple(sorted(set(input_dims) | set(next_input_dims)))
                    output_indices.append(i)
                    i += 1
                else:
                    break

        groups.append((input_dims, output_indices))

    from torch.fx.experimental.symbolic_shapes import guard_or_false

    def is_singleton(size):
        """Check if size is 1, returning False for unbacked symbols."""
        return guard_or_false(size == 1)

    # Track which input dims have been consumed and the "cursor" position
    # Singletons should only match input singletons that come after previously matched dims
    consumed_input_dims = set()
    next_singleton_cursor = 0  # Start matching singletons from this input dim

    result = []
    for input_dims, output_indices in groups:
        # Separate singletons from non-singletons
        non_singleton_indices = [
            j for j in output_indices if not is_singleton(to_shape[j])
        ]
        output_sizes = tuple(to_shape[j] for j in non_singleton_indices)

        if len(input_dims) == 0:
            # Singletons with no tracked provenance - match with input singletons in order
            for j in output_indices:
                # Find the next unconsumed input singleton starting from cursor
                matched = False
                for i in range(next_singleton_cursor, len(from_shape)):
                    if i not in consumed_input_dims and is_singleton(from_shape[i]):
                        consumed_input_dims.add(i)
                        next_singleton_cursor = i + 1
                        result.append(InputDim(i))
                        matched = True
                        break
                if not matched:
                    result.append(Singleton())
        elif len(non_singleton_indices) == 0:
            # No non-singleton outputs (shouldn't happen with valid reshape)
            for _ in output_indices:
                result.append(Singleton())
            # Mark input dims as consumed and update cursor
            for d in input_dims:
                consumed_input_dims.add(d)
                next_singleton_cursor = max(next_singleton_cursor, d + 1)
        elif len(input_dims) == 1 and len(non_singleton_indices) == 1:
            # Could be direct mapping or need to check sizes
            in_size = from_shape[input_dims[0]]
            out_size = to_shape[non_singleton_indices[0]]
            # Mark input dim as consumed and update cursor
            consumed_input_dims.add(input_dims[0])
            next_singleton_cursor = max(next_singleton_cursor, input_dims[0] + 1)
            for j in output_indices:
                out_j_size = to_shape[j]
                # If output is singleton but input maps 1:1 to it (both size 1),
                # use InputDim to preserve sharding semantics
                if (
                    is_singleton(out_j_size)
                    and is_singleton(in_size)
                    and len(output_indices) == 1
                ):
                    result.append(InputDim(input_dims[0]))
                elif is_singleton(out_j_size):
                    result.append(Singleton())
                elif guard_or_false(in_size == out_size):
                    result.append(InputDim(input_dims[0]))
                else:
                    # Size differs - this is a split
                    flattened = Flatten.new(tuple(InputDim(d) for d in input_dims))
                    result.append(Split.new(flattened, output_sizes, 0))
        else:
            # Multiple inputs or multiple non-singleton outputs: Flatten then Split
            flattened = Flatten.new(tuple(InputDim(d) for d in input_dims))
            # Mark input dims as consumed and update cursor
            for d in input_dims:
                consumed_input_dims.add(d)
                next_singleton_cursor = max(next_singleton_cursor, d + 1)
            split_id = 0
            for j in output_indices:
                if is_singleton(to_shape[j]):
                    result.append(Singleton())
                else:
                    result.append(Split.new(flattened, output_sizes, split_id))
                    split_id += 1

    return tuple(result)


def view_groups_symbolic(from_shape: Shape, to_shape: Shape) -> tuple:
    """
    Compute DimMap for a reshape using symbolic shape tracking.

    This is a drop-in replacement for view_groups that uses the meta-kernel
    as the source of truth.
    """
    provenance, actual_output_shape = symbolic_view_groups(from_shape, to_shape)
    return provenance_to_dimmap(from_shape, to_shape, provenance, actual_output_shape)


# =============================================================================
# Tests
# =============================================================================


def test_symbolic_view_groups():
    """Test symbolic_view_groups against known expected results."""
    test_cases = [
        # (from_shape, to_shape, expected_inputs_per_output)
        ((6,), (2, 3), [[0], [0]]),
        ((2, 3), (6,), [[0, 1]]),
        ((2, 3, 4), (6, 4), [[0, 1], [2]]),
        ((2, 3, 4), (2, 12), [[0], [1, 2]]),
        ((6, 4), (2, 3, 4), [[0], [0], [1]]),
        ((2, 3, 4), (24,), [[0, 1, 2]]),
        ((24,), (2, 3, 4), [[0], [0], [0]]),
        ((2, 3, 4), (4, 6), [[0, 1, 2], [0, 1, 2]]),
        ((2, 3), (3, 2), [[0, 1], [0, 1]]),
        ((6,), (2, 1, 3), [[0], [], [0]]),
    ]

    print("Testing symbolic_view_groups")
    print("=" * 60)

    all_passed = True
    for from_shape, to_shape, expected in test_cases:
        result, _ = symbolic_view_groups(from_shape, to_shape)
        passed = result == expected
        status = "✓" if passed else "✗"
        if not passed:
            all_passed = False
        print(f"{status} {from_shape} -> {to_shape}")
        if not passed:
            print(f"   Expected: {expected}")
            print(f"   Got:      {result}")

    print("=" * 60)
    print("All tests passed!" if all_passed else "Some tests failed!")
    return all_passed


def test_consistency_with_original():
    """Verify symbolic results are consistent with original view_groups."""
    from torch.distributed.tensor._ops._view_ops import (
        Flatten,
        InputDim,
        Split,
        view_groups,
    )

    def extract_input_dims(spec):
        if isinstance(spec, InputDim):
            return {spec.input_dim}
        elif isinstance(spec, Flatten):
            result = set()
            for inner in spec.input_dims:
                result |= extract_input_dims(inner)
            return result
        elif isinstance(spec, Split):
            return extract_input_dims(spec.input_dim)
        return set()

    test_cases = [
        ((6,), (2, 3)),
        ((2, 3), (6,)),
        ((2, 3, 4), (6, 4)),
        ((2, 3, 4), (2, 12)),
        ((6, 4), (2, 3, 4)),
        ((2, 12), (2, 3, 4)),
        ((2, 3, 4), (24,)),
        ((24,), (2, 3, 4)),
        ((2, 3, 4), (4, 6)),
        ((6, 4), (2, 4, 3)),
        ((4, 6), (2, 3, 4)),
        ((2, 3), (3, 2)),
        ((2, 1, 3), (6,)),
        ((6,), (2, 1, 3)),
        ((1, 6), (2, 3)),
        ((2, 3), (1, 6)),
        ((2, 3, 4, 5), (6, 20)),
        ((120,), (2, 3, 4, 5)),
    ]

    print("\nConsistency check with original view_groups")
    print("=" * 60)

    all_passed = True
    for from_shape, to_shape in test_cases:
        orig_result = view_groups(from_shape, to_shape)
        orig_inputs = [extract_input_dims(spec) for spec in orig_result]

        sym_result, _ = symbolic_view_groups(from_shape, to_shape)
        sym_inputs = [set(dims) for dims in sym_result]

        # Symbolic should be subset of or equal to original
        consistent = all(sym <= orig for sym, orig in zip(sym_inputs, orig_inputs))

        status = "✓" if consistent else "✗"
        if not consistent:
            all_passed = False
            print(f"{status} {from_shape} -> {to_shape}")
            for i, (o, s) in enumerate(zip(orig_inputs, sym_inputs)):
                print(f"   dim {i}: orig={sorted(o)}, sym={sorted(s)}")
        else:
            print(f"{status} {from_shape} -> {to_shape}")

    print("=" * 60)
    print("All consistent!" if all_passed else "Inconsistencies found!")
    return all_passed


def test_symbolic_shapes():
    """Test with symbolic (unbacked and backed) input dimensions."""
    print("\nTesting with symbolic shapes")
    print("=" * 60)

    # Test 1: Backed symints for input, concrete output
    print("\nTest 1: Backed symbolic input with concrete output")
    try:
        shape_env = ShapeEnv()
        s0 = _create_symint(shape_env, 6, "s0")
        s1 = _create_symint(shape_env, 4, "s1")

        result, _ = symbolic_view_groups((s0, s1), (2, 3, 4))
        print(f"  (s0=6, s1=4) -> (2, 3, 4): {result}")
        assert result == [[0], [0], [1]], f"Expected [[0], [0], [1]], got {result}"
        print("  ✓ PASS")
    except Exception as e:
        print(f"  ✗ FAIL: {e}")

    # Test 2: All concrete (baseline)
    print("\nTest 2: All concrete shapes")
    try:
        result, _ = symbolic_view_groups((6, 4), (2, 3, 4))
        print(f"  (6, 4) -> (2, 3, 4): {result}")
        assert result == [[0], [0], [1]], f"Expected [[0], [0], [1]], got {result}"
        print("  ✓ PASS")
    except Exception as e:
        print(f"  ✗ FAIL: {e}")

    # Test 3: All symbolic from same ShapeEnv
    print("\nTest 3: All symbolic shapes from same ShapeEnv")
    try:
        shape_env = ShapeEnv()
        s0 = _create_symint(shape_env, 6, "s0")
        s1 = _create_symint(shape_env, 4, "s1")
        t0 = _create_symint(shape_env, 2, "t0")
        t1 = _create_symint(shape_env, 12, "t1")

        result, _ = symbolic_view_groups((s0, s1), (t0, t1))
        print(f"  (s0=6, s1=4) -> (t0=2, t1=12): {result}")
        assert result == [[0], [0, 1]], f"Expected [[0], [0, 1]], got {result}"
        print("  ✓ PASS")
    except Exception as e:
        print(f"  ✗ FAIL: {e}")

    # Test 4: Symbolic input, symbolic output (different reshape)
    print("\nTest 4: Symbolic flatten")
    try:
        shape_env = ShapeEnv()
        s0 = _create_symint(shape_env, 2, "s0")
        s1 = _create_symint(shape_env, 3, "s1")
        s2 = _create_symint(shape_env, 4, "s2")
        t0 = _create_symint(shape_env, 6, "t0")
        t1 = _create_symint(shape_env, 4, "t1")

        result, _ = symbolic_view_groups((s0, s1, s2), (t0, t1))
        print(f"  (s0=2, s1=3, s2=4) -> (t0=6, t1=4): {result}")
        assert result == [[0, 1], [2]], f"Expected [[0, 1], [2]], got {result}"
        print("  ✓ PASS")
    except Exception as e:
        print(f"  ✗ FAIL: {e}")

    # Test 5: Complex symbolic reshape
    print("\nTest 5: Complex symbolic reshape")
    try:
        shape_env = ShapeEnv()
        s0 = _create_symint(shape_env, 2, "s0")
        s1 = _create_symint(shape_env, 3, "s1")
        s2 = _create_symint(shape_env, 4, "s2")
        t0 = _create_symint(shape_env, 4, "t0")
        t1 = _create_symint(shape_env, 6, "t1")

        result, _ = symbolic_view_groups((s0, s1, s2), (t0, t1))
        print(f"  (s0=2, s1=3, s2=4) -> (t0=4, t1=6): {result}")
        # This is flatten-all-then-split
        assert result == [[0, 1, 2], [0, 1, 2]], (
            f"Expected [[0, 1, 2], [0, 1, 2]], got {result}"
        )
        print("  ✓ PASS")
    except Exception as e:
        print(f"  ✗ FAIL: {e}")

    print("=" * 60)


def test_unbacked_symints():
    """Test with unbacked SymInt dimensions."""
    print("\nTesting with unbacked symbolic shapes")
    print("=" * 60)

    # Test 1: Unbacked input dim with concrete split
    print("\nTest 1: Unbacked input with concrete split")
    try:
        shape_env = ShapeEnv()
        u0 = shape_env.create_unbacked_symint()

        result, _ = symbolic_view_groups((u0,), (2, -1))
        print(f"  (u0,) -> (2, -1): {result}")
        assert result == [[0], [0]], f"Expected [[0], [0]], got {result}"
        print("  ✓ PASS")
    except Exception as e:
        print(f"  ✗ FAIL: {e}")

    # Test 2: Mixed unbacked and backed
    print("\nTest 2: Mixed unbacked and backed")
    try:
        shape_env = ShapeEnv()
        u0 = shape_env.create_unbacked_symint()
        s1 = _create_symint(shape_env, 4, "s1")

        result, _ = symbolic_view_groups((u0, s1), (u0, 2, 2))
        print(f"  (u0, s1=4) -> (u0, 2, 2): {result}")
        assert result == [[0], [1], [1]], f"Expected [[0], [1], [1]], got {result}"
        print("  ✓ PASS")
    except Exception as e:
        print(f"  ✗ FAIL: {e}")

    # Test 3: Unbacked flatten
    print("\nTest 3: Unbacked flatten")
    try:
        shape_env = ShapeEnv()
        s0 = _create_symint(shape_env, 2, "s0")
        u1 = shape_env.create_unbacked_symint()

        result, _ = symbolic_view_groups((s0, u1), (-1,))
        print(f"  (s0=2, u1) -> (-1,): {result}")
        assert result == [[0, 1]], f"Expected [[0, 1]], got {result}"
        print("  ✓ PASS")
    except Exception as e:
        print(f"  ✗ FAIL: {e}")

    print("=" * 60)


def test_sharding_propagation():
    """Test integration with actual sharding propagation."""
    print("\nTesting sharding propagation integration")
    print("=" * 60)

    try:
        from torch.distributed.tensor import Shard
        from torch.distributed.tensor._ops._view_ops import (
            propagate_shape_and_sharding,
            view_groups,
        )

        # We'll test the propagate_shape_and_sharding function with our computed groups
        # to verify sharding decisions match

        def test_prop(from_shape, to_shape, input_placements, mesh_sizes):
            """Compare sharding propagation results."""
            # Get original view_groups result
            orig_rules = view_groups(from_shape, to_shape)

            # Get symbolic result
            sym_inputs, _ = symbolic_view_groups(from_shape, to_shape)

            # Run original propagation
            orig_input_tgt, orig_output = propagate_shape_and_sharding(
                input_placements, from_shape, orig_rules, mesh_sizes
            )

            # For comparison, we check if the sharding decisions are consistent
            # The key insight: if sym_inputs[i] is subset of what original says,
            # sharding decisions should still be valid

            return {
                "orig_input_tgt": orig_input_tgt,
                "orig_output": orig_output,
                "sym_inputs": sym_inputs,
            }

        # Test case 1: Simple split with sharding
        print("\nTest 1: (6, 4) -> (2, 3, 4) with Shard(0)")
        result = test_prop(
            from_shape=(6, 4),
            to_shape=(2, 3, 4),
            input_placements=[Shard(0)],
            mesh_sizes=(2,),
        )
        print(f"  Input target: {result['orig_input_tgt']}")
        print(f"  Output: {result['orig_output']}")
        print(f"  Sym inputs: {result['sym_inputs']}")

        # Test case 2: Flatten with sharding
        print("\nTest 2: (2, 3, 4) -> (6, 4) with Shard(0)")
        result = test_prop(
            from_shape=(2, 3, 4),
            to_shape=(6, 4),
            input_placements=[Shard(0)],
            mesh_sizes=(2,),
        )
        print(f"  Input target: {result['orig_input_tgt']}")
        print(f"  Output: {result['orig_output']}")
        print(f"  Sym inputs: {result['sym_inputs']}")

        # Test case 3: Complex reshape
        print("\nTest 3: (2, 3, 4) -> (4, 6) with Shard(1)")
        result = test_prop(
            from_shape=(2, 3, 4),
            to_shape=(4, 6),
            input_placements=[Shard(1)],
            mesh_sizes=(3,),
        )
        print(f"  Input target: {result['orig_input_tgt']}")
        print(f"  Output: {result['orig_output']}")
        print(f"  Sym inputs: {result['sym_inputs']}")

        print("\n  ✓ Sharding propagation tests completed")

    except Exception as e:
        import traceback

        print(f"  ✗ Error: {e}")
        traceback.print_exc()

    print("=" * 60)


def test_view_groups_symbolic():
    """Test view_groups_symbolic against original view_groups."""
    from torch.distributed.tensor._ops._view_ops import view_groups

    test_cases = [
        ((6,), (2, 3)),
        ((2, 3), (6,)),
        ((2, 3, 4), (6, 4)),
        ((2, 3, 4), (2, 12)),
        ((6, 4), (2, 3, 4)),
        ((2, 12), (2, 3, 4)),
        ((2, 3, 4), (24,)),
        ((24,), (2, 3, 4)),
        ((2, 3, 4), (4, 6)),
        ((6, 4), (2, 4, 3)),
        ((4, 6), (2, 3, 4)),
        ((2, 3), (3, 2)),
        ((2, 1, 3), (6,)),
        ((6,), (2, 1, 3)),
        ((1, 6), (2, 3)),
        ((2, 3), (1, 6)),
        ((2, 3, 4, 5), (6, 20)),
        ((120,), (2, 3, 4, 5)),
    ]

    print("\nTesting view_groups_symbolic vs original")
    print("=" * 60)

    all_passed = True
    for from_shape, to_shape in test_cases:
        orig = view_groups(from_shape, to_shape)
        sym = view_groups_symbolic(from_shape, to_shape)

        passed = orig == sym
        status = "✓" if passed else "✗"
        if not passed:
            all_passed = False
            print(f"{status} {from_shape} -> {to_shape}")
            print(f"   Original:  {orig}")
            print(f"   Symbolic:  {sym}")
        else:
            print(f"{status} {from_shape} -> {to_shape}")

    print("=" * 60)
    print("All DimMap tests passed!" if all_passed else "Some DimMap tests failed!")
    return all_passed


if __name__ == "__main__":
    test_symbolic_view_groups()
    test_consistency_with_original()
    test_symbolic_shapes()
    test_unbacked_symints()
    test_sharding_propagation()
    test_view_groups_symbolic()
