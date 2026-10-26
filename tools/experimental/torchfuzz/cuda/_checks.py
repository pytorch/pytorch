"""Concrete Check subclasses provided by the CUDA device plugin."""

from torchfuzz.checks import Check


class EagerVsReduceOverheadCheck(Check):
    """Exercise cudagraph_trees via ``torch.compile(mode="reduce-overhead")``.

    A single call only warms up; cudagraph_trees records on the second call and
    replays afterwards, so we run several steps (each preceded by
    ``cudagraph_mark_step_begin``) and compare every step's output against a
    fresh eager reference. This targets the cudagraph-specific hazards: recording
    vs replay correctness, cross-step output overwrite/poison, and (with
    backward) the separate backward cudagraph tree. The numeric check mirrors
    the sum-relative-diff heuristic used by the standard numerics check so eager
    vs inductor kernel differences do not cause false positives, while the gross
    corruption cudagraph bugs produce (stale/overwritten data) is caught.
    """

    steps = 4

    def codegen(self, args_tuple: str) -> list[str]:
        return [
            f"args = {args_tuple}",
            "out_eager = fuzzed_program(*args)",
            "out_eager.sum().backward()",
            "print('✅ eager + backward success')",
            "ref_sum = out_eager.detach().double().sum()",
            "# Determinism gate: nondeterministic programs (dropout, randint-based",
            "# gather/index_select indices, atomic reductions) are excluded from the",
            "# numeric oracle -- they legitimately differ eager-vs-compiled. Run",
            "# eager several times; only compare numerically if every run matches",
            "# (a single extra run misses low-entropy randint-index nondeterminism).",
            "try:",
            "    _deterministic = True",
            "    for _ in range(6):",
            "        _en = fuzzed_program(*args)",
            "        if not torch.equal(out_eager.detach(), _en.detach()):",
            "            _deterministic = False",
            "            break",
            "except Exception:",
            "    _deterministic = False",
            "print(f'deterministic: {_deterministic}')",
            "compiled_program = torch.compile(",
            "    fuzzed_program, mode='reduce-overhead', fullgraph=True",
            ")",
            f"for _step in range({self.steps}):",
            "    torch.compiler.cudagraph_mark_step_begin()",
            "    out_compiled = compiled_program(*args)",
            "    out_compiled.sum().backward()",
            "    torch.cuda.synchronize()",
            "    if _deterministic:",
            "        cur_sum = out_compiled.detach().double().sum()",
            "        diff = (ref_sum - cur_sum).abs().item()",
            "        rel = diff / (ref_sum.abs().item() + 1e-12) * 100",
            "        if rel > 5 and diff > 1:",
            "            print(",
            "                f'❌ reduce-overhead step {_step} differs: '",
            "                f'rel={rel:.6f}% abs={diff} '",
            "                f'eager={ref_sum.item()} compiled={cur_sum.item()}'",
            "            )",
            "            import sys; sys.exit(1)",
            "print('✅ reduce-overhead multi-step + backward success')",
            "# Coverage: report whether cudagraphs actually engaged (many programs",
            "# legitimately skip -- cpu/scalar/mutation/etc -- so this is not fatal).",
            "try:",
            "    import torch._inductor.cudagraph_trees as _ct",
            "    _mgr = _ct.get_container(torch.cuda.current_device()).tree_manager",
            "    print(f'cudagraphs_engaged: {_mgr is not None}')",
            "except Exception as _e:",
            "    print(f'cudagraphs_engaged: unknown ({_e!r})')",
        ]


class EagerVsFullGraphDynamicCompileCheck(Check):
    """Standard check that runs eager then fullgraph+dynamic compilation."""

    def codegen(self, args_tuple: str) -> list[str]:
        return [
            f"args = {args_tuple}",
            "result_original = fuzzed_program(*args)",
            "print('✅ eager success')",
            "compiled_program = torch.compile(fuzzed_program, fullgraph=True, dynamic=True)",
            "result_compiled = compiled_program(*args)",
            "print('✅ compile success')",
        ]


class EagerVsFullGraphDynamicCompileWithBackwardCheck(Check):
    """Check that runs eager then fullgraph+dynamic compilation with backward pass."""

    def codegen(self, args_tuple: str) -> list[str]:
        return [
            f"args = {args_tuple}",
            "result_original = fuzzed_program(*args)",
            "result_original.sum().backward()",
            "print('✅ eager + backward success')",
            "compiled_program = torch.compile(fuzzed_program, fullgraph=True, dynamic=True)",
            "result_compiled = compiled_program(*args)",
            "result_compiled.sum().backward()",
            "print('✅ compile + backward success')",
        ]


class EagerVsFullGraphDynamicCompileWithNumericsCheck(Check):
    """Check that runs eager and compiled, compares forward numerics."""

    def codegen(self, args_tuple: str) -> list[str]:
        return [
            f"args = {args_tuple}",
            "out_eager = fuzzed_program(*args)",
            "out_eager.sum().backward()",
            "print('Eager Success! ✅')",
            "compiled_program = torch.compile(fuzzed_program, fullgraph=True, dynamic=True)",
            "out_compiled = compiled_program(*args)",
            "out_compiled.sum().backward()",
            "print('Compile Success! ✅')",
            "out_eager_sum = out_eager.sum()",
            "out_compiled_sum = out_compiled.sum()",
            "diff = (out_eager_sum - out_compiled_sum).abs().item()",
            "rel_diff = diff / (out_eager_sum.abs().item() + 1e-12) * 100",
            "print(f'Relative diff (sum): {rel_diff:.6f}%')",
            "if rel_diff > 5 and diff > 1:",
            "    print(f'❌ Forward output sums differ significantly (relative and absolute)!')",
            "    print('out_eager_sum:', out_eager_sum.item())",
            "    print('out_compiled_sum:', out_compiled_sum.item())",
            "    print('Absolute diff:', diff)",
            "    print('Relative diff (%):', rel_diff)",
            "    import sys; sys.exit(1)",
        ]
