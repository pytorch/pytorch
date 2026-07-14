"""Check abstractions for different execution modes and validations."""

from abc import ABC, abstractmethod


class Check(ABC):
    """Base class for execution checks."""

    @abstractmethod
    def codegen(self, args_tuple: str) -> list[str]:
        """Generate code lines for this check."""


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
