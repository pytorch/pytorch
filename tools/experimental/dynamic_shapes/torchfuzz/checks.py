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
