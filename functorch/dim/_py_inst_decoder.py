import dis
from typing import Any, Optional


class _PyInstDecoder:
    """
    Decodes Python bytecode instructions to extract variable names
    """

    def __init__(self, code_object: Any, lasti: int) -> None:
        self.code_object = code_object
        self.instructions = list(dis.get_instructions(code_object))
        self.offset = self._find_instruction_index(lasti)

    def _find_instruction_index(self, lasti: int) -> int:
        """Find instruction index corresponding to lasti (byte offset)."""
        # Find the instruction at or before lasti
        # This should find the CALL instruction, not the next one
        best_idx = 0
        for i, instr in enumerate(self.instructions):
            if instr.offset <= lasti:
                best_idx = i
            else:
                break
        return best_idx

    def next(self) -> None:
        """Advance to the next instruction."""
        self.offset += 1

    def opcode(self) -> Optional[str]:
        """Get the opcode name of the current instruction."""
        if self.offset < len(self.instructions):
            return self.instructions[self.offset].opname
        return None

    def oparg(self) -> int:
        """Get the argument of the current instruction."""
        if self.offset < len(self.instructions):
            return self.instructions[self.offset].arg or 0
        return 0

    def name(self) -> Optional[str]:
        """
        Extract variable name from current instruction.
        """
        opname = self.opcode()
        if not opname:
            return None

        names = None
        if opname in ("STORE_NAME", "STORE_GLOBAL"):
            names = self.code_object.co_names
        elif opname == "STORE_FAST":
            names = self.code_object.co_varnames
        elif opname == "STORE_DEREF":
            names = self.code_object.co_cellvars
            if not names:
                names = self.code_object.co_freevars
        else:
            return None

        arg = self.oparg()
        if names and 0 <= arg < len(names):
            return names[arg]

        return None
