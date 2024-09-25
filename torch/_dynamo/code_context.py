import types
from typing import Dict, Any

from .utils import ExactWeakKeyDictionary


class CodeContextDict:
    def __init__(self) -> None:
        self.code_context: ExactWeakKeyDictionary[types.CodeType, Dict[str, Any]] = ExactWeakKeyDictionary()

    def has_context(self, code: types.CodeType) -> bool:
        return code in self.code_context

    def get_context(self, code: types.CodeType) -> Dict[str, Any]:
        ctx = self.code_context.get(code)
        if ctx is None:
            ctx = {}
            self.code_context[code] = ctx
        return ctx

    def pop_context(self, code: types.CodeType) -> Dict[str, Any]:
        ctx = self.get_context(code)
        self.code_context._remove_id(id(code))
        return ctx

    def clear(self) -> None:
        self.code_context.clear()


code_context: CodeContextDict = CodeContextDict()
