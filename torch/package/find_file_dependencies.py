from typing import List, Optional, Tuple
import ast
from ._importlib import _resolve_name

class _ExtractModuleReferences(ast.NodeVisitor):
    """
    Extract the list of global variables a block of code will read and write
    """

    @classmethod
    def run(cls, src: str, package: str) -> List[Tuple[str, Optional[str]]]:
        visitor = cls(package)
        tree = ast.parse(src)
        visitor.visit(tree)
        return list(visitor.references.keys())

    def __init__(self, package):
        super().__init__()
        self.package = package
        self.references = {}

    def _absmodule(self, module_name: str, level: int) -> str:
        if level > 0:
            return _resolve_name(module_name, self.package, level)
        return module_name

    def visit_Import(self, node):
        for alias in node.names:
            self.references[(alias.name, None)] = True

    def visit_ImportFrom(self, node):
        name = self._absmodule(node.module, 0 if node.level is None else node.level)
        for alias in node.names:
            # from my_package import foo
            # foo may be a module, so we have to add it to the list of
            # potential references, if import of it fails, we will ignore it
            if alias.name != '*':
                self.references[(name, alias.name)] = True
            else:
                self.references[(name, None)] = True

find_files_source_depends_on = _ExtractModuleReferences.run
