"""Generic codegen surface shared by AOTAutograd's runtime wrappers.

``_compile_and_exec_source`` -- the chokepoint that compiles a generated wrapper
source string into a live function -- lives in codegen_utils.py alongside
PySourceBuilder. It is re-exported here so importers have a stably-named
``codegen`` module to reach for the generic exec primitive (and, once AOT-to-Python
lowering is added, the wrapper-source capture sink) without reaching into the
PySourceBuilder module. Kept as a leaf module (stdlib + torch only, no intra-package
imports beyond codegen_utils) so it is safe to import from anywhere.
"""

from .codegen_utils import _compile_and_exec_source


__all__ = ["_compile_and_exec_source"]
