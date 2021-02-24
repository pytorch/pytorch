import ast
import inspect
import textwrap
import copy
from types import FunctionType
from typing import cast, Union, Callable, Dict, Optional, Any
from torch.fx.symbolic_trace import Tracer
from torch.fx.graph import Graph
from torch.jit.frontend import normalize_source_lines
import torch

class Assert_Rewriter(ast.NodeTransformer):
    """
    Take a FunctionType object representing a `forward` method, then
    perform an AST rewrite to swap out nodes that are not symbolically
    traceable with a callsite to the FX alternative.

    To support swapping out an AST node, define a new `visit` method on
    that node. For more details, see:
    https://docs.python.org/3/library/ast.html#ast.NodeTransformer
    """

    def rewrite(self, fn: FunctionType):

        # Normalize the source lines
        sourcelines, _ = inspect.getsourcelines(fn)
        sourcelines = normalize_source_lines(sourcelines)
        source = ''.join(sourcelines)
        normalized_str = textwrap.dedent(source)

        # Rewrite the original AST
        source_ast = ast.parse(normalized_str)
        dest_ast = ast.fix_missing_locations(self.visit(source_ast))

        # Pull out the compiled fucntion from the newly-created Module
        code = compile(dest_ast, "", "exec")
        globals_dict = copy.copy(fn.__globals__)
        keys_before = set(globals_dict.keys())
        exec(code, globals_dict)
        new_keys = list(set(globals_dict.keys()) - keys_before)
        assert len(new_keys) == 1
        fn_compiled = globals_dict[new_keys[0]]

        # Return the correct FunctionType object
        return fn_compiled

    def visit_Assert(self, node):
        """
        Swap out the Assert node (Python's `assert`) with a callsite to the
        symbolically-traceable torch._assert function
        """
        # Create the Call node
        n = ast.parse('torch._assert()', mode='eval')
        assert isinstance(n, ast.Expression)
        call_node = n.body
        assert isinstance(call_node, ast.Call)
        msg = node.msg if node.msg else ast.Constant(value="", kind=None)
        call_node.args = [node.test, msg]

        # Ensure that the new node conforms to the Python AST grammar
        expr_wrapper = ast.Expr(value=call_node)

        # Return the new Call node to signify that we want to use it as
        # a replacement for the original _assert node
        return ast.copy_location(expr_wrapper, node)


class AssertRewritingTracer(Tracer):
    def trace(self, root: Union[torch.nn.Module, Callable], concrete_args: Optional[Dict[str, Any]] = None) -> Graph:
        foo = _rewrite(root)
        return super().trace(foo, concrete_args)


def _rewrite(fn: Union[torch.nn.Module, Callable]) -> Union[torch.nn.Module, Callable]:

    if isinstance(fn, torch.nn.Module):


        def rewrite_module(m : torch.nn.Module):

            class RewrittenModule(torch.nn.Module):
                def __init__(self, orig):
                    super().__init__()

                    def is_helper_method(k: str) -> bool:

                        def is_dunder(name: str) -> bool:
                            return (len(name) > 4
                                    and name.startswith('__') 
                                    and name.endswith('__'))

                        o = getattr(orig, k)

                        if not callable(o):
                            return False

                        if is_dunder(k):
                            return False

                        if isinstance(o, torch.nn.Module):
                            return False

                        if k in dir(self):
                            return False

                        return True

                    # Copy over any potential helper methods
                    for k in dir(orig):
                        if is_helper_method(k):
                            self.__dict__[k] = getattr(orig, k)

                    for k, v in orig.__dict__.items():
                        if k == "_modules":
                            for mod_k, mod_v in v.items():
                                res = copy.copy(rewrite_module(mod_v))
                                self.add_module(mod_k, res)
                                self.__dict__[mod_k] = res
                        else:
                            self.__dict__[k] = copy.copy(v)
            RewrittenModule.forward = Assert_Rewriter().rewrite(cast(FunctionType, m.forward))
            return RewrittenModule(m)
        return rewrite_module(fn)
    else:
        # Rewrite this single free function
        return Assert_Rewriter().rewrite(cast(FunctionType, fn))
