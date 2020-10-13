#so like you would go from (forward function Python object) -> (forward function source code (from inspect module)) -> AST -> modified AST -> (forward function Python Object)

import ast
import astpretty
import astor
import inspect
import textwrap
import torch
from types import FunctionType

class AST_Rewriter(ast.NodeTransformer):
    '''
    Takes a FunctionType object representing a `forward` method, then
    performs an AST rewrite to swap out nodes that are not symbolically
    traceable with a callsite to the FX alternative.

    To support swapping out a node, simply create a new `visit_<NODE>`
    function, where <NODE> is the name of the AST node you wish to swap.
    '''

    def rewrite(self, fn: FunctionType):
        #TODO: annotate this with the right type (<class 'code'>? how to express?)
        source_str = textwrap.dedent(inspect.getsource(fn))
        source_ast = ast.parse(source_str)
        dest_ast = ast.fix_missing_locations(self.visit(source_ast))
        dest_str = astor.to_source(dest_ast)
        #TODO: We could use `dest_str = ast.unparse(dest_ast)` in Python 3.9
        return compile(dest_str, '', 'exec')

    def visit_Assert(self, node):
        # Create the CALL node
        call_node_args = [node.test]
        if node.msg:
            call_node_args.insert(0, node.msg)
        call_node = ast.Call(
            func=ast.Attribute(value=ast.Name(id='torch', ctx=ast.Load()),
            attr='Assert', ctx=ast.Load()),
            args=call_node_args,
            keywords=[])
        return ast.copy_location(call_node, node)
