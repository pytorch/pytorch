import ast
import astor
import inspect
import textwrap
import torch
import copy
from types import FunctionType

class AST_Rewriter(ast.NodeTransformer):
    '''
    Take a FunctionType object representing a `forward` method, then
    perform an AST rewrite to swap out nodes that are not symbolically
    traceable with a callsite to the FX alternative.

    To support swapping out an AST node, define a new `visit` method on
    that node. For more details, see:
    https://docs.python.org/3/library/ast.html#ast.NodeTransformer
    '''

    def rewrite(self, fn: FunctionType):

        # Get the source string, align everything with the `def`, then
        # remote unnecessary indentation
        source_str = inspect.getsource(fn)
        whitespace_prefix = source_str.split("def")[0]
        split_source = source_str.split('\n')
        aligned_source = [whitespace_prefix+s for s in split_source[1:]]
        aligned_source.insert(0, split_source[0])
        aligned_str = '\n'.join(aligned_source)
        normalized_str = textwrap.dedent(aligned_str)

        # Rewrite the original AST
        source_ast = ast.parse(normalized_str)
        dest_ast = ast.fix_missing_locations(self.visit(source_ast))
        dest_str = astor.to_source(dest_ast)

        # Pull out the compiled fucntion from the newly-created Module
        code = compile(dest_ast, "", "exec")
        globals_dict = {}
        globals_dict = copy.copy(fn.__globals__)
        exec(code, globals_dict)
        fn_compiled = globals_dict["forward"]

        # Return the correct FunctionType object
        return fn_compiled

    def visit_Assert(self, node):
        # Create the Call node
        call_node_args = [node.test]
        if node.msg:
            call_node_args.append(node.msg)
        else:
            call_node_args.append(ast.Constant(value="", kind=None))
        call_node = ast.Call(
            func=ast.Attribute(value=ast.Name(id="torch", ctx=ast.Load()),
            attr="Assert", ctx=ast.Load()),
            args=call_node_args,
            keywords=[])

        # Workaround for astor
        expr_wrapper = ast.Expr(value=call_node)

        # Return the new Call node to signify that we want to use it as
        # a replacement for the original Assert node
        return ast.copy_location(expr_wrapper, node)
