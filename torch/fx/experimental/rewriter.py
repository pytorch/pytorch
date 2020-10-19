import ast
import inspect
import textwrap
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

        def remove_prefix(text, prefix):
            return text[text.startswith(prefix) and len(prefix):]

        # Get the source string
        sourcelines, _ = inspect.getsourcelines(fn)

        # Find the line and line number containing the function definition
        for i, l in enumerate(sourcelines):
            if l.lstrip()[:3] == "def":
                idx = i
                break
        fn_def = sourcelines[idx]

        # Get a string representing the amount of leading whitespace
        whitespace = fn_def.split("def")[0]

        # Add this leading whitespace to all lines before and after the `def`
        aligned_prefix = [whitespace + remove_prefix(s, whitespace) for s in sourcelines[:idx]]
        aligned_suffix = [whitespace + remove_prefix(s, whitespace) for s in sourcelines[idx + 1:]]

        # Put it together again
        aligned_prefix.append(fn_def)
        aligned_source = aligned_prefix + aligned_suffix

        # Remove common leading whitespace
        source = ''.join(aligned_source)
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
        # Create the Call node
        call_node = ast.parse('torch.Assert()', mode='eval').body
        msg = node.msg if node.msg else ast.Constant(value="", kind=None)
        call_node.args = [node.test, msg]

        # Ensure that the new node conforms to the Python AST grammar
        expr_wrapper = ast.Expr(value=call_node)

        # Return the new Call node to signify that we want to use it as
        # a replacement for the original Assert node
        return ast.copy_location(expr_wrapper, node)
