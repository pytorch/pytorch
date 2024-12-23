"""
This module implements the functionality to take any Python expression as a
string and fix all numbers and other things before evaluating it,
thus

1/2

returns

Integer(1)/Integer(2)

We use the ast module for this. It is well documented at docs.python.org.

Some tips to understand how this works: use dump() to get a nice
representation of any node. Then write a string of what you want to get,
e.g. "Integer(1)", parse it, dump it and you'll see that you need to do
"Call(Name('Integer', Load()), [node], [], None, None)". You do not need
to bother with lineno and col_offset, just call fix_missing_locations()
before returning the node.
"""

from sympy.core.basic import Basic
from sympy.core.sympify import SympifyError

from ast import parse, NodeTransformer, Call, Name, Load, \
    fix_missing_locations, Constant, Tuple

class Transform(NodeTransformer):

    def __init__(self, local_dict, global_dict):
        NodeTransformer.__init__(self)
        self.local_dict = local_dict
        self.global_dict = global_dict

    def visit_Constant(self, node):
        if isinstance(node.value, int):
            return fix_missing_locations(Call(func=Name('Integer', Load()),
                    args=[node], keywords=[]))
        elif isinstance(node.value, float):
            return fix_missing_locations(Call(func=Name('Float', Load()),
                    args=[node], keywords=[]))
        return node

    def visit_Name(self, node):
        if node.id in self.local_dict:
            return node
        elif node.id in self.global_dict:
            name_obj = self.global_dict[node.id]

            if isinstance(name_obj, (Basic, type)) or callable(name_obj):
                return node
        elif node.id in ['True', 'False']:
            return node
        return fix_missing_locations(Call(func=Name('Symbol', Load()),
                args=[Constant(node.id)], keywords=[]))

    def visit_Lambda(self, node):
        args = [self.visit(arg) for arg in node.args.args]
        body = self.visit(node.body)
        n = Call(func=Name('Lambda', Load()),
            args=[Tuple(args, Load()), body], keywords=[])
        return fix_missing_locations(n)

def parse_expr(s, local_dict):
    """
    Converts the string "s" to a SymPy expression, in local_dict.

    It converts all numbers to Integers before feeding it to Python and
    automatically creates Symbols.
    """
    global_dict = {}
    exec('from sympy import *', global_dict)
    try:
        a = parse(s.strip(), mode="eval")
    except SyntaxError:
        raise SympifyError("Cannot parse %s." % repr(s))
    a = Transform(local_dict, global_dict).visit(a)
    e = compile(a, "<string>", "eval")
    return eval(e, global_dict, local_dict)
