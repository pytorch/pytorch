import unittest
import re

import torch
from torch.testing._internal.jit_utils import JitTestCase

from torch.testing import FileCheck

class D(torch.nn.Module):
    def __init__(self, x):
        super(D, self).__init__()
        self.x = x

    def forward(self, y):
        return self.x + y

class B(torch.nn.Module):
    def __init__(self, x):
        super(B, self).__init__()
        self.d = D(x)

    def forward(self, y):
        return self.d(y)

    def named_method(self, y):
        return self.d(y)

class C(torch.nn.Module):
    def __init__(self, x):
        super(C, self).__init__()
        self.d = D(x)

    def forward(self, y):
        return self.d(y)

    def named_method(self, y):
        return self.d(y)

class A(torch.nn.Module):
    def __init__(self, x):
        super(A, self).__init__()
        self.b0 = B(x)
        self.b1 = B(x)
        self.c0 = C(x)
        self.c1 = C(x)
        self.x = x

    def forward(self, y):
        return self.b0(y) + self.b1(y) + self.c0(y) + self.c1(y)

    def named_method(self, y):
        return self.b0.named_method(y) + \
               self.b1.named_method(y) + \
               self.c0.named_method(y) + \
               self.c1.named_method(y) + \
               self.x

class TOP(torch.nn.Module):
    def __init__(self, x):
        super(TOP, self).__init__()
        self.a0 = A(x)
        self.a1 = A(x)

    def forward(self, y):
        return self.a0(y) + self.a1(y)

class TOPWithNamedMethods(torch.nn.Module):
    def __init__(self, x):
        super(TOPWithNamedMethods, self).__init__()
        self.a0 = A(x)
        self.a1 = A(x)

    def forward(self, y):
        return self.a0.named_method(y) + self.a1.named_method(y)

def func_foo(object: A, arg):
    return object.forward(arg)

class TOPWithFreeFunction(torch.nn.Module):
    def __init__(self, x):
        super(TOPWithFreeFunction, self).__init__()
        self.a0 = A(x)
        self.a1 = A(x)

    def forward(self, y):
        return func_foo(self.a0, y) + func_foo(self.a1, y)

def get_add_lines(graph):
    graph_str = str(graph)
    graph_str_lines = graph_str.splitlines()
    add_op_lines = []
    for line in graph_str_lines:
        if re.match('(.*)aten::add(.*)', line):
            add_op_lines.append(line)
    add_op_lines = '\n'.join(add_op_lines)
    return add_op_lines

class TestScopeWithInlinedCallSTack(JitTestCase):
    def test_scope_info(self):
        m = torch.jit.script(TOP(10))
        torch._C._jit_pass_inline(m.graph)
        torch._C._jit_pass_reconstruct_scope_from_inlined_callstack(m.graph)
        add_op_lines = get_add_lines(m.graph)
        FileCheck().check_count("aten::add(", 15, exactly=True).run(add_op_lines)
        # A does 3 adds and hence 4 operands.
        # Each operand of the add inturn does one add.
        FileCheck().check_count("A(a0)::forward", 7, exactly=True).run(add_op_lines)
        FileCheck().check_count("A(a1)::forward", 7, exactly=True).run(add_op_lines)
        # One b0 instance from a0, and one from a1
        FileCheck().check_count("B(b0)::forward", 2, exactly=True).run(add_op_lines)
        FileCheck().check_count("B(b1)::forward", 2, exactly=True).run(add_op_lines)
        # One c0 instance from a0, and one from a1
        FileCheck().check_count("C(c0)::forward", 2, exactly=True).run(add_op_lines)
        FileCheck().check_count("C(c1)::forward", 2, exactly=True).run(add_op_lines)
        # 4 d instances from a0, and 4 from a1
        FileCheck().check_count("D(d)::forward", 8, exactly=True).run(add_op_lines)

        m = torch.jit.trace(TOP(10), torch.zeros((1)))
        torch._C._jit_pass_inline(m.graph)
        torch._C._jit_pass_reconstruct_scope_from_inlined_callstack(m.graph)
        add_op_lines = get_add_lines(m.graph)
        FileCheck().check_count("aten::add(", 15, exactly=True).run(add_op_lines)
        # A does 3 adds and hence 4 operands.
        # Each operand of the add inturn does one add.
        FileCheck().check_count("A(a0)::forward", 7, exactly=True).run(add_op_lines)
        FileCheck().check_count("A(a1)::forward", 7, exactly=True).run(add_op_lines)
        # One b0 instance from a0, and one from a1
        FileCheck().check_count("B(b0)::forward", 2, exactly=True).run(add_op_lines)
        FileCheck().check_count("B(b1)::forward", 2, exactly=True).run(add_op_lines)
        # One c0 instance from a0, and one from a1
        FileCheck().check_count("C(c0)::forward", 2, exactly=True).run(add_op_lines)
        FileCheck().check_count("C(c1)::forward", 2, exactly=True).run(add_op_lines)
        # 4 d instances from a0, and 4 from a1
        FileCheck().check_count("D(d)::forward", 8, exactly=True).run(add_op_lines)

    def test_scope_info_with_named_method(self):
        m = torch.jit.script(TOPWithNamedMethods(10))
        torch._C._jit_pass_inline(m.graph)
        torch._C._jit_pass_reconstruct_scope_from_inlined_callstack(m.graph)
        add_op_lines = get_add_lines(m.graph)
        FileCheck().check_count("aten::add(", 17, exactly=True).run(add_op_lines)
        # A does 3 adds and hence 4 operands.
        # Each operand of the add inturn does one add.
        FileCheck().check_count("A(a0)::named_method", 8, exactly=True).run(add_op_lines)
        FileCheck().check_count("A(a1)::named_method", 8, exactly=True).run(add_op_lines)
        # One b0 instance from a0, and one from a1
        FileCheck().check_count("B(b0)::named_method", 2, exactly=True).run(add_op_lines)
        FileCheck().check_count("B(b1)::named_method", 2, exactly=True).run(add_op_lines)
        # One c0 instance from a0, and one from a1
        FileCheck().check_count("C(c0)::named_method", 2, exactly=True).run(add_op_lines)
        FileCheck().check_count("C(c1)::named_method", 2, exactly=True).run(add_op_lines)
        # 4 d instances from a0, and 4 from a1
        FileCheck().check_count("D(d)::forward", 8, exactly=True).run(add_op_lines)

        m = torch.jit.trace(TOPWithNamedMethods(10), torch.zeros((1)))
        torch._C._jit_pass_inline(m.graph)
        torch._C._jit_pass_reconstruct_scope_from_inlined_callstack(m.graph)
        add_op_lines = get_add_lines(m.graph)
        # Tracing seems to remove optimize and inline a bunch of stuff so that only
        # CallMethods that remain are for D module. Rest are inlined.
        # Note sure if checking for D is meanigful even, since that might be inlined
        # too. It is just so that right now it is not.
        FileCheck().check_count("aten::add(", 17, exactly=True).run(add_op_lines)
        FileCheck().check_count("D(d)::forward", 8, exactly=True).run(add_op_lines)

    def test_scope_info_with_free_function(self):
        m = torch.jit.script(TOPWithFreeFunction(10))
        torch._C._jit_pass_inline(m.graph)
        torch._C._jit_pass_reconstruct_scope_from_inlined_callstack(m.graph)
        add_op_lines = get_add_lines(m.graph)
        FileCheck().check_count("aten::add(", 15, exactly=True).run(add_op_lines)
        # Call func_foo method twice
        FileCheck().check_count("FreeFunction::func_foo", 14, exactly=True).run(add_op_lines)
        # A does 3 adds and hence 4 operands.
        # Each operand of the add inturn does one add.
        # Here instance name cannot be extracted for the two calls to func_foo
        FileCheck().check_count("A(INSTANCE_NAME_UNKNOWN)::forward", 14, exactly=True).run(add_op_lines)
        # One b0 instance from a0, and one from a1
        FileCheck().check_count("B(b0)::forward", 2, exactly=True).run(add_op_lines)
        FileCheck().check_count("B(b1)::forward", 2, exactly=True).run(add_op_lines)
        # One c0 instance from a0, and one from a1
        FileCheck().check_count("C(c0)::forward", 2, exactly=True).run(add_op_lines)
        FileCheck().check_count("C(c1)::forward", 2, exactly=True).run(add_op_lines)
        # 4 d instances from a0, and 4 from a1
        FileCheck().check_count("D(d)::forward", 8, exactly=True).run(add_op_lines)

        # Tracing is retaining only some modules and much is inlined.
        m = torch.jit.trace(TOPWithFreeFunction(10), torch.zeros((1)))
        torch._C._jit_pass_inline(m.graph)
        torch._C._jit_pass_reconstruct_scope_from_inlined_callstack(m.graph)
        add_op_lines = get_add_lines(m.graph)
        FileCheck().check_count("aten::add(", 15, exactly=True).run(add_op_lines)
        # func_foo is inlined during tracing and so is A's forward methods.
        # One b0 instance from a0, and one from a1
        FileCheck().check_count("B(b0)::forward", 2, exactly=True).run(add_op_lines)
        FileCheck().check_count("B(b1)::forward", 2, exactly=True).run(add_op_lines)
        # One c0 instance from a0, and one from a1
        FileCheck().check_count("C(c0)::forward", 2, exactly=True).run(add_op_lines)
        FileCheck().check_count("C(c1)::forward", 2, exactly=True).run(add_op_lines)
        # 4 d instances from a0, and 4 from a1
        FileCheck().check_count("D(d)::forward", 8, exactly=True).run(add_op_lines)

if __name__ == '__main__':
    raise RuntimeError("This test file is not meant to be run directly, use:\n\n"
                       "\tpython test/test_jit.py TESTNAME\n\n"
                       "instead.")