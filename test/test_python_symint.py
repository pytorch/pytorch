from typing import List
import torch
from enum import Enum

class Op(Enum):
    SIZE = 1
    ADD = 2

class AOTAutogradSymInt:
    def __init__(self, op: Op, children: List["AOTAutogradSymInt"]):
        self._op = op
        self._children = children

    def __add__(self, other):
        return AOTAutogradSymInt(Op.ADD, [self, other])


a = torch.rand(2, 2)

a_size0 = AOTAutogradSymInt(Op.SIZE, [a, 0])
a_size1 = AOTAutogradSymInt(Op.ADD, [a, 1])


ps0 = torch._C.PythonSymbolicIntNode(a_size0)
ps1 = torch._C.PythonSymbolicIntNode(a_size1)
ps_add = ps0 + ps1
a_add = ps_add.pyobj()
assert(type(a_add) == AOTAutogradSymInt)
#print(type(a_add))
assert(a_add._op == Op.ADD)
assert(a_add._children == [a_size0, a_size1])

