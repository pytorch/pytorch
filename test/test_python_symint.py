from typing import Iterator, List
import torch
from enum import Enum
import torch
#from base_tensor import BaseTensor
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.utils._pytree import tree_map
import contextlib

@contextlib.contextmanager
def no_dispatch() -> Iterator[None]:
    guard = torch._C._DisableTorchDispatch()  # type: ignore[attr-defined]
    try:
        yield
    finally:
        del guard

class Op(Enum):
    SIZE = 1
    ADD = 2
    MAX = 3

class AOTAutogradSymInt:
    def __init__(self, op: Op, children: List["AOTAutogradSymInt"]):
        self._op = op
        self._children = children

    def __str__(self):
        return f"{self._op}"

    def __add__(self, other):
        return AOTAutogradSymInt(Op.ADD, [self, other])


a = torch.rand(2, 2)

sym1 = torch._C.SymbolicIntNode.new_symint(AOTAutogradSymInt(Op.SIZE, [a, 0]))
print(torch._C.SymbolicIntNode.isinstance(1, False))
print(torch._C.SymbolicIntNode.types(sym1))
sym2 = torch._C.SymbolicIntNode.new_symint(AOTAutogradSymInt(Op.SIZE, [a, 0]))

tid = 0
def get_next_id():
    global tid
    tid += 1
    return tid 

class EmptyTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, elem):
        tid = get_next_id()
        symints = [torch._C.SymbolicIntNode.new_symint(AOTAutogradSymInt(Op.SIZE, [tid, i])) for i in range(len(elem.size()))]
        return torch.Tensor._make_wrapper_subclass(
            cls, symints,
            strides=elem.stride(), storage_offset=elem.storage_offset(),
            dtype=elem.dtype, layout=elem.layout, requires_grad=elem.requires_grad,
            device=elem.device
        )

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        print(func.__name__)
        if func.__name__ == 'add.Tensor':
            print('inside add.Tensor')
            # we assume that shapes are broadcastable
            new_dims = [torch._C.SymbolicIntNode.new_symint(AOTAutogradSymInt(Op.MAX, [a, b])) for (a, b) in zip(args[0].size(), args[1].size())]
            strides = [0 for x in range(len(new_dims))]
            try:
                t = torch.Tensor._make_wrapper_subclass(
                    EmptyTensor, new_dims,
                    strides=strides, storage_offset=0, # symbolic offset yet
                    dtype=a.dtype, layout=a.layout, requires_grad=a.requires_grad,
                    device=a.device)
                return t
            except RuntimeError as e:
                print(e)
        else:
            raise NotImplementedError("NYI")

    def __init__(self, elem):
        pass

    def __repr__(self):
        return f'EmptyTensor({self.size()})'

a = torch.rand(4, 5)
print(a.size())
x = EmptyTensor(torch.randn(4, 5))
print(x.size()[0])
print(x.size()[1])
y = EmptyTensor(torch.randn(4, 5))
c = x + y
print(c.size())

