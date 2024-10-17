# Owner(s): ["module: dynamo"]
from typing import Callable, Dict, List, NamedTuple, Optional

import torch
import torch._dynamo
from torch._dynamo.test_case import run_tests, TestCase
from torch._dynamo.testing import CompileCounter, same


"""
This is an example of a pure-python version of autograd implemented by
@zdevito.  It represents a rather challenging test case for TorchDynamo
to push the limits of what it can do.
"""


_name: int = 0


def fresh_name() -> str:
    """create a new unique name for a variable: v0, v1, v2"""
    global _name
    r = f"v{_name}"
    _name += 1
    return r


class Variable:
    def __init__(self, value: torch.Tensor, name: str = None):
        self.value = value
        self.name = name or fresh_name()

    # We need to start with some tensors whose values were not computed
    # inside the autograd. This function constructs leaf nodes.
    @staticmethod
    def constant(value: torch.Tensor, name: str = None):
        return Variable(value, name)

    def __repr__(self):
        return repr(self.value)

    # This performs a pointwise multiplication of a Variable, tracking gradients
    def __mul__(self, rhs: "Variable") -> "Variable":
        # defined later in the notebook
        return operator_mul(self, rhs)

    def __add__(self, rhs: "Variable") -> "Variable":
        return operator_add(self, rhs)

    def sum(self, name: Optional[str] = None) -> "Variable":
        return operator_sum(self, name)

    def expand(self, sizes: List[int]) -> "Variable":
        return operator_expand(self, sizes)


class TapeEntry(NamedTuple):
    # names of the inputs to the original computation
    inputs: List[str]
    # names of the outputs of the original computation
    outputs: List[str]
    # apply chain rule
    propagate: "Callable[List[Variable], List[Variable]]"


gradient_tape: List[TapeEntry] = []


def reset_tape():
    gradient_tape.clear()
    global _name
    _name = 0


def grad(L, desired_results: List[Variable]) -> List[Variable]:
    # this map holds dL/dX for all values X
    dL_d: Dict[str, Variable] = {}
    # It starts by initializing the 'seed' dL/dL, which is 1
    dL_d[L.name] = Variable(torch.ones(()))
    # print(f'd{L.name} ------------------------')

    # look up dL_dentries. If a variable is never used to compute the loss,
    # we consider its gradient None, see the note below about zeros for more information.
    def gather_grad(entries: List[str]):
        return [dL_d[entry] if entry in dL_d else None for entry in entries]

    # propagate the gradient information backward
    for entry in reversed(gradient_tape):
        dL_doutputs = gather_grad(entry.outputs)
        if all(dL_doutput is None for dL_doutput in dL_doutputs):
            # optimize for the case where some gradient pathways are zero. See
            # The note below for more details.
            continue

        # perform chain rule propagation specific to each compute
        dL_dinputs = entry.propagate(dL_doutputs)

        # Accumulate the gradient produced for each input.
        # Each use of a variable produces some gradient dL_dinput for that
        # use. The multivariate chain rule tells us it is safe to sum
        # all the contributions together.
        for input, dL_dinput in zip(entry.inputs, dL_dinputs):
            if input not in dL_d:
                dL_d[input] = dL_dinput
            else:
                dL_d[input].value += dL_dinput.value

    # print some information to understand the values of each intermediate
    # for name, value in dL_d.items():
    #    print(f'd{L.name}_d{name} = {value.name}')
    # print(f'------------------------')

    return gather_grad(desired.name for desired in desired_results)


def operator_mul(self: Variable, rhs: Variable) -> Variable:
    if isinstance(rhs, float) and rhs == 1.0:
        # peephole optimization
        return self

    # define forward
    r = Variable(self.value * rhs.value)
    # print(f'{r.name} = {self.name} * {rhs.name}')

    # record what the inputs and outputs of the op were
    inputs = [self.name, rhs.name]
    outputs = [r.name]

    # define backprop
    def propagate(dL_doutputs: List[Variable]):
        (dL_dr,) = dL_doutputs

        dr_dself = rhs  # partial derivative of r = self*rhs
        dr_drhs = self  # partial derivative of r = self*rhs

        # chain rule propagation from outputs to inputs of multiply
        dL_dself = dL_dr * dr_dself
        dL_drhs = dL_dr * dr_drhs
        dL_dinputs = [dL_dself, dL_drhs]
        return dL_dinputs

    # finally, we record the compute we did on the tape
    gradient_tape.append(TapeEntry(inputs=inputs, outputs=outputs, propagate=propagate))
    return r


def operator_add(self: Variable, rhs: Variable) -> Variable:
    # Add follows a similar pattern to Mul, but it doesn't end up
    # capturing any variables.
    r = Variable(self.value + rhs.value)
    # print(f'{r.name} = {self.name} + {rhs.name}')

    def propagate(dL_doutputs: List[Variable]):
        (dL_dr,) = dL_doutputs
        dr_dself = 1.0
        dr_drhs = 1.0
        dL_dself = dL_dr * dr_dself
        dL_drhs = dL_dr * dr_drhs
        return [dL_dself, dL_drhs]

    gradient_tape.append(
        TapeEntry(inputs=[self.name, rhs.name], outputs=[r.name], propagate=propagate)
    )
    return r


def operator_sum(self: Variable, name: Optional[str]) -> "Variable":
    r = Variable(torch.sum(self.value), name=name)
    # print(f'{r.name} = {self.name}.sum()')

    def propagate(dL_doutputs: List[Variable]):
        (dL_dr,) = dL_doutputs
        size = self.value.size()
        return [dL_dr.expand(*size)]

    gradient_tape.append(
        TapeEntry(inputs=[self.name], outputs=[r.name], propagate=propagate)
    )
    return r


def operator_expand(self: Variable, sizes: List[int]) -> "Variable":
    assert self.value.dim() == 0  # only works for scalars
    r = Variable(self.value.expand(sizes))
    # print(f'{r.name} = {self.name}.expand({sizes})')

    def propagate(dL_doutputs: List[Variable]):
        (dL_dr,) = dL_doutputs
        return [dL_dr.sum()]

    gradient_tape.append(
        TapeEntry(inputs=[self.name], outputs=[r.name], propagate=propagate)
    )
    return r


def simple(a, b):
    t = a + b
    return t * b


class TestPythonAutograd(TestCase):
    def _common(self, fn, expected_ops):
        args1 = [torch.randn(10), torch.randn(10)]
        args2 = [torch.randn(10), torch.randn(10)]
        cnt = CompileCounter()
        fn_dynamo = torch._dynamo.optimize_assert(cnt)(fn)
        reset_tape()
        res1 = fn_dynamo(*args1)
        reset_tape()
        res2 = fn_dynamo(*args2)
        reset_tape()
        self.assertTrue(same(res1, fn(*args1)))
        reset_tape()
        self.assertTrue(same(res2, fn(*args2)))
        reset_tape()
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(cnt.op_count, expected_ops)

    def test_forwards1(self):
        def fn(a, b):
            a = Variable.constant(a, name="a")
            b = Variable.constant(b, name="b")
            loss = simple(a, b).sum()
            return loss

        self._common(fn, 3)

    def test_forwards2(self):
        def fn(a, b):
            reset_tape()
            a = Variable.constant(a, name="a")
            b = Variable.constant(b, name="b")
            loss = simple(a, b).sum()
            reset_tape()
            return loss

        self._common(fn, 3)

    def test_backwards1(self):
        def fn(a, b):
            a = Variable.constant(a, name="a")
            b = Variable.constant(b, name="b")
            loss = simple(a, b).sum()
            return grad(loss, [a, b])

        self._common(fn, 8)

    def test_backwards2(self):
        def fn(a, b):
            reset_tape()
            a = Variable.constant(a, name="a")
            b = Variable.constant(b, name="b")
            loss = simple(a, b).sum()
            res = grad(loss, [a, b])
            reset_tape()
            return res

        self._common(fn, 8)

    def test_split(self):
        v1 = Variable.constant(torch.randn(10), name="a")
        v2 = Variable.constant(torch.randn(10), name="b")
        cnt = CompileCounter()

        def forward(a, b):
            return simple(a, b).sum()

        reset_tape()
        loss1 = forward(v1, v2)
        grad1 = grad(loss1, [v1, v2])

        reset_tape()
        opt_forward = torch._dynamo.optimize_assert(cnt)(forward)
        opt_grad = torch._dynamo.optimize_assert(cnt)(grad)
        loss2 = opt_forward(v1, v2)
        # force two frames
        grad2 = opt_grad(loss2, [v1, v2])

        self.assertTrue(same(loss1, loss2))
        self.assertTrue(same(grad1, grad2))
        self.assertEqual(cnt.frame_count, 2)
        self.assertEqual(cnt.op_count, 8)


if __name__ == "__main__":
    run_tests()
