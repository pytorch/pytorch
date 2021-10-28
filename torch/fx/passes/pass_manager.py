from typing import Any, Callable, List, Protocol, Type, TypeVar
import torch.nn as nn

T1 = TypeVar('T1', bound=nn.Module)
T2 = TypeVar('T2', bound=nn.Module)

class Pass(Protocol[T1, T2]):
    def __call__(self, g: T1) -> T2:
        ...


# for callables which modify object inplace and return something other than
# the object on which they act
def inplace_wrapper(fn: Callable[[T1], Any]) -> Pass:
    def wrapped_fn(gm: T1) -> T1:
        fn(gm)
        return gm

    return wrapped_fn


class LoopPass(Pass):
    def __init__(self, _pass: Pass, n_iter: int = None, predicate: Callable = None):
        self._pass = _pass
        self._n_iter = n_iter
        self._predicate = predicate
        assert n_iter is None or predicate is None

    def __call__(self, source):
        if self._n_iter:
            for _ in range(self._n_iter):
                self._pass(source)
        else:
            while self._predicate(self._pass(source)):
                continue
        return source


# Constraints are implemented as 'less than' operators. A constraint is
# satisfied iff a list has a valid partial ordering according to this
# comparison operator.
class PassScheduleConstraint:
    def lt(self, a, b):
        raise NotImplementedError()

    def validate(self, passes: List[Pass]):
        for i, a in enumerate(passes):
            for j, b in enumerate(passes[i + 1 :]):
                if self.lt(a, b):
                    continue
                raise RuntimeError(
                    f"PassScheduleConstraint {self} violated. Expected {a} before {b}"
                    f" but found {a} at index {i} and {b} at index{j} in pass"
                    f" list."
                )


class ThisBeforeThatConstraint(PassScheduleConstraint):
    def __init__(self, this: Pass, that: Pass):
        self._this = this
        self._that = that

    def lt(self, a, b):
        if a == self._that and b == self._this:
            return False
        return True


class TheseBeforeThoseConstraint(PassScheduleConstraint):
    def __init__(self, these: Type[Pass], those: Type[Pass]):
        self._these = these
        self._those = those

    def lt(self, a, b):
        if isinstance(a, self._those) and isinstance(b, self._these):
            return False
        return True


class PassManager(Pass):
    def __init__(self, passes: List[Pass] = None, constraints: List[Callable] = None):
        self._passes = passes or []
        self._constraints = constraints or []
        self._validated = False

    def add_pass(self, _pass: Pass):
        self._passes.append(_pass)
        self._validated = False

    def add_constraint(self, constraint):
        self._constraints.append(constraint)
        self._validated = False

    def validate(self):
        if self._validated:
            return
        for constraint in self._constraints:
            constraint.validate(self._passes)
        self._validated = True

    def __call__(self, source):
        self.validate()
        out = source
        for _pass in self._passes:
            out = _pass(out)
        return out


class PassManagerBuilder:
    @classmethod
    def build_from_passlist(cls, passes: List[Pass]) -> PassManager:
        pm = PassManager()

        # Add passes
        for _pass in passes:
            pm.add_pass(_pass)

        # TODO(@alexbeloi): Add constraints to preserve given order
        # TODO(@alexbeloi): Add validation

        return pm
