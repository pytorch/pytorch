from typing import List, Dict, Set
import torch
import dataclasses
from contextlib import contextmanager

"""
Parent structure for guard env expressions.

A GuardEnvExpr can have any subtype.

Note: All subtypes must be handled exhaustively in
torch._dynamo.guards._parse_guard_env_guards to avoid a RuntimeError.
"""
@dataclasses.dataclass
class GuardEnvExpr():
    pass

"""
A class representing a pair of duplicate inputs.

arg_a and arg_b are argument names in the traced scope.
"""
@dataclasses.dataclass
class DuplicateInputs(GuardEnvExpr):
    arg_a: str
    arg_b: str


"""
A GuardEnv is philosophically equivalent to a SymbolicShapeEnv in that it:

1) Accumulates behaviors (ex: register_duplicates)
2) Produces internal guard representations
3) Can be consumed by guarding systems to create guards

It differs significantly in how it represents guards.

See: https://docs.google.com/document/d/1VbiXkIhKy744Q7Lx2e8UUBvP_5RH_WU0FEc3VadVX4U/edit?usp=sharing for
a document explaining this structure in greater detail.

Note - while this feels like it should live in dynamo's guards.py, it feels better from a layering perspective.
To have it here, especially as aot_autograd knows about this and registers new guards here.
"""
class GuardEnv:
    _guards : List[GuardEnvExpr] = []
    _tensor_to_names : Dict[torch.Tensor, Set[str]] = {}

    def register_duplicates(self, dupe_arg: torch.Tensor, kept_arg: torch.Tensor):
        # Note: This is a little onerous - one could imagine that registration implies assoication.
        # HOWEVER - I loathe to allow this to be easier, as we want the dynamo layer to associate and
        # lower levels to register things with associated tensors. If we ever end up registering something
        # that does not have prior knowlege in association... we just re-invented a new flavor of the
        # symbolic-expression-not-found problem, and I would rather we not do that, or at least, identify it
        # where we make the guard, as opposed to where we go to process it when we call .guards().
        assert dupe_arg in self._tensor_to_names, "Tensor not found - did you forget to call .associate()?"

        assert dupe_arg is kept_arg, "Register_duplicates args must pass identity check."

        names_for_dupe = list(self._tensor_to_names[dupe_arg].keys())

        traverse_len = len(names_for_dupe)

        for i in range(0, traverse_len - 1):
            name_a = names_for_dupe[i]
            name_b = names_for_dupe[i + 1]
            dupe_inputs = DuplicateInputs(arg_a=name_a, arg_b=name_b)
            self._guards.append(dupe_inputs)

    def associate(self, tensor: torch.Tensor, name: str):
        if tensor not in self._tensor_to_names:
            # Note: Dict for determinism
            self._tensor_to_names[tensor] = dict()

        self._tensor_to_names[tensor].update({name: None})

    # This is a func because we don't want users just reading ._guards for now
    # we may do simplification or other things here.
    def get_guards(self) -> List[GuardEnvExpr]:
        return self._guards

    def clear(self):
        self._guards.clear()
        self._tensor_to_names.clear()

CURRENT_GUARD_ENV = None

@contextmanager
def guarding(guard_env):
    global CURRENT_GUARD_ENV
    old_guard_env = CURRENT_GUARD_ENV
    CURRENT_GUARD_ENV = guard_env
    try:
        yield CURRENT_GUARD_ENV
    finally:
        CURRENT_GUARD_ENV.clear()
        CURRENT_GUARD_ENV = old_guard_env
