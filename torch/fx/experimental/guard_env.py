from typing import List, Dict, Set
import torch
import dataclasses

class GuardEnvExpr():
    pass

@dataclasses.dataclass
class DuplicateInputs(GuardEnvExpr):
    arg_a: str
    arg_b: str

class GuardEnv:
    guards : List[GuardEnvExpr] = []
    fake_tensor_to_names : Dict[torch._subclasses.FakeTensor, Set[str]] = {}

    def register_duplicates(self, dupe_arg: torch._subclasses.FakeTensor, kept_arg: torch._subclasses.FakeTensor):
        names_for_dupe = self.fake_tensor_to_names[dupe_arg]
        names_for_kept = self.fake_tensor_to_names[kept_arg]

        # Instead of a quadratic association of every-equal-to-every, we can chain
        all_names = list(names_for_dupe.union(names_for_kept))

        # Two iter does not cover this, edge case
        for i in range(0, len(all_names), 2):
            name_a = all_names[i]
            name_b = all_names[i + 1]
            dupe_inputs = DuplicateInputs(arg_a=name_a, arg_b=name_b)
            self.guards.append(dupe_inputs)

    def associate(self, fake_tensor: torch._subclasses.FakeTensor, name: str):
        if fake_tensor not in self.fake_tensor_to_names:
            self.fake_tensor_to_names[fake_tensor] = set()

        self.fake_tensor_to_names[fake_tensor].add(name)

    # This is a func because we don't want users just reading .guards for now
    # we may do simplification or other things here.
    def get_guards(self):
        return self.guards

    def clear(self):
        self.guards = []
        self.fake_tensor_to_names = {}

# TODO(voz): This is super lame, but keeping it global for the sake of the prototype
# allows me to punt dealing with kwargs, piping stuff around backends, etc
# We may or may not want to land like this, TBD, but for now it allows me to focus
# on implementation. We got away with reading the fake_mode off of tensors, so maybe we
# will do the same here. Maybe not.
GUARD_ENV = GuardEnv()
