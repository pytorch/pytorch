import logging

from torch.testing._internal.inputgen.argument.engine import MetaArgEngine
from torch.testing._internal.inputgen.argument.gen import ArgumentGenerator
from torch.testing._internal.inputgen.attribute.model import Attribute
from torch.testing._internal.inputgen.specs.model import Spec


def reverse_topological_sort(graph):
    def dfs(node, visited, strack):
        visited[node] = True
        for neig in graph[node]:
            if not visited[neig]:
                dfs(neig, visited, strack)
        stack.append(node)

    visited = {node: False for node in graph}
    stack = []

    for node in graph:
        if not visited[node]:
            dfs(node, visited, stack)

    return stack


def inverse_permutation(permutation):
    n = len(permutation)
    inverse = [0] * n
    for i in range(n):
        inverse[permutation[i]] = i
    return inverse


class MetaArgTupleEngine:
    def __init__(self, spec: Spec, out: bool = False):
        if out:
            raise NotImplementedError("out=True is not supported yet")
        self.args = spec.inspec
        self.order = self._sort_dependencies()
        self.order_inverse_perm = inverse_permutation(self.order)

    def _generate_dependency_dag(self):
        graph = {}
        for i, arg in enumerate(self.args):
            if arg.deps is None:
                graph[i] = []
            else:
                graph[i] = arg.deps
        return graph

    def _sort_dependencies(self):
        graph = self._generate_dependency_dag()
        return reverse_topological_sort(graph)

    def _sort_meta_tuple(self, meta_tuple):
        return tuple(
            meta_tuple[self.order_inverse_perm[i]] for i in range(len(self.args))
        )

    def _get_deps(self, meta_tuple, arg_deps):
        value_tuple = tuple(ArgumentGenerator(m).gen() for m in meta_tuple)
        return tuple(value_tuple[self.order_inverse_perm[ix]] for ix in arg_deps)

    def gen_meta_tuples(self, valid: bool, focus_ix: int):
        logging.debug(f"Generating tuples focusing on argument {focus_ix}")
        tuples = [()]
        for ix in self.order:
            arg = self.args[ix]
            logging.debug(f"Generating argument {ix} ({arg.type})")
            new_tuples = []
            focuses = [None]
            if ix == focus_ix:
                focuses = Attribute.hierarchy(arg.type)
            for focus in focuses:
                for meta_tuple in tuples:
                    deps = self._get_deps(meta_tuple, arg.deps)
                    engine = MetaArgEngine(arg.type, arg.constraints, deps, valid)
                    for meta_arg in engine.gen(focus):
                        new_tuples.append(meta_tuple + (meta_arg,))
            tuples = new_tuples
        return map(self._sort_meta_tuple, tuples)

    def gen_valid_meta_tuples(self):
        valid_tuples = []
        for ix in range(len(self.args)):
            valid_tuples += self.gen_meta_tuples(True, ix)
        return valid_tuples

    def gen_invalid_from_valid(self, valid_tuple):
        logging.debug("Valid", [str(x) for x in valid_tuple])
        valid_value_tuple = tuple(ArgumentGenerator(m).gen() for m in valid_tuple)
        invalid_tuples = []
        for ix in range(len(self.args)):
            arg = self.args[ix]
            logging.debug(f"Generating invalid argument {ix} ({arg.type})")
            deps = tuple(valid_value_tuple[i] for i in arg.deps)
            for focus in Attribute.hierarchy(arg.type):
                engine = MetaArgEngine(arg.type, arg.constraints, deps, False)
                for meta_arg in engine.gen(focus):
                    invalid_tuple = (
                        valid_tuple[:ix] + (meta_arg,) + valid_tuple[ix + 1 :]
                    )
                    logging.debug(
                        f"  Invalid {ix} {focus}:", [str(x) for x in invalid_tuple]
                    )
                    invalid_tuples.append(invalid_tuple)
        invalid_tuples = list(set(invalid_tuples))
        return invalid_tuples

    def gen_invalid_meta_tuples(self):
        valid_tuples = self.gen_valid_meta_tuples()
        invalid_tuples = []
        for valid_tuple in valid_tuples:
            invalids = self.gen_invalid_from_valid(valid_tuple)
            invalid_tuples += invalids
        invalid_tuples = list(set(invalid_tuples))
        return invalid_tuples

    def gen(self, valid: bool = True):
        if valid:
            return self.gen_valid_meta_tuples()
        else:
            return self.gen_invalid_meta_tuples()
