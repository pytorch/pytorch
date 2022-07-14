import inspect
from queue import Queue
from functools import wraps
from typing import Callable, Dict, List
import warnings

import torch
import torch.nn as nn
from torch.fx.graph_module import GraphModule
from torch.fx.node import Argument, map_aggregate
from torch.fx._compatibility import compatibility
from torch.fx.passes.infra.pass_base import PassResult

__all__ = ['inplace_wrapper', 'pass_result_wrapper', 'this_before_that_pass_constraint', 'PassManager']

@compatibility(is_backward_compatible=False)
def inplace_wrapper(fn: Callable) -> Callable:
    """
    Convenience wrapper for passes which modify an object inplace. This
    wrapper makes them return a PassResult containing the modified object and
    True for the "modified" flag.

    Args:
        fn (Callable[Module, Any])

    Returns:
        wrapped_fn (Callable[Module, PassResult])
    """
    if fn is None:
        return None

    @wraps(fn)
    def wrapped_fn(gm):
        fn(gm)
        return PassResult(gm, True)

    return wrapped_fn

@compatibility(is_backward_compatible=False)
def pass_result_wrapper(fn: Callable) -> Callable:
    """
    Wrapper for passes which currently do not return a PassResult.
    This wrapper makes them return a PassResult containing the modified object
    and True for the "modified" flag.

    Args:
        fn (Callable[Module, Any])

    Returns:
        wrapped_fn (Callable[Module, PassResult])
    """
    if fn is None:
        return None

    @wraps(fn)
    def wrapped_fn(gm):
        gm = fn(gm)
        return PassResult(gm, True)

    return wrapped_fn

def _validate_pass_schedule_constraint(
    constraint: Callable[[Callable, Callable], bool], passes: List[Callable]
) -> None:
    for i, a in enumerate(passes):
        for j, b in enumerate(passes[i + 1 :]):
            if constraint(a, b):
                continue
            raise RuntimeError(
                f"pass schedule constraint violated. Expected {a} before {b}"
                f" but found {a} at index {i} and {b} at index{j} in pass"
                f" list."
            )

def _topological_sort_passes(
    passes: List[Callable], constraints: List[Callable]
) -> List[Callable]:
    """
    Args
        passes: Passes that we are ordering
        constraints: Constraints applied on these passes

    Returns
        A sorted list of callables and a boolean of if a circular dependency
        existed
    """
    if len(constraints) == 0:
        return passes

    # Contruct a graph mapping nodes to a list of their users
    graph: Dict[Callable, List[Callable]] = {p : [] for p in passes}
    indegree_map: Dict[Callable, int] = {p : 0 for p in passes}
    candidates: Queue = Queue()
    for a in passes:
        for b in passes:
            if a == b:
                continue

            for constraint in constraints:
                if not constraint(a, b):
                    graph[b].append(a)
                    indegree_map[a] += 1

        if indegree_map[a] == 0:
            candidates.put(a)

    visited: Dict[Callable, bool] = {p : False for p in passes}
    sorted_passes: List[Callable] = []

    while not candidates.empty():
        p = candidates.get()
        sorted_passes.append(p)
        visited[p] = True

        for n in graph[p]:
            if not visited[n]:
                indegree_map[n] -= 1
                if indegree_map[n] == 0:
                    candidates.put(n)

    # Check if there are unvisited nodes (aka cycles in the graph)
    cycle_passes = list(filter(lambda p: indegree_map[p] != 0, indegree_map.keys()))
    if len(cycle_passes) != 0:
        error = f"Circular dependency detected within the following passes: {cycle_passes}"
        raise RuntimeError(error)

    return sorted_passes

@compatibility(is_backward_compatible=False)
def this_before_that_pass_constraint(this: Callable, that: Callable) -> Callable:
    """
    Defines a partial order ('depends on' function) where `this` must occur
    before `that`.

    For example, the following pass list and constraint list would be invalid.
    ```
    passes = [pass_b, pass_a]

    constraints = [
        this_before_that_pass_constraint(pass_a, pass_b)
    ]
    ```

    Args:
        this (Callable): pass which should occur first
        that (Callable): pass which should occur later

    Returns:
        depends_on (Callable[[Object, Object], bool]
    """

    def depends_on(a: Callable, b: Callable):
        if a == that and b == this:
            return False
        return True

    return depends_on


@compatibility(is_backward_compatible=False)
class PassManager:
    """
    Construct a PassManager.

    Collects passes and constraints. This defines the pass schedule, manages
    pass constraints and pass execution.

    Args:
        passes (Optional[List[Callable]]): List of passes. A pass is a
            callable which modifies an object and returns a PassResult
        constraint (Optional[List[Callable]]): List of constraints. A
            constraint is a callable which takes two passes (A, B) and returns
            True if A depends on B and False otherwise. See implementation of
            `this_before_that_pass_constraint` for example.
        steps (int): Max number of times we run the passes (default = 1).
        run_checks_after_each_pass (bool): Whether to run checks and linting
            after each pass
    """

    passes: List[Callable[[nn.Module], PassResult]] = []
    constraints: List[Callable[[Callable, Callable], bool]] = []
    _validated: bool = False
    steps: int = 1

    def __init__(
        self,
        passes=None,
        constraints=None,
        steps=None,
        run_checks_after_each_pass: bool = False,
        suppress_check_failures: bool = False,
    ):
        if passes:
            self.passes = passes
        if constraints:
            self.constraints = constraints
        if steps:
            self.steps = steps

        self.run_checks_after_each_pass = run_checks_after_each_pass
        self.suppress_check_failures = suppress_check_failures

    def add_pass(self, _pass: Callable):
        """
        Adds a pass into the current list of passes.
        """
        self.passes.append(_pass)
        self._validated = False

    def add_constraint(self, constraint: Callable):
        """
        Adds a constraint into the current list of constraints.
        """
        self.constraints.append(constraint)
        self._validated = False

    def validate_constraints(self):
        """
        Validates that current pass schedule defined by `self.passes` is valid
        according to all constraints in `self.constraints`
        """
        if self._validated:
            return
        for constraint in self.constraints:
            _validate_pass_schedule_constraint(constraint, self.passes)
        self._validated = True

    def solve_constraints(self):
        """
        Finds a valid traversal order based on the given constraints and orders
        the passes based on this order.

        If a circular dependency exists between the constraints and steps = 1,
        then we will raise an error because if steps != 1 this means that we
        will re-run the passes, allowing for circular dependencies.
        """
        self.passes = _topological_sort_passes(self.passes, self.constraints)
        self._validated = True

    def add_checks(self, check: Callable) -> None:
        """
        Adds a function which takes runs various checks on a given graph module.
        This function is run before and after each pass if the
        `run_checks_after_each_pass` flag is enabled.
        """
        sig = inspect.signature(check)

        if len(list(sig.parameters.values())) != 1:
            raise TypeError("PassManager check function should only take in one variable, a module")

        setattr(self, "check", check)  # noqa: B010

    def check(self, module: nn.Module) -> None:
        pass

    def __call__(self, module: nn.Module, **kwargs) -> PassResult:
        """
        Runs a list of passes in the order based on `self.passes` on the given
        graph module. Each time a pass is run, checks and linting will be run on
        the graph module if `run_checks_after_each_pass` is set.

        If the module is a graph module, we will run the list of passes until
        the graph stops changing, or until `steps` number of times.

        If **kwargs is passed in and `run_checks_after_each_pass` is set, we
        will check between each pass that the result of the graph module is
        still the same between passes.
        """
        # Order the passes based on the constraints
        if not self._validated:
            self.solve_constraints()

        # Check graph invariants
        self.check(module)

        # Run the set of passes `steps` number of times or until the graph stops
        # changing
        overall_modified = False
        for _ in range(self.steps):
            modified = False

            # Run the set of passes on the graph module
            for fn in self.passes:
                if self.run_checks_after_each_pass and "input" in kwargs:
                    orig_res = module(*kwargs["input"])

                res = fn(module)

                module = res.graph_module
                modified = modified or res.modified

                if isinstance(module, GraphModule):
                    module.recompile()

                # Check graph invariants
                if self.run_checks_after_each_pass:
                    self.check(module)

                    if "input" in kwargs and modified:
                        new_res = module(*kwargs["input"])
                        self.check_res_equal(fn, orig_res, new_res, **kwargs)

            # If the graph no longer changes, then we can stop running these passes
            overall_modified = overall_modified or modified
            if not modified:
                break

        return PassResult(module, overall_modified)

    def check_res_equal(self, pass_: Callable, res0: Argument, res1: Argument, **kwargs) -> None:
        """
        Validates that inference results before and after the pass are `all_close`

        Args:
            pass_: Pass that was used
            res0: Inference result before pass was run on the module
            res1: Inference result after pass was run on the module
            kwargs: Other kwargs that might be needed (ex. rtol, atol)
        """
        def _collect_tensors(arg: Argument) -> List[torch.Tensor]:
            """Collects all the tensors found in a nested container object"""
            res: List[torch.Tensor] = []

            def collect(x: Argument) -> Argument:
                if isinstance(x, torch.Tensor):
                    res.append(x)
                return x

            map_aggregate(arg, collect)
            return res

        tensor_res_0 = _collect_tensors(res0)
        tensor_res_1 = _collect_tensors(res1)

        for kk, (x, y) in enumerate(zip(tensor_res_0, tensor_res_1)):
            all_close_kwargs = {"equal_nan": True}
            if "rtol" in kwargs:
                all_close_kwargs["rtol"] = kwargs["rtol"]
            if "atol" in kwargs:
                all_close_kwargs["atol"] = kwargs["atol"]

            # If tensors are on different devices, make sure to compare
            # their copies that are on the same device.
            if x.get_device() != y.get_device():
                x = x.cpu()
                y = y.cpu()

            accuracy_check = torch.allclose(x, y, **all_close_kwargs)
            if not accuracy_check:
                if self.suppress_check_failures:
                    warnings.warn(
                        f"Pass {pass_} failed correctness check due to output {kk}."
                    )
                else:
                    raise RuntimeError(
                        f"Pass {pass_} failed correctness check due to output {kk}"
                    )
