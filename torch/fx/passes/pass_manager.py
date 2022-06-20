import inspect
from typing import Callable, List


from torch.fx import GraphModule
from pass_base import PassManagerParams


# Pass Schedule Constraints:
#
# Implemented as 'depends on' operators. A constraint is satisfied iff a list
# has a valid partial ordering according to this comparison operator.
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


def this_before_that_pass_constraint(this: Callable, that: Callable) -> None:
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


class PassManager:
    """
    Construct a PassManager.

    Collects passes and constraints. This defines the pass schedule, manages
    pass constraints and pass execution.

    Args:
        passes (Optional[List[Callable]]): list of passes. A pass is a
            callable which modifies an object and returns modified object
        constraint (Optional[List[Callable]]): list of constraints. A
            constraint is a callable which takes two passes (A, B) and returns
            True if A depends on B and False otherwise. See implementation of
            `this_before_that_pass_constraint` for example.
        enable_debug_pass (bool): set to true to enable the debug passes
        run_checks_after_each_pass (bool): whether to run checks and linting
            after each pass
    """

    passes: List[Callable] = []
    constraints: List[Callable] = []
    _validated: bool = False

    def __init__(
        self,
        passes=None,
        constraints=None,
        enable_debug_pass: bool = False,
        run_checks_after_each_pass: bool = False,
    ):
        if passes:
            self.passes = passes
        if constraints:
            self.constraints = constraints
        
        self.params = PassManagerParams(
            enable_debug_pass=enable_debug_pass,
            run_checks_after_each_pass=run_checks_after_each_pass,
        )

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
    
    def add_checks(self, check: Callable) -> None:
        """
        Adds a function which takes runs various checks on a given graph module.
        This function is run before and after each pass if the
        `run_checks_after_each_pass` flag is enabled.
        """
        sig = inspect.signature(check)

        params = [p for p in sig.parameters.values()]
        if len(params) != 1:
            raise TypeError("PassManager check function should only take in one variable, a graph_module")

        if sig.return_annotation: 
            raise TypeError("PassManager check function should return None")

        self.assert_invariants = check

    def assert_invariants(self, graph_module: GraphModule) -> None:
        pass

    def __call__(self, graph_module: GraphModule) -> None:
        """
        Runs a list of passes in the order based on `self.passes` on the given
        graph module. Each time a pass is run, checks and linting will be run on
        the graph module to ensure that it still maintains the same required
        invariants.
        """
        # Check constraints 
        self.validate_constraints()

        # Lint and check graph invariants
        graph_module.graph.lint()
        self.assert_invariants(graph_module)

        # TODO(angelayi): run these passes multiple times until the graph
        # converts, set max iteration number too

        for fn in self.passes:
            fn(self.params, graph_module)

            if self.params.run_checks_after_each_pass:
                # Lint and check graph invariants
                graph_module.graph.lint()
                self.assert_invariants(graph_module)

        graph_module.recompile()
