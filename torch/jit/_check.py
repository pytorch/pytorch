
import ast
import inspect
import textwrap
import torch

class AttributeTypeIsSupportedChecker(ast.NodeVisitor):
    """
    Checks the ``__init__`` method of a given ``nn.Module`` to ensure
    that all instance-level attributes can be properly initialized.

    Specifically, we do type inference based on attribute values...even
    if the attribute in question has already been typed using
    Python3-style annotations or ``torch.jit.annotate``. This means that
    setting an instance-level attribute to ``[]`` (for ``List``),
    ``{}`` for ``Dict``), or ``None`` (for ``Optional``) isn't enough
    information for us to properly initialize that attribute.

    An object of this class can walk a given ``nn.Module``'s AST and
    determine if it meets our requirements or not.

    Known limitation: We can only check the AST nodes for certain
    constructs; we can't ``eval`` arbitrary expressions. This means
    that function calls, class instantiations, and complex expressions
    that resolve to one of the "empty" values specified above will NOT
    be flagged as problematic.

    Example:

        .. code-block:: python

            class M(torch.nn.Module):
                def fn(self):
                    return []

                def __init__(self):
                    super().__init__()
                    self.x: List[int] = []

                def forward(self, x: List[int]):
                    self.x = x
                    return 1

        The above code will pass the ``AttributeTypeIsSupportedChecker``
        check since we have a function call in ``__init__``. However,
        it will still fail later with the ``RuntimeError`` "Tried to set
        nonexistent attribute: x. Did you forget to initialize it in
        __init__()?".

    Args:
        nn_module - The instance of ``torch.nn.Module`` whose
            ``__init__`` method we wish to check
    """

    def check(self, nn_module: torch.nn.Module) -> None:
        source_lines = textwrap.dedent(inspect.getsource(nn_module.__class__.__init__))

        # This AST only contains the `__init__` method of the nn.Module
        init_ast = ast.parse(source_lines)

        self.visit(init_ast)

    def visit_AnnAssign(self, node):
        """
        Visit an AnnAssign node in an ``nn.Module``'s ``__init__``
        method and see if it conforms to our attribute annotation rules.
        """
        # If we have a class-level attribute (instance-level attributes
        # are instances of ast.Attribute)
        if isinstance(node.target, ast.Name):
            return

        # TODO @ansley: add `Union` once landed

        # NB: Even though `Tuple` is a "container", we don't want to
        # check for it here. `Tuple` functions as an type with an
        # "infinite" number of subtypes, in the sense that you can have
        # `Tuple[())]`, `Tuple[T1]`, `Tuple[T2]`, `Tuple[T1, T2]`,
        # `Tuple[T2, T1]` and so on, and none of these subtypes can be
        # used in place of the other. Therefore, assigning an empty
        # tuple in `__init__` CORRECTLY means that that variable
        # cannot be reassigned later to a non-empty tuple. Same
        # deal with `NamedTuple`

        containers = {"List", "Dict", "Optional"}

        # If we're not evaluating one of the specified problem types
        try:
            if node.annotation.value.id not in containers:
                return
        except AttributeError:
            # To evaluate a base type (`str`, `int`, etc.), we would
            # have needed to get the name through `node.annotation.id`
            # instead of `node.annotation.value.id`. Seems that we're
            # not evaluating one of our "containers"
            return

        # switch-case to check if the assigned variable is not empty
        ann_type = node.annotation.value.id
        if ann_type == "List":
            # Assigning `[]` to a `List` type gives you an AnnAssign
            # Node where value=List(elts=[], ctx=Load())
            if not isinstance(node.value, ast.List):
                return
            if node.value.elts:
                return
        elif ann_type == "Dict":
            # Assigning `{}` to a `Dict` type gives you an AnnAssign
            # Node where value=Dict(keys=[], values=[])
            if not isinstance(node.value, ast.Dict):
                return
            if node.value.keys:
                return
        elif ann_type == "Optional":
            # Assigning `None` to an `Optional` type gives you an
            # AnnAssign Node where value=Constant(value=None, kind=None)
            if not isinstance(node.value, ast.Constant):
                return
            if node.value.value:
                return

        raise RuntimeError("The TorchScript type system doesn't support"
                           " instance-level annotations on empty"
                           " non-base types in `__init__`. Instead, "
                           " either 1) use a type annotation in the "
                           "class body, or 2) wrap the type in "
                           "`torch.jit.Attribute`.")

    def visit_Call(self, node):
        """
        Visit a Call node in an ``nn.Module``'s ``__init__``
        method and determine if it's ``torch.jit.annotate``. If so,
        see if it conforms to our attribute annotation rules.
        """
        # If this isn't a call to `torch.jit.annotate`
        try:
            if (node.func.value.value.id != "torch"
                    or node.func.value.attr != "jit"
                    or node.func.attr != "annotate"):
                try:
                    self.generic_visit(node)
                except RuntimeError:
                    raise
                return
        except AttributeError:
            # Looks like we didn't even have the right node structure
            # to check for `torch.jit.annotate` in the first place
            try:
                self.generic_visit(node)
            except RuntimeError:
                raise
            return

        # Invariant: we have a `torch.jit.annotate` call

        # A Call Node for `torch.jit.annotate` should have an `args`
        # list of length 2 where args[0] represents the annotation and
        # args[1] represents the actual value
        if len(node.args) != 2:
            return

        if not isinstance(node.args[0], ast.Subscript):
            return

        # See notes in `visit_AnnAssign` r.e. containers

        containers = {"List", "Dict", "Optional"}

        ann_type = node.args[0].value.id    # type: ignore

        if ann_type not in containers:
            return

        # switch-case to check if the assigned variable is not empty
        if ann_type == "List":
            # An empty list is List(elts=[], ctx=Load())
            if not isinstance(node.args[1], ast.List):
                return
            if node.args[1].elts:
                return
        elif ann_type == "Dict":
            # An empty dict is Dict(keys=[], values=[])
            if not isinstance(node.args[1], ast.Dict):
                return
            if node.args[1].keys:
                return
        elif ann_type == "Optional":
            # `None` is Constant(value=None, kind=None)
            if not isinstance(node.args[1], ast.Constant):
                return
            if node.args[1].value:
                return

        raise RuntimeError("The TorchScript type system doesn't support"
                           " instance-level annotations on empty"
                           " non-base types in `__init__`. Instead, "
                           " either 1) use a type annotation in the "
                           "class body, or 2) wrap the type in "
                           "`torch.jit.Attribute`.")
