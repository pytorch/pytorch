import ast
import inspect
import textwrap
import warnings

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

    Known limitations
    1. We can only check the AST nodes for certain constructs; we can't
    ``eval`` arbitrary expressions. This means that function calls,
    class instantiations, and complex expressions that resolve to one of
    the "empty" values specified above will NOT be flagged as
    problematic.
    2. We match on string literals, so if the user decides to use a
    non-standard import (e.g. `from typing import List as foo`), we
    won't catch it.

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
        source_lines = inspect.getsource(nn_module.__class__.__init__)

        # Ignore comments no matter the indentation
        def is_useless_comment(line):
            line = line.strip()
            return line.startswith("#") and not line.startswith("# type:")

        source_lines = "\n".join(
            [l for l in source_lines.split("\n") if not is_useless_comment(l)]
        )

        # This AST only contains the `__init__` method of the nn.Module
        init_ast = ast.parse(textwrap.dedent(source_lines))

        # Get items annotated in the class body
        self.class_level_annotations = list(nn_module.__annotations__.keys())

        # Flag for later
        self.visiting_class_level_ann = False

        self.visit(init_ast)

    def _is_empty_container(self, node: ast.AST, ann_type: str) -> bool:
        if ann_type == "List":
            # Assigning `[]` to a `List` type gives you a Node where
            # value=List(elts=[], ctx=Load())
            if not isinstance(node, ast.List):
                return False
            if node.elts:
                return False
        elif ann_type == "Dict":
            # Assigning `{}` to a `Dict` type gives you a Node where
            # value=Dict(keys=[], values=[])
            if not isinstance(node, ast.Dict):
                return False
            if node.keys:
                return False
        elif ann_type == "Optional":
            # Assigning `None` to an `Optional` type gives you a
            # Node where value=Constant(value=None, kind=None)
            if not isinstance(node, ast.Constant):
                return False
            if node.value:  # type: ignore[attr-defined]
                return False

        return True

    def visit_Assign(self, node):
        """
        If we're visiting a Call Node (the right-hand side of an
        assignment statement), we won't be able to check the variable
        that we're assigning to (the left-hand side of an assignment).
        Because of this, we need to store this state in visitAssign.
        (Luckily, we only have to do this if we're assigning to a Call
        Node, i.e. ``torch.jit.annotate``. If we're using normal Python
        annotations, we'll be visiting an AnnAssign Node, which has its
        target built in.)
        """
        try:
            if (
                isinstance(node.value, ast.Call)
                and node.targets[0].attr in self.class_level_annotations
            ):
                self.visiting_class_level_ann = True
        except AttributeError:
            return
        self.generic_visit(node)
        self.visiting_class_level_ann = False

    def visit_AnnAssign(self, node):
        """
        Visit an AnnAssign node in an ``nn.Module``'s ``__init__``
        method and see if it conforms to our attribute annotation rules.
        """
        # If we have a local variable
        try:
            if node.target.value.id != "self":
                return
        except AttributeError:
            return

        # If we have an attribute that's already been annotated at the
        # class level
        if node.target.attr in self.class_level_annotations:
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

        # Check if the assigned variable is empty
        ann_type = node.annotation.value.id
        if not self._is_empty_container(node.value, ann_type):
            return

        warnings.warn(
            "The TorchScript type system doesn't support "
            "instance-level annotations on empty non-base "
            "types in `__init__`. Instead, either 1) use a "
            "type annotation in the class body, or 2) wrap "
            "the type in `torch.jit.Attribute`.",
            stacklevel=2,
        )

    def visit_Call(self, node):
        """
        Visit a Call node in an ``nn.Module``'s ``__init__``
        method and determine if it's ``torch.jit.annotate``. If so,
        see if it conforms to our attribute annotation rules.
        """
        # If we have an attribute that's already been annotated at the
        # class level
        if self.visiting_class_level_ann:
            return

        # If this isn't a call to `torch.jit.annotate`
        try:
            if (
                node.func.value.value.id != "torch"
                or node.func.value.attr != "jit"
                or node.func.attr != "annotate"
            ):
                self.generic_visit(node)
            elif (
                node.func.value.value.id != "jit" or node.func.value.attr != "annotate"
            ):
                self.generic_visit(node)
        except AttributeError:
            # Looks like we didn't even have the right node structure
            # to check for `torch.jit.annotate` in the first place
            self.generic_visit(node)

        # Invariant: we have a `torch.jit.annotate` or a
        # `torch.annotate` call

        # A Call Node for `torch.jit.annotate` should have an `args`
        # list of length 2 where args[0] represents the annotation and
        # args[1] represents the actual value
        if len(node.args) != 2:
            return

        if not isinstance(node.args[0], ast.Subscript):
            return

        # See notes in `visit_AnnAssign` r.e. containers

        containers = {"List", "Dict", "Optional"}

        try:
            ann_type = node.args[0].value.id  # type: ignore[attr-defined]
        except AttributeError:
            return

        if ann_type not in containers:
            return

        # Check if the assigned variable is empty
        if not self._is_empty_container(node.args[1], ann_type):
            return

        warnings.warn(
            "The TorchScript type system doesn't support "
            "instance-level annotations on empty non-base "
            "types in `__init__`. Instead, either 1) use a "
            "type annotation in the class body, or 2) wrap "
            "the type in `torch.jit.Attribute`.",
            stacklevel=2,
        )
