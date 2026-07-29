"""Runtime guard matcher.

``GuardManagerWrapper`` wraps the C++ ``RootGuardManager`` (from
``torch._C._dynamo.guards``) that a Dynamo cache entry stores and evaluates via
``check_nopybind`` from C++. It lives in ``torch._guards`` so the runtime guard
matcher can be used without importing ``torch._dynamo``; the few dynamo ``config``
reads it needs are deferred to call time.
"""

from __future__ import annotations

import types
from contextlib import contextmanager
from typing import Any, TYPE_CHECKING

import torch
from torch._C._dynamo.guards import (
    ClosureGuardAccessor,
    CodeGuardAccessor,
    DictGetItemGuardAccessor,
    DictGuardManager,
    FuncDefaultsGuardAccessor,
    FuncKwDefaultsGuardAccessor,
    GetAttrGuardAccessor,
    GetGenericDictGuardAccessor,
    GuardAccessor,
    GuardDebugInfo,
    GuardManager,
    LeafGuard,
    RelationalGuard,
    RootGuardManager,
    TupleGetItemGuardAccessor,
    TypeDictGuardAccessor,
    TypeGuardAccessor,
    TypeMROGuardAccessor,
)
from torch._utils_internal import justknobs_check
from torch.utils._indented_buffer import IndentedBuffer
from torch.utils._ordered_set import OrderedSet


if TYPE_CHECKING:
    from collections.abc import Callable, Generator
    from weakref import ReferenceType

    from torch._dynamo.types import CacheEntry, ExtraState, GuardFail


dunder_attrs_assumed_constants = (
    "__defaults__",
    "__kwdefaults__",
    "__code__",
    "__closure__",
    "__annotations__",
    "__func__",
    "__mro__",
)


class IndentedBufferWithPrefix(IndentedBuffer):
    def prefix(self) -> str:
        return "| " * (self._indent * self.tabwidth)

    def writeline(self, line: str, skip_prefix: bool = False) -> None:  # type: ignore[override]
        if skip_prefix:
            super().writeline(line)
        else:
            super().writeline("+- " + line)


class GuardManagerWrapper:
    """
    A helper class that contains the root guard manager. An instance of this
    class is stored in the Dynamo cache entry, so that the cache entry can
    access the RootGuardManager stored in the "root" attribute and directly call
    the check_nopybind from C++.
    """

    def __init__(
        self,
        root: RootGuardManager | None = None,
        local_state: Any | None = None,
    ) -> None:
        if root is None:
            self.root = RootGuardManager()
        else:
            self.root = root

        if local_state is not None:
            self.root.set_local_state(local_state)

        self.diff_guard_root: RootGuardManager | None = None
        self.closure_vars: dict[str, Any] | None = None
        self.args: list[str] | None = None
        self.code_parts: list[str] = []
        self.verbose_code_parts: list[str] | None = None
        self.global_scope: dict[str, Any] | None = None
        self.guard_fail_fn: Callable[[GuardFail], None] | None = None
        self.cache_entry: CacheEntry | None = None
        self.extra_state: ExtraState | None = None
        self.id_matched_objs: dict[str, ReferenceType[object]] = {}
        self.no_tensor_aliasing_sources: list[str] = []

        self.printed_relational_guards: set[RelationalGuard] = set()

        self.diff_guard_sources: OrderedSet[str] = OrderedSet()

    @contextmanager
    def _preserve_printed_relational_guards(self) -> Generator[None, None, None]:
        self.printed_relational_guards = set()
        try:
            yield
        finally:
            self.printed_relational_guards = set()

    # TODO: clarify what fn and attributes guard manager has to get the right things here
    def collect_diff_guard_sources(self) -> OrderedSet[str]:
        # At the time of finalize, we have only marked guard managers with
        # TENSOR_MATCH guards as diff guard managers. So, we do a tree traversal
        # and collect all the nodes in the tree (branches) that lead to tensor
        # guards.

        # After a recompilation, some of guard managers will have a fail_count >
        # 0, so we collect them as well. Later on, we accumulate the diff guard
        # sources for all the guard managers.

        def visit_dict_manager(node: DictGuardManager) -> bool:
            is_diff_guard_node = (
                node.get_source() in self.diff_guard_sources or node.fail_count() > 0
            )
            for _idx, (key_mgr, val_mgr) in sorted(
                node.get_key_value_managers().items()
            ):
                is_diff_guard_node |= visit(key_mgr) | visit(val_mgr)

            if is_diff_guard_node:
                self.diff_guard_sources.add(node.get_source())

            return is_diff_guard_node

        def visit_manager(node: GuardManager) -> bool:
            if isinstance(node, DictGuardManager):
                raise AssertionError(
                    f"Expected non-DictGuardManager node, got {type(node)}"
                )

            is_diff_guard_node = (
                node.get_source() in self.diff_guard_sources or node.fail_count() > 0
            )
            for child_mgr in node.get_child_managers():
                is_diff_guard_node |= visit(child_mgr)

            if is_diff_guard_node:
                self.diff_guard_sources.add(node.get_source())

            return is_diff_guard_node

        def visit(node: GuardManager) -> bool:
            if node is None:
                return False
            if isinstance(node, DictGuardManager):
                return visit_dict_manager(node)
            return visit_manager(node)

        visit(self.root)

        return self.diff_guard_sources

    def finalize(self) -> None:
        from torch._dynamo import config

        if config.use_recursive_dict_tags_for_guards and justknobs_check(
            "pytorch/compiler:use_recursive_dict_tags_for_guards"
        ):
            self.find_tag_safe_roots()
        self.prepare_diff_guard_manager()

    def prepare_diff_guard_manager(self) -> None:
        self.collect_diff_guard_sources()
        self.populate_diff_guard_manager()

    def find_tag_safe_roots(self) -> None:
        """
        Identify ``tag safe nodes`` and ``tag safe roots`` within a guard tree.

        -----------------------------------------------------------------------
        tag safe node
        -----------------------------------------------------------------------
        A *tag safe node* is a ``GuardManager`` whose guarded value satisfies one
        of the following conditions:

        1. Immutable value - The value is intrinsically immutable according to
        ``is_immutable_object``. Tensors are considered immutable. To ensure
        that symbolic guards run, we also check that the GuardManager has no
        accessors.

        2. Nested tag safe dictionary - The value is a ``dict`` whose keys and
        values are all tag safe nodes  (checked recursively).  Such dictionaries
        allow entire nested structures to be skipped once their identity tag
        matches.

        3. Pure ``nn.Module`` - The value is an ``nn.Module`` whose sole
        accessor is ``GetGenericDictGuardAccessor``—i.e., it only exposes its
        ``__dict__`` and nothing else that could mutate between runs.

        For every tag safe node, verifying the identity/tag of just the top-level
        dictionary is enough to guarantee the entire subtree is unchanged, enabling
        a *fast-path* guard check.

        -----------------------------------------------------------------------
        tag safe root
        -----------------------------------------------------------------------
        A ``tag safe root`` is a tag safe node whose parent is not tag safe.
        These boundary nodes mark the points where guard evaluation can safely
        prune traversal: if a tag-safe root's dictionary tag matches, the entire
        subtree beneath it is skipped.

        One strong requirement for tag safe root is for the guarded object to
        support weakref. Refer to more details in the Recursive dict tag
        matching note. In short, we need to save the weakref of the object on
        first invocation, and check if it is still valid in later iterations, to
        apply recursive dict tag optimizations. `dict` objects do NOT support
        weakref. Therefore, as of now, we only mark nn module related guard
        managers as tag safe roots.

        Algorithm
        ---------
        The search runs in post-order traversal

        1. Visit leaves and classify them as tag safe or not.
        2. Propagate tag-safety upward: a parent dictionary becomes tag safe only if
        all of its children are already tag-safe.
        3. Propagate tag-safe-rootness upward: if the whole subtree is tag safe,
        the current node becomes the new tag safe root, otherwise propagate the
        subtree tag safe roots.
        4. Collect every tag safe node and, by inspecting parent tags, label the
        subset that are tag safe roots.
        """
        from torch._dynamo import config

        def check_tag_safety(
            node: GuardManager, accepted_accessors: tuple[type[GuardAccessor], ...]
        ) -> bool:
            accessors = node.get_accessors()
            child_mgrs = node.get_child_managers()
            return all(
                isinstance(accessor, accepted_accessors) and mgr.is_tag_safe()
                for accessor, mgr in zip(accessors, child_mgrs)
            )

        def visit_dict_manager(node: DictGuardManager) -> list[GuardManager]:
            # Just recurse through the key and value dict managers and check if
            # all of them are tag safe nodes.
            if not issubclass(node.get_type_of_guarded_value(), dict):
                raise AssertionError(
                    f"Expected dict subclass, got {node.get_type_of_guarded_value()}"
                )

            tag_safe_roots = []
            is_subtree_tag_safe = True

            # Recurse to get the tag safe roots from subtree.
            for _idx, (key_mgr, val_mgr) in sorted(
                node.get_key_value_managers().items()
            ):
                if key_mgr is not None:
                    visit(key_mgr)
                if val_mgr is not None:
                    tag_safe_roots.extend(visit(val_mgr))

            for key_mgr, val_mgr in node.get_key_value_managers().values():
                if key_mgr:
                    is_subtree_tag_safe &= key_mgr.is_tag_safe()

                if val_mgr:
                    is_subtree_tag_safe &= val_mgr.is_tag_safe()

            if is_subtree_tag_safe:
                node.mark_tag_safe()
            return tag_safe_roots

        def visit_manager(node: GuardManager) -> list[GuardManager]:
            if isinstance(node, DictGuardManager):
                raise AssertionError(
                    f"Expected non-DictGuardManager node, got {type(node)}"
                )

            # Collect the subtree tag safe roots
            tag_safe_roots = []
            for child_mgr in node.get_child_managers():
                tag_safe_roots.extend(visit(child_mgr))

            if node.is_guarded_value_immutable():
                # If the node guards a tensor, mark it tag safe only if there
                # are no accessors. Presence of accessors means presence of
                # symbolic shape guards.
                if issubclass(node.get_type_of_guarded_value(), torch.Tensor):
                    if node.has_no_accessors() and not node.has_object_aliasing_guard():
                        node.mark_tag_safe()
                elif any(
                    a.repr() == "PythonLambdaGuardAccessor"
                    for a in node.get_accessors()
                ):
                    # PythonLambdaGuardAccessor produces ephemeral objects
                    # (e.g., ___from_numpy converts np.float64 to a temporary
                    # tensor). These must not be stashed by the tag-safe
                    # recording pass since they are freed after each check.
                    pass
                else:
                    node.mark_tag_safe()
            elif issubclass(node.get_type_of_guarded_value(), dict):
                accessors = node.get_accessors()
                child_mgrs = node.get_child_managers()
                is_subtree_tag_safe = all(
                    isinstance(accessor, DictGetItemGuardAccessor) and mgr.is_tag_safe()
                    for accessor, mgr in zip(accessors, child_mgrs)
                )
                if is_subtree_tag_safe:
                    node.mark_tag_safe()
            elif issubclass(node.get_type_of_guarded_value(), torch.nn.Module):
                is_subtree_tag_safe = check_tag_safety(
                    node, (GetGenericDictGuardAccessor, TypeGuardAccessor)
                )
                if is_subtree_tag_safe:
                    node.mark_tag_safe()
                    # Return the current node as tag safe root, discarding the
                    # subtree tag safe roots.
                    return [
                        node,
                    ]
            elif (
                node.get_type_of_guarded_value()
                in (
                    types.FunctionType,
                    types.MethodType,
                    staticmethod,
                    classmethod,
                )
                and config.assume_dunder_attributes_remain_unchanged
            ):
                # Assumption: callers will not reassign the attributes
                #   func.__code__, func.__closure__, func.__defaults__, or func.__kwdefaults__.
                # Mutating the objects those attributes point to is fine;
                # rebinding the attribute itself is not.
                # Example ─ allowed:   foo.__defaults__[0].bar = 99
                #          forbidden: foo.__defaults__ = (3, 4)
                is_subtree_tag_safe = check_tag_safety(
                    node,
                    (
                        CodeGuardAccessor,
                        ClosureGuardAccessor,
                        FuncDefaultsGuardAccessor,
                        FuncKwDefaultsGuardAccessor,
                        GetAttrGuardAccessor,
                    ),
                )

                for accessor in node.get_accessors():
                    if isinstance(accessor, GetAttrGuardAccessor):
                        is_subtree_tag_safe &= (
                            accessor.get_attr_name() in dunder_attrs_assumed_constants
                        )

                if is_subtree_tag_safe:
                    node.mark_tag_safe()
            elif issubclass(node.get_type_of_guarded_value(), types.CellType):
                is_subtree_tag_safe = check_tag_safety(node, (GetAttrGuardAccessor,))

                is_subtree_tag_safe &= all(
                    isinstance(accessor, GetAttrGuardAccessor)
                    and accessor.get_attr_name() == "cell_contents"
                    for accessor in node.get_accessors()
                )
                if is_subtree_tag_safe:
                    node.mark_tag_safe()
            elif (
                issubclass(node.get_type_of_guarded_value(), tuple)
                and node.get_source().endswith(dunder_attrs_assumed_constants)
                and config.assume_dunder_attributes_remain_unchanged
            ):
                # We trust tuples obtained from a function's __closure__ or
                # __defaults__. Any *other* tuple-valued attribute can be
                # silently replaced—for example:
                #
                #     foo.bar = (1, 2)      # original
                #     foo.bar = (3, 4)      # rebinding that our dict-tag optimisation won't see
                #
                # Therefore only tuples from __closure__ / __defaults__ participate in the
                # recursive-dict-tag optimization; all others are ignored.
                is_subtree_tag_safe = check_tag_safety(
                    node, (TupleGetItemGuardAccessor,)
                )
                if is_subtree_tag_safe:
                    node.mark_tag_safe()
            elif issubclass(node.get_type_of_guarded_value(), type):
                is_subtree_tag_safe = check_tag_safety(
                    node, (TypeDictGuardAccessor, TypeMROGuardAccessor)
                )
                if is_subtree_tag_safe:
                    node.mark_tag_safe()

            return tag_safe_roots

        def visit(node: GuardManager) -> list[GuardManager]:
            if node is None:
                return []
            if isinstance(node, DictGuardManager):
                return visit_dict_manager(node)
            return visit_manager(node)

        tag_safe_roots = visit(self.root)
        for node in tag_safe_roots:
            if issubclass(node.get_type_of_guarded_value(), torch.nn.Module):
                node.mark_tag_safe_root()

    def populate_diff_guard_manager(self) -> None:
        self.diff_guard_root = self.clone_with_chosen_sources(self.diff_guard_sources)

        # Ensure that C++ side points to the updated diff guard manager.
        # When a new GuardManagerWrapper is created, it does not have a
        # cache_entry attribute, so it relies on the CacheEntry constructor to
        # set the diff_guard_root in C++.  But once it is saved in the Dynamo
        # cache, C++ side adds a cache_entry attribute. On recompiles, this
        # cache_entry is visible, so we update the C++ side to point to the
        # update guard manager.
        if self.cache_entry:
            self.cache_entry.update_diff_guard_root_manager()

    def clone_with_chosen_sources(
        self, chosen_sources: OrderedSet[str]
    ) -> RootGuardManager:
        def filter_fn(node_mgr: GuardManager) -> bool:
            return node_mgr.get_source() in chosen_sources

        return self.root.clone_manager(filter_fn)

    def get_guard_lines(self, guard: LeafGuard) -> list[str]:
        guard_name = guard.__class__.__name__
        parts = guard.verbose_code_parts()
        parts = [guard_name + ": " + part for part in parts]
        return parts

    def get_manager_line(
        self, guard_manager: GuardManager, accessor_str: str | None = None
    ) -> str:
        source = guard_manager.get_source()
        t = guard_manager.__class__.__name__
        s = t + ": source=" + source
        if accessor_str:
            s += ", " + accessor_str
        s += f", type={guard_manager.get_type_of_guarded_value()}"
        s += f", tag_safe=({guard_manager.is_tag_safe()}, {guard_manager.is_tag_safe_root()})"
        return s

    def construct_dict_manager_string(
        self, mgr: DictGuardManager, body: IndentedBufferWithPrefix
    ) -> None:
        for idx, (key_mgr, val_mgr) in sorted(mgr.get_key_value_managers().items()):
            body.writeline(f"KeyValueManager pair at index={idx}")
            with body.indent():
                if key_mgr:
                    body.writeline(f"KeyManager: {self.get_manager_line(key_mgr)}")
                    self.construct_manager_string(key_mgr, body)

                if val_mgr:
                    body.writeline(f"ValueManager: {self.get_manager_line(val_mgr)}")
                    self.construct_manager_string(val_mgr, body)

    def construct_manager_string(
        self, mgr: GuardManager, body: IndentedBufferWithPrefix
    ) -> None:
        with body.indent():
            for guard in mgr.get_leaf_guards():
                if isinstance(guard, RelationalGuard):
                    if guard not in self.printed_relational_guards:
                        self.printed_relational_guards.add(guard)

                        body.writelines(self.get_guard_lines(guard))
                    else:
                        body.writelines(
                            [
                                guard.__class__.__name__,
                            ]
                        )
                else:
                    body.writelines(self.get_guard_lines(guard))

            # This works for both DictGuardManager and SubclassedDictGuardManager
            if isinstance(mgr, DictGuardManager):
                self.construct_dict_manager_string(mgr, body)

            # General case of GuardManager/RootGuardManager
            for accessor, child_mgr in zip(
                mgr.get_accessors(), mgr.get_child_managers()
            ):
                body.writeline(
                    self.get_manager_line(child_mgr, f"accessed_by={accessor.repr()}")
                )
                self.construct_manager_string(child_mgr, body)

    def __str__(self) -> str:
        with self._preserve_printed_relational_guards():
            body = IndentedBufferWithPrefix()
            body.tabwidth = 1
            body.writeline("", skip_prefix=True)
            body.writeline("TREE_GUARD_MANAGER:", skip_prefix=True)
            body.writeline("RootGuardManager")
            self.construct_manager_string(self.root, body)
            if hasattr(self.root, "get_epilogue_lambda_guards"):
                for guard in self.root.get_epilogue_lambda_guards():
                    body.writelines(self.get_guard_lines(guard))
            return body.getvalue()

    def check(self, x: Any) -> bool:
        # Only needed for debugging purposes.
        return self.root.check(x)

    def check_verbose(self, x: Any) -> GuardDebugInfo:
        # Only needed for debugging purposes.
        return self.root.check_verbose(x)

    def populate_code_parts_for_debugging(self) -> None:
        # This should be called when the guard manager is fully populated
        relational_guards_seen = set()

        def get_code_parts(leaf_guard: LeafGuard) -> list[str]:
            code_parts = []
            for verbose_code_part in leaf_guard.verbose_code_parts():
                code_part = verbose_code_part.split("#")[0].rstrip()
                code_parts.append(code_part)
            return code_parts

        def visit(mgr: GuardManager) -> None:
            nonlocal relational_guards_seen
            for guard in mgr.get_leaf_guards():
                if isinstance(guard, RelationalGuard):
                    if guard not in relational_guards_seen:
                        self.code_parts.extend(get_code_parts(guard))
                        relational_guards_seen.add(guard)
                else:
                    self.code_parts.extend(get_code_parts(guard))

            for child_mgr in mgr.get_child_managers():
                visit(child_mgr)

        visit(self.root)


class DeletedGuardManagerWrapper(GuardManagerWrapper):
    def __init__(self, reason: str) -> None:
        super().__init__()
        self.invalidation_reason = reason

    def populate_diff_guard_manager(self) -> None:
        self.diff_guard_root = None
