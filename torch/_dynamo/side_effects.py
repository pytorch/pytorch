import collections
import inspect
from typing import Any, Dict, List, Optional

import torch.nn

from . import utils, variables
from .bytecode_transformation import (
    create_call_function,
    create_call_method,
    create_instruction,
)
from .codegen import PyCodegen
from .exc import unimplemented
from .source import LocalSource, Source
from .utils import nn_module_new, object_new
from .variables.base import (
    is_side_effect_safe,
    MutableLocalBase,
    MutableLocalSource,
    VariableTracker,
)


class MutableSideEffects(MutableLocalBase):
    """
    VariableTracker.mutable_local marker to indicate a list passed as
    an input that if we mutate we need to re-apply those mutations after
    the graph runs.
    """

    def __init__(self, source: Source, is_modified: bool = False):
        super().__init__(MutableLocalSource.Existing)
        self.source = source
        self.is_modified = is_modified


class AttributeMutation(MutableLocalBase):
    """
    VariableTracker.mutable_local marker to track changes to attributes
    """

    def __init__(self, typ: MutableLocalSource, source: Source):
        super().__init__(typ)
        self.source = source


class AttributeMutationExisting(AttributeMutation):
    def __init__(self, source: Source):
        super().__init__(MutableLocalSource.Existing, source)
        self.source = source


class AttributeMutationNew(AttributeMutation):
    def __init__(self, source: Source, cls_source: Source):
        super().__init__(MutableLocalSource.Local, source)
        self.cls_source = cls_source


class SideEffects:
    """
    Track side effects (list mutation, setattr, etc) that need to be
    applied after an FX graph is run.
    """

    id_to_variable: Dict[int, VariableTracker]
    store_attr_mutations: Dict[AttributeMutation, Dict[str, VariableTracker]]
    keepalive: List[Any]

    def __init__(
        self,
        id_to_variable=None,
        store_attr_mutations=None,
        keepalive=None,
        save_for_backward=None,
        tensor_hooks=None,
    ):
        super().__init__()
        self.id_to_variable = id_to_variable or collections.OrderedDict()
        self.store_attr_mutations = store_attr_mutations or collections.OrderedDict()
        self.keepalive = keepalive or []
        self.save_for_backward = save_for_backward or []
        self.tensor_hooks = tensor_hooks or []

    def __eq__(self, other: object) -> bool:
        assert isinstance(other, SideEffects)
        # NB: do NOT test keepalive
        return (
            self.id_to_variable == other.id_to_variable
            and self.store_attr_mutations == other.store_attr_mutations
            and self.save_for_backward == other.save_for_backward
            and self.tensor_hooks == other.tensor_hooks
        )

    def diff(self, other: "SideEffects") -> Optional[str]:
        if self.id_to_variable != other.id_to_variable:
            sk_itv = self.id_to_variable.keys()
            ok_itv = other.id_to_variable.keys()
            if sk_itv != ok_itv:
                return f"id_to_variable keys: {sk_itv} != {ok_itv}"
            # Feel free to augment this with more fancy diffing logic
            # if needed for debugging
            return "id_to_variable: unknown diff"
        elif self.store_attr_mutations != other.store_attr_mutations:
            sk_sam = self.store_attr_mutations.keys()
            ok_sam = other.store_attr_mutations.keys()
            if sk_sam != ok_sam:
                return f"store_attr_mutations keys: {sk_sam} != {ok_sam}"
            return "store_attr_mutations: unknown diff"
        elif self.save_for_backward != other.save_for_backward:
            return "save_for_backward"
        else:
            return None

    def clone(self):
        """Create a shallow copy"""
        return self.__class__(
            id_to_variable=collections.OrderedDict(self.id_to_variable),
            store_attr_mutations=collections.OrderedDict(
                (k, collections.OrderedDict(v))
                for k, v in self.store_attr_mutations.items()
            ),
            keepalive=list(self.keepalive),
            save_for_backward=self.save_for_backward,
            tensor_hooks=self.tensor_hooks,
        )

    def apply(self, fn, cache=None, skip_fn=lambda _: False):
        if cache is None:
            cache = dict()

        self.id_to_variable = collections.OrderedDict(
            (k, VariableTracker.apply(fn, v, cache, skip_fn))
            for k, v in self.id_to_variable.items()
        )
        self.store_attr_mutations = collections.OrderedDict(
            (k, VariableTracker.apply(fn, v, cache, skip_fn))
            for k, v in self.store_attr_mutations.items()
        )
        self.save_for_backward = VariableTracker.apply(
            fn, self.save_for_backward, cache, skip_fn
        )
        self.tensor_hooks = VariableTracker.apply(fn, self.tensor_hooks, cache, skip_fn)

    def __contains__(self, item):
        return id(item) in self.id_to_variable

    def __getitem__(self, item):
        return self.id_to_variable[id(item)]

    def check_allowed_side_effect(self, item):
        from torch._dynamo.variables.misc import AutogradFunctionContextVariable

        # People do things like self.dim = dim inside autograd.Function.
        # These are benign.
        if isinstance(item, AutogradFunctionContextVariable):
            return True
        if not is_side_effect_safe(item.mutable_local):
            unimplemented(
                "HigherOrderOperator: Mutating a variable not in the current scope (SideEffects)"
            )

    def store_attr(self, item: VariableTracker, name: str, value: VariableTracker):
        assert self.is_attribute_mutation(item)
        self.check_allowed_side_effect(item)
        if item.mutable_local not in self.store_attr_mutations:
            self.store_attr_mutations[item.mutable_local] = collections.OrderedDict()
        self.store_attr_mutations[item.mutable_local][name] = value

    def load_attr(self, item, name, deleted_ok=False):
        assert self.is_attribute_mutation(item)
        result = self.store_attr_mutations[item.mutable_local][name]
        if not deleted_ok and isinstance(result, variables.DeletedVariable):
            unimplemented("read deleted attribute")
        return result

    def store_cell(self, cellvar, value):
        assert isinstance(cellvar, variables.NewCellVariable)
        assert isinstance(value, variables.VariableTracker)
        self.store_attr(cellvar, "cell_contents", value)

    def load_cell(self, cellvar):
        assert isinstance(cellvar, variables.NewCellVariable)
        return self.load_attr(cellvar, "cell_contents")

    def load_global(self, gvar: VariableTracker, name: str):
        assert isinstance(gvar, variables.VariableTracker)
        return self.load_attr(gvar, name)

    def store_global(self, gvar: VariableTracker, name: str, value: VariableTracker):
        assert isinstance(gvar, variables.VariableTracker)
        assert isinstance(value, variables.VariableTracker)
        self.store_attr(gvar, name, value)

    @staticmethod
    def cls_supports_mutation_side_effects(cls):
        return inspect.getattr_static(cls, "__setattr__", None) in (
            object.__setattr__,
            torch.nn.Module.__setattr__,
        )

    def is_attribute_mutation(self, item):
        return isinstance(item.mutable_local, AttributeMutation)

    def has_pending_mutation(self, item):
        return self.is_attribute_mutation(item) and bool(
            self.store_attr_mutations.get(item.mutable_local)
        )

    def is_modified(self, item):
        if isinstance(item.mutable_local, AttributeMutationNew):
            return True
        if self.is_attribute_mutation(item):
            return item.mutable_local in self.store_attr_mutations
        return item.mutable_local.is_modified

    def _track_obj(
        self,
        source: Source,
        item: Any,
        variable: VariableTracker,
        mutable_cls=MutableSideEffects,
    ):
        """Start tracking a new variable for mutation"""
        variable = variable.clone(mutable_local=mutable_cls(source), source=source)
        self.id_to_variable[id(item)] = variable
        self.keepalive.append(item)
        return variable

    track_list = _track_obj
    track_dict = _track_obj

    def track_object_existing(
        self,
        source: Source,
        item: Any,
        variable: VariableTracker,
    ):
        return self._track_obj(
            source, item, variable, mutable_cls=AttributeMutationExisting
        )

    def track_object_new(
        self,
        cls_source: Source,
        user_cls: Any,
        variable_cls: Any,
        options,
    ):
        if user_cls is torch.autograd.function.FunctionCtx:
            obj = torch.autograd.Function()
        elif issubclass(user_cls, torch.nn.Module):
            obj = nn_module_new(user_cls)
        else:
            obj = object_new(user_cls)
        variable = variable_cls(
            obj,
            mutable_local=AttributeMutationNew(None, cls_source),
            **options,
        )
        self.id_to_variable[id(obj)] = variable
        self.keepalive.append(obj)
        return variable

    def track_cell_new(
        self,
    ):
        obj = object()
        variable = variables.NewCellVariable(
            mutable_local=AttributeMutationNew(None, None),
        )
        self.id_to_variable[id(obj)] = variable
        self.keepalive.append(obj)
        return variable

    def track_cell_existing(self, source: Source, item: Any):
        variable = variables.NewCellVariable(
            mutable_local=AttributeMutationExisting(source),
        )
        self.id_to_variable[id(item)] = variable
        self.keepalive.append(item)
        return variable

    def track_global_existing(self, source: Source, item: Any):
        variable = variables.NewGlobalVariable(
            mutable_local=AttributeMutationExisting(source),
        )
        self.id_to_variable[id(item)] = variable
        self.keepalive.append(item)
        return variable

    def track_save_for_backward(self, ctx, args):
        assert isinstance(ctx, variables.AutogradFunctionContextVariable)
        self.save_for_backward.append((ctx, args))

    def prune_dead_object_new(self, tx):
        live_new_objects = set()
        skip_obj = None

        def visit(var: VariableTracker):
            if (
                isinstance(var.mutable_local, AttributeMutationNew)
                and var.mutable_local is not skip_obj
            ):
                live_new_objects.add(var.mutable_local)
            return var

        def is_live(var: VariableTracker):
            if isinstance(var, AttributeMutationNew):
                return var in live_new_objects
            if isinstance(var, VariableTracker):
                return is_live(var.mutable_local)
            return True

        VariableTracker.apply(visit, (tx.stack, tx.symbolic_locals))
        for var in self.id_to_variable.values():
            if not isinstance(var.mutable_local, AttributeMutationNew):
                VariableTracker.apply(visit, var)

        for skip_obj, setattrs in self.store_attr_mutations.items():
            VariableTracker.apply(visit, setattrs)

        self.id_to_variable = collections.OrderedDict(
            (k, v) for k, v in self.id_to_variable.items() if is_live(v)
        )
        self.store_attr_mutations = collections.OrderedDict(
            (k, v) for k, v in self.store_attr_mutations.items() if is_live(k)
        )

    def mutation(self, oldvar, newvar):
        self.check_allowed_side_effect(oldvar)
        return newvar.clone(
            mutable_local=MutableSideEffects(oldvar.mutable_local.source, True)
        )

    def _get_modified_vars(self):
        return [var for var in self.id_to_variable.values() if self.is_modified(var)]

    def codegen_save_tempvars(self, cg: PyCodegen):
        for var in self._get_modified_vars():
            if isinstance(
                var.mutable_local, (AttributeMutationExisting, AttributeMutationNew)
            ) and isinstance(var, variables.NewCellVariable):
                cg.load_import_from(utils.__name__, "make_cell")
                cg.extend_output(create_call_function(0, True))
                cg.add_cache(var)
                if isinstance(var.mutable_local, AttributeMutationNew):
                    var.mutable_local.source = LocalSource(cg.tempvars[var])
            elif isinstance(var.mutable_local, AttributeMutationNew):
                if isinstance(var, variables.AutogradFunctionContextVariable):
                    unimplemented("AutogradFunctionContextVariable escaped")
                if "__call_nn_module_init" in self.store_attr_mutations.get(
                    var.mutable_local, {}
                ):
                    assert isinstance(var, variables.UnspecializedNNModuleVariable)
                    cg.load_import_from(utils.__name__, "nn_module_new")
                else:
                    cg.load_import_from(utils.__name__, "object_new")
                cg(var.mutable_local.cls_source)
                cg.extend_output(create_call_function(1, True))
                cg.add_cache(var)
                var.mutable_local.source = LocalSource(cg.tempvars[var])
            elif var in cg.tempvars:
                assert cg.tempvars.get(var) is None
                # subsequent usage should point to the original variable
                cg(var.mutable_local.source)
                cg.add_cache(var)

        for ctx, args in self.save_for_backward:
            cg(ctx.source)
            cg.extend_output(
                [create_instruction("LOAD_METHOD", argval="save_for_backward")]
            )
            for arg in args:
                cg(arg)
            cg.extend_output(
                [
                    *create_call_method(len(args)),
                    create_instruction("POP_TOP"),
                ]
            )

        for (
            tensor,
            hook,
            handle,
        ) in self.tensor_hooks:
            # On dynamo tensor_hooks
            #
            # register_hook in the Graph: We bypass direct inclusion of register_hook calls in the graph.
            # Instead, these are tracked and stashed as a global variable, enabling their association with tensors in
            # the residuals. During dynamo's frame creation, these hooks are invoked seamlessly.
            #
            # Handling the Handle: When a user retains the register_hook result in a handle, we intercept the
            # STORE_FAST operation to record the user-designated local variable name. This ensures the reconstructed
            # bytecode retains this name. If no handle is defined, we simply pop the generated value to keep the
            # stack intact.
            #
            # Dynamo Tensor Hooks Workflow:
            # - Functions passed to register_hook are lifted globally.
            # - In the "side_effects" phase of codegen, we iterate over tensors with hooks to:
            #   - Generate the tensor.
            #   - Issue a register_hook call on the tensor, linking to the globally stored function.
            #   - Incorporate a handle if one was established in the eager phase.
            # The handle's exact user-specified name, "last_seen_name", is discerned and associated during STORE_FAST.
            cg(tensor)
            cg.extend_output([cg.create_load_attr("register_hook")])
            cg(hook)
            cg.extend_output(create_call_function(1, True))
            if hasattr(handle, "last_seen_name") and handle.last_seen_name:
                # register_hook stored with variable name assigned to the handle
                cg.extend_output(
                    [create_instruction("STORE_FAST", argval=handle.last_seen_name)]
                )
            else:
                # register_hook stored w/o a variable name assigned to the handle
                cg.extend_output([create_instruction("POP_TOP")])

    def register_hook(self, tensor, hook, handle):
        self.tensor_hooks.append((tensor, hook, handle))

    def codegen_update_mutated(self, cg: PyCodegen):
        suffixes = []
        for var in self._get_modified_vars():
            if isinstance(var, variables.ListVariable):
                # old[:] = new
                cg(var, allow_cache=False)
                cg(var.mutable_local.source)
                cg.extend_output(
                    [
                        cg.create_load_const(None),
                        cg.create_load_const(None),
                        create_instruction("BUILD_SLICE", arg=2),
                    ]
                )
                suffixes.append([create_instruction("STORE_SUBSCR")])
            elif isinstance(var, variables.ConstDictVariable):
                cg.tx.output.update_co_names("clear")
                cg.tx.output.update_co_names("update")

                cg(var.mutable_local.source)
                cg.extend_output([create_instruction("LOAD_METHOD", argval="update")])
                cg(var, allow_cache=False)

                cg(var.mutable_local.source)
                cg.extend_output([create_instruction("LOAD_METHOD", argval="clear")])

                suffixes.append(
                    [
                        *create_call_method(0),  # clear
                        create_instruction("POP_TOP"),
                        *create_call_method(1),  # update
                        create_instruction("POP_TOP"),
                    ]
                )
            elif self.is_attribute_mutation(var):
                for name, value in self.store_attr_mutations.get(
                    var.mutable_local, {}
                ).items():
                    if isinstance(var, variables.NewGlobalVariable):
                        cg.tx.output.update_co_names(name)
                        cg(value)
                        suffixes.append(
                            [create_instruction("STORE_GLOBAL", argval=name)]
                        )
                    elif name == "__call_nn_module_init":
                        pass  # handled in codegen_save_tempvars
                    elif isinstance(value, variables.DeletedVariable):
                        if isinstance(
                            var.mutable_local, AttributeMutationExisting
                        ) and hasattr(getattr(var, "value", None), name):
                            cg.tx.output.update_co_names(name)
                            cg(var.mutable_local.source)
                            suffixes.append(
                                [create_instruction("DELETE_ATTR", argval=name)]
                            )
                    else:
                        cg.tx.output.update_co_names(name)
                        cg(value)
                        cg(var.mutable_local.source)
                        suffixes.append([create_instruction("STORE_ATTR", argval=name)])
            else:
                raise AssertionError(type(var))

        # do all the actual mutations at the very end to handle dependencies
        for suffix in reversed(suffixes):
            cg.extend_output(suffix)

    def is_empty(self):
        return not (
            any(map(self.is_modified, self.id_to_variable.values()))
            or self.save_for_backward
        )

    def clear(self):
        self.keepalive.clear()
        self.id_to_variable.clear()
