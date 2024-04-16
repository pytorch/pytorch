import inspect
import warnings
from typing import Any, Dict, List, Optional, Union

import torch.nn

from . import utils, variables
from .bytecode_transformation import (
    create_call_function,
    create_call_method,
    create_instruction,
    create_load_method,
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

    def __init__(self, typ: MutableLocalSource, source: Optional[Source]):
        super().__init__(typ)
        self.source = source


class AttributeMutationExisting(AttributeMutation):
    def __init__(self, source: Source):
        super().__init__(MutableLocalSource.Existing, source)
        self.source = source


class AttributeMutationNew(AttributeMutation):
    def __init__(self, source: Optional[Source], cls_source: Optional[Source]):
        super().__init__(MutableLocalSource.Local, source)
        self.cls_source = cls_source


class SideEffects:
    """
    Track side effects (list mutation, setattr, etc) that need to be
    applied after an FX graph is run.
    """

    id_to_variable: Dict[int, VariableTracker]
    store_attr_mutations: Dict[MutableLocalBase, Dict[str, VariableTracker]]
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
        self.id_to_variable = id_to_variable or {}
        self.store_attr_mutations = store_attr_mutations or {}
        self.keepalive = keepalive or []
        self.save_for_backward = save_for_backward or []
        self.tensor_hooks = tensor_hooks or {}

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
        elif self.tensor_hooks != other.tensor_hooks:
            return "tensor_hooks"
        else:
            return None

    def clone(self):
        """Create a shallow copy"""
        return self.__class__(
            id_to_variable=dict(self.id_to_variable),
            store_attr_mutations={
                k: dict(v) for k, v in self.store_attr_mutations.items()
            },
            keepalive=list(self.keepalive),
            save_for_backward=self.save_for_backward,
            tensor_hooks=self.tensor_hooks,
        )

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
        # TODO(yf225): make it safe when the HOP is utils.checkpoint and the mutation is not on a tensor
        # or create a way to mark it as safe
        return True
        # if not is_side_effect_safe(item.mutable_local):
        #     unimplemented(
        #         "HigherOrderOperator: Mutating a variable not in the current scope (SideEffects)"
        #     )

    def store_attr(self, item: VariableTracker, name: str, value: VariableTracker):
        assert self.is_attribute_mutation(item)
        self.check_allowed_side_effect(item)
        if item.mutable_local not in self.store_attr_mutations:
            self.store_attr_mutations[item.mutable_local] = {}
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
        return (
            inspect.getattr_static(cls, "__getattribute__", None)
            is object.__getattribute__
        )

    def is_attribute_mutation(self, item):
        return isinstance(item.mutable_local, AttributeMutation)

    def has_pending_mutation(self, item):
        return self.is_attribute_mutation(item) and bool(
            self.store_attr_mutations.get(item.mutable_local)
        )

    def has_pending_mutation_of_attr(self, item, name):
        return self.is_attribute_mutation(
            item
        ) and name in self.store_attr_mutations.get(item.mutable_local, ())

    def is_modified(self, item):
        if isinstance(item.mutable_local, AttributeMutationNew):
            return True
        if self.is_attribute_mutation(item):
            return item.mutable_local in self.store_attr_mutations
        return item.mutable_local.is_modified

    def _track_obj(
        self,
        item: Any,
        variable: VariableTracker,
        mutable_cls=MutableSideEffects,
    ):
        """Start tracking a new variable for mutation"""
        assert variable.source is not None
        variable.mutable_local = mutable_cls(variable.source)
        self.id_to_variable[id(item)] = variable
        self.keepalive.append(item)
        return variable

    track_mutable = _track_obj

    def track_object_existing(
        self,
        item: Any,
        variable: VariableTracker,
    ):
        return self._track_obj(item, variable, mutable_cls=AttributeMutationExisting)

    def track_object_new(
        self,
        cls_source: Source,
        user_cls: Any,
        variable_cls: Any,
        options,
    ):
        if user_cls is torch.autograd.function.FunctionCtx:
            with warnings.catch_warnings(record=True):
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

    def track_tensor_variables_from_runahead_side_effects(self, other):
        # In higher order ops we want to keep track of tensors seen in the
        # speculate_subgraph so that we don't lift them again as a new input in
        # other speculate_subgraph or in the root tracer.
        for other_item in other.keepalive:
            other_id = id(other_item)
            other_variable = other.id_to_variable[other_id]
            if other_id not in self.id_to_variable and isinstance(
                other_variable, variables.TensorVariable
            ):
                self.track_object_existing(other_item, other_variable)

    def prune_dead_object_new(self, tx):
        live_new_objects = set()
        skip_obj = None

        def visit(var: VariableTracker):
            if (
                isinstance(var.mutable_local, AttributeMutationNew)
                and var.mutable_local is not skip_obj
            ):
                live_new_objects.add(var.mutable_local)

        def is_live(var: Union[MutableLocalBase, VariableTracker]):
            if isinstance(var, AttributeMutationNew):
                return var in live_new_objects
            if isinstance(var, VariableTracker):
                return is_live(var.mutable_local)
            return True

        VariableTracker.visit(visit, (tx.stack, tx.symbolic_locals))
        for var in self.id_to_variable.values():
            if not isinstance(var.mutable_local, AttributeMutationNew):
                VariableTracker.visit(visit, var)

        for skip_obj, setattrs in self.store_attr_mutations.items():
            VariableTracker.visit(visit, setattrs)

        self.id_to_variable = {
            k: v for k, v in self.id_to_variable.items() if is_live(v)
        }
        self.store_attr_mutations = {
            k: v for k, v in self.store_attr_mutations.items() if is_live(k)
        }

    def mutation(self, var):
        self.check_allowed_side_effect(var)
        if isinstance(var.mutable_local, MutableSideEffects):
            var.mutable_local = MutableSideEffects(var.mutable_local.source, True)

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
                    var.mutable_local.source = LocalSource(cg.tempvars[var])  # type: ignore[attr-defined]
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
            cg.extend_output([create_load_method("save_for_backward")])
            for arg in args:
                cg(arg)
            cg.extend_output(
                [
                    *create_call_method(len(args)),
                    create_instruction("POP_TOP"),
                ]
            )

    def register_hook(self, tensor, hook, handle, name):
        assert isinstance(tensor, variables.TensorVariable)
        assert isinstance(hook, variables.VariableTracker)
        assert (
            isinstance(handle, variables.RemovableHandleVariable)
            and handle.mutable_local
        )
        assert hasattr(torch.Tensor, name)
        idx = len(self.tensor_hooks.keys())
        # duplicate index possible because of self.remove_hook()
        while idx in self.tensor_hooks:
            idx += 1
        self.tensor_hooks[idx] = (tensor, hook, handle, name)
        assert not handle.idx
        handle.idx = idx

    def remove_hook(self, idx):
        del self.tensor_hooks[idx]

    def codegen_hooks(self, cg):
        for (
            tensor,
            hook,
            handle,
            name,
        ) in self.tensor_hooks.values():
            # Note: [On tensor.register_hook]
            #
            # register_hook on a tensor, AKA backward hooks, have slightly nuanced differences in how they are implemented
            # when it comes to hooks on objects with sources (inputs, params) vs objects without sources (intermediaries).
            #
            # For tensors with a source, we bypass direct inclusion of register_hook calls in the graph.
            # Instead, these are tracked and stashed as a global variable, enabling their association with tensors in
            # the residuals. During dynamo's frame creation, these hooks are invoked seamlessly on known reconstructible/fetch-able
            # tensors. Because a source indicates knowledge of this object outside the torch compile region, and
            # because we are running residuals firmly before .backward() can be run, it is sound to invoke
            # `register_hook` on a known tensor.
            #
            # For tensors without a source, we support a limited subset of hooks. Global functions only, and
            # compiled_autograd must be enabled or we will graph break.
            #
            # Handling the Handle: When a user retains the register_hook result in a handle, we intercept the
            # STORE_FAST operation to record the user-designated local variable name. This ensures the reconstructed
            # bytecode retains this name. If no handle is defined, we simply pop the generated value to keep the
            # stack intact.
            #
            # Dynamo Tensor Hooks Workflow:
            # - Functions passed to register_hook are lifted globally.
            # - For tensors with sources:
            #   - In the "side_effects" phase of codegen, we iterate over tensors with hooks to:
            #     - Generate the tensor.
            #     - Issue a register_hook call on the tensor, linking to the globally stored function.
            #     - Incorporate a handle if one was established in the eager phase.
            #  - For tensors without sources:
            #    - We don't generate any instructions for registering a hook.
            #    - Handles from intermediary hooks are NYI.
            #    - We produce a call function that utilizes the trace_wrapped higher order op, closing over it.
            #    - We then manually insert the call function above into the graph.
            # - The handle's exact user-specified name, "user_code_variable_name", is discerned and associated during STORE_FAST.
            assert tensor.source, "Hooks on non input tensors NYI - should not get here"
            cg(tensor)
            cg.extend_output([cg.create_load_attr(name)])
            cg(hook)
            cg.extend_output(create_call_function(1, True))

            # Adding the handle to the cache means RemovableHandleVariable().reconstruct() will
            # be associated with the return value of register_hook().  This consumes the top of stack.
            cg.add_cache(handle)

    def codegen_update_mutated(self, cg: PyCodegen):
        suffixes = []
        for var in self._get_modified_vars():
            if isinstance(var, variables.ListVariable):
                # old[:] = new
                cg(var, allow_cache=False)
                cg(var.mutable_local.source)  # type: ignore[attr-defined]
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

                cg(var.mutable_local.source)  # type: ignore[attr-defined]
                cg.extend_output([create_load_method("update")])
                cg(var, allow_cache=False)

                cg(var.mutable_local.source)  # type: ignore[attr-defined]
                cg.extend_output([create_load_method("clear")])

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
                    elif (
                        isinstance(var, variables.UserDefinedObjectVariable)
                        and var.needs_slow_setattr()
                    ):
                        # __setattr__ is defined on this object, so call object.__setattr__ directly
                        cg.load_import_from("builtins", "object")
                        cg.load_method("__setattr__")
                        cg(var.mutable_local.source)  # type: ignore[attr-defined]
                        cg(variables.ConstantVariable(name))
                        cg(value)
                        suffixes.append(
                            [*create_call_method(3), create_instruction("POP_TOP")]
                        )
                    else:
                        cg.tx.output.update_co_names(name)
                        cg(value)
                        cg(var.mutable_local.source)
                        suffixes.append([create_instruction("STORE_ATTR", argval=name)])
            elif isinstance(var, variables.TupleIteratorVariable):
                for _ in range(var.index):
                    cg.load_import_from(utils.__name__, "iter_next")
                    cg(var.mutable_local.source)  # type: ignore[attr-defined]
                    cg.call_function(1, True)
                    cg.pop_top()
            else:
                raise AssertionError(type(var))

        # do all the actual mutations at the very end to handle dependencies
        for suffix in reversed(suffixes):
            cg.extend_output(suffix)

    def is_empty(self):
        return not (
            any(map(self.is_modified, self.id_to_variable.values()))
            or self.tensor_hooks
            or self.save_for_backward
            or self.tensor_hooks
        )

    def clear(self):
        self.keepalive.clear()
        self.id_to_variable.clear()
