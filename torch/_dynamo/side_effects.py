import collections
import dataclasses
import inspect
from typing import Any

import torch.nn

from . import utils, variables
from .bytecode_transformation import create_instruction
from .codegen import PyCodegen
from .source import LocalSource, Source
from .utils import object_new
from .variables.base import VariableTracker


@dataclasses.dataclass
class MutableSideEffects:
    """
    VariableTracker.mutable_local marker to indicate a list passed as
    an input that if we mutate we need to re-apply those mutations after
    the graph runs.
    """

    source: Source
    is_modified: bool = False

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self is other


@dataclasses.dataclass
class AttributeMutation:
    """
    VariableTracker.mutable_local marker to track changes to attributes
    """

    source: Source


class AttributeMutationExisting(AttributeMutation):
    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self is other


@dataclasses.dataclass
class AttributeMutationNew(AttributeMutation):
    cls_source: Source

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self is other


class SideEffects(object):
    """
    Track side effects (list mutation, setattr, etc) that need to be
    applied after an FX graph is run.
    """

    def __init__(self, id_to_variable=None, store_attr_mutations=None, keepalive=None):
        super(SideEffects, self).__init__()
        self.id_to_variable = id_to_variable or collections.OrderedDict()
        self.store_attr_mutations = store_attr_mutations or collections.OrderedDict()
        self.keepalive = keepalive or []

    def clone(self):
        """Create a shallow copy"""
        return self.__class__(
            id_to_variable=collections.OrderedDict(self.id_to_variable),
            store_attr_mutations=collections.OrderedDict(
                (k, collections.OrderedDict(v))
                for k, v in self.store_attr_mutations.items()
            ),
            keepalive=list(self.keepalive),
        )

    def apply(self, fn, cache=None):
        if cache is None:
            cache = dict()

        self.id_to_variable = collections.OrderedDict(
            (k, VariableTracker.apply(fn, v, cache))
            for k, v in self.id_to_variable.items()
        )
        self.store_attr_mutations = collections.OrderedDict(
            (k, VariableTracker.apply(fn, v, cache))
            for k, v in self.store_attr_mutations.items()
        )

    def __contains__(self, item):
        return id(item) in self.id_to_variable

    def __getitem__(self, item):
        return self.id_to_variable[id(item)]

    def store_attr(self, item: VariableTracker, name: str, value: VariableTracker):
        assert self.is_attribute_mutation(item)
        if item.mutable_local not in self.store_attr_mutations:
            self.store_attr_mutations[item.mutable_local] = collections.OrderedDict()
        self.store_attr_mutations[item.mutable_local][name] = value

    def load_attr(self, item, name):
        assert self.is_attribute_mutation(item)
        return self.store_attr_mutations[item.mutable_local][name]

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
        obj = object_new(user_cls)
        variable = variable_cls(
            obj, mutable_local=AttributeMutationNew(None, cls_source), **options
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
                cg.extend_output([create_instruction("CALL_FUNCTION", 0)])
                cg.add_cache(var)
                if isinstance(var.mutable_local, AttributeMutationNew):
                    var.mutable_local.source = LocalSource(cg.tempvars[var])
            elif isinstance(var.mutable_local, AttributeMutationNew):
                cg.load_import_from(utils.__name__, "object_new")
                cg(var.mutable_local.cls_source)
                cg.extend_output([create_instruction("CALL_FUNCTION", 1)])
                cg.add_cache(var)
                var.mutable_local.source = LocalSource(cg.tempvars[var])
            elif var in cg.tempvars:
                assert cg.tempvars.get(var) is None
                # subsequent usage should point to the original variable
                cg(var.mutable_local.source)
                cg.add_cache(var)

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
                        create_instruction("BUILD_SLICE", 2),
                    ]
                )
                suffixes.append([create_instruction("STORE_SUBSCR")])
            elif isinstance(var, variables.ConstDictVariable):
                cg.tx.output.update_co_names("clear")
                cg.tx.output.update_co_names("update")

                cg(var.mutable_local.source)
                cg.extend_output([create_instruction("LOAD_METHOD", "update")])
                cg(var, allow_cache=False)

                cg(var.mutable_local.source)
                cg.extend_output([create_instruction("LOAD_METHOD", "clear")])

                suffixes.append(
                    [
                        create_instruction("CALL_METHOD", 0),  # clear
                        create_instruction("POP_TOP"),
                        create_instruction("CALL_METHOD", 1),  # update
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
                        suffixes.append([create_instruction("STORE_GLOBAL", name)])
                    else:
                        cg.tx.output.update_co_names(name)
                        cg(value)
                        cg(var.mutable_local.source)
                        suffixes.append([create_instruction("STORE_ATTR", name)])
            else:
                raise AssertionError(type(var))

        # do all the actual mutations at the very end to handle dependencies
        for suffix in reversed(suffixes):
            cg.extend_output(suffix)

    def is_empty(self):
        return not any(map(self.is_modified, self.id_to_variable.values()))
