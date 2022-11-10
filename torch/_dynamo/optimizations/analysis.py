import copy
import functools
import itertools
import operator

import torch
from torch.fx.node import map_aggregate
from torch.fx.passes.shape_prop import _extract_tensor_metadata, ShapeProp
from torch.multiprocessing.reductions import StorageWeakRef
from torch.utils._pytree import tree_map

from .. import config
from ..utils import clone_inputs, fake_tensors_available

if fake_tensors_available:
    from torch._subclasses import FakeTensorMode  # noqa: F401

    from ..utils import deepcopy_to_fake_tensor


class ShapeAliasingAndMutationProp(ShapeProp):
    def __init__(self, *args, **kwargs):
        super(ShapeAliasingAndMutationProp, self).__init__(*args, **kwargs)
        self.input_alias_groups = set()
        self.storage_to_alias_group = dict()
        self.make_alias_group = itertools.count(1)

    def tensor_alias_group(self, value: torch.Tensor):
        """Assign a unique identifier to the storage of a given tensor"""
        storage = StorageWeakRef(value._typed_storage())
        alias_group = self.storage_to_alias_group.get(storage)
        if alias_group is None:
            alias_group = next(self.make_alias_group)
            self.storage_to_alias_group[storage] = alias_group
        return alias_group

    def placeholder(self, target, args, kwargs):
        value = super().placeholder(target, args, kwargs)
        assert isinstance(value, torch.Tensor)
        self.input_alias_groups.add(self.tensor_alias_group(value))
        return value

    def run_node(self, n: torch.fx.Node):
        args, kwargs = self.fetch_args_kwargs_from_env(n)
        tensor_args = self.extract_tensors((args, kwargs))

        input_versions1 = [obj._version for obj in tensor_args]
        result = getattr(self, n.op)(n.target, args, kwargs)
        input_versions2 = [obj._version for obj in tensor_args]

        n.meta["type"] = type(result)
        n.meta["alias_groups"] = {
            self.tensor_alias_group(obj) for obj in self.extract_tensors(result)
        }

        if (
            not n.meta["alias_groups"]
            and n.op == "call_function"
            and n.target == operator.setitem
        ):
            n.meta["alias_groups"] = {self.tensor_alias_group(tensor_args[0])}

        n.meta["mutates_alias_groups"] = {
            self.tensor_alias_group(tensor)
            for tensor, v1, v2 in zip(tensor_args, input_versions1, input_versions2)
            if v1 != v2
        }
        # Partial mutation refers to the mutation caused by getitem that can
        # potentially result in changing only a slice of the original tensor
        n.meta["partial_mutation"] = False

        def visit_arg(arg: torch.fx.Node):
            if (
                arg.op == "call_function" and arg.target == operator.getitem
            ) or arg.meta["partial_mutation"]:
                if bool(n.meta["mutates_alias_groups"] & arg.meta["alias_groups"]):
                    n.meta["partial_mutation"] = True

        torch.fx.map_arg((n.args, n.kwargs), visit_arg)
        n.meta["is_input_alias"] = bool(
            self.input_alias_groups & n.meta["alias_groups"]
        )
        n.meta["is_input_mutation"] = bool(
            self.input_alias_groups & n.meta["mutates_alias_groups"]
        )
        n.meta["is_mutation"] = bool(n.meta["mutates_alias_groups"])
        n.meta["tensor_metas"] = [
            _extract_tensor_metadata(obj) for obj in self.extract_tensors(result)
        ]
        tensors = self.extract_tensors(result)
        if tensors:
            n.meta["device"] = tensors[0].device
            n.meta["dtype"] = tensors[0].dtype

        return result

    @staticmethod
    def extract_tensors(result):
        """Return a flat list of tensors found in some nested data structure"""
        seen = set()
        tensors = []

        def visit(obj):
            if isinstance(obj, torch.Tensor) and id(obj) not in seen:
                seen.add(id(obj))
                tensors.append(obj)

        map_aggregate(result, visit)
        return tensors

    def run(self, *args):
        try:
            super().run(*args)
        finally:
            # cleanup
            self.env.clear()


def has_mutation(gm, example_inputs, inputs_only=False):
    """Check if the graph module has any form of mutation.  If inputs_only is
    true, we only check for mutation of inputs"""
    # TODO - moco gives bad accuracy with Aliasing. gm is getting mutated in a bad way.

    if fake_tensors_available and config.fake_tensor_propagation:

        def _wrap_to_fake_tensor(t, *, f_mode):
            if type(t) in (torch.Tensor, torch.nn.Parameter):
                static_shapes_ = config.dynamic_shapes is False
                return fake_mode.from_tensor(
                    t, static_shapes=config.dynamic_shapes is not False
                )
            else:
                return t

        # Our analysis pass should use dynamic shape tensor inputs
        # when dynamic shapes are enabled.
        # We don't actually care about the guards that are created
        # on those shapes though, so just create a fresh ShapeEnv here.
        from torch.fx.experimental.symbolic_shapes import ShapeEnv

        with FakeTensorMode(
            shape_env=ShapeEnv() if config.dynamic_shapes else None
        ) as fake_mode:
            pass
        fake_wrapper = functools.partial(_wrap_to_fake_tensor, f_mode=fake_mode)
        example_inputs = tree_map(fake_wrapper, example_inputs)
        new_gm = deepcopy_to_fake_tensor(gm, fake_mode)
        with fake_mode.restore() if hasattr(fake_mode, "restore") else fake_mode:
            ShapeAliasingAndMutationProp(new_gm).run(*example_inputs)
    else:
        # Clone the inputs such that intermediate tensors (not leaf tensors) with
        # requires_grad to True are now converted to False to avoid Runtime Error
        # like "leaf variable that requires grad is inplace modified"
        example_inputs = clone_inputs(example_inputs)
        new_gm = copy.deepcopy(gm)
        example_inputs = copy.deepcopy(example_inputs)
        ShapeAliasingAndMutationProp(new_gm).run(*example_inputs)

    for node in new_gm.graph.nodes:
        if node.meta["is_mutation"] or node.meta["is_input_mutation"]:
            if inputs_only:
                if node.meta["is_input_alias"]:
                    return True
            else:
                return True
    return False
