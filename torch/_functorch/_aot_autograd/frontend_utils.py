# mypy: ignore-errors

from collections.abc import KeysView
from contextlib import contextmanager
from typing import Any, Optional

import torch
import torch.utils._pytree as pytree
from torch._guards import detect_fake_mode
from torch._subclasses import FakeTensor, FakeTensorMode
from torch.fx.experimental.proxy_tensor import _pytree_subclasses_that_lose_info
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torch.utils._python_dispatch import is_traceable_wrapper_subclass

from .. import config
from .schemas import AOTConfig, FakifiedFlatArgs


static_inputs_log = torch._logging.getArtifactLogger(
    __name__, "cudagraph_static_inputs"
)


def process_inputs(
    flat_args: list[Any],
    aot_config: AOTConfig,
    fake_mode: FakeTensorMode,
    shape_env: Optional[ShapeEnv],
    ignore_shape_env: bool = False,
) -> FakifiedFlatArgs:
    with fake_mode:

        def convert(idx, x):
            if shape_env is not None and not ignore_shape_env:
                from torch._dynamo.source import ConstantSource

                if isinstance(x, int):
                    # We always specialize on scalar values in export.
                    if aot_config.is_export:
                        return x
                    source = ConstantSource(f"sym_{idx}")
                    return shape_env.create_symintnode(
                        shape_env.create_symbol(x, source), hint=x, source=source
                    )
            if isinstance(x, torch.ScriptObject):
                return torch._library.fake_class_registry.maybe_to_fake_obj(
                    fake_mode, x
                )
            if not isinstance(x, torch.Tensor):
                return x
            if isinstance(x, FakeTensor):
                assert x.fake_mode is fake_mode
                return x
            if is_traceable_wrapper_subclass(x):
                attrs, _ = x.__tensor_flatten__()
                if all(isinstance(getattr(x, attr), FakeTensor) for attr in attrs):
                    assert all(
                        getattr(x, attr).fake_mode is fake_mode for attr in attrs
                    )
                    return x

            # see note [Tensor Fakification and Symbol Caching]
            symbolic_context = None
            source = None
            trace = True
            if tracing_context := torch._guards.TracingContext.try_get():
                if x in tracing_context.tensor_to_context:
                    symbolic_context = tracing_context.tensor_to_context[x]
                    source = symbolic_context.tensor_source
                    # We already fakeified this tensor in Dynamo, don't
                    # dump the trace for it again
                    trace = False
            if (
                idx < aot_config.num_params_buffers
                and config.static_weight_shapes
                and not symbolic_context
            ):
                # TODO: Ensure that this codepath is never exercised from
                # Dynamo
                return fake_mode.from_tensor(x, static_shapes=True)

            result = fake_mode.from_tensor(
                x,
                static_shapes=ignore_shape_env,
                symbolic_context=symbolic_context,
                source=source,
                trace=trace,
            )
            return result

        return FakifiedFlatArgs([convert(idx, x) for idx, x in enumerate(flat_args)])


def construct_fake_mode(
    flat_args: list[Any], aot_config: AOTConfig
) -> tuple[FakeTensorMode, Optional[ShapeEnv]]:
    fake_mode = detect_fake_mode(flat_args)
    if fake_mode is None:
        shape_env = ShapeEnv() if aot_config.dynamic_shapes else None
        fake_mode = FakeTensorMode(shape_env=shape_env)
    else:
        shape_env = fake_mode.shape_env
    return (fake_mode, shape_env)


def _try_get_metadata_from_dynamo(
    mod: torch.nn.Module, param_keys: KeysView[str], full_args_num: int
) -> tuple[Optional[list[torch._guards.Source]], list[int]]:
    """
    Metadata is forwarded from Dynamo to AOTDispatch via special fields on GraphModule.
    We first verify that `mod` does come from Dynamo, then we handle cases where
    metadata might be missing.

    Returns:
        aot_autograd_arg_pos_to_source: used to dedup params and their guards
        static_input_indices: used to identify static inputs for cudagraphs
    """
    # Note [Assumption on Dynamo Metadata]
    # This function assumes a graph module from dynamo provides `dynamo_compiled_id`,
    # _param_name_to_source, and every placeholder node has `_dynamo_source` attributes.
    # When gm is modified (e.g., DDPOptimizer via split_module), metadata needs to
    # be propagated in order to be recognized as a dynamo graph

    if not (isinstance(mod, torch.fx.GraphModule) and "dynamo_compile_id" in mod.meta):
        # graph was not captured by dynamo
        return None, []

    if not hasattr(mod, "_param_name_to_source"):
        # is from export
        return None, []

    # We now know this came from dynamo, and (1) we care about guards,
    # so setting up aot_autograd_arg_pos_to_source for downstream dedup guards
    # can now be done safely. (2) Dynamo logic protects the 1:1 sizing below.
    # Additionally, we mark static indices for cudagraphs.
    param_name_to_source = mod._param_name_to_source
    seen_sources = set()

    aot_autograd_arg_pos_to_source = []
    static_input_indices = []
    # Collect the new inputs lifted by aotdispatch
    for i, name in enumerate(param_keys):
        assert name in param_name_to_source, f"{name} not found."
        source = param_name_to_source[name]
        assert source not in seen_sources, source
        seen_sources.add(source)
        aot_autograd_arg_pos_to_source.append(source)

        static_input_indices.append(i)

    # Collect the dynamo graph inputs
    # TODO(mlazos): Revisit if this is still needed. With Dynamo install ID
    # matched tensors back into the Fx graph, this might not be necessary.
    for pos, node in enumerate(mod.graph.find_nodes(op="placeholder")):
        assert hasattr(node, "_dynamo_source")
        source = node._dynamo_source
        # `source`` specifies the source from user code. ddp optimizer may have
        # intermediate values becoming submodule placeholders which does not
        # have a source
        assert source is None or source not in seen_sources, source
        seen_sources.add(source)
        aot_autograd_arg_pos_to_source.append(source)
        source_name = source.name() if source else str(source)

        # input[i] in dynamo is now:
        # input[i + len(extra_params)] in AOT,
        # where extra_params are the params/buffers that dynamo baked into the
        # OutputGraph
        actual_pos = pos + len(param_keys)

        if "tensor_dict" in node.meta and node.meta["tensor_dict"].get(
            "_dynamo_static_input_type", None
        ):
            static_inputs_log.debug(
                "Adding static input pos %s for source %s", actual_pos, source_name
            )
            static_input_indices.append(actual_pos)
        else:
            static_inputs_log.debug(
                "Non-static input pos %s for source %s", actual_pos, source_name
            )

    assert full_args_num == len(aot_autograd_arg_pos_to_source)
    return aot_autograd_arg_pos_to_source, static_input_indices


@contextmanager
def _detect_attribute_assignment(mod: torch.nn.Module):
    # Do not allow assignment of tensor attributes during export unless
    # the attribute is registered as a buffer.

    NN_MODULE_STD_ATTRS = [
        "_backward_hooks",
        "_backward_pre_hooks",
        "_buffers",
        "_forward_hooks",
        "_forward_hooks_always_called",
        "_forward_hooks_with_kwargs",
        "_forward_pre_hooks",
        "_forward_pre_hooks_with_kwargs",
        "_is_full_backward_hook",
        "_load_state_dict_post_hooks",
        "_load_state_dict_pre_hooks",
        "_modules",
        "_non_persistent_buffers_set",
        "_parameters",
        "_state_dict_hooks",
        "_state_dict_pre_hooks",
        "training",
    ]
    NN_MODULE_LAZY_STD_ATTRS = [
        "_initialize_hook",
        "_load_hook",
    ]
    STD_ATTRS = {
        *NN_MODULE_STD_ATTRS,
        *NN_MODULE_LAZY_STD_ATTRS,
    }

    def _get_attributes(mod):
        # return any attributes of a module that are not standard attributes
        return {k: v for k, v in mod.__dict__.items() if k not in STD_ATTRS}

    # save state of attributes before enter
    snapshot = pytree.tree_map(
        lambda x: x,
        _get_attributes(mod),
        is_leaf=lambda x: type(x) in _pytree_subclasses_that_lose_info,
    )
    try:
        yield
    finally:
        # after exit, compare state of attributes with snapshot
        # to detect which tensor attributes were assigned
        assigned_tensor_attributes = []

        def _collect_assigned_tensor_attributes(kp, v, _v):
            if _v is not v:
                attr, *rest = kp
                if isinstance(v, torch.Tensor):
                    assigned_tensor_attributes.append(
                        f"self.{attr.key}{pytree.keystr(rest)}"
                    )
                # TODO(avik): Assigning all other types are allowed right now.
                # Maybe in the future we want to limit this to primitive types?
            return v

        new_attrs = _get_attributes(mod)
        if len(new_attrs) != len(snapshot):
            added_attrs = new_attrs.keys() - snapshot.keys()
            deleted_attrs = snapshot.keys() - new_attrs.keys()

            if len(added_attrs) > 0:
                raise ValueError(
                    f"During torch.export, following attrs were created in the model.forward: {added_attrs} "
                    f"Such attributes must be registered as buffers using the `register_buffer` "
                    f"API and must be initialized at model.__init__ "
                    f"(https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_buffer)."
                )

            if len(deleted_attrs) > 0:
                raise ValueError(
                    f"During torch.export, following attrs were deleted in the model.forward: {deleted_attrs} "
                    f"Such attributes must be registered as buffers using the `register_buffer` "
                    f"API and must be initialized at model.__init__ "
                    f"(https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_buffer)."
                )

        pytree.tree_map_with_path(
            _collect_assigned_tensor_attributes, snapshot, new_attrs
        )
        # restore state of all attributes (including, e.g., of primitive types)
        mod.__dict__.update(snapshot)

        if assigned_tensor_attributes:
            if len(assigned_tensor_attributes) > 1:
                noun, verb = "attributes", "were"
            else:
                noun, verb = "attribute", "was"
            raise ValueError(
                f"The tensor {noun} {', '.join(assigned_tensor_attributes)} {verb} assigned during export. "
                "Such attributes must be registered as buffers using the `register_buffer` API "
                "(https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_buffer)."
            )
