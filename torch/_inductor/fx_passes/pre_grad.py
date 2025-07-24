# mypy: allow-untyped-defs
import copy
import itertools
import logging
import types
from collections.abc import Sequence
from typing import Optional

import torch
from torch._dynamo.utils import counters, detect_fake_mode
from torch._logging import trace_structured
from torch.fx.experimental.optimization import (
    matches_module_pattern,
    replace_node_module,
)
from torch.fx.passes.graph_transform_observer import GraphTransformObserver
from torch.fx.passes.shape_prop import ShapeProp
from torch.nn import functional as F
from torch.nn.utils.fusion import fuse_conv_bn_eval, fuse_conv_bn_weights

from .. import config
from ..fx_utils import matches_module_function_pattern
from ..pattern_matcher import (
    init_once_fakemode,
    PatternMatcherPass,
    stable_topological_sort,
)
from ..utils import is_cpu_device, pass_execution_and_save
from .group_batch_fusion import group_batch_fusion_passes, PRE_GRAD_FUSIONS
from .misc_patterns import numpy_compat_normalization
from .split_cat import PRE_GRAD_PATTERNS


log = logging.getLogger(__name__)

efficient_conv_bn_eval_pass = PatternMatcherPass(
    pass_name="efficient_conv_bn_eval_pass"
)

fuse_split_linear_add_pass = PatternMatcherPass(
    pass_name="fuse_split_linear_add_pass",
)
fuse_chunk_squeeze_cat_pass = PatternMatcherPass(
    pass_name="fuse_chunk_squeeze_cat_pass",
)
remove_reshape_pass = PatternMatcherPass(
    pass_name="remove_reshape_pass",
)

# based on predispatch aten IR
normalization_pass_aten = PatternMatcherPass(pass_name="normalization_pass_aten")
merge_splits_pass_aten = PatternMatcherPass(pass_name="merge_splits_pass_aten")
split_cat_pass_aten = PatternMatcherPass(pass_name="split_cat_pass_aten")
unbind_stack_pass_aten = PatternMatcherPass(pass_name="unbind_stack_pass_aten")
merge_getitem_cat_pass_aten = PatternMatcherPass(
    pass_name="merge_getitem_cat_pass_aten"
)
merge_stack_tahn_unbind_pass_aten = PatternMatcherPass(
    pass_name="merge_stack_tahn_unbind_pass_aten"
)
mutate_cat_pass_aten = PatternMatcherPass(pass_name="mutate_cat_pass_aten")
remove_split_with_size_one_pass_aten = PatternMatcherPass(
    pass_name="remove_split_with_size_one_pass_aten"
)


def save_inductor_dict(pass_to_compare=None):
    if not pass_to_compare:
        pass_to_compare = list(config.pre_grad_fusion_options.keys()) + list(
            config.post_grad_fusion_options.keys()
        )
    return {p: dict(counters["inductor"]).get(p, 0) for p in pass_to_compare}


def is_same_dict(inductor_dict, optimus_dict):
    for pass_name, count in optimus_dict.items():
        if count != dict(inductor_dict).get(pass_name, 0):
            return False
    return True


def shape_prop(mod) -> None:
    return None


def normalize_node_kwargs_pass(graph):
    return None


def fuse_parallel_linear_pass(graph):
    return None


def remove_split_ops(graph, shape_prop):
    return None


def remove_split_ops_pass(graph):
    remove_split_ops(graph.owning_module, shape_prop)


def fuse_chunk_reshape_unsqueeze_concat_pass(graph):
    return None


def fuse_chunk_reshape_concat_pass(graph):
    return None


def remove_noop_pass(graph):
    return None


def stack_to_unsqueeze_pass(graph):
    return None


def merge_concats_pass(graph):
    return None


def relu_nan_to_num(graph):
    return None


def fuse_split_getitem_squeeze_cat(graph):
    return None


def use_triton_dot_compress(graph):
    return None


def use_triton_lce_replace_simple_LCE_helper(gm, shape_prop):
    return None


def use_triton_lce_replace_simple_LCE(graph):
    return use_triton_lce_replace_simple_LCE_helper(graph.owning_module, shape_prop)


def use_triton_lce_replace_normal_LCE_helper(gm, shape_prop):
    return None


def use_triton_lce_replace_normal_LCE(graph):
    return use_triton_lce_replace_simple_LCE_helper(graph.owning_module, shape_prop)


def use_matmul_lce_replace_normal_LCE(graph):
    return None


def use_matmul_fuse_lce_replace_first_LCE(graph):
    return None


@init_once_fakemode
def lazy_init():
    from . import efficient_conv_bn_eval, split_cat  # noqa: F401

    if config.is_fbcode():
        from . import fb  # type: ignore[attr-defined]  # noqa: F401


def _get_pass_name_func(p):
    if isinstance(p, PatternMatcherPass):
        pass_name = p.pass_name
        pass_func = p.apply
    elif isinstance(p, types.FunctionType):
        pass_name = p.__name__.lstrip("_")
        pass_func = p
    else:
        pass_name = None
        pass_func = None

    return pass_name, pass_func


def _run_pre_dispatch_passes(
    gm: torch.fx.GraphModule,
    example_inputs: Sequence[object] = (),
    add_passes: Optional[str] = None,
    remove_passes: Optional[str] = None,
) -> None:
    # order matters
    default_pass_list = [
        # normalize passes, must be called as the first passes
        normalization_pass_aten,
        normalize_node_kwargs_pass,
        remove_noop_pass,
        relu_nan_to_num,
        fuse_chunk_reshape_concat_pass,
        group_batch_fusion_passes,
        normalize_node_kwargs_pass,
        fuse_chunk_squeeze_cat_pass,
        merge_concats_pass,
        fuse_split_linear_add_pass,
        remove_reshape_pass,
        fuse_parallel_linear_pass,
        remove_split_ops_pass,
        stack_to_unsqueeze_pass,  # run before fuse_chunk_reshape_unsqueeze_concat_pass
        fuse_chunk_reshape_unsqueeze_concat_pass,
    ]

    full_pass_list = default_pass_list + [
        fuse_split_getitem_squeeze_cat,
        use_triton_dot_compress,
        use_triton_lce_replace_simple_LCE,
        use_triton_lce_replace_normal_LCE,
        use_matmul_fuse_lce_replace_first_LCE,
        use_matmul_lce_replace_normal_LCE,
    ]

    log.info(
        f"pre_grad_passes: add_passes: {add_passes}, remove_pass: {remove_passes}"  # noqa: G004
    )
    add_passes_list = []
    remove_passes_list = []
    if add_passes:
        add_passes_list = add_passes.split(",")
    if remove_passes:
        remove_passes_list = remove_passes.split(",")

    shape_prop = lambda mod: ShapeProp(  # noqa: E731
        gm=mod,
        # pyre-fixme[16]: Module `torch._dynamo.utils` has no attribute `detect_fake_mode`
        fake_mode=detect_fake_mode(example_inputs),
    ).propagate(*tuple(example_inputs))

    for p in default_pass_list:
        pass_name, pass_func = _get_pass_name_func(p)
        # should not happen
        if pass_name is None or pass_func is None:
            continue
        if pass_name in remove_passes_list:
            continue
        pass_execution_and_save(
            pass_func,
            gm,
            example_inputs,
            f"[Pre grad(predispatch IR)] Apply {pass_name} pass",
        )

    for p in full_pass_list:
        pass_name, pass_func = _get_pass_name_func(p)
        if pass_name is None or pass_func is None:
            continue
        if pass_name in add_passes_list:
            pass_execution_and_save(
                pass_func,
                gm,
                example_inputs,
                f"[Pre grad(predispatch IR)] Apply {pass_name} pass",
            )

    # Remove noops at the end, which may be generated other passes.
    pass_execution_and_save(
        remove_noop_pass,
        gm,
        example_inputs,
        "[Pre grad(predispatch IR)]Apply remove_noop pass",
    )
    shape_prop(gm)


def pre_grad_passes(
    gm: torch.fx.GraphModule,
    example_inputs: Sequence[object] = (),
    add_passes: Optional[str] = None,
    remove_passes: Optional[str] = None,
) -> torch.fx.GraphModule:
    """
    Apply passes on the input FX graph using Torch IR.

    WARNING:
    The IR before grad is not functional or normalized, so it is harder
    to write passes on this IR.  Passes must be safe with respect to
    aliasing and mutation and need to handle all possible arg schemas.

    Consider adding a new pass to post_grad.py or joint_graph.py which
    are after functionalization and normalization.
    """
    if config.pattern_matcher:
        lazy_init()
        if hasattr(
            config, "fx_passes_numeric_check"
        ) and config.fx_passes_numeric_check.get("pre_grad", False):
            gm_before_fx_passes = gm.__copy__()
        # explicitly run with predispatch atenIR based passes
        if config.is_predispatch:
            _run_pre_dispatch_passes(gm, example_inputs, add_passes, remove_passes)
        else:
            # We only log the graph with changes to avoid the excessive compilation time
            # https://fb.workplace.com/groups/257735836456307/permalink/633533465543207/
            if example_inputs is not None:
                gm = fuse_fx(gm, example_inputs)
            numpy_compat_normalization(gm.graph)
            # We should always do the normalization_pass first
            if "normalization_pass" in config.pre_grad_fusion_options:
                pattern_matcher_pass = PRE_GRAD_PATTERNS["normalization_pass"]
                pattern_matcher_pass.apply(gm.graph)  # type: ignore[arg-type]
            group_batch_fusion_passes(gm.graph, pre_grad=True)
            for pass_name in config.pre_grad_fusion_options:
                # skip all patterns for group batch fusions
                if pass_name in PRE_GRAD_FUSIONS or pass_name == "normalization_pass":
                    continue
                pattern_matcher_pass = PRE_GRAD_PATTERNS[pass_name]
                inductor_before_change = save_inductor_dict(
                    [pattern_matcher_pass.pass_name]
                )
                # we support run same pattern multiple times, the default is to run only once
                counter = config.pre_grad_fusion_options[pass_name].get("counter", 1)
                for _ in range(counter):
                    pattern_matcher_pass.apply(gm.graph)  # type: ignore[arg-type]
                if not is_same_dict(counters["inductor"], inductor_before_change):
                    trace_structured(
                        "artifact",
                        metadata_fn=lambda: {
                            "name": f"{pattern_matcher_pass.pass_name}_pre_grad",
                            "encoding": "string",
                        },
                        payload_fn=lambda: gm.print_readable(
                            print_output=False, include_stride=True, include_device=True
                        ),
                    )
            # TODO: move efficient_conv_bn_eval_pass to the fusions dict too.
            efficient_conv_bn_eval_pass.apply(gm.graph)  # type: ignore[arg-type]

    if config.pre_grad_custom_pass is not None:
        with GraphTransformObserver(gm, "pre_grad_custom_pass"):
            config.pre_grad_custom_pass(gm.graph)
    stable_topological_sort(gm.graph)

    from .quantization import quant_lift_up

    quant_lift_up(gm)

    gm.graph.lint()
    gm.recompile()

    if (
        config.pattern_matcher
        and hasattr(config, "fx_passes_numeric_check")
        and config.fx_passes_numeric_check.get("pre_grad", False)
        and example_inputs is not None
    ):
        from .numeric_utils import numeric_check_if_enabled

        gm_after_fx_passes = gm.__copy__()
        numeric_check_if_enabled(
            gm_before_fx_passes,  # type: ignore[possibly-undefined]
            gm_after_fx_passes,
            example_inputs,
            config.fx_passes_numeric_check.get("num_iterations", 1),
            config.fx_passes_numeric_check.get("precision", 1e-4),
        )

    return gm


def fuse_fx(gm: torch.fx.GraphModule, example_inputs) -> torch.fx.GraphModule:
    is_cpu = is_cpu_device(example_inputs)
    # pyre-fixme[16]: Module `torch._dynamo.utils` has no attribute `detect_fake_mode`
    fake_mode = detect_fake_mode(example_inputs)

    gm = sink_cat_after_pointwise(gm)
    if config.permute_fusion and not is_cpu:
        # For linear permute fusion, we need to check input info to identify
        # and perform proper permutation/transpose
        ShapeProp(gm, fake_mode=fake_mode).propagate(*example_inputs)
        with GraphTransformObserver(gm, "linear_permute_fusion"):
            gm = linear_permute_fusion(gm)
        with GraphTransformObserver(gm, "permute_linear_fusion"):
            gm = permute_linear_fusion(gm)
        with GraphTransformObserver(gm, "permute_matmul_fusion"):
            gm = permute_matmul_fusion(gm)

    # make sure the autograd is disabled.
    if torch.is_grad_enabled() or not is_cpu:
        return gm
    if config.freezing:
        with GraphTransformObserver(gm, "remove_identity"):
            gm = remove_identity(gm)
        with GraphTransformObserver(gm, "fuse_conv_bn"):
            gm = fuse_conv_bn(gm)
    return gm


def fetch_attr(target: str, mod):
    target_atoms = target.split(".")
    attr_itr = mod
    for i, atom in enumerate(target_atoms):
        if not hasattr(attr_itr, atom):
            raise RuntimeError(
                f"Node referenced nonexistent target {'.'.join(target_atoms[:i])}"
            )
        attr_itr = getattr(attr_itr, atom)
    return attr_itr


def remove_identity(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    """
    Removes all identity layers from the module.
    """

    class IdentityRemover(torch.fx.Transformer):
        def call_module(self, target, args, kwargs):
            assert isinstance(target, str)
            if isinstance(self.submodules[target], torch.nn.Identity):
                assert len(args) == 1
                return args[0]
            return super().call_module(target, args, kwargs)

    return IdentityRemover(gm).transform()


def fuse_conv_bn(gm: torch.fx.GraphModule, inplace=False) -> torch.fx.GraphModule:
    """
    Fuses Convolution/BN layers for inference purposes.
    """
    modules_patterns = [
        (torch.nn.Conv1d, torch.nn.BatchNorm1d),
        (torch.nn.Conv2d, torch.nn.BatchNorm2d),
        (torch.nn.Conv3d, torch.nn.BatchNorm3d),
    ]
    module_function_patterns = [
        (torch.nn.Conv1d, F.batch_norm),
        (torch.nn.Conv2d, F.batch_norm),
        (torch.nn.Conv3d, F.batch_norm),
    ]
    modules = dict(gm.named_modules())

    class ConvBNFusion:
        def __init__(
            self,
            bn_node,
            conv_module,
            bn_module=None,  # For BN Module
            bn_running_mean=None,  # For Functional BN
            bn_running_var=None,
            bn_eps=None,
            bn_weight=None,
            bn_bias=None,
        ) -> None:
            self.bn_nodes = [
                bn_node,
            ]
            self.conv_module = conv_module
            self.bn_module = bn_module
            self.bn_running_mean = bn_running_mean
            self.bn_running_var = bn_running_var
            self.bn_eps = bn_eps
            self.bn_weight = bn_weight
            self.bn_bias = bn_bias
            self.fusion_enabled = True

        def add_bn_node(self, bn_node):
            self.bn_nodes.append(bn_node)

        def disable_fusion(self):
            self.fusion_enabled = False

        def is_fusion_enabled(self):
            return self.fusion_enabled

    conv_bn_to_fuse: dict[int, ConvBNFusion] = {}
    for pattern in modules_patterns:
        conv_bn_to_fuse.clear()
        for node in gm.graph.nodes:
            if matches_module_pattern(pattern, node, modules):
                if len(node.args[0].users) > 1:  # Output of conv is used by other nodes
                    continue
                conv = modules[node.args[0].target]
                bn = modules[node.target]
                eval_mode = all(not n.training for n in [conv, bn])
                if not eval_mode:
                    continue
                if not bn.track_running_stats:
                    continue

                # Do hash based on the module name of conv
                hash_id = hash(node.args[0].target)
                if hash_id not in conv_bn_to_fuse:
                    conv_bn_to_fuse[hash_id] = ConvBNFusion(node, conv, bn)
                else:
                    if bn == conv_bn_to_fuse[hash_id].bn_module:
                        # Do fusion if same bn module
                        conv_bn_to_fuse[hash_id].add_bn_node(node)
                    else:
                        # Disable the conv bn folding if conv shared by different bn
                        conv_bn_to_fuse[hash_id].disable_fusion()

        for conv_bn_fusion in conv_bn_to_fuse.values():
            if conv_bn_fusion.is_fusion_enabled():
                bn_nodes = conv_bn_fusion.bn_nodes
                conv = conv_bn_fusion.conv_module
                bn = conv_bn_fusion.bn_module

                fused_conv = fuse_conv_bn_eval(conv, bn)
                for bn_node in bn_nodes:
                    replace_node_module(bn_node.args[0], modules, fused_conv)
                    bn_node.replace_all_uses_with(bn_node.args[0])
                    gm.graph.erase_node(bn_node)

    gm.graph.lint()
    for pattern in module_function_patterns:
        conv_bn_to_fuse.clear()
        for node in gm.graph.nodes:
            if matches_module_function_pattern(pattern, node, modules):
                # TODO: support kwargs.
                if len(node.args) != 8:
                    continue
                conv = modules[node.args[0].target]
                bn_training = node.args[5]
                bn_eps = node.args[7]
                if conv.training or bn_training:
                    continue
                if type(bn_eps) is not float:
                    continue

                def _used_by_same_conv_module(users):
                    conv_module_name = users[0].args[0].target
                    return all(
                        conv_module_name == user.args[0].target for user in users
                    )

                bn_args_is_constant = all(
                    n.op == "get_attr"
                    and (len(n.users) == 1 or _used_by_same_conv_module(list(n.users)))
                    for n in node.args[1:5]
                )
                if not bn_args_is_constant:
                    continue
                bn_running_mean = fetch_attr(node.args[1].target, gm)
                bn_running_var = fetch_attr(node.args[2].target, gm)
                bn_weight = fetch_attr(node.args[3].target, gm)
                bn_bias = fetch_attr(node.args[4].target, gm)
                if bn_running_mean is None or bn_running_var is None:
                    continue

                # Do hash based on the module name of conv
                hash_id = hash(node.args[0].target)
                if hash_id not in conv_bn_to_fuse:
                    conv_bn_to_fuse[hash_id] = ConvBNFusion(
                        node,
                        conv,
                        bn_running_mean=bn_running_mean,
                        bn_running_var=bn_running_var,
                        bn_eps=bn_eps,
                        bn_weight=bn_weight,
                        bn_bias=bn_bias,
                    )
                else:
                    if (
                        hash(bn_running_mean)
                        == hash(conv_bn_to_fuse[hash_id].bn_running_mean)
                        and hash(bn_running_var)
                        == hash(conv_bn_to_fuse[hash_id].bn_running_var)
                        and torch.allclose(
                            torch.tensor(bn_eps),
                            torch.tensor(conv_bn_to_fuse[hash_id].bn_eps),
                        )
                        and hash(bn_weight) == hash(conv_bn_to_fuse[hash_id].bn_weight)
                        and hash(bn_bias) == hash(conv_bn_to_fuse[hash_id].bn_bias)
                    ):
                        # Do fusion if same functional bn
                        conv_bn_to_fuse[hash_id].add_bn_node(node)
                    else:
                        # Disable the conv bn folding if conv shared by different bn
                        conv_bn_to_fuse[hash_id].disable_fusion()

        for conv_bn_fusion in conv_bn_to_fuse.values():
            if conv_bn_fusion.is_fusion_enabled():
                bn_nodes = conv_bn_fusion.bn_nodes
                conv = conv_bn_fusion.conv_module
                bn_running_mean = conv_bn_fusion.bn_running_mean
                bn_running_var = conv_bn_fusion.bn_running_var
                bn_eps = conv_bn_fusion.bn_eps
                bn_weight = conv_bn_fusion.bn_weight
                bn_bias = conv_bn_fusion.bn_bias

                fused_conv = copy.deepcopy(conv)
                fused_conv.weight, fused_conv.bias = fuse_conv_bn_weights(
                    fused_conv.weight,
                    fused_conv.bias,
                    bn_running_mean,
                    bn_running_var,
                    bn_eps,
                    bn_weight,
                    bn_bias,
                )
                for bn_node in bn_nodes:
                    replace_node_module(bn_node.args[0], modules, fused_conv)
                    bn_node.replace_all_uses_with(bn_node.args[0])
                    gm.graph.erase_node(bn_node)
    gm.graph.lint()
    gm.recompile()

    return gm


class NormalizedLinearNode:
    def __init__(self, node: torch.fx.Node) -> None:
        assert node.op == "call_function"
        assert node.target in [torch.nn.functional.linear]
        self.node: torch.fx.Node = node

    def get_input(self) -> torch.fx.Node:
        if len(self.node.args) > 0:
            return self.node.args[0]  # type: ignore[return-value]
        else:
            return self.node.kwargs["input"]  # type: ignore[return-value]

    def get_weight(self) -> torch.fx.Node:
        if len(self.node.args) > 1:
            return self.node.args[1]  # type: ignore[return-value]
        else:
            return self.node.kwargs["weight"]  # type: ignore[return-value]

    def get_bias(self) -> torch.fx.Node:
        if len(self.node.args) > 2:
            return self.node.args[2]  # type: ignore[return-value]
        else:
            return self.node.kwargs["bias"] if "bias" in self.node.kwargs else None  # type: ignore[return-value]


class NormalizedMatmulNode:
    def __init__(self, node: torch.fx.Node) -> None:
        assert node.op == "call_function"
        assert node.target in [torch.bmm, torch.matmul]
        self.node: torch.fx.Node = node

    def get_input(self) -> torch.fx.Node:
        if len(self.node.args) > 0:
            return self.node.args[0]  # type: ignore[return-value]
        else:
            return self.node.kwargs["input"]  # type: ignore[return-value]

    def get_other(self) -> torch.fx.Node:
        if len(self.node.args) > 1:
            return self.node.args[1]  # type: ignore[return-value]
        else:
            return self.node.kwargs["other"]  # type: ignore[return-value]


def check_permute(node: torch.fx.Node) -> bool:
    ranks = len(node.meta["tensor_meta"].shape)
    if len(node.args) > 3:
        permutation = [node.args[i] % ranks for i in range(1, ranks + 1)]  # type: ignore[operator]
    elif (
        "permutation" in node.kwargs
        and node.kwargs["permutation"] is not None
        and len(node.kwargs["permutation"]) > 2  # type: ignore[arg-type]
    ):
        permutation = [i % ranks for i in node.kwargs["permutation"]]  # type: ignore[operator, union-attr]
    else:
        return False
    allowed_permutation = list(range(ranks))
    allowed_permutation[-1] = ranks - 2
    allowed_permutation[-2] = ranks - 1
    return permutation == allowed_permutation


def sink_cat_after_pointwise(module: torch.fx.GraphModule) -> torch.fx.GraphModule:
    def one_user(node):
        users = list(node.users)
        return users[0] if len(users) == 1 else None

    def is_view(node):
        return node.op == "call_method" and node.target == "view"

    def is_pointwise_unary(node):
        ops = "call_function", "call_method"
        pointwise = torch.relu, torch.tanh, "relu", "tanh"
        return node.op in ops and node.target in pointwise

    g = module.graph
    for node in g.nodes:
        if node.op != "call_function" or node.target != torch.cat:
            continue

        cat_or_view = node
        while True:
            user = one_user(cat_or_view)
            if not user or not is_view(user):
                break
            cat_or_view = user

        if user and is_pointwise_unary(user):
            with g.inserting_before(node):

                def cat_args(tensors, dim=0):
                    return tensors, dim

                tensors, dim = cat_args(*node.args, **node.kwargs)
                new_kwargs = {
                    name: val for name, val in user.kwargs.items() if name != "input"
                }
                new_tensors = [
                    g.create_node(user.op, user.target, args=(arg,), kwargs=new_kwargs)
                    for arg in tensors
                ]
                new_cat = g.create_node(
                    "call_function", torch.cat, args=(new_tensors, dim)
                )
                user.replace_all_uses_with(cat_or_view)
                node.replace_all_uses_with(new_cat)
                g.erase_node(user)
                g.erase_node(node)
    g.lint()
    module.recompile()
    return module


def linear_permute_fusion(module: torch.fx.GraphModule) -> torch.fx.GraphModule:
    for node in module.graph.find_nodes(op="call_method", target="permute"):
        if check_permute(node):
            if len(node.args) > 0:
                input_node = node.args[0]
            else:
                input_node = node.kwargs["input"]
            if (
                input_node.op == "call_function"
                and input_node.target == torch.nn.functional.linear
            ):
                normalized = NormalizedLinearNode(input_node)
                input = normalized.get_input()
                weight = normalized.get_weight()
                bias = normalized.get_bias()
                with module.graph.inserting_before(node):
                    fused_node = module.graph.call_function(
                        linear_transpose, args=(input, weight, bias)
                    )
                    node.replace_all_uses_with(fused_node)
                    module.graph.erase_node(node)
                    if len(input_node.users) == 0:
                        module.graph.erase_node(input_node)

    module.graph.lint()
    module.recompile()
    return module


# Y1 = X * W^T + bias
# Y2 = Y1.permute(0, 2, 1)
# ---->
# Y2 = (W * X^T + bias.unsqueeze(-1))^T
def linear_transpose(
    input: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor]
) -> torch.Tensor:
    if bias is None:
        return torch.matmul(weight, input.transpose(-1, -2))
    return torch.matmul(weight, input.transpose(-1, -2)) + bias.unsqueeze(-1)


def permute_linear_fusion(module: torch.fx.GraphModule) -> torch.fx.GraphModule:
    for node in module.graph.find_nodes(
        op="call_function", target=torch.nn.functional.linear
    ):
        if len(node.args) > 0:
            input_node = node.args[0]
        else:
            input_node = node.kwargs["input"]
        if (
            input_node.op == "call_method"
            and input_node.target == "permute"
            and check_permute(input_node)
        ):
            normalized = NormalizedLinearNode(node)
            if len(input_node.args) > 0:
                input = input_node.args[0]
            else:
                input = input_node.kwargs["input"]
            weight = normalized.get_weight()
            bias = normalized.get_bias()
            with module.graph.inserting_before(node):
                fused_node = module.graph.call_function(
                    transpose_linear, args=(input, weight, bias)
                )
                node.replace_all_uses_with(fused_node)
                module.graph.erase_node(node)
                if len(input_node.users) == 0:
                    module.graph.erase_node(input_node)

    module.graph.lint()
    module.recompile()
    return module


def permute_matmul_fusion(module: torch.fx.GraphModule) -> torch.fx.GraphModule:
    for node in itertools.chain(
        module.graph.find_nodes(op="call_function", target=torch.bmm),
        module.graph.find_nodes(op="call_function", target=torch.matmul),
    ):
        normalized = NormalizedMatmulNode(node)
        input_A_node = normalized.get_input()
        input_B_node = normalized.get_other()
        input_A = input_A_node
        input_B = input_B_node
        Atrans = Btrans = False
        if (
            input_A_node.op == "call_method"
            and input_A_node.target == "permute"
            and check_permute(input_A_node)
        ):
            Atrans = True
            if len(input_A_node.args) > 0:
                input_A = input_A_node.args[0]  # type: ignore[assignment]
            else:
                input_A = input_A_node.kwargs["input"]  # type: ignore[assignment]

        if (
            input_B_node.op == "call_method"
            and input_B_node.target == "permute"
            and check_permute(input_B_node)
        ):
            Btrans = True
            if len(input_B_node.args) > 0:
                input_B = input_B_node.args[0]  # type: ignore[assignment]
            else:
                input_B = input_B_node.kwargs["input"]  # type: ignore[assignment]

        if Atrans or Btrans:
            with module.graph.inserting_before(node):
                fused_node = module.graph.call_function(
                    transpose_matmul,
                    args=(input_A, input_B, Atrans, Btrans),
                )
            node.replace_all_uses_with(fused_node)
            module.graph.erase_node(node)
            if Atrans and len(input_A_node.users) == 0:
                module.graph.erase_node(input_A_node)
            if Btrans and len(input_B_node.users) == 0:
                module.graph.erase_node(input_B_node)

    module.graph.lint()
    module.recompile()
    return module


# X1 = X.permute(0, 2, 1)
# Y1 = X1 * W1^T + bias1
# ---->
# Y2 = X1.transpose(-1, -2) * W1^T + bias1
def transpose_linear(
    input: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor]
) -> torch.Tensor:
    if bias is None:
        return torch.matmul(input.transpose(-1, -2), weight.t())
    return torch.matmul(input.transpose(-1, -2), weight.t()) + bias


def transpose_matmul(
    A: torch.Tensor, B: torch.Tensor, Atrans: bool, Btrans: bool
) -> torch.Tensor:
    if Atrans:
        A = A.transpose(-1, -2)
    if Btrans:
        B = B.transpose(-1, -2)
    return torch.matmul(A, B)
