import inspect
from itertools import count
import logging

import torch
from torch._ops import HigherOrderOperator
from torch.utils.checkpoint import checkpoint

log = logging.getLogger(__name__)

uid = count(1)

# Used for testing the HigherOrderOperator mechanism
class Wrap(HigherOrderOperator):
    def __init__(self):
        super().__init__("wrap")

    def __call__(self, func, *args, **kwargs):
        # Dynamo already traces the body of HigherOrderOp beforehand when it
        # so no need to trace into it.
        import torch._dynamo  # noqa: F401
        from torch._dynamo import disable

        @disable
        def wrapper():
            result = func(*args, **kwargs)
            return result

        return wrapper()

wrap = Wrap()

random_ops = set([
    # torch functions
    torch.dropout,
    torch._fused_dropout,
    torch.bernoulli,
    torch.binomial,
    torch.multinomial,
    torch.normal,
    torch.rand,
    torch.rand_like,
    torch.randn,
    torch.randint,
    torch.randperm,
    torch.native_dropout,
    torch._standard_gamma,
    torch.poisson,
    torch.rrelu,
    torch.rrelu_,

    # torch.nn.functional functions
    torch.nn.functional.dropout,
    torch.nn.functional.rrelu,
    torch.nn.functional.rrelu_,

    # Tensor methods
    "bernoulli",
    "bernoulli_",
    "cauchy_",
    "exponential_",
    "geometric_",
    "log_normal_",
    "normal_",
    "uniform_",
])

inplace_ops = set([
    torch._add_relu_,
    torch._amp_foreach_non_finite_check_and_unscale_,
    torch._amp_update_scale_,
    torch._fill_mem_eff_dropout_mask_,
    torch._foreach_abs_,
    torch._foreach_acos_,
    torch._foreach_add_,
    torch._foreach_addcdiv_,
    torch._foreach_addcmul_,
    torch._foreach_asin_,
    torch._foreach_atan_,
    torch._foreach_ceil_,
    torch._foreach_clamp_max_,
    torch._foreach_clamp_min_,
    torch._foreach_cos_,
    torch._foreach_cosh_,
    torch._foreach_div_,
    torch._foreach_erf_,
    torch._foreach_erfc_,
    torch._foreach_exp_,
    torch._foreach_expm1_,
    torch._foreach_floor_,
    torch._foreach_frac_,
    torch._foreach_lerp_,
    torch._foreach_lgamma_,
    torch._foreach_log10_,
    torch._foreach_log1p_,
    torch._foreach_log2_,
    torch._foreach_log_,
    torch._foreach_maximum_,
    torch._foreach_minimum_,
    torch._foreach_mul_,
    torch._foreach_neg_,
    torch._foreach_pow_,
    torch._foreach_reciprocal_,
    torch._foreach_round_,
    torch._foreach_sigmoid_,
    torch._foreach_sin_,
    torch._foreach_sinh_,
    torch._foreach_sqrt_,
    torch._foreach_sub_,
    torch._foreach_tan_,
    torch._foreach_tanh_,
    torch._foreach_trunc_,
    torch._foreach_zero_,
    torch._fused_adam_,
    torch._fused_adamw_,
    torch._index_put_impl_,
    torch._mkldnn_transpose_,
    torch._resize_output_,
    torch._sobol_engine_ff_,
    torch._sobol_engine_initialize_state_,
    torch._sobol_engine_scramble_,
    torch.abs_,
    torch.acos_,
    torch.acosh_,
    torch.addmv_,
    torch.alpha_dropout_,
    torch.arccos_,
    torch.arccosh_,
    torch.arcsin_,
    torch.arcsinh_,
    torch.arctan_,
    torch.arctanh_,
    torch.as_strided_,
    torch.asin_,
    torch.asinh_,
    torch.atan_,
    torch.atanh_,
    torch.ceil_,
    torch.celu_,
    torch.clamp_,
    torch.clamp_max_,
    torch.clamp_min_,
    torch.clip_,
    torch.conj_physical_,
    torch.cos_,
    torch.cosh_,
    torch.deg2rad_,
    torch.detach_,
    torch.dropout_,
    torch.embedding_renorm_,
    torch.erf_,
    torch.erfc_,
    torch.exp2_,
    torch.exp_,
    torch.expm1_,
    torch.feature_alpha_dropout_,
    torch.feature_dropout_,
    torch.fill_,
    torch.fix_,
    torch.floor_,
    torch.frac_,
    torch.gcd_,
    torch.i0_,
    torch.index_put_,
    torch.lcm_,
    torch.ldexp_,
    torch.log10_,
    torch.log1p_,
    torch.log2_,
    torch.log_,
    torch.logit_,
    torch.nan_to_num_,
    torch.neg_,
    torch.negative_,
    torch.rad2deg_,
    torch.reciprocal_,
    torch.relu_,
    torch.resize_as_,
    torch.resize_as_sparse_,
    torch.round_,
    torch.rrelu_,
    torch.rsqrt_,
    torch.selu_,
    torch.sigmoid_,
    torch.sin_,
    torch.sinc_,
    torch.sinh_,
    torch.sqrt_,
    torch.square_,
    torch.tan_,
    torch.tanh_,
    torch.threshold_,
    torch.trunc_,
    torch.xlogy_,
    torch.zero_,
    torch.nn.functional._no_grad_embedding_renorm_,
    torch.nn.functional.celu_,
    torch.nn.functional.elu_,
    torch.nn.functional.hardtanh_,
    torch.nn.functional.leaky_relu_,
    torch.nn.functional.relu_,
    torch.nn.functional.rrelu_,
    torch.nn.functional.selu_,
    torch.nn.functional.threshold_,
    "_coalesced_",
    "abs_",
    "absolute_",
    "acos_",
    "acosh_",
    "add_",
    "addbmm_",
    "addcdiv_",
    "addcmul_",
    "addmm_",
    "addmv_",
    "addr_",
    "apply_",
    "arccos_",
    "arccosh_",
    "arcsin_",
    "arcsinh_",
    "arctan2_",
    "arctan_",
    "arctanh_",
    "as_strided_",
    "asin_",
    "asinh_",
    "atan2_",
    "atan_",
    "atanh_",
    "baddbmm_",
    "bernoulli_",
    "bitwise_and_",
    "bitwise_left_shift_",
    "bitwise_not_",
    "bitwise_or_",
    "bitwise_right_shift_",
    "bitwise_xor_",
    "cauchy_",
    "ceil_",
    "clamp_",
    "clamp_max_",
    "clamp_min_",
    "clip_",
    "conj_physical_",
    "copy_",
    "copysign_",
    "cos_",
    "cosh_",
    "cumprod_",
    "cumsum_",
    "deg2rad_",
    "detach_",
    "digamma_",
    "div_",
    "divide_",
    "eq_",
    "erf_",
    "erfc_",
    "erfinv_",
    "exp2_",
    "exp_",
    "expm1_",
    "exponential_",
    "fill_",
    "fill_diagonal_",
    "fix_",
    "float_power_",
    "floor_",
    "floor_divide_",
    "fmod_",
    "frac_",
    "gcd_",
    "ge_",
    "geometric_",
    "greater_",
    "greater_equal_",
    "gt_",
    "heaviside_",
    "hypot_",
    "i0_",
    "igamma_",
    "igammac_",
    "index_add_",
    "index_copy_",
    "index_fill_",
    "index_put_",
    "index_reduce_",
    "lcm_",
    "ldexp_",
    "le_",
    "lerp_",
    "less_",
    "less_equal_",
    "lgamma_",
    "log10_",
    "log1p_",
    "log2_",
    "log_",
    "log_normal_",
    "logical_and_",
    "logical_not_",
    "logical_or_",
    "logical_xor_",
    "logit_",
    "lt_",
    "map2_",
    "map_",
    "masked_fill_",
    "masked_scatter_",
    "mul_",
    "multiply_",
    "mvlgamma_",
    "nan_to_num_",
    "ne_",
    "neg_",
    "negative_",
    "nextafter_",
    "normal_",
    "not_equal_",
    "polygamma_",
    "pow_",
    "put_",
    "rad2deg_",
    "random_",
    "reciprocal_",
    "relu_",
    "remainder_",
    "rename_",
    "renorm_",
    "requires_grad_",
    "resize_",
    "resize_as_",
    "resize_as_sparse_",
    "round_",
    "rsqrt_",
    "scatter_",
    "scatter_add_",
    "scatter_reduce_",
    "set_",
    "sgn_",
    "share_memory_",
    "sigmoid_",
    "sign_",
    "sin_",
    "sinc_",
    "sinh_",
    "sparse_resize_",
    "sparse_resize_and_clear_",
    "sqrt_",
    "square_",
    "squeeze_",
    "sub_",
    "subtract_",
    "swapaxes_",
    "swapdims_",
    "t_",
    "tan_",
    "tanh_",
    "transpose_",
    "tril_",
    "triu_",
    "true_divide_",
    "trunc_",
    "uniform_",
    "unsqueeze_",
    "xlogy_",
    "zero_",
])

class HandleActivationCheckpoint(HigherOrderOperator):
    """
    This operator is supposed to be used only with torch.compile stack.
    There are two modes in this operator:

    Mode 1: WrapActivationCheckpoint
    This mode is used for selective checkpointing + torch.compile.
    Under this mode, we wrap torch.utils.checkpoint. This avoids
    TorchDynamo to look into saved tensor hooks and directly passes the control
    to AOT Autograd, which is ok with tracing saved tensor hooks. As a result of
    AOT tracing torch.utils.checkpoint code, we have a backward graph with
    recomputed forward nodes.

    However, we might deprecate this mode soon. The difficulty arises in the
    functionalization of rng ops. Today, there are two different
    functionalization of rng ops - one at AOT autograd and other at Inductor.
    And they are difficult to map to each other. The rng states also complicate
    pattern matching in Inductor. Due to the ease of implementation, we are
    currently inclined towards functionalization at Inductor level, which means
    that duplication/recomputation is done as a compiler pass in the
    partitioners. See TagActivationCheckpoint for more information.

    Mode 2: TagActivationCheckpoint
    This mode is used for full activation checkpointing + torch.compile.
    Under this mode, the operator accepts a Fx graph module which needs to be checkpointed.
    This operator adds "recomputable" tag to the nodes of the Fx graph that
    should be recomputed.

    The goal is to avoid both Dynamo and AOT Autograd to trace through saved
    tensor hooks, and rather rely on the partitioners to actually duplicate the
    nodes. This sits well in the torch.compile stack, because by the time graph
    reaches partitioner, inductor has already run its functionalization of rng
    ops. Therefore, the duplication of nodes, by design, respects the rng states
    in the forward and recomputed forward in backward.
    """

    def __init__(self):
        super().__init__("handle_activation_checkpoint")
        self.context_fn = None

    @staticmethod
    def divide_kwargs(kwargs):
        """
        checkpoint fn can have mixed kwargs between checkpointed fn and
        checkpoint fn itself. For example
        >> def gn(x, y, z=None):
        >>     a = torch.matmul(x, y)
        >>     if z is not None:
        >>         return torch.matmul(a, z)
        >>     return a
        >> def fn(x, y, z):
        >>     return torch.cos(checkpoint(gn, x, y, use_reentrant=False, z=z))
        In the above case, z belongs to checkpointed function gn, but
        use_reentrant belongs to the checkpoint function. This function splits
        the kwargs into checkpoint_kwargs and gmod_kwargs (or
        checkpointed_fn_kwargs).
        We do sorting to ensure same graph from run to run for better
        debuggability. It is not required for correctness.
        """
        ckpt_signature = inspect.signature(checkpoint)
        checkpoint_keys = set()
        for name in ckpt_signature.parameters:
            if name in ("function", "args", "kwargs"):
                continue
            checkpoint_keys.add(name)

        # `preserve_rng_state` is not a regular kwarg
        checkpoint_keys.add("preserve_rng_state")

        checkpoint_kwargs = {name: kwargs[name] for name in kwargs.keys() if name in checkpoint_keys}
        gmod_kwargs = {name: kwargs[name] for name in kwargs.keys() if name not in checkpoint_keys}
        return checkpoint_kwargs, gmod_kwargs

    def tag_nodes(self, gmod):
        unique_graph_id = next(uid)
        for node in gmod.graph.nodes:
            if node.op in ("call_function", "call_method", "call_module"):
                node.meta["recompute"] = unique_graph_id
        return gmod

    def __call__(self, gmod, *args, **kwargs):
        import torch.fx.traceback as fx_traceback
        from torch.fx import Interpreter
        # TODO - This is a temporary sitaution where we have two versions of
        # checkpointing implemetation. We will converge on one and remove the other.
        if self.context_fn is not None or torch._functorch.config.functionalize_rng_ops:
            # Mode 1: WrapActivationCheckpoint
            assert kwargs.get("use_reentrant", False) is False, \
                "use_reentrant=True is not supported under torch.compile " + \
                "when selective checkpointing is used " + \
                "or when `torch._functorch.config.functionalize_rng_ops` is True."
            assert kwargs.get("preserve_rng_state", False) is False, \
                "preserve_rng_state=True is not supported under torch.compile " + \
                "when selective checkpointing is used " + \
                "or when `torch._functorch.config.functionalize_rng_ops` is True. " + \
                "If your program relies on this to have deterministic output in the checkpointed region, " + \
                "it means the checkpointed region has ops that use random number generator (e.g. torch.dropout). " + \
                "Please remove those random ops as they are not supported under selective checkpointing + torch.compile."

            random_ops_in_gmod = [node.target for node in gmod.graph.nodes if node.target in random_ops]
            assert len(random_ops_in_gmod) == 0, \
                "Random ops are not supported in selective checkpointing region under torch.compile. " + \
                f"Found random ops: {random_ops_in_gmod}."

            inplace_ops_in_gmod = [node.target for node in gmod.graph.nodes if node.target in inplace_ops]
            assert len(inplace_ops_in_gmod) == 0, \
                "In-place ops are not supported in selective checkpointing region under torch.compile. " + \
                f"Found in-place ops: {inplace_ops_in_gmod}."

            log.warn("""
Detected selective checkpointing is used under torch.compile.
Please make sure the checkpointed region does not contain:

1. random ops (e.g. torch.dropout)
2. in-place ops (e.g. torch.relu_)
""")

            kwargs["use_reentrant"] = False
            kwargs["preserve_rng_state"] = False
            kwargs["context_fn"] = self.context_fn
            # Using interpreter allows preservation of metadata through torch.compile stack.
            with fx_traceback.preserve_node_meta():
                return checkpoint(Interpreter(gmod).run, *args, **kwargs)
        else:
            # Mode 2: TagActivationCheckpoint
            gmod = self.tag_nodes(gmod)
            # Using interpreter allows preservation of metadata through torch.compile stack.
            with fx_traceback.preserve_node_meta():
                return Interpreter(gmod).run(*args)

handle_activation_checkpoint = HandleActivationCheckpoint()
