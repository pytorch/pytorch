# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable


"""
Global flags for aot autograd
"""

import os
import sys
from typing import Literal, Optional, TYPE_CHECKING

from torch.utils._config_module import Config, install_config_module


# [@compile_ignored: debug]
_save_config_ignore = [
    # callable not serializable
    "joint_custom_pass",
]


# Converts torch rng ops to their functional philox rng equivalents. Note that
# we functionalize only CUDA rng ops today.
functionalize_rng_ops = False

# can be useful for debugging if we are incorrectly creating meta fake tensors
fake_tensor_allow_meta = os.environ.get("FAKE_ALLOW_META", "1") != "0"

# Enables optional asserts in hotpath code to check for errors.  If
# you are seeing weird accuracy problems, try turning this on.
# This is currently off by default as it will harm tracing time,
# but it is on by default for aot_eager.
debug_assert = False

debug_partitioner = os.environ.get("AOT_PARTITIONER_DEBUG", "0") != "0"

# See # NOTE [Export custom triton op]
decompose_custom_triton_ops = True

static_weight_shapes = True

# See https://github.com/pytorch/pytorch/issues/141881
# Tells partitioner that parameters are free to save for backward.
treat_parameters_as_free_to_save = True

# Applies CSE to the graph before partitioning
cse = True

from torch._environment import is_fbcode


enable_autograd_cache: bool = Config(
    justknob="pytorch/remote_cache:enable_local_autograd_cache",
    env_name_force="TORCHINDUCTOR_AUTOGRAD_CACHE",
    default=True,
)

autograd_cache_allow_custom_autograd_functions: bool = Config(
    env_name_force="TORCHINDUCTOR_AUTOGRAD_CACHE_ALLOW_CUSTOM_AUTOGRAD", default=False
)

# For now, this is just for enabling unit testing in test_aot_autograd_cache.py
# We will either make this the default with AOTAutogradCache, or
# we'll just use it in the precompile flow. So there's no
# need to add env vars or make it configurable
bundled_autograd_cache: bool = False

# Whether or not to normalize placeholder names in graphs
# from dynaom in AOTAutogradCache
autograd_cache_normalize_inputs = not is_fbcode()


def remote_autograd_cache_default() -> Optional[bool]:
    if os.environ.get("TORCHINDUCTOR_AUTOGRAD_REMOTE_CACHE") == "1":
        return True
    if os.environ.get("TORCHINDUCTOR_AUTOGRAD_REMOTE_CACHE") == "0":
        return False
    return None


enable_remote_autograd_cache = remote_autograd_cache_default()


# When AOTAutograd regenerates aliased graph outputs,
# attempt to use functionalization's view-replay logic
# before falling back to the autograd engine's view replay or as_strided.
# This can have some perf implications
# (although for many models this will not matter).
# (1) If you have many view ops chained together, replaying all of them
#     at runtime can have more overhead compared to a single as_strided call
# (2) If you are doing training, AsStridedBackward is quite slow,
#     and the individual view op backward formulas will likely be faster.
# (3) Some backends like XLA do not support as_strided

# Temporary hack: disable this flag for internal
# (needed to fix an internal issue while avoiding bumping XLA pin)
# eventually: either default this config to false completely
# once XLA pin update works,
# or default config to true and fix relevant bugs


# View replay is currently not compatible with AOTAutogradCache, since
# FunctionalTensors are not serializable. We'll need to make them
# serializable before enabling warm cache with this config turned on.
view_replay_for_aliased_outputs = not is_fbcode()

# Restricts the amount of computation AOTAutograd can do.
# NB: We have essentially disabled this heuristic now. However, this is kept
# here for now in case it's useful. Setting it low can artificially reduce the
# amount of recomputation AOTAutograd performs, although not in any kind of
# principled way.
max_dist_from_bw = 1000


# Bans recomputation of nodes that are reading from nodes that is far before
# the current node
ban_recompute_used_far_apart = True
# Breaks up long chain of fusible ops, as otherwise we can have an arbitrarily
# long chain of recomputation in the backwards pass.
ban_recompute_long_fusible_chains = True
# Bans recomputation of nodes that must be materialized in the backwards pass
# (used by a non-fusible node)
ban_recompute_materialized_backward = True
# Chooses to ban recomputation of nodes based off an allowlist. Setting it to
# False changes it to use a denylist. Main change is on operators like
# sort/pool/stuff that isn't cheap enough to be fusible for free but also isn't
# that expensive
ban_recompute_not_in_allowlist = True
# Chooses to ban recomputation of reductions. This is generally a good idea, as
# the result of reductions is generally very small but recomputing reductions in
# a fusion can be expensive.
ban_recompute_reductions = True
# Prevents the partitioner from ever saving views (i.e. always recompute them).
# Generally a good idea since views are free to recompute.
recompute_views = False

# By default, the partitioner is purely trying to optimize for runtime (although
# it should always use less memory than eager)
# This knob controls the partitioner to make that tradeoff for you, choosing the
# fastest option that saves less activations than the memory budget.
# Specifically, 0.0 corresponds to the activation memory from applying
# activation checkpointing to the full compiled region, and 1.0 corresponds to
# the activation memory from the default runtime-optimized strategy.  So, 0.4
# would result in a strategy that saves 40% of the activations compared to the
# default strategy.
# It solves a 0-1 knapsack to find the minimum recompute necessary to stay below
# the activation memory budget.
# NOTE: This *cannot* be treated as
activation_memory_budget = 1.0

# This controls how we estimate the runtime when deciding what the cheapest
# operators to recompute are. The 3 options are
# "flops": Bases it off of the flop count provided by torch.utils.flop_counter
# "profile": Benchmarks each operator to come up with a runtime
# "testing": Returns 1 for everything
activation_memory_budget_runtime_estimator = "flops"

# This controls the solver used for the 0-1 knapsack. By default we use a
# quantized DP solution ("dp"). The other approaches are a "greedy" and a "ilp"
# (which has a scipy dependency).
activation_memory_budget_solver = "dp"

# This dumps out a SVG visualization of the expected runtime vs. activation
# memory tradeoffs for all memory budget values from 0 to 1 in increments of
# 0.5. See an example here:
# https://github.com/pytorch/pytorch/pull/126320#discussion_r1625104015
visualize_memory_budget_pareto = (
    os.environ.get("PARTITIONER_MEMORY_BUDGET_PARETO", "0") == "1"
)

# This controls the directory in which to dump the SVG plot with the pareto
# frontier of the activation checkpointing memory-vs-runtime tradeoffs.
memory_budget_pareto_dir = os.environ.get("PARTITIONER_MEMORY_BUDGET_PARETO_DIR")

# Sets all of the ban_recompute heuristics to False except ban_recompute_reductions
# Generally, this will probably result in some memory improvement, but at the
# cost of some performance
aggressive_recomputation = False

# activation offloading enablement (testing purpose)
enable_activation_offloading = False

# activation offloading with separate CUDA stream
activation_offload_separate_stream = False

# activation offloading wait sinking when using separate stream (fwd graph)
activation_offload_sink_wait = False

# activation reloading with prefetching when using separate streams (bwd graph)
activation_reload_prefetch = False

# If FakeTensor.data_ptr() should error.
# This option is independent of AOTAutograd and torch.compile, but our policy
# is to turn it off during torch.compile.
fake_tensor_allow_unsafe_data_ptr_access = True

# Unlifts effect tokens from the inputs/outputs in the traced graph and instead
# inserts make_token/sink_token calls in the graph to create tokens and then
# sink them at the end. Note that this means the graph is no longer functional
# which may lead to silent errors unless the backend knows how to handle the
# tokens.
unlift_effect_tokens = False

# NOTE: [The default layout constraint for custom operators.]
# This must be the name of one of the layout constraint tags
# (that is, one of {"needs_fixed_stride_order", "flexible_layout"}),
# If the custom op does not have a layout constraint tag already
# then we assume the following applies.
#
# This config is respected by Inductor and we recommend other backends also
# respect it.
# This config is in torch._functorch and not torch._inductor because it affects
# ProxyTensor tracing.
custom_op_default_layout_constraint: Literal[
    "needs_exact_strides", "needs_fixed_stride_order", "flexible_layout"
] = "needs_exact_strides"


# Run aot eager decomp partition with CrossRefFakeMode
# options = False, "all", "custom_ops"
fake_tensor_crossref = False

# This mode specifies that we should also keep track of the real
# tensor along with the fake tensor, and do real compute.  While
# seemingly this eliminates the whole point of fake tensors, there are
# two obvious use cases for it:
#
#   1. When users call item()/other data dependent operations,
#      if we propagate_real_tensors we are able to determine what
#      the true value is and keep going.
#
#   2. It can be useful for testing, when you want to see if the fake
#      and real tensors agree with each other.  (Note that there are
#      currently known inaccuracies in how we clone real tensors, that
#      would have to be tightened up for this to be useful in this
#      case.)
#
# Note that fake tensors are typically understood to be cheap to store
# indefinitely, so we tend to hold on to them longer than we would
# hold onto the real tensors.  So we also support you explicitly
# deallocating the real tensor associated with a fake tensor, at which
# point we will stop propagating real tensors.
#
# One more thing: when you provide a real tensor to fakeify, we will
# clone it, so that we can safely perform mutations on it if necessary.
# This will increase live memory usage.  This could potentially be
# optimized by using COW.  We also currently do not faithfully
# maintain autograd metadata on the real tensor; this is fine because
# AOTAutograd will only use the fake tensor to determine leafness/etc
# of tensors in question.
fake_tensor_propagate_real_tensors = False

# AOTDispatcher traces out a backward graph at the time of the forward pass.
# This flags controls whether or not that backward graph gets autocast behavior
# applied to it.
#
# The options are either:
# - "same_as_forward". We assume that the backward of the torch.compile'ed region
#   will be run under the same autocast context manager that the region was run
#   under. This is equivalent to running the following code in eager:
#
#   with torch.amp.autocast(...):
#       y = region(x)
#       ...
#       z.backward()
#
# - "off". We assume that the backward of the torch.compile'd region will
#   not be run under any autocast context managers.
#   This is equivalent to running the following code in eager:
#
#   with torch.amp.autocast(...):
#       y = region(x)
#       ...
#   z.backward()
#
# - or a list of kwargs dicts that represent an autocast context manager to turn
#   on during the backward pass.
#
#   e.g. [{"device_type": "cuda"}] is equivalent to running the following code in eager:
#
#   y = region(x)
#   ...
#   with torch.amp.autocast(device="cuda"):
#       z.backward()
backward_pass_autocast = "same_as_forward"

# This controls whether we collect donated buffer. This flag must be set
# False if a user wants to retain_graph=True for backward.
donated_buffer = not is_fbcode()

# Controls the default graph output format used by draw_graph
# Supported formats are defined here https://graphviz.org/docs/outputs/
torch_compile_graph_format = os.environ.get("TORCH_COMPILE_GRAPH_FORMAT", "svg")

# Valid only if fake_tensor_propagate_real_tensors = True; if a fake-real
# kernel mismatch is detected, bypasses by making a fake kernel from the
# real tensor outputs.
generate_fake_kernels_from_real_mismatches = False

# When there are device mismatches in FakeTensor device propagation,
# prefer a specific device type over others. This is particularly useful
# in full compiled mode where intermediate tensors with device mismatches
# represent only logical differences during compilation - these intermediate
# tensors will never physically materialize in the binary execution, so the
# device mismatch is not a real runtime concern. Enabling this allows the
# compiler to proceed with compilation by choosing the preferred device type
# for consistency. For example, set to "mtia" to prefer MTIA devices over
# CPU, or "cuda" to prefer CUDA devices over CPU.
fake_tensor_prefer_device_type: Optional[str] = None

# CUDAGraph save run_with_rng functionalization.
# TODO: turn on by default
graphsafe_rng_functionalization = True

# Whether or not to eagerly compile the backward
# used by AOT compile and other settings
# TODO: once AOT compile calls aot autograd directly instead of
# through compile_fx, we can remove this
force_non_lazy_backward_lowering = False

# only for testing, used to turn functionalization off in AOTDispatcher
_test_disable_functionalization = True

# Error on BypassAOTAutogradCache instead of just a warning
# Used for tests
strict_autograd_cache = False

# Note [Recomputing collectives in the partitioner]
# The purpose of this config is as follows:
# - We have many passes in the compiler (min-cut partitioning, DCE, etc)
#   which can reorder or ,delete duplicate nodes in the graph
# - If any of these passes reorder/delete/duplicate a collective
#   in a setting where the compiler is being run independently on multiple
#   ranks, we run the risk that the compiler will make a different decision on
#   different ranks, resulting in a NCCL hang when using torch.compile
# To handle this, we will (by default) ensure that collectives are not modified
# by the compiler.
#
# A few examples:
# - don't dead-code-eliminate collectives
#   (in case they are dead on rank i but not rank j)
# - don't recompute collectives in partitioning
#   (in case we recompute on rank i but not rank j)
#
# Today this flag **must** be set to false, but eventually
# we want the option to set it to true.
# In order to potentially optimize collectives, we'll need the compiler
# to broadcast information across ranks at compile time to ensure
# that any decisions on collectives are made consistently.
unsafe_allow_optimization_of_collectives = False

# See Note [AOTAutograd Tangent Subclassness for mutated inputs]
# TODO(ivankobzarev): Remove this config, being able to deduce it compile time.
disable_guess_zero_tangent_for_mutated_input_subclass = False

# See Note [Tangents memory format]
# By default tangents strideness is guessed to be contiguous,
# At runtime non contiguous tangents will be coerced to be contiguous.
# This config changes this guess for tangents strides to be the same as outputs.
# TODO(ivankobzarev): Remove this config once extra memory usage is investigated.
guess_tangent_strides_as_outputs = False

# This is a temporary config to ensure all ranks take the same decision in the partitioner
# it will untimately be removed once we share size_hints across ranks through compiler collectives
_sync_decision_cross_ranks = False

# By default apply inlined saved_tensors_hooks only for "donated" buffers.
# "donated" buffers are invisible to the user, they are intermediates of the forward graph.
# Applying saved tensors hooks for memory optimizations only for intermediates
# guarantees that original saved tensors could be deallocated.
# This config enables saved_tensors_hooks are applied for **all** saved tensors,
# that could include inputs, parameters, outputs.
# "donated" - applied only to saved intermediates of the graph
# "no_static" - applied to all saved but not "static"
# (this includes parameters and user marked as static)
# "all" - no filtering, everything saved for backward.
saved_tensors_hooks_filtering_mode = "donated"


# This callback is invoked on the joint graph before partitioning
joint_custom_pass: Callable = None  # type: ignore[assignment]

# Note [Selective Decomposition]
# This config allows selective decomposition of certain operators in the graph.
# When True, it does NOT decompose any nodes, except those nodes that users explicitly
# annotated with regional inductor compile. Please read torch.fx.passes.regional_inductor
# on to explicitly annotate. This is currently only used by inductor lite mode.
selective_decompose: bool = False


if TYPE_CHECKING:
    from torch.utils._config_typing import *  # noqa: F401, F403


# adds patch, save_config, invalid config checks, etc
install_config_module(sys.modules[__name__])
