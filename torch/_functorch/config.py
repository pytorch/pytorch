# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Global flags for aot autograd
"""
import os
import sys
from typing import TYPE_CHECKING


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

# Today, if you are in a situation where there is "false aliasing"
# (e.g. you have a bunch of model parameters that all alias the same underlying buffer),
# our checks for this situation are very slow if these inputs have dynamic shapes.
# This config is set to ensure that there aren't too many aliased inputs in this situation,
# so that we error loudly instead of compiling forever.
# Eventually, we should make these checks faster.
# For now, however, you can simply turn off dynamic shapes by marking your inputs static
# when you run into this situation.
_max_aliased_inputs_with_dynamic_shapes_enabled = 5

static_weight_shapes = True

# Applies CSE to the graph before partitioning
cse = True


enable_autograd_cache = os.environ.get("ENABLE_AOT_AUTOGRAD_CACHE", "0") == "1"

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
from torch._inductor.config import is_fbcode


# View replay is currently not compatible with AOTAutogradCache, since
# FunctionalTensors are not serializable. We'll need to make them
# serializable before enabling warm cache with this config turned on.
view_replay_for_aliased_outputs = (not is_fbcode()) and (not enable_autograd_cache)

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

# This dumps out a png visualization of the expected runtime vs. activation
# memory tradeoffs for all memory budget values from 0 to 1 in increments of
# 0.5. See an example here:
# https://github.com/pytorch/pytorch/pull/126320#discussion_r1625104015
visualize_memory_budget_pareto = (
    os.environ.get("PARTITIONER_MEMORY_BUDGET_PARETO", "0") == "1"
)

# Sets all of the ban_recompute heuristics to False except ban_recompute_reductions
# Generally, this will probably result in some memory improvement, but at the
# cost of some performance
aggressive_recomputation = False

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

# This controls whether we collect donated buffer. This flag must be set
# False if a user wants to retain_graph=True for backward.
donated_buffer = False

# Controls the default graph output format used by draw_graph
# Supported formats are defined here https://graphviz.org/docs/outputs/
torch_compile_graph_format = os.environ.get("TORCH_COMPILE_GRAPH_FORMAT", "svg")


# Error on BypassAOTAutogradCache instead of just a warning
# Used for tests
strict_autograd_cache = False

if TYPE_CHECKING:
    from torch.utils._config_typing import *  # noqa: F401, F403

from torch.utils._config_module import install_config_module


# adds patch, save_config, invalid config checks, etc
install_config_module(sys.modules[__name__])
