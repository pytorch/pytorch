import os
import sys
from typing import Optional

from torch.utils._config_module import Config, install_config_module


# [@compile_ignored: debug] Fails hard instead of graph breaking on guard on data dependent errors.
no_data_dependent_graph_break = (
    os.environ.get("TORCHDYNAMO_NO_DATA_DEPENDENT_GRAPH_BREAK", "0") == "1"
)
# [@compile_ignored: debug] Uses z3 for validating the guard optimizations transformations.
translation_validation = (
    os.environ.get("TORCHDYNAMO_TRANSLATION_VALIDATION", "0") == "1"
)
# Timeout (in milliseconds) for z3 finding a solution.
# [@compile_ignored: debug]
translation_validation_timeout = int(
    os.environ.get("TORCHDYNAMO_TRANSLATION_VALIDATION_TIMEOUT", "600000")
)
# Disables bisection for translation validation.
#
# Translation validation bisection is enabled by default, if translation validation
# is also enabled. This should help finding guard simplification issues. However,
# since validation uses Z3 for bisecting, it might take a lot of time.
#
# Set this configuration option so as to avoid bisecting.
# [@compile_ignored: debug]
translation_validation_no_bisect = (
    os.environ.get("TORCHDYNAMO_TRANSLATION_NO_BISECT", "0") == "1"
)
# Checks whether replaying ShapeEnv events on a freshly constructed one yields
# the a ShapeEnv with the same state. This should be used only in testing.
check_shape_env_recorded_events = False

# TODO: Perhaps consider allowing unions for the configs below (so you can hit
# multiple reps at the same time)

# Give extended debug information if the string representation of a guard
# matches this.  For example, set this to "Ne(s0, 10)" and whenever we issue
# this guard, we will generate full Python and C++ backtrace
# [@compile_ignored: debug]
extended_debug_guard_added = os.environ.get(
    "TORCHDYNAMO_EXTENDED_DEBUG_GUARD_ADDED", None
)

# Give extended debug information when a particular symbol is allocated.  For
# example, set this to "u2" and whenever we create this symbol, we will
# generate full Python and C++ backtrace
# [@compile_ignored: debug]
extended_debug_create_symbol = os.environ.get(
    "TORCHDYNAMO_EXTENDED_DEBUG_CREATE_SYMBOL", None
)

# Give extended debug information (C++ backtrace) for all extended debug
# settings as well as errors.  The C++ backtrace is slow and very spammy so we
# don't include it by default even when you're requesting extended debug.
# [@compile_ignored: debug]
extended_debug_cpp = os.environ.get("TORCHDYNAMO_EXTENDED_DEBUG_CPP", "") != ""

# Give extended debug information (line of code) when a torch function
# is called during export.  This is useful for showing progress and detecting
# where export might be stuck. Currently only works for strict=False.
# [@compile_ignored: debug]
extended_debug_current_loc = (
    os.environ.get("TORCHEXPORT_EXTENDED_DEBUG_CURRENT_LOC", "0") == "1"
)

# [@compile_ignored: debug] Show a warning for every specialization
print_specializations = False

# wraps (un)equalities with 'Not' class after recording the correct expression
# in the FX graph. This should incorrectly construct the divisible and replacement
# lists, and incorrectly issue guards.
inject_EVALUATE_EXPR_flip_equality_TESTING_ONLY = False

# [@compile_ignored: debug] Validate that ShapeEnv's version key is updated correctly
validate_shape_env_version_key = False

# If we produce more than this many guards on a symbol, force the symbol to
# get specialized and bail out if this many guards mention this particular
# symbol.  This may be slightly more aggressive than the true number of guards
# issued (as we test if we've hit the limit on-the-fly, whereas we may
# do further simplifications at final guard issuance time that make guards
# irrelevant.)
symbol_guard_limit_before_specialize: Optional[int] = None

# This flag changes whether we should use the same symbolic variable to represent input sizes that are the same.
use_duck_shape = True

# Controls the registration of torch.nonzero() on the meta device.
# When True, nonzero returns a tensor with shape (self.numel(), self.dim())
# assuming all elements are none-zero.
# Default is False to prevent unintended registration. Set to True to enable.
meta_nonzero_assume_all_nonzero = False

# Applies size-oblivious reasoning to backed symbols. This allocates a [0, inf] range for backed size symbols,
# and relies on size-oblivious semantics to avoid 0/1 specialization guards by marking them size-like.
# Currently an experimental option for export.
backed_size_oblivious = False

# Skip dtype check in meta registrations. Only used for systems that does its own dtype checking.
skip_dtype_check_in_meta_registrations = False

# Experimental: If True, graph module will register fx metadata during recompile()
enrich_profiler_metadata: bool = Config(  # type: ignore[var-annotated]
    default=False,
    env_name_default="TORCH_ENRICH_RPOFILER_STACK_TRACE",
)


install_config_module(sys.modules[__name__])
