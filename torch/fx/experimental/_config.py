import os
import sys

from typing import Optional

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

from torch.utils._config_module import install_config_module

install_config_module(sys.modules[__name__])
