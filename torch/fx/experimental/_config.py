import os
import sys

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


# [@compile_ignored: debug] Show a warning for every specialization
print_specializations = False

# wraps (un)equalities with 'Not' class after recording the correct expression
# in the FX graph. This should incorrectly construct the divisible and replacement
# lists, and incorrectly issue guards.
inject_EVALUATE_EXPR_flip_equality_TESTING_ONLY = False

# [@compile_ignored: debug] Validate that ShapeEnv's version key is updated correctly
validate_shape_env_verison_key = False

from torch.utils._config_module import install_config_module

install_config_module(sys.modules[__name__])
