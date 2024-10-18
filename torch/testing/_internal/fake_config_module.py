import sys

from torch.utils._config_module import install_config_module
from torch._utils_internal import JustKnobsConfig

example_knob = JustKnobsConfig(name="true_knob")
example_knob_false = JustKnobsConfig(name="false_knob", default=False)
example_knob_force= JustKnobsConfig(name="false_knob", default=False, override=True)
_cache_config_ignore_prefix = []

install_config_module(sys.modules[__name__])
