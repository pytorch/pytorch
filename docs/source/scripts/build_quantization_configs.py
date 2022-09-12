"""
This script will generate default values of quantization configs.
These are for use in the documentation.
"""

import torch
from torch.ao.quantization.backend_config import get_native_backend_config_dict
from torch.ao.quantization.backend_config.utils import (
    entry_to_pretty_str,
    remove_boolean_dispatch_from_name,
)
import os.path


# Create a directory for the images, if it doesn't exist
QUANTIZATION_BACKEND_CONFIG_IMAGE_PATH = os.path.join(
    os.path.realpath(os.path.join(__file__, "..")),
    "quantization_backend_configs"
)

if not os.path.exists(QUANTIZATION_BACKEND_CONFIG_IMAGE_PATH):
    os.mkdir(QUANTIZATION_BACKEND_CONFIG_IMAGE_PATH)

output_path = os.path.join(QUANTIZATION_BACKEND_CONFIG_IMAGE_PATH, "default_backend_config.txt")

with open(output_path, "w") as f:
    native_backend_config_dict = get_native_backend_config_dict()

    configs = native_backend_config_dict['configs']

    def _sort_key_func(entry):
        pattern = entry['pattern']
        while isinstance(pattern, tuple):
            pattern = pattern[-1]

        pattern = remove_boolean_dispatch_from_name(pattern)
        if not isinstance(pattern, str):
            # methods are already strings
            pattern = torch.typename(pattern)

        # we want
        #
        #   torch.nn.modules.pooling.AdaptiveAvgPool1d
        #
        # and
        #
        #   torch._VariableFunctionsClass.adaptive_avg_pool1d
        #
        # to be next to each other, so convert to all lower case
        # and remove the underscores, and compare the last part
        # of the string
        pattern_str_normalized = pattern.lower().replace('_', '')
        key = pattern_str_normalized.split('.')[-1]
        return key

    configs.sort(key=_sort_key_func)

    entries = []
    for entry in configs:
        entries.append(entry_to_pretty_str(entry))
    entries = ",\n".join(entries)
    f.write(entries)
