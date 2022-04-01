"""
This script will generate default values of quantization configs.
These are for use in the documentation.
"""

from torch.ao.quantization.fx.backend_config import get_native_backend_config_dict
import os.path
from pprint import pprint


# Create a directory for the images, if it doesn't exist
QUANTIZATION_CONFIG_IMAGE_PATH = os.path.join(
    os.path.realpath(os.path.join(__file__, "..")),
    "quantization_configs"
)

if not os.path.exists(QUANTIZATION_CONFIG_IMAGE_PATH):
    os.mkdir(QUANTIZATION_CONFIG_IMAGE_PATH)

output_path = os.path.join(QUANTIZATION_CONFIG_IMAGE_PATH, "default_config.txt")

with open(output_path, "w") as f:
    pprint(get_native_backend_config_dict(), stream=f)
