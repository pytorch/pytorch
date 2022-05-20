"""
This script will generate default values of quantization configs.
These are for use in the documentation.
"""

from torch.ao.quantization.backend_config import get_native_backend_config_dict
import os.path
from pprint import pprint


# Create a directory for the images, if it doesn't exist
QUANTIZATION_BACKEND_CONFIG_IMAGE_PATH = os.path.join(
    os.path.realpath(os.path.join(__file__, "..")),
    "quantization_backend_configs"
)

if not os.path.exists(QUANTIZATION_BACKEND_CONFIG_IMAGE_PATH):
    os.mkdir(QUANTIZATION_BACKEND_CONFIG_IMAGE_PATH)

output_path = os.path.join(QUANTIZATION_BACKEND_CONFIG_IMAGE_PATH, "default_backend_config.txt")

with open(output_path, "w") as f:
    pprint(get_native_backend_config_dict(), stream=f)
