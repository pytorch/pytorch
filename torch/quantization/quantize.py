# flake8: noqa: F401
r"""
This file is in the process of migration to `torch/ao/quantization`, and
is kept here for compatibility while the migration process is ongoing.
If you are adding a new entry/functionality, please, add it to the
`torch/ao/quantization/quantize.py`, while adding an import statement
here.
"""

from torch.ao.quantization.quantize import _convert
from torch.ao.quantization.quantize import _observer_forward_hook
from torch.ao.quantization.quantize import _propagate_qconfig_helper
from torch.ao.quantization.quantize import _remove_activation_post_process
from torch.ao.quantization.quantize import _remove_qconfig
from torch.ao.quantization.quantize import add_observer_
from torch.ao.quantization.quantize import add_quant_dequant
from torch.ao.quantization.quantize import convert
from torch.ao.quantization.quantize import get_observer_dict
from torch.ao.quantization.quantize import get_unique_devices_
from torch.ao.quantization.quantize import is_activation_post_process
from torch.ao.quantization.quantize import prepare
from torch.ao.quantization.quantize import prepare_qat
from torch.ao.quantization.quantize import propagate_qconfig_
from torch.ao.quantization.quantize import quantize
from torch.ao.quantization.quantize import quantize_dynamic
from torch.ao.quantization.quantize import quantize_qat
from torch.ao.quantization.quantize import register_activation_post_process_hook
from torch.ao.quantization.quantize import swap_module
