import torch

import torch_openreg._C  # type: ignore[misc]
import torch_openreg.openreg


torch.utils.rename_privateuse1_backend("openreg")
torch._register_device_module("openreg", torch_openreg.openreg)
torch.utils.generate_methods_for_privateuse1_backend(for_storage=True)
