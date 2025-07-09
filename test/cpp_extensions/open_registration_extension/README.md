# PyTorch OpenReg

This folder contains a self-contained example of a PyTorch out-of-tree backend leveraging the "PrivateUse1" backend from core.

## How to use

Install as standalone with `python -m pip install -e .` (or `python -m pip install .`)
from this folder. You can run test via `python {PYTORCH_ROOT_PATH}/test/test_openreg.py`.

## Design principles

For simplicity anything that can be implemented from python is done so.
A real implementation will most likely want to call these different APIs from c++ directly.

The current version sends everything back to python and contains enough implementation to run basic model, transfer host/device and printing.

The codebase is split as follows:

- `pytorch_openreg/__init__.py`
  - imports torch to get core state initialized.
  - imports `._aten_impl` to register our aten op implementations to torch.
  - imports `.C` to load our c++ extension that registers more ops, allocator and hooks.
  - renames the PrivateUse1 backend and register our python-side module.
- `pytorch_openreg/_aten_impl.py`
  - Define a new `torch.Library` that registers a fallback that will be called whenever a backend kernel for PrivateUse1 is called. It contains the logic to handle all kind of native functions, computing the output metadata, allocating it and only calling into the device daemon to perform computation.
- `pytorch_openreg/_device_daemon.py`
  - contains the Allocator (responsible for allocating memory on the device side and host side, as int8 buffers).
  - contains `Driver`, which as user-process driver to deal with some information needed to be done in driver.
  - contains `Executor`, which as device-process exector to do something related device logic.
- `pytorch_openreg/_meta_parser.py` mainly contain utilities to send objects over the wire from the user process to the device process.
  - The main class there is `OpenRegTensorMeta` that contains all the metadata sent to the device which should be enough for it to populate the output Tensor.

## Next steps

The main next step would be to:

- Replace the current `open_registration_extension.cpp` test in PyTorch CI with this.
