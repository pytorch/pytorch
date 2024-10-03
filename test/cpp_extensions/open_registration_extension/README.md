This folder contains a self-contained example of a PyTorch out-of-tree backend leveraging the "PrivateUse1" backend from core.

## How to use
Install as standalone with `python setup.py develop` (or install) from this folder.
You can run test via `python test/test_openreg.py`.

## Design principles
For simplicity anything that can be implemented from python is done so.
A real implementation will most likely want to call these different APIs from c++ directly.

The current version sends everything back to python and contains enough implementation to run basic model, transfer host/device and printing.

The codebase is split as follows:
- `pytorch_openreg/__init__.py` imports torch to get core state initialized, imports `._aten_impl` to register our aten op implementations to torch, imports `.C` to load our c++ extension that registers more ops, allocator and hooks and finally renames the PrivateUse1 backend and register our python-side module.
- `pytorch_openreg/_aten_impl.py` does two main things. Use the `_register_same_name()` function to register hooks from c++ (like getDevice, getStream, etc) and send them to our device daemon. Define a new `torch.Library` that registers a fallback that will be called whenever a backend kernel for PrivateUse1 is called. It contains the logic to handle all kind of native functions, computing the output metadata, allocating it and only calling into the device daemon to perform computation
- `pytorch_openreg/_device_daemon.py` contains the Allocator (responsible for allocating memory on the device side, as int8 buffers, and recreating nice looking Tensors on the device side to be able to use aten ops to run code there), `run_op` that is the logic running on the device side to perform compute (for simplicity of coverage, we are re-building full blown Tensors here and calling aten ops on them). It also contains the Daemon responsible for the device worker process and sending data back and forth.
- `pytorch_openreg/_meta_parser.py` mainly contain utilities to send objects over the wire from the user process to the device process. The main class there is `OpenRegTensorMeta` that contains all the metadata sent to the device which should be enough for it to populate the output Tensor.

## Next steps

Currently, the autograd test is disabled because it's missing the getStream implementation.
The main next step would be to:
- Split the daemon into a proper user-process driver vs device-process executor. The main goal would be to better mimick which information is held on the user-process side and when we're actually communicating with the device. In particular current device or stream should be user-process informations.
- Add Stream/Event system. Most likely by having multiple requests queue that go to the device from the driver.
- Add RNG Generator.
- Add Pinned memory and HostAllocator.

Longer term:
- Replace the current `open_registration_extension.cpp` test in PyTorch CI with this.
- Build this module in the CI environment and enable Device-generic tests on this device.
