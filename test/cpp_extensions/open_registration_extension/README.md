This folder contains a self-contained example of a PyTorch out-of-tree backend leveraging the "PrivateUse1" backend in core.

## How to use
Install as standalone with `python setup.py develop` (or install) from this folder.
You can run test via `python test/test_openreg.py`.

## Design principles
For simplicity anything that can be implemented from python is done so.
A real implementation will most likely want to call these different APIs from c++ directly.

The current version send everything back to python and is missing most implementations in python. The only one available is the one used by the autograd engine to check how many workers to spawn.

Next step is to create the device daemon so we can actually provide and allocator and create memory, then start using features and re-route all missing methods to daemon as appropriate.
