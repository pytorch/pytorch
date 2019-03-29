This folder contains the c10 dispatcher. This dispatcher is a single point
through which we are planning to route all kernel calls.
Existing dispatch mechanisms from legacy PyTorch or caffe2 are planned to
be replaced.

This folder contains the following files:
- Dispatcher.h: Main facade interface. Code using the dispatcher should only use this.
- DispatchTable.h: Implementation of the actual dispatch mechanism. Hash table with kernels, lookup, ...
- KernelCache.h: An interface operator kernels can use to inherit from if they need to keep around a cache between invocations
- KernelFunction.h: The core interface (i.e. function pointer) for calling a kernel
- OpSchemaRegistration.h: The mechanisms to register new operators with the c10 dispatcher
- KernelRegistration.h: The mechanisms to register kernels with the c10 dispatcher
