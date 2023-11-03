1. Added `torch.cuda.init()` to the start of `BackendWorker`'s `run()`.
   Prior to this, the `torch.load()` call after the warmup batch was
   received was the first to initialize CUDA and there would be a long
   `cudaGetDeviceSetStreamPriorityRange` in the profile. After this change,
   the `torch.load` time went down significantly.
