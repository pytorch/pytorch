#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>

// THIS IS A MASSIVE HACK.  This will BREAK you Caffe2 CUDA code if you
// load ATen_hip, even if you don't ever actually use ATen_hip at runtime.
//
// If you ever link ATen_hip statically into the full library along
// with ATen_cuda (libomnibus), the loading order of this versus the regular
// ATen_cuda will be nondeterministic, and you'll nondeterministically get
// one or the other.  (This will be obvious because all of your code
// will fail.)
//
// This hack can be removed once PyTorch is out-of-place HIPified, and
// doesn't pretend CUDA is HIP.
C10_REGISTER_GUARD_IMPL(CUDA, at::cuda::HIPGuardImplMasqueradingAsCUDA);
