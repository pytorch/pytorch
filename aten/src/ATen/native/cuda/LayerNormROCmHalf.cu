// Half instantiation of the ROCm cached-row LayerNorm / RMSNorm kernels (see LayerNormROCm.cuh).
#include <ATen/native/cuda/LayerNormROCm.cuh>

#ifdef USE_ROCM
namespace at::native::rocm_ln {
template bool forward<c10::Half>(const FwdArgs&);
template bool backward<c10::Half>(const BwdArgs&, const WsAlloc&);
}  // namespace at::native::rocm_ln
#endif
