#include <ATen/DeviceAccelerator.h>

namespace at {

C10_API std::optional<DeviceType> getAccelerator(bool checked) {
#ifdef USE_CUDA
#define CHECK_NO_CUDA \
  static_assert(false, "Only one of USE_CUDA, USE_HIP and USE_MPS can be defined at once.");
#else
#define CHECK_NO_CUDA
#endif

#define CHECK_NO_MTAI_PU1 \
  TORCH_CHECK(!(at::detail::getMTIAHooks().hasMTIA() || is_privateuse1_backend_registered()), "Cannot have both CUDA, MTIA or PrivateUse1");


#if defined(USE_CUDA)
    CHECK_NO_MTAI_PU1
    return kCUDA;
#else
    CHECK_NO_CUDA
    TORCH_CHECK(!(at::detail::getMTIAHooks().hasMTIA() && is_privateuse1_backend_registered()), "Cannot have both MTIA and PrivateUse1");
    if(at::detail::getMTIAHooks().hasMTIA()) {
        return kMTIA;
    } else if (is_privateuse1_backend_registered()) {
        return kPrivateUse1;
    } else {
        TORCH_CHECK(!checked, "Cannot access accelerator device when none is available.")
        return std::nullopt;
    }
#endif

#undef CHECK_NO_CUDA
#undef CHECK_NO_MTAI_PU1
}


} // namespace at
