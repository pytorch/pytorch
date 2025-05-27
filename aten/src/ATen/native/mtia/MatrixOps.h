#include <ATen/core/Tensor.h>
#include <ATen/native/DispatchStub.h>

namespace at::native {

// This registration is just to make linker happy!
// Explanation: We use DispatchStub as the bridge between pytorch and mtia's own
// code. DispatchStub requires DEFAULT(and AVX2 as well if compile with flag
// HAVE_AVX2_CPU_DEFINITION) to be defined, thus we register for DEFAULT/AVX2
// here to avoid the "undefined symbol" link error. And these no-op funcitons
// should never be called anyway, as it's only called when the dispatch key is
// mtia while the device_type is not mtia.
#define REGISTER_CPU_DISPATCH_NO_OP(op_name, return_type, ...)             \
  return_type op_name##_impl_SHOULD_NOT_BE_CALLED(__VA_ARGS__) {           \
    TORCH_CHECK(false, #op_name ": MTIA op should not be called for CPU"); \
  }                                                                        \
  REGISTER_ARCH_DISPATCH(                                                  \
      op_name##_mtia_stub, DEFAULT, &op_name##_impl_SHOULD_NOT_BE_CALLED)  \
  REGISTER_AVX2_DISPATCH(                                                  \
      op_name##_mtia_stub, &op_name##_impl_SHOULD_NOT_BE_CALLED)

using mm_out_fn = const at::Tensor& (*)(const at::Tensor&,
                                        const at::Tensor&,
                                        const at::Tensor&);
DECLARE_DISPATCH(mm_out_fn, mm_out_mtia_stub)

} // namespace at::native
