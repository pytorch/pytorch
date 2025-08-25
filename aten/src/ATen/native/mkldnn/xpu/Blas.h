#include <ATen/ATen.h>
#include <ATen/native/DispatchStub.h>

namespace at::native {

using mm_complex_fn = at::Tensor& (*)(const at::Tensor&,
                                      const at::Tensor&,
                                      Tensor&);

using addmm_complex_fn = at::Tensor& (*)(const at::Tensor&,
                                         const at::Tensor&,
                                         const at::Tensor&,
                                         const at::Scalar&,
                                         const at::Scalar&,
                                         at::Tensor&);

DECLARE_DISPATCH(mm_complex_fn, mm_complex_stub);
DECLARE_DISPATCH(mm_complex_fn, bmm_complex_stub);
DECLARE_DISPATCH(addmm_complex_fn, addmm_complex_stub);
DECLARE_DISPATCH(addmm_complex_fn, baddbmm_complex_stub);

} // namespace at::native
