#include "ATen/native/cpu/UnaryOpsKernel.h"
#include <cmath>
#include "ATen/Dispatch.h"
#include "ATen/Parallel.h"
#include "ATen/native/cpu/Vec256.h"

namespace at { namespace native {

using namespace vec256;

// This modifies arr in place with given OP
template <class scalar_t, template <class> class VOP, CPUCapability C>
inline void
kernel_(scalar_t* arr_out, const scalar_t* arr_in, size_t start, size_t end) {
  Vec256<scalar_t> a;
  size_t epr = 32 / sizeof(scalar_t); // primitives per Vec256
  size_t k = start;
  size_t vec_end = end > epr ? end - epr : 0;
  for (; k < vec_end; k += epr) {
    a.load(arr_in + k);
    VOP<scalar_t>()(a).store(arr_out + k);
  }
  size_t leftover = std::min((end - k), a.size);
  a.load(arr_in + k, leftover);
  VOP<scalar_t>()(a).store(arr_out + k, leftover);
}

namespace {

#define FUNCVOP(NAME)                          \
  template <typename T>                        \
  struct NAME##VOP {                           \
    Vec256<T> operator()(Vec256<T>& x) const { \
      return x.NAME();                         \
    }                                          \
  };

UNARY_OPS_MACRO(FUNCVOP)

} // namespace

#define FUNCImpl(NAME)                                                      \
  template <>                                                               \
  void NAME##ImplC<CURRENT_CAPABILITY>::function(                           \
      Tensor& result, const Tensor& self) {                                 \
    AT_DISPATCH_FLOATING_TYPES(self.type(), NAME, [&] {                     \
      at::parallel_for_1d<scalar_t>(                                        \
          &kernel_<scalar_t, NAME##VOP, CURRENT_CAPABILITY>, result, self); \
    });                                                                     \
  }

UNARY_OPS_MACRO(FUNCImpl)

}} // namespace at::native
