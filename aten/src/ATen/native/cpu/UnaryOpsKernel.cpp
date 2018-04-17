#include "ATen/native/cpu/UnaryOpsKernel.h"
#include <cmath>
#include <iostream>
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

template <template <class> class VOP, CPUCapability C>
inline void allImpl(Tensor& result, const Tensor& self, const char* name) {
  AT_DISPATCH_FLOATING_TYPES(self.type(), name, [&] {
    at::parallel_for_1d<scalar_t>(
        &kernel_<scalar_t, VOP, CURRENT_CAPABILITY>, result, self);
  });
}

namespace {

template <typename T>
struct ceilVOP {
  Vec256<T> operator()(Vec256<T>& x) const {
    return x.ceil();
  }
};
} // namespace

template <>
void ceilImplC<CURRENT_CAPABILITY>::function(
    Tensor& result,
    const Tensor& self) {
  allImpl<ceilVOP, CURRENT_CAPABILITY>(result, self, "ceil");
}

namespace {

template <typename T>
struct floorVOP {
  Vec256<T> operator()(Vec256<T>& x) const {
    return x.floor();
  }
};
} // namespace

template <>
void floorImplC<CURRENT_CAPABILITY>::function(
    Tensor& result,
    const Tensor& self) {
  allImpl<floorVOP, CURRENT_CAPABILITY>(result, self, "floor");
}

namespace {

template <typename T>
struct roundVOP {
  Vec256<T> operator()(Vec256<T>& x) const {
    return x.round();
  }
};
} // namespace

template <>
void roundImplC<CURRENT_CAPABILITY>::function(
    Tensor& result,
    const Tensor& self) {
  allImpl<roundVOP, CURRENT_CAPABILITY>(result, self, "round");
}

namespace {

template <typename T>
struct truncVOP {
  Vec256<T> operator()(Vec256<T>& x) const {
    return x.trunc();
  }
};
} // namespace

template <>
void truncImplC<CURRENT_CAPABILITY>::function(
    Tensor& result,
    const Tensor& self) {
  allImpl<truncVOP, CURRENT_CAPABILITY>(result, self, "trunc");
}

namespace {

template <typename T>
struct sqrtVOP {
  Vec256<T> operator()(Vec256<T>& x) const {
    return x.sqrt();
  }
};
} // namespace

template <>
void sqrtImplC<CURRENT_CAPABILITY>::function(
    Tensor& result,
    const Tensor& self) {
  allImpl<sqrtVOP, CURRENT_CAPABILITY>(result, self, "sqrt");
}
}} // namespace at::native
