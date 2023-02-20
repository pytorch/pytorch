#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <algorithm>

#include <ATen/Dispatch.h>
#include <ATen/OpMathType.h>
#include <ATen/core/Tensor.h>
#include <ATen/cpu/vec/functional.h>
#include <ATen/cpu/vec/vec.h>
#include <ATen/native/ReduceOps.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <ATen/native/Resize.h>
#include <ATen/native/SharedReduceOps.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cpu/Reduce.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/imag.h>
#endif

#include <ATen/AccumulateType.h>
#include <c10/util/Optional.h>
#include <c10/util/irange.h>

namespace at::native {
namespace {

template <typename scalar_t, typename acc_t>
inline void norm_two_reduce_step(
    Vectorized<acc_t>& acc_vec,
    Vectorized<scalar_t>& data_vec) {
  acc_vec += data_vec * data_vec;
}

template <>
inline void norm_two_reduce_step(
    Vectorized<float>& acc_fvec,
    Vectorized<BFloat16>& data_bvec) {
  Vectorized<float> data_fvec0, data_fvec1;
  std::tie(data_fvec0, data_fvec1) = convert_bfloat16_float(data_bvec);
  acc_fvec += data_fvec0 * data_fvec0;
  acc_fvec += data_fvec1 * data_fvec1;
}

template <
    typename scalar_t,
    typename acc_t = typename scalar_value_type<scalar_t>::type,
    typename out_t = typename scalar_value_type<scalar_t>::type>
void norm_kernel_cpu_impl(TensorIterator& iter, const double& val) {
  if (val == static_cast<double>(0)) {
    binary_kernel_reduce(iter, NormZeroOps<scalar_t, acc_t, out_t>(), acc_t(0));
  } else if (val == static_cast<double>(1)) {
    binary_kernel_reduce(iter, NormOneOps<scalar_t, acc_t, out_t>(), acc_t(0));
  } else if (val == static_cast<double>(2)) {
    binary_kernel_reduce(iter, NormTwoOps<scalar_t, acc_t, out_t>(), acc_t(0));
  } else if (val == static_cast<double>(INFINITY)) {
    binary_kernel_reduce(iter, AbsMaxOps<scalar_t, acc_t, out_t>(), acc_t(0));
  } else if (val == static_cast<double>(-INFINITY)) {
    binary_kernel_reduce(
        iter,
        AbsMinOps<scalar_t, acc_t, out_t>(),
        std::numeric_limits<acc_t>::infinity());
  } else {
    binary_kernel_reduce(
        iter, NormOps<scalar_t, acc_t, out_t>{acc_t(val)}, acc_t(0));
  }
}

void norm_kernel_cpu(TensorIterator& iter, const Scalar& p) {
  double val;
  if (p.isIntegral(false)) {
    val = p.to<int64_t>();
  } else if (p.isFloatingPoint()) {
    val = p.to<double>();
  } else {
    TORCH_CHECK(false, "norm_kernel_cpu expects norm to be integer or float");
  }
  if (iter.numel() == 0) {
    iter.output().fill_((val < 0) ? INFINITY : 0);
    return;
  }

  if (val == static_cast<double>(2) && is_reduce_lastdim(iter) &&
      iter.dtype(0) == iter.input_dtype() &&
      (iter.input_dtype() == kFloat || iter.input_dtype() == kDouble ||
       iter.input_dtype() == kBFloat16)) {
    AT_DISPATCH_FLOATING_TYPES_AND(
        kBFloat16, iter.input_dtype(), "norm_cpu", [&] {
          // use float as accumulate type for BFloat16
          using acc_t = at::opmath_type<scalar_t>;
          binary_kernel_reduce_lastdim(
              iter,
              [](char* result_data_bytes, char* self_data_bytes, int64_t size) {
                scalar_t* result_data = (scalar_t*)result_data_bytes;
                scalar_t* self_data = (scalar_t*)self_data_bytes;

                using Vec = Vectorized<scalar_t>;
                using fVec = Vectorized<acc_t>;
                fVec acc_vec{acc_t(0)};
                acc_t buffer[fVec::size()];
                int64_t d = 0;
                for (; d < size - (size % Vec::size()); d += Vec::size()) {
                  Vec data_vec = Vec::loadu(self_data + d);
                  norm_two_reduce_step(acc_vec, data_vec);
                }
                acc_vec.store(buffer);
                for (int j = 1; j < fVec::size(); j++) {
                  buffer[0] = buffer[0] + buffer[j];
                }
                for (; d < size; d++) {
                  acc_t data_val = acc_t(self_data[d]);
                  buffer[0] += data_val * data_val;
                }
                result_data[0] = scalar_t(std::sqrt(buffer[0]));
              });
        });
  } else {
    if (iter.dtype(0) == kHalf) {
      return norm_kernel_cpu_impl<at::Half, float>(iter, val);
    } else if (iter.dtype(0) == kBFloat16) {
      return norm_kernel_cpu_impl<at::BFloat16, float>(iter, val);
    }
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(iter.input_dtype(), "norm_cpu", [&] {
      norm_kernel_cpu_impl<scalar_t>(iter, val);
    });

    if (isComplexType(iter.output().scalar_type())) {
      at::imag(iter.output()).zero_();
    }
  }
}

} // anonymous namespace

REGISTER_DISPATCH(norm_stub, &norm_kernel_cpu);

} // namespace at::native
