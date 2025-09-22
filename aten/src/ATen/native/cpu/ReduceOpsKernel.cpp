#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <algorithm>

#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/OpMathType.h>
#include <ATen/cpu/vec/vec.h>
#include <ATen/cpu/vec/functional.h>
#include <ATen/native/ReduceOps.h>
#include <ATen/native/Resize.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/SharedReduceOps.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <ATen/native/cpu/Reduce.h>
#include <ATen/native/cpu/LogAddExp.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/imag.h>
#endif

#include <c10/util/irange.h>
#include <ATen/AccumulateType.h>

namespace at::native { namespace {

using namespace vec;

template <typename scalar_t, typename func_t>
static inline void cpu_cum_base_kernel(const Tensor& result,
    const Tensor& self,
    int64_t dim,
    const func_t& f,
    scalar_t init_val) {
  if (result.sizes() != self.sizes()) {
    at::native::resize_output(result, self.sizes());
  }
  if (self.numel() == 0) {
    return;
  }
  const auto input_ndim = self.dim();
  if (input_ndim == 0) {
    result.fill_(self);
    return;
  }

  // TODO This probably should be using at::native::make_reduction
  auto iter = TensorIteratorConfig()
    .check_all_same_dtype(false)
    .resize_outputs(false)
    .declare_static_shape(self.sizes(), /*squash_dims=*/dim)
    .add_output(result)
    .add_const_input(self)
    .build();

  auto result_dim_stride = ensure_nonempty_stride(result, dim);
  auto self_dim_stride = ensure_nonempty_stride(self, dim);

  auto loop = [&](char** data, const int64_t* strides, int64_t n) {
    auto* result_data_bytes = data[0];
    const auto* self_data_bytes = data[1];

    for ([[maybe_unused]] const auto i : c10::irange(n)) {
      f((scalar_t*)result_data_bytes,
        result_dim_stride,
        (scalar_t*)self_data_bytes,
        self_dim_stride,
        init_val);
      result_data_bytes += strides[0];
      self_data_bytes += strides[1];
    }
  };

  int64_t grain_size = internal::GRAIN_SIZE / std::max(int64_t{1}, self.size(dim));
  iter.for_each(loop, grain_size);
}

static void cumsum_cpu_kernel(const Tensor& result, const Tensor& self, int64_t dim) {
  auto wrap_dim = maybe_wrap_dim(dim, self.dim());
  int64_t self_dim_size = ensure_nonempty_size(self, wrap_dim);

  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(kBFloat16, kHalf, self.scalar_type(), "cumsum_out_cpu", [&] {
    cpu_cum_base_kernel<scalar_t>(result, self, wrap_dim, [&] (
      scalar_t* result_data, auto result_dim_stride,
      const scalar_t* self_data, auto self_dim_stride, scalar_t init_val) {
        // NOLINTNEXTLINE(bugprone-signed-char-misuse)
        auto cum_number = (at::acc_type<scalar_t, false>)init_val;
        for (const auto i : c10::irange(self_dim_size)) {
          cum_number += self_data[i * self_dim_stride];
          result_data[i * result_dim_stride] = (scalar_t)cum_number;
        }
      }, /*init_val=*/ 0
    );
  });
}

static void cumprod_cpu_kernel(const Tensor& result, const Tensor& self, int64_t dim) {
  auto wrap_dim = maybe_wrap_dim(dim, self.dim());
  int64_t self_dim_size = ensure_nonempty_size(self, wrap_dim);

  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(kBFloat16, kHalf, self.scalar_type(), "cumprod_out_cpu", [&] {
    cpu_cum_base_kernel<scalar_t>(result, self, wrap_dim, [&] (
      scalar_t* result_data, auto result_dim_stride,
      const scalar_t* self_data, auto self_dim_stride, scalar_t init_val) {
        // NOLINTNEXTLINE(bugprone-signed-char-misuse)
        auto cum_number = (at::acc_type<scalar_t, false>)init_val;
        for (const auto i : c10::irange(self_dim_size)) {
          cum_number *= self_data[i * self_dim_stride];
          result_data[i * result_dim_stride] = (scalar_t)cum_number;
        }
      }, /*init_val=*/ 1
    );
  });
}

static void logcumsumexp_cpu_kernel(Tensor& result, const Tensor& self, int64_t dim) {
  auto wrap_dim = maybe_wrap_dim(dim, self.dim());
  int64_t self_dim_size = ensure_nonempty_size(self, wrap_dim);

  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(kBFloat16, kHalf, self.scalar_type(), "logcumsumexp_out_cpu", [&] {
    cpu_cum_base_kernel<scalar_t>(result, self, wrap_dim, [&] (
      scalar_t* result_data, auto result_dim_stride,
      const scalar_t* self_data, auto self_dim_stride, scalar_t init_val) {
        using accscalar_t = at::acc_type<scalar_t, false>;
        auto cum_number = (accscalar_t)init_val;
        for (const auto i : c10::irange(self_dim_size)) {
          accscalar_t x = self_data[i * self_dim_stride];

          cum_number = _log_add_exp_helper(x, cum_number);
          result_data[i * result_dim_stride] = static_cast<scalar_t>(cum_number);
        }
      }, /*init_val=*/ -std::numeric_limits<scalar_t>::infinity()
    );
  });
}

static void std_var_kernel_impl(TensorIterator& iter, double correction, bool take_sqrt) {
  AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16, iter.dtype(), "std_cpu", [&] {
    binary_kernel_reduce(
        iter,
        WelfordOps<
            scalar_t,
            double,
            int64_t,
            std::tuple<scalar_t, scalar_t>>{correction, take_sqrt},
        WelfordData<double, int64_t>());
  });
}

static void prod_kernel_impl(TensorIterator& iter) {
  // Workaround for the error: '*' in boolean context, suggest '&&' instead
  if (iter.dtype() == ScalarType::Bool) {
    using scalar_t = bool;
    binary_kernel_reduce_vec(
        iter,
        [=](scalar_t a, scalar_t b)
            -> scalar_t { return a && b; },
        [=](Vectorized<scalar_t> a, Vectorized<scalar_t> b)
            { return a && b; },
        /*ident=*/1);
  } else {
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(kBFloat16, kHalf, iter.dtype(), "prod_out_cpu", [&] {
      binary_kernel_reduce_vec(
          iter,
          [=](scalar_t a, scalar_t b)
              -> scalar_t { return a * b; },
          [=](Vectorized<scalar_t> a, Vectorized<scalar_t> b)
              { return a * b; },
          /*ident=*/1);
    });
  }
}

template <typename scalar_t, typename acc_t>
inline void norm_two_reduce_step(Vectorized<acc_t>& acc_vec, Vectorized<scalar_t>& data_vec) {
  acc_vec += data_vec * data_vec;
}

template <>
inline void norm_two_reduce_step(Vectorized<float>& acc_fvec, Vectorized<BFloat16>& data_bvec) {
  auto [data_fvec0, data_fvec1] = convert_bfloat16_float(data_bvec);
  acc_fvec += data_fvec0 * data_fvec0;
  acc_fvec += data_fvec1 * data_fvec1;
}

template <typename scalar_t, typename out_t=typename scalar_value_type<scalar_t>::type>
void norm_kernel_cpu_impl(TensorIterator& iter, const double& val) {
  // This reduction accumulates results as the type `acc_t`.
  using acc_t = at::opmath_type<typename scalar_value_type<scalar_t>::type>;
  if (val == 0.0) {
    binary_kernel_reduce(iter, NormZeroOps<scalar_t, acc_t, out_t>(), acc_t(0));
  } else if (val == 1.0) {
    binary_kernel_reduce(iter, NormOneOps<scalar_t, acc_t, out_t>(), acc_t(0));
  } else if (val == 2.0) {
    binary_kernel_reduce(iter, NormTwoOps<scalar_t, acc_t, out_t>(), acc_t(0));
  } else if (val == INFINITY) {
    binary_kernel_reduce(iter, AbsMaxOps<scalar_t, acc_t, out_t>(), acc_t(0));
  } else if (val == -INFINITY) {
    binary_kernel_reduce(iter, AbsMinOps<scalar_t, acc_t, out_t>(), std::numeric_limits<acc_t>::infinity());
  } else {
    binary_kernel_reduce(iter, NormOps<scalar_t, acc_t, out_t>{acc_t(val)}, acc_t(0));
  }
}

static void norm_kernel_tensor_iterator_impl(
    TensorIterator& iter,
    const Scalar& p) {
  double val = 0;
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

  if (val == 2.0 && is_reduce_lastdim(iter) &&
      iter.dtype(0) == iter.input_dtype() &&
      (iter.input_dtype() == kFloat || iter.input_dtype() == kDouble ||
       iter.input_dtype() == kBFloat16)) {
    // If we can vectorize over the last dimension and the dtype
    // of the output is the same as that of the input,
    // then we go through the vectorised path.
    AT_DISPATCH_FLOATING_TYPES_AND(kBFloat16, iter.input_dtype(), "norm_cpu", [&] {
        // use float as accumulate type for BFloat16
        using acc_t = at::opmath_type<scalar_t>;
        binary_kernel_reduce_lastdim(iter, [](char* result_data_bytes, char* self_data_bytes, int64_t size) {
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
    if (iter.input_dtype() == kHalf && iter.dtype(0) == kFloat) {
      // type promotion that does cast and reduction in a single kernel
      return norm_kernel_cpu_impl<at::Half, float>(iter, val);
    } else if (iter.input_dtype() == kBFloat16 && iter.dtype(0) == kFloat) {
      // type promotion that does cast and reduction in a single kernel
      return norm_kernel_cpu_impl<at::BFloat16, float>(iter, val);
    }

    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND3(kHalf, kBFloat16, kComplexHalf, iter.input_dtype(), "norm_cpu", [&] {
      norm_kernel_cpu_impl<scalar_t>(iter, val);
    });

    // For complex outputs, the above kernels do not touch the imaginary values,
    // so we must zero them out
    if (isComplexType(iter.output().scalar_type())) {
      at::imag(iter.output()).zero_();
    }
  }
}

static void and_kernel_impl(TensorIterator& iter) {
  if (iter.dtype() == ScalarType::Byte) {
    // Refer [all, any : uint8 compatibility]
    binary_kernel_reduce_vec(
        iter,
        [=](uint8_t a, uint8_t b) -> uint8_t { return (a && b) ? 1 : 0; },
        [=](Vectorized<uint8_t> a, Vectorized<uint8_t> b) {
          // NB: != returns 0xFF rather than 0x01, so we must negate to get
          // the desired result
          return (a != Vectorized<uint8_t>(0)).neg() & (b != Vectorized<uint8_t>(0)).neg();
        },
        /*ident=*/true);
  } else {
    binary_kernel_reduce_vec(
        iter,
        [=](bool a, bool b) -> bool { return a && b; },
        [=](Vectorized<bool> a, Vectorized<bool> b) {
          // Adding the implementation here instead of in vec256_base to avoid
          // return value inconsistency. Other comparison operators in
          // vec256_base return -1/0 (all bit 1 / all bit 0) as true/false to
          // follow the AVX2 convention. This would be convenient when combined
          // with other vectorized operations. For example, one can use the
          // logical operation results as a mask for a bit operation to
          // retrieve/reset multiple elements in a vector.
          //
          // In this method, users would expect, e.g., all(), to return 1/0 as
          // true/false.
          Vectorized<bool> c = Vectorized<bool>();

          for (decltype(c.size()) i = 0; i != Vectorized<bool>::size(); i++) {
            c[i] = a[i] && b[i];
          }
          return c;
        },
        /*ident=*/true);
  }
}

static void or_kernel_impl(TensorIterator& iter) {
  if (iter.dtype() == ScalarType::Byte) {
    // Refer [all, any : uint8 compatibility]
    binary_kernel_reduce_vec(
        iter,
        [=](uint8_t a, uint8_t b) -> uint8_t { return (a || b) ? 1 : 0; },
        [=](Vectorized<uint8_t> a, Vectorized<uint8_t> b) {
          return (a != Vectorized<uint8_t>(0)).neg() | (b != Vectorized<uint8_t>(0)).neg();
        },
        /*ident=*/false);
  } else {
    binary_kernel_reduce_vec(
        iter,
        [=](bool a, bool b) -> bool { return a || b; },
        [=](Vectorized<bool> a, Vectorized<bool> b) {
          Vectorized<bool> c = Vectorized<bool>();

          for (decltype(c.size()) i = 0; i != Vectorized<bool>::size(); i++) {
            c[i] = a[i] || b[i];
          }
          return c;
        },
        /*ident=*/false);
  }
}

template<typename scalar_t>
struct MinValuesOps: public at::native::MinOps<scalar_t> {
  using arg_t = typename MinOps<scalar_t>::arg_t;
  static scalar_t project(arg_t arg) {
    return arg.first;
  }
};

static void min_values_kernel_impl(TensorIterator& iter) {
  if (iter.dtype() == kLong) {
    // This case is special because of Vectorized<int64_t> does not
    // handle upper_bound<int64_t>().
    // See: https://github.com/pytorch/pytorch/issues/43254
    using scalar_t = int64_t;
    binary_kernel_reduce(
      iter,
      MinValuesOps<scalar_t>{},
      std::pair<scalar_t, int64_t>(upper_bound<scalar_t>(), -1));
    return;
  }
  AT_DISPATCH_ALL_TYPES_AND3(kBFloat16, kHalf, kBool, iter.dtype(), "min_values_cpu", [&iter] {
    binary_kernel_reduce_vec(
      iter,
      [](scalar_t a, scalar_t b) -> scalar_t { return min_impl(a, b); },
      [](Vectorized<scalar_t> a, Vectorized<scalar_t> b) { return minimum(a, b); },
      static_cast<double>(upper_bound<scalar_t>()));
  });
}

static void max_values_kernel_impl(TensorIterator& iter) {
  AT_DISPATCH_ALL_TYPES_AND3(kBFloat16, kHalf, kBool, iter.dtype(), "max_values_cpu", [&iter] {
    binary_kernel_reduce_vec(
      iter,
      [](scalar_t a, scalar_t b) -> scalar_t { return max_impl(a, b); },
      [](Vectorized<scalar_t> a, Vectorized<scalar_t> b) { return maximum(a, b); },
      lower_bound<scalar_t>());
  });
}

static void argmax_kernel_impl(TensorIterator &iter) {
  AT_DISPATCH_ALL_TYPES_AND2(kHalf, kBFloat16, iter.dtype(1), "argmax_cpu", [&] {
    if (is_reduce_lastdim(iter)) {
      using arg_t = std::pair<scalar_t, int64_t>;
      auto op = ArgMaxOps<scalar_t>{};
      binary_kernel_reduce_lastdim(iter, [&](char* result_data_bytes, char* self_data_bytes, int64_t size) {
        int64_t* result_data = (int64_t*)result_data_bytes;
        scalar_t* self_data = (scalar_t*)self_data_bytes;

        arg_t acc = arg_t(lower_bound<scalar_t>(), 0);
        for (int64_t i = 0; i < size; i++) {
          acc = op.reduce(acc, self_data[i], i);
        }
        result_data[0] = acc.second;
      });
      return;
    }
    binary_kernel_reduce(
      iter,
      ArgMaxOps<scalar_t>{},
      std::pair<scalar_t, int64_t>(lower_bound<scalar_t>(), 0));
  });
}

static void argmin_kernel_impl(TensorIterator &iter) {
  AT_DISPATCH_ALL_TYPES_AND2(kHalf, kBFloat16, iter.dtype(1), "argmin_cpu", [&] {
    if (is_reduce_lastdim(iter)) {
      using arg_t = std::pair<scalar_t, int64_t>;
      auto op = ArgMinOps<scalar_t>{};
      binary_kernel_reduce_lastdim(iter, [&](char* result_data_bytes, char* self_data_bytes, int64_t size) {
        int64_t* result_data = (int64_t*)result_data_bytes;
        scalar_t* self_data = (scalar_t*)self_data_bytes;

        arg_t acc = arg_t(upper_bound<scalar_t>(), 0);
        for (int64_t i = 0; i < size; i++) {
          acc = op.reduce(acc, self_data[i], i);
        }
        result_data[0] = acc.second;
      });
      return;
    }
    binary_kernel_reduce(
      iter,
      ArgMinOps<scalar_t>{},
      std::pair<scalar_t, int64_t>(upper_bound<scalar_t>(), 0));
  });
}

template <typename scalar_t, typename acc_t = uint64_t, typename out_t = acc_t>
struct XorSumOps {
  inline C10_DEVICE acc_t reduce(acc_t acc, scalar_t data, int64_t /*idx*/) const {
    if (std::is_same<scalar_t, bool>::value) {
      return acc ^ (data ? 1 : 0);
    } else if (
        std::is_same<scalar_t, float>::value ||
        std::is_same<scalar_t, double>::value ||
        std::is_same<scalar_t, at::BFloat16>::value ||
        std::is_same<scalar_t, at::Half>::value) {
      union {
        double d;
        uint64_t u;
      } converter;
      converter.d = static_cast<double>(data);
      return acc ^ converter.u;
    } else {
      return acc ^ static_cast<uint64_t>(data);
    }
  }

  inline C10_DEVICE acc_t combine(acc_t a, acc_t b) const {
    return a ^ b;
  }

  inline C10_DEVICE out_t project(acc_t a) const {
    return a;
  }

  static C10_DEVICE acc_t translate_idx(acc_t acc, int64_t /*base_idx*/) {
    return acc;
  }
};

static void xor_sum_kernel_impl(TensorIterator& iter) {
  // Use iter.dtype(1) to dispatch based on the type of the input tensor
  AT_DISPATCH_ALL_TYPES_AND3(
      kBFloat16, kHalf, kBool, iter.dtype(1), "xor_sum_cpu", [&] {
        binary_kernel_reduce(
            iter, XorSumOps<scalar_t>(), static_cast<uint64_t>(0));
      });
}

}  // anonymous namespace

REGISTER_DISPATCH(std_var_stub, &std_var_kernel_impl)
REGISTER_DISPATCH(prod_stub, &prod_kernel_impl)
// mean implementation for CPU is in aten/src/ATen/native/ReduceOps.cpp
// but mean_stub must be defined for CPU as well
REGISTER_DISPATCH(mean_stub, nullptr)
REGISTER_DISPATCH(norm_stub, &norm_kernel_tensor_iterator_impl)
REGISTER_DISPATCH(and_stub, &and_kernel_impl)
REGISTER_DISPATCH(or_stub, &or_kernel_impl)
REGISTER_DISPATCH(min_values_stub, &min_values_kernel_impl)
REGISTER_DISPATCH(max_values_stub, &max_values_kernel_impl)
REGISTER_DISPATCH(argmax_stub, &argmax_kernel_impl)
REGISTER_DISPATCH(argmin_stub, &argmin_kernel_impl)
REGISTER_DISPATCH(xor_sum_stub, &xor_sum_kernel_impl)

REGISTER_DISPATCH(cumprod_stub, &cumprod_cpu_kernel)
REGISTER_DISPATCH(cumsum_stub, &cumsum_cpu_kernel)
REGISTER_DISPATCH(logcumsumexp_stub, &logcumsumexp_cpu_kernel)

}  // namespace at::native
