#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <algorithm>

#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/cpu/vec/vec.h>
#include <ATen/cpu/vec/functional.h>
#include <ATen/native/ReduceOps.h>
#include <ATen/native/Resize.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/SharedReduceOps.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <ATen/native/cpu/Reduce.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/imag.h>
#endif

#include <c10/util/Optional.h>
#include <c10/util/irange.h>
#include <ATen/AccumulateType.h>

namespace at { namespace native { namespace {

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
    // NOLINTNEXTLINE(bugprone-argument-comment)
    .declare_static_shape(self.sizes(), /*squash_dim=*/dim)
    .add_output(result)
    .add_input(self)
    .build();

  auto result_dim_stride = ensure_nonempty_stride(result, dim);
  auto self_dim_stride = ensure_nonempty_stride(self, dim);

  auto loop = [&](char** data, const int64_t* strides, int64_t n) {
    auto* result_data_bytes = data[0];
    const auto* self_data_bytes = data[1];

    for (const auto i C10_UNUSED : c10::irange(n)) {
      f(
        (scalar_t*)result_data_bytes, result_dim_stride,
        (scalar_t*)self_data_bytes, self_dim_stride, init_val
      );
      result_data_bytes += strides[0];
      self_data_bytes += strides[1];
    }
  };

  iter.for_each(loop);
}

static void cumsum_cpu_kernel(const Tensor& result, const Tensor& self, int64_t dim) {
  auto wrap_dim = maybe_wrap_dim(dim, self.dim());
  int64_t self_dim_size = ensure_nonempty_size(self, wrap_dim);

  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND(kBFloat16, self.scalar_type(), "cumsum_out_cpu", [&] {
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

  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND(kBFloat16, self.scalar_type(), "cumprod_out_cpu", [&] {
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

  AT_DISPATCH_FLOATING_TYPES_AND(kBFloat16, self.scalar_type(), "logcumsumexp_out_cpu", [&] {
    cpu_cum_base_kernel<scalar_t>(result, self, wrap_dim, [&] (
      scalar_t* result_data, auto result_dim_stride,
      const scalar_t* self_data, auto self_dim_stride, scalar_t init_val) {
        using accscalar_t = at::acc_type<scalar_t, false>;
        auto cum_number = (accscalar_t)init_val;
        for (const auto i : c10::irange(self_dim_size)) {
          accscalar_t x = self_data[i * self_dim_stride];

          // Reference : https://www.tensorflow.org/api_docs/python/tf/math/cumulative_logsumexp
          auto log_add_exp = [](accscalar_t x, accscalar_t y) -> accscalar_t {
            accscalar_t min = std::isnan(y) ? y : std::min(x,y); //std::min returns first arg if one of the args is nan
            accscalar_t max = std::isnan(y) ? y : std::max(x,y); //std::max returns first arg if one of the args is nan
            if (min != max || std::isfinite(min)) {
              // nan will be propagated here
              return std::log1p(std::exp(min - max)) + max;
            } else {
           // special case to correctly handle infinite cases
              return x;
            }
          };
          cum_number = log_add_exp(x, cum_number);
          result_data[i * result_dim_stride] = static_cast<scalar_t>(cum_number);
        }
      }, /*init_val=*/ -std::numeric_limits<scalar_t>::infinity()
    );
  });
}

static void mean_kernel_impl(TensorIterator& iter) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX(iter.dtype(), "mean_cpu", [&] {
    scalar_t factor = scalar_t(iter.num_output_elements()) / scalar_t(iter.numel());
    binary_kernel_reduce(
      iter,
      MeanOps<scalar_t, scalar_t> {factor},
      scalar_t(0)
    );
  });
}

static void std_var_kernel_impl(TensorIterator& iter, int64_t correction, bool take_sqrt) {
  AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16, iter.dtype(), "std_cpu", [&] {
    binary_kernel_reduce(
        iter,
        WelfordOps<
            scalar_t,
            double,
            int64_t,
            double,
            std::tuple<scalar_t, scalar_t>>{correction, take_sqrt},
        WelfordData<double, int64_t, double>());
  });
}

static void prod_kernel_impl(TensorIterator& iter) {
  // Workaround for the error: '*' in boolean context, suggest '&&' instead
  // [-Werror=int-in-bool-context]
  if (iter.dtype() == ScalarType::Bool) {
    using scalar_t = bool;
    binary_kernel_reduce_vec(
        iter,
        [=](scalar_t a, scalar_t b)
            __ubsan_ignore_undefined__ -> scalar_t { return a && b; },
        [=](Vectorized<scalar_t> a, Vectorized<scalar_t> b)
            __ubsan_ignore_undefined__ { return a && b; },
        // NOLINTNEXTLINE(bugprone-argument-comment)
        /*identity=*/1);
  } else {
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX(iter.dtype(), "prod_cpu", [&] {
      binary_kernel_reduce_vec(
          iter,
          [=](scalar_t a, scalar_t b)
              __ubsan_ignore_undefined__ -> scalar_t { return a * b; },
          [=](Vectorized<scalar_t> a, Vectorized<scalar_t> b)
              __ubsan_ignore_undefined__ { return a * b; },
          // NOLINTNEXTLINE(bugprone-argument-comment)
          /*identity=*/1);
    });
  }
}

template <typename scalar_t, typename acc_t>
inline void norm_two_reduce_step(Vectorized<acc_t>& acc_vec, Vectorized<scalar_t>& data_vec) {
  acc_vec += data_vec * data_vec;
}

template <>
inline void norm_two_reduce_step(Vectorized<float>& acc_fvec, Vectorized<BFloat16>& data_bvec) {
  Vectorized<float> data_fvec0, data_fvec1;
  std::tie(data_fvec0, data_fvec1) = convert_bfloat16_float(data_bvec);
  acc_fvec += data_fvec0 * data_fvec0;
  acc_fvec += data_fvec1 * data_fvec1;
}

static void norm_kernel_tensor_iterator_impl(
    TensorIterator& iter,
    const Scalar& p) {
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  float val;
  if (p.isIntegral(false)) {
    val = p.to<int64_t>();
  } else if (p.isFloatingPoint()) {
    // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
    val = p.to<double>();
  } else {
    AT_ERROR("norm_kernel_tensor_iterator_impl expects norm to be integer or float");
  }
  if (iter.numel() == 0) {
    iter.output().fill_((val < 0) ? INFINITY : 0);
    return;
  }

  bool use_fast_path = is_reduce_lastdim(iter) && iter.dtype(0) == iter.input_dtype()
      && (iter.input_dtype() == kFloat || iter.input_dtype() == kBFloat16);

  // In the dispatch code blocks below, reduction kernels accumulate results as
  // the type `acc_t`. When `scalar_t` is complex, `acc_t` is the downgraded
  // real number type. Otherwise, `acc_t` and `scalar_t` are the same type.
  if (val == 0) {
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(kHalf, kBFloat16, iter.input_dtype(), "norm_cpu", [&] {
      using acc_t = typename scalar_value_type<scalar_t>::type;
      binary_kernel_reduce(
        iter,
        NormZeroOps<scalar_t, acc_t>(),
        acc_t(0)
      );
    });
  } else if (val == 1) {
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(kHalf, kBFloat16, iter.input_dtype(), "norm_cpu", [&] {
      using acc_t = typename scalar_value_type<scalar_t>::type;
      binary_kernel_reduce(
        iter,
        NormOneOps<scalar_t, acc_t>(),
        acc_t(0)
      );
    });
  } else if (val == 2) {
    if (use_fast_path) {
      AT_DISPATCH_FLOATING_TYPES_AND(kBFloat16, iter.input_dtype(), "norm_cpu", [&] {
        // use float as accumulate type for BFloat16
        using acc_t = vec_scalar_t<scalar_t>;
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
      return;
    }
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(kHalf, kBFloat16, iter.input_dtype(), "norm_cpu", [&] {
      using acc_t = typename scalar_value_type<scalar_t>::type;
      binary_kernel_reduce(
        iter,
        NormTwoOps<scalar_t, acc_t>(),
        acc_t(0)
      );
    });
  } else if (val == INFINITY) {
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(kHalf, kBFloat16, iter.input_dtype(), "norm_cpu", [&] {
      using acc_t = typename scalar_value_type<scalar_t>::type;
      binary_kernel_reduce(
        iter,
        AbsMaxOps<scalar_t, acc_t>(),
        acc_t(0)
      );
    });
  } else if (val == -INFINITY) {
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(kHalf, kBFloat16, iter.input_dtype(), "norm_cpu", [&] {
      using acc_t = typename scalar_value_type<scalar_t>::type;
      binary_kernel_reduce(
        iter,
        AbsMinOps<scalar_t, acc_t>(),
        std::numeric_limits<acc_t>::infinity()
      );
    });
  } else {
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(kHalf, kBFloat16, iter.input_dtype(), "norm_cpu", [&] {
      using acc_t = typename scalar_value_type<scalar_t>::type;
      binary_kernel_reduce(
        iter,
        NormOps<scalar_t, acc_t> { acc_t(val) },
        acc_t(0)
      );
    });
  }

  // For complex outputs, the above kernels do not touch the imaginary values,
  // so we must zero them out
  if (isComplexType(iter.output().scalar_type())) {
    at::imag(iter.output()).zero_();
  }
}

static void and_kernel_impl(TensorIterator& iter) {
  if (iter.dtype() == ScalarType::Byte) {
    // Refer [all, any : uint8 compatibility]
    binary_kernel_reduce_vec(
        iter,
        [=](uint8_t a, uint8_t b) -> uint8_t { return (a && b) ? 1 : 0; },
#if defined(CPU_CAPABILITY_ZVECTOR)
        [=](Vectorized<uint8_t> a, Vectorized<uint8_t> b) {
          return a & b;
        },
#else
        [=](Vectorized<uint8_t> a, Vectorized<uint8_t> b) {
          Vectorized<uint8_t> c = Vectorized<uint8_t>();

          for (decltype(c.size()) i = 0; i != Vectorized<uint8_t>::size(); i++) {
            c[i] = (a[i] && b[i]) ? 1 : 0;
          }
          return c;
        },
#endif
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
#if defined(CPU_CAPABILITY_ZVECTOR)
        [=](Vectorized<uint8_t> a, Vectorized<uint8_t> b) {
          return a | b;
        },
#else
        [=](Vectorized<uint8_t> a, Vectorized<uint8_t> b) {
          Vectorized<uint8_t> c = Vectorized<uint8_t>();

          for (decltype(c.size()) i = 0; i != Vectorized<uint8_t>::size(); i++) {
            c[i] = (a[i] || b[i]) ? 1 : 0;
          }
          return c;
        },
#endif
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

}  // anonymous namespace

REGISTER_DISPATCH(std_var_stub, &std_var_kernel_impl);
REGISTER_DISPATCH(prod_stub, &prod_kernel_impl);
REGISTER_DISPATCH(mean_stub, &mean_kernel_impl);
REGISTER_DISPATCH(norm_stub, &norm_kernel_tensor_iterator_impl);
REGISTER_DISPATCH(and_stub, &and_kernel_impl);
REGISTER_DISPATCH(or_stub, &or_kernel_impl);
REGISTER_DISPATCH(min_values_stub, &min_values_kernel_impl);
REGISTER_DISPATCH(max_values_stub, &max_values_kernel_impl);
REGISTER_DISPATCH(argmax_stub, &argmax_kernel_impl);
REGISTER_DISPATCH(argmin_stub, &argmin_kernel_impl);
REGISTER_DISPATCH(cumprod_stub, &cumprod_cpu_kernel);
REGISTER_DISPATCH(cumsum_stub, &cumsum_cpu_kernel);
REGISTER_DISPATCH(logcumsumexp_stub, &logcumsumexp_cpu_kernel);

}}  // namespace at::native
