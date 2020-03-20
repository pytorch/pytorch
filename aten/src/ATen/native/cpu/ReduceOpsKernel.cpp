#include <numeric>
#include <iterator>
#include <algorithm>

#include <ATen/Dispatch.h>
#include <ATen/cpu/vec256/vec256.h>
#include <ATen/native/ReduceOps.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/SharedReduceOps.h>
#include <ATen/native/cpu/Reduce.h>
#include <ATen/native/ReduceOpsUtils.h>

#include <c10/util/Optional.h>
#include <ATen/AccumulateType.h>

namespace at { namespace native { namespace {

using namespace vec256;

template <typename scalar_t, typename func_t>
static inline void cpu_cum_base_kernel(Tensor& result,
    const Tensor& self,
    int64_t dim,
    const func_t& f,
    scalar_t init_val) {
  if (result.sizes() != self.sizes()) {
    result.resize_as_(self);
  }
  if (self.numel() == 0) {
    return;
  }
  const auto input_ndim = self.dim();
  if (input_ndim == 0) {
    result.fill_(self);
    return;
  }

  auto self_sizes = ensure_nonempty_vec(self.sizes().vec());
  self_sizes[dim] = 1;

  auto result_restrided = restride_dim(result, dim, self_sizes);
  auto self_restrided = restride_dim(self, dim, self_sizes);

  auto iter = TensorIterator();
  iter.dont_compute_common_dtype();
  iter.dont_resize_outputs();
  iter.add_output(result_restrided);
  iter.add_input(self_restrided);
  iter.build();

  auto result_dim_stride = ensure_nonempty_stride(result, dim);
  auto self_dim_stride = ensure_nonempty_stride(self, dim);

  auto loop = [&](char** data, const int64_t* strides, int64_t n) {
    auto* result_data_bytes = data[0];
    const auto* self_data_bytes = data[1];

    for (int64_t i = 0; i < n; ++i) {
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

static void cumsum_cpu_kernel(Tensor& result, const Tensor& self, int64_t dim) {
  auto wrap_dim = maybe_wrap_dim(dim, self.dim());
  int64_t self_dim_size = ensure_nonempty_size(self, wrap_dim);

  AT_DISPATCH_ALL_TYPES(self.scalar_type(), "cumsum_out_cpu", [&] {
    cpu_cum_base_kernel<scalar_t>(result, self, wrap_dim, [&] (
      scalar_t* result_data, auto result_dim_stride,
      const scalar_t* self_data, auto self_dim_stride, scalar_t init_val) {
        auto cum_number = (at::acc_type<scalar_t, false>)init_val;
        for (int64_t i = 0; i < self_dim_size; ++i) {
          cum_number += self_data[i * self_dim_stride];
          result_data[i * result_dim_stride] = (scalar_t)cum_number;
        }
      }, /*init_val=*/ 0
    );
  });
}

static void cumprod_cpu_kernel(Tensor& result, const Tensor& self, int64_t dim) {
  auto wrap_dim = maybe_wrap_dim(dim, self.dim());
  int64_t self_dim_size = ensure_nonempty_size(self, wrap_dim);

  AT_DISPATCH_ALL_TYPES(self.scalar_type(), "cumprod_out_cpu", [&] {
    cpu_cum_base_kernel<scalar_t>(result, self, wrap_dim, [&] (
      scalar_t* result_data, auto result_dim_stride,
      const scalar_t* self_data, auto self_dim_stride, scalar_t init_val) {
        auto cum_number = (at::acc_type<scalar_t, false>)init_val;
        for (int64_t i = 0; i < self_dim_size; ++i) {
          cum_number *= self_data[i * self_dim_stride];
          result_data[i * result_dim_stride] = (scalar_t)cum_number;
        }
      }, /*init_val=*/ 1
    );
  });
}

static void sum_kernel_impl(TensorIterator& iter) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(
      ScalarType::BFloat16, ScalarType::Bool, iter.dtype(), "sum_cpu", [&] {
        binary_kernel_reduce_vec(
            iter, [=](scalar_t a, scalar_t b) -> scalar_t { return a + b; },
            [=](Vec256<scalar_t> a, Vec256<scalar_t> b) { return a + b; });
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

static void std_var_kernel_impl(TensorIterator &iter, bool unbiased, bool take_sqrt) {
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(iter.dtype(), "std_cpu", [&] {
    binary_kernel_reduce(
      iter,
      WelfordOps<scalar_t, double, int64_t, double, std::tuple<scalar_t, scalar_t>> { unbiased, take_sqrt },
      WelfordData<double, int64_t, double>()
    );
  });
}

static void prod_kernel_impl(TensorIterator& iter) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX(iter.dtype(), "prod_cpu", [&] {
    binary_kernel_reduce_vec(
      iter,
      [=](scalar_t a, scalar_t b) -> scalar_t { return a * b; },
      [=](Vec256<scalar_t> a, Vec256<scalar_t> b) { return a * b; },
      /*identity=*/1);
  });
}

static void norm_kernel_tensor_iterator_impl(
    TensorIterator& iter,
    Scalar p) {
  float val;
  if (p.isIntegral(false)) {
    val = p.to<int64_t>();
  } else if (p.isFloatingPoint()) {
    val = p.to<float>();
  } else {
    AT_ERROR("norm_kernel_tensor_iterator_impl expects norm to be integer or float");
  }


  if (val == 0) {
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(iter.dtype(), "norm_cpu", [&] {
      binary_kernel_reduce(
        iter,
        NormZeroOps<scalar_t>(),
        scalar_t(0)
      );
    });
  } else if (val == 1) {
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(iter.dtype(), "norm_cpu", [&] {
      binary_kernel_reduce(
        iter,
        NormOneOps<scalar_t>(),
        scalar_t(0)
      );
    });
  } else if (val == 2) {
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(iter.dtype(), "norm_cpu", [&] {
      binary_kernel_reduce(
        iter,
        NormTwoOps<scalar_t>(),
        scalar_t(0)
      );
    });
  } else if (val == INFINITY) {
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(iter.dtype(), "norm_cpu", [&] {
      binary_kernel_reduce(
        iter,
        AbsMaxOps<scalar_t>(),
        scalar_t(std::numeric_limits<scalar_t>::min())
      );
    });
  } else if (val == -INFINITY) {
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(iter.dtype(), "norm_cpu", [&] {
      binary_kernel_reduce(
        iter,
        AbsMinOps<scalar_t>(),
        scalar_t(std::numeric_limits<scalar_t>::max())
      );
    });
  } else {
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(iter.dtype(), "norm_cpu", [&] {
      binary_kernel_reduce(
        iter,
        NormOps<scalar_t> { scalar_t(val) },
        scalar_t(0)
      );
    });
  }
}

static void and_kernel_impl(TensorIterator& iter) {
  binary_kernel_reduce_vec(
    iter,
    [=](uint8_t a, uint8_t b) -> uint8_t { return a && b; },
    [=](Vec256<uint8_t> a, Vec256<uint8_t> b) {
      // Adding the implementation here instead of in vec256_base to avoid
      // return value inconsistency. Other comparison operators in vec256_base
      // return -1/0 (all bit 1 / all bit 0) as true/false to follow the AVX2
      // convention. This would be convenient when combined with other
      // vectorized operations. For example, one can use the logical operation
      // results as a mask for a bit operation to retrieve/reset multiple
      // elements in a vector.
      //
      // In this method, users would expect, e.g., all(), to return 1/0 as
      // true/false.
      Vec256<uint8_t> c = Vec256<uint8_t>();
      for (int i = 0; i != Vec256<uint8_t>::size(); i++) {
        c[i] = a[i] && b[i];
      }
      return c;
    },
    /*ident=*/true);
}

static void or_kernel_impl(TensorIterator& iter) {
  binary_kernel_reduce_vec(
    iter,
    [=](uint8_t a, uint8_t b) -> uint8_t { return a || b; },
    [=](Vec256<uint8_t> a, Vec256<uint8_t> b) {
      Vec256<uint8_t> c = Vec256<uint8_t>();
      for (int i = 0; i != Vec256<uint8_t>::size(); i++) {
        c[i] = a[i] || b[i];
      }
      return c;
    },
    /*ident=*/false);
}

static void min_values_kernel_impl(TensorIterator& iter) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX(iter.dtype(), "min_values_cpu", [&iter] {
    binary_kernel_reduce_vec(
      iter,
      [](scalar_t a, scalar_t b) -> scalar_t { return min_impl(a, b); },
      [](Vec256<scalar_t> a, Vec256<scalar_t> b) { return minimum(a, b); });
  });
}

static void max_values_kernel_impl(TensorIterator& iter) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX(iter.dtype(), "max_values_cpu", [&iter] {
    binary_kernel_reduce_vec(
      iter,
      [](scalar_t a, scalar_t b) -> scalar_t { return max_impl(a, b); },
      [](Vec256<scalar_t> a, Vec256<scalar_t> b) { return maximum(a, b); });
  });
}

static void argmax_kernel_impl(TensorIterator &iter) {
  AT_DISPATCH_ALL_TYPES(iter.dtype(1), "argmax_cpu", [&] {
    binary_kernel_reduce(
      iter,
      ArgMaxOps<scalar_t>{},
      std::pair<scalar_t, int64_t>(lower_bound<scalar_t>(), -1));
  });
}

static void argmin_kernel_impl(TensorIterator &iter) {
  AT_DISPATCH_ALL_TYPES(iter.dtype(1), "argmin_cpu", [&] {
    binary_kernel_reduce(
      iter,
      ArgMinOps<scalar_t>{},
      std::pair<scalar_t, int64_t>(upper_bound<scalar_t>(), -1));
  });
}

}  // anonymous namespace

REGISTER_DISPATCH(sum_stub, &sum_kernel_impl);
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

}}  // namespace at::native
