#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/LinearAlgebra.h>
#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/SharedReduceOps.h>
#include <ATen/native/cpu/Reduce.h>
#include <ATen/native/cpu/Loops.h>
#include <c10/util/irange.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/imag.h>
#endif

namespace at { namespace native { namespace {

void addr_kernel(TensorIterator &iter,
                 const Scalar& beta, const Scalar& alpha) {
  if (iter.dtype() == ScalarType::Bool) {
    using scalar_t = bool;
    auto beta_val = beta.to<scalar_t>();
    auto alpha_val = alpha.to<scalar_t>();

    // when beta is false, values in self should be ignored,
    // nans and infs in self should not propagate.
    if (beta_val == false) {
      cpu_kernel(iter,
        [=](scalar_t self_val,
            scalar_t vec1_val,
            scalar_t vec2_val) __ubsan_ignore_undefined__ -> scalar_t {
          return alpha_val && vec1_val && vec2_val;
        }
      );
    } else {
      cpu_kernel(iter,
        [=](scalar_t self_val,
            scalar_t vec1_val,
            scalar_t vec2_val) __ubsan_ignore_undefined__ -> scalar_t {
          return (beta_val && self_val) || (alpha_val && vec1_val && vec2_val);
        }
      );
    }
    return;
  }

  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(kBFloat16, kHalf,
    iter.dtype(), "addr_cpu", [&]() {
      using Vec = Vectorized<scalar_t>;

      auto beta_val = beta.to<scalar_t>();
      auto alpha_val = alpha.to<scalar_t>();

      auto beta_vec = Vec(beta_val);
      auto alpha_vec = Vec(alpha_val);

      const scalar_t zero_val(0);
      // when beta == 0, values in self should be ignored,
      // nans and infs in self should not propagate.
      if (beta_val == zero_val) {
        cpu_kernel_vec(iter,
          [=](scalar_t self_val,
              scalar_t vec1_val,
              scalar_t vec2_val) __ubsan_ignore_undefined__ -> scalar_t {
            return alpha_val * vec1_val * vec2_val;
          },
          [=](Vec self_vec,
              Vec vec1_vec,
              Vec vec2_vec) __ubsan_ignore_undefined__ {
            return alpha_vec * vec1_vec * vec2_vec;
          }
        );
      } else {
        cpu_kernel_vec(iter,
          [=](scalar_t self_val,
              scalar_t vec1_val,
              scalar_t vec2_val) __ubsan_ignore_undefined__ -> scalar_t {
            return beta_val * self_val + alpha_val * vec1_val * vec2_val;
          },
          [=](Vec self_vec,
              Vec vec1_vec,
              Vec vec2_vec) __ubsan_ignore_undefined__ {
            return beta_vec * self_vec + alpha_vec * vec1_vec * vec2_vec;
          }
        );
      }
    }
  );
}

template <typename scalar_t, typename acc_t=typename scalar_value_type<scalar_t>::type>
void linalg_vector_norm_kernel_cpu_impl(TensorIterator& iter, Scalar ord) {
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  double ord_val;
  if (ord.isFloatingPoint()) {
     ord_val = ord.to<double>();
  } else {
     TORCH_CHECK(false, "linalg.vector_norm expects ord to be float");
  }
  // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
  acc_t init_val = (ord_val == -INFINITY) ? std::numeric_limits<acc_t>::infinity() : static_cast<acc_t>(0);
  if (iter.numel() == 0) {
    iter.output().fill_((ord_val < 0) ? INFINITY : 0);
    return;
  }
  if (ord_val == 0) {
    binary_kernel_reduce(iter, NormZeroOps<scalar_t, acc_t>(), init_val);
  } else if (ord_val == 1) {
    binary_kernel_reduce(iter, NormOneOps<scalar_t, acc_t>(), init_val);
  } else if (ord_val == 2) {
    binary_kernel_reduce(iter, NormTwoOps<scalar_t, acc_t>(), init_val);
  } else if (ord_val == INFINITY) {
    binary_kernel_reduce(iter, AbsMaxOps<scalar_t, acc_t>(), init_val);
  // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
  } else if (ord_val == -INFINITY) {
    binary_kernel_reduce(iter, AbsMinOps<scalar_t, acc_t>(), init_val);
  } else {
    binary_kernel_reduce(iter, NormOps<scalar_t, acc_t> { static_cast<acc_t>(ord_val) }, init_val);
  }
  // For complex outputs, the above kernels do not touch the imaginary values,
  // so we must zero them out
  if (isComplexType(iter.output().scalar_type())) {
    at::imag(iter.output()).zero_();
  }
}

static void linalg_vector_norm_kernel_cpu(TensorIterator& iter, Scalar ord) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(kHalf, kBFloat16, iter.input_dtype(), "linalg_vector_norm_cpu", [&] {
    linalg_vector_norm_kernel_cpu_impl<scalar_t>(iter, ord);
  });
}

void unpack_pivots_cpu_kernel(
  TensorIterator& iter,
  int64_t dim_size
) {
  if (iter.numel() == 0) {
    return;
  }

  auto loop = [&](char** data, const int64_t* strides, int64_t nelems) {
    auto* unpacked_pivots_ptr = data[0];
    const auto* pivots_ptr = data[1];

    for (const auto elem : c10::irange(nelems)) {
      (void)elem; //Suppress unused variable warning
      // WARNING: torch.lu returns int32 pivots,
      // this behavior could change in the future.
      auto* unpacked_pivots_data = reinterpret_cast<int32_t*>(unpacked_pivots_ptr);
      auto* pivots_data = reinterpret_cast<const int32_t*>(pivots_ptr);

      for (const auto i : c10::irange(dim_size)) {
        std::swap(
          unpacked_pivots_data[i],
          unpacked_pivots_data[pivots_data[i]]
        );
      }

      unpacked_pivots_ptr += strides[0];
      pivots_ptr += strides[1];
    }
  };

  iter.for_each(loop);
}

} // anonymous namespace

REGISTER_DISPATCH(addr_stub, &addr_kernel);
REGISTER_DISPATCH(linalg_vector_norm_stub, &linalg_vector_norm_kernel_cpu);
REGISTER_DISPATCH(unpack_pivots_stub, &unpack_pivots_cpu_kernel);

}} // namespace at::native
