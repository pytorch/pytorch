#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/Dispatch.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/LinearAlgebra.h>
#include <ATen/native/BatchLinearAlgebra.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/SharedReduceOps.h>
#include <ATen/native/ReduceOps.h>
#include <c10/core/Scalar.h>

#include <thrust/swap.h>

namespace at::native {

namespace {

void addr_kernel_cuda(TensorIterator &iter, const Scalar& beta, const Scalar& alpha) {
  if (iter.dtype() == ScalarType::Bool) {
    using scalar_t = bool;
    auto beta_val = beta.to<scalar_t>();
    auto alpha_val = alpha.to<scalar_t>();

    // when beta is false, values in self should be ignored,
    // nans and infs in self should not propagate.
    if (beta_val == false) {
      gpu_kernel(
        iter,
        [=] GPU_LAMBDA (scalar_t self_val,
                        scalar_t vec1_val, scalar_t vec2_val) -> scalar_t {
          return alpha_val && vec1_val && vec2_val;
        }
      );
    } else {
      gpu_kernel(
        iter,
        [=] GPU_LAMBDA (scalar_t self_val,
                        scalar_t vec1_val, scalar_t vec2_val) -> scalar_t {
          return (beta_val && self_val) || (alpha_val && vec1_val && vec2_val);
        }
      );
    }
    return;
  }

  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(kBFloat16, kHalf,
                                         iter.dtype(), "addr_cuda", [&] {
    auto beta_val = beta.to<scalar_t>();
    auto alpha_val = alpha.to<scalar_t>();

    scalar_t zero_val(0);
    // when beta==0, values in self should be ignored,
    // nans and infs in self should not propagate.
    if (beta_val == zero_val) {
      gpu_kernel(
        iter,
        [=] GPU_LAMBDA (scalar_t self_val,
                        scalar_t vec1_val, scalar_t vec2_val) -> scalar_t {
          return alpha_val * vec1_val * vec2_val;
        }
      );
    } else {
      gpu_kernel(
        iter,
        [=] GPU_LAMBDA (scalar_t self_val,
                        scalar_t vec1_val, scalar_t vec2_val) -> scalar_t {
          return beta_val * self_val + alpha_val * vec1_val * vec2_val;
        }
      );
    }
  });
}


template <int n_threads, int n_elems_per_thread, typename func_t>
C10_LAUNCH_BOUNDS_2(n_threads, n_elems_per_thread)
__global__ void _elementwise_kernel(int total_n_elems, func_t f) {
  constexpr int total_work_block = n_threads * n_elems_per_thread;
  int idx = total_work_block * blockIdx.x + threadIdx.x;

  #pragma unroll
  for (int i = 0; i < n_elems_per_thread; ++i) {
    if (idx < total_n_elems) {
      f(idx);
      idx += n_threads;
    }
  }
}

template <int n_threads, int n_elems_per_thread, typename func_t>
static void _launch_kernel(int total_n_elems, func_t f) {
  TORCH_INTERNAL_ASSERT(
    total_n_elems >= 0 && total_n_elems <= std::numeric_limits<int32_t>::max()
  );

  dim3 block(n_threads);
  constexpr int total_work_block = n_threads * n_elems_per_thread;
  dim3 grid((total_n_elems + total_work_block - 1) / total_work_block);

  auto stream = at::cuda::getCurrentCUDAStream();
  _elementwise_kernel<n_threads, n_elems_per_thread, func_t>
    <<<grid, block, 0, stream>>>(total_n_elems, f);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void unpack_pivots_cuda_kernel(TensorIterator& iter, const int64_t dim_size, const int64_t max_pivot) {
  if (iter.numel() == 0) {
    return;
  }

  if (!iter.can_use_32bit_indexing()) {
    for (auto& sub_iter : iter.with_32bit_indexing()) {
      unpack_pivots_cuda_kernel(sub_iter, dim_size, max_pivot);
    }
    return;
  }

  const auto offset_calculator = make_offset_calculator<2>(iter);

  const auto perm_ptr = reinterpret_cast<char*>(iter.data_ptr(0));
  const auto pivots_ptr = reinterpret_cast<const char*>(iter.data_ptr(1));

  auto loop = [=]C10_DEVICE(const int idx) {
    const auto offsets = offset_calculator.get(idx);

    int64_t* const __restrict__ perm_data = reinterpret_cast<int64_t*>(perm_ptr + offsets[0]);
    const int32_t* const __restrict__ pivots_data = reinterpret_cast<const int32_t*>(pivots_ptr + offsets[1]);

    // QUESTION: can we mix 64bit offsets with 32bit Iterator indexing?
    for (int64_t i = 0; i < dim_size; ++i) {
      thrust::swap(
        perm_data[i],
        perm_data[pivots_data[i] - 1]
      );
    }
  };

  _launch_kernel<num_threads(), thread_work_size()>(iter.numel(), loop);
}
} // anonymous namespace

REGISTER_DISPATCH(unpack_pivots_stub, &unpack_pivots_cuda_kernel);
REGISTER_DISPATCH(addr_stub, &addr_kernel_cuda);
} // namespace at::native
