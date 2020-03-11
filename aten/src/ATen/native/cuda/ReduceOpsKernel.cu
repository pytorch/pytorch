#include <ATen/native/SharedReduceOps.h>
#include <ATen/AccumulateType.h>
#include <ATen/Context.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/NumericLimits.cuh>
#include <ATen/native/cuda/DeviceSqrt.cuh>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/cuda/Reduce.cuh>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/ReduceOps.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>

#include <limits>
#include <tuple>
#include <thrust/tuple.h>
#include <thrust/pair.h>

#include <THC/THCTensorMathReduce.cuh>

namespace at { namespace native {

template <typename scalar_t, typename acc_t=scalar_t, typename out_t=scalar_t>
void sum_kernel_impl(TensorIterator& iter) {
  gpu_reduce_kernel<scalar_t, out_t>(iter, func_wrapper<out_t> ([]GPU_LAMBDA(acc_t a, acc_t b) -> acc_t {
    return a + b;
  }));
}

template <typename scalar_t>
void std_var_kernel_impl(TensorIterator& iter, bool unbiased, bool take_sqrt) {
  // reducing unrolling factor to 2 for welford kernel
  // This is necessary to lower register usage that leads to register spills.
  gpu_reduce_kernel<scalar_t, scalar_t, 2>(iter, WelfordOps<scalar_t, scalar_t, int32_t, float, thrust::tuple<scalar_t, scalar_t>> { unbiased, take_sqrt }, WelfordData<scalar_t, int32_t, float> {});
}

template <>
void std_var_kernel_impl<at::Half>(TensorIterator& iter, bool unbiased, bool take_sqrt) {
  // reducing unrolling factor to 2 for welford kernel
  // This is necessary to lower register usage that leads to register spills.
  gpu_reduce_kernel<at::Half, at::Half, 2>(iter, WelfordOps<at::Half, float, int32_t, float, thrust::tuple<at::Half, at::Half>> { unbiased, take_sqrt }, WelfordData<float, int32_t, float> {});
}

template <typename scalar_t, typename acc_t=scalar_t, typename out_t=scalar_t>
void prod_kernel_impl(TensorIterator& iter) {
  gpu_reduce_kernel<scalar_t, out_t>(iter, func_wrapper<out_t> ([]GPU_LAMBDA(acc_t a, acc_t b) -> acc_t {
    return a * b;
  }), 1);
}

static void std_var_kernel_cuda(TensorIterator& iter, bool unbiased, bool take_sqrt) {
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(iter.dtype(), "std", [&]() {
    std_var_kernel_impl<scalar_t>(iter, unbiased, take_sqrt);
  });
}

template <typename scalar_t, typename acc_t=scalar_t, typename out_t=scalar_t>
void mean_kernel_impl(TensorIterator& iter) {
  float factor = float(iter.num_output_elements()) / iter.numel();
  gpu_reduce_kernel<scalar_t, out_t>(iter, MeanOps<acc_t, float> {factor});
}

template <typename scalar_t, typename acc_t=scalar_t, typename out_t=scalar_t>
void norm_kernel_cuda_impl(TensorIterator& iter, Scalar val) {
  float p;
  if (val.isIntegral(false)) {
     p = val.to<int64_t>();
  } else if (val.isFloatingPoint()) {
     p = val.to<acc_t>();
  } else {
     AT_ERROR("norm_kernel_cuda_impl expects norm to be integer or float");
  }

  if (p == static_cast<float>(0)) {
    gpu_reduce_kernel<scalar_t, out_t>(iter, NormZeroOps<acc_t>(), 0);
  } else if (p == static_cast<float>(1)) {
    gpu_reduce_kernel<scalar_t, out_t>(iter, NormOneOps<acc_t>(), 0);
  } else if (p == static_cast<float>(2)) {
    gpu_reduce_kernel<scalar_t, out_t>(iter, NormTwoOps<acc_t>(), 0);
  } else if (p == static_cast<float>(INFINITY)) {
    gpu_reduce_kernel<scalar_t, out_t>(iter, AbsMaxOps<acc_t>(), std::numeric_limits<acc_t>::min());
  } else if (p == static_cast<float>(-INFINITY)) {
    gpu_reduce_kernel<scalar_t, out_t>(iter, AbsMinOps<acc_t>(), std::numeric_limits<acc_t>::max());
  } else {
    gpu_reduce_kernel<scalar_t, out_t>(iter, NormOps<acc_t>{ acc_t(p) }, 0);
  }
}

static void sum_kernel_cuda(TensorIterator& iter) {
  if (iter.dtype() == kHalf) {
    return sum_kernel_impl<at::Half, float>(iter);
  } else if (iter.dtype(1) == kHalf && iter.dtype() == kFloat) {
    // type promotion that does cast and reduction in a single kernel
    return sum_kernel_impl<at::Half, float, float>(iter);
  }
  #ifdef __HIP_PLATFORM_HCC__
  else if (iter.dtype() == kBFloat16) {
    return sum_kernel_impl<at::BFloat16, float>(iter);
  } else if (iter.dtype(1) == kBFloat16 && iter.dtype() == kFloat) {
    // type promotion that does cast and reduction in a single kernel
    return sum_kernel_impl<at::BFloat16, float, float>(iter);
  }
  #endif
  AT_DISPATCH_ALL_TYPES_AND(ScalarType::Bool, iter.dtype(), "sum_cuda", [&]() {
    sum_kernel_impl<scalar_t>(iter);
  });
}

static void prod_kernel_cuda(TensorIterator& iter) {
  if (iter.dtype() == kHalf) {
    return prod_kernel_impl<at::Half, float>(iter);
  } else if (iter.dtype(1) == kHalf && iter.dtype() == kFloat) {
    // type promotion that does cast and reduction in a single kernel
    return prod_kernel_impl<at::Half, float, float>(iter);
  }
  #ifdef __HIP_PLATFORM_HCC__
  else if (iter.dtype() == kBFloat16) {
    return prod_kernel_impl<at::BFloat16, float>(iter);
  } else if (iter.dtype(1) == kBFloat16 && iter.dtype() == kFloat) {
    // type promotion that does cast and reduction in a single kernel
    return prod_kernel_impl<at::BFloat16, float, float>(iter);
  }
  #endif
  AT_DISPATCH_ALL_TYPES(iter.dtype(), "prod_cuda", [&]() {
    prod_kernel_impl<scalar_t>(iter);
  });
}

static void mean_kernel_cuda(TensorIterator& iter) {
  if (iter.dtype() == kHalf) {
    return mean_kernel_impl<at::Half, float>(iter);
  } else if (iter.dtype(1) == kHalf && iter.dtype() == kFloat) {
    // type promotion that does cast and reduction in a single kernel
    return mean_kernel_impl<at::Half, float, float>(iter);
  }
  #ifdef __HIP_PLATFORM_HCC__
  else if(iter.dtype() == kBFloat16) {
    return mean_kernel_impl<at::BFloat16, float>(iter);
  } else if (iter.dtype(1) == kBFloat16 && iter.dtype() == kFloat) {
    // type promotion that does cast and reduction in a single kernel
    return mean_kernel_impl<at::BFloat16, float, float>(iter);
  }
  #endif
  AT_DISPATCH_ALL_TYPES(iter.dtype(), "mean_cuda", [&]() {
    mean_kernel_impl<scalar_t>(iter);
  });
}

static void norm_kernel_cuda(TensorIterator& iter, Scalar p) {
  if (iter.dtype() == kHalf) {
    return norm_kernel_cuda_impl<at::Half, float>(iter, p);
  } else if (iter.dtype(1) == kHalf && iter.dtype() == kFloat) {
    // type promotion that does cast and reduction in a single kernel
    return norm_kernel_cuda_impl<at::Half, float, float>(iter, p);
  }
  #ifdef __HIP_PLATFORM_HCC__
  else if(iter.dtype() == kBFloat16) {
    return norm_kernel_cuda_impl<at::BFloat16, float>(iter, p);
  } else if (iter.dtype(1) == kBFloat16 && iter.dtype() == kFloat) {
    // type promotion that does cast and reduction in a single kernel
    return norm_kernel_cuda_impl<at::BFloat16, float, float>(iter, p);
  }
  #endif
  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "norm_cuda", [&]() {
    norm_kernel_cuda_impl<scalar_t>(iter, p);
  });
}

void and_kernel_cuda(TensorIterator& iter) {
  gpu_reduce_kernel<uint8_t, uint8_t>(
    iter, func_wrapper<uint8_t> ([]GPU_LAMBDA(uint8_t a, uint8_t b) -> uint8_t {
      return a && b;
    }), true);
}

void or_kernel_cuda(TensorIterator& iter) {
  gpu_reduce_kernel<uint8_t, uint8_t>(
    iter, func_wrapper<uint8_t> ([]GPU_LAMBDA(uint8_t a, uint8_t b) -> uint8_t {
      return a || b;
    }), false);
}

template <typename scalar_t, typename acc_t=scalar_t>
void max_values_kernel_cuda_impl(TensorIterator& iter) {
  gpu_reduce_kernel<scalar_t, scalar_t>(
    iter, func_wrapper<acc_t> ([]GPU_LAMBDA(acc_t a, acc_t b) -> acc_t {
      return (THCNumerics<acc_t>::isnan(a) || a > b) ? a : b;
    }), at::numeric_limits<acc_t>::lower_bound());
}

template <typename scalar_t, typename acc_t=scalar_t>
void min_values_kernel_cuda_impl(TensorIterator& iter) {
  gpu_reduce_kernel<scalar_t, scalar_t>(
    iter, func_wrapper<acc_t> ([]GPU_LAMBDA(acc_t a, acc_t b) -> acc_t {
      return (THCNumerics<acc_t>::isnan(a) || a < b) ? a : b;
    }), at::numeric_limits<acc_t>::upper_bound());
}

void max_values_kernel_cuda(TensorIterator& iter) {
  if (iter.dtype(1) == kHalf) {
    max_values_kernel_cuda_impl<at::Half, float>(iter);
  } else {
    AT_DISPATCH_ALL_TYPES(iter.dtype(), "max_values_cuda", [&]() {
      max_values_kernel_cuda_impl<scalar_t>(iter);
    });
  }
}

void min_values_kernel_cuda(TensorIterator& iter) {
  if (iter.dtype(1) == kHalf) {
    min_values_kernel_cuda_impl<at::Half, float>(iter);
  } else {
    AT_DISPATCH_ALL_TYPES(iter.dtype(), "min_values_cuda", [&]() {
      min_values_kernel_cuda_impl<scalar_t>(iter);
    });
  }
}

template <typename scalar_t, typename acc_t=scalar_t>
void argmax_kernel_cuda_impl(TensorIterator& iter) {
  gpu_reduce_kernel<scalar_t, int64_t>(
    iter,
    ArgMaxOps<acc_t>{},
    thrust::pair<acc_t, int64_t>(at::numeric_limits<acc_t>::lower_bound(), 0));
};

template <typename scalar_t, typename acc_t=scalar_t>
void argmin_kernel_cuda_impl(TensorIterator& iter) {
  gpu_reduce_kernel<scalar_t, int64_t>(
    iter,
    ArgMinOps<acc_t>{},
    thrust::pair<acc_t, int64_t>(at::numeric_limits<acc_t>::upper_bound(), 0));
};

void argmax_kernel_cuda(TensorIterator& iter) {
  if (iter.dtype(1) == kHalf) {
    // Instead of implementing is_nan and warp_shfl_down
    // we can convert halves to float and do all the operations in float
    argmax_kernel_cuda_impl<at::Half, float>(iter);
  } else {
    AT_DISPATCH_ALL_TYPES(iter.dtype(1), "argmax_cuda", [&]() {
      argmax_kernel_cuda_impl<scalar_t>(iter);
    });
  }
}

void argmin_kernel_cuda(TensorIterator& iter) {
  if (iter.dtype(1) == kHalf) {
    // Instead of implementing is_nan and warp_shfl_down
    // we can convert halves to float and do all the operations in float
    argmin_kernel_cuda_impl<at::Half, float>(iter);
  } else {
    AT_DISPATCH_ALL_TYPES(iter.dtype(1), "argmin_cuda", [&]() {
      argmin_kernel_cuda_impl<scalar_t>(iter);
    });
  }
}

template<typename scalar_t, typename idx_t, typename BinaryOperation>
__device__ void binary_op_update(const scalar_t lhs, scalar_t& rhs, const idx_t lhs_idx, idx_t& rhs_idx, BinaryOperation binary_op) {
  if(!THCNumerics<scalar_t>::isnan(rhs) && (THCNumerics<scalar_t>::isnan(lhs) || !binary_op(rhs, lhs))) {
    rhs = lhs;
    rhs_idx = lhs_idx;
  }
}
/* Perform an inclusive scan along the innermost dimension of a tensor.
 *
 * - num_rows is the size of the flattened outer dimensions;
 * - row_size is the size of the innermost dimension;
 *
 * The outer dimensions of the tensor are considered as a single dimension, i.e. the tensor is
 * considered as having 'num_rows' rows of size 'row_size'.
 * Each thread block processes one or more sets of contiguous rows (processing multiple rows
 * per thread block is quicker than processing a single row, especially for short rows).
 */
template<typename scalar_t, int num_threads_x, int num_threads_y, class BinaryFunction>
__global__ void tensor_kernel_scan_innermost_dim_with_indices(const scalar_t *self_, scalar_t *values_, int64_t *indices_,
                                                int num_rows, int row_size,
                                                scalar_t init, BinaryFunction binary_op) {
  __shared__ scalar_t vbuf[num_threads_y][2 * num_threads_x];
  __shared__ int64_t ibuf[num_threads_y][2 * num_threads_x];
  scalar_t* row_buf = vbuf[threadIdx.y];
  int64_t* row_idx_buf = ibuf[threadIdx.y];

  for (int block_row = blockIdx.x * blockDim.y;
       block_row < num_rows;
       block_row += blockDim.y * gridDim.x) {
    int row = block_row + threadIdx.y;
    const scalar_t *row_self = self_ + row * row_size;
    scalar_t *row_values = values_ + row * row_size;
    int64_t *row_indices = indices_ + row * row_size;
    scalar_t block_total = init;
    int64_t block_idx_final = 0;
    // Perform scan on one block at a time, keeping track of the total value of
    // all blocks processed so far.
    for (int block_col = 0; block_col < row_size; block_col += 2 * num_threads_x) {
      // Load data into shared memory (two values per thread).
      int col1 = block_col + threadIdx.x;
      int col2 = block_col + num_threads_x + threadIdx.x;
      if (row < num_rows) {
        if (col1 < row_size) {
          row_buf[threadIdx.x] = row_self[col1];
          row_idx_buf[threadIdx.x] = col1;
        } else {
          row_buf[threadIdx.x] = init;
          // No need to set the index here as the value in init will never be selected
        }

        if (col2 < row_size) {
          row_buf[num_threads_x + threadIdx.x] = row_self[col2];
          row_idx_buf[num_threads_x + threadIdx.x] = col2;
        } else {
          row_buf[num_threads_x + threadIdx.x] = init;
          // No need to set the index here as the value in init will never be selected
        }

        // Add the total value of all previous blocks to the first value of this block.
        if (threadIdx.x == 0) {
          binary_op_update(block_total, row_buf[0], block_idx_final, row_idx_buf[0], binary_op);
        }
      }
      __syncthreads();

      // Parallel reduction (up-sweep).
      for (int s = num_threads_x, d = 1; s >= 1; s >>= 1, d <<= 1) {
        if (row < num_rows && threadIdx.x < s) {
          int offset = (2 * threadIdx.x + 1) * d - 1;
          binary_op_update(row_buf[offset], row_buf[offset + d], row_idx_buf[offset], row_idx_buf[offset + d], binary_op);
        }
        __syncthreads();
      }

      // Down-sweep.
      for (int s = 2, d = num_threads_x / 2; d >= 1; s <<= 1, d >>= 1) {
        if (row < num_rows && threadIdx.x < s - 1) {
          int offset = 2 * (threadIdx.x + 1) * d - 1;
          binary_op_update(row_buf[offset], row_buf[offset + d], row_idx_buf[offset], row_idx_buf[offset + d], binary_op);
        }
        __syncthreads();
      }

      // Write back to output.
      if (row < num_rows) {
        if (col1 < row_size){
          row_values[col1] = row_buf[threadIdx.x];
          row_indices[col1] = row_idx_buf[threadIdx.x];
        }
        if (col2 < row_size) {
          row_values[col2] = row_buf[num_threads_x + threadIdx.x];
          row_indices[col2] = row_idx_buf[num_threads_x + threadIdx.x];
        }
      }
      block_total = row_buf[2 * num_threads_x - 1];
      block_idx_final = row_idx_buf[2 * num_threads_x - 1];
      __syncthreads();
    }
  }
}

/* Perform an inclusive scan along an outer dimension of a tensor.
 *
 * - num_orows is the size of the flattened outer dimensions;
 * - num_irows is the size of the flattened inner dimensions;
 * - row_size is the size of the dimension along which to compute the variance;
 *
 * The dimensions to the outside and inside of the specified dimension are considered as flattened.
 * Thread blocks with the same blockIdx.y process an "outer row" (i.e. an element of the flattened
 * outer dimensions, which contains several "inner rows").
 * Each thread processes a single inner row at a time.
 */
template<typename scalar_t, class BinaryFunction>
__global__ void tensor_kernel_scan_outer_dim_with_indices(scalar_t *self_, scalar_t *values_, int64_t *indices_,
                  int num_orows, int num_irows, int row_size, scalar_t init, BinaryFunction binary_op) {
  for (int orow = blockIdx.x; orow < num_orows; orow += gridDim.x) {
    for (int irow = blockIdx.y * blockDim.x + threadIdx.x; irow < num_irows; irow += gridDim.y * blockDim.x) {
      scalar_t *self = self_ + orow * row_size * num_irows + irow;
      scalar_t *values = values_ + orow * row_size * num_irows + irow;
      int64_t *indices = indices_ + orow * row_size * num_irows + irow;
      scalar_t out = init;
      int64_t out_idx = 0;

      for (int64_t col = 0; col < row_size; ++col) {
        if(THCNumerics<scalar_t>::isnan(*self) || (!THCNumerics<scalar_t>::isnan(out) && binary_op(*self, out))) {
          out = *self;
          out_idx = col;
        }
        *values = out;
        *indices = out_idx;
        self += num_irows;
        values += num_irows;
        indices += num_irows;
      }
    }
  }
}

template<typename scalar_t, class BinaryFunction>
__host__ void scan_outer_dim_with_indices(const Tensor& self, Tensor& values, Tensor& indices,
                                       int dim, scalar_t init, BinaryFunction binary_op) {
  int row_size = self.size(dim);
  auto sizes = self.sizes();

  // Treat all outer dimensions (i.e. dim_ < dim) as one.
  int num_orows = std::accumulate(sizes.begin(), sizes.begin() + dim, 1, std::multiplies<int>());

  // Treat all inner dimensions (i.e. dim > dimension) as one.
  int num_irows = std::accumulate(sizes.begin() + dim + 1, sizes.end(), 1, std::multiplies<int>());

  dim3 threads(std::min(512, int(num_irows)));
  int maxGridDim = at::cuda::getCurrentDeviceProperties()->maxGridSize[0];
  dim3 grid(std::min(maxGridDim, num_orows), std::min(maxGridDim, cuda::ATenCeilDiv(num_irows, int(threads.x))));

  tensor_kernel_scan_outer_dim_with_indices<scalar_t><<<grid, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
    self.data_ptr<scalar_t>(), values.data_ptr<scalar_t>(), indices.data_ptr<int64_t>(),
    num_orows, num_irows, row_size, init, binary_op);
  AT_CUDA_CHECK(cudaGetLastError());
}

template <typename scalar_t, class BinaryFunction>
__host__ void scan_innermost_dim_with_indices(const Tensor& self, Tensor& values, Tensor& indices, scalar_t init, BinaryFunction binary_op) {
  int ndim = self.dim();
  // Treat all outer dimensions as a single dimension.
  int row_size = self.size(ndim - 1);
  int num_rows = self.numel() / row_size;

  dim3 threads(16, 32);
  dim3 grid(std::min(at::cuda::getCurrentDeviceProperties()->maxGridSize[0], cuda::ATenCeilDiv(num_rows, int(threads.y))));

  tensor_kernel_scan_innermost_dim_with_indices<scalar_t, 16, 32><<<grid, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
    self.data_ptr<scalar_t>(), values.data_ptr<scalar_t>(), indices.data_ptr<int64_t>(),
    num_rows, row_size, init, binary_op);
  AT_CUDA_CHECK(cudaGetLastError());
}

template<typename scalar_t, typename BinaryFunction>
void scan_dim_with_indices(const Tensor& self, Tensor& values, Tensor& indices, //int64_t dim) {
     int64_t dim, scalar_t init, BinaryFunction binary_op) {
  int ndim = self.dim();
  Tensor self_ = self.contiguous();
  Tensor values_ = values.contiguous();
  Tensor indices_ = indices.contiguous();
   if (dim == ndim - 1) {
     scan_innermost_dim_with_indices<scalar_t>(self, values, indices, init, binary_op);
   } else {
     scan_outer_dim_with_indices<scalar_t>(self, values, indices, dim, init, binary_op);
   }
}

void cummax_helper_cuda(const Tensor& self, Tensor& values, Tensor& indices, int64_t dim) {
  TensorArg output_arg{ values, "output", 1 };
  TensorArg indices_arg{ indices, "indices", 2 };
  TensorArg input_arg{ self, "input", 3 };
  checkAllSameGPU("cummax", {output_arg, indices_arg, input_arg});
  AT_DISPATCH_ALL_TYPES_AND2(at::ScalarType::Bool, at::ScalarType::Half,
    self.scalar_type(), "cummax_cuda", [&]() {
    scalar_t init = self.is_floating_point() ? (-1*std::numeric_limits<scalar_t>::infinity()) : std::numeric_limits<scalar_t>::lowest();
    scan_dim_with_indices<scalar_t>(self, values, indices, dim, init, std::greater_equal<scalar_t>());
  });
}

void cummin_helper_cuda(const Tensor& self, Tensor& values, Tensor& indices, int64_t dim) {
  TensorArg output_arg{ values, "output", 1 };
  TensorArg indices_arg{ indices, "indices", 2 };
  TensorArg input_arg{ self, "input", 3 };
  checkAllSameGPU("cummin", {output_arg, indices_arg, input_arg});
  AT_DISPATCH_ALL_TYPES_AND2(at::ScalarType::Bool, at::ScalarType::Half,
    self.scalar_type(), "cummin_cuda", [&]() {
    scalar_t init = self.is_floating_point() ? std::numeric_limits<scalar_t>::infinity() : std::numeric_limits<scalar_t>::max();
    scan_dim_with_indices<scalar_t>(self, values, indices, dim, init, std::less_equal<scalar_t>());
  });
}

REGISTER_DISPATCH(std_var_stub, &std_var_kernel_cuda);
REGISTER_DISPATCH(sum_stub, &sum_kernel_cuda);
REGISTER_DISPATCH(prod_stub, &prod_kernel_cuda);
REGISTER_DISPATCH(mean_stub, &mean_kernel_cuda);
REGISTER_DISPATCH(norm_stub, &norm_kernel_cuda);
REGISTER_DISPATCH(and_stub, &and_kernel_cuda);
REGISTER_DISPATCH(or_stub, &or_kernel_cuda);
REGISTER_DISPATCH(max_values_stub, &max_values_kernel_cuda);
REGISTER_DISPATCH(min_values_stub, &min_values_kernel_cuda);
REGISTER_DISPATCH(argmax_stub, &argmax_kernel_cuda);
REGISTER_DISPATCH(argmin_stub, &argmin_kernel_cuda);

}} // namespace at::native
