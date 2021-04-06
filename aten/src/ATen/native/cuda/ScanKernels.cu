#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/NumericLimits.cuh>
#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/TensorUtils.h>
#include <ATen/NumericUtils.h>
#include <c10/util/accumulate.h>
#include <THC/THCGeneral.h>
#include <THC/THCNumerics.cuh>

#include <cub/device/device_scan.cuh>


namespace at { namespace native {

template <typename integer>
constexpr inline integer ceil_div(integer n, integer m) {
  return (n + m - 1) / m;
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
                  const uint32_t num_orows, const uint32_t num_irows, const uint32_t row_size, scalar_t init, BinaryFunction binary_op) {
  for (uint32_t orow = blockIdx.x; orow < num_orows; orow += gridDim.x) {
    for (uint32_t irow = blockIdx.y * blockDim.x + threadIdx.x; irow < num_irows; irow += gridDim.y * blockDim.x) {
      scalar_t *self = self_ + orow * row_size * num_irows + irow;
      scalar_t *values = values_ + orow * row_size * num_irows + irow;
      int64_t *indices = indices_ + orow * row_size * num_irows + irow;
      scalar_t out = init;
      int64_t out_idx = 0;

      for (auto col = decltype(row_size){0}; col < row_size; ++col) {
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

void check_fits_in_unsigned(int64_t val, const char* name) {
  constexpr auto umax = std::numeric_limits<uint32_t>::max();
  TORCH_CHECK(
      val >= 0 && val <= umax, name, " must fit in a 32-bit uint32_t value");
}


template<typename scalar_t, class BinaryFunction>
__host__ void scan_outer_dim_with_indices(const Tensor& self, Tensor& values, Tensor& indices,
                                       int dim, scalar_t init, BinaryFunction binary_op) {
  int64_t row_size = self.size(dim);
  auto sizes = self.sizes();

  // Treat all outer dimensions (i.e. dim_ < dim) as one.
  const int64_t num_orows = c10::multiply_integers(sizes.begin(), sizes.begin() + dim);

  // Treat all inner dimensions (i.e. dim > dimension) as one.
  const int64_t num_irows = c10::multiply_integers(sizes.begin() + dim + 1, sizes.end());
  //for performance reasons, cuda kernels use uint32_t for loops over irows, orows and row,
  //make sure that input is not bigger than supported by uint32_t
  check_fits_in_unsigned(num_irows, "num_irows");
  check_fits_in_unsigned(num_orows, "num_orows");
  check_fits_in_unsigned(row_size, "row_size");


  dim3 threads(std::min(512, int(num_irows)));
  int64_t maxGridDim = at::cuda::getCurrentDeviceProperties()->maxGridSize[1];
  dim3 grid(std::min(maxGridDim, num_orows), std::min(maxGridDim, ceil_div(num_irows, int64_t{threads.x})));
  tensor_kernel_scan_outer_dim_with_indices<scalar_t><<<grid, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
    self.data_ptr<scalar_t>(), values.data_ptr<scalar_t>(), indices.data_ptr<int64_t>(),
    num_orows, num_irows, row_size, init, binary_op);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename scalar_t, class BinaryFunction>
__host__ void scan_innermost_dim_with_indices(const Tensor& self, Tensor& values, Tensor& indices, scalar_t init, BinaryFunction binary_op) {
  int ndim = self.dim();
  // Treat all outer dimensions as a single dimension.
  int row_size = self.size(ndim - 1);
  int num_rows = self.numel() / row_size;

  dim3 threads(16, 32);
  dim3 grid(std::min(at::cuda::getCurrentDeviceProperties()->maxGridSize[0], ceil_div(num_rows, int(threads.y))));

  tensor_kernel_scan_innermost_dim_with_indices<scalar_t, 16, 32><<<grid, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
    self.data_ptr<scalar_t>(), values.data_ptr<scalar_t>(), indices.data_ptr<int64_t>(),
    num_rows, row_size, init, binary_op);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template<typename scalar_t, typename BinaryFunction>
void scan_dim_with_indices(const Tensor& self, Tensor& values, Tensor& indices, //int64_t dim) {
     int64_t dim, scalar_t init, BinaryFunction binary_op) {
  int ndim = self.dim();
  Tensor self_ = self.contiguous();
  Tensor values_ = values.contiguous();
  Tensor indices_ = indices.contiguous();
  bool copy_values = !values.is_contiguous();
  bool copy_indices = !indices.is_contiguous();
   if (dim == ndim - 1) {
     scan_innermost_dim_with_indices<scalar_t>(self_, values_, indices_, init, binary_op);
   } else {
     scan_outer_dim_with_indices<scalar_t>(self_, values_, indices_, dim, init, binary_op);
   }
   if (copy_values){
     values.copy_(values_);
   }
   if (copy_indices){
     indices.copy_(indices_);
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

// TODO: The implementation of `tensor_kernel_scan_outer_dim` and
// `tensor_kernel_scan_innermost_dim` is similar to
// `tensor_kernel_scan_outer_dim_with_indices`
// `tensor_kernel_scan_outer_dim_with_indices` and should be refactored to
// remove the duplication.

/* Perform an inclusive scan along an outer dimension of a tensor.
 *
 * - num_orows is the size of the flattened outer dimensions;
 * - num_irows is the size of the flattened inner dimensions;
 * - row_size is the size of the dimension along which to scan;
 *
 * The dimensions to the outside and inside of the specified dimension are considered as flattened.
 * Thread blocks with the same blockIdx.y process an "outer row" (i.e. an element of the flattened
 * outer dimensions, which contains several "inner rows").
 * Each thread processes a single inner row at a time.
 */
template<typename scalar_t, class BinaryOp>
__global__ void tensor_kernel_scan_outer_dim(scalar_t *tgt_, scalar_t *src_,
                                              const uint32_t num_orows, const uint32_t num_irows, const uint32_t row_size,
                                              const scalar_t init, BinaryOp binary_op)
{
  for (uint32_t orow = blockIdx.x; orow < num_orows; orow += gridDim.x) {
    for (uint32_t irow = blockIdx.y * blockDim.x + threadIdx.x; irow < num_irows; irow += gridDim.y * blockDim.x) {
      scalar_t *src = src_ + orow * row_size * num_irows + irow;
      scalar_t *tgt = tgt_ + orow * row_size * num_irows + irow;
      scalar_t acc = init;

      for (uint32_t col = 0; col < row_size; ++col) {
        acc = binary_op(acc, *src);
        *tgt = acc;

        src += num_irows;
        tgt += num_irows;
      }
    }
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
template<typename T, int num_threads_x, int num_threads_y, class BinaryFunction>
__device__ void tensor_kernel_scan_innermost_dim_impl(T* row_buf, T *tgt_, T *src_,
                                      const uint32_t num_rows, const uint32_t row_size,
                                      T init, BinaryFunction binary_op){
  for (uint32_t block_row = blockIdx.x * blockDim.y;
       block_row < num_rows;
       block_row += blockDim.y * gridDim.x) {
    uint32_t row = block_row + threadIdx.y;
    T block_total = init;

    T *row_src = src_ + row * row_size;
    T *row_tgt = tgt_ + row * row_size;

    // Perform scan on one block at a time, keeping track of the total value of
    // all blocks processed so far.
    for (uint32_t block_col = 0; block_col < row_size; block_col += 2 * num_threads_x) {
      // Load data into shared memory (two values per thread).
      uint32_t col1 = block_col + threadIdx.x;
      uint32_t col2 = block_col + num_threads_x + threadIdx.x;
      if (row < num_rows) {
        if (col1 < row_size) {
          row_buf[threadIdx.x] = row_src[col1];
        } else {
          row_buf[threadIdx.x] = init;
        }

        if (col2 < row_size) {
          row_buf[num_threads_x + threadIdx.x] = row_src[col2];
        } else {
          row_buf[num_threads_x + threadIdx.x] = init;
        }

        // Add the total value of all previous blocks to the first value of this block.
        if (threadIdx.x == 0) {
          row_buf[0] = binary_op(row_buf[0], block_total);
        }
      }
      __syncthreads();

      // Parallel reduction (up-sweep).
      for (uint32_t s = num_threads_x, d = 1; s >= 1; s >>= 1, d <<= 1) {
        if (row < num_rows && threadIdx.x < s) {
          uint32_t offset = (2 * threadIdx.x + 1) * d - 1;
          row_buf[offset + d] = binary_op(row_buf[offset], row_buf[offset + d]);
        }
        __syncthreads();
      }

      // Down-sweep.
      for (uint32_t s = 2, d = num_threads_x / 2; d >= 1; s <<= 1, d >>= 1) {
        if (row < num_rows && threadIdx.x < s - 1) {
          uint32_t offset = 2 * (threadIdx.x + 1) * d - 1;
          row_buf[offset + d] = binary_op(row_buf[offset], row_buf[offset + d]);
        }
        __syncthreads();
      }

      // Write back to output.
      if (row < num_rows) {
        if (col1 < row_size) row_tgt[col1] = row_buf[threadIdx.x];
        if (col2 < row_size) row_tgt[col2] = row_buf[num_threads_x + threadIdx.x];
      }
      block_total = row_buf[2 * num_threads_x - 1];
      __syncthreads();
    }
  }
}

template <
    typename T,
    int num_threads_x,
    int num_threads_y,
    class BinaryFunction>
__global__ typename std::enable_if<!c10::is_complex<T>::value, void>::type
tensor_kernel_scan_innermost_dim(
    T* tgt_,
    T* src_,
    const uint32_t num_rows,
    const uint32_t row_size,
    T init,
    BinaryFunction binary_op) {
  __shared__ T sbuf[num_threads_y][2 * num_threads_x];
  T* row_buf = sbuf[threadIdx.y];

  tensor_kernel_scan_innermost_dim_impl<T, num_threads_x, num_threads_y>(
      row_buf, tgt_, src_, num_rows, row_size, init, binary_op);
}

template <
    typename T,
    int num_threads_x,
    int num_threads_y,
    class BinaryFunction>
__global__ typename std::enable_if<c10::is_complex<T>::value, void>::type
tensor_kernel_scan_innermost_dim(
    T* tgt_,
    T* src_,
    const uint32_t num_rows,
    const uint32_t row_size,
    T init,
    BinaryFunction binary_op) {
  // As we cannot directly initialize shared array for complex types
  // Reference:
  //  `error: initializer not allowed for __shared__ variable`
  // We instead get the base scalar type and allocate twice number of
  // elements required of base type and reinterpret them as complex.
  using base_t = typename scalar_value_type<T>::type;
  __shared__ base_t sbuf[num_threads_y][4 * num_threads_x];

  T* row_buf = reinterpret_cast<T*>(sbuf[threadIdx.y]);

  tensor_kernel_scan_innermost_dim_impl<T, num_threads_x, num_threads_y>(
      row_buf, tgt_, src_, num_rows, row_size, init, binary_op);
}


template<typename scalar_t, class BinaryFunction>
__host__ void scan_outer_dim(const Tensor& self, Tensor& result,
                                       int dim, scalar_t init, BinaryFunction binary_op) {
  const int64_t row_size = self.size(dim);
  auto sizes = self.sizes();

  // Treat all outer dimensions (i.e. dim_ < dim) as one.
  const int64_t num_orows = c10::multiply_integers(sizes.begin(), sizes.begin() + dim);

  // Treat all inner dimensions (i.e. dim > dimension) as one.
  const int64_t num_irows = c10::multiply_integers(sizes.begin() + dim + 1, sizes.end());

  dim3 threads(std::min(512, int(num_irows)));
  int64_t maxGridDim = at::cuda::getCurrentDeviceProperties()->maxGridSize[1];
  dim3 grid(std::min(maxGridDim, num_orows), std::min(maxGridDim, ceil_div(num_irows, int64_t{threads.x})));

  check_fits_in_unsigned(num_irows, "num_irows");
  check_fits_in_unsigned(num_orows, "num_orows");
  check_fits_in_unsigned(row_size, "row_size");

  tensor_kernel_scan_outer_dim<scalar_t><<<grid, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
    result.data_ptr<scalar_t>(), self.data_ptr<scalar_t>(),
    num_orows, num_irows, row_size, init, binary_op);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename scalar_t, class BinaryFunction>
void scan_innermost_dim(const Tensor& self, Tensor& result, scalar_t init, BinaryFunction binary_op) {
  int64_t ndim = self.dim();
  // Treat all outer dimensions as a single dimension.
  int64_t row_size = self.size(ndim - 1);
  int64_t num_rows = self.numel() / row_size;

  dim3 threads(16, 32);
  int64_t maxGridDim = at::cuda::getCurrentDeviceProperties()->maxGridSize[0];
  dim3 grid(std::min(maxGridDim, ceil_div(num_rows, int64_t{threads.y})));

  check_fits_in_unsigned(num_rows, "Number of rows (self.numel()/self.size(self.dim()-1))");
  check_fits_in_unsigned(row_size, "row_size");

  tensor_kernel_scan_innermost_dim<scalar_t, 16, 32><<<grid, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
    result.data_ptr<scalar_t>(), self.data_ptr<scalar_t>(),
    num_rows, row_size, init, binary_op);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template<typename scalar_t, class func_t>
__global__ void transform_vals(scalar_t * a, scalar_t * b, scalar_t * out, func_t binary_op){
   *out = binary_op(*a, *b);
}

template<typename scalar_t, typename BinaryFunction>
void scan_cub(const Tensor& self, Tensor& result, scalar_t init, BinaryFunction binary_op) {
  int64_t size = self.numel();
  // non synchronizing cub call
  // even though cub is supposed to support tensors with int_max elements, in reality it doesn't,
  // so split at int_max/2
  constexpr int max_cub_size = std::numeric_limits<int>::max() / 2 + 1; // 2**30
  for (int64_t i = 0; i < size; i += max_cub_size) {
    int size_cub = std::min<int64_t>(size - i, max_cub_size);
    Tensor first_elem; // need to save it for all iterations other than first
    if (i > 0) {
      // need to temporarily transform first element of the range we are
      // operating on; self might be multi-d, but we need to index a single
      // element
      auto self_view = at::_unsafe_view(self, -1);
      first_elem = self_view[i].clone();
      transform_vals<<<1, 1, 0, at::cuda::getCurrentCUDAStream()>>>(
          self.data_ptr<scalar_t>() + i,
          result.data_ptr<scalar_t>() + i - 1,
          self.data_ptr<scalar_t>() + i,
          binary_op);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    size_t temp_storage_bytes = 0;
    AT_CUDA_CHECK(cub::DeviceScan::InclusiveScan(
        nullptr,
        temp_storage_bytes,
        self.data_ptr<scalar_t>() + i,
        result.data_ptr<scalar_t>() + i,
        binary_op,
        size_cub,
        at::cuda::getCurrentCUDAStream()));
    auto temp_storage = at::native::empty_cuda(
        {static_cast<int64_t>(temp_storage_bytes)},
        kByte, self.options().layout_opt(), self.options().device_opt(),
        self.options().pinned_memory_opt());
    AT_CUDA_CHECK(cub::DeviceScan::InclusiveScan(
        temp_storage.data_ptr(),
        temp_storage_bytes,
        self.data_ptr<scalar_t>() + i,
        result.data_ptr<scalar_t>() + i,
        binary_op,
        size_cub,
        at::cuda::getCurrentCUDAStream()));
    if (i > 0) {
      if (self.data_ptr<scalar_t>() != result.data_ptr<scalar_t>()) {
        // restore modified first element only if it's not an inplace operation
        auto self_view = at::_unsafe_view(self, -1);
        self_view[i].copy_(first_elem, /*non_blocking=*/true);
      }
    }
  }
}

template<typename scalar_t, typename BinaryFunction>
void scan_dim(const Tensor& self, Tensor& result,
     int64_t dim, scalar_t init, BinaryFunction binary_op) {
  int ndim = self.dim();
  Tensor self_ = self.contiguous();
  bool copy_result = !result.is_contiguous();
  Tensor result_ = result.contiguous();

  if (self.numel() == self.size(dim)) {
    scan_cub<scalar_t>(self_, result_, init, binary_op);
  } else if (dim == ndim - 1) {
    scan_innermost_dim<scalar_t>(self_, result_, init, binary_op);
  } else {
    scan_outer_dim<scalar_t>(self_, result_, dim, init, binary_op);
  }
  if (copy_result) {
    result.copy_(result_);
  }
}

Tensor& _logcumsumexp_out_cuda(const Tensor& self, int64_t dim, Tensor& result) {
  result.resize_(self.sizes());
  if (self.dim() == 0) {
    result.fill_(self);
    return result;
  }
  if (self.numel() == 0) {
    result.zero_();
    return result;
  }
  auto wrap_dim = maybe_wrap_dim(dim, self.dim());

  TensorArg output_arg{ result, "output", 1 };
  TensorArg input_arg{ self, "input", 2 };
  checkAllSameGPU("logcumsumexp", {output_arg, input_arg});

  AT_DISPATCH_FLOATING_TYPES_AND(at::ScalarType::Half,
    self.scalar_type(), "logcumsumexp_cuda", [&]() {
    using accscalar_t = acc_type<scalar_t, true>;
    scalar_t init = -std::numeric_limits<scalar_t>::infinity();
    auto log_add_exp = [] C10_HOST_DEVICE (const scalar_t x, const scalar_t y) -> scalar_t {
      scalar_t min = at::_isnan(y) ? y : std::min<scalar_t>(x,y); //std::min returns first arg if one of the args is nan
      scalar_t max = at::_isnan(y) ? y : std::max<scalar_t>(x,y); //std::max returns first arg if one of the args is nan
      if (min != max || ::isfinite(static_cast<accscalar_t>(min))) {
      // nan will be propagated here
          return ::log1p(std::exp(min - max)) + max;
      } else {
      // special case to correctly handle infinite inputs
         return x;
      }
    };
    scan_dim<scalar_t>(self, result, wrap_dim, init, log_add_exp);
  });

  return result;
}

Tensor _logcumsumexp_cuda(const Tensor& self, int64_t dim) {
  Tensor result = at::empty_like(self, MemoryFormat::Contiguous);
  return _logcumsumexp_out_cuda(self, dim, result);
}

Tensor& _cumsum_out_cuda(Tensor& result, const Tensor& self, int64_t dim) {
  TensorArg output_arg{result, "output", 1};
  TensorArg input_arg{self, "input", 2};
  checkAllSameGPU("cumsum", {output_arg, input_arg});
  checkSameType("cumsum", output_arg, input_arg);

  result.resize_(self.sizes());
  if (self.dim() == 0) {
    result.fill_(self);
    return result;
  }
  if (self.numel() == 0) {
    result.zero_();
    return result;
  }
  auto wrap_dim = maybe_wrap_dim(dim, self.dim());

  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND(
      at::ScalarType::Half, self.scalar_type(), "cumsum_cuda", [&]() {
        scalar_t init = 0;
        scan_dim<scalar_t>(
            self,
            result,
            wrap_dim,
            init,
            std::plus<scalar_t>());
      });

  return result;
}

Tensor _cumsum_cuda(const Tensor& self, int64_t dim) {
  Tensor result = at::empty_like(self, MemoryFormat::Contiguous);
  return _cumsum_out_cuda(result, self, dim);
}

Tensor& _cumprod_out_cuda(Tensor& result, const Tensor& self, int64_t dim) {
  TensorArg output_arg{result, "output", 1};
  TensorArg input_arg{self, "input", 2};
  checkAllSameGPU("cumprod", {output_arg, input_arg});
  checkSameType("cumprod", output_arg, input_arg);

  result.resize_(self.sizes());
  if (self.dim() == 0) {
    result.fill_(self);
    return result;
  }
  if (self.numel() == 0) {
    result.zero_();
    return result;
  }
  auto wrap_dim = maybe_wrap_dim(dim, self.dim());

  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND(
      at::ScalarType::Half, self.scalar_type(), "cumprod_cuda", [&]() {
        scalar_t init = 1;
        scan_dim<scalar_t>(
            self,
            result,
            wrap_dim,
            init,
            std::multiplies<scalar_t>());
      });

  return result;
}

Tensor _cumprod_cuda(const Tensor& self, int64_t dim) {
  Tensor result = at::empty_like(self, MemoryFormat::Contiguous);
  return _cumprod_out_cuda(result, self, dim);
}

}} // namespace at::native
