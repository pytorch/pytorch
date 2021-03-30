#include <ATen/ATen.h>
#include <ATen/ExpandUtils.h>
#include <ATen/NativeFunctions.h>
#include <ATen/SparseTensorUtils.h>
#include <ATen/WrapDimUtilsMulti.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAUtils.h>
#include <ATen/native/sparse/SparseTensorMath.h>
#include <ATen/native/sparse/ParamUtils.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/native/sparse/cuda/SparseCUDAApplyUtils.cuh>
#include <ATen/native/sparse/cuda/SparseCUDABlas.cuh>

#include <THC/THCTensorMathPointwise.cuh>
#include <THC/THCThrustAllocator.cuh>

#include <thrust/binary_search.h>
#include <thrust/device_ptr.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/system/cuda/execution_policy.h>

#include <cuda_runtime_api.h>
#include <cusparse.h>
#include <bitset>

#include <c10/cuda/CUDAMathCompat.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/detail/OffsetCalculator.cuh>
#include <ATen/native/cuda/Loops.cuh>

#include <c10/macros/Macros.h>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/for_each.h>
#include <thrust/functional.h>
#include <thrust/gather.h>
#include <thrust/generate.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/unique.h>

#include <c10/cuda/CUDAMathCompat.h>

namespace at {
namespace native {
namespace {

// Number of threads in a block given an input size up to MAX_BLOCK_SIZE
static int getNumThreads(int nElem) {
#if defined(__HIP_PLATFORM_HCC__)
  int threadSizes[5] = {16, 32, 64, 128, 256};
#else
  int threadSizes[5] = {32, 64, 128, 256, 512};
#endif
  for (int i = 0; i != 5; ++i) {
    if (nElem <= threadSizes[i]) {
      return threadSizes[i];
    }
  }
  return threadSizes[4];
}

template <typename scalar_t, bool LogSoftMax>
__global__ void cuda_sparse_coo_softmax_kernel(
    int64_t* sorted_pool_indices,
    int64_t size,
    int64_t* pool_sizes,
    int64_t* pool_offsets,
    int64_t nvalues,
    scalar_t* mx_rows,
    PackedTensorAccessor<scalar_t, 2> input_values_acc,
    PackedTensorAccessor<scalar_t, 2> output_values_acc) {
  /*
    See ATen/native/sparse/Softmax.cpp:cpu_sparse_coo_softmax for the CPU
    implementation of the sparse softmax algorithm that this implementation is
    based on.
  */
  int tid = threadIdx.x;
  int blkid = blockIdx.x;
  int blksz = blockDim.x;
  int gridsz = gridDim.x;

  int index = tid + blkid * blksz;
  int step = blksz * gridsz;

  while (index < size) {
    int64_t offset = pool_offsets[index];
    int64_t* pool_indices = sorted_pool_indices + offset;
    int64_t pool_indices_size = pool_sizes[index];
    scalar_t* mx_row = mx_rows + index * nvalues;

    for (int64_t j = 0; j < nvalues; j++) {
      scalar_t exp_sums = 0;
      for (int64_t p = 0; p < pool_indices_size; p++) {
        auto i = pool_indices[p];
        auto values_row = input_values_acc[i];
        auto out_values_row = output_values_acc[i];

        auto v = c10::cuda::compat::exp(values_row[j] - mx_row[j]);
        if (!LogSoftMax) {
          out_values_row[j] = v;
        }
        exp_sums += v;
      }
      for (int64_t p = 0; p < pool_indices_size; p++) {
        auto i = pool_indices[p];
        auto values_row = input_values_acc[i];
        auto out_values_row = output_values_acc[i];

        if (LogSoftMax) {
          out_values_row[j] = values_row[j] - mx_row[j] - c10::cuda::compat::log(exp_sums);
        } else {
          out_values_row[j] *= 1.0 / exp_sums;
        }
      }
    }
    index += step;
  }
}

template <typename scalar_t, bool LogSoftMax>
__global__ void cuda_sparse_coo_softmax_backward_kernel(
    int64_t* sorted_pool_indices,
    int64_t size,
    int64_t* pool_sizes,
    int64_t* pool_offsets,
    int64_t nvalues,
    int64_t grad_nnz,
    int64_t* grad_offsets,
    int64_t* out_offsets,
    int64_t* lower_bound_values,
    PackedTensorAccessor<scalar_t, 2> values_accessor,
    PackedTensorAccessor<scalar_t, 2> out_values_accessor,
    PackedTensorAccessor<scalar_t, 2> grad_values_accessor) {
  /*
    See ATen/native/sparse/Softmax.cpp:cpu_sparse_coo_softmax_backward for
    the CPU implementation of the sparse softmax backward algorithm that this
    implementation is based on.
  */
  int tid = threadIdx.x;
  int blkid = blockIdx.x;
  int blksz = blockDim.x;
  int gridsz = gridDim.x;

  int index = tid + blkid * blksz;
  int step = blksz * gridsz;

  while (index < size) {
    int64_t offset = pool_offsets[index];
    int64_t* pool_indices = sorted_pool_indices + offset;
    int64_t pool_indices_size = pool_sizes[index];

    for (int64_t k = 0; k < nvalues; k++) {
      scalar_t tmp_row{0};

      /* Compute tmp = - sum_j output_j * grad_j */
      for (int64_t p = 0; p < pool_indices_size; p++) {
        auto i = pool_indices[p];
        auto out_values_row = out_values_accessor[i];
        auto j = lower_bound_values[i];

        /* Update `tmp_row` accumulator only when limits and pools are valid */
        if (j < grad_nnz && (out_offsets[i] == grad_offsets[j])) {
          auto grad_values_row = grad_values_accessor[j];
          if (LogSoftMax) {
            tmp_row -= grad_values_row[k];
          } else {
            tmp_row -= out_values_row[k] * grad_values_row[k];
          }
        }
      }

      /* Compute grad_input = output * (grad + tmp)*/
      for (int64_t p = 0; p < pool_indices_size; p++) {
        auto i = pool_indices[p];
        auto out_values_row = out_values_accessor[i];
        auto values_row = values_accessor[i];
        auto j = lower_bound_values[i];
        if (j < grad_nnz && (out_offsets[i] == grad_offsets[j])) {
          auto grad_values_row = grad_values_accessor[j];
          if (LogSoftMax) {
            values_row[k] = grad_values_row[k] +
                c10::cuda::compat::exp(out_values_row[k]) * tmp_row;
          } else {
            values_row[k] =
                out_values_row[k] * (grad_values_row[k] + tmp_row);
          }
        } else {
          if (LogSoftMax) {
            values_row[k] =
                c10::cuda::compat::exp(out_values_row[k]) * tmp_row;
          } else {
            values_row[k] = out_values_row[k] * tmp_row;
          }
        }
      }
    }
    index += step;
  }
}

using thrust_ptr = thrust::device_ptr<int64_t>;

Tensor get_offsets(
    const Tensor& indices,
    const IntArrayRef& sizes,
    const int64_t dim) {
  /*
    See ATen/native/sparse/Softmax.cpp:get_offsets for the CPU
    implementation of get_offsets function that this implementation is based on.
  */
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  auto allocator = THCThrustAllocator(globalContext().lazyInitCUDA());
  auto policy = thrust::cuda::par(allocator).on(stream);

  auto ndim = indices.size(0);
  auto nnz = indices.size(1);
  std::vector<int64_t> host_strides(ndim, 1);
  if (ndim > 1) {
    for (int64_t i = ndim - 2; i >= 0; i--) {
      host_strides[i] =
          host_strides[i + 1] * (i + 1 == dim ? 1 : sizes[i + 1]);
    }
  }
  auto strides = at::empty({ndim}, indices.options());
  auto strides_ptr = strides.data_ptr<int64_t>();

  AT_CUDA_CHECK(cudaMemcpyAsync(
          strides_ptr, host_strides.data(), host_strides.size() * sizeof(int64_t),
          cudaMemcpyHostToDevice,
          stream));

  auto indices_accessor = indices.packed_accessor<int64_t, 2>();

  Tensor offsets = at::empty({nnz}, indices.options());

  thrust::transform(
      policy,
      thrust::make_counting_iterator(int64_t(0)),
      thrust::make_counting_iterator(int64_t(nnz)),
      thrust::device_ptr<int64_t>(offsets.data_ptr<int64_t>()),
      [indices_accessor, strides_ptr, dim, ndim] __device__(int64_t x) {
        int64_t pool_index = 0;
        for (int64_t j = 0; j < ndim; j++) {
          if (j != dim) {
            auto indices_row = indices_accessor[j];
            auto stride = strides_ptr[j];
            pool_index += stride * indices_row[x];
          }
        }
        return pool_index;
      });
  return offsets;
}

template <class scalar_t, bool requireMxRows = true>
std::tuple<Tensor, Tensor, Tensor, Tensor> compute_pool_max(
    const Tensor& indices,
    const Tensor& values,
    const IntArrayRef& sizes,
    int64_t nvalues,
    const int64_t dim) {
  /*
    Return pools of indices that align with the given dimension and the
    corresponding max values for each pool.

    See ATen/native/sparse/Softmax.cpp:get_offsets and
    ATen/native/sparse/Softmax.cpp:cpu_sparse_coo_softmax for the CPU
    implementation that this implementation is based on.
  */
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  auto allocator = THCThrustAllocator(globalContext().lazyInitCUDA());
  auto policy = thrust::cuda::par(allocator).on(stream);

  auto nnz = indices.size(1);
  auto offsets = get_offsets(indices, sizes, dim);
  int64_t* offsets_ptr = offsets.data_ptr<int64_t>();

  auto sorted_indices = at::empty({nnz}, indices.options());
  thrust_ptr sorted_indices_thrust_ptr(sorted_indices.data_ptr<int64_t>());
  thrust::sequence(
      policy, sorted_indices_thrust_ptr, sorted_indices_thrust_ptr + nnz, 0);

  thrust::sort(
      policy,
      sorted_indices_thrust_ptr,
      sorted_indices_thrust_ptr + nnz,
      [offsets_ptr] __device__(int64_t x, int64_t y) {
        return offsets_ptr[x] < offsets_ptr[y];
      });
  auto pool_sizes = at::empty({nnz}, indices.options());

  auto new_end = thrust::reduce_by_key(
      policy,
      sorted_indices_thrust_ptr,
      sorted_indices_thrust_ptr + nnz,
      thrust::make_constant_iterator(int64_t(1)),
      thrust::make_discard_iterator(),
      thrust_ptr(pool_sizes.data_ptr<int64_t>()),
      [offsets_ptr] __device__(int64_t x, int64_t y) {
        return offsets_ptr[x] == offsets_ptr[y];
      });
  auto new_sz = thrust::distance(
      thrust_ptr(pool_sizes.data_ptr<int64_t>()), new_end.second);
  pool_sizes.resize_({new_sz});

  auto pool_offsets = pool_sizes.clone();
  thrust_ptr pool_offsets_thrust_ptr(
      pool_offsets.data_ptr<int64_t>());
  thrust::exclusive_scan(
      policy,
      pool_offsets_thrust_ptr,
      pool_offsets_thrust_ptr + new_sz,
      pool_offsets_thrust_ptr);

  Tensor mx_buffer;
  if (requireMxRows) {

    auto values_accessor =
        values.packed_accessor<scalar_t, 2>(); // {nnz, nvalues}

    mx_buffer = at::full({new_sz * nvalues}, Scalar(-std::numeric_limits<scalar_t>::infinity()), values.options());

    auto mx_buffer_ptr = mx_buffer.data_ptr<scalar_t>();

    auto pool_sizes_ptr = pool_sizes.data_ptr<int64_t>();
    auto sorted_indices_ptr = sorted_indices.data_ptr<int64_t>();
    auto pool_offsets_ptr = pool_offsets.data_ptr<int64_t>();

    thrust::for_each(
        policy,
        thrust::make_counting_iterator(int64_t(0)),
        thrust::make_counting_iterator(int64_t(new_sz)),
        [values_accessor,
         sorted_indices_ptr,
         pool_sizes_ptr,
         pool_offsets_ptr,
         mx_buffer_ptr,
         nvalues] __device__(int64_t index) {
          int64_t curr_pool_size = pool_sizes_ptr[index];
          auto mx_row = mx_buffer_ptr + index * nvalues;
          int64_t offset = pool_offsets_ptr[index];
          for (int64_t p = 0; p < curr_pool_size; p++) {
            int64_t i = *(sorted_indices_ptr + offset + p);
            auto values_row = values_accessor[i].data();
            for (int64_t j = 0; j < nvalues; j++) {
              mx_row[j] = c10::cuda::compat::max(mx_row[j], values_row[j]);
            }
          }
        });
  }
  return std::make_tuple(
      sorted_indices, pool_offsets, pool_sizes, mx_buffer);
}

template <typename scalar_t, bool LogSoftMax>
void cuda_sparse_coo_softmax(
    Tensor& output,
    const Tensor& input,
    const int64_t dim) {
  /*
    See ATen/native/sparse/Softmax.cpp:cpu_sparse_coo_softmax for the CPU
    implementation of the sparse softmax algorithm that this implementation is
    based on.
  */
  auto sparse_dim = input.sparse_dim();
  auto indices = input._indices().contiguous();
  auto values = input._values().contiguous();
  auto out_values = output._values();
  auto out_indices = output._indices();
  out_values.resize_as_(values);
  out_indices.resize_as_(indices);
  out_indices.copy_(indices);

  if (dim >= sparse_dim) {
    if (LogSoftMax) {
      auto new_values = log_softmax_cuda(values, dim - sparse_dim + 1, false);
      out_values.set_(new_values);
    } else {
      auto new_values = softmax_cuda(values, dim - sparse_dim + 1, false);
      out_values.set_(new_values);
    }
    return;
  }

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  auto allocator = THCThrustAllocator(globalContext().lazyInitCUDA());
  auto policy = thrust::cuda::par(allocator).on(stream);

  auto nnz = values.size(0);
  auto sizes = input.sizes();
  auto nvalues = values.numel() / nnz;

  /* Prepare accessors */
  auto values_2 = values.view({nnz, nvalues});
  auto values_accessor = values_2.packed_accessor<scalar_t, 2>();

  auto out_values_2 = out_values.view({nnz, nvalues});
  auto out_values_accessor = out_values_2.packed_accessor<scalar_t, 2>();

  Tensor sorted_indices;
  Tensor pool_offsets;
  Tensor pool_sizes;
  Tensor mx_buffer;

  std::tie(sorted_indices, pool_offsets, pool_sizes, mx_buffer) =
      compute_pool_max<scalar_t, true>(indices, values_2, sizes, nvalues, dim);

  auto pool_size = pool_offsets.size(0);
  int block_size = getNumThreads(pool_size);
  const int grid_size = (pool_size + block_size - 1) / block_size;

  cuda_sparse_coo_softmax_kernel<scalar_t, LogSoftMax>
      <<<grid_size, block_size, 0, stream>>>(
          sorted_indices.data_ptr<int64_t>(),
          pool_size,
          pool_sizes.data_ptr<int64_t>(),
          pool_offsets.data_ptr<int64_t>(),
          nvalues,
          mx_buffer.data_ptr<scalar_t>(),
          values_accessor,
          out_values_accessor);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename scalar_t, bool LogSoftMax>
void cuda_sparse_coo_softmax_backward(
    Tensor& grad_input,
    const Tensor& grad,
    const Tensor& output,
    const int64_t dim) {
  /*
    See ATen/native/sparse/Softmax.cpp:cpu_sparse_coo_softmax_backward for
    the CPU implementation of the sparse softmax backward algorithm that this
    implementation is based on.
  */
  auto sparse_dim = output.sparse_dim();
  auto sizes = output.sizes().vec();
  auto grad_indices = grad._indices().contiguous();
  auto grad_values = grad._values().contiguous();
  auto out_indices = output._indices().contiguous();
  auto out_values = output._values().contiguous();
  auto values = grad_input._values();
  auto indices = grad_input._indices();
  auto out_nnz = out_values.size(0);
  auto grad_nnz = grad_values.size(0);

  values.resize_as_(out_values);
  values.zero_();
  indices.resize_as_(out_indices);
  indices.copy_(out_indices);

  auto out_offsets = get_offsets(out_indices, sizes, -1);
  auto grad_offsets = get_offsets(grad_indices, sizes, -1);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  auto allocator = THCThrustAllocator(globalContext().lazyInitCUDA());
  auto policy = thrust::cuda::par(allocator).on(stream);

  /* when dim >= sparse_dim the dense backward is used */
  if (dim >= sparse_dim) {
    if (at::native::cuda_equal(out_offsets, grad_offsets) == true) {
      Tensor unused = at::native::empty_like(grad_values);
      if (LogSoftMax) {
        auto r = log_softmax_backward_cuda(grad_values, out_values, dim - sparse_dim + 1, unused);
        values.set_(r);
      } else {
        auto r = softmax_backward_cuda(grad_values, out_values, dim - sparse_dim + 1, unused);
        values.set_(r);
      }
    } else {
      auto host_out_offsets =
          out_offsets.to(at::Device(kCPU), indices.dtype(), false, true);
      auto host_grad_offsets =
          grad_offsets.to(at::Device(kCPU), indices.dtype(), false, true);
      auto out_offsets_accessor = host_out_offsets.data_ptr<int64_t>();
      auto grad_offsets_accessor = host_grad_offsets.data_ptr<int64_t>();
      for (int64_t i = 0; i < out_nnz; i++) {
        Tensor unused = at::native::empty_like(grad_values);
        auto low = thrust::lower_bound(
            grad_offsets_accessor,
            grad_offsets_accessor + grad_offsets.size(0),
            out_offsets_accessor[i]);
        auto j = low - grad_offsets_accessor;
        /*
          Compute output using dense backward only when limits and pools are valid
          If this check is false then a sparse tensor with full of zeros is returned
        */
        if (j < grad_nnz && out_offsets_accessor[i] == grad_offsets_accessor[j]) {
          if (LogSoftMax) {
            auto r = log_softmax_backward_cuda(
                grad_values[j], out_values[i], dim - sparse_dim, unused);
            values[i].copy_(r);
          } else {
            auto r = softmax_backward_cuda(
                grad_values[j], out_values[i], dim - sparse_dim, unused);
            values[i].copy_(r);
          }
        }
      }
    }
    return;
  }

  auto nnz = values.size(0);
  auto nvalues = values.numel() / nnz;

  auto values_2 = values.view({nnz, nvalues});
  auto values_accessor = values_2.packed_accessor<scalar_t, 2>();

  auto out_values_2 = out_values.view({out_nnz, nvalues});
  auto out_values_accessor = out_values_2.packed_accessor<scalar_t, 2>();

  auto grad_values_2 = grad_values.view({grad_nnz, nvalues});
  auto grad_values_accessor = grad_values_2.packed_accessor<scalar_t, 2>();

  Tensor lower_bound_values =
      at::empty({out_offsets.size(0)}, indices.options());

  thrust::lower_bound(
      policy,
      thrust_ptr(grad_offsets.data_ptr<int64_t>()),
      thrust_ptr(grad_offsets.data_ptr<int64_t>() + grad_offsets.size(0)),
      thrust_ptr(out_offsets.data_ptr<int64_t>()),
      thrust_ptr(out_offsets.data_ptr<int64_t>()) + out_offsets.size(0),
      thrust_ptr(lower_bound_values.data_ptr<int64_t>()));

  Tensor sorted_indices;
  Tensor pool_offsets;
  Tensor pool_sizes;

  /* Compute independent pools of indices */
  std::tie(
      sorted_indices, pool_offsets, pool_sizes, std::ignore) =
      compute_pool_max<scalar_t, false>(
          out_indices, values_2, sizes, nvalues, dim);

  auto pool_size = pool_offsets.size(0);

  int block_size = getNumThreads(pool_size);
  const int grid_size = (pool_size + block_size - 1) / block_size;

  cuda_sparse_coo_softmax_backward_kernel<scalar_t, LogSoftMax>
      <<<grid_size, block_size, 0, stream>>>(
          sorted_indices.data_ptr<int64_t>(),
          pool_size,
          pool_sizes.data_ptr<int64_t>(),
          pool_offsets.data_ptr<int64_t>(),
          nvalues,
          grad_nnz,
          grad_offsets.data_ptr<int64_t>(),
          out_offsets.data_ptr<int64_t>(),
          lower_bound_values.data_ptr<int64_t>(),
          values_accessor,
          out_values_accessor,
          grad_values_accessor);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // end anonymous namespace

Tensor softmax_sparse_cuda(
    const Tensor& input_,
    const int64_t dim,
    const bool half_to_float) {
  Tensor input, output;
  std::tie(input, output) = softmax_sparse_input_preprocessing(
      input_, dim, half_to_float, "softmax");
  if (input.numel() == 0) {
    return output;
  }
  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "softmax", [&] {
    cuda_sparse_coo_softmax<scalar_t, false>(output, input, dim);
  });
  return output;
}

Tensor log_softmax_sparse_cuda(
    const Tensor& input_,
    const int64_t dim,
    const bool half_to_float) {
  Tensor input, output;
  std::tie(input, output) = softmax_sparse_input_preprocessing(
      input_, dim, half_to_float, "log_softmax");
  if (input.numel() == 0) {
    return output;
  }
  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "log_softmax", [&] {
    cuda_sparse_coo_softmax<scalar_t, true>(output, input, dim);
  });
  return output;
}

Tensor softmax_backward_sparse_cuda(
    const Tensor& grad_,
    const Tensor& output_,
    int64_t dim_,
    const Tensor& input_) {
  Tensor grad_input, grad, output;
  std::tie(grad_input, grad, output) =
      softmax_backward_sparse_input_preprocessing(
          grad_, output_, dim_, input_, "softmax_backward");
  if (output.numel() == 0) {
    return grad_input;
  }
  AT_DISPATCH_FLOATING_TYPES(grad.scalar_type(), "softmax_backward", [&] {
    cuda_sparse_coo_softmax_backward<scalar_t, false>(
        grad_input, grad, output, dim_);
  });
  return grad_input;
}

Tensor log_softmax_backward_sparse_cuda(
    const Tensor& grad_,
    const Tensor& output_,
    int64_t dim_,
    const Tensor& input_) {
  Tensor grad_input, grad, output;
  std::tie(grad_input, grad, output) =
      softmax_backward_sparse_input_preprocessing(
          grad_, output_, dim_, input_, "log_softmax_backward");
  if (output.numel() == 0) {
    return grad_input;
  }

  AT_DISPATCH_FLOATING_TYPES(grad.scalar_type(), "log_softmax_backward", [&] {
    cuda_sparse_coo_softmax_backward<scalar_t, true>(
        grad_input, grad, output, dim_);
  });
  return grad_input;
}

} // namespace native
} // namespace at
