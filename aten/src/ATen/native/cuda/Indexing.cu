#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/TensorAdvancedIndexing.h>
#include <ATen/native/IndexingUtils.h>
#include <ATen/native/quantized/IndexKernel.h>
#include <ATen/native/cuda/KernelUtils.cuh>

#include <ATen/core/Tensor.h>
#include <ATen/ceil_div.h>
#include <ATen/Dispatch.h>
#include <ATen/Dispatch_v2.h>
#include <ATen/ExpandUtils.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/TensorOperators.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/Resize.h>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/CUDAUtils.h>
#include <ATen/cuda/DeviceUtils.cuh>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/arange.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/zeros_like.h>
#include <ATen/ops/ones_like.h>
#include <ATen/ops/empty_quantized.h>
#include <ATen/ops/index_add_native.h>
#include <ATen/ops/index_reduce_native.h>
#include <ATen/ops/index_select_native.h>
#include <ATen/ops/masked_fill_native.h>
#include <ATen/ops/_sparse_coo_tensor_with_dims_and_tensors.h>
#endif

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/cub.h>
#include <c10/util/irange.h>
#include <c10/core/QScheme.h>
#include <ATen/native/quantized/AffineQuantizerBase.h>

#include <limits>

#include <c10/macros/Macros.h>

namespace {
template <typename scalar_t, int SZ>
__global__ void indexing_backward_kernel(
  const int64_t* sorted_indices, const int64_t* indices, const scalar_t* grad_output, scalar_t* grad_weight,
  int64_t numel, int64_t stride, int64_t stride_before, int64_t outer_dim, bool accumulate) {
//numel is total number of flattened indices, not expanded to dimensions that are not indexed.
//stride is the cumulative size of the not-indexed last dimensions
//stride_before is the stride of the dimension immediately preceding first indexed dimension
//if indexing starts from the 0th dimension, stride_before does not matter because blockIdx.z will be 0 in this case
//outer_dim is number of elements in the first unindexed dimensions
  using opmath_t = at::opmath_type<scalar_t>;

  // Each warp is responsible for an input into the LookupTable.
  // If the preceding input has the same destination index as this input, then the warp
  // exits immediately. The warp also processes subsequent inputs with the
  // same value.
  //
  // Input Warp
  // 1     <warp 1>
  // 1     <warp 1> (<warp 2> exits without doing any work)
  // 5     <warp 3>
  // 8     <warp 4>

  // Number of values processed by each thread (grain size)
  for (int64_t z = blockIdx.z; z < outer_dim; z += gridDim.z){
    int64_t idx = blockIdx.x * blockDim.y + threadIdx.y;
    if (idx < numel
        && (idx == 0 || sorted_indices[idx] != sorted_indices[idx - 1])){
      do {
        int64_t start_feature = threadIdx.x + blockIdx.y * blockDim.x * SZ;
        // if not accumulate, we only keep the last duplicate index so skip those before it
        if (!accumulate && (idx < numel - 1) && sorted_indices[idx] == sorted_indices[idx + 1]) {
          idx++;
          continue;
        }
        const int64_t weight_row = ((int64_t) sorted_indices[idx]) * stride + z * stride_before;
        const int64_t grad_row = ((int64_t) indices[idx]) * stride + z * numel * stride;
        const opmath_t scale = (opmath_t)1.0;

        opmath_t gradient[SZ];
        opmath_t weight[SZ];

        while (start_feature < stride) {
          #pragma unroll
          for (int ii = 0; ii < SZ; ii++) {
            int64_t feature_dim = start_feature + ii * C10_WARP_SIZE;
            if (feature_dim < stride) {
              gradient[ii] = static_cast<opmath_t>(grad_output[grad_row + feature_dim]);
              if (accumulate) {
                weight[ii] = static_cast<opmath_t>(grad_weight[weight_row + feature_dim]);
              }
            }
          }

          #pragma unroll
          for (int ii = 0; ii < SZ; ii++) {
            if (accumulate) {
              weight[ii] += gradient[ii] * scale;
            } else {
              weight[ii] = gradient[ii] * scale;
            }
          }

          #pragma unroll
          for (int ii = 0; ii < SZ; ii++) {
            int64_t feature_dim = start_feature + ii * C10_WARP_SIZE;
            if (feature_dim < stride) {
                grad_weight[weight_row + feature_dim] = static_cast<scalar_t>(weight[ii]);
            }
          }
          start_feature += gridDim.y * blockDim.x * SZ;
        }

        idx++;
      } while (idx < numel && sorted_indices[idx] == sorted_indices[idx - 1]);
    }
  }
}

template <typename scalar_t>
__global__ void indexing_backward_kernel_stride_1(
  const int64_t* sorted_indices, const int64_t* indices, const scalar_t* grad_output, scalar_t* grad_weight,
  int64_t numel, int64_t stride, int64_t stride_before, int64_t outer_dim, bool accumulate) {
  using opmath_t = at::opmath_type<scalar_t>;

  // Number of values processed by each thread (grain size)
  for (int64_t z = blockIdx.z; z < outer_dim; z += gridDim.z){
    int64_t idx = blockIdx.x * blockDim.y + threadIdx.y;
    int64_t crnt_sorted_idx = sorted_indices[idx];

    if ((idx < numel) &&
        (idx == 0 || crnt_sorted_idx != sorted_indices[idx - 1]))
    {
      // Determine the number of duplicates in advance
      int64_t num_duplicates = 1;
      while (((idx + num_duplicates) < numel) && (sorted_indices[idx + num_duplicates] == crnt_sorted_idx)) {
        num_duplicates++;
      }

      // Continue computing weights
      const int64_t weight_row = crnt_sorted_idx * stride + z * stride_before;
      int64_t grad_row = 0;
      const opmath_t scale = (opmath_t)1.0;

      if (!accumulate) {
        grad_row = ((int64_t)indices[idx + num_duplicates - 1]) * stride + z * numel * stride;
        grad_weight[weight_row] =
          static_cast<scalar_t>(static_cast<opmath_t>(grad_output[grad_row]) * scale);
      } else {
        opmath_t gradient = (opmath_t)0.0;

        int laneIdx = threadIdx.x % C10_WARP_SIZE;
        int64_t num_warp_passes = num_duplicates / C10_WARP_SIZE;
        for (int64_t i = 0; i < num_warp_passes; ++i) {
            grad_row = ((int64_t) indices[idx + i * C10_WARP_SIZE + laneIdx]) * stride + z * numel * stride;
            gradient += static_cast<opmath_t>(grad_output[grad_row]) * scale;
        }
        WARP_SYNC();
        for (int offset = C10_WARP_SIZE / 2; offset > 0; offset /= 2) {
          gradient += WARP_SHFL_DOWN(gradient, offset);
        }

        if (laneIdx == 0) {
          for (int64_t i = num_warp_passes * C10_WARP_SIZE; i < num_duplicates; ++i) {
            grad_row = ((int64_t) indices[idx + i]) * stride + z * numel * stride;
            gradient += static_cast<opmath_t>(grad_output[grad_row]) * scale;
          }

          grad_weight[weight_row] = static_cast<scalar_t>(static_cast<opmath_t>(grad_weight[weight_row]) + gradient);
        }
      }
    }
  }
}

template <typename scalar_t>
__global__ void indexing_backward_kernel_small_stride(
  const int64_t* sorted_indices, const int64_t* indices, const scalar_t* grad_output, scalar_t* grad_weight,
  int64_t numel, int64_t stride, int64_t stride_before, int64_t outer_dim, bool accumulate) {
  using opmath_t = at::opmath_type<scalar_t>;

  // Number of values processed by each thread (grain size)
  for (int64_t z = blockIdx.z; z < outer_dim; z += gridDim.z){
    int64_t idx = blockIdx.x * blockDim.y + threadIdx.y;
    int64_t tidx = threadIdx.x;
    int64_t crnt_sorted_idx = sorted_indices[idx];

    if ((idx < numel) &&
        (tidx < stride) &&
        (idx == 0 || crnt_sorted_idx != sorted_indices[idx - 1]))
    {
      // Determine the number of duplicates in advance
      int64_t num_duplicates = 1;
      while (((idx + num_duplicates) < numel) && (sorted_indices[idx + num_duplicates] == crnt_sorted_idx)) {
        num_duplicates++;
      }

      // Continue computing weights
      const int64_t weight_row = crnt_sorted_idx * stride + z * stride_before;
      int64_t grad_row = 0;
      const opmath_t scale = (opmath_t)1.0;

      if (!accumulate) {
        grad_row = ((int64_t)indices[idx + num_duplicates - 1]) * stride + z * numel * stride;
        grad_weight[weight_row + tidx] =
          static_cast<scalar_t>(static_cast<opmath_t>(grad_output[grad_row + tidx]) * scale);
      } else {
        opmath_t gradient = (opmath_t)0.0;
        for (int64_t i = 0; i < num_duplicates; ++i) {
          grad_row = ((int64_t) indices[idx + i]) * stride + z * numel * stride;
          gradient += static_cast<opmath_t>(grad_output[grad_row + tidx]) * scale;
        }

        grad_weight[weight_row + tidx] = static_cast<scalar_t>(static_cast<opmath_t>(grad_weight[weight_row + tidx]) + gradient);
      }
    }
  }
}

template <typename scalar_t, int SZ>
__global__ void indexing_backward_kernel_quantized(
  const int64_t* sorted_indices, const int64_t* indices, const float* grad_output, scalar_t* grad_weight,
  int64_t numel, int64_t stride, int64_t stride_before, int64_t outer_dim,
  float inv_scale, int zero_point, int64_t qmin, int64_t qmax) {

  // This implementation is adopted from indexing_backward_kernel above.
  using opmath_t = at::opmath_type<float>;
  for (int64_t z = blockIdx.z; z < outer_dim; z += gridDim.z){
    int64_t idx = blockIdx.x * blockDim.y + threadIdx.y;
    if (idx < numel
        && (idx == 0 || sorted_indices[idx] != sorted_indices[idx - 1])){
      do {
        int64_t start_feature = threadIdx.x + blockIdx.y * blockDim.x * SZ;
        // we only keep the last duplicate index so skip those before it
        if ((idx < numel - 1) && sorted_indices[idx] == sorted_indices[idx + 1]) {
          idx++;
          continue;
        }
        const int64_t weight_row = ((int64_t) sorted_indices[idx]) * stride + z * stride_before;
        const int64_t grad_row = ((int64_t) indices[idx]) * stride + z * numel * stride;
        const opmath_t scale = (opmath_t)1.0;

        opmath_t gradient[SZ];
        opmath_t weight[SZ];

        while (start_feature < stride) {
          #pragma unroll
          for (int ii = 0; ii < SZ; ii++) {
            int64_t feature_dim = start_feature + ii * C10_WARP_SIZE;
            if (feature_dim < stride) {
              gradient[ii] = static_cast<opmath_t>(grad_output[grad_row + feature_dim]);
            }
          }

          #pragma unroll
          for (int ii = 0; ii < SZ; ii++) {
            weight[ii] = gradient[ii] * scale;
          }

          #pragma unroll
          for (int ii = 0; ii < SZ; ii++) {
            int64_t feature_dim = start_feature + ii * C10_WARP_SIZE;
            if (feature_dim < stride) {
                // we do quantization here
                int64_t qvalue = static_cast<int64_t>(zero_point + nearbyintf(weight[ii]* inv_scale));
                qvalue = min(max(qvalue, qmin), qmax);
                grad_weight[weight_row + feature_dim] = static_cast<scalar_t>(qvalue);
            }
          }
          start_feature += gridDim.y * blockDim.x * SZ;
        }

        idx++;
      } while (idx < numel && sorted_indices[idx] == sorted_indices[idx - 1]);
    }
  }
}


}


namespace at::native {

namespace {

class ReduceMultiply {
public:
  template <typename scalar_t>
  constexpr C10_DEVICE void operator() (scalar_t* self_data_start, int64_t index, int64_t numel, const scalar_t * src_data) const {
    (void)numel; // suppress unused warning
    gpuAtomicMul(self_data_start + index, *src_data);
  }
};
static ReduceMultiply reduce_multiply;

class ReduceAdd {
public:
  template <typename scalar_t>
  constexpr C10_DEVICE void operator() (scalar_t* self_data_start, int64_t index, int64_t numel, const scalar_t * src_data) const {
    fastAtomicAdd(self_data_start, index, numel, *src_data, true);
  }
};
static ReduceAdd reduce_add;

class ReduceMinimum {
public:
  template <typename scalar_t>
  constexpr C10_DEVICE void operator() (scalar_t* self_data_start, int64_t index, int64_t numel, const scalar_t * src_data) const {
    (void)numel; // suppress unused warning
    gpuAtomicMin(self_data_start + index, *src_data);
  }
};
static ReduceMinimum reduce_minimum;

class ReduceMaximum {
public:
  template <typename scalar_t>
  constexpr C10_DEVICE void operator() (scalar_t* self_data_start, int64_t index, int64_t numel, const scalar_t * src_data) const {
    (void)numel; // suppress unused warning
    gpuAtomicMax(self_data_start + index, *src_data);
  }
};
static ReduceMaximum reduce_maximum;

}

static Tensor wrapIndexOnce(const Tensor & index, int64_t dim, int64_t dim_size, bool check_range=true) {
//we don't need to check range in backward - if there were out of bounds indices forward should already have errored out
  if (index.numel() != 0 && check_range) {
    auto max_idx = index.max().item<int64_t>();
    auto min_idx = index.min().item<int64_t>();
    if (max_idx >= dim_size) {
      TORCH_CHECK_INDEX(false, "index ", max_idx, " is out of bounds for dimension ", dim, " with size ", dim_size);
    }
    if (min_idx < -dim_size) {
      TORCH_CHECK_INDEX(false, "index ", min_idx, " is out of bounds for dimension ", dim, " with size ", dim_size);
    }
  }
  return index.remainder(dim_size);
}

static std::vector<int64_t> computeLinearStride(const Tensor & tensor) {
  // computes the stride as if tensor were contiguous
  auto sizes = tensor.sizes();
  std::vector<int64_t> stride(tensor.dim());
  if (stride.empty()) {
    return stride;
  }
  stride[tensor.dim() - 1] = 1;
  std::partial_sum(sizes.rbegin(), sizes.rend() - 1, stride.rbegin() + 1, std::multiplies<int64_t>());
  return stride;
}

static std::tuple<Tensor, int64_t, int64_t, int64_t>
computeLinearIndex(const Tensor & src, TensorList indices, bool check_range) {
  auto strides = computeLinearStride(src);
  const auto& device = src.options().device();

  // Compute the linear index by multiplying the indexing tensors by the
  // stride and summing them. All the indexing tensors have the same shape at
  // this point. We also compute the number of dimensions before and after that
  // are not being index.
  Tensor linearIndex;
  int64_t nElemBefore = 1, nElemAfter = 1, strideBefore =0;
  for (const auto i: c10::irange(src.dim())) {
    if (indices[i].defined()) {
      // Cast index to the longType matching src's device
      // This allows us to support ie indexing a cuda tensor with a cpu tensor
      Tensor index = (wrapIndexOnce(indices[i], i, src.size(i), check_range) * strides[i]).to(device);
      if (linearIndex.defined()) {
        linearIndex += index;
      } else {
        linearIndex = index;
        if (i>0) {
           strideBefore = src.stride(i-1); // stride after undefined dimensions
        }
      }
    } else if (linearIndex.defined()) {
      nElemAfter *= src.size(i);
    } else {
      nElemBefore *= src.size(i);
    }
  }

  return std::make_tuple(std::move(linearIndex), nElemBefore, strideBefore, nElemAfter);
}


static std::tuple<Tensor, Tensor, int64_t, int64_t, int64_t, std::vector<int64_t>> makeLinearIndex(Tensor self, IOptTensorListRef orig, bool check_range) {
  checkIndexTensorTypes(orig, /*allow_int*/true);
  // first expand BoolTensor (masks) or ByteTensor (masks) into 1 or more LongTensors
  auto indices = expandTensors(self, orig);
  for (auto & i : indices) {
    if (i.defined() && i.dtype() == at::kInt) {
      i = i.to(at::kLong);
    }
  }
  // next broadcast all index tensors together
  indices = expand_outplace(indices);
  // add missing null Tensors so that it matches self.dim()
  while (indices.size() < (size_t)self.dim()) {
    indices.emplace_back();
  }
  // if the non-null indices are not all adjacent, transpose self and indices
  // together so that they're adjacent at the front
  std::vector<int64_t> inversePerm;
  if (!hasContiguousSubspace(indices)) {
    std::tie(self, indices, inversePerm) = transposeToFrontAndInvPerm(self, indices);
  }
  auto [linearIndex, nElemBefore, strideBefore, nElemAfter] = computeLinearIndex(self, indices, check_range);
  return std::make_tuple(linearIndex, self, nElemBefore, strideBefore, nElemAfter, inversePerm);
}


void index_put_with_sort_kernel_thrust_helper(Tensor &linearIndex, Tensor &orig_indices, Tensor &sorted_indices, int64_t num_indices);

namespace {

int64_t largestIndex(const Tensor &self) {
  int64_t result = 0;
  for (const auto i: c10::irange(self.dim())) {
    result += (self.sizes()[i] - 1) * self.strides()[i];
  }
  return result;
}

void index_put_with_sort_kernel(Tensor & self, const c10::List<c10::optional<Tensor>>& indices, const Tensor & value, bool accumulate, bool unsafe) {
  TORCH_CHECK(!indices.empty() || is_expandable_to(value.sizes(), self.sizes()), "shape mismatch: value tensor of shape ", value.sizes(),
             " cannot be broadcast to indexing result of shape ", self.sizes());
  if (indices.size() > (size_t)self.dim()) {
    TORCH_CHECK_INDEX(false, "too many indices for tensor of dimension ", self.dim(), " (got ", indices.size(), ")");
  }
  bool self_contiguous = self.is_contiguous();
  auto self_ = self_contiguous ? self : self.contiguous();
  Tensor linearIndex, src, expandedValue = value;
  int64_t nElemBefore, strideBefore, sliceSize;
  std::vector<int64_t> inversePerm;
  std::tie(linearIndex, src, nElemBefore, strideBefore, sliceSize, inversePerm) = makeLinearIndex(self_, indices, !unsafe);
  int64_t num_indices = linearIndex.numel();

  if (expandedValue.numel() < num_indices * nElemBefore * sliceSize) {
    auto expanded_size = at::DimVector(expandedValue.sizes());
    auto size1 = expandedValue.sizes();
    auto size2 = linearIndex.sizes();
    if (are_expandable(size1, size2)) {
      expanded_size = infer_size_dimvector(size1, size2);
    }
    if (nElemBefore > 1) {
      expanded_size.insert(expanded_size.begin(), nElemBefore);
    }
    if (sliceSize > 1) {
      expanded_size.insert(expanded_size.end(), sliceSize);
    }
    expandedValue = expandedValue.expand(expanded_size);
  }
  expandedValue = expandedValue.contiguous();

  if (num_indices > 0 && sliceSize > 0) {
      const bool permuted = !src.is_contiguous();
      auto src_ = permuted ? src.contiguous() : src;
      linearIndex = linearIndex.reshape(-1);
      auto sorted_indices = at::empty_like(linearIndex, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
      auto orig_indices = at::empty_like(linearIndex, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
      const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

      linearIndex.divide_(sliceSize, "trunc");

      // cub on CUDA <= 11.2 have a bug that for small sizes
      // cub's sort can be much slower than thrust's merge sort
      // this bug is fixed in CUDA 11.3
#if (defined(CUDA_VERSION) && CUDA_VERSION < 11030) || defined(USE_ROCM)
      if (num_indices < 50000) {
        index_put_with_sort_kernel_thrust_helper(linearIndex, orig_indices, sorted_indices, num_indices);
      } else
#endif
      {
      // Sort the inputs into sorted with the corresponding indices
      auto range = at::arange(num_indices, linearIndex.options());
      // linearIndex can not be negative, and we take advantage of this
      // fact to sort on less bits for better performance.
      int64_t nbits = cuda::cub::get_num_bits(largestIndex(self_) / sliceSize);
      cuda::cub::radix_sort_pairs(
        linearIndex.const_data_ptr<int64_t>(), sorted_indices.mutable_data_ptr<int64_t>(),
        range.const_data_ptr<int64_t>(), orig_indices.mutable_data_ptr<int64_t>(),
        num_indices, false, 0, nbits);
      }

      TORCH_INTERNAL_ASSERT(
          linearIndex.numel()*sliceSize*nElemBefore == expandedValue.numel(),
          "number of flattened indices did not match number of elements in the value tensor: ",
          linearIndex.numel()*sliceSize*nElemBefore, " vs ", expandedValue.numel());
      const int UNROLL = 4;
      const int indices_per_block = 4;
      const int warp_size = at::cuda::warp_size();
      dim3 grid(ceil_div(num_indices, (int64_t) indices_per_block),
           std::min<int>(at::cuda::getCurrentDeviceProperties()->maxGridSize[1], ceil_div(sliceSize, (int64_t) (warp_size*UNROLL))),
           std::min(std::max<int>(1,nElemBefore), at::cuda::getCurrentDeviceProperties()->maxGridSize[2]));
      dim3 block(warp_size, indices_per_block);


      if (sliceSize == 1) {
        // This implementation is faster with high amounts of duplicates but could overflow
        // if FP16 / BF16 is used
        AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND4(kComplexHalf, kHalf, kBool, kBFloat16,
        expandedValue.scalar_type(), "indexing_backward_kernel_stride_1", [&] {
          indexing_backward_kernel_stride_1<scalar_t><<<grid, block, 0, stream>>>(
            sorted_indices.const_data_ptr<int64_t>(),
            orig_indices.const_data_ptr<int64_t>(),
            expandedValue.const_data_ptr<scalar_t>(),
            src_.mutable_data_ptr<scalar_t>(),
            num_indices,
            sliceSize,
            strideBefore,
            nElemBefore,
            accumulate);
          C10_CUDA_KERNEL_LAUNCH_CHECK();
        });
      } else {
        if (sliceSize <= warp_size) {
          AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND4(kComplexHalf, kHalf, kBool, kBFloat16,
          expandedValue.scalar_type(), "indexing_backward_kernel_small_stride", [&] {
            indexing_backward_kernel_small_stride<scalar_t><<<grid, block, 0, stream>>>(
              sorted_indices.const_data_ptr<int64_t>(),
              orig_indices.const_data_ptr<int64_t>(),
              expandedValue.const_data_ptr<scalar_t>(),
              src_.mutable_data_ptr<scalar_t>(),
              num_indices,
              sliceSize,
              strideBefore,
              nElemBefore,
              accumulate);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            });
        } else {
            AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND4(kComplexHalf, kHalf, kBool, kBFloat16,
            expandedValue.scalar_type(), "indexing_backward", [&] {
              indexing_backward_kernel<scalar_t, UNROLL><<<grid, block, 0, stream>>>(
                sorted_indices.const_data_ptr<int64_t>(),
                orig_indices.const_data_ptr<int64_t>(),
                expandedValue.const_data_ptr<scalar_t>(),
                src_.mutable_data_ptr<scalar_t>(),
                num_indices,
                sliceSize,
                strideBefore,
                nElemBefore,
                accumulate);
              C10_CUDA_KERNEL_LAUNCH_CHECK();
            });
          }
        }

      if (permuted) {
        self.copy_(src_.permute(inversePerm));
      } else if (!self_contiguous) {
        self.copy_(self_);
      }
  }
}

REGISTER_CUDA_DISPATCH(index_put_with_sort_stub, &index_put_with_sort_kernel);

void index_put_with_sort_quantized(Tensor & self, const c10::List<c10::optional<Tensor>>& indices, const Tensor & value, double scale, int zero_point, bool unsafe) {
  if (indices.size() > (size_t)self.dim()) {
    TORCH_CHECK_INDEX(false, "too many indices for tensor of dimension ", self.dim(), " (got ", indices.size(), ")");
  }
  bool self_contiguous = self.is_contiguous();
  auto self_ = self_contiguous ? self : self.contiguous();
  Tensor linearIndex, src, expandedValue = value;
  int64_t nElemBefore, strideBefore, sliceSize;
  std::vector<int64_t> inversePerm;
  std::tie(linearIndex, src, nElemBefore, strideBefore, sliceSize, inversePerm) = makeLinearIndex(self_, indices, !unsafe);
  int64_t num_indices = linearIndex.numel();

  if (expandedValue.numel() < num_indices * nElemBefore * sliceSize) {
    auto expanded_size = at::DimVector(expandedValue.sizes());
    auto size1 = expandedValue.sizes();
    auto size2 = linearIndex.sizes();
    if (are_expandable(size1, size2)) {
      expanded_size = infer_size_dimvector(size1, size2);
    }
    if (nElemBefore > 1) {
      expanded_size.insert(expanded_size.begin(), nElemBefore);
    }
    expandedValue = expandedValue.expand(expanded_size);
  }
  expandedValue = expandedValue.contiguous();

  if (num_indices > 0 && sliceSize > 0) {
      const bool permuted = !src.is_contiguous();
      auto src_ = permuted ? src.contiguous() : src;
      linearIndex = linearIndex.reshape(-1);
      auto sorted_indices = at::empty_like(linearIndex, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
      auto orig_indices = at::empty_like(linearIndex, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
      const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

      linearIndex.divide_(sliceSize, "trunc");

      // cub on CUDA <= 11.2 have a bug that for small sizes
      // cub's sort can be much slower than thrust's merge sort
      // this bug is fixed in CUDA 11.3
#if (defined(CUDA_VERSION) && CUDA_VERSION < 11030) || defined(USE_ROCM)
      if (num_indices < 50000) {
        index_put_with_sort_kernel_thrust_helper(linearIndex, orig_indices, sorted_indices, num_indices);
      } else
#endif
      {
      // Sort the inputs into sorted with the corresponding indices
      auto range = at::arange(num_indices, linearIndex.options());
      // linearIndex can not be negative, and we take advantage of this
      // fact to sort on less bits for better performance.
      int64_t nbits = cuda::cub::get_num_bits(largestIndex(self_) / sliceSize);
      cuda::cub::radix_sort_pairs(
        linearIndex.const_data_ptr<int64_t>(), sorted_indices.mutable_data_ptr<int64_t>(),
        range.const_data_ptr<int64_t>(), orig_indices.mutable_data_ptr<int64_t>(),
        num_indices, false, 0, nbits);
      }

      TORCH_INTERNAL_ASSERT(
          linearIndex.numel()*sliceSize*nElemBefore == expandedValue.numel(),
          "number of flattened indices did not match number of elements in the value tensor: ",
          linearIndex.numel()*sliceSize*nElemBefore, " vs ", expandedValue.numel());
      const int UNROLL = 4;
      const int indices_per_block = 4;
      const int warp_size = at::cuda::warp_size();
      dim3 grid(ceil_div(num_indices, (int64_t) indices_per_block),
           std::min<int>(at::cuda::getCurrentDeviceProperties()->maxGridSize[1], ceil_div(sliceSize, (int64_t) (warp_size*UNROLL))),
           std::min(std::max<int>(1,nElemBefore), at::cuda::getCurrentDeviceProperties()->maxGridSize[2]));
      dim3 block(warp_size, indices_per_block);

      AT_DISPATCH_QINT_TYPES(
        src.scalar_type(), "indexing_backward_quantized", [&] {
        constexpr int64_t qmin = std::numeric_limits<typename scalar_t::underlying>::min();
        constexpr int64_t qmax = std::numeric_limits<typename scalar_t::underlying>::max();
        float inv_scale = 1.0f / static_cast<float>(scale);

        indexing_backward_kernel_quantized<scalar_t, UNROLL><<<grid, block, 0, stream>>>(
          sorted_indices.const_data_ptr<int64_t>(),
          orig_indices.const_data_ptr<int64_t>(),
          expandedValue.const_data_ptr<float>(),
          src_.mutable_data_ptr<scalar_t>(),
          num_indices,
          sliceSize,
          strideBefore,
          nElemBefore,
          inv_scale,
          zero_point,
          qmin,
          qmax);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });

      if (permuted) {
        self.copy_(src_.permute(inversePerm));
      } else if (!self_contiguous) {
        self.copy_(self_);
      }
  }
}

REGISTER_CUDA_DISPATCH(index_put_with_sort_quantized_stub, &index_put_with_sort_quantized);
} //anonymous


// Check tensor dimensions for index operations, and return the slice size.
static ptrdiff_t getSliceSize(const Tensor & dst,
                              int dim,
                              const Tensor & index,
                              const Tensor & src)
{
  const auto dstDims = dst.dim();
  const auto srcDims = src.dim();

  TORCH_CHECK(index.dim() <= 1, "Index must be vector or scalar");

  ptrdiff_t dstSliceSize = 1;
  TORCH_CHECK(dim >= 0 && dim < dstDims, "Indexing dim ", dim, " is out of bounds");
  for (const auto d: c10::irange(dstDims)) {
    if (d != dim) {
      dstSliceSize *= dst.size(d);
    }
  }

  TORCH_CHECK(dim < srcDims, "Indexing dim ", dim, " is out of bounds");
  TORCH_CHECK(index.numel() == src.size(dim),
             "length of src.size[dim] is not equal to length of indices");

  ptrdiff_t srcSliceSize = 1;
  bool mismatch = false;

  if (dstDims != srcDims) mismatch = true;

  for (const auto d: c10::irange(srcDims)) {
    if (d != dim) {
      srcSliceSize *= src.size(d);
      if (!mismatch && dst.size(d) != src.size(d)) mismatch = true;
    }
  }

  TORCH_CHECK(dstSliceSize == srcSliceSize,
             "Source/destination tensor have different slice sizes (%ld vs %ld)",
             dstSliceSize, srcSliceSize);

  if (mismatch) {
    TORCH_WARN_ONCE(
        "Warning: source/destination slices have same size but different "
        "shape for an index operation.  This behavior is deprecated.\n");
  }

  return dstSliceSize;
}

// We prefer this kernel to avoid reloading index points if the number
// of indices is a small number.
// This kernel in fact works for all choices of problem size, but if
// the number of indices chosen is large, then the
// indexFuncLargeIndex kernel is a better choice to increase
// parallelism.
template <typename T, typename IndicesType, typename IndexType, int DstDim, int SrcDim, int IdxDim,
          typename func_t>
__global__ void indexFuncSmallIndex(cuda::detail::TensorInfo<T, IndexType> dst,
                                    cuda::detail::TensorInfo<const T, IndexType> src,
                                    cuda::detail::TensorInfo<const IndicesType, IndexType> indices,
                                    int dstAddDim,
                                    int srcAddDim,
                                    IndexType innerSize,
                                    int64_t dstAddDimSize,
                                    int64_t dstNumel,
                                    const func_t& op,
                                    T alpha) {
  // In order to avoid reloading the index that we are copying, load
  // it once to handle all of the points that are being selected, so
  // it can be reused as much as possible. This kernel is chosen when
  // this is a good choice (small number of chosen indices), since
  // re-accessing indices in addition to src elements can be slow.
  for (IndexType srcIndex = 0; srcIndex < indices.sizes[0]; ++srcIndex) {
    // Lua indices begin at 1
    IndexType dstIndex =
        indices.data[cuda::detail::IndexToOffset<const IndicesType, IndexType, IdxDim>::get(srcIndex, indices)];
    CUDA_KERNEL_ASSERT(dstIndex < dstAddDimSize);

    // We stride over the output ignoring the indexed dimension
    // (innerSize), whose offset calculation is handled differently
    for (IndexType linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
         linearIndex < innerSize;
         linearIndex += gridDim.x * blockDim.x) {
      IndexType dstOffset =
          cuda::detail::IndexToOffset<T, IndexType, DstDim>::get(linearIndex, dst);
      dstOffset += dstIndex * dst.strides[dstAddDim];

      IndexType srcOffset =
          cuda::detail::IndexToOffset<const T, IndexType, SrcDim>::get(linearIndex, src);
      srcOffset += srcIndex * src.strides[srcAddDim];

      T val = src.data[srcOffset] * alpha;
      op(dst.data, dstOffset, dstNumel, &val);
    }

  }
}

// We prefer this kernel to balance parallelism across index points,
// if there are a large number of indices.
// This kernel in fact works for all choices of problem size, but if
// the number of indices chosen is small, then the
// indexFuncSmallIndex kernel is a better choice to reduce memory
// accesses.
template <typename T, typename IndicesType, typename IndexType, int DstDim, int SrcDim, int IdxDim,
          bool IndexIsMajor, typename func_t>
__global__ void indexFuncLargeIndex(cuda::detail::TensorInfo<T, IndexType> dst,
                                    cuda::detail::TensorInfo<const T, IndexType> src,
                                    cuda::detail::TensorInfo<const IndicesType, IndexType> indices,
                                    int dstAddDim,
                                    int srcAddDim,
                                    IndexType totalSize,
                                    IndexType innerSize,
                                    int64_t dstAddDimSize,
                                    int64_t dstNumel,
                                    const func_t& op,
                                    T alpha) {
  // We stride over the output including the indexed dimension
  // (totalSize), and calculate the destination index point based on that
  for (IndexType linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
       linearIndex < totalSize;
       linearIndex += gridDim.x * blockDim.x) {
    IndexType srcIndex, elementInSlice;
    if (IndexIsMajor) {
      srcIndex = linearIndex / innerSize;
      elementInSlice = linearIndex % innerSize;
    }
    else {
      elementInSlice = linearIndex / innerSize;
      srcIndex = linearIndex % innerSize;
    }

    // Lua indices begin at 1
    IndexType dstIndex =
        indices.data[cuda::detail::IndexToOffset<const IndicesType, IndexType, IdxDim>::get(srcIndex, indices)];
    CUDA_KERNEL_ASSERT(dstIndex < dstAddDimSize);

    IndexType dstOffset =
      cuda::detail::IndexToOffset<T, IndexType, DstDim>::get(elementInSlice, dst);
    dstOffset += dstIndex * dst.strides[dstAddDim];

    IndexType srcOffset =
      cuda::detail::IndexToOffset<const T, IndexType, SrcDim>::get(elementInSlice, src);
    srcOffset += srcIndex * src.strides[srcAddDim];

    T val = src.data[srcOffset] * alpha;
    op(dst.data, dstOffset, dstNumel, &val);
  }
}

// Compare the stride between adjacent slices (sliceStride) with strides in the
// other dimensions (i.e., strides *inside* each slice).
//
// - Returns true if some dimension inside the slice has lower stride than
//   sliceStride.  The simplest example is a 2-D contiguous tensor with sliceDim
//   == 0 (that is, each slice is a row).
//
//   In this case, we choose the CUDA kernel that processes the data in
//   "index-major order".  For example, if thread count equals slice size, then
//   all threads process slice #0 in lockstep, and then slice #1, and so on.
//
// - Otherwise (i.e., sliceStride has the lowest value), this function returns
//   false.  The simplest example is a 2-D contiguous tensor with sliceDim == 1
//   (each slice is a column).
//
//   In this case, we choose the CUDA kernel that processes the data in
//   "elementInSlice-major order".  For example, each thread can process element
//   #0 of every slice, and then element #1 of every slice, and so on.
template <typename scalar_t>
bool indexShouldBeMajor(cuda::detail::TensorInfo<scalar_t, unsigned int> &info,
                                    int sliceDim)
{
  // The stride between adjacent slices (e.g., between element #0 of slice #100
  // and element #0 of slice #101).
  unsigned int sliceStride = info.strides[sliceDim];

  for (const auto i: c10::irange(info.dims)) {
    if (i != sliceDim && info.sizes[i] > 1 && info.strides[i] < sliceStride) {
      return true;
    }
  }

  return false;
}

void index_add_cuda_impl(const Tensor& self, int64_t dim, const Tensor& index, const Tensor& source, const Scalar& alpha, const Tensor& result) {
  if (!result.is_same(self)) {
    result.copy_(self);
  }

  // Scalars are treated as 1-d tensor
  const Tensor self_ = (result.dim() == 0) ? result.view(1) : result;
  const Tensor source_ = (source.dim() == 0) ? source.view(1) : source;

  TORCH_CHECK(result.dim() <= MAX_TENSORINFO_DIMS, "tensor has too many (>", MAX_TENSORINFO_DIMS, ") dims");
  TORCH_CHECK(source.dim() <= MAX_TENSORINFO_DIMS, "tensor has too many (>", MAX_TENSORINFO_DIMS, ") dims" );
  TORCH_CHECK(index.dim() <= MAX_TENSORINFO_DIMS, "tensor has too many (>", MAX_TENSORINFO_DIMS, ") dims");

  if (globalContext().deterministicAlgorithms()){
    torch::List<c10::optional<Tensor>> indices;
    indices.reserve(dim + 1);
    for (const auto i: c10::irange(dim)) {
      indices.emplace_back();
    }
    indices.emplace_back(index.to(at::kLong));
    result.index_put_(indices, source * alpha, true);
    return;
  }

  // The `source` is partitioned into two parts:
  // -the size of each slice we are indexing, which is the
  // total size of the tensor ignoring dimension `dim`;
  // -the number of index we are choosing, which is the total size
  // of the tensor `index`.
  const ptrdiff_t sliceSize = getSliceSize(self_, dim, index, source_);
  const ptrdiff_t sourceTotalSize = source.numel();
  const int64_t selfAddDimSize = self_.size(dim);
  const ptrdiff_t numIndex = index.numel();
  const int64_t selfNumel = self_.numel();

  if (sliceSize == 0) {
    return;
  }
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const bool indContig = index.is_contiguous();

  const int mpc = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;

#define SMALL_INDEX(TENSOR_TYPE, INDICES_TYPE, TYPE, SELF_DIM, SOURCE_DIM, IDX_DIM)     \
  indexFuncSmallIndex<TENSOR_TYPE, INDICES_TYPE, TYPE, SELF_DIM, SOURCE_DIM, IDX_DIM>   \
    <<<smallIndexGrid, smallIndexBlock, 0, stream>>>(                                   \
      selfInfo, sourceInfo, indexInfo,                                                  \
      selfAddDim, sourceAddDim, sliceSize, selfAddDimSize,                              \
      selfNumel, reduce_add, alpha_value);                                              \
  C10_CUDA_KERNEL_LAUNCH_CHECK();

#define LARGE_INDEX(TENSOR_TYPE, INDICES_TYPE, TYPE,                        \
                    SELF_DIM, SOURCE_DIM, IDX_DIM, IDX_IS_MAJOR)            \
  indexFuncLargeIndex<TENSOR_TYPE, INDICES_TYPE, TYPE,                      \
                      SELF_DIM, SOURCE_DIM, IDX_DIM, IDX_IS_MAJOR>          \
    <<<largeIndexGrid, largeIndexBlock, 0, stream>>>(                       \
      selfInfo, sourceInfo, indexInfo,                                      \
      selfAddDim, sourceAddDim, sourceTotalSize,                            \
      (IDX_IS_MAJOR) ? sliceSize : numIndex,                                \
      selfAddDimSize, selfNumel, reduce_add, alpha_value);                  \
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  const dim3 smallIndexGrid(std::min(ceil_div(sliceSize, (ptrdiff_t)128), (ptrdiff_t)(mpc * 8)));
  const dim3 smallIndexBlock(std::min(sliceSize, (ptrdiff_t)128));

  const dim3 largeIndexGrid(std::min(ceil_div(sourceTotalSize, (ptrdiff_t)128), (ptrdiff_t)(mpc * 8)));
  const dim3 largeIndexBlock(std::min(sourceTotalSize, (ptrdiff_t)128));

  if (cuda::detail::canUse32BitIndexMath(result) &&
      cuda::detail::canUse32BitIndexMath(source) &&
      cuda::detail::canUse32BitIndexMath(index)) {
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND4(at::ScalarType::Bool, at::ScalarType::Half, at::ScalarType::BFloat16, at::ScalarType::ComplexHalf, result.scalar_type(), "index_add", [&] {
      cuda::detail::TensorInfo<scalar_t, unsigned int> selfInfo =
          cuda::detail::getTensorInfo<scalar_t, unsigned int>(self_);
      const int selfAddDim = selfInfo.collapseDims(dim);
      selfInfo.reduceDim(selfAddDim);
      const auto alpha_value = alpha.to<scalar_t>();
      AT_DISPATCH_INDEX_TYPES(index.scalar_type(), "index_add_cuda_", [&] () {
        auto sourceInfo =
          cuda::detail::getTensorInfo<const scalar_t, unsigned int>(source_);
        const int sourceAddDim = sourceInfo.collapseDims(dim);
        sourceInfo.reduceDim(sourceAddDim);

        auto indexInfo =
        cuda::detail::getTensorInfo<const index_t, unsigned int>(index);
        indexInfo.collapseDims();

        // A reasonable choice for when to have each thread iterate over
        // index to choose
        if (numIndex <= 16) {
          if (selfInfo.dims == 1 && sourceInfo.dims == 1 && indContig) {
            SMALL_INDEX(scalar_t, index_t, unsigned int, 1, 1, -2);
          } else if (selfInfo.dims == 2 && sourceInfo.dims == 2 && indContig) {
            SMALL_INDEX(scalar_t, index_t, unsigned int, 2, 2, -2);
          } else if (selfInfo.dims == 3 && sourceInfo.dims == 3 && indContig) {
            SMALL_INDEX(scalar_t, index_t, unsigned int, 3, 3, -2);
          } else {
            SMALL_INDEX(scalar_t, index_t, unsigned int, -1, -1, -1);
          }
        } else {
          const bool indexIsMajor = indexShouldBeMajor(selfInfo, selfAddDim);

          if (selfInfo.dims == 1 && sourceInfo.dims == 1 && indContig) {
            LARGE_INDEX(scalar_t, index_t, unsigned int, 1, 1, -2, true);
          } else if (selfInfo.dims == 2 && sourceInfo.dims == 2 && indContig) {
            if (indexIsMajor) {
              LARGE_INDEX(scalar_t, index_t, unsigned int, 2, 2, -2, true);
            } else {
              LARGE_INDEX(scalar_t, index_t, unsigned int, 2, 2, -2, false);
            }
          } else if (selfInfo.dims == 3 && sourceInfo.dims == 3 && indContig) {
            if (indexIsMajor) {
              LARGE_INDEX(scalar_t, index_t, unsigned int, 3, 3, -2, true);
            } else {
              LARGE_INDEX(scalar_t, index_t, unsigned int, 3, 3, -2, false);
            }
          } else {
            LARGE_INDEX(scalar_t, index_t, unsigned int, -1, -1, -1, true);
          }
        }
      });
    });
  } else {
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(at::ScalarType::Bool, at::ScalarType::Half, at::ScalarType::BFloat16, self.scalar_type(), "index_add", [&] {
      cuda::detail::TensorInfo<scalar_t, uint64_t> selfInfo =
        cuda::detail::getTensorInfo<scalar_t, uint64_t>(self_);
      const int selfAddDim = selfInfo.collapseDims(dim);
      selfInfo.reduceDim(selfAddDim);
      const auto alpha_value = alpha.to<scalar_t>();

      cuda::detail::TensorInfo<const scalar_t, uint64_t> sourceInfo =
        cuda::detail::getTensorInfo<const scalar_t, uint64_t>(source_);
      const int sourceAddDim = sourceInfo.collapseDims(dim);
      sourceInfo.reduceDim(sourceAddDim);

      AT_DISPATCH_INDEX_TYPES(index.scalar_type(), "index_add_cuda_", [&] () {
        cuda::detail::TensorInfo<const index_t, uint64_t> indexInfo =
          cuda::detail::getTensorInfo<const index_t, uint64_t>(index);
        indexInfo.collapseDims();

        LARGE_INDEX(scalar_t, index_t, uint64_t, -1, -1, -1, true);
      });
    });
  }

#undef SMALL_INDEX
#undef LARGE_INDEX
}

template <typename func_t>
void index_reduce_func_cuda_impl(
  const Tensor& self,
  int64_t dim,
  const Tensor& index,
  const Tensor& source,
  bool include_self,
  const ReductionType& reduce,
  const func_t& reduce_func,
  const Tensor& result) {
  globalContext().alertNotDeterministic("index_reduce_cuda");

  if (!result.is_same(self)) result.copy_(self);

  // Scalars are treated as 1-d tensor
  Tensor self_ = (result.dim() == 0) ? result.view(1) : result;
  Tensor source_ = (source.dim() == 0) ? source.view(1) : source;

  TORCH_CHECK(result.dim() <= MAX_TENSORINFO_DIMS, "tensor has too many (>", MAX_TENSORINFO_DIMS, ") dims");
  TORCH_CHECK(source.dim() <= MAX_TENSORINFO_DIMS, "tensor has too many (>", MAX_TENSORINFO_DIMS, ") dims" );
  TORCH_CHECK(index.dim() <= MAX_TENSORINFO_DIMS, "tensor has too many (>", MAX_TENSORINFO_DIMS, ") dims");

  if (!include_self) {
    AT_DISPATCH_ALL_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16,
      self.scalar_type(), "index_reduce_func_cuda_exclude_input_init", [&] {
      scalar_t init_val;
      switch (reduce) {
        case ReductionType::PROD:
          init_val = (scalar_t)1;
          break;
        case ReductionType::MAX:
          init_val = std::numeric_limits<scalar_t>::has_infinity ? -std::numeric_limits<scalar_t>::infinity()
                     : std::numeric_limits<scalar_t>::lowest();
          break;
        case ReductionType::MIN:
          init_val = std::numeric_limits<scalar_t>::has_infinity ? std::numeric_limits<scalar_t>::infinity()
                     : std::numeric_limits<scalar_t>::max();
          break;
        default:
          init_val = (scalar_t)0;
          break;
      }
      // index_fill_ requires index to be a LongTensor
      self_.index_fill_(dim, index.to(at::ScalarType::Long), init_val);
    });
  }

  // The `source` is partitioned into two parts:
  // -the size of each slice we are indexing, which is the
  // total size of the tensor ignoring dimension `dim`;
  // -the number of index we are choosing, which is the total size
  // of the tensor `index`.
  ptrdiff_t sliceSize = getSliceSize(self_, dim, index, source_);
  ptrdiff_t sourceTotalSize = source.numel();
  int64_t selfReduceDimSize = self_.size(dim);
  ptrdiff_t numIndex = index.numel();
  int64_t selfNumel = self_.numel();

  if (sliceSize == 0) {
    return;
  }
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  bool indContig = index.is_contiguous();

  int mpc = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;

#define SMALL_INDEX(TENSOR_TYPE, INDICES_TYPE, TYPE, SELF_DIM, SOURCE_DIM, IDX_DIM)                  \
  indexFuncSmallIndex<TENSOR_TYPE, INDICES_TYPE, TYPE, SELF_DIM, SOURCE_DIM, IDX_DIM>                \
    <<<smallIndexGrid, smallIndexBlock, 0, stream>>>(                                                \
      selfInfo, sourceInfo, indexInfo,                                                               \
      selfReduceDim, sourceReduceDim, sliceSize, selfReduceDimSize,                                  \
      selfNumel, reduce_func, alpha_value);                                                          \
  C10_CUDA_KERNEL_LAUNCH_CHECK();

#define LARGE_INDEX(TENSOR_TYPE, INDICES_TYPE, TYPE,                                     \
                    SELF_DIM, SOURCE_DIM, IDX_DIM, IDX_IS_MAJOR)                         \
  indexFuncLargeIndex<TENSOR_TYPE, INDICES_TYPE, TYPE,                                   \
                     SELF_DIM, SOURCE_DIM, IDX_DIM, IDX_IS_MAJOR>                        \
    <<<largeIndexGrid, largeIndexBlock, 0, stream>>>(                                    \
      selfInfo, sourceInfo, indexInfo,                                                   \
      selfReduceDim, sourceReduceDim, sourceTotalSize,                                   \
      (IDX_IS_MAJOR) ? sliceSize : numIndex,                                             \
      selfReduceDimSize, selfNumel, reduce_func, alpha_value);                           \
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  dim3 smallIndexGrid(std::min(ceil_div(sliceSize, (ptrdiff_t)128), (ptrdiff_t)(mpc * 8)));
  dim3 smallIndexBlock(std::min(sliceSize, (ptrdiff_t)128));

  dim3 largeIndexGrid(std::min(ceil_div(sourceTotalSize, (ptrdiff_t)128), (ptrdiff_t)(mpc * 8)));
  dim3 largeIndexBlock(std::min(sourceTotalSize, (ptrdiff_t)128));

  if (cuda::detail::canUse32BitIndexMath(result) &&
      cuda::detail::canUse32BitIndexMath(source) &&
      cuda::detail::canUse32BitIndexMath(index)) {
    AT_DISPATCH_ALL_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, result.scalar_type(), "index_reduce", [&] {
      cuda::detail::TensorInfo<scalar_t, unsigned int> selfInfo =
          cuda::detail::getTensorInfo<scalar_t, unsigned int>(self_);
      int selfReduceDim = selfInfo.collapseDims(dim);
      selfInfo.reduceDim(selfReduceDim);
      auto alpha_value = (scalar_t) 1;
      AT_DISPATCH_INDEX_TYPES(index.scalar_type(), "index_reduce_cuda", [&] () {
        auto sourceInfo =
          cuda::detail::getTensorInfo<const scalar_t, unsigned int>(source_);
        int sourceReduceDim = sourceInfo.collapseDims(dim);
        sourceInfo.reduceDim(sourceReduceDim);

        auto indexInfo =
        cuda::detail::getTensorInfo<const index_t, unsigned int>(index);
        indexInfo.collapseDims();

        // A reasonable choice for when to have each thread iterate over
        // index to choose
        if (numIndex <= 16) {
          if (selfInfo.dims == 1 && sourceInfo.dims == 1 && indContig) {
            SMALL_INDEX(scalar_t, index_t, unsigned int, 1, 1, -2);
          } else if (selfInfo.dims == 2 && sourceInfo.dims == 2 && indContig) {
            SMALL_INDEX(scalar_t, index_t, unsigned int, 2, 2, -2);
          } else if (selfInfo.dims == 3 && sourceInfo.dims == 3 && indContig) {
            SMALL_INDEX(scalar_t, index_t, unsigned int, 3, 3, -2);
          } else {
            SMALL_INDEX(scalar_t, index_t, unsigned int, -1, -1, -1);
          }
        } else {
          bool indexIsMajor = indexShouldBeMajor(selfInfo, selfReduceDim);

          if (selfInfo.dims == 1 && sourceInfo.dims == 1 && indContig) {
            LARGE_INDEX(scalar_t, index_t, unsigned int, 1, 1, -2, true);
          } else if (selfInfo.dims == 2 && sourceInfo.dims == 2 && indContig) {
            if (indexIsMajor) {
              LARGE_INDEX(scalar_t, index_t, unsigned int, 2, 2, -2, true);
            } else {
              LARGE_INDEX(scalar_t, index_t, unsigned int, 2, 2, -2, false);
            }
          } else if (selfInfo.dims == 3 && sourceInfo.dims == 3 && indContig) {
            if (indexIsMajor) {
              LARGE_INDEX(scalar_t, index_t, unsigned int, 3, 3, -2, true);
            } else {
              LARGE_INDEX(scalar_t, index_t, unsigned int, 3, 3, -2, false);
            }
          } else {
            LARGE_INDEX(scalar_t, index_t, unsigned int, -1, -1, -1, true);
          }
        }
      });
    });
  } else {
    AT_DISPATCH_ALL_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, self.scalar_type(), "index_reduce", [&] {
      cuda::detail::TensorInfo<scalar_t, uint64_t> selfInfo =
        cuda::detail::getTensorInfo<scalar_t, uint64_t>(self_);
      int selfReduceDim = selfInfo.collapseDims(dim);
      selfInfo.reduceDim(selfReduceDim);
      auto alpha_value = (scalar_t) 1;

      cuda::detail::TensorInfo<const scalar_t, uint64_t> sourceInfo =
        cuda::detail::getTensorInfo<const scalar_t, uint64_t>(source_);
      int sourceReduceDim = sourceInfo.collapseDims(dim);
      sourceInfo.reduceDim(sourceReduceDim);

      AT_DISPATCH_INDEX_TYPES(index.scalar_type(), "index_reduce_cuda", [&] () {
        cuda::detail::TensorInfo<const index_t, uint64_t> indexInfo =
          cuda::detail::getTensorInfo<const index_t, uint64_t>(index);
        indexInfo.collapseDims();

        LARGE_INDEX(scalar_t, index_t, uint64_t, -1, -1, -1, true);
      });
    });
  }

#undef SMALL_INDEX
#undef LARGE_INDEX
}

TORCH_IMPL_FUNC(index_add_cuda_out)
(const Tensor& self, int64_t dim, const Tensor& index, const Tensor& source, const Scalar& alpha, const Tensor& result) {
  index_add_cuda_impl(self, dim, index, source, alpha, result);
}

TORCH_IMPL_FUNC(index_reduce_cuda_out)
(const Tensor& self,
 int64_t dim,
 const Tensor& index,
 const Tensor& source,
 const c10::string_view reduce,
 bool include_self,
 const Tensor& result) {
  TORCH_WARN_ONCE("index_reduce() is in beta and the API may change at any time.");

  if (reduce == "prod") {
    index_reduce_func_cuda_impl(self, dim, index, source, include_self, ReductionType::PROD, reduce_multiply, result);
  } else if (reduce == "mean") {
    index_reduce_func_cuda_impl(self, dim, index, source, include_self, ReductionType::MEAN, reduce_add, result);
    auto counts = include_self ? at::ones_like(result) : at::zeros_like(result);
    counts.index_add_(dim, index, at::ones_like(source));
    counts.masked_fill_(counts == 0, 1);
    if (result.is_floating_point() || result.is_complex()) {
      result.div_(counts);
    } else {
      result.div_(counts, "floor");
    }
  } else if (reduce == "amax") {
    index_reduce_func_cuda_impl(self, dim, index, source, include_self, ReductionType::MAX, reduce_maximum, result);
  } else if (reduce == "amin") {
    index_reduce_func_cuda_impl(self, dim, index, source, include_self, ReductionType::MIN, reduce_minimum, result);
  } else {
    TORCH_CHECK(false, "reduce argument must be either prod, mean, amax or amin, got ", reduce, ".");
  }
}

namespace {
// We prefer this kernel to avoid reloading index points if the number
// of indices is a small number.
// This kernel in fact works for all choices of problem size, but if
// the number of indices chosen is large, then the
// indexSelectLargeIndex kernel is a better choice to increase
// parallelism.
template <typename T, typename IndicesType, typename IndexType, int DstDim, int SrcDim, int IdxDim>
__global__ void indexSelectSmallIndex(cuda::detail::TensorInfo<T, IndexType> dst,
                                      cuda::detail::TensorInfo<const T, IndexType> src,
                                      cuda::detail::TensorInfo<const IndicesType, IndexType> indices,
                                      int dstSelectDim,
                                      int srcSelectDim,
                                      IndexType innerSize,
                                      int64_t srcSelectDimSize) {
  // In order to avoid reloading the index that we are copying, load
  // it once to handle all of the points that are being selected, so
  // it can be reused as much as possible. This kernel is chosen when
  // this is a good choice (small number of chosen indices), since
  // re-accessing indices in addition to src elements can be slow.
  for (IndexType dstIndex = 0; dstIndex < indices.sizes[0]; ++dstIndex) {
    IndexType srcIndex =
      indices.data[cuda::detail::IndexToOffset<const IndicesType, IndexType, IdxDim>::get(dstIndex, indices)];
    CUDA_KERNEL_ASSERT(srcIndex < srcSelectDimSize);

    // We stride over the output ignoring the indexed dimension
    // (innerSize), whose offset calculation is handled differently
    for (IndexType linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
         linearIndex < innerSize;
         linearIndex += gridDim.x * blockDim.x) {
      IndexType dstOffset =
        cuda::detail::IndexToOffset<T, IndexType, DstDim>::get(linearIndex, dst);
      dstOffset += dstIndex * dst.strides[dstSelectDim];

      IndexType srcOffset =
        cuda::detail::IndexToOffset<const T, IndexType, SrcDim>::get(linearIndex, src);
      srcOffset += srcIndex * src.strides[srcSelectDim];

      dst.data[dstOffset] = src.data[srcOffset];
    }
  }
}

// We prefer this kernel to balance parallelism across index points,
// if there are a large number of indices.
// This kernel in fact works for all choices of problem size, but if
// the number of indices chosen is small, then the
// indexSelectSmallIndex kernel is a better choice to reduce memory
// accesses.
template <typename T, typename IndicesType, typename IndexType, int DstDim, int SrcDim, int IdxDim,
          bool IndexIsMajor>
__global__ void indexSelectLargeIndex(cuda::detail::TensorInfo<T, IndexType> dst,
                                      cuda::detail::TensorInfo<const T, IndexType> src,
                                      cuda::detail::TensorInfo<const IndicesType, IndexType> indices,
                                      int dstSelectDim,
                                      int srcSelectDim,
                                      IndexType totalSize,
                                      IndexType innerSize,
                                      int64_t srcSelectDimSize) {
  // We stride over the output including the indexed dimension
  // (totalSize), and calculate the destination index point based on that
  for (IndexType linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
       linearIndex < totalSize;
       linearIndex += gridDim.x * blockDim.x) {
    IndexType dstIndex, elementInSlice;
    if (IndexIsMajor) {
      dstIndex = linearIndex / innerSize;
      elementInSlice = linearIndex % innerSize;
    }
    else {
      elementInSlice = linearIndex / innerSize;
      dstIndex = linearIndex % innerSize;
    }

    IndexType srcIndex =
      indices.data[cuda::detail::IndexToOffset<const IndicesType, IndexType, IdxDim>::get(dstIndex, indices)];
    CUDA_KERNEL_ASSERT(srcIndex < srcSelectDimSize);

    IndexType dstOffset =
      cuda::detail::IndexToOffset<T, IndexType, DstDim>::get(elementInSlice, dst);
    dstOffset += dstIndex * dst.strides[dstSelectDim];

    IndexType srcOffset =
      cuda::detail::IndexToOffset<const T, IndexType, SrcDim>::get(elementInSlice, src);
    srcOffset += srcIndex * src.strides[srcSelectDim];

    dst.data[dstOffset] = src.data[srcOffset];
  }
}

namespace {

// When using a 0-dim scalar tensor, we need the legacy (THC) semantics of
// TensorInfo: Pretend that the scalar tensor is in fact a one-element vector.
template <typename T, typename IndexType>
cuda::detail::TensorInfo<T, IndexType>
tensorInfoLegacyIfScalar(cuda::detail::TensorInfo<T, IndexType> ti) {
  if (ti.dims == 0) {
    ti.dims = 1;
    ti.sizes[0] = 1;
    ti.strides[0] = 1;
  }
  return ti;
}

}

template <typename scalar_t>
void index_select_out_cuda_impl(
    Tensor& out,
    const Tensor& self,
    long dim,
    const Tensor& index) {
  ptrdiff_t numIndices = index.numel();
  int selfDims = self.dim() == 0 ? 1 : self.dim();

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  TORCH_CHECK(
      index.dim() <= 1, "Index is supposed to be an empty tensor or a vector");
  TORCH_CHECK(
      !(self.dim() == 0 && numIndices != 1), "index_select(): Index to scalar can have only 1 value, got ", numIndices, " value(s)");
  TORCH_CHECK(dim < selfDims, "Indexing dim is out of bounds");

  std::vector<int64_t> newSize = self.sizes().vec();
  if (self.dim() > 0) {
    newSize[dim] = numIndices;
  }

  if (self.is_quantized()){
      out = at::empty_quantized(newSize, out);
  } else {
    at::native::resize_output(out, newSize);
  }

  ptrdiff_t outTotalSize = out.numel();
  if (outTotalSize == 0) {
    return;
  }

  bool indContig = index.is_contiguous();

  // The `self` is partitioned into two parts:
  // -the size of each slice we are indexing, which is the
  // total size of the tensor ignoring dimension `dim`;
  // -the number of indices we are choosing, which is the total size
  // of the tensor `indices`.
  int64_t selfSelectDimSize = self.dim() == 0 ? 1 : self.size(dim);
  ptrdiff_t sliceSize = outTotalSize / numIndices;

  int mpc = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;

#define SMALL_INDEX(TENSOR_TYPE, INDICES_TYPE, TYPE, DST_DIM, SRC_DIM, IDX_DIM)         \
  indexSelectSmallIndex<TENSOR_TYPE, INDICES_TYPE, TYPE, DST_DIM, SRC_DIM, IDX_DIM>     \
    <<<smallIndexGrid, smallIndexBlock, 0, stream>>>(                                   \
      outInfo, selfInfo, indicesInfo,                                                   \
      outSelectDim, selfSelectDim, static_cast<TYPE>(sliceSize),                        \
      selfSelectDimSize);                                                               \
  C10_CUDA_KERNEL_LAUNCH_CHECK();

#define LARGE_INDEX(TENSOR_TYPE, INDICES_TYPE, TYPE,                           \
                    DST_DIM, SRC_DIM, IDX_DIM, IDX_IS_MAJOR)                   \
  indexSelectLargeIndex<TENSOR_TYPE, INDICES_TYPE, TYPE,                       \
                        DST_DIM, SRC_DIM, IDX_DIM, IDX_IS_MAJOR>               \
    <<<largeIndexGrid, largeIndexBlock, 0, stream>>>(                          \
      outInfo, selfInfo, indicesInfo,                                          \
      outSelectDim, selfSelectDim, static_cast<TYPE>(outTotalSize),            \
      static_cast<TYPE>((IDX_IS_MAJOR) ? sliceSize : numIndices),              \
      selfSelectDimSize);                                                      \
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  dim3 smallIndexGrid(std::min(ceil_div(sliceSize, (ptrdiff_t)128), (ptrdiff_t)(mpc * 8)));
  dim3 smallIndexBlock(std::min(sliceSize, (ptrdiff_t)128));

  dim3 largeIndexGrid(std::min(ceil_div(outTotalSize, (ptrdiff_t)128), (ptrdiff_t)(mpc * 8)));
  dim3 largeIndexBlock(std::min(outTotalSize, (ptrdiff_t)128));
  if (cuda::detail::canUse32BitIndexMath(out) &&
      cuda::detail::canUse32BitIndexMath(self) &&
      cuda::detail::canUse32BitIndexMath(index)) {
    auto outInfo = tensorInfoLegacyIfScalar(cuda::detail::getTensorInfo<scalar_t, unsigned int>(out));
    int outSelectDim = outInfo.collapseDims(dim);
    outInfo.reduceDim(outSelectDim);

    auto  selfInfo = tensorInfoLegacyIfScalar(cuda::detail::getTensorInfo<const scalar_t, unsigned int>(self));
    int selfSelectDim = selfInfo.collapseDims(dim);
    selfInfo.reduceDim(selfSelectDim);

    AT_DISPATCH_INDEX_TYPES(index.scalar_type(), "index_select_out_cuda_impl", [&] () {
      auto indicesInfo = tensorInfoLegacyIfScalar(cuda::detail::getTensorInfo<const index_t, unsigned int>(index));
      indicesInfo.collapseDims();

      // A reasonable choice for when to have each thread iterate over
      // indices to choose
      if (numIndices <= 16) {
        if (outInfo.dims == 1 && selfInfo.dims == 1 && indContig) {
          SMALL_INDEX(scalar_t, index_t, unsigned int, 1, 1, -2);
        } else if (outInfo.dims == 2 && selfInfo.dims == 2 && indContig) {
          SMALL_INDEX(scalar_t, index_t, unsigned int, 2, 2, -2);
        } else if (outInfo.dims == 3 && selfInfo.dims == 3 && indContig) {
          SMALL_INDEX(scalar_t, index_t, unsigned int, 3, 3, -2);
        } else {
          SMALL_INDEX(scalar_t, index_t, unsigned int, -1, -1, -1);
        }
      } else {
        bool indexIsMajor = indexShouldBeMajor(outInfo, outSelectDim);

        if (outInfo.dims == 1 && selfInfo.dims == 1 && indContig) {
          LARGE_INDEX(scalar_t, index_t, unsigned int, 1, 1, -2, true);
        } else if (outInfo.dims == 2 && selfInfo.dims == 2 && indContig) {
          if (indexIsMajor) {
            LARGE_INDEX(scalar_t, index_t, unsigned int, 2, 2, -2, true);
          } else {
            LARGE_INDEX(scalar_t, index_t, unsigned int, 2, 2, -2, false);
          }
        } else if (outInfo.dims == 3 && selfInfo.dims == 3 && indContig) {
          if (indexIsMajor) {
            LARGE_INDEX(scalar_t, index_t, unsigned int, 3, 3, -2, true);
          } else {
            LARGE_INDEX(scalar_t, index_t, unsigned int, 3, 3, -2, false);
          }
        } else {
          LARGE_INDEX(scalar_t, index_t, unsigned int, -1, -1, -1, true);
        }
      }
    });
  } else {
    auto outInfo = tensorInfoLegacyIfScalar(cuda::detail::getTensorInfo<scalar_t, uint64_t>(out));
    int outSelectDim = outInfo.collapseDims(dim);
    outInfo.reduceDim(outSelectDim);

    auto selfInfo = tensorInfoLegacyIfScalar(cuda::detail::getTensorInfo<const scalar_t, uint64_t>(self));
    int selfSelectDim = selfInfo.collapseDims(dim);
    selfInfo.reduceDim(selfSelectDim);
    AT_DISPATCH_INDEX_TYPES(index.scalar_type(), "index_select_out_cuda_impl", [&] () {
      auto indicesInfo = tensorInfoLegacyIfScalar(cuda::detail::getTensorInfo<const index_t, uint64_t>(index));
      indicesInfo.collapseDims();

      LARGE_INDEX(scalar_t, index_t, uint64_t, -1, -1, -1, true);
    });
  }
#undef SMALL_INDEX
#undef LARGE_INDEX
}
} // anonymous namespace

Tensor& index_select_out_cuda(
    const Tensor& self,
    int64_t dim,
    const Tensor& index,
    Tensor& out) {
  static constexpr string_view DIM_WARNING =
      "Tensor too large or too many (> 25) dimensions";
  TORCH_CHECK(
      at::cuda::check_device({out, self, index}),
      "Input, output and indices must be on the current device");
  at::assert_no_internal_overlap(out);
  at::assert_no_overlap(out, self);
  at::assert_no_overlap(out, index);

  dim = at::maybe_wrap_dim(dim, self);
  TORCH_CHECK(self.dim() <= MAX_TENSORINFO_DIMS, DIM_WARNING);
  TORCH_CHECK(index.dim() <= MAX_TENSORINFO_DIMS, DIM_WARNING);
  if (self.is_quantized()){
    TORCH_CHECK(
      self.qscheme() == kPerTensorAffine,
      "Only per_tensor quantized quantized tensors are supported by index_select.")
    AT_DISPATCH_QINT_TYPES(out.scalar_type(), "index_select_quant_cuda", [&] {
      index_select_out_cuda_impl<scalar_t>(out, self, dim, index);
    });
  } else {
    AT_DISPATCH_V2(
        out.scalar_type(),
        "index_select_cuda",
        AT_WRAP([&] { index_select_out_cuda_impl<scalar_t>(out, self, dim, index); }),
        AT_EXPAND(AT_ALL_TYPES_AND_COMPLEX), AT_EXPAND(AT_BAREBONES_UNSIGNED_TYPES),
        kComplexHalf,
        kHalf,
        kBool,
        kBFloat16
        );
  }

  return out;
}

Tensor index_select_cuda(const Tensor& self, int64_t dim, const Tensor& index) {
  Tensor out = at::empty({0}, self.options());
  at::native::index_select_out_cuda(self, dim, index, out);
  return out;
}

Tensor index_select_quantized_cuda(const Tensor& self, int64_t dim, const Tensor& index) {
  TORCH_CHECK(
    self.qscheme() == kPerTensorAffine,
    "Only per_tensor quantized quantized tensors are supported by index_select.")
  Tensor out = at::empty_quantized({0}, self);
  at::native::index_select_out_cuda(self, dim, index, out);
  return out;
}

namespace {

void masked_fill_kernel(TensorIterator& iter, const Scalar& value) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND4(
      kBool, kHalf, kBFloat16, kComplexHalf, iter.common_dtype(), "masked_fill_", [&]() {
        const auto value_ = value.to<scalar_t>();
        gpu_kernel(
            iter, [value_] GPU_LAMBDA(scalar_t self, bool mask) -> scalar_t {
              if (mask) {
                return value_;
              }
              return self;
            });
      });
}

template <typename scalar_t>
void cuda_masked_fill_kernel_quantized(TensorIterator& iter, scalar_t quantized_val) {
    gpu_kernel(
        iter, [quantized_val] GPU_LAMBDA(scalar_t self, bool mask) -> scalar_t {
          if (mask) {
            return quantized_val;
          }
          return self;
    });
}

void masked_fill_kernel_quantized(TensorIterator& iter, const Scalar& value, double scale, int zero_point) {
  TORCH_CHECK(iter.input_dtype(1) == at::ScalarType::Bool, "masked_fill only supports boolean masks, ",
    "but got dtype ", iter.input_dtype(1));
  AT_DISPATCH_QINT_TYPES(
      iter.common_dtype(), "masked_fill_", [&]() {
        float float_val = value.to<float>();
        const auto quantized_val = quantize_val<scalar_t>(scale, zero_point, float_val);

        cuda_masked_fill_kernel_quantized<scalar_t>(iter, quantized_val);
    });
}

REGISTER_CUDA_DISPATCH(masked_fill_kernel_quantized_stub, &masked_fill_kernel_quantized);

} // anonymous namespace

Tensor & masked_fill__cuda(Tensor& self, const Tensor & mask, const Scalar& value) {
  TORCH_CHECK(self.device() == mask.device(), "expected self and mask to be on the same device, but got mask on ",
    mask.device(), " and self on ", self.device());
  TORCH_CHECK(mask.scalar_type() == kBool,
    "masked_fill only supports boolean masks, but got dtype ", mask.scalar_type());
  auto maybe_outnames = namedinference::broadcast_to_outnames(self, mask, "masked_fill_");
  if (at::has_internal_overlap(self) == MemOverlap::Yes) {
    TORCH_WARN(
      "Use of masked_fill_ on expanded tensors is deprecated. "
      "Please clone() the tensor before performing this operation. "
      "This also applies to advanced indexing e.g. tensor[mask] = scalar");
  }
  at::assert_no_partial_overlap(self, mask);

  c10::MaybeOwned<Tensor> b_mask = expand_inplace(self, mask, "masked_fill_");

  auto iter = TensorIteratorConfig()
      .set_check_mem_overlap(false)
      .check_all_same_dtype(false)
      .resize_outputs(false)
      .add_output(self)
      .add_const_input(self)
      .add_const_input(*b_mask)
      .build();

  masked_fill_kernel(iter, value);
  namedinference::propagate_names_if_nonempty(self, maybe_outnames);
  return self;
}

Tensor & masked_fill__cuda(Tensor& self, const Tensor & mask, const Tensor & value) {
  TORCH_CHECK(value.dim() == 0, "masked_fill_ only supports a 0-dimensional value tensor, but got tensor "
      "with ", value.dim(), " dimension(s).");
  // We hit this function if either of the input tensor lives on CUDA.
  // It is ok, if `value` is `CPU` tensor but we should not allow `self` or
  // `mask` to be CPU tensor. Check for `self` and `mask` being on same device
  // exists in `masked_fill__cuda` (Scalar version).
  TORCH_CHECK(!self.device().is_cpu(), "masked_fill_: Expected inputs to be on same device")
  return masked_fill__cuda(self, mask, value.item());
}

namespace {

// ForwardIt: only legacy random access iterator is supported.
template<class ForwardIt, class T, bool is_lower = true>
static __host__ __device__ __forceinline__
ForwardIt find_bound(ForwardIt first, ForwardIt last, const T& value) {
    ForwardIt it;
    typename std::iterator_traits<ForwardIt>::difference_type count, step;
    // NOTE: std::distance(first, last) compiles but produces wrong results here,
    // so only legacy random access iterators are safe in this code.
    count = last - first;

    while (count > 0) {
      it = first;
      step = count / 2;
      // avoiding std::advance(it, step),
      // although it does work unlike std::distance
      it += step;
      if (is_lower ? *it < value : value >= *it) {
        first = ++it;
        count -= step + 1;
      }
      else {
        count = step;
      }
    }
    return first;
}

}

Tensor index_select_sparse_cuda(const Tensor& self, int64_t dim, const Tensor& index) {
  const auto ndim = self.dim();
  TORCH_CHECK_INDEX(ndim, "index_select() cannot be applied to a 0-dim tensor.");
  TORCH_CHECK_INDEX(
      index.dim() == 1 && index.dtype() == at::kLong && index.options().layout() == at::kStrided,
      "index_select() argument index must be 1-D strided (non-sparse) long-tensor.");
  dim = maybe_wrap_dim(dim, ndim);
  const auto size = self.size(dim);
  const auto sparse_dim = self.sparse_dim();
  const auto dense_dim = self.dense_dim();
  const auto indices = self._indices();
  const auto values = self._values();
  const auto nnz = values.size(0);
  const auto index_len = index.size(0);
  auto res_sizes = self.sizes().vec();
  res_sizes[dim] = index_len;

  // If indexing into sparse dimensions
  if (dim < sparse_dim) {
    const auto make_output = [
      dim, sparse_dim, dense_dim, res_sizes, &self, &indices, &values
    ](
        const Tensor& selected_dim_indices,
        const Tensor& res_dim_indices
    ) -> Tensor {
      auto res_indices = indices.index_select(1, selected_dim_indices);
      res_indices[dim] = res_dim_indices;
      const auto res_values = values.index_select(0, selected_dim_indices);

      return at::_sparse_coo_tensor_with_dims_and_tensors(
          sparse_dim, dense_dim, res_sizes, res_indices, res_values, self.options());
    };

    // short-circuit if index is empty
    if (!index_len) {
      return make_output(index, index);
    }

    const auto nneg_index = [&index, size]() -> Tensor {
      auto nneg_index = at::empty_like(index, at::MemoryFormat::Contiguous);

      auto iter = TensorIteratorConfig()
        .add_output(nneg_index)
        .add_input(index)
        .build();

      AT_DISPATCH_INDEX_TYPES(index.scalar_type(), "index_select_sparse_cuda", [&]() {
          gpu_kernel(iter, [size] GPU_LAMBDA (index_t idx) -> index_t {
              CUDA_KERNEL_ASSERT(idx >= -size && idx < size
                  && "index_select(): index out of bounds");
              return idx < 0 ? idx + size : idx;
          });
      });
      return nneg_index;
    }();

    const auto dim_indices = indices[dim].contiguous();
    const auto idx_nneg_index = at::arange(index_len, nneg_index.options());
    const auto idx_dim_indices = at::arange(nnz, dim_indices.options());

    Tensor sorted_dim_indices, argsort_dim_indices;
    std::tie(sorted_dim_indices, argsort_dim_indices) = [&]() -> std::tuple<Tensor, Tensor> {
      if (dim == 0 && self.is_coalesced()) {
        return std::make_tuple(dim_indices, idx_dim_indices);
      }
      else {
        return dim_indices.sort();
      }
    }();

    Tensor intrsc_counts_nneg_index;
    Tensor intrsc_first_match_nneg_index;
    std::tie(intrsc_counts_nneg_index, intrsc_first_match_nneg_index) = [&]() -> std::tuple<Tensor, Tensor> {
      auto intrsc_counts_nneg_index = at::zeros_like(nneg_index);
      auto intrsc_first_match_nneg_index = at::zeros_like(nneg_index);

      auto iter = TensorIteratorConfig()
        .add_output(intrsc_first_match_nneg_index)
        .add_input(nneg_index)
        .add_input(idx_nneg_index)
        .build();

      AT_DISPATCH_INDEX_TYPES(nneg_index.scalar_type(), "index_select_sparse_cuda", [&]() {
          index_t* ptr_intrsc_counts_nneg_index = intrsc_counts_nneg_index.mutable_data_ptr<index_t>();
          const index_t* ptr_sorted_dim_indices = sorted_dim_indices.const_data_ptr<index_t>();
          gpu_kernel(
              iter,
              [ptr_intrsc_counts_nneg_index, ptr_sorted_dim_indices, nnz] GPU_LAMBDA (
                index_t idx_val, index_t idx_idx
              ) -> index_t {
                auto* lb = find_bound<const index_t*, index_t, true>(
                  ptr_sorted_dim_indices,
                  ptr_sorted_dim_indices + nnz,
                  idx_val
                );
                auto* ub = find_bound<const index_t*, index_t, false>(
                  ptr_sorted_dim_indices,
                  ptr_sorted_dim_indices + nnz,
                  idx_val
                );
                const auto idx_count = ub - lb;
                ptr_intrsc_counts_nneg_index[idx_idx] = idx_count;

                return lb - ptr_sorted_dim_indices;
              }
          );
      });

      return std::make_tuple(intrsc_counts_nneg_index, intrsc_first_match_nneg_index);
    }();

    // Unavoidable sync since the shape of the result is not known in advance
    auto res_len = intrsc_counts_nneg_index.sum().item<int64_t>();
    // Short-circuit if empty intersection
    if (!res_len) {
      auto empty_idx = at::empty({0}, nneg_index.options());
      return make_output(empty_idx, empty_idx);
    }

    Tensor selected_dim_indices, res_dim_indices;
    std::tie(selected_dim_indices, res_dim_indices) = [&]() -> std::tuple<Tensor, Tensor> {
      auto res_dim_indices = at::empty({res_len}, nneg_index.options());
      auto selected_dim_indices = at::empty_like(res_dim_indices);
      auto selected_dim_indices_offsets = intrsc_counts_nneg_index.cumsum(0)
        .sub_(intrsc_counts_nneg_index);

      // Need to have output as TensorIterator does not allow having void lambdas.
      auto dummy_output = at::empty({1}, dim_indices.options()).expand(IntArrayRef({index_len}));
      auto iter = TensorIteratorConfig()
        .add_output(dummy_output)
        // All iterations map to a single element in dummy_output by design,
        // hence removed output memory overlap check.
        .set_check_mem_overlap(false)
        .add_input(idx_nneg_index)
        .add_input(intrsc_counts_nneg_index)
        .add_input(selected_dim_indices_offsets)
        .add_input(intrsc_first_match_nneg_index)
        .build();

      AT_DISPATCH_INDEX_TYPES(nneg_index.scalar_type(), "index_select_sparse_cuda", [&]() {
          index_t* ptr_res_dim_indices = res_dim_indices.mutable_data_ptr<index_t>();
          index_t* ptr_selected_dim_indices = selected_dim_indices.mutable_data_ptr<index_t>();
          const index_t* ptr_argsort_dim_indices = argsort_dim_indices.const_data_ptr<index_t>();
          gpu_kernel(
              iter,
              [ptr_res_dim_indices, ptr_selected_dim_indices, ptr_argsort_dim_indices] GPU_LAMBDA (
                index_t idx_idx, index_t count, index_t offset, index_t first_match
              ) -> index_t {
                index_t* __restrict__ ptr_res_dim_indices_out = ptr_res_dim_indices + offset;
                const index_t* __restrict__ ptr_argsort_dim_indices_in = ptr_argsort_dim_indices + first_match;
                index_t* __restrict__ ptr_selected_dim_indices_out = ptr_selected_dim_indices + offset;
                for (index_t i = 0; i < count; ++i) {
                  *ptr_res_dim_indices_out++ = idx_idx;
                  *ptr_selected_dim_indices_out++ = *ptr_argsort_dim_indices_in++;
                }

                // A dummy return scalar for a dummy output
                return static_cast<index_t>(1);
              }
          );
      });

      return std::make_tuple(selected_dim_indices, res_dim_indices);
    }();

    return make_output(selected_dim_indices, res_dim_indices);
  }
  // If indexing into dense dimensions
  else {
    // It is sufficient to just perform `index_select` on values
    // if `dim` refers to dense dimensions.
    const auto res_values = values.index_select(dim - sparse_dim + 1, index);

    return _sparse_coo_tensor_with_dims_and_tensors(
        sparse_dim, dense_dim, res_sizes, indices, res_values, self.options());
  }
}


} // at::native
