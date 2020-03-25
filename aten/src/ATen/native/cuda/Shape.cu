#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/Dispatch.h>
#include <c10/core/MemoryFormat.h>
#include <c10/util/Optional.h>

#include <THC/THC.h>

namespace at {
namespace native {

constexpr int CAT_ARRAY_BATCH_SIZE = 1024;
constexpr int CAT_ARRAY_MAX_INPUT_DIMS = 4;

namespace {

inline bool getCatGrid(ptrdiff_t nTensors, dim3& grid) {
  const int numSM = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;

  //X dim of grid for cat array cooperates on a single tensor in the cat.
  //Given half of the GPU, full utilization will always occur.
  grid = dim3( 2LL * numSM, (long long) nTensors );

  return true;
}

// Similar to any other IndexToOffset calculation for copying along a given
// dimension.
template <typename IndexType, int Dims>
struct CatArrIndexToOffset {
  static inline __device__ IndexType compute(
      const IndexType outputSize[Dims],
      const IndexType outputStride[Dims],
      const IndexType dimSize,
      const unsigned int concatDim,
      IndexType linearIndex) {
    // linearIndex is not really linear index, but instead the offset in
    // input tensor. If the input tensor is contiguous, then this offset
    // is the linear index, but if the input tensor is channels last, then
    // it is the linear index of the permuted contiguous tensor
    IndexType offset = 0;

#pragma unroll
    for (int i = Dims - 1; i >= 1; --i) {
      IndexType curDimSize = i == concatDim ? dimSize : outputSize[i];
      IndexType nextDimIndex = linearIndex / curDimSize;
      IndexType curDimIndex = linearIndex - curDimSize * nextDimIndex;
      IndexType curDimOffset = curDimIndex * outputStride[i];
      offset += curDimOffset;
      linearIndex = nextDimIndex;
    }

    return offset + linearIndex * outputStride[0];
  }
};

template <typename T, typename IndexType>
struct CatArrInputTensor {
  T* input;
  IndexType offset;
  IndexType dimSize;
  IndexType nElements;
};

template<typename IndexType, unsigned int MaxDims>
struct OutputTensorSizeStride {
  IndexType outputSize[MaxDims];
  IndexType outputStride[MaxDims];
};

/**
  * Kernel used to concatenated grimDim.y tensors into an output tensor. Uses a
  * grid-stride loop based off of the blockIdx.x, threadIdx.x for each input to
  * copy each element from each input tensor into the output.
  *
  * output: base pointer to the storage associated with the output tensor
  * inputs: GPU-allocated array of input metadata for each input to concatenate
  *         in the kernel
  * os: the size/stride vectors for the output tensor
  * concatDim: dimension along which we are concatenating
  * dimStride: the stride of the output tensor at the concatDim
  *
  * The most important assumption made is that the input tensors are contiguous.
  */
template <typename T, typename IndexType, int Dims>
#ifdef __HIP_PLATFORM_HCC__
C10_LAUNCH_BOUNDS_1(512)
#endif
__global__ void CatArrayBatchedCopy(
    T* output,
    CatArrInputTensor<T, IndexType>* inputs,
    OutputTensorSizeStride<IndexType, CAT_ARRAY_MAX_INPUT_DIMS> os,
    const int concatDim,
    IndexType dimStride) {

    IndexType tid = blockIdx.x * blockDim.x + threadIdx.x;
    IndexType nElements = inputs[blockIdx.y].nElements;

    if(tid >= nElements) return;

    T* data = inputs[blockIdx.y].input;
    IndexType offset = inputs[blockIdx.y].offset;
    IndexType dimSize = inputs[blockIdx.y].dimSize;
    IndexType dataOffset = offset * dimStride;

    IndexType stride = gridDim.x * blockDim.x;

    while( tid < nElements){
    IndexType elementOffset = CatArrIndexToOffset<IndexType, Dims>::compute(
                  os.outputSize, os.outputStride, dimSize, concatDim, tid);
    output[dataOffset + elementOffset] = data[tid];

    tid += stride;
    }
}

void check_shape_except_dim(const Tensor &first, const Tensor &second,
                            int dimension)
{
  int first_dims = first.dim();
  int second_dims = second.dim();
  TORCH_CHECK(first_dims == second_dims,
      "Tensors must have same number of dimensions: got ", first_dims,
      " and ", second_dims);
  for (int dim = 0; dim < first_dims; dim++) {
    if (dim == dimension) {
      continue;
    }
    int64_t first_dim_size = at::native::size(first, dim);
    int64_t second_dim_size = at::native::size(second, dim);
    TORCH_CHECK(first_dim_size == second_dim_size,
        "Sizes of tensors must match except in dimension ", dim, ". Got ",
        static_cast<long long>(first_dim_size), " and ",
        static_cast<long long>(second_dim_size));
  }
}

template <typename scalar_t>
void parallel_cat(Tensor &out, const TensorList &inputs, int64_t dimension,
                  int nDims, c10::MemoryFormat memory_format) {
  // First, let's set up our kernel parameters. We start with a raw pointer to
  // the storage for the output Tensor.
  scalar_t *data = out.data_ptr<scalar_t>();

  // Kernel Parameter
  long tensorMetadataSize =
    sizeof(CatArrInputTensor<scalar_t, unsigned int>) * CAT_ARRAY_BATCH_SIZE;
  auto d_inputs_storage = at::empty(
    {tensorMetadataSize}, out.options().dtype(at::kByte));
  auto d_inputs = static_cast<CatArrInputTensor<scalar_t, unsigned int> *>(
    d_inputs_storage.data_ptr());

  OutputTensorSizeStride<unsigned int, CAT_ARRAY_MAX_INPUT_DIMS> param;

  // Next, let's initialize the size, stride arrays for the output Tensor.
  if (memory_format == c10::MemoryFormat::Contiguous) {
    for (int i = 0; i < nDims; ++i) {
      param.outputSize[i] = at::native::size(out, i);
      param.outputStride[i] = out.stride(i);
    }
  } else if (memory_format == c10::MemoryFormat::ChannelsLast || memory_format == c10::MemoryFormat::ChannelsLast3d) {
    // permute the semantics of dims from NCHW to NHWC so that the input
    // tensor is now contiguous
    param.outputSize[0] = at::native::size(out, 0);
    param.outputStride[0] = out.stride(0);
    for (int i = 1; i < nDims - 1; ++i) {
      param.outputSize[i] = at::native::size(out, i + 1);
      param.outputStride[i] = out.stride(i + 1);
    }
    param.outputSize[nDims - 1] = at::native::size(out, 1);
    param.outputStride[nDims - 1] = out.stride(1);
  } else {
    TORCH_CHECK(false, "unsupported memory format");
  }

  at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();

  // Now we loop
  int batchCounter = 0;
  int64_t offset = 0;
  for (int i = 0; i < inputs.size() ; i += CAT_ARRAY_BATCH_SIZE) {
    // Re-allocate stackInputs every iteration to avoid read-after-write hazard
    {
      auto stackInputs_storage = at::empty({tensorMetadataSize},
          out.options().dtype(at::kByte).device(at::kCPU).pinned_memory(true));
      auto stackInputs =
        static_cast<CatArrInputTensor<scalar_t, unsigned int> *>(
          stackInputs_storage.data_ptr());
      for (batchCounter = 0;
           batchCounter < CAT_ARRAY_BATCH_SIZE &&
             (i+batchCounter) < inputs.size();
           ++batchCounter) {
        int64_t dimSize = at::native::size(inputs[i+batchCounter], dimension);

        stackInputs[batchCounter].input =
          inputs[i+batchCounter].data_ptr<scalar_t>();
        stackInputs[batchCounter].offset = offset;
        stackInputs[batchCounter].dimSize = dimSize;
        stackInputs[batchCounter].nElements = inputs[i+batchCounter].numel();

        // update offset
        offset += dimSize;
      }
      at::native::copy_(d_inputs_storage, stackInputs_storage,
                        /* non_blocking= */ true);
    }

    // Next, let's consider how we set our kernel launch parameters.
    // We borrow from THCApply, which the kernel's internal indexing
    // is based on.
    dim3 applyBlock = dim3(32*16);

    //Get grid where x dim fills half gpu and y dim is number of tensors.
    //This will have cating two tensors fill the entire grid, but prevent
    //many threads from needlessly load meta data if their sizes is small.
    dim3 catGrid;
    getCatGrid(batchCounter, catGrid);


    if (memory_format != c10::MemoryFormat::Contiguous) {
      switch (dimension) {
      case 0:
        break;
      case 1:
        dimension = nDims - dimension;
        break;
      default:
        dimension--;
      }
    }
    // Template Declarations for dim = 1, 2, 3, 4
#define HANDLE_CASE(DIMS) \
    CatArrayBatchedCopy<scalar_t, unsigned int, DIMS><<<\
        catGrid, applyBlock, 0, stream.stream()>>>(\
            data, d_inputs, param, dimension, param.outputStride[dimension]);
    switch (nDims) {
      case 1:
        HANDLE_CASE(1);
        break;
      case 2:
        HANDLE_CASE(2);
        break;
      case 3:
        HANDLE_CASE(3);
        break;
      case 4:
        HANDLE_CASE(4);
        break;
    }
#undef HANDLE_CASE
    AT_CUDA_CHECK(cudaGetLastError());
  }
}

} // namespace

Tensor cat_cuda(TensorList inputs, int64_t dimension) {
  Tensor out = at::empty({0}, inputs.front().options());
  cat_out_cuda(out, inputs, dimension);
  return out;
}

inline c10::MemoryFormat compute_output_memory_format(const TensorList &inputs) {
  c10::optional<c10::MemoryFormat> format = c10::nullopt;
  for (auto &t : inputs) {
    auto f = t.suggest_memory_format();
    if (!format.has_value()) {
      format = f;
      continue;
    }
    if (format.value() == f) {
      continue;
    }
    bool contiguous = (format.value() == c10::MemoryFormat::Contiguous || f == c10::MemoryFormat::Contiguous || format.value() != f);
    if (contiguous) {
      return c10::MemoryFormat::Contiguous;
    }
  }
  return format.value();
}

Tensor& cat_out_cuda(Tensor& out, TensorList inputs, int64_t dimension) {

  // previously, size [0] tensors were the only possible empty tensors; thus, it
  // wasn't possible to cat empty tensors unless all the other tensors were
  // 1-dimensional, so we allowed these tensors to be "skipped".  We maintain
  // this behavior for backwards compatibility, but only for this specific size
  // (i.e. other empty sizes are not skipped).
  // FIXME: warn if this is the case
  auto should_skip = [](const Tensor &t) {
    return t.dim() == 1 && at::native::size(t, 0) == 0;
  };
  bool hasSkippedInput = false;

  const Tensor *notSkippedTensor = NULL;  // non-owning reference
  int nDims = 0;

  // Inputs cannot alias the output tensor
  for (int i = 0; i < inputs.size(); i++) {
    auto lap = at::get_overlap_status(out, inputs[i]);
    TORCH_CHECK(lap != at::MemOverlapStatus::PARTIAL &&
                lap != at::MemOverlapStatus::FULL,
                "unsupported operation: the input tensors cannot refer to any "
                "of the output memory locations. Found overlap in input "
                "tensor ", i);
  }

  for (int i = 0; i < inputs.size(); i++)
  {
    if (should_skip(inputs[i])) {
      hasSkippedInput = true;
      continue;
    }
    nDims = inputs[i].dim();
    notSkippedTensor = &inputs[i];
  }

  // If all inputs are empty tensors, return an empty tensor
  if (notSkippedTensor == NULL) {
    return out;
  }

  TORCH_CHECK(inputs.size() > 0, "invalid number of inputs ", inputs.size());
  TORCH_CHECK(dimension >= 0, "invalid dimension ", dimension);

  for (const Tensor& t: inputs) {
    TORCH_CHECK(t.device() == notSkippedTensor->device(),
                "All input tensors must be on the same device. Received ",
                t.device(), " and ", notSkippedTensor->device());
  }

  c10::MemoryFormat memory_format = compute_output_memory_format(inputs);

  std::vector<int64_t> size(notSkippedTensor->sizes().vec());

  // Compute size of the result in the cat dimension
  int64_t cat_dim_size = 0;
  for (int i = 0; i < inputs.size(); i++) {
    const Tensor &tensor = inputs[i];
    if (should_skip(tensor)) {
      continue;
    }
    check_shape_except_dim(*notSkippedTensor, tensor, dimension);
    cat_dim_size += at::native::size(tensor, dimension);
  }

  // Compute the size of the result
  size[dimension] = cat_dim_size;
  out.resize_(size, memory_format);
  if (out.numel() == 0) {
    return out;
  }

  // We parallelize the copy if all 6 conditions pass:
  //
  // 1. There is more than one input tensor
  // 2. No empty inputs
  // 3. The out tensor is 32-bit indexable
  // 4. The number of dimensions is <= 4
  // 5. All input tensors are contiguous (output tensor may be non-contig)
  // 6. All input tensors can use 32-bit indexing

  const bool all32BitIndexable = std::all_of(inputs.begin(), inputs.end(),
    [] (const Tensor& t) {
      return at::cuda::detail::canUse32BitIndexMath(t);
    });
  const bool allContiguous = std::all_of(inputs.begin(), inputs.end(),
    [=](const Tensor& t) {
      return !t.defined() || t.is_contiguous(memory_format);
    });
  if (inputs.size() > 1 &&
      !hasSkippedInput &&
      out.dim() <= CAT_ARRAY_MAX_INPUT_DIMS &&
      at::cuda::detail::canUse32BitIndexMath(out) &&
      allContiguous &&
      all32BitIndexable) {

    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
        at::ScalarType::Half, at::ScalarType::Bool, at::ScalarType::BFloat16,
        out.scalar_type(), "cat_cuda", [&]() {
      parallel_cat<scalar_t>(out, inputs, dimension, nDims, memory_format);
    });

  } else {
    int64_t offset = 0;
    for (int j = 0; j < inputs.size(); j++)
    {
      if (should_skip(inputs[j])) continue;
      int64_t dimSize = at::native::size(inputs[j], dimension);
      Tensor nt = at::narrow(out, dimension, offset, dimSize);
      copy_(nt, inputs[j]);
      offset += dimSize;
    }
  }

  return out;
}

} // namespace native
} // namespace at
