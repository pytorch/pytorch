#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/AccumulateType.h>
#include <ATen/ceil_div.h>
#include <ATen/Dispatch.h>
#include <ATen/Utils.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/EmptyTensor.h>
#include <ATen/cuda/detail/KernelUtils.h>
#include <ATen/native/UnaryOps.h>
#include <ATen/native/cuda/LaunchUtils.h>
#include <ATen/cuda/CUDAGraphsUtils.cuh>
#include <ATen/native/cuda/block_reduce.cuh>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/CUDAFunctions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/empty_native.h>
#include <ATen/ops/empty_like_native.h>
#include <ATen/ops/cumsum_cuda_dispatch.h>
#include <ATen/ops/uniform_native.h>
#endif

#include <curand.h>
#include <curand_kernel.h>
#include <curand_philox4x32_x.h>
#include <type_traits>

namespace at::native {

namespace {

template <
    typename T,
    typename = std::enable_if_t<
        std::is_floating_point_v<T> || std::is_convertible_v<T, float>>>
inline __device__ bool _isinf(T x) {
  if constexpr (std::is_floating_point_v<T>) {
    return ::isinf(x);
  } else {
    return ::isinf(static_cast<float>(x));
  }
}

#define MAX_NUM_BLOCKS 200

// Normalizes the L1 norm of every row to 1; used by multinomial
template <typename scalar_t>
C10_LAUNCH_BOUNDS_1(cuda::detail::CUDA_NUM_THREADS)
__global__ void renormRowsL1(scalar_t* dist, long rows, long cols) {
  extern __shared__  unsigned char my_smem[];
  scalar_t *smem = reinterpret_cast<scalar_t *>(my_smem);
  scalar_t zero = static_cast<scalar_t>(0);
  scalar_t val;
  for (int64_t row = blockIdx.x; row < rows; row += gridDim.x) {
    scalar_t sum = static_cast<scalar_t>(0);
    for (int64_t col = threadIdx.x; col < cols; col += blockDim.x) {
      val = dist[row * cols + col];
      CUDA_KERNEL_ASSERT(!(val < zero)); // ! < 0 for NaN handling
      sum = sum + val;
    }

    sum = cuda_utils::BlockReduceSum(sum, smem);
    if (threadIdx.x == 0) {
      CUDA_KERNEL_ASSERT(!(val < zero)); // ! < 0 for NaN handling
      smem[0] = sum;
    }
    __syncthreads();

    sum = smem[0];
    if (sum > zero) {
      for (int64_t col = threadIdx.x; col < cols; col += blockDim.x) {
        dist[row * cols + col] = dist[row * cols + col] / sum;
      }
    }
  }
}

void renormRows(Tensor& t) {
  TORCH_CHECK(t.dim() == 2);
  int64_t rows = t.size(0);
  int64_t cols = t.size(1);

  auto props = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(props != nullptr);
  int numSM = props->multiProcessorCount;
  const int64_t maxThreads = std::min(
      props->maxThreadsPerBlock, cuda_utils::kCUDABlockReduceMaxThreads);

  int warp_size = at::cuda::warp_size();
  dim3 grid(rows < numSM * 4 ? rows : numSM * 4);
  dim3 block(std::min(maxThreads, warp_size * ceil_div(cols, int64_t{warp_size})));

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(t.scalar_type(), "renormRows_cuda", [&] {
    renormRowsL1<scalar_t>
        <<<grid, block, (block.x / warp_size) * sizeof(scalar_t),
        at::cuda::getCurrentCUDAStream()>>>(t.mutable_data_ptr<scalar_t>(),
            rows, cols);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  });
}

template <typename scalar_t>
__device__ int binarySearchForMultinomial(const scalar_t* cumdist,
                                          const scalar_t* dist,
                                          int size,
                                          scalar_t val) {
  int start = 0;
  int end = size;
  // cumdist[size - 1] = 0 => all zero prob dist
  CUDA_KERNEL_ASSERT(cumdist[size - 1] > static_cast<scalar_t>(0));

  while (end - start > 0) {
    int mid = start + (end - start) / 2;

    scalar_t midVal = cumdist[mid];
    if (midVal < val) {
      start = mid + 1;
    } else {
      end = mid;
    }
  }

  if (start == size) {
    // No probability mass or precision problems; just return the
    // first non-zero element by setting start to size-1 here,
    // the code below will move it to the last non-zero probability
    // this actually can happen when the random number is 1
    // (github pytorch issue #4858).
    start = size - 1;
  }

  while(start >= 1 && dist[start] == 0) start--;

  return start;
}

template <typename scalar_t>
__global__ void
sampleMultinomialWithReplacement(PhiloxCudaState philox_args,
                                 int totalSamples,
                                 int64_t* dest,
                                 int64_t distributions,
                                 int categories,
                                 const scalar_t* normDistPrefixSum,
                                 const scalar_t* normDist) {
  // At the moment, each warp computes one sample value in the binary
  // search due to divergence. It seems possible to compute multiple
  // values and limit divergence though later on.

  auto seeds = at::cuda::philox::unpack(philox_args);

  // global index formula for 2D grid of 1D blocks
  int idx = blockIdx.y * gridDim.x * blockDim.x + blockIdx.x * blockDim.x + threadIdx.x;

  curandStatePhilox4_32_10_t state;
  curand_init(std::get<0>(seeds),
              idx,
              std::get<1>(seeds),
              &state);

  // The block determines the distribution for which we generate a point
  for (int64_t curDist = blockIdx.y;
       curDist < distributions;
       curDist += gridDim.y) {
    for (int sample = blockIdx.x*blockDim.x + threadIdx.x;
         sample < totalSamples; sample += blockDim.x*gridDim.x) {

      //we are losing 3 out of 4 generated numbers but it's ok
      //this kernel is not very efficient anyway
      auto rand = curand_uniform4(&state);
      scalar_t r = static_cast<scalar_t>(rand.x);

      // Find the bucket that a uniform sample lies in
      int choice = binarySearchForMultinomial<scalar_t>(
          normDistPrefixSum + curDist * categories,
          normDist + curDist * categories,
          categories,
          r);

      dest[curDist * totalSamples + sample] = choice;

    }
  }
}

template <typename scalar_t, typename accscalar_t>
C10_LAUNCH_BOUNDS_1(cuda::detail::CUDA_NUM_THREADS)
__global__ void sampleMultinomialOnce(
    int64_t* dest,
    int64_t distributions,
    int categories,
    const scalar_t* sampled,
    const scalar_t* dist,
    int stride_dist, // dist->stride(0)
    int stride_categories // dist->stride(1)
) {
  extern __shared__  unsigned char my_smem[];
  __shared__ bool found;
  __shared__ unsigned foundPos;

  accscalar_t *smem = reinterpret_cast<accscalar_t *>(my_smem);

  accscalar_t accZero = static_cast<accscalar_t>(0);
  scalar_t zero = static_cast<scalar_t>(0);

  for (int64_t curDist = blockIdx.x;
       curDist < distributions; curDist += gridDim.x) {
    // Each block handles one distribution
    // First pass, find the total sum of the distribution
    accscalar_t sum = accZero;
    scalar_t val;
    for (int cat = threadIdx.x; cat < categories; cat += blockDim.x) {
      val = dist[curDist * stride_dist + cat * stride_categories];
      CUDA_KERNEL_ASSERT(!at::_isnan(val));
      CUDA_KERNEL_ASSERT(!_isinf(val));
      CUDA_KERNEL_ASSERT(!(val < zero));
      sum = sum + static_cast<accscalar_t>(val);
    }

    // threadIdx.x == 0 has the sum value from this
    sum = cuda_utils::BlockReduceSum(sum, smem);

    // Broadcast sum and sample value
    if (threadIdx.x == 0) {
      // Make sure the sum of our distribution didn't overflow
      CUDA_KERNEL_ASSERT(!_isinf(val));
      CUDA_KERNEL_ASSERT(sum > accZero);

      foundPos = 0;
      smem[0] = sum;
      smem[1] = sampled[curDist];
    }
    __syncthreads();

    sum = smem[0];
    scalar_t sample = static_cast<scalar_t>(smem[1]);
    __syncthreads();

    if (sum == accZero) {
      // Choose the first element
      if (threadIdx.x == 0) {
        dest[curDist] = 0;
      }

      continue;
    }

    int chunks = (categories + (int)blockDim.x - 1) / blockDim.x;
    accscalar_t prevHighProb = accZero;
    found = false;

    for (int chunk = 0; chunk < chunks && !found; ++chunk) {
      // All threads in bounds load a value
      int cat = chunk * blockDim.x + threadIdx.x;

      accscalar_t dist_val = cat < categories ?
                             static_cast<accscalar_t>(dist[curDist * stride_dist + cat * stride_categories]) / sum :
                             accZero;

      smem[threadIdx.x] = dist_val;
      __syncthreads();

      // Perform an inclusive prefix sum of the shared memory contents
      for (int offset = 1; offset < blockDim.x; offset *= 2) {
        accscalar_t val = accZero;

        if (threadIdx.x >= offset) {
          val = smem[threadIdx.x - offset] + smem[threadIdx.x];
        }

        __syncthreads();
        if (threadIdx.x >= offset) {
          smem[threadIdx.x] = val;
        }
        __syncthreads();
      }

      // Each thread will check to see if the sample falls in its
      // bucket
      scalar_t curBucket =
          static_cast<scalar_t>(smem[threadIdx.x] + prevHighProb);
      scalar_t prevBucket = static_cast<scalar_t>(
          threadIdx.x == 0 ? prevHighProb
                          : smem[threadIdx.x - 1] + prevHighProb);
      bool inBucket =
          (cat < categories) &&
          (!(sample >= curBucket) &&
          (sample >= prevBucket) &&
          (dist_val > zero));

      if (inBucket) {
        // We're done; we have the sample
        // Torch indices are 1-based
        atomicMax(&foundPos, cat);
        found = true;
      }

      // Store the previous scan's high value for future use
      prevHighProb = prevHighProb + smem[blockDim.x - 1];

      __syncthreads();
    }

    if (threadIdx.x == 0) {
      if (found) {
          dest[curDist] = foundPos;
      } else {
        // This should address a rare bug where we don't select a valid index. This likely occurs when
        // due to floating point arithmetic rounding errors, our cumulative sum does not add up to 1, but
        // and our uniform sample is greater than this value. In this case we likely have unitialized memory
        // in dest[curDist]. So basically we will loop through the distribution and pick the largest index
        // where the distribution is non-zero. This is obviously terribly inefficient, but due to the
        // rarity in which this occurs, this should not be an issue.
        for (int cat = categories - 1; cat >= 0; --cat) {
          if (dist[curDist * stride_dist + cat * stride_categories] > zero) {
            dest[curDist] = cat;
            break;
          }
        }
      }
    }
  }
}

void multinomial_with_replacement_kernel_impl(
    Tensor& result,
    const Tensor& self,
    const int64_t n_sample,
    c10::optional<Generator> generator) {
  auto gen = get_generator_or_default<CUDAGeneratorImpl>(generator, cuda::detail::getDefaultCUDAGenerator());

  int inputSize = self.dim();
  int64_t numDist =
      inputSize == 1 ? 1 : self.size(0);
  int numCategories =
      inputSize == 1 ? self.size(0) : self.size(1);

  // Restructure data for 2d
  auto self_v = inputSize == 1 ? self.view({numDist, numCategories}) : self;

  result.resize_({numDist, n_sample});

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(self_v.scalar_type(), "multinomial_kernel_cuda", [&] {
    using accscalar_t = at::acc_type<scalar_t, true>;
    auto props = at::cuda::getCurrentDeviceProperties();
    TORCH_CHECK(props != nullptr);
    int numSM = props->multiProcessorCount;
    int maxThreads = props->maxThreadsPerBlock;
    int maxShared = props->sharedMemPerBlock;

    int warp_size = at::cuda::warp_size();
    int requiredWarps = at::ceil_div(numCategories, warp_size);
    int requiredThreads = std::min(maxThreads, requiredWarps * warp_size);
    int requiredShared = requiredThreads * sizeof(accscalar_t);

    if (n_sample == 1 && maxShared >= requiredShared) {
      // Optimized allocation-free implementation
      // To exploit greater parallelism for the sampling, generate the
      // Uniform random samples in a separate kernel launch, into
      // temporarily allocated memory. The device RNG is thread-limited
      Tensor sampled = at::detail::empty_cuda({numDist, n_sample}, self_v.options());
      at::native::uniform_(sampled, 0.0, 1.0, generator);

      dim3 block(requiredThreads);
      dim3 grid(std::min(static_cast<int>(numDist), numSM * 4));

      sampleMultinomialOnce<scalar_t, accscalar_t>
          <<<grid, block,
          requiredShared,
          at::cuda::getCurrentCUDAStream()>>>(
              result.mutable_data_ptr<int64_t>(),
                  numDist,
                  numCategories,
                  sampled.const_data_ptr<scalar_t>(),
                  self_v.const_data_ptr<scalar_t>(),
                  self_v.stride(0),
                  self_v.stride(1)
          );
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    } else {
      // Generic, slow implementation with memory allocations

      // For sampling without replacement, we modify the distribution
      // for subsequent samples in this space
      Tensor origDist = native::empty_like(
          self_v,
          c10::nullopt /* dtype */,
          c10::nullopt /* layout */,
          c10::nullopt /* device */,
          c10::nullopt /* pin_memory */,
          LEGACY_CONTIGUOUS_MEMORY_FORMAT);
      origDist.copy_(self_v);

      Tensor normDist = native::empty_like(
          self_v,
          c10::nullopt /* dtype */,
          c10::nullopt /* layout */,
          c10::nullopt /* device */,
          c10::nullopt /* pin_memory */,
          LEGACY_CONTIGUOUS_MEMORY_FORMAT);

      Tensor prefixSum = native::empty_like(
          self_v,
          c10::nullopt /* dtype */,
          c10::nullopt /* layout */,
          c10::nullopt /* device */,
          c10::nullopt /* pin_memory */,
          LEGACY_CONTIGUOUS_MEMORY_FORMAT);

      // Renorm along rows
      normDist.copy_(origDist);
      renormRows(normDist);

      // Prefix sum along rows
      at::cuda::cumsum_out(prefixSum, normDist, 1);

      PhiloxCudaState rng_engine_inputs;

        // Binary search is warp divergent (so effectively we're running
        // with just a single thread), but for better utilization,
        // we need each block to have at least 4 warps.
        dim3 block(128);

        // Each block will generate a sample from one
        // distribution concurrently.
        int grid_y=std::min<int>(numDist, at::cuda::getCurrentDeviceProperties()->maxGridSize[1]);
        dim3 grid((n_sample-1)/block.x+1, grid_y);
        {
          // See Note [Acquire lock when using random generators]
          std::lock_guard<std::mutex> lock(gen->mutex_);

          // each thread generates a single sample for (numdist/numblocks.y) distributions, however, since we have to use
          // curand_uniform4 (See Note [Register spilling in curand call for CUDA < 10]),
          // offset is 4 times that.
          auto offset = ((numDist-1)/grid.y+1)*4;
          rng_engine_inputs = gen->philox_cuda_state(offset);
        }
        // Sample with replacement

        sampleMultinomialWithReplacement
            <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
                rng_engine_inputs,
                n_sample,
                result.mutable_data_ptr<int64_t>(),
                numDist, numCategories,
                prefixSum.const_data_ptr<scalar_t>(),
                normDist.const_data_ptr<scalar_t>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
  });

  if (inputSize == 1) {
    result.resize_({n_sample});
  }
}
}

REGISTER_DISPATCH(
    multinomial_with_replacement_stub,
    &multinomial_with_replacement_kernel_impl);
} // namespace at::native
