#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/LegacyTHFunctionsCUDA.h>
#include <ATen/native/UnaryOps.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <ATen/native/cuda/LaunchUtils.h>
#include <ATen/AccumulateType.h>

#include <THC/THCReduceApplyUtils.cuh>
#include <THC/THCTensorMathReduce.cuh>
#include <THC/THCNumerics.cuh>

#include <curand.h>
#include <curand_kernel.h>
#include <curand_philox4x32_x.h>

namespace at { namespace native {

namespace {

#define MAX_NUM_BLOCKS 200

// Normalizes the L1 norm of every row to 1; used by multinomial
template <typename scalar_t>
#ifdef __HIP_PLATFORM_HCC__
C10_LAUNCH_BOUNDS_1(1024)
#endif
__global__ void renormRowsL1(scalar_t* dist, long rows, long cols) {
  extern __shared__  unsigned char my_smem[];
  scalar_t *smem = reinterpret_cast<scalar_t *>(my_smem);
  scalar_t zero = static_cast<scalar_t>(0);
  scalar_t val;
  for (int64_t row = blockIdx.x; row < rows; row += gridDim.x) {
    scalar_t sum = static_cast<scalar_t>(0);
    for (int64_t col = threadIdx.x; col < cols; col += blockDim.x) {
      val = dist[row * cols + col];
      CUDA_ALWAYS_ASSERT(!THCNumerics<scalar_t>::lt(val, zero)); // ! < 0 for NaN handling
      sum = sum + val;
    }

    sum = reduceBlock(smem, blockDim.x, sum, ReduceAdd<scalar_t>(), zero);
    if (threadIdx.x == 0) {
      CUDA_ALWAYS_ASSERT(!THCNumerics<scalar_t>::lt(val, zero)); // ! < 0 for NaN handling
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
  CUDA_ALWAYS_ASSERT(props != NULL);
  int numSM = props->multiProcessorCount;
  int maxThreads = props->maxThreadsPerBlock;

  dim3 grid(rows < numSM * 4 ? rows : numSM * 4);
  dim3 block(cols < maxThreads ? cols : maxThreads);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(t.scalar_type(), "renormRows_cuda", [&] {
    renormRowsL1<scalar_t>
        <<<grid, block, block.x * sizeof(scalar_t),
        at::cuda::getCurrentCUDAStream()>>>(t.data_ptr<scalar_t>(),
            rows, cols);
  });
}

template <typename scalar_t>
__device__ int binarySearchForMultinomial(scalar_t* cumdist,
                                          scalar_t* dist,
                                          int size,
                                          scalar_t val) {
  int start = 0;
  int end = size;
  // cumdist[size - 1] = 0 => all zero prob dist
  CUDA_ALWAYS_ASSERT(cumdist[size - 1] > static_cast<scalar_t>(0));

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
sampleMultinomialWithReplacement(std::pair<uint64_t, uint64_t> seeds,
                                 int totalSamples,
                                 int64_t* dest,
                                 int64_t distributions,
                                 int categories,
                                 scalar_t* normDistPrefixSum,
                                 scalar_t* normDist) {
  // At the moment, each warp computes one sample value in the binary
  // search due to divergence. It seems possible to compute multiple
  // values and limit divergence though later on.

  // global index formula for 1D grid of 2D blocks
  int idx = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;

  curandStatePhilox4_32_10_t state;
  curand_init(seeds.first, idx, seeds.second, &state);

  // The block determines the distribution for which we generate a point
  for (int64_t curDist = blockIdx.x;
       curDist < distributions;
       curDist += gridDim.x) {
    for (int sampleBase = 0;
         sampleBase < totalSamples; sampleBase += blockDim.y) {
      // The warp determines the sample
      int sample = sampleBase + threadIdx.y;

      // All threads participate in this
      auto rand = curand_uniform4(&state);
      scalar_t r = static_cast<scalar_t>(rand.x);

      if (threadIdx.x == 0 && sample < totalSamples) {
        // Find the bucket that a uniform sample lies in
        int choice = binarySearchForMultinomial<scalar_t>(
            normDistPrefixSum + curDist * categories,
            normDist + curDist * categories,
            categories,
            r);

        // Torch indices are 1-based
        dest[curDist * totalSamples + sample] = choice;
      }
    }
  }
}

template <typename scalar_t>
__global__ void
sampleMultinomialWithoutReplacement(std::pair<uint64_t, uint64_t> seeds,
                                    int totalSamples,
                                    int sample,
                                    int64_t* dest,
                                    int64_t distributions,
                                    int categories,
                                    scalar_t* origDist,
                                    scalar_t* normDistPrefixSum) {
  // At the moment, each warp computes one sample value in the binary
  // search due to divergence. It seems possible to compute multiple
  // values and limit divergence though later on.

  // global index formula for 1D grid of 2D blocks
  int idx = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;

  curandStatePhilox4_32_10_t state;
  curand_init(seeds.first, idx, seeds.second, &state);

  // The block and warp determines the distribution for which we
  // generate a point
  for (int64_t curDistBase = blockIdx.x * blockDim.y;
       curDistBase < distributions;
       curDistBase += gridDim.x * blockDim.y) {
    // The warp determines the distribution
    int64_t curDist = curDistBase + threadIdx.y;

    // All threads must participate in this
    auto rand = curand_uniform4(&state);
    scalar_t r = static_cast<scalar_t>(rand.x);

    if (threadIdx.x == 0 && curDist < distributions) {
      // Find the bucket that a uniform sample lies in
      int choice = binarySearchForMultinomial<scalar_t>(
          normDistPrefixSum + curDist * categories,
          origDist + curDist * categories,
          categories,
          r);

      // Torch indices are 1-based
      dest[curDist * totalSamples + sample] = choice;

      // Without replacement, so update the original probability so it
      // is not considered a second time
      origDist[curDist * categories + choice] = static_cast<scalar_t>(0);
    }
  }
}

template <typename scalar_t, typename accscalar_t>
#ifdef __HIP_PLATFORM_HCC__
C10_LAUNCH_BOUNDS_1(1024)
#endif
__global__ void
sampleMultinomialOnce(int64_t* dest,
                      int64_t distributions,
                      int categories,
                      scalar_t* sampled,
                      scalar_t* dist,
                      int stride_dist,        // dist->stride(0)
                      int stride_categories   // dist->stride(1)
) {
  extern __shared__  unsigned char my_smem[];
  __shared__ bool found;

  // Shared Memory hold blockdim.x T for holding the cumulative sum,
  // blockDim.x AccT for normalizing the probabilities,
  scalar_t *smem = reinterpret_cast<scalar_t *>(my_smem);
  accscalar_t *asmem = reinterpret_cast<accscalar_t *>(&my_smem[blockDim.x * sizeof(scalar_t)]);

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
      CUDA_ALWAYS_ASSERT(val >= zero);
      CUDA_ALWAYS_ASSERT(!THCNumerics<scalar_t>::isinf(val));
      CUDA_ALWAYS_ASSERT(!THCNumerics<scalar_t>::isnan(val));
      sum = sum + static_cast<accscalar_t>(val);
    }

    // threadIdx.x == 0 has the sum value from this
    sum = reduceBlock(asmem, blockDim.x, sum, ReduceAdd<accscalar_t>(), accZero);

    // Broadcast sum and sample value
    if (threadIdx.x == 0) {
      // Make sure the sum of our distribution didn't overflow
      CUDA_ALWAYS_ASSERT(!THCNumerics<accscalar_t>::isinf(sum));
      CUDA_ALWAYS_ASSERT(sum > accZero);

      asmem[0] = sum;
      smem[0] = sampled[curDist];
    }
    __syncthreads();

    sum = asmem[0];
    scalar_t sample = smem[0];
    __syncthreads();

    if (sum == accZero) {
      // Choose the first element
      if (threadIdx.x == 0) {
        dest[curDist] = 0;
      }

      continue;
    }

    int chunks = (categories + (int)blockDim.x - 1) / blockDim.x;
    scalar_t prevHighProb = zero;
    found = false;

    for (int chunk = 0; chunk < chunks && !found; ++chunk) {
      // All threads in bounds load a value
      int cat = chunk * blockDim.x + threadIdx.x;

      accscalar_t a_dist_val = cat < categories ?
                               static_cast<accscalar_t>(dist[curDist * stride_dist + cat * stride_categories]) / sum :
                               accZero;
      scalar_t dist_val = static_cast<scalar_t>(a_dist_val);

      smem[threadIdx.x] = dist_val;
      __syncthreads();

      // Perform an inclusive prefix sum of the shared memory contents
      for (int offset = 1; offset < blockDim.x; offset *= 2) {
        scalar_t val = zero;

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
      scalar_t curBucket = smem[threadIdx.x] + prevHighProb;
      scalar_t prevBucket =
          threadIdx.x == 0 ? prevHighProb :
          smem[threadIdx.x - 1] + prevHighProb;
      bool inBucket =
          (cat < categories) &&
          (!(sample >= curBucket) &&
           (sample >= prevBucket) &&
           (dist_val > zero));

      if (inBucket) {
        // We're done; we have the sample
        // Torch indices are 1-based
        dest[curDist] = cat;
        found = true;
      }

      // Store the previous scan's high value for future use
      prevHighProb = prevHighProb + smem[blockDim.x - 1];

      __syncthreads();
    }

    if (threadIdx.x == 0 && !found) {
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

void multinomial_kernel_impl(Tensor& result, const Tensor& self, const int64_t n_sample, const bool with_replacement, Generator generator) {
  auto gen = get_generator_or_default<CUDAGenerator>(generator, cuda::detail::getDefaultCUDAGenerator());

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
    CUDA_ALWAYS_ASSERT(props != NULL);
    int numSM = props->multiProcessorCount;
    int maxThreads = props->maxThreadsPerBlock;
    int maxShared = props->sharedMemPerBlock;
    int requiredShared = (numCategories < maxThreads ? numCategories : maxThreads)
                         * (sizeof(scalar_t) + sizeof(accscalar_t));

    if (n_sample == 1 && maxShared >= requiredShared) {
      // Optimized allocation-free implementation
      // To exploit greater parallelism for the sampling, generate the
      // Uniform random samples in a separate kernel launch, into
      // temporarily allocated memory. The device RNG is thread-limited
      Tensor sampled = native::empty_cuda({numDist, n_sample}, self_v.options());
      at::native::uniform_cuda_(sampled, 0.0, 1.0, generator);

      dim3 block(numCategories < maxThreads ? numCategories : maxThreads);
      dim3 grid(numDist < numSM * 4 ? numDist : numSM * 4);

      sampleMultinomialOnce<scalar_t, accscalar_t>
          <<<grid, block,
          requiredShared,
          at::cuda::getCurrentCUDAStream()>>>(
              result.data_ptr<int64_t>(),
                  numDist,
                  numCategories,
                  sampled.data_ptr<scalar_t>(),
                  self_v.data_ptr<scalar_t>(),
                  self_v.stride(0),
                  self_v.stride(1)
          );
    } else {
      // Generic, slow implementation with memory allocations

      // For sampling without replacement, we modify the distribution
      // for subsequent samples in this space
      Tensor origDist = native::empty_like(self_v, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
      origDist.copy_(self_v);

      Tensor normDist = native::empty_like(self_v, LEGACY_CONTIGUOUS_MEMORY_FORMAT);

      Tensor prefixSum = native::empty_like(self_v, LEGACY_CONTIGUOUS_MEMORY_FORMAT);

      // Renorm along rows
      normDist.copy_(origDist);
      renormRows(normDist);

      // Prefix sum along rows
      legacy::cuda::_th_cumsum_out(prefixSum, normDist, 1);

      std::pair<uint64_t, uint64_t> rng_engine_inputs;

      if (with_replacement) {
        {
          // See Note [Acquire lock when using random generators]
          std::lock_guard<std::mutex> lock(gen->mutex_);

          // each thread will utilize one random, however, since we have to use
          // curand_uniform4 (See Note [Register spilling in curand call for CUDA < 10]),
          // offset is 4.
          rng_engine_inputs = gen->philox_engine_inputs(4);
        }
        // Sample with replacement

        // Binary search is warp divergent (so effectively we're running
        // with just a single thread), but for better utilization,
        // we need each block to have at least 4 warps.
        dim3 block(32, 4);

        // Each warp in a block will generate a sample from one
        // distribution concurrently.
        dim3 grid(numDist < MAX_NUM_BLOCKS ? numDist : MAX_NUM_BLOCKS);

        sampleMultinomialWithReplacement
            <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
            rng_engine_inputs,
                n_sample,
                result.data_ptr<int64_t>(),
                numDist, numCategories,
                prefixSum.data_ptr<scalar_t>(),
                normDist.data_ptr<scalar_t>());
      } else {
        // Sample without replacement

        // Binary search is warp divergent (so effectively we're running
        // with just a single thread), but for better utilization,
        // we need each block to have at least 4 warps.
        dim3 block(32, 4);

        // Each warp in a block will generate a sample from a different
        // distribution concurrently.
        ptrdiff_t numBlocks = (numDist + 4 - 1) / 4;
        dim3 grid(numBlocks < MAX_NUM_BLOCKS ? numBlocks : MAX_NUM_BLOCKS);

        for (int sample = 0; sample < n_sample; ++sample) {
          if (sample > 0) {
            // Update probabilities
            // Renorm along rows
            normDist.copy_(origDist);
            renormRows(normDist);

            // Prefix sum along rows
            legacy::cuda::_th_cumsum_out(prefixSum, normDist, 1);
          }
          {
            // See Note [Acquire lock when using random generators]
            std::lock_guard<std::mutex> lock(gen->mutex_);

            // each thread will utilize one random, however, since we have to use
            // curand_uniform4 (See Note [Register spilling in curand call for CUDA < 10]),
            // offset is 4.
            rng_engine_inputs = gen->philox_engine_inputs(4);
          }

          // The kernel can only draw one sample before we have to
          // recalculate our distribution
          sampleMultinomialWithoutReplacement
              <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
              rng_engine_inputs,
                  n_sample,
                  sample,
                  result.data_ptr<int64_t>(),
                  numDist, numCategories,
                  origDist.data_ptr<scalar_t>(),
                  prefixSum.data_ptr<scalar_t>());
        }
      }
    }
  });

  if (inputSize == 1) {
    result.resize_({n_sample});
  }
}
}

REGISTER_DISPATCH(multinomial_stub, &multinomial_kernel_impl);

}}
