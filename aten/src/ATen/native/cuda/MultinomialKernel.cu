#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/ceil_div.h>
#include <ATen/Dispatch.h>
#include <ATen/Utils.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/EmptyTensor.h>
#include <ATen/cuda/detail/KernelUtils.h>
#include <ATen/native/UnaryOps.h>
#include <ATen/cuda/CUDAGraphsUtils.cuh>
#include <ATen/native/cuda/block_reduce.cuh>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/CUDAFunctions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/empty_native.h>
#include <ATen/ops/cumsum_cuda_dispatch.h>
#endif

#include <curand.h>
#include <curand_kernel.h>
#include <curand_philox4x32_x.h>
#include <limits>
#include <type_traits>

namespace at::native {

namespace {

// Normalizes the L1 norm of every row to 1; used by multinomial
template <typename scalar_t>
C10_LAUNCH_BOUNDS_1(cuda::detail::CUDA_NUM_THREADS)
__global__ void renormRowsL1(scalar_t* dist, int64_t rows, int64_t cols) {
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
      props->maxThreadsPerBlock, cuda_utils::kCUDABlockReduceMaxThreads());

  int warp_size = at::cuda::warp_size();
  dim3 grid(rows < numSM * 4 ? rows : numSM * 4);
  dim3 block(std::min(maxThreads, warp_size * ceil_div(cols, int64_t{warp_size})));

  AT_DISPATCH_FLOATING_TYPES(t.scalar_type(), "renormRows_cuda", [&] {
    renormRowsL1<scalar_t>
        <<<grid, block, (block.x / warp_size) * sizeof(scalar_t),
        at::cuda::getCurrentCUDAStream()>>>(t.mutable_data_ptr<scalar_t>(),
            rows, cols);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  });
}

template <typename scalar_t>
__device__ int64_t binarySearchForMultinomial(const scalar_t* cumdist,
                                              const scalar_t* dist,
                                              int64_t size,
                                              scalar_t val) {
  int64_t start = 0;
  int64_t end = size;
  // cumdist[size - 1] = 0 => all zero prob dist
  CUDA_KERNEL_ASSERT(cumdist[size - 1] > static_cast<scalar_t>(0));

  while (end - start > 0) {
    int64_t mid = start + (end - start) / 2;

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
                                 int64_t categories,
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
      scalar_t r;
      if constexpr (std::is_same_v<scalar_t, double>) {
        // A float uniform has only 24 bits of resolution, too coarse for a double CDF
        r = curand_uniform2_double(&state).x;
      } else {
        r = static_cast<scalar_t>(curand_uniform4(&state).x);
      }

      // Find the bucket that a uniform sample lies in
      int64_t choice = binarySearchForMultinomial<scalar_t>(
          normDistPrefixSum + curDist * categories,
          normDist + curDist * categories,
          categories,
          r);

      dest[curDist * totalSamples + sample] = choice;

    }
  }
}

void multinomial_with_replacement_kernel_impl(
    Tensor& result,
    const Tensor& self,
    const int64_t n_sample,
    std::optional<Generator> generator) {
  auto gen = get_generator_or_default<CUDAGeneratorImpl>(generator, cuda::detail::getDefaultCUDAGenerator());

  int inputSize = self.dim();
  int64_t numDist =
      inputSize == 1 ? 1 : self.size(0);
  int64_t numCategories =
      inputSize == 1 ? self.size(0) : self.size(1);

  // Restructure data for 2d
  auto self_v = inputSize == 1 ? self.view({numDist, numCategories}) : self;
  // Half row sums overflow and bfloat16 CDFs are too coarse, so both build the CDF in
  // float; a float CDF cannot give distinct buckets to more than 2^24 categories
  const auto inputType = self_v.scalar_type();
  const auto cdfType = numCategories > (int64_t{1} << std::numeric_limits<float>::digits)
      ? kDouble
      : (inputType == kHalf || inputType == kBFloat16 ? kFloat : inputType);

  result.resize_({numDist, n_sample});

  AT_DISPATCH_FLOATING_TYPES(cdfType, "multinomial_kernel_cuda", [&] {
    Tensor normDist = at::detail::empty_cuda({numDist, numCategories}, self_v.options().dtype(cdfType));
    Tensor prefixSum = at::detail::empty_cuda({numDist, numCategories}, normDist.options());

    // Renorm along rows
    normDist.copy_(self_v);
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
