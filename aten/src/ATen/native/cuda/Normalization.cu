#include <THC/THCDeviceUtils.cuh>
#include <THC/THCGeneral.h>
#include "ATen/ATen.h"
#include "ATen/AccumulateType.h"
#include "ATen/cuda/CUDAContext.h"

namespace at { namespace native {

namespace {


#if defined(__HIP_PLATFORM_HCC__)
constexpr int WARP_SIZE = 64;

// take these out when ROCm implements std:: math functions
#include <math.h>
template <typename scalar_t>
static __forceinline__ __device__ scalar_t device_sqrt(scalar_t val);

template <>
__forceinline__ __device__ float device_sqrt(float val) {
  return ::sqrtf(val);
}

template <>
__forceinline__ __device__ double device_sqrt(double val) {
  return ::sqrt(val);
}

#else
constexpr int WARP_SIZE = 32;

template<typename scalar_t>
__forceinline__ __device__ double device_sqrt(scalar_t val) {
  return std::sqrt(val);
}
#endif

// The maximum number of threads in a block
#if defined(__HIP_PLATFORM_HCC__)
constexpr int MAX_BLOCK_SIZE = 256;
#else
constexpr int MAX_BLOCK_SIZE = 512;
#endif

// Number of threads in a block given an input size up to MAX_BLOCK_SIZE
static int getNumThreads(int nElem) {
#if defined(__HIP_PLATFORM_HCC__)
  int threadSizes[5] = { 16, 32, 64, 128, MAX_BLOCK_SIZE };
#else
  int threadSizes[5] = { 32, 64, 128, 256, MAX_BLOCK_SIZE };
#endif
  for (int i = 0; i != 5; ++i) {
    if (nElem <= threadSizes[i]) {
      return threadSizes[i];
    }
  }
  return MAX_BLOCK_SIZE;
}

// Returns the index of the most significant 1 bit in `val`.
__device__ __forceinline__ int getMSB(int val) {
  return 31 - __clz(val);
}

template <typename scalar_t, typename accscalar_t>
struct Float2 {
  accscalar_t v1, v2;
  __device__ Float2() {}
  __device__ Float2(scalar_t v1, scalar_t v2) : v1(static_cast<accscalar_t>(v1)), v2(static_cast<accscalar_t>(v2)) {}
  __device__ Float2(int v) : v1(static_cast<accscalar_t>(v)), v2(static_cast<accscalar_t>(v)) {}
  __device__ Float2& operator+=(const Float2& a) {
    v1 += a.v1;
    v2 += a.v2;
    return *this;
  }
};

template <typename scalar_t, typename accscalar_t>
struct SumOp {
  __device__ SumOp(const PackedTensorAccessor<scalar_t, 3>& t) : tensor(t) {}
  __device__ __forceinline__ accscalar_t operator()(int batch, int plane, int n) {
    return static_cast<accscalar_t>(tensor[batch][plane][n]);
  }
  const PackedTensorAccessor<scalar_t, 3>& tensor;
};

template <typename scalar_t, typename accscalar_t>
struct VarOp {
  __device__ VarOp(accscalar_t m, const PackedTensorAccessor<scalar_t, 3>& t) : mean(m), tensor(t) {}
  __device__ __forceinline__ accscalar_t operator()(int batch, int plane, int n) {
    accscalar_t val = tensor[batch][plane][n];
    return (val - mean) * (val - mean);
  }
  const accscalar_t mean;
  const PackedTensorAccessor<scalar_t, 3>& tensor;
};

template <typename scalar_t, typename accscalar_t>
struct GradOp {
  __device__ GradOp(accscalar_t m, const PackedTensorAccessor<scalar_t, 3>& i, const PackedTensorAccessor<scalar_t, 3>& g)
    : mean(m), input(i), grad_output(g) {}
  __device__ __forceinline__ Float2<scalar_t, accscalar_t> operator()(int batch, int plane, int n) {
    accscalar_t g = grad_output[batch][plane][n];
    accscalar_t c = static_cast<accscalar_t>(input[batch][plane][n]) - mean;
    return Float2<scalar_t, accscalar_t>(g, g * c);
  }
  const accscalar_t mean;
  const PackedTensorAccessor<scalar_t, 3>& input;
  const PackedTensorAccessor<scalar_t, 3>& grad_output;
};

// Sum across all threads within a warp
template <typename T>
static __device__ __forceinline__ T warpSum(T val) {
#if __CUDA_ARCH__ >= 300
  for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
    val += WARP_SHFL_XOR(val, 1 << i, WARP_SIZE);
  }
#else
  __shared__ T values[MAX_BLOCK_SIZE];
  values[threadIdx.x] = val;
  __threadfence_block();
  const int base = (threadIdx.x / WARP_SIZE) * WARP_SIZE;
  for (int i = 1; i < WARP_SIZE; i++) {
    val += values[base + ((i + threadIdx.x) % WARP_SIZE)];
  }
#endif
  return val;
}

template <typename scalar_t, typename accscalar_t>
static __device__ __forceinline__ Float2<scalar_t, accscalar_t> warpSum(Float2<scalar_t, accscalar_t> value) {
  value.v1 = warpSum(value.v1);
  value.v2 = warpSum(value.v2);
  return value;
}

// Sum across (batch, x/y/z) applying Op() pointwise
// this works by first having each thread sum it's part
// of the data. Then there is a double-shuffeling reduction.
// First each warp (of WARP_SIZE threads) uses warpSum to reduce its
// data to the "warp leader", who writes its value into shared memory.
// Then a single warp reads the remaining (at most WARP_SIZE) items
// and reduces them using another warpSum.
// The implicit assumption is that there are no more
// than WARP_SIZE**2 threads.
template<typename scalar_t, typename Op, typename PTA>
__device__ scalar_t reduce(Op op, PTA tensor, int plane) {
  // first the reductions each thread does separately
  scalar_t sum = static_cast<scalar_t>(0);
  for (int batch = 0; batch < tensor.size(0); ++batch) {
    for (int x = threadIdx.x; x < tensor.size(2); x += blockDim.x) {
      sum += op(batch, plane, x);
    }
  }

  // first warpSum to get one value per thread to
  // one value per warp
  sum = warpSum(sum);

  // this writes each warps  item into shared memory
  // there are at most WARP_SIZE items left because
  // there are at most WARP_SIZE**2 threads at the beginning
  __shared__ scalar_t shared[WARP_SIZE];
  __syncthreads();
  if (threadIdx.x % WARP_SIZE == 0) {
    shared[threadIdx.x / WARP_SIZE] = sum;
  }
  if (threadIdx.x >= blockDim.x / WARP_SIZE && threadIdx.x < WARP_SIZE) {
    // zero out the other entries in shared
    shared[threadIdx.x] = (scalar_t)0;
  }
  __syncthreads();
  // now have a second warpSum to reduce the intermediate values
  // from shared memory to a single number. The very first
  // thread writes it to shared memory.

  if (threadIdx.x / WARP_SIZE == 0) {
    sum = warpSum(shared[threadIdx.x]);
    if (threadIdx.x == 0) {
      shared[0] = sum;
    }
  }
  __syncthreads();

  // Everyone picks it up, should be broadcast into the whole grad_input
  return shared[0];
}

template <typename scalar_t, typename accscalar_t, bool train>
__global__ void batch_norm_transform_input_kernel(
    const PackedTensorAccessor<scalar_t, 3> input,
    PackedTensorAccessor<scalar_t, 3> output,
    const PackedTensorAccessor<typename std::conditional<train, accscalar_t, scalar_t>::type, 1> mean_,
    const PackedTensorAccessor<typename std::conditional<train, accscalar_t, scalar_t>::type, 1> var_or_invstd,
    const PackedTensorAccessor<scalar_t, 1> weight,
    const PackedTensorAccessor<scalar_t, 1> bias,
    accscalar_t epsilon) {

  int plane = blockIdx.y * blockDim.y + threadIdx.y;
  int fstep = blockDim.x * gridDim.x;

  if (plane >= input.size(1)) {
    return;
  }

  accscalar_t gamma = weight.size(0) > 0 ? static_cast<accscalar_t>(weight[plane]) : static_cast<accscalar_t>(1);
  accscalar_t beta = bias.size(0) > 0 ? static_cast<accscalar_t>(bias[plane]) : static_cast<accscalar_t>(0);
  accscalar_t mean = static_cast<accscalar_t>(mean_[plane]);
  accscalar_t invstd;
  if (train) {
    invstd = var_or_invstd[plane];
  } else {
    invstd = static_cast<accscalar_t>(1) / device_sqrt(static_cast<accscalar_t>(var_or_invstd[plane]) + epsilon);
  }
  for (int64_t batch = blockIdx.z; batch < input.size(0); batch += gridDim.z) {
    auto o = output[batch][plane];

    for (int64_t feature = threadIdx.x + blockDim.x * blockIdx.x; feature < input.size(2); feature += fstep) {
      o[feature] = static_cast<scalar_t>(gamma * (input[batch][plane][feature] - mean) * invstd + beta);
    }
  }
}


template <typename scalar_t, typename accscalar_t>
__global__ void batch_norm_collect_statistics_kernel(
    const PackedTensorAccessor<scalar_t, 3> input,
    const accscalar_t epsilon,
    const accscalar_t momentum,
    PackedTensorAccessor<scalar_t, 1> running_mean,
    PackedTensorAccessor<scalar_t, 1> running_var,
    PackedTensorAccessor<accscalar_t, 1> save_mean,
    PackedTensorAccessor<accscalar_t, 1> save_invstd) {

  int plane = blockIdx.x;
  int N = input.size(0) * input.size(2);

  accscalar_t norm = accscalar_t(1) / N;

  // Compute the mean and variance across (batch, x/y/z)
  accscalar_t mean = reduce<accscalar_t>(SumOp<scalar_t, accscalar_t>(input), input, plane) * norm;
  __syncthreads();
  accscalar_t varN = reduce<accscalar_t>(VarOp<scalar_t, accscalar_t>(mean, input), input, plane);

  // Save the mean, variance, and moving averages
  if (threadIdx.x == 0) {
    accscalar_t invstd = 0;
    if (varN != static_cast<accscalar_t>(0) || epsilon != static_cast<accscalar_t>(0)) {
      invstd = static_cast<accscalar_t>(1) / device_sqrt(varN * norm + epsilon);
    }
    save_mean[plane] = mean;
    save_invstd[plane] = invstd;
    if (running_mean.data() != NULL) {
      running_mean[plane] = static_cast<scalar_t>((1 - momentum) * running_mean[plane] + momentum * mean);
    }
    if (running_var.data() != NULL) {
      accscalar_t unbiasedVar = varN / (N - 1);
      running_var[plane] = static_cast<scalar_t>((1 - momentum) * running_var[plane] + momentum * unbiasedVar);
    }
  }

}

template <typename scalar_t, typename accscalar_t>
__global__ void batch_norm_backward_kernel(
    const PackedTensorAccessor<scalar_t, 3> input,
    const PackedTensorAccessor<scalar_t, 3> grad_output,
    PackedTensorAccessor<scalar_t, 3> grad_input,
    PackedTensorAccessor<scalar_t, 1> grad_weight,
    PackedTensorAccessor<scalar_t, 1> grad_bias,
    const PackedTensorAccessor<scalar_t, 1> weight,
    const PackedTensorAccessor<scalar_t, 1> running_mean,
    const PackedTensorAccessor<scalar_t, 1> running_var,
    const PackedTensorAccessor<accscalar_t, 1> save_mean,
    const PackedTensorAccessor<accscalar_t, 1> save_invstd,
    bool train,
    accscalar_t epsilon) {

  int plane = blockIdx.x;
  int N = grad_output.size(0) * grad_output.size(2);

  accscalar_t mean, invstd;
  if (train) {
    mean = save_mean[plane];
    invstd = save_invstd[plane];
  } else {
    mean = static_cast<accscalar_t>(running_mean[plane]);
    invstd = static_cast<accscalar_t>(1) / device_sqrt(static_cast<accscalar_t>(running_var[plane]) + epsilon);
  }

  accscalar_t weight_val = weight.size(0) > 0 ? static_cast<accscalar_t>(weight[plane]) : accscalar_t(1);
  accscalar_t norm = accscalar_t(1) / N;

  // Compute two values across (batch, x/y/z) in one pass:
  // 1. Sum(grad_output)
  // 2. DotProduct(input - mean, grad_output)
  GradOp<scalar_t, accscalar_t> g(mean, input, grad_output);
  Float2<scalar_t, accscalar_t> res = reduce<Float2<scalar_t, accscalar_t>, GradOp<scalar_t, accscalar_t>, PackedTensorAccessor<scalar_t, 3>>(g, grad_output, plane);
  accscalar_t grad_output_sum = res.v1;
  accscalar_t dot_p = res.v2;

  accscalar_t grad_mean = grad_output_sum * norm;
  accscalar_t proj_scale = dot_p * norm * invstd * invstd;
  accscalar_t grad_scale = invstd * weight_val;

  if (grad_input.data() != NULL) {
    for (int batch = 0; batch < grad_output.size(0); ++batch) {
      for (int x = threadIdx.x; x < grad_output.size(2); x += blockDim.x) {
        scalar_t go = grad_output[batch][plane][x];
        if (train) {
          scalar_t inp = input[batch][plane][x];
          accscalar_t proj = (inp - mean) * proj_scale;
          grad_input[batch][plane][x] = static_cast<scalar_t>((go - proj - grad_mean) * grad_scale);
        } else {
          grad_input[batch][plane][x] = static_cast<scalar_t>(go * grad_scale);
        }
      }
    }
  }

  if (grad_weight.size(0) > 0) {
    if (threadIdx.x == 0) {
      grad_weight[plane] = static_cast<scalar_t>(dot_p * invstd);
    }
  }

  if (grad_bias.size(0) > 0) {
    if (threadIdx.x == 0) {
      grad_bias[plane] = static_cast<scalar_t>(grad_output_sum);
    }
  }
}

// TensorAccessor in which the last dimensions are collapsed or expanded as needed
template <typename scalar_t, int64_t dim>
static PackedTensorAccessor<scalar_t, dim> reshaped_packed_accessor(const Tensor& t) {
  constexpr int too_small_feature_set = 16;  // this is the maximum feature dimension when we swap
  constexpr int few_planes = 256;
  // undefined...
  if (! t.defined()) {
    const std::vector<int64_t> zeros(dim);
    return PackedTensorAccessor<scalar_t, dim>(nullptr, zeros.data(), zeros.data());
  }
  int64_t in_dim = t.dim();
  if (in_dim == dim && (dim < 3 || (t.size(0) < t.size(2)) || (t.size(2) >= too_small_feature_set))) { // easy, if we don't need the evil trick
    return t.packed_accessor<scalar_t, dim>();
  }

  AT_CHECK(in_dim < dim || t.is_contiguous(), "need contiguous or <= 3d tensor");
  std::vector<int64_t> sizes(dim);
  std::vector<int64_t> strides(dim);
  for (int i = 0; i < in_dim || i < dim; ++i) {
    if (i < dim && i < in_dim) {
      sizes[i] = t.size(i);
      strides[i] = t.stride(i);
    } else if (i < dim) {
      sizes[i] = 1;
      strides[i] = 0;
    } else {
      sizes[dim - 1] *= t.size(i);
      strides[dim -1] = 1;
    }
  }
  // evil trick to get adjusted 2d tensors to have large dimension last
  if (dim == 3 && sizes[0] > sizes[2] && sizes[2] < too_small_feature_set &&
      sizes[1] <= few_planes) {
    std::swap(sizes[0], sizes[2]);
    std::swap(strides[0], strides[2]);
  }
  return PackedTensorAccessor<scalar_t, dim>(t.data<scalar_t>(), sizes.data(), strides.data());
}

template<typename scalar_t>
std::tuple<Tensor, Tensor, Tensor> batch_norm_cuda_template(const Tensor& input_, const Tensor& weight_, const Tensor& bias_,
							    const Tensor& running_mean_, const Tensor& running_var_,
							    bool train, double momentum, double epsilon) {

  using accscalar_t = at::acc_type<scalar_t, true>;
  Tensor output_= at::empty_like(input_);
  int64_t n_input = input_.size(1);
  Tensor save_mean_;
  Tensor save_invstd_;
  auto input_options = input_.options();
  if (input_options.dtype() == ScalarType::Half) {
    input_options = input_options.dtype(ScalarType::Float);
  }
  if (train) {
    save_mean_ = at::empty({n_input}, input_options);
    save_invstd_ = at::empty({n_input}, input_options);
  } else {
    save_mean_ = at::empty({0}, input_options);
    save_invstd_ = at::empty({0}, input_options);
  }
  auto input = reshaped_packed_accessor<scalar_t, 3>(input_);
  auto output = reshaped_packed_accessor<scalar_t, 3>(output_);
  auto weight = reshaped_packed_accessor<scalar_t, 1>(weight_);
  auto bias = reshaped_packed_accessor<scalar_t, 1>(bias_);
  auto running_mean = reshaped_packed_accessor<scalar_t, 1>(running_mean_);
  auto running_var = reshaped_packed_accessor<scalar_t, 1>(running_var_);
  auto save_mean = reshaped_packed_accessor<accscalar_t, 1>(save_mean_);
  auto save_invstd = reshaped_packed_accessor<accscalar_t, 1>(save_invstd_);
  auto stream = at::cuda::getCurrentCUDAStream();

  constexpr int max_blocks = 60000;
  int feature_threads = std::min(getNumThreads(input.size(2)), 128);
  int plane_threads = MAX_BLOCK_SIZE / feature_threads;
  int plane_blocks = std::min<int>((input.size(1)+plane_threads-1)/plane_threads, max_blocks);
  int feature_blocks = std::min<int>((input.size(2)+feature_threads-1)/feature_threads, max_blocks / plane_blocks);
  dim3 blocks(feature_blocks, plane_blocks);
  dim3 threads(feature_threads, plane_threads);
  if (!train) {
    batch_norm_transform_input_kernel<scalar_t, accscalar_t, false> <<<blocks, threads, 0, stream>>>
      (input, output, running_mean, running_var, weight, bias, epsilon);
  } else {
    dim3 blocks_red(input.size(1));
    dim3 threads_red(getNumThreads(input.size(2)));
    batch_norm_collect_statistics_kernel<scalar_t, accscalar_t> <<<blocks_red, threads_red, 0, stream>>>
      (input, epsilon, momentum, running_mean, running_var, save_mean, save_invstd);
    batch_norm_transform_input_kernel<scalar_t, accscalar_t, true> <<<blocks, threads, 0, stream>>>
      (input, output, save_mean, save_invstd, weight, bias, epsilon);
  }
  THCudaCheck(cudaGetLastError());
  return std::make_tuple(output_, save_mean_, save_invstd_);
}

template<typename scalar_t>
std::tuple<Tensor, Tensor, Tensor> batch_norm_backward_cuda_template(const Tensor& grad_out_, const Tensor& input_, const Tensor& weight_,
								     const Tensor& running_mean_, const Tensor& running_var_, const Tensor& save_mean_, const Tensor& save_invstd_,
								     bool train, double epsilon, std::array<bool,3> grad_input_mask) {

  using accscalar_t = at::acc_type<scalar_t, true>;
  Tensor grad_input_;
  Tensor grad_weight_;
  Tensor grad_bias_;
  if (grad_input_mask[0]) {
    grad_input_ = at::empty_like(input_);
  }
  if (grad_input_mask[1]) {
    grad_weight_ = at::empty_like(weight_);
  }
  if (grad_input_mask[2]) {
    grad_bias_ = at::empty_like(weight_);
  }

  auto grad_output = reshaped_packed_accessor<scalar_t, 3>(grad_out_);
  auto input = reshaped_packed_accessor<scalar_t, 3>(input_);
  auto grad_input = reshaped_packed_accessor<scalar_t, 3>(grad_input_);
  auto weight = reshaped_packed_accessor<scalar_t, 1>(weight_);
  auto grad_weight = reshaped_packed_accessor<scalar_t, 1>(grad_weight_);
  auto grad_bias = reshaped_packed_accessor<scalar_t, 1>(grad_bias_);
  auto running_mean = reshaped_packed_accessor<scalar_t, 1>(running_mean_);
  auto running_var = reshaped_packed_accessor<scalar_t, 1>( running_var_);
  auto save_mean = reshaped_packed_accessor<accscalar_t, 1>(save_mean_);
  auto save_invstd = reshaped_packed_accessor<accscalar_t, 1>(save_invstd_);

  auto stream = at::cuda::getCurrentCUDAStream();
  dim3 blocks(input.size(1));
  dim3 threads(getNumThreads(input.size(2)));

  batch_norm_backward_kernel<scalar_t,  accscalar_t> <<<blocks, threads, 0, stream>>>
    (input, grad_output, grad_input, grad_weight, grad_bias, weight, running_mean, running_var,
     save_mean, save_invstd, train, epsilon);
  THCudaCheck(cudaGetLastError());

  return std::make_tuple(grad_input_, grad_weight_, grad_bias_);
}

} // anonymous namespace

std::tuple<Tensor, Tensor, Tensor> batch_norm_cuda(const Tensor& self, const Tensor& weight, const Tensor& bias,
						   const Tensor& running_mean, const Tensor& running_var, bool train, double momentum, double epsilon) {
  return AT_DISPATCH_FLOATING_TYPES_AND_HALF(self.type(), "batch_norm", [&] {
      return batch_norm_cuda_template<scalar_t>(self, weight, bias, running_mean, running_var, train, momentum, epsilon);
    });
}

std::tuple<Tensor, Tensor, Tensor> batch_norm_backward_cuda(const Tensor& grad_out, const Tensor& self, const Tensor& weight, const Tensor& running_mean, const Tensor& running_var,
							    const Tensor& save_mean, const Tensor& save_invstd, bool train, double epsilon, std::array<bool,3> grad_input_mask) {
  return AT_DISPATCH_FLOATING_TYPES_AND_HALF(self.type(), "batch_norm_backward", [&] {
      return batch_norm_backward_cuda_template<scalar_t>(grad_out, self, weight, running_mean, running_var, save_mean, save_invstd, train, epsilon, grad_input_mask);
    });
}

} } // namespace at::native
