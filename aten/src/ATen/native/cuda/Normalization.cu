#include <THC/THCDeviceUtils.cuh>
#include <THC/THCGeneral.h>
#include "ATen/ATen.h"
#include "ATen/AccumulateType.h"
#include "ATen/cuda/CUDAContext.h"

namespace at { namespace native {

namespace {


using namespace at;

#if defined(__HIP_PLATFORM_HCC__)
constexpr int WARP_SIZE = 64;
#else
constexpr int WARP_SIZE = 32;
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
  __device__ SumOp(const PackedTensorAccessor<scalar_t, 3, at::RestrictPtrTraits>& t) : tensor(t) {}
  __device__ __forceinline__ accscalar_t operator()(int batch, int plane, int n) {
    return static_cast<accscalar_t>(tensor[batch][plane][n]);
  }
  const PackedTensorAccessor<scalar_t, 3, at::RestrictPtrTraits>& tensor;
};

template <typename scalar_t, typename accscalar_t>
struct VarOp {
  __device__ VarOp(accscalar_t m, const PackedTensorAccessor<scalar_t, 3, at::RestrictPtrTraits>& t) : mean(m), tensor(t) {}
  __device__ __forceinline__ accscalar_t operator()(int batch, int plane, int n) {
    scalar_t val = tensor[batch][plane][n];
    return (val - mean) * (val - mean);
  }
  const accscalar_t mean;
  const PackedTensorAccessor<scalar_t, 3, at::RestrictPtrTraits>& tensor;
};

template <typename scalar_t, typename accscalar_t>
struct GradOp {
  __device__ GradOp(accscalar_t m, const PackedTensorAccessor<scalar_t, 3, at::RestrictPtrTraits>& i, const PackedTensorAccessor<scalar_t, 3, at::RestrictPtrTraits>& g)
    : mean(m), input(i), grad_output(g) {}
  __device__ __forceinline__ Float2<scalar_t, accscalar_t> operator()(int batch, int plane, int n) {
    scalar_t g = grad_output[batch][plane][n];
    scalar_t c = static_cast<accscalar_t>(input[batch][plane][n] - mean);
    return Float2<scalar_t, accscalar_t>(g, g * c);
  }
  const accscalar_t mean;
  const PackedTensorAccessor<scalar_t, 3, at::RestrictPtrTraits>& input;
  const PackedTensorAccessor<scalar_t, 3, at::RestrictPtrTraits>& grad_output;
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
template<typename scalar_t, typename Op, typename PTA>
__device__ scalar_t reduce(Op op, PTA tensor, int plane) {
  scalar_t sum = static_cast<scalar_t>(0);
  for (int batch = 0; batch < tensor.size(0); ++batch) {
    for (int x = threadIdx.x; x < tensor.size(2); x += blockDim.x) {
      sum += op(batch, plane, x);
    }
  }

  // sum over NumThreads within a warp
  sum = warpSum(sum);

  // 'transpose', and reduce within warp again
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

template <typename scalar_t, typename accscalar_t>
__global__ void batch_norm_transform_input_kernel(
    const PackedTensorAccessor<scalar_t, 3, at::RestrictPtrTraits> input,
    PackedTensorAccessor<scalar_t, 3, at::RestrictPtrTraits> output,
    const PackedTensorAccessor<scalar_t, 1, at::RestrictPtrTraits> mean_,
    const PackedTensorAccessor<scalar_t, 1, at::RestrictPtrTraits> var_,
    const PackedTensorAccessor<scalar_t, 1, at::RestrictPtrTraits> weight,
    const PackedTensorAccessor<scalar_t, 1, at::RestrictPtrTraits> bias,
    accscalar_t epsilon) {

  int plane = blockIdx.y * blockDim.y + threadIdx.y;

  if (plane >= input.size(1)) {
    return;
  }

  accscalar_t gamma = weight.size(0) > 0 ? static_cast<accscalar_t>(weight[plane]) : static_cast<accscalar_t>(1);
  accscalar_t beta = bias.size(0) > 0 ? static_cast<accscalar_t>(bias[plane]) : static_cast<accscalar_t>(0);
  accscalar_t mean = static_cast<accscalar_t>(mean_[plane]);
  accscalar_t invstd = 0;
  accscalar_t var = static_cast<accscalar_t>(var_[plane]);
  if (var != static_cast<accscalar_t>(0) || epsilon != static_cast<accscalar_t>(0)) {
    invstd = static_cast<accscalar_t>(1) / std::sqrt(var + epsilon);
  }
  for (int64_t batch = blockIdx.x; batch < input.size(0); batch += gridDim.x) {
    for (int64_t feature = blockIdx.z; feature < input.size(2); feature += gridDim.z) {
      output[batch][plane][feature] = static_cast<scalar_t>(gamma * (input[batch][plane][feature] - mean) * invstd + beta);
    }
  }
}


template <typename scalar_t, typename accscalar_t>
__global__ void batch_norm_collect_statistics_kernel(
    const PackedTensorAccessor<scalar_t, 3, at::RestrictPtrTraits> input,
    PackedTensorAccessor<scalar_t, 3, at::RestrictPtrTraits> output,
    const PackedTensorAccessor<scalar_t, 1, at::RestrictPtrTraits> weight,
    const PackedTensorAccessor<scalar_t, 1, at::RestrictPtrTraits> bias,
    const accscalar_t epsilon,
    const accscalar_t momentum,
    PackedTensorAccessor<scalar_t, 1, at::RestrictPtrTraits> running_mean,
    PackedTensorAccessor<scalar_t, 1, at::RestrictPtrTraits> running_var,
    PackedTensorAccessor<scalar_t, 1, at::RestrictPtrTraits> save_mean,
    PackedTensorAccessor<scalar_t, 1, at::RestrictPtrTraits> save_var) {

  int plane = blockIdx.x;
  int N = input.size(0) * input.size(2);

  accscalar_t norm = accscalar_t(1) / N;

  // Compute the mean and variance across (batch, x/y/z)
  accscalar_t mean = reduce<accscalar_t>(SumOp<scalar_t, accscalar_t>(input), input, plane) * norm;
  __syncthreads();
  accscalar_t varN = reduce<accscalar_t>(VarOp<scalar_t, accscalar_t>(mean, input), input, plane);

  // Save the mean, variance, and moving averages
  if (threadIdx.x == 0) {
    accscalar_t unbiasedVar = varN / (N - 1);
    save_mean[plane] = static_cast<scalar_t>(mean);
    save_var[plane] = varN * norm;
    if (running_mean.data() != NULL) {
      running_mean[plane] = static_cast<scalar_t>((1 - momentum) * running_mean[plane] + momentum * mean);
    }
    if (running_var.data() != NULL) {
      running_var[plane] = static_cast<scalar_t>((1 - momentum) * running_var[plane] + momentum * unbiasedVar);
    }
  }

}

template <typename scalar_t, typename accscalar_t>
__global__ void batch_norm_backward_kernel(
    const PackedTensorAccessor<scalar_t, 3, at::RestrictPtrTraits> input,
    const PackedTensorAccessor<scalar_t, 3, at::RestrictPtrTraits> grad_output,
    PackedTensorAccessor<scalar_t, 3, at::RestrictPtrTraits> grad_input,
    PackedTensorAccessor<scalar_t, 1, at::RestrictPtrTraits> grad_weight,
    PackedTensorAccessor<scalar_t, 1, at::RestrictPtrTraits> grad_bias,
    const PackedTensorAccessor<scalar_t, 1, at::RestrictPtrTraits> weight,
    const PackedTensorAccessor<scalar_t, 1, at::RestrictPtrTraits> running_mean,
    const PackedTensorAccessor<scalar_t, 1, at::RestrictPtrTraits> running_var,
    const PackedTensorAccessor<scalar_t, 1, at::RestrictPtrTraits> save_mean,
    const PackedTensorAccessor<scalar_t, 1, at::RestrictPtrTraits> save_var,
    bool train,
    accscalar_t epsilon) {

  int plane = blockIdx.x;
  int N = grad_output.size(0) * grad_output.size(2);

  accscalar_t mean, var;
  if (train) {
    mean = static_cast<accscalar_t>(save_mean[plane]);
    var = static_cast<accscalar_t>(save_var[plane]);
  } else {
    mean = static_cast<accscalar_t>(running_mean[plane]);
    var = static_cast<accscalar_t>(running_var[plane]);
  }
  accscalar_t invstd = 0;
  if (var != static_cast<accscalar_t>(0) || epsilon != static_cast<accscalar_t>(0)) {
    invstd = static_cast<accscalar_t>(1) / std::sqrt(var + epsilon);
  }

  accscalar_t weight_val = weight.size(0) > 0 ? static_cast<accscalar_t>(weight[plane]) : accscalar_t(1);
  accscalar_t norm = accscalar_t(1) / N;

  // Compute two values across (batch, x/y/z) in one pass:
  // 1. Sum(grad_output)
  // 2. DotProduct(input - mean, grad_output)
  GradOp<scalar_t, accscalar_t> g(mean, input, grad_output);
  Float2<scalar_t, accscalar_t> res = reduce<Float2<scalar_t, accscalar_t>, GradOp<scalar_t, accscalar_t>, PackedTensorAccessor<scalar_t, 3, at::RestrictPtrTraits>>(g, grad_output, plane);
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
static PackedTensorAccessor<scalar_t, dim, at::RestrictPtrTraits> reshaped_packed_accessor(const Tensor& t) {
  // undefined...
  if (! t.defined()) {
    const std::vector<int64_t> zeros(dim);
    return PackedTensorAccessor<scalar_t, dim, at::RestrictPtrTraits>(nullptr, zeros.data(), zeros.data());
  }
  int64_t in_dim = t.dim();
  if (in_dim == dim) {
    return t.packed_accessor<scalar_t, dim, at::RestrictPtrTraits>();
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
  if (dim == 3 && sizes[0] > sizes[2]) {
    std::swap(sizes[0], sizes[2]);
    std::swap(strides[0], strides[2]);
  }
  return PackedTensorAccessor<scalar_t, dim, at::RestrictPtrTraits>(t.data<scalar_t>(), sizes.data(), strides.data());
}

template<typename scalar_t>
std::tuple<Tensor, Tensor, Tensor> batch_norm_cuda_template(const Tensor& input_, const Tensor& weight_, const Tensor& bias_,
							    const Tensor& running_mean_, const Tensor& running_var_,
							    bool train, double momentum, double epsilon) {

  using accscalar_t = at::acc_type<scalar_t, true>;
  Tensor output_= at::native::empty_like(input_);
  int64_t n_input = input_.size(1);
  Tensor save_mean_;
  Tensor save_var_;
  if (train) {
    save_mean_ = at::native::empty({n_input}, input_.options());
    save_var_ = at::native::empty({n_input}, input_.options());
  } else {
    save_mean_ = at::native::empty({0}, input_.options());
    save_var_ = at::native::empty({0}, input_.options());
  }
  auto input = reshaped_packed_accessor<scalar_t, 3>(input_);
  auto output = reshaped_packed_accessor<scalar_t, 3>(output_);
  auto weight = reshaped_packed_accessor<scalar_t, 1>(weight_);
  auto bias = reshaped_packed_accessor<scalar_t, 1>(bias_);
  auto running_mean = reshaped_packed_accessor<scalar_t, 1>(running_mean_);
  auto running_var = reshaped_packed_accessor<scalar_t, 1>(running_var_);
  auto save_mean = reshaped_packed_accessor<scalar_t, 1>(save_mean_);
  auto save_var = reshaped_packed_accessor<scalar_t, 1>(save_var_);
  auto stream = at::cuda::getCurrentCUDAStream();

  constexpr int max_blocks_per_input = 60000;
  int feature_blocks = std::min<int>(input.size(2), max_blocks_per_input);
  int batch_blocks   = std::min<int>(input.size(0), max_blocks_per_input / feature_blocks);
  dim3 blocks(batch_blocks, (input.size(1)+127)/128, feature_blocks);
  dim3 threads(1, 128);
  if (!train) {
    batch_norm_transform_input_kernel<scalar_t, accscalar_t> <<<blocks, threads, 0, stream>>>
      (input, output, running_mean, running_var, weight, bias, epsilon);
  } else {
    dim3 blocks_red(input.size(1));
    dim3 threads_red(getNumThreads(input.size(2)));
    batch_norm_collect_statistics_kernel<scalar_t, accscalar_t> <<<blocks_red, threads_red, 0, stream>>>
      (input, output, weight, bias, epsilon, momentum, running_mean, running_var, save_mean, save_var);
    batch_norm_transform_input_kernel<scalar_t, accscalar_t> <<<blocks, threads, 0, stream>>>
      (input, output, save_mean, save_var, weight, bias, epsilon);
  }
  THCudaCheck(cudaGetLastError());
  return std::make_tuple(output_, save_mean_, save_var_);
}

template<typename scalar_t>
std::tuple<Tensor, Tensor, Tensor> batch_norm_backward_cuda_template(const Tensor& grad_out_, const Tensor& input_, const Tensor& weight_,
								     const Tensor& running_mean_, const Tensor& running_var_, const Tensor& save_mean_, const Tensor& save_var_,
								     bool train, double epsilon, std::array<bool,3> grad_input_mask) {

  using accscalar_t = at::acc_type<scalar_t, true>;
  Tensor grad_input_;
  Tensor grad_weight_;
  Tensor grad_bias_;
  if (grad_input_mask[0]) {
    grad_input_ = at::native::empty_like(input_);
  }
  if (grad_input_mask[1]) {
    grad_weight_ = at::native::empty_like(weight_);
  }
  if (grad_input_mask[2]) {
    grad_bias_ = at::native::empty_like(weight_);
  }

  auto grad_output = reshaped_packed_accessor<scalar_t, 3>(grad_out_);
  auto input = reshaped_packed_accessor<scalar_t, 3>(input_);
  auto grad_input = reshaped_packed_accessor<scalar_t, 3>(grad_input_);
  auto weight = reshaped_packed_accessor<scalar_t, 1>(weight_);
  auto grad_weight = reshaped_packed_accessor<scalar_t, 1>(grad_weight_);
  auto grad_bias = reshaped_packed_accessor<scalar_t, 1>(grad_bias_);
  auto running_mean = reshaped_packed_accessor<scalar_t, 1>(running_mean_);
  auto running_var = reshaped_packed_accessor<scalar_t, 1>( running_var_);
  auto save_mean = reshaped_packed_accessor<scalar_t, 1>(save_mean_);
  auto save_var = reshaped_packed_accessor<scalar_t, 1>(save_var_);

  auto stream = at::cuda::getCurrentCUDAStream();
  dim3 blocks(input.size(1));
  dim3 threads(getNumThreads(input.size(2)));

  batch_norm_backward_kernel<scalar_t,  accscalar_t> <<<blocks, threads, 0, stream>>>
    (input, grad_output, grad_input, grad_weight, grad_bias, weight, running_mean, running_var,
     save_mean, save_var, train, epsilon);
  THCudaCheck(cudaGetLastError());

  return std::make_tuple(grad_input_, grad_weight_, grad_bias_);
}

} // anonymous namespace

// Note: In contrast to CuDNN, we return the batch variance in save_var/as third return. This is done for a bit better stability for
// half - where CuDNN uses float.
std::tuple<Tensor, Tensor, Tensor> batch_norm_cuda(const Tensor& self, const Tensor& weight, const Tensor& bias,
						   const Tensor& running_mean, const Tensor& running_var, bool train, double momentum, double epsilon) {
  return AT_DISPATCH_FLOATING_TYPES_AND_HALF(self.type(), "batch_norm", [&] {
      return batch_norm_cuda_template<scalar_t>(self, weight, bias, running_mean, running_var, train, momentum, epsilon);
    });
}

// Note: In contrast to CuDNN, we have the batch variance in save_var. This is done for a bit better stability for
// half - where CuDNN uses float.
std::tuple<Tensor, Tensor, Tensor> batch_norm_backward_cuda(const Tensor& grad_out, const Tensor& self, const Tensor& weight, const Tensor& running_mean, const Tensor& running_var,
							    const Tensor& save_mean, const Tensor& save_var, bool train, double epsilon, std::array<bool,3> grad_input_mask) {
  return AT_DISPATCH_FLOATING_TYPES_AND_HALF(self.type(), "batch_norm_backward", [&] {
      return batch_norm_backward_cuda_template<scalar_t>(grad_out, self, weight, running_mean, running_var, save_mean, save_var, train, epsilon, grad_input_mask);
    });
}

} } // namespace at::native
