#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/TensorUtils.h>
#include <ATen/cuda/Atomic.cuh>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/core/TensorAccessor.h>
#include <ATen/cuda/detail/KernelUtils.h>
#include <c10/cuda/CUDAException.h>
#include <c10/macros/Macros.h>
#include <ATen/native/IndexingUtils.h>
#include <ATen/native/Resize.h>
#include <ATen/native/cuda/block_reduce.cuh>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/nll_loss2d_forward_native.h>
#include <ATen/ops/nll_loss2d_backward_native.h>
#endif

namespace at::native {

namespace {

// Returns a contiguous tensor if the source tensor
// is defined. Otherwise returns the undefined
// source tensor unmodified.
inline Tensor optional_contiguous(const Tensor& source) {
  return source.defined() ? source.contiguous() : source;
}

// Returns the address of the first element of a tensor
// or nullptr if the tensor is undefined.
template <typename scalar_t>
inline const scalar_t* optional_data(const Tensor& source) {
  return source.defined() ? source.const_data_ptr<scalar_t>() : nullptr;
}

using at::cuda::detail::CUDA_NUM_THREADS;
using at::cuda::detail::GET_BLOCKS;

// TODO(crcrpar): Think about introducing `canUse32BitIndexMath` and choose int or int64_t for `target`.
template <typename scalar_t>
C10_LAUNCH_BOUNDS_1(CUDA_NUM_THREADS)
__global__ void nll_loss2d_forward_no_reduce_kernel(
  int64_t n_threads,
  PackedTensorAccessor64<scalar_t, 4> input,
  PackedTensorAccessor64<int64_t, 3> target,
  PackedTensorAccessor64<scalar_t, 3> output,
  const scalar_t* weight,
  int64_t ignore_index
) {
  int64_t batch_size = input.size(0);
  int64_t n_classes = input.size(1);
  int64_t H = input.size(2);
  int64_t W = input.size(3);

  CUDA_KERNEL_LOOP(index, n_threads) {
    const int64_t b = index % batch_size;
    const int64_t h = (index / batch_size) % H;
    const int64_t w = (index / (batch_size * H)) % W;

    int64_t cur_target = target[b][h][w];
    if (cur_target == ignore_index) {
      output[b][h][w] = static_cast<scalar_t>(0);
      continue;
    }
    CUDA_KERNEL_ASSERT(cur_target >= 0 && cur_target < n_classes);
    scalar_t value = input[b][cur_target][h][w];
    scalar_t cur_weight = weight != nullptr ? weight[cur_target] : static_cast<scalar_t>(1);
    output[b][h][w] = -value * cur_weight;
  }
}

template <typename scalar_t, typename accscalar_t, typename index_t>
C10_LAUNCH_BOUNDS_1(CUDA_NUM_THREADS)
__global__ void nll_loss2d_forward_kernel(
  scalar_t* output,
  scalar_t* total_weight,
  const scalar_t* input,
  const int64_t* target,
  const scalar_t* weight,
  int n_classes,
  int map_nelem,
  int blocks_per_sample,
  int64_t ignore_index) {

  scalar_t cur_weight;
  accscalar_t input_sum = 0;
  accscalar_t acc_weight = 0;

  index_t sample = blockIdx.x / blocks_per_sample;
  index_t toffset = sample * map_nelem;
  index_t ioffset = sample * map_nelem * n_classes;
  int step = blockDim.x * blocks_per_sample;
  for (int i = (blockIdx.x % blocks_per_sample) * blockDim.x + threadIdx.x;
       i < map_nelem;
       i += step) {
    index_t t = target[toffset + i];
    if (t != ignore_index) {
      CUDA_KERNEL_ASSERT(t >= 0 && t < n_classes);
      cur_weight = weight != nullptr ? weight[t] : static_cast<scalar_t>(1);
      const auto input_index = ioffset + i + map_nelem * t;
      CUDA_KERNEL_ASSERT(input_index >= 0);
      input_sum -= input[input_index] * cur_weight;
      acc_weight += cur_weight;
    }
  }

  __shared__ accscalar_t acc_weight_smem[CUDA_NUM_THREADS];
  __shared__ accscalar_t input_sum_smem[CUDA_NUM_THREADS];

  auto acc_weight_ = cuda_utils::BlockReduceSum(acc_weight, acc_weight_smem);
  auto input_sum_ = cuda_utils::BlockReduceSum(input_sum, input_sum_smem);

  if (threadIdx.x == 0) {
    gpuAtomicAdd(total_weight, static_cast<scalar_t>(acc_weight_));
    gpuAtomicAdd(output, static_cast<scalar_t>(input_sum_));
  }
}

template <typename scalar_t>
C10_LAUNCH_BOUNDS_1(CUDA_NUM_THREADS)
__global__ void nll_loss2d_forward_size_average_kernel(
  scalar_t* output,
  const scalar_t* total_weight
) {
  *output /= *total_weight;
}

template <typename scalar_t>
C10_LAUNCH_BOUNDS_1(CUDA_NUM_THREADS)
__global__ void nll_loss2d_backward_no_reduce_kernel(
  int64_t n_threads,
  PackedTensorAccessor64<int64_t, 3> target,
  PackedTensorAccessor64<scalar_t, 3> grad_output,
  PackedTensorAccessor64<scalar_t, 4> grad_input,
  const scalar_t* weight,
  int64_t ignore_index
) {
  int64_t batch_size = target.size(0);
  int64_t H = target.size(1);
  int64_t W = target.size(2);

  CUDA_KERNEL_LOOP(index, n_threads) {
    const int64_t b = index % batch_size;
    const int64_t h = (index / batch_size) % H;
    const int64_t w = (index / (batch_size * H)) % W;

    int64_t cur_target = target[b][h][w];
    if (cur_target == ignore_index) {
      continue;
    }
    scalar_t value = -(weight != nullptr ? weight[cur_target] : static_cast<scalar_t>(1));
    grad_input[b][cur_target][h][w] = value * grad_output[b][h][w];
  }
}

template <typename scalar_t>
C10_LAUNCH_BOUNDS_1(CUDA_NUM_THREADS)
__global__ void nll_loss2d_backward_kernel(
  scalar_t* grad_input,
  const scalar_t* grad_output,
  const int64_t* target,
  const scalar_t* weights,
  const scalar_t* total_weight,
  bool size_average,
  int n_classes,
  int map_nelem,
  int blocks_per_sample,
  int64_t ignore_index
) {
  const auto grad = -(size_average ? *grad_output / *total_weight
                                   : *grad_output);

  const int sample = blockIdx.x / blocks_per_sample;
  const int step = blockDim.x * blocks_per_sample;

  const int toffset = sample * map_nelem;
  const auto* const target_thread = target + toffset;

  const int ioffset = sample * map_nelem * n_classes;
  auto* const grad_input_thread = grad_input + ioffset;

  for (int i = (blockIdx.x % blocks_per_sample) * blockDim.x + threadIdx.x;
       i < map_nelem;
       i += step) {
    const int64_t t = target_thread[i];
    if (t != ignore_index) {
      CUDA_KERNEL_ASSERT(t >= 0 && t < n_classes);
      const auto grad_input_index = i + map_nelem * t;
      CUDA_KERNEL_ASSERT(grad_input_index >= 0);
      grad_input_thread[i + map_nelem * t] = weights != nullptr ? weights[t] * grad
                                                                : grad;
    }
  }
}

void check_inputs_nll_loss2d(
    const Tensor& input,
    const Tensor& target,
    const Tensor& weight) {
  TORCH_CHECK(
      target.dim() == 3,
      "only batches of spatial targets supported (3D tensors)"
      " but got targets of size: : ",
      target.sizes());
  TORCH_CHECK(
      input.dim() == 4,
      "only batches of spatial inputs supported (4D tensors), "
      "but got input of size: ",
      input.sizes());
  TORCH_CHECK(
      !weight.defined() || weight.numel() == input.size(1),
      "weight tensor should be defined either for all or no classes");

  TORCH_CHECK(
      input.size(0) == target.size(0) && input.size(2) == target.size(1) &&
          input.size(3) == target.size(2),
      "input and target batch or spatial sizes don't match: target ",
      target.sizes(),
      ", input ",
      input.sizes());
}

void nll_loss2d_forward_out_cuda_template(
    Tensor& output,
    Tensor& total_weight,
    const Tensor& input,
    const Tensor& target,
    const c10::optional<Tensor>& weight_opt,
    int64_t reduction,
    int64_t ignore_index) {
  // See Note [Writing Nondeterministic Operations]
  // Nondeterministic because of atomicAdd usage in 'sum' or 'mean' reductions.
  if (reduction != at::Reduction::None) {
    at::globalContext().alertNotDeterministic("nll_loss2d_forward_out_cuda_template");
  }

  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> weight_maybe_owned =
      at::borrow_from_optional_tensor(weight_opt);
  const Tensor& weight = *weight_maybe_owned;

  check_inputs_nll_loss2d(input, target, weight);
  total_weight.resize_({});

  if (reduction == at::Reduction::None) {
    int64_t batch_size = input.size(0);
    int64_t H = input.size(2);
    int64_t W = input.size(3);
    int64_t count = batch_size * H * W;

    at::native::resize_output(output, {batch_size, H, W});
    if (count == 0) {
      // This guards from unnecessary operations and launching CUDA kernel with
      // 0 blocks.
      return;
    }
    auto weight_ = optional_contiguous(weight);
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        input.scalar_type(),
        "nll_loss2d_forward_no_reduce_kernel",
        [&] {
          nll_loss2d_forward_no_reduce_kernel<scalar_t>
              <<<GET_BLOCKS(count),
                 CUDA_NUM_THREADS,
                 0,
                 at::cuda::getCurrentCUDAStream()>>>(
                  count,
                  input.packed_accessor64<scalar_t, 4>(),
                  target.packed_accessor64<int64_t, 3>(),
                  output.packed_accessor64<scalar_t, 3>(),
                  optional_data<scalar_t>(weight_),
                  ignore_index);
          C10_CUDA_KERNEL_LAUNCH_CHECK();
        });
    return;
  }

  // produce scalar outputs for the reduction case
  at::native::resize_output(output, {});

  if (target.numel() == 0) {
    // Here target (and input) have zero elements
    // Mean reduction on empty tensors produces NaN. See the discussion in
    // https://github.com/pytorch/pytorch/pull/64572#issuecomment-926504162
    if (reduction == Reduction::Mean) {
      output.fill_(std::numeric_limits<double>::quiet_NaN());
    } else {
      output.zero_();
    }
    total_weight.zero_();
    return;
  }

  auto input_ = input.contiguous();
  auto weight_ = optional_contiguous(weight);
  auto target_ = target.contiguous();

  output.zero_();
  total_weight.zero_();

  auto batch_size = target.size(0);
  int64_t map_nelem = target.numel() / batch_size;
  int blocks_per_sample = GET_BLOCKS(map_nelem) / 128;
  blocks_per_sample = (blocks_per_sample == 0) ? 1 : blocks_per_sample;
  int total_blocks = blocks_per_sample * batch_size;

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      input.scalar_type(),
      "nll_loss2d_forward_kernel",
      [&] {
        using accscalar_t = acc_type<scalar_t, true>;
    AT_DISPATCH_INDEX_TYPES(
        at::native::canUse32BitIndexMath(input_, INT_MAX) ? ScalarType::Int : ScalarType::Long,
        "nll_loss2d_forward_launcher", [&] {
            nll_loss2d_forward_kernel<scalar_t, accscalar_t, index_t>
                <<<total_blocks,
                  CUDA_NUM_THREADS,
                  0,
                  at::cuda::getCurrentCUDAStream()>>>(
                    output.mutable_data_ptr<scalar_t>(),
                    total_weight.mutable_data_ptr<scalar_t>(),
                    input_.const_data_ptr<scalar_t>(),
                    target_.const_data_ptr<int64_t>(),
                    optional_data<scalar_t>(weight_),
                    input_.size(1),
                    input_.size(2) * input_.size(3),
                    blocks_per_sample,
                    ignore_index);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            // Divide by total_weight
            if (reduction == at::Reduction::Mean) {
              nll_loss2d_forward_size_average_kernel<scalar_t>
                  <<<1, 1, 0, at::cuda::getCurrentCUDAStream()>>>(
                      output.mutable_data_ptr<scalar_t>(),
                      total_weight.const_data_ptr<scalar_t>());
              C10_CUDA_KERNEL_LAUNCH_CHECK();
            }
    });
      });
}

void nll_loss2d_backward_out_cuda_template(
    Tensor& grad_input,
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& target,
    const c10::optional<Tensor>& weight_opt,
    int64_t reduction,
    int64_t ignore_index,
    const Tensor& total_weight) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> weight_maybe_owned =
      at::borrow_from_optional_tensor(weight_opt);
  const Tensor& weight = *weight_maybe_owned;

  check_inputs_nll_loss2d(input, target, weight);
  grad_input.resize_as_(input);
  grad_input.zero_();
  TORCH_CHECK(grad_input.is_contiguous(), "grad_input must be contiguous");
  TORCH_CHECK(
      total_weight.numel() == 1,
      "expected total_weight to be a single element tensor, got: ",
      total_weight.sizes(),
      " (",
      total_weight.numel(),
      " elements)");


  if (reduction == at::Reduction::None) {
    TORCH_CHECK(
        grad_output.dim() == 3,
        "grad_output must have same dimension as target (3) but got dimension: ",
        grad_output.sizes());
    TORCH_CHECK(
        grad_output.size(0) == target.size(0) &&
            grad_output.size(1) == target.size(1) &&
            grad_output.size(2) == target.size(2),
        "grad_output sizes don't match target sizes: target ",
        target.sizes(),
        ", grad_output ",
        grad_output.sizes())
    int64_t batch_size = input.size(0);
    int64_t H = input.size(2);
    int64_t W = input.size(3);
    int64_t count = batch_size * H * W;

    if (count == 0) {
      // This guards from unnecessary operations and launching CUDA kernel with
      // 0 blocks.
      return;
    }
    auto weight_ = optional_contiguous(weight);
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        input.scalar_type(),
        "nll_loss2d_backward_no_reduce_kernel",
        [&] {
          nll_loss2d_backward_no_reduce_kernel<scalar_t>
              <<<GET_BLOCKS(count),
                 CUDA_NUM_THREADS,
                 0,
                 at::cuda::getCurrentCUDAStream()>>>(
                  count,
                  target.packed_accessor64<int64_t, 3>(),
                  grad_output.packed_accessor64<scalar_t, 3>(),
                  grad_input.packed_accessor64<scalar_t, 4>(),
                  optional_data<scalar_t>(weight_),
                  ignore_index);
          C10_CUDA_KERNEL_LAUNCH_CHECK();
        });
    return;
  }

  int64_t batch_size = target.size(0);
  auto target_numel = target.numel();
  if (batch_size != 0 && target_numel != 0) {
    // This guards from unnecessary operations and launching CUDA kernel with 1
    // blocks.
    auto target_ = target.contiguous();
    auto weight_ = optional_contiguous(weight);

    int64_t map_nelem = target_numel / batch_size;
    int blocks_per_sample = GET_BLOCKS(map_nelem) / 128;
    blocks_per_sample = (blocks_per_sample == 0) ? 1 : blocks_per_sample;
    int total_blocks = blocks_per_sample * batch_size;

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        input.scalar_type(),
        "nll_loss2d_backward_kernel",
        [&] {
          nll_loss2d_backward_kernel<scalar_t>
              <<<total_blocks,
                CUDA_NUM_THREADS,
                0,
                at::cuda::getCurrentCUDAStream()>>>(
                  grad_input.mutable_data_ptr<scalar_t>(),
                  grad_output.const_data_ptr<scalar_t>(),
                  target_.const_data_ptr<int64_t>(),
                  optional_data<scalar_t>(weight_),
                  total_weight.const_data_ptr<scalar_t>(),
                  reduction == at::Reduction::Mean,
                  input.size(1),
                  map_nelem,
                  blocks_per_sample,
                  ignore_index);
          C10_CUDA_KERNEL_LAUNCH_CHECK();
        });
  }
}
} // namespace

std::tuple<Tensor&, Tensor&> nll_loss2d_forward_out_cuda(
    const Tensor& self,
    const Tensor& target,
    const c10::optional<Tensor>& weight_opt,
    int64_t reduction,
    int64_t ignore_index,
    Tensor& output,
    Tensor& total_weight) {
  nll_loss2d_forward_out_cuda_template(
      output, total_weight, self, target, weight_opt, reduction, ignore_index);
  return std::tuple<Tensor&, Tensor&>(output, total_weight);
}

std::tuple<Tensor, Tensor> nll_loss2d_forward_cuda(
    const Tensor& self,
    const Tensor& target,
    const c10::optional<Tensor>& weight_opt,
    int64_t reduction,
    int64_t ignore_index) {
  auto output = at::empty({0}, self.options());
  auto total_weight = at::empty({0}, self.options());
  nll_loss2d_forward_out_cuda_template(
      output, total_weight, self, target, weight_opt, reduction, ignore_index);
  return std::make_tuple(output, total_weight);
}

Tensor& nll_loss2d_backward_out_cuda(
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& target,
    const c10::optional<Tensor>& weight_opt,
    int64_t reduction,
    int64_t ignore_index,
    const Tensor& total_weight,
    Tensor& grad_input) {
  nll_loss2d_backward_out_cuda_template(
      grad_input,
      grad_output,
      self,
      target,
      weight_opt,
      reduction,
      ignore_index,
      total_weight);
  return grad_input;
}

Tensor nll_loss2d_backward_cuda(
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& target,
    const c10::optional<Tensor>& weight_opt,
    int64_t reduction,
    int64_t ignore_index,
    const Tensor& total_weight) {
  auto grad_input = at::empty_like(self);
  nll_loss2d_backward_out_cuda_template(
      grad_input,
      grad_output,
      self,
      target,
      weight_opt,
      reduction,
      ignore_index,
      total_weight);
  return grad_input;
}

} // namespace at::native
