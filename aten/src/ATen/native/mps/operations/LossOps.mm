//  Copyright © 2022 Apple Inc.
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/OpMathType.h>
#include <ATen/ceil_div.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/CanUse32BitIndexMath.h>
#include <ATen/native/PointwiseOps.h>
#include <ATen/native/mps/OperationUtils.h>
#include <ATen/native/mps/kernels/LossOps.h>
#include <ATen/native/mps/kernels/ReduceOps.h>
#include <limits>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_ctc_loss_backward_native.h>
#include <ATen/ops/_ctc_loss_native.h>
#include <ATen/ops/binary_cross_entropy_backward_native.h>
#include <ATen/ops/binary_cross_entropy_native.h>
#include <ATen/ops/full_like.h>
#include <ATen/ops/huber_loss_backward_native.h>
#include <ATen/ops/huber_loss_native.h>
#include <ATen/ops/mean.h>
#include <ATen/ops/mse_loss_backward_native.h>
#include <ATen/ops/nll_loss2d_backward_native.h>
#include <ATen/ops/nll_loss2d_forward_native.h>
#include <ATen/ops/nll_loss_backward_native.h>
#include <ATen/ops/nll_loss_forward_native.h>
#include <ATen/ops/smooth_l1_loss_backward_native.h>
#include <ATen/ops/smooth_l1_loss_native.h>
#include <ATen/ops/sum.h>
#include <ATen/ops/tensor.h>
#endif

namespace at::native {
namespace mps {

#ifndef PYTORCH_JIT_COMPILE_SHADERS
static auto& lib = MetalShaderLibrary::getBundledLibrary();
#else
#include <ATen/native/mps/LossOps_metallib.h>
#endif

static void bce_loss_forward_impl(const Tensor& input_arg,
                                  const Tensor& target_arg,
                                  const std::optional<Tensor>& weight_opt,
                                  int64_t reduction,
                                  Tensor& loss) {
  TORCH_CHECK_NOT_IMPLEMENTED(c10::isFloatingType(input_arg.scalar_type()),
                              "binary_cross_entropy is not implemented for ",
                              input_arg.scalar_type(),
                              " on MPS");
  TORCH_CHECK(target_arg.is_same_size(input_arg),
              "binary_cross_entropy: target and input tensors must have identical shapes");

  loss.resize_(reduction == Reduction::None ? target_arg.sizes() : IntArrayRef({}));
  TORCH_CHECK(loss.is_mps());

  // A zero-sized input has no Metal buffer to bind, so the kernels below cannot run
  // on it at all. CPU returns an empty tensor for Reduction::None, 0 for Sum and
  // NaN for Mean; reproduce that rather than tripping the placeholder assert.
  if (input_arg.numel() == 0) {
    if (loss.numel() != 0) {
      loss.fill_(reduction == at::Reduction::Mean ? std::numeric_limits<double>::quiet_NaN() : 0.0);
    }
    return;
  }

  const auto weight_maybe_owned = at::borrow_from_optional_tensor(weight_opt);
  const auto& weight = *weight_maybe_owned;
  const Tensor input = input_arg.contiguous();
  const Tensor target = target_arg.to(input.scalar_type()).contiguous();
  const int64_t numel = input.numel();
  MPSStream* stream = getCurrentMPSStream();

  // Mean and sum are fused into the loss kernel, which writes one partial per
  // threadgroup (pass 1 of ReduceOps' flat reduction); sum_out folds them
  if (reduction != Reduction::None && canUse32BitIndexMath(input)) {
    const auto n = static_cast<uint32_t>(numel);
    const auto max_groups = std::min(512u, at::ceil_div(n, MAX_THREADGROUP_SIZE * SUM_NCHAINS));
    const auto slice_len = at::ceil_div(n, max_groups);
    const auto num_groups = at::ceil_div(n, slice_len);
    const BCELossReduceParams params{.numel = n,
                                     .slice_len = slice_len,
                                     .scale = reduction == Reduction::Mean ? 1.0f / n : 1.0f,
                                     .has_weight = weight.defined()};
    const Tensor weight_ = weight.defined() ? weight.to(input.scalar_type()).expand_as(input).contiguous() : input;
    Tensor partials = at::empty({num_groups}, input.options().dtype(kFloat));
    dispatch_sync_with_rethrow(stream->queue(), ^() {
      @autoreleasepool {
        auto pso = lib.getPipelineStateForFunc("bce_loss_forward_reduce_" + scalarToMetalTypeString(input));
        const auto threads = std::min<uint32_t>(MAX_THREADGROUP_SIZE, [pso maxTotalThreadsPerThreadgroup]);
        auto encoder = stream->commandEncoder();
        [encoder setComputePipelineState:pso];
        mtl_setArgs(encoder, input, target, weight_, partials, params, stream->getErrorBuffer());
        [encoder dispatchThreads:MTLSizeMake(num_groups * threads, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
      }
    });
    at::sum_out(loss, partials, IntArrayRef{}, false, loss.scalar_type());
    return;
  }

  const bool write_to_loss = reduction == Reduction::None && loss.is_contiguous();
  Tensor unreduced = write_to_loss ? loss : at::empty(input.sizes(), input.options());
  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      auto pso = lib.getPipelineStateForFunc("bce_loss_forward_" + scalarToMetalTypeString(input));
      auto encoder = stream->commandEncoder();
      [encoder setComputePipelineState:pso];
      mtl_setArgs(encoder, input, target, unreduced);

      constexpr auto max_threads = int64_t{std::numeric_limits<uint32_t>::max()};
      for (auto offset = int64_t{0}; offset < numel; offset += max_threads) {
        mtl_setArgs<3>(encoder, offset, stream->getErrorBuffer());
        mtl_dispatch1DJob(encoder, pso, std::min(max_threads, numel - offset));
      }
    }
  });
  if (weight.defined()) {
    unreduced.mul_(weight);
  }
  if (reduction == Reduction::Mean) {
    loss.copy_(at::mean(unreduced));
  } else if (reduction == Reduction::Sum) {
    loss.copy_(at::sum(unreduced));
  } else if (!write_to_loss) {
    loss.copy_(unreduced);
  }
}

// NLLLoss
// `t` is the tensor the kernel indexes by class: input for forward, grad_input for backward
static NLLLossForwardParams<> nllnd_loss_params(const Tensor& t, int64_t ignore_index, bool has_weight, bool is2D) {
  const auto class_dim = t.dim() == 1 ? 0 : 1;
  return {
      .n_classes = t.size(class_dim),
      .map_size = is2D ? t.size(2) * t.size(3) : 1,
      .batch_stride = t.dim() == 1 ? 0 : t.stride(0),
      .class_stride = t.stride(class_dim),
      .ignore_index = ignore_index,
      .tid_offset = 0,
      .has_weight = has_weight,
  };
}

static void nllnd_loss_backward_impl(Tensor& grad_input_arg,
                                     const Tensor& grad_output_arg,
                                     const Tensor& input_arg,
                                     const Tensor& target_arg,
                                     const Tensor& weight_arg,
                                     int64_t reduction,
                                     int64_t ignore_index,
                                     const Tensor& total_weight,
                                     bool is2D) {
  TORCH_CHECK(input_arg.is_mps(), "input must be an MPS tensor");
  TORCH_CHECK(grad_input_arg.is_mps(), "grad_input must be an MPS tensor");
  TORCH_CHECK(grad_output_arg.is_mps(), "grad_output must be an MPS tensor");
  TORCH_CHECK(target_arg.is_mps(), "target must be an MPS tensor");
  TORCH_CHECK(!weight_arg.defined() || weight_arg.is_mps(), "weight must be an MPS tensor");
  TORCH_CHECK(total_weight.is_mps(), "total_weight must be an MPS tensor");
  if (grad_input_arg.numel() == 0) {
    return;
  }
  TORCH_CHECK_TYPE(grad_input_arg.scalar_type() == input_arg.scalar_type(),
                   "expected scalar type ",
                   input_arg.scalar_type(),
                   " but found ",
                   grad_input_arg.scalar_type());
  TORCH_CHECK_TYPE(grad_output_arg.scalar_type() == input_arg.scalar_type(),
                   "expected scalar type ",
                   input_arg.scalar_type(),
                   " but found ",
                   grad_output_arg.scalar_type());
  TORCH_CHECK_TYPE(!weight_arg.defined() || weight_arg.scalar_type() == input_arg.scalar_type(),
                   "expected scalar type ",
                   input_arg.scalar_type(),
                   " but found ",
                   weight_arg.scalar_type());
  grad_input_arg.zero_();

  const bool has_weight = weight_arg.defined() && weight_arg.numel() > 0;
  const Tensor target =
      (target_arg.scalar_type() == ScalarType::Long ? target_arg : target_arg.to(ScalarType::Long)).contiguous();
  const Tensor grad_output = grad_output_arg.contiguous();
  const Tensor weight = has_weight ? weight_arg.contiguous() : grad_input_arg;
  const Tensor total_weight_cast =
      (total_weight.scalar_type() == input_arg.scalar_type() ? total_weight : total_weight.to(input_arg.scalar_type()))
          .contiguous();

  const NLLLossBackwardParams<> params{
      .forward = nllnd_loss_params(grad_input_arg, ignore_index, has_weight, is2D),
      .is_reduction = reduction != Reduction::None,
      .is_mean = reduction == Reduction::Mean,
  };

  MPSStream* stream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      auto pso = lib.getPipelineStateForFunc("nllnd_loss_backward_" + scalarToMetalTypeString(input_arg));
      auto encoder = stream->commandEncoder();
      [encoder setComputePipelineState:pso];
      mtl_setArgs(encoder, grad_input_arg, grad_output, target, weight, total_weight_cast);
      // For 1D (no batch dim) input the loss has a single element and only
      // target[0] is used; dispatching target.numel() threads would read
      // grad_output (a single element) out of bounds and scatter garbage into
      // grad_input. See https://github.com/pytorch/pytorch/issues/195391.
      const int64_t num_outputs = input_arg.dim() == 1 ? 1 : target.numel();
      // Chunk only when the target exceeds Metal's uint32 thread-grid limit.
      constexpr auto max_threads = int64_t{std::numeric_limits<uint32_t>::max()};
      auto dispatch_params = params;
      for (auto offset = int64_t{0}; offset < num_outputs; offset += max_threads) {
        dispatch_params.forward.tid_offset = offset;
        mtl_setArgs<5>(encoder, dispatch_params, stream->getErrorBuffer());
        mtl_dispatch1DJob(encoder, pso, std::min(max_threads, num_outputs - offset));
      }
    }
  });
}

static void nllnd_loss_forward_impl(Tensor& output,
                                    Tensor& total_weight,
                                    const Tensor& input_arg,
                                    const Tensor& target_arg,
                                    const Tensor& weight_arg,
                                    int64_t reduction,
                                    int64_t ignore_index,
                                    bool is2D) {
  TORCH_CHECK(input_arg.is_mps(), "input must be an MPS tensor");
  TORCH_CHECK(target_arg.is_mps(), "target must be an MPS tensor");
  TORCH_CHECK(!weight_arg.defined() || weight_arg.is_mps(), "weight must be an MPS tensor");
  TORCH_CHECK(output.is_mps(), "output must be an MPS tensor");
  TORCH_CHECK(total_weight.is_mps(), "total_weight must be an MPS tensor");

  TORCH_CHECK_NOT_IMPLEMENTED(c10::isFloatingType(input_arg.scalar_type()),
                              "nll_loss is not implemented for ",
                              input_arg.scalar_type(),
                              " on MPS");

  TORCH_CHECK(!weight_arg.defined() || input_arg.scalar_type() == weight_arg.scalar_type(),
              "expected scalar type ",
              input_arg.scalar_type(),
              " but found ",
              weight_arg.scalar_type());
  const bool has_weight = weight_arg.defined() && weight_arg.numel() > 0;

  if (reduction == Reduction::None) {
    output.resize_(target_arg.sizes());
  } else {
    output.resize_({});
  }
  total_weight.resize_({});

  // Empty input or target
  if (target_arg.numel() == 0 || input_arg.numel() == 0) {
    if (reduction == Reduction::Mean) {
      output.fill_(std::numeric_limits<double>::quiet_NaN());
    } else {
      output.zero_();
    }
    total_weight.zero_();
    return;
  }

  const Tensor target =
      (target_arg.scalar_type() == ScalarType::Long ? target_arg : target_arg.to(ScalarType::Long)).contiguous();
  const Tensor input = input_arg.contiguous();
  const Tensor weight = has_weight ? weight_arg.contiguous() : input;

  const int64_t num_outputs = input_arg.dim() == 1 ? 1 : target.numel();

  Tensor unreduced_loss;
  if (reduction == Reduction::None) {
    unreduced_loss = output;
  } else {
    unreduced_loss = at::empty(target_arg.sizes(), input.options());
  }

  Tensor sample_weights = at::empty(target_arg.sizes(), input.options());

  const auto params = nllnd_loss_params(input, ignore_index, has_weight, is2D);

  MPSStream* stream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      auto pso = lib.getPipelineStateForFunc("nllnd_loss_forward_" + scalarToMetalTypeString(input_arg));
      auto encoder = stream->commandEncoder();
      [encoder setComputePipelineState:pso];
      mtl_setArgs(encoder, input, target, weight, unreduced_loss, sample_weights);

      constexpr auto max_threads = int64_t{std::numeric_limits<uint32_t>::max()};
      auto dispatch_params = params;
      for (auto offset = int64_t{0}; offset < num_outputs; offset += max_threads) {
        dispatch_params.tid_offset = offset;
        mtl_setArgs<5>(encoder, dispatch_params, stream->getErrorBuffer());
        mtl_dispatch1DJob(encoder, pso, std::min(max_threads, num_outputs - offset));
      }
    }
  });

  total_weight.copy_(at::sum(sample_weights));

  if (reduction == Reduction::None) {
    // output already holds unreduced_loss
  } else if (reduction == Reduction::Sum) {
    output.copy_(at::sum(unreduced_loss));
  } else if (reduction == Reduction::Mean) {
    output.copy_(at::sum(unreduced_loss).div(total_weight));
  }
}

} // namespace mps

// APIs exposed to at::native scope

// HuberLoss
static void huber_mps_kernel(TensorIterator& iter, double delta) {
  const auto dtype = iter.common_dtype();
  TORCH_CHECK_NOT_IMPLEMENTED(mps::supportedFloatingType(dtype), "\"huber_mps\" not implemented for '", dtype, "'");
  mps::lib.exec_binary_kernel(iter, "huber", delta, at::toOpMathType(dtype));
}

REGISTER_DISPATCH(huber_stub, &huber_mps_kernel)

Tensor& huber_loss_backward_out_mps(const Tensor& grad_output,
                                    const Tensor& input,
                                    const Tensor& target,
                                    int64_t reduction,
                                    double delta,
                                    Tensor& grad_input) {
  // Mixed dtypes are rejected up front rather than promoted the way CPU does; test_mps expects
  // this error.
  TORCH_CHECK(input.scalar_type() == target.scalar_type() && input.scalar_type() == grad_output.scalar_type() &&
                  input.scalar_type() == grad_input.scalar_type(),
              "huber_loss_backward: expected all tensors to have the same dtype, but got input ",
              input.scalar_type(),
              ", target ",
              target.scalar_type(),
              ", grad_output ",
              grad_output.scalar_type(),
              " and grad_input ",
              grad_input.scalar_type());
  TORCH_CHECK(mps::supportedFloatingType(input.scalar_type()), "huber_loss_backward_mps only supports floating types");
  // grad_output is a scalar unless the reduction is None (where norm is 1), so
  // folding the mean's 1/numel into it is a 0-dim op and leaves alpha for delta.
  const double norm = reduction == Reduction::Mean && input.numel() > 0 ? 1.0 / input.numel() : 1.0;
  const Tensor grad = norm == 1.0 ? grad_output : grad_output.mul(norm);
  auto iter = TensorIteratorConfig()
                  .add_output(grad_input)
                  .add_const_input(input)
                  .add_const_input(target)
                  .add_const_input(grad)
                  .build();
  mps::lib.exec_ternary_kernel(iter, "huber_backward", delta, at::toOpMathType(iter.common_dtype()));
  return grad_input;
}

// MSELoss
static void mse_mps_kernel(TensorIteratorBase& iter) {
  TORCH_CHECK(mps::supportedFloatingType(iter.common_dtype()), "mse_mps only supports floating types");
  mps::lib.exec_binary_kernel(iter, "mse", std::nullopt, std::nullopt, iter.common_dtype());
}

static void mse_backward_mps_kernel(TensorIterator& iter, const Scalar& value) {
  TORCH_CHECK(mps::supportedFloatingType(iter.common_dtype()), "mse_backward_mps only supports floating types");
  mps::lib.exec_ternary_kernel(iter, "mse_backward", value);
}

REGISTER_DISPATCH(mse_stub, &mse_mps_kernel)
REGISTER_DISPATCH(mse_backward_stub, &mse_backward_mps_kernel)

Tensor mse_loss_backward_mps(const Tensor& grad_output, const Tensor& input, const Tensor& target, int64_t reduction) {
  Tensor grad_input = at::empty_like(input);
  return mse_loss_backward_out(grad_output, input, target, reduction, grad_input);
}

// BCELoss
Tensor& binary_cross_entropy_out_mps(const Tensor& input,
                                     const Tensor& target,
                                     const std::optional<Tensor>& weight_opt,
                                     int64_t reduction,
                                     Tensor& loss) {
  mps::bce_loss_forward_impl(input, target, weight_opt, reduction, loss);
  return loss;
}

Tensor binary_cross_entropy_mps(const Tensor& input,
                                const Tensor& target,
                                const std::optional<Tensor>& weight_opt,
                                int64_t reduction) {
  Tensor loss = at::empty_like(input);
  return binary_cross_entropy_out_mps(input, target, weight_opt, reduction, loss);
}

Tensor& binary_cross_entropy_backward_out_mps(const Tensor& grad_output,
                                              const Tensor& input,
                                              const Tensor& target,
                                              const std::optional<Tensor>& weight_opt,
                                              int64_t reduction,
                                              Tensor& grad_input) {
  const auto grad = weight_opt && weight_opt->defined() ? grad_output.mul(*weight_opt) : grad_output;
  auto iter = TensorIteratorConfig()
                  .add_output(grad_input)
                  .add_const_input(input)
                  .add_const_input(target)
                  .add_const_input(grad)
                  .build();
  const double norm = reduction == Reduction::Mean ? 1.0 / input.numel() : 1.0;
  mps::lib.exec_ternary_kernel(iter, "bce_backward", norm, at::toOpMathType(iter.common_dtype()));
  return grad_input;
}

Tensor binary_cross_entropy_backward_mps(const Tensor& grad_output,
                                         const Tensor& input,
                                         const Tensor& target,
                                         const std::optional<Tensor>& weight_opt,
                                         int64_t reduction) {
  Tensor grad_input = at::empty_like(input);
  return binary_cross_entropy_backward_out_mps(grad_output, input, target, weight_opt, reduction, grad_input);
}

// SmoothL1Loss
static void smooth_l1_mps_kernel(TensorIteratorBase& iter, double beta) {
  const auto dtype = iter.common_dtype();
  TORCH_CHECK_NOT_IMPLEMENTED(mps::supportedFloatingType(dtype), "\"smooth_l1_mps\" not implemented for '", dtype, "'");
  mps::lib.exec_binary_kernel(iter, "smooth_l1", beta, at::toOpMathType(dtype));
}

REGISTER_DISPATCH(smooth_l1_stub, &smooth_l1_mps_kernel)

Tensor& smooth_l1_loss_backward_out_mps(const Tensor& grad_output,
                                        const Tensor& input,
                                        const Tensor& target,
                                        int64_t reduction,
                                        double beta,
                                        Tensor& grad_input) {
  TORCH_CHECK(beta >= 0, "smooth_l1_loss_backward does not support negative values for beta.");
  TORCH_CHECK_NOT_IMPLEMENTED(input.scalar_type() != kBool && !isComplexType(input.scalar_type()),
                              "smooth_l1_loss_backward_mps not implemented for ",
                              input.scalar_type());
  // grad_output is a scalar unless the reduction is None (where norm is 1), so
  // folding the mean's 1/numel into it is a 0-dim op and leaves alpha for beta.
  // Integer inputs truncate norm first, like the CPU kernel's norm.to<scalar_t>().
  const double norm = reduction == Reduction::Mean && input.numel() > 0 ? 1.0 / input.numel() : 1.0;
  const Scalar norm_scalar = isFloatingType(input.scalar_type()) ? Scalar(norm) : Scalar(static_cast<int64_t>(norm));
  const Tensor grad = norm == 1.0 ? grad_output : grad_output.mul(norm_scalar);
  auto iter = TensorIteratorConfig()
                  .add_output(grad_input)
                  .add_const_input(input)
                  .add_const_input(target)
                  .add_const_input(grad)
                  .promote_inputs_to_common_dtype(true)
                  .cast_common_dtype_to_outputs(true)
                  .enforce_safe_casting_to_output(true)
                  .build();
  mps::lib.exec_ternary_kernel(iter, "smooth_l1_backward", beta, at::toOpMathType(iter.common_dtype()));
  return grad_input;
}

// NLLLoss
TORCH_IMPL_FUNC(nll_loss_backward_out_mps)
(const Tensor& grad_output,
 const Tensor& self,
 const Tensor& target,
 OptionalTensorRef weight_opt,
 int64_t reduction,
 int64_t ignore_index,
 const Tensor& total_weight,
 const Tensor& grad_input) {
  const Tensor& weight = weight_opt.getTensorRef();

  mps::nllnd_loss_backward_impl(
      (Tensor&)grad_input, grad_output, self, target, weight, reduction, ignore_index, total_weight, false);
  return;
}

TORCH_IMPL_FUNC(nll_loss_forward_out_mps)
(const Tensor& self,
 const Tensor& target,
 const OptionalTensorRef weight_opt,
 int64_t reduction,
 int64_t ignore_index,
 const Tensor& output,
 const Tensor& total_weight) {
  const Tensor& weight = weight_opt.getTensorRef();

  mps::nllnd_loss_forward_impl(
      (Tensor&)output, (Tensor&)total_weight, self, target, weight, reduction, ignore_index, false);

  return;
}

inline void check_inputs_nll_loss2d(const Tensor& input, const Tensor& target, const Tensor& weight) {
  TORCH_CHECK(target.dim() == 3,
              "only batches of spatial targets supported (3D tensors)"
              " but got targets of dimension: ",
              target.dim());
  TORCH_CHECK(input.dim() == 4,
              "only batches of spatial inputs supported (4D tensors), "
              "but got input of dimension: ",
              input.dim());
  TORCH_CHECK(!weight.defined() || weight.numel() == input.size(1),
              "weight tensor should be defined either for all or no classes");

  const int64_t input0 = input.size(0);
  const int64_t input2 = input.size(2);
  const int64_t input3 = input.size(3);
  const int64_t target0 = target.size(0);
  const int64_t target1 = target.size(1);
  const int64_t target2 = target.size(2);
  TORCH_CHECK(input0 == target0 && input2 == target1 && input3 == target2,
              "size mismatch (got input: ",
              input.sizes(),
              " , target: ",
              target.sizes());
}

static void nll_loss2d_forward_out_mps_template(Tensor& output,
                                                Tensor& total_weight,
                                                const Tensor& input,
                                                const Tensor& target,
                                                const Tensor& weight,
                                                int64_t reduction,
                                                int64_t ignore_index) {
  check_inputs_nll_loss2d(input, target, weight);
  total_weight.resize_({});

  mps::nllnd_loss_forward_impl(output, total_weight, input, target, weight, reduction, ignore_index, true);

  return;
}

std::tuple<Tensor&, Tensor&> nll_loss2d_forward_out_mps(const Tensor& self,
                                                        const Tensor& target,
                                                        const std::optional<Tensor>& weight_opt,
                                                        int64_t reduction,
                                                        int64_t ignore_index,
                                                        Tensor& output,
                                                        Tensor& total_weight) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> weight_maybe_owned = at::borrow_from_optional_tensor(weight_opt);
  const Tensor& weight = *weight_maybe_owned;

  nll_loss2d_forward_out_mps_template(output, total_weight, self, target, weight, reduction, ignore_index);
  return std::tuple<Tensor&, Tensor&>(output, total_weight);
}

std::tuple<Tensor, Tensor> nll_loss2d_forward_mps(const Tensor& self,
                                                  const Tensor& target,
                                                  const std::optional<Tensor>& weight_opt,
                                                  int64_t reduction,
                                                  int64_t ignore_index) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> weight_maybe_owned = at::borrow_from_optional_tensor(weight_opt);
  const Tensor& weight = *weight_maybe_owned;

  auto output = at::empty({0}, self.options());
  auto total_weight = at::empty({0}, self.options());
  at::native::nll_loss2d_forward_out_mps(self, target, weight, reduction, ignore_index, output, total_weight);
  return std::make_tuple(output, total_weight);
}

static void nll_loss2d_backward_out_mps_template(Tensor& grad_input,
                                                 const Tensor& grad_output,
                                                 const Tensor& input,
                                                 const Tensor& target,
                                                 const Tensor& weight,
                                                 int64_t reduction,
                                                 int64_t ignore_index,
                                                 const Tensor& total_weight) {
  check_inputs_nll_loss2d(input, target, weight);
  if (reduction == Reduction::None) {
    TORCH_CHECK(grad_output.dim() == 3,
                "grad_output must have same dimension as target (3) but got dimension: ",
                grad_output.sizes());
    TORCH_CHECK(grad_output.sizes() == target.sizes(),
                "size mismatch (got grad_output: ",
                grad_output.sizes(),
                " target: ",
                target.sizes());
  } else {
    TORCH_CHECK(grad_output.dim() <= 1 && grad_output.numel() == 1,
                "Expected a single element grad_output tensor, but got: ",
                grad_output.sizes());
  }
  grad_input.resize_as_(input);
  TORCH_CHECK(grad_input.is_contiguous(), "grad_input must be contiguous");
  TORCH_CHECK(total_weight.numel() == 1,
              "expected total_weight to be a single element tensor, got: ",
              total_weight.sizes(),
              " (",
              total_weight.numel(),
              " elements)");

  mps::nllnd_loss_backward_impl(
      grad_input, grad_output, input, target, weight, reduction, ignore_index, total_weight, true);

  return;
}

Tensor& nll_loss2d_backward_out_mps(const Tensor& grad_output,
                                    const Tensor& self,
                                    const Tensor& target,
                                    const std::optional<Tensor>& weight_opt,
                                    int64_t reduction,
                                    int64_t ignore_index,
                                    const Tensor& total_weight,
                                    Tensor& grad_input) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> weight_maybe_owned = at::borrow_from_optional_tensor(weight_opt);
  const Tensor& weight = *weight_maybe_owned;

  nll_loss2d_backward_out_mps_template(
      grad_input, grad_output, self, target, weight, reduction, ignore_index, total_weight);
  return grad_input;
}

Tensor nll_loss2d_backward_mps(const Tensor& grad_output,
                               const Tensor& self,
                               const Tensor& target,
                               const std::optional<Tensor>& weight_opt,
                               int64_t reduction,
                               int64_t ignore_index,
                               const Tensor& total_weight) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> weight_maybe_owned = at::borrow_from_optional_tensor(weight_opt);
  const Tensor& weight = *weight_maybe_owned;

  auto grad_input = at::zeros_like(self);
  nll_loss2d_backward_out_mps(grad_output, self, target, weight, reduction, ignore_index, total_weight, grad_input);
  return grad_input;
}

template <typename index_t>
std::string_view get_index_type_str() {
  if constexpr (std::is_same_v<index_t, int32_t>) {
    return "int32_t";
  } else if constexpr (std::is_same_v<index_t, int64_t>) {
    return "int64_t";
  } else {
    static_assert(false && sizeof(index_t), "unsupported index type");
  }
}

template <typename index_t, bool beta = false>
static void ctc_loss_mps_kernel(const std::optional<Tensor>& loss,
                                Tensor& log_alpha,
                                const Tensor& log_probs,
                                const Tensor& targets,
                                const Tensor& input_lengths_t,
                                const Tensor& target_lengths_t,
                                const Tensor& target_batch_offsets_t,
                                int64_t BLANK,
                                int64_t max_input_length,
                                int64_t max_target_length,
                                int64_t batch_size,
                                int64_t tg_target_stride) {
  using namespace mps;
  MPSStream* mpsStream = getCurrentMPSStream();

  dispatch_sync_with_rethrow(mpsStream->queue(), ^() {
    @autoreleasepool {
      id<MTLComputeCommandEncoder> computeEncoder = mpsStream->commandEncoder();
      id<MTLComputePipelineState> pso = mps::lib.getPipelineStateForFunc(fmt::format("ctc_loss{}_{}_{}_{}",
                                                                                     beta ? "_backward_log_beta" : "",
                                                                                     scalarToMetalTypeString(log_probs),
                                                                                     scalarToMetalTypeString(targets),
                                                                                     get_index_type_str<index_t>()));
      const uint32_t TG_SIZE = std::min<int64_t>([pso maxTotalThreadsPerThreadgroup], 2 * max_target_length + 1);
      [computeEncoder setComputePipelineState:pso];

      CTCLossParams<index_t> params;

      params.max_input_length = max_input_length;
      params.max_target_length = max_target_length;
      params.batch_size = batch_size;
      params.BLANK = BLANK;
      params.tg_target_stride = tg_target_stride;
      params.log_probs_time_stride = log_probs.stride(0);
      params.log_probs_batch_stride = log_probs.stride(1);
      params.log_probs_token_stride = log_probs.stride(2);
      params.log_alpha_batch_stride = log_alpha.stride(0);
      params.log_alpha_time_stride = log_alpha.stride(1);
      params.log_alpha_target_stride = log_alpha.stride(2);

      if constexpr (beta) {
        mtl_setArgs(computeEncoder,
                    log_alpha,
                    log_probs,
                    targets,
                    input_lengths_t,
                    target_lengths_t,
                    target_batch_offsets_t,
                    params);

      } else {
        TORCH_INTERNAL_ASSERT(loss.has_value(), "loss tensor must have a value when beta=false");
        mtl_setArgs(computeEncoder,
                    *loss,
                    log_alpha,
                    log_probs,
                    targets,
                    input_lengths_t,
                    target_lengths_t,
                    target_batch_offsets_t,
                    params);
      }
      [computeEncoder dispatchThreadgroups:MTLSizeMake(batch_size, 1, 1)
                     threadsPerThreadgroup:MTLSizeMake(TG_SIZE, 1, 1)];
    }
  });
}

static void ctc_loss_mps_check(const Tensor& log_probs,
                               const Tensor& targets,
                               IntArrayRef input_lengths,
                               IntArrayRef target_lengths,
                               int64_t BLANK) {
  TORCH_CHECK(log_probs.dim() == 3, "log_probs must be 3-D (T, N, C)");
  TORCH_CHECK(targets.dim() >= 1 && targets.dim() <= 2, "targets must have 1 or 2 dims");
  int64_t batch_size = log_probs.size(1);
  int64_t num_labels = log_probs.size(2);
  TORCH_CHECK((0 <= BLANK) && (BLANK < num_labels), "blank must be in label range");
  TORCH_CHECK((int64_t)input_lengths.size() == batch_size, "input_lengths must be of size batch_size");
  TORCH_CHECK((int64_t)target_lengths.size() == batch_size, "target_lengths must be of size batch_size");
  TORCH_CHECK(log_probs.numel() > 0, "log_probs tensor must not be empty");
}

std::tuple<Tensor, Tensor> ctc_loss_mps(const Tensor& log_probs,
                                        const Tensor& targets,
                                        IntArrayRef input_lengths,
                                        IntArrayRef target_lengths,
                                        int64_t BLANK,
                                        bool zero_infinity) {
  using namespace mps;
  ctc_loss_mps_check(log_probs, targets, input_lengths, target_lengths, BLANK);

  int64_t batch_size = log_probs.size(1);

  // Compute per-batch target offsets and max target length
  int64_t tg_target_stride = (targets.dim() == 1) ? targets.stride(0) : targets.stride(1);
  int64_t max_target_length = 0;
  int64_t max_input_length = log_probs.size(0);
  std::vector<int64_t> target_batch_offsets(batch_size);

  int64_t pos = 0;
  int64_t tg_batch_stride = targets.stride(0);
  for (int64_t i = 0; i < batch_size; i++) {
    TORCH_CHECK(target_lengths[i] >= 0,
                "Expected target_lengths to have value at least 0, but got value ",
                target_lengths[i],
                " (while checking arguments for ctc_loss_mps)");
    TORCH_CHECK(input_lengths[i] >= 0 && input_lengths[i] <= max_input_length,
                "Expected input_lengths to be in [0, ",
                max_input_length,
                "], but got value ",
                input_lengths[i],
                " (while checking arguments for ctc_loss_mps)");
    if (targets.dim() == 1) {
      target_batch_offsets[i] = pos;
      pos += target_lengths[i];
    } else {
      target_batch_offsets[i] = i * tg_batch_stride;
    }
    max_target_length = std::max(max_target_length, target_lengths[i]);
  }
  if (targets.dim() == 2) {
    TORCH_CHECK(targets.size(1) >= max_target_length,
                "Expected targets to have size at least ",
                max_target_length,
                " at dimension 1, but got size ",
                targets.size(1),
                " (while checking arguments for ctc_loss_mps)");
  }

  Tensor loss = at::empty({batch_size}, log_probs.options());
  Tensor log_alpha = at::empty({batch_size, log_probs.size(0), 2 * max_target_length + 1}, log_probs.options());

  if (batch_size == 0) {
    return {loss, log_alpha};
  }

  bool can_use_32bit_index_math = at::native::canUse32BitIndexMath(log_probs) &&
      at::native::canUse32BitIndexMath(targets) && at::native::canUse32BitIndexMath(log_alpha);
  // NOTE: Used signed types because if we attempt to use unsigned, the
  // `at::tensor` calls below raise the error: "Exception: "tensor_cpu" not
  // implemented for 'UInt32'"
  auto metadata_dtype = can_use_32bit_index_math ? kInt : kLong;

  // Move metadata to MPS device
  Tensor input_lengths_t = at::tensor(input_lengths, log_probs.options().dtype(metadata_dtype));
  Tensor target_lengths_t = at::tensor(target_lengths, log_probs.options().dtype(metadata_dtype));
  Tensor target_batch_offsets_t = at::tensor(target_batch_offsets, log_probs.options().dtype(metadata_dtype));

  if (can_use_32bit_index_math) {
    ctc_loss_mps_kernel<int32_t>(loss,
                                 log_alpha,
                                 log_probs,
                                 targets,
                                 input_lengths_t,
                                 target_lengths_t,
                                 target_batch_offsets_t,
                                 BLANK,
                                 max_input_length,
                                 max_target_length,
                                 batch_size,
                                 tg_target_stride);
  } else {
    ctc_loss_mps_kernel<int64_t>(loss,
                                 log_alpha,
                                 log_probs,
                                 targets,
                                 input_lengths_t,
                                 target_lengths_t,
                                 target_batch_offsets_t,
                                 BLANK,
                                 max_input_length,
                                 max_target_length,
                                 batch_size,
                                 tg_target_stride);
  }

  return {std::move(loss), std::move(log_alpha)};
}

template <typename index_t>
static void ctc_loss_backward_mps_kernel(Tensor& grad,
                                         const Tensor& grad_out,
                                         const Tensor& log_alpha,
                                         const Tensor& log_probs,
                                         const Tensor& targets,
                                         const Tensor& input_lengths_t,
                                         const Tensor& target_lengths_t,
                                         const Tensor& loss,
                                         const Tensor& target_batch_offsets_t,
                                         int64_t BLANK,
                                         int64_t max_input_length,
                                         int64_t batch_size,
                                         bool zero_infinity) {
  using namespace mps;
  // Derive max_target_length from log_alpha shape (same as CUDA backward).
  int64_t max_target_length = log_alpha.size(2) / 2;
  int64_t tg_target_stride = (targets.dim() == 1) ? targets.stride(0) : targets.stride(1);

  Tensor log_beta = at::empty_like(log_alpha, LEGACY_CONTIGUOUS_MEMORY_FORMAT);

  ctc_loss_mps_kernel<index_t, /*beta=*/true>(/*loss=*/std::nullopt,
                                              log_beta,
                                              log_probs,
                                              targets,
                                              input_lengths_t,
                                              target_lengths_t,
                                              target_batch_offsets_t,
                                              BLANK,
                                              max_input_length,
                                              max_target_length,
                                              batch_size,
                                              tg_target_stride);

  MPSStream* mpsStream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(mpsStream->queue(), ^() {
    @autoreleasepool {
      id<MTLComputeCommandEncoder> computeEncoder = mpsStream->commandEncoder();
      id<MTLComputePipelineState> pso =
          mps::lib.getPipelineStateForFunc(fmt::format("ctc_loss_backward_collect_{}_{}_{}",
                                                       scalarToMetalTypeString(log_probs),
                                                       scalarToMetalTypeString(targets),
                                                       get_index_type_str<index_t>()));
      [computeEncoder setComputePipelineState:pso];

      CTCLossBackwardCollectParams<index_t> params;
      params.BLANK = BLANK;
      params.max_input_length = max_input_length;
      params.max_target_length = max_target_length;
      params.num_labels = log_probs.size(2);
      params.tg_target_stride = tg_target_stride;
      params.log_probs_time_stride = log_probs.stride(0);
      params.log_probs_batch_stride = log_probs.stride(1);
      params.log_probs_token_stride = log_probs.stride(2);
      params.log_alpha_beta_batch_stride = log_alpha.stride(0);
      params.log_alpha_beta_time_stride = log_alpha.stride(1);
      params.log_alpha_beta_target_stride = log_alpha.stride(2);
      params.grad_time_stride = grad.stride(0);
      params.grad_batch_stride = grad.stride(1);
      params.grad_token_stride = grad.stride(2);
      params.grad_out_batch_stride = grad_out.stride(0);
      params.zero_infinity = zero_infinity;

      mtl_setArgs(computeEncoder,
                  grad,
                  grad_out,
                  log_alpha,
                  log_beta,
                  log_probs,
                  targets,
                  input_lengths_t,
                  target_lengths_t,
                  loss,
                  target_batch_offsets_t,
                  params);
      [computeEncoder
                dispatchThreads:MTLSizeMake(max_input_length, batch_size, 1)
          threadsPerThreadgroup:MTLSizeMake(
                                    std::min<int64_t>([pso maxTotalThreadsPerThreadgroup], max_input_length), 1, 1)];
    }
  });
}

Tensor ctc_loss_backward_mps(const Tensor& grad_out,
                             const Tensor& log_probs,
                             const Tensor& targets,
                             IntArrayRef input_lengths,
                             IntArrayRef target_lengths,
                             const Tensor& loss,
                             const Tensor& log_alpha,
                             int64_t BLANK,
                             bool zero_infinity) {
  using namespace mps;

  int64_t batch_size = log_probs.size(1);
  int64_t max_input_length = log_probs.size(0);

  std::vector<int64_t> target_batch_offsets(batch_size);

  int64_t pos = 0;
  int64_t tg_batch_stride = targets.stride(0);
  for (int64_t i = 0; i < batch_size; i++) {
    if (targets.dim() == 1) {
      target_batch_offsets[i] = pos;
      pos += target_lengths[i];
    } else {
      target_batch_offsets[i] = i * tg_batch_stride;
    }
  }

  // grad initialized to neginf (log-domain zero) for the scatter-logsumexp in kernel 2.
  Tensor grad = at::full_like(log_probs, -std::numeric_limits<double>::infinity(), LEGACY_CONTIGUOUS_MEMORY_FORMAT);

  if (batch_size == 0 || max_input_length == 0) {
    return grad;
  }

  bool can_use_32bit_index_math = at::native::canUse32BitIndexMath(log_probs) &&
      at::native::canUse32BitIndexMath(targets) && at::native::canUse32BitIndexMath(log_alpha);
  auto metadata_dtype = can_use_32bit_index_math ? kInt : kLong;

  Tensor input_lengths_t = at::tensor(input_lengths, log_probs.options().dtype(metadata_dtype));
  Tensor target_lengths_t = at::tensor(target_lengths, log_probs.options().dtype(metadata_dtype));
  Tensor target_batch_offsets_t = at::tensor(target_batch_offsets, log_probs.options().dtype(metadata_dtype));

  if (can_use_32bit_index_math) {
    ctc_loss_backward_mps_kernel<int32_t>(grad,
                                          grad_out,
                                          log_alpha,
                                          log_probs,
                                          targets,
                                          input_lengths_t,
                                          target_lengths_t,
                                          loss,
                                          target_batch_offsets_t,
                                          BLANK,
                                          max_input_length,
                                          batch_size,
                                          zero_infinity);
  } else {
    ctc_loss_backward_mps_kernel<int64_t>(grad,
                                          grad_out,
                                          log_alpha,
                                          log_probs,
                                          targets,
                                          input_lengths_t,
                                          target_lengths_t,
                                          loss,
                                          target_batch_offsets_t,
                                          BLANK,
                                          max_input_length,
                                          batch_size,
                                          zero_infinity);
  }

  return grad;
}

} // namespace at::native
