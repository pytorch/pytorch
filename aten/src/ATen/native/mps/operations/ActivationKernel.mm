#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/Dispatch.h>
#include <ATen/TensorIterator.h>
#include <ATen/mps/MPSProfiler.h>
#include <ATen/native/Activation.h>
#include <ATen/native/mps/OperationUtils.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/add.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/glu_backward_native.h>
#include <ATen/ops/glu_native.h>
#include <ATen/ops/log_sigmoid_backward_native.h>
#include <ATen/ops/log_sigmoid_forward_native.h>
#include <ATen/ops/mul.h>
#include <ATen/ops/mul_native.h>
#include <ATen/ops/relu_native.h>
#include <ATen/ops/rsub.h>
#include <ATen/ops/sigmoid.h>
#include <ATen/ops/sigmoid_backward_native.h>
#include <ATen/ops/sigmoid_native.h>
#endif
#include <ATen/native/BinaryOps.h>
#include <ATen/native/Gelu.h>
#include <ATen/native/mps/kernels/Activation.h>
#include <fmt/format.h>

namespace at::native {

#ifndef PYTORCH_JIT_COMPILE_SHADERS
static auto& lib = mps::MetalShaderLibrary::getBundledLibrary();
#else
#include <ATen/native/mps/ActivationKernel_metallib.h>
#endif

Tensor relu_mps(const Tensor& self) {
  TORCH_CHECK(!self.is_complex(), "relu is not supported for complex types");
  auto output = at::empty_like(self);
  if (output.numel() == 0)
    return output;
  auto iter = at::TensorIteratorConfig().add_output(output).add_input(self).build();
  lib.exec_unary_kernel(iter, "relu");
  return output;
}

Tensor& relu_mps_(Tensor& self) {
  TORCH_CHECK(!self.is_complex(), "relu is not supported for complex types");
  if (self.numel() == 0)
    return self;
  auto iter = at::TensorIteratorConfig().add_output(self).add_input(self).set_check_mem_overlap(false).build();
  lib.exec_unary_kernel(iter, "relu");
  return self;
}

static void hardshrink_kernel(TensorIteratorBase& iter, const Scalar& lambda = 0.5) {
  lib.exec_unary_kernel(iter, "hardshrink", lambda);
}

static void softshrink_kernel(TensorIteratorBase& iter, const Scalar& lambda = 0.5) {
  lib.exec_unary_kernel(iter, "softshrink", lambda);
}

static void shrink_backward_kernel(TensorIteratorBase& iter, const Scalar& lambda = 0.5) {
  lib.exec_binary_kernel(iter, "shrink_backward", lambda);
}

static void hardsigmoid_kernel(TensorIteratorBase& iter) {
  lib.exec_unary_kernel(iter, "hardsigmoid");
}

static void hardsigmoid_backward_kernel(TensorIteratorBase& iter) {
  lib.exec_binary_kernel(iter, "hardsigmoid_backward");
}

static void hardswish_kernel(at::TensorIterator& iter) {
  lib.exec_unary_kernel(iter, "hardswish");
}

static void hardswish_backward_kernel(at::TensorIterator& iter) {
  lib.exec_binary_kernel(iter, "hardswish_backward");
}

static void elu_kernel(TensorIteratorBase& iter, const Scalar& alpha, const Scalar& scale, const Scalar& input_scale) {
  AT_DISPATCH_FLOATING_TYPES_AND2(c10::kHalf, c10::kBFloat16, iter.common_dtype(), "elu_mps", [&]() {
    ELUParams<scalar_t> params{alpha.to<scalar_t>(), scale.to<scalar_t>(), input_scale.to<scalar_t>()};
    lib.exec_unary_kernel_with_params(
        iter, "elu", params, fmt::format("ELUParams_{}", mps::scalarToMetalTypeString(iter.common_dtype())));
  });
}

static void elu_backward_kernel(TensorIteratorBase& iter,
                                const Scalar& alpha,
                                const Scalar& scale,
                                const Scalar& input_scale,
                                bool is_result) {
  AT_DISPATCH_FLOATING_TYPES_AND2(c10::kHalf, c10::kBFloat16, iter.common_dtype(), "elu_backward_mps", [&]() {
    ELUBackwardParams<scalar_t> params{
        alpha.to<scalar_t>(), scale.to<scalar_t>(), input_scale.to<scalar_t>(), is_result};
    lib.exec_binary_kernel_with_params(
        iter,
        "elu_backward",
        params,
        fmt::format("ELUBackwardParams_{}", mps::scalarToMetalTypeString(iter.common_dtype())));
  });
}

static void silu_kernel(TensorIteratorBase& iter) {
  if (isComplexType(iter.common_dtype())) {
    auto out = iter.output(0);
    auto self = iter.input(0);
    at::mul_out(out, self, at::sigmoid(self));
    return;
  }
  lib.exec_unary_kernel(iter, "silu");
}

static void silu_backward_kernel(TensorIteratorBase& iter) {
  if (isComplexType(iter.common_dtype())) {
    auto grad_input = iter.output(0);
    auto grad_output = iter.input(0);
    auto self = iter.input(1);
    auto sig = at::sigmoid(self);
    auto one_minus_sig = at::rsub(sig, 1);
    auto inner = at::add(at::mul(self, one_minus_sig), 1);
    grad_input.copy_(at::mul(grad_output, at::mul(sig, inner)));
    return;
  }
  lib.exec_binary_kernel(iter, "silu_backward");
}

static void mish_kernel(TensorIteratorBase& iter) {
  lib.exec_unary_kernel(iter, "mish");
}

static void mish_backward_kernel(TensorIterator& iter) {
  lib.exec_binary_kernel(iter, "mish_backward");
}

static void leaky_relu_kernel(TensorIteratorBase& iter, const Scalar& negative_slope) {
  lib.exec_unary_kernel(iter, "leaky_relu", negative_slope);
}

static void leaky_relu_backward_kernel(TensorIteratorBase& iter, const Scalar& negative_slope) {
  lib.exec_binary_kernel(iter, "leaky_relu_backward", negative_slope);
}

static void gelu_kernel(TensorIteratorBase& iter, GeluType approximate) {
  const char* name = (approximate == GeluType::Tanh) ? "gelu_tanh" : "gelu";
  lib.exec_unary_kernel(iter, name);
}

static void gelu_backward_kernel(TensorIteratorBase& iter, GeluType approximate) {
  const char* name = (approximate == GeluType::Tanh) ? "gelu_tanh_backward" : "gelu_backward";
  lib.exec_binary_kernel(iter, name);
}

static void sigmoid_backward_kernel(TensorIteratorBase& iter) {
  lib.exec_binary_kernel(iter, "sigmoid_backward");
}

// Collapse a tensor around the split dim into [outer, 2*L], where
// L = (size[dim]/2) * product(size[dim+1:]) is the contiguous run per outer row
// (the two halves of a row sit L elements apart). Valid only for a contiguous
// source; used by the dense fast paths below.
static std::pair<int64_t, int64_t> glu_outer_and_half_run(IntArrayRef sizes, int64_t wrap_dim) {
  int64_t inner = 1;
  for (auto d = wrap_dim + 1; d < static_cast<int64_t>(sizes.size()); ++d) {
    inner *= sizes[d];
  }
  int64_t outer = 1;
  for (const auto d : c10::irange(wrap_dim)) {
    outer *= sizes[d];
  }
  return {outer, (sizes[wrap_dim] / 2) * inner};
}

// Forward: when the source is contiguous, read both halves of each collapsed
// row through a fixed L-element offset and dispatch a dense 2D grid; otherwise
// fall back to the strided binary kernel over the two narrowed halves.
TORCH_IMPL_FUNC(glu_out_mps)(const Tensor& self, const int64_t dim, const Tensor& output) {
  using namespace mps;
  if (output.numel() == 0) {
    return;
  }
  const auto wrap_dim = maybe_wrap_dim(dim, self.dim());

  if (self.is_contiguous() && output.is_contiguous()) {
    const auto collapsed = glu_outer_and_half_run(self.sizes(), wrap_dim);
    const int64_t outer = collapsed.first;
    const int64_t L = collapsed.second;
    auto mpsStream = getCurrentMPSStream();
    dispatch_sync_with_rethrow(mpsStream->queue(), ^() {
      @autoreleasepool {
        auto computeEncoder = mpsStream->commandEncoder();
        auto pso = lib.getPipelineStateForFunc(fmt::format("glu_dense_{}", scalarToMetalTypeString(self)));
        getMPSProfiler().beginProfileKernel(pso, "glu_dense", {self});
        [computeEncoder setComputePipelineState:pso];
        mtl_setArgs(computeEncoder, output, self, L);
        mtl_dispatch2DJob(computeEncoder, pso, L, outer);
        getMPSProfiler().endProfileKernel(pso);
      }
    });
    return;
  }

  // Fallback: strided binary over the two halves. They stay alive through the
  // launch, so borrowing into the iterator is safe here.
  const auto half = self.size(wrap_dim) / 2;
  auto firstHalf = self.narrow(wrap_dim, 0, half);
  auto secondHalf = self.narrow(wrap_dim, half, half);
  auto iter = TensorIteratorConfig().add_output(output).add_const_input(firstHalf).add_const_input(secondHalf).build();
  lib.exec_binary_kernel(iter, "glu");
}

// Dedicated MPS glu backward, following the CUDA implementation: a single fused
// pass over the halved shape that reaches the second halves of input/grad_input
// through fixed offsets, computing sigmoid internally (rather than reusing
// glu_backward_cpu_out, which computes sigmoid as a separate op). A contiguous
// source takes the dense 2D path; otherwise a strided kernel handles arbitrary
// layouts.
Tensor& glu_backward_mps_out(const Tensor& grad_output, const Tensor& input, int64_t dim, Tensor& grad_input) {
  using namespace mps;
  TORCH_CHECK(input.dim() > 0, "glu does not support 0-dimensional tensors");
  const auto wrap_dim = maybe_wrap_dim(dim, input.dim());
  const auto input_sizes = input.sizes();
  const int64_t nIn = input_sizes[wrap_dim];
  TORCH_CHECK(nIn % 2 == 0, "Halving dimension must be even, but dimension ", wrap_dim, " is size ", nIn);

  grad_input.resize_(input_sizes);

  DimVector iter_shape(input_sizes);
  const auto dim_size = nIn / 2;
  iter_shape[wrap_dim] = dim_size;
  TORCH_CHECK(grad_output.sizes() == IntArrayRef{iter_shape});
  if (grad_input.numel() == 0) {
    return grad_input;
  }

  if (input.is_contiguous() && grad_input.is_contiguous() && grad_output.is_contiguous()) {
    const auto collapsed = glu_outer_and_half_run(input_sizes, wrap_dim);
    const int64_t outer = collapsed.first;
    const int64_t L = collapsed.second;
    auto mpsStream = getCurrentMPSStream();
    dispatch_sync_with_rethrow(mpsStream->queue(), ^() {
      @autoreleasepool {
        auto computeEncoder = mpsStream->commandEncoder();
        auto pso = lib.getPipelineStateForFunc(fmt::format("glu_backward_dense_{}", scalarToMetalTypeString(input)));
        getMPSProfiler().beginProfileKernel(pso, "glu_backward_dense", {input, grad_output});
        [computeEncoder setComputePipelineState:pso];
        mtl_setArgs(computeEncoder, grad_input, input, grad_output, L);
        mtl_dispatch2DJob(computeEncoder, pso, L, outer);
        getMPSProfiler().endProfileKernel(pso);
      }
    });
    return grad_input;
  }

  auto iter = TensorIteratorConfig()
                  .add_output(grad_input)
                  .add_const_input(input)
                  .add_const_input(grad_output)
                  .resize_outputs(false)
                  .declare_static_shape(iter_shape)
                  .build();
  if (iter.numel() == 0) {
    return grad_input;
  }

  const auto I_byte_offset = input.strides()[wrap_dim] * dim_size * input.element_size();
  const auto gI_byte_offset = grad_input.strides()[wrap_dim] * dim_size * grad_input.element_size();

  // Capture by reference so the block keeps non-const access to the iterator.
  TensorIteratorBase& iter_ref = iter;
  auto mpsStream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(mpsStream->queue(), ^() {
    @autoreleasepool {
      auto computeEncoder = mpsStream->commandEncoder();
      auto pso = lib.getPipelineStateForFunc(fmt::format("glu_backward_{}", scalarToMetalTypeString(input)));
      getMPSProfiler().beginProfileKernel(pso, "glu_backward", {input, grad_output});
      [computeEncoder setComputePipelineState:pso];
      bind_iter_tensors(computeEncoder, iter_ref);
      mtl_setArgs<3>(computeEncoder,
                     iter_ref.shape(),
                     iter_ref.strides(0),
                     iter_ref.strides(1),
                     iter_ref.strides(2),
                     gI_byte_offset,
                     I_byte_offset,
                     static_cast<uint32_t>(iter_ref.ndim()));
      mtl_dispatch1DJob(computeEncoder, pso, iter_ref.numel());
      getMPSProfiler().endProfileKernel(pso);
    }
  });
  return grad_input;
}

Tensor glu_backward_mps(const Tensor& grad_output, const Tensor& input, int64_t dim) {
  auto grad_input = at::empty({0}, input.options());
  return glu_backward_mps_out(grad_output, input, dim, grad_input);
}

std::tuple<Tensor&, Tensor&> log_sigmoid_forward_out_mps(const Tensor& self, Tensor& output, Tensor& buffer) {
  // NOTE: buffer is only used by CPU dispatch, we just ignore it here
  output.resize_as_(self);
  if (self.numel() == 0) {
    return std::forward_as_tuple(output, buffer);
  }
  auto iter = at::TensorIteratorConfig().add_output(output).add_const_input(self).build();
  lib.exec_unary_kernel(iter, "log_sigmoid_forward");
  return std::forward_as_tuple(output, buffer);
}

std::tuple<Tensor, Tensor> log_sigmoid_forward_mps(const Tensor& self) {
  auto output = at::empty_like(self);
  auto buffer = at::empty({0}, self.options());
  log_sigmoid_forward_out_mps(self, output, buffer);
  return std::make_tuple(std::move(output), std::move(buffer));
}

Tensor& log_sigmoid_backward_mps_out(const Tensor& grad_output,
                                     const Tensor& self,
                                     const Tensor& buffer,
                                     Tensor& grad_input) {
  // NOTE: buffer is only used by CPU dispatch, we just ignore it here
  grad_input.resize_as_(self);
  if (self.numel() == 0) {
    return grad_input;
  }
  auto iter =
      at::TensorIteratorConfig().add_output(grad_input).add_const_input(self).add_const_input(grad_output).build();
  lib.exec_binary_kernel(iter, "log_sigmoid_backward");
  return grad_input;
}

Tensor log_sigmoid_backward_mps(const Tensor& grad_output, const Tensor& self, const Tensor& buffer) {
  auto grad_input = at::empty_like(grad_output);
  log_sigmoid_backward_mps_out(grad_output, self, buffer, grad_input);
  return grad_input;
}

REGISTER_DISPATCH(hardshrink_stub, hardshrink_kernel);
REGISTER_DISPATCH(softshrink_stub, softshrink_kernel);
REGISTER_DISPATCH(shrink_backward_stub, shrink_backward_kernel);
REGISTER_DISPATCH(hardsigmoid_stub, hardsigmoid_kernel);
REGISTER_DISPATCH(hardsigmoid_backward_stub, hardsigmoid_backward_kernel);
REGISTER_DISPATCH(hardswish_stub, hardswish_kernel);
REGISTER_DISPATCH(hardswish_backward_stub, hardswish_backward_kernel);
REGISTER_DISPATCH(elu_stub, elu_kernel);
REGISTER_DISPATCH(elu_backward_stub, elu_backward_kernel);
REGISTER_DISPATCH(leaky_relu_stub, leaky_relu_kernel);
REGISTER_DISPATCH(leaky_relu_backward_stub, leaky_relu_backward_kernel);
REGISTER_DISPATCH(silu_stub, silu_kernel);
REGISTER_DISPATCH(silu_backward_stub, silu_backward_kernel);
REGISTER_DISPATCH(mish_stub, mish_kernel);
REGISTER_DISPATCH(mish_backward_stub, mish_backward_kernel);
REGISTER_DISPATCH(GeluKernel, gelu_kernel);
REGISTER_DISPATCH(GeluBackwardKernel, gelu_backward_kernel);
REGISTER_DISPATCH(sigmoid_backward_stub, sigmoid_backward_kernel);

} // namespace at::native
