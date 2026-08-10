//  Copyright © 2022 Apple Inc.
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/ExpandUtils.h>
#include <ATen/mps/MPSProfiler.h>
#include <ATen/native/mps/MPSGraphSequoiaOps.h>
#include <ATen/native/mps/OperationUtils.h>
#include <ATen/ops/addmm.h>
#include <ATen/ops/linear_backward_native.h>
#include <ATen/ops/linear_native.h>
#include <ATen/ops/mm.h>
#include <ATen/ops/zeros.h>

namespace at::native {

using namespace mps;

// MPSNDArrayMatrixMultiplication and MPSGraph matrixMultiplication produce
// non-deterministic results for >2D fp16/bf16 inputs on Apple M5+ (Apple10 GPU family).
// Flatten to 2D to work around the issue (See https://github.com/pytorch/pytorch/issues/180776 )
static bool needs_nd_workaround(const Tensor& input) {
  static const bool is_m5_or_newer = is_apple_family_or_newer(AppleGPUFamily::APPLE_10_PLUS);
  return input.dim() > 2 && is_m5_or_newer && (input.scalar_type() == kHalf || input.scalar_type() == kBFloat16);
}

// Apple7/8 (M1/M2) MPSGraph matmul intermittently returns wrong results when the
// reduction dimension exceeds 2^15 and both output dimensions are at least 16; the
// corruption is allocator/session-state dependent and hits contiguous and transposed
// operands alike. Apple9+ is fine. mm/addmm already divert such shapes to the
// stride-aware metal kernels, but linear builds its own graph, so it has to test the
// GEMM it would form and delegate instead. Mirrors use_metal_mm in LinearAlgebra.mm.
static bool needs_mm_overflow_fallback(int64_t m, int64_t k, int64_t n) {
  static const bool is_affected_gpu = !is_apple_family_or_newer(AppleGPUFamily::APPLE_9_PLUS);
  constexpr int64_t max_mpsgraph_dim = 32768;
  constexpr int64_t min_matrix_dim = 16;
  return is_affected_gpu && k > max_mpsgraph_dim && m >= min_matrix_dim && n >= min_matrix_dim;
}

static void _mps_linear_nograph(const Tensor& input, const Tensor& weight, const Tensor& bias, Tensor& output) {
  bool is_bias_defined = bias.defined();

  auto mpsStream = getCurrentMPSStream();
  auto device = MPSDevice::getInstance()->device();

  const std::string key = "mps_linear" + getTensorsStringKey({input, weight, bias}, true, true);
  dispatch_sync_with_rethrow(mpsStream->queue(), ^() {
    @autoreleasepool {
      mpsStream->endKernelCoalescing();

      auto computeEncoder = mpsStream->commandEncoder();
      auto commandBuffer = mpsStream->commandBuffer();

      const auto mpsDataType = getMPSDataType(weight.scalar_type());

      auto inputNDArray = getMPSNDArray(input, input.sizes(), input.strides());
      auto outNDArray = getMPSNDArray(output, output.sizes(), output.strides());

      auto weightBuf = getMTLBufferStorage(weight);
      auto weightDesc = [MPSNDArrayDescriptor descriptorWithDataType:mpsDataType shape:getMPSShape(weight.sizes())];
      weightDesc.preferPackedRows = YES;
      [weightDesc transposeDimension:0 withDimension:1];
      auto weightNDArray = [[[MPSNDArray alloc] initWithBuffer:weightBuf
                                                        offset:weight.storage_offset() * weight.element_size()
                                                    descriptor:weightDesc] autorelease];

      if (is_bias_defined) {
        auto biasNDArray = getMPSNDArray(bias, bias.sizes(), bias.strides());
        auto cachedKernel = LookUpOrCreateCachedKernel<MPSCachedKernel>(key, [&]() {
          return [[[MPSNDArrayMatrixMultiplication alloc] initWithDevice:device sourceCount:3] autorelease];
        });
        auto kernel = cachedKernel->kernel<MPSNDArrayMatrixMultiplication>();

        getMPSProfiler().beginProfileKernel(kernel, "mps_linear", {input, weight, bias});
        [kernel encodeToCommandEncoder:computeEncoder
                         commandBuffer:commandBuffer
                          sourceArrays:@[ inputNDArray, weightNDArray, biasNDArray ]
                      destinationArray:outNDArray];
        getMPSProfiler().endProfileKernel(kernel);
      } else {
        auto cachedKernel = LookUpOrCreateCachedKernel<MPSCachedKernel>(key, [&]() {
          return [[[MPSNDArrayMatrixMultiplication alloc] initWithDevice:device sourceCount:2] autorelease];
        });
        auto kernel = cachedKernel->kernel<MPSNDArrayMatrixMultiplication>();
        getMPSProfiler().beginProfileKernel(kernel, "mps_linear", {input, weight, bias});
        [kernel encodeToCommandEncoder:computeEncoder
                         commandBuffer:commandBuffer
                          sourceArrays:@[ inputNDArray, weightNDArray ]
                      destinationArray:outNDArray];
        getMPSProfiler().endProfileKernel(kernel);
      }
    }
  });
}

Tensor _mps_linear(const Tensor& input, const Tensor& weight_arg, const std::optional<Tensor>& bias_opt) {
  // wT = transpose(weight);
  // y=x*wT+b

  TORCH_CHECK(supportedFloatingOrComplexType(input), "MPS device does not support linear for non-float inputs");
  TORCH_CHECK(input.is_mps(), "Tensor for argument input is on ", input.device(), " but expected on mps");
  TORCH_CHECK(supportedFloatingOrComplexType(weight_arg), "MPS device does not support linear for non-float weights");
  TORCH_CHECK(weight_arg.is_mps(), "Tensor for argument weight is on ", weight_arg.device(), " but expected on mps");

  const Tensor& bias = *(at::borrow_from_optional_tensor(bias_opt));
  const bool is_bias_defined = bias.defined();
  if (is_bias_defined) {
    TORCH_CHECK(bias.is_mps(), "Tensor for argument bias is on ", bias.device(), " but expected on mps");
    TORCH_CHECK(supportedFloatingOrComplexType(bias), "MPS device does not support linear for non-float bias");
  }

  auto weight = (weight_arg.dim() == 1) ? weight_arg.unsqueeze(0) : weight_arg;

  auto input_size = input.sizes();
  std::vector<int64_t> output_size(input_size.begin(), input_size.end() - 1);
  output_size.push_back(weight.size(0));

  TORCH_CHECK(input.size(-1) == weight_arg.size(-1),
              "linear(): input and weight.T shapes cannot be multiplied (",
              input.size(-2),
              "x",
              input.size(-1),
              " and ",
              weight_arg.size(-1),
              "x",
              weight_arg.size(-2),
              ")");

  if (is_bias_defined) {
    // Check bias and output shapes compatibility only.
    inferExpandGeometry_dimvector(bias.sizes(), bias.strides(), output_size);
  }

  Tensor output =
      at::empty(output_size, input.scalar_type(), std::nullopt, kMPS, std::nullopt, input.suggest_memory_format());

  if (output.numel() == 0) {
    return output;
  }

  // An empty reduction dimension (in_features == 0) makes the matmul term
  // zero, so the result is just the broadcast bias. Neither the NDArray fast
  // path nor MPSGraph can take a zero-length dimension (MPSNDArray asserts
  // and aborts the process).
  if (input.size(-1) == 0) {
    if (is_bias_defined) {
      output.copy_(bias.expand(output.sizes()));
    } else {
      output.zero_();
    }
    // Squeeze last dim of 1D linear
    return weight_arg.dim() != 1 ? output : output.squeeze(-1);
  }

  const bool is_complex = input.is_complex() || weight.is_complex() || (is_bias_defined && bias.is_complex());

  // See pytorch/pytorch#177116. Delegating to addmm/mm reaches the metal kernels, which
  // take the weight's real strides, so the transposed (column-major) operand costs nothing.
  if (!is_complex && needs_mm_overflow_fallback(input.numel() / input.size(-1), input.size(-1), weight.size(0))) {
    const auto input_2d = input.dim() != 2 ? input.reshape({-1, input.size(-1)}) : input;
    // addmm fuses the bias and routes rank-1 shapes to the GEMV kernels. A multi-dim bias
    // cannot broadcast against the 2D result, so it is added after the reshape instead.
    const bool fuse_bias = is_bias_defined && bias.dim() <= 1;
    auto result = (fuse_bias ? at::addmm(bias, input_2d, weight.t()) : at::mm(input_2d, weight.t())).view(output_size);
    if (is_bias_defined && !fuse_bias) {
      result.add_(bias);
    }
    // Squeeze last dim of 1D linear
    return weight_arg.dim() != 1 ? result : result.squeeze(-1);
  }

  // No-graph execution causes nonsense if these are non-contiguous.
  const bool is_contiguous = input.is_contiguous() && weight.is_contiguous() && bias.is_contiguous();

  if (is_macos_at_least(MacOSVersion::MACOS_15_0) && is_contiguous && !is_complex) {
    // The fused 3-source kernel drops the bias for vector-shaped (M==1) inputs on the M1
    // (Apple7) family on macOS 26; add it separately there. Fixed in macOS 27.
    static const bool decompose_bias = is_apple_family_or_newer(AppleGPUFamily::APPLE_7_PLUS) &&
        !is_apple_family_or_newer(AppleGPUFamily::APPLE_8_PLUS) && is_macos_at_least(MacOSVersion::MACOS_26_0) &&
        !is_macos_at_least(MacOSVersion::MACOS_27_0);
    // linear's leading dims are a fake batch (weight is shared), so a >2D input is one 2D GEMM.
    // Passing it as a batched NDArray instead triggers a 2^16 batch-index wraparound (#189495) and
    // a small-batch GEMV perf cliff (#189847); flatten to 2D (a free view here) to avoid both.
    // A multi-dim bias cannot be flattened, so run the bias-free kernel there and add the bias afterwards.
    const bool needs_flatten = input.dim() > 2;
    const bool add_bias_after = is_bias_defined && (decompose_bias || (needs_flatten && bias.dim() > 1));
    const Tensor kernel_bias = add_bias_after ? Tensor() : bias;
    if (needs_flatten) {
      auto input2d = input.flatten(0, -2);
      auto output2d = output.flatten(0, -2);
      _mps_linear_nograph(input2d, weight, kernel_bias, output2d);
    } else {
      _mps_linear_nograph(input, weight, kernel_bias, output);
    }
    if (add_bias_after) {
      output.add_(bias);
    }
    // Squeeze last dim of 1D linear
    return weight_arg.dim() != 1 ? output : output.squeeze(-1);
  }
  MPSStream* stream = getCurrentMPSStream();
  struct CachedGraph : public MPSCachedGraph {
    CachedGraph(MPSGraph* graph) : MPSCachedGraph(graph) {}
    MPSGraphTensor* inputTensor_ = nil;
    MPSGraphTensor* weightTensor_ = nil;
    MPSGraphTensor* biasTensor_ = nil;
    MPSGraphTensor* outputTensor_ = nil;
  };

  // Flatten the input's batch dims to 2D on the tensor side rather than with an in-graph
  // flatten2DTensor. MPSGraph's canonicalizer fuses a flatten2D -> matmul chain and
  // miscomputes the output shape, aborting during MLIR compilation for complex inputs on
  // macOS 27 (see agent_space/mm_complex_broadcast_crash.swift). A pre-flattened 2D input
  // keeps the fused pattern from forming; the output reshape stays in-graph (it does not
  // trigger the bug and lets the Placeholder honor non-contiguous output strides). This
  // also covers the original reasons for reshaping: the 5D matmul crash (#114942) and the
  // >2D fp16/bf16 non-determinism on Apple10+.
  bool needs_flatten = input.dim() > 4;
  if (!needs_flatten && is_bias_defined) {
    // improves performance with 3D+ inputs
    needs_flatten =
        input_size.size() > 2 && input_size[0] > 1 && input_size[1] >= 1 && input_size[1] <= 32 && bias.dim() <= 1;
  }
  if (!needs_flatten) {
    needs_flatten = needs_nd_workaround(input);
  }
  const Tensor linear_input = needs_flatten ? input.reshape({-1, input.size(-1)}) : input;

  @autoreleasepool {
    // Key on the original input: two inputs that flatten to the same 2D shape can still
    // need different output reshapes (e.g. (6,K) vs (2,3,K)), so they must not share a graph.
    std::string key = "mps_linear" + getTensorsStringKey({input, weight, bias});
    auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto* mpsGraph, auto* newCachedGraph) {
      MPSGraphTensor* inputTensor = mpsGraphRankedPlaceHolder(mpsGraph, linear_input);
      MPSGraphTensor* weightTensor = mpsGraphRankedPlaceHolder(mpsGraph, weight);

      MPSGraphTensor* weightTransposeTensor = [mpsGraph transposeTensor:weightTensor
                                                              dimension:-1
                                                          withDimension:-2
                                                                   name:nil];
      auto outputTensor = [mpsGraph matrixMultiplicationWithPrimaryTensor:inputTensor
                                                          secondaryTensor:weightTransposeTensor
                                                                     name:nil];

      if (is_bias_defined) {
        newCachedGraph->biasTensor_ = mpsGraphRankedPlaceHolder(mpsGraph, bias);
        outputTensor = [mpsGraph additionWithPrimaryTensor:outputTensor
                                           secondaryTensor:newCachedGraph->biasTensor_
                                                      name:nil];
      }
      if (needs_flatten) {
        outputTensor = [mpsGraph reshapeTensor:outputTensor withShape:getMPSShape(output_size) name:nil];
      }

      newCachedGraph->inputTensor_ = inputTensor;
      newCachedGraph->weightTensor_ = weightTensor;
      newCachedGraph->outputTensor_ = outputTensor;
    });

    Placeholder inputPlaceholder = Placeholder(cachedGraph->inputTensor_, linear_input);
    Placeholder weightPlaceholder = Placeholder(cachedGraph->weightTensor_, weight);
    Placeholder biasPlaceholder = Placeholder();
    Placeholder outputPlaceholder = Placeholder(cachedGraph->outputTensor_, output);

    NSMutableDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds = [NSMutableDictionary dictionary];
    feeds[inputPlaceholder.getMPSGraphTensor()] = inputPlaceholder.getMPSGraphTensorData();
    feeds[weightPlaceholder.getMPSGraphTensor()] = weightPlaceholder.getMPSGraphTensorData();
    if (is_bias_defined) {
      biasPlaceholder = Placeholder(cachedGraph->biasTensor_, bias);
      feeds[biasPlaceholder.getMPSGraphTensor()] = biasPlaceholder.getMPSGraphTensorData();
    }
    runMPSGraph(stream, cachedGraph->graph(), feeds, outputPlaceholder);
  }

  // Squeeze last dim of 1D linear
  return weight_arg.dim() != 1 ? output : output.squeeze(-1);
}

static Tensor _mps_linear_backward_input(IntArrayRef input_size, const Tensor& grad_output, const Tensor& weight) {
  TORCH_CHECK(grad_output.is_mps(), "mps_linear_backward: grad_output needs to be mps layout");
  TORCH_CHECK(weight.device().is_mps() && supportedFloatingOrComplexType(weight),
              "mps_linear_backward: unsupported weights data type: ",
              weight.scalar_type());
  TORCH_CHECK(supportedFloatingOrComplexType(grad_output),
              "MPS device does not support linear backward for non-float inputs");

  // An empty grad_output (out_features == 0) zeroes the grad-input; a zero-length
  // input_size dim (in_features == 0) makes it empty. Neither can go through mm.
  if (grad_output.numel() == 0 || c10::multiply_integers(input_size) == 0) {
    return at::zeros(input_size, grad_output.options());
  }

  const auto weight_contig = weight.is_contiguous() ? weight : weight.contiguous();
  const auto grad_output_2d = grad_output.dim() != 2 ? grad_output.reshape({-1, grad_output.size(-1)}) : grad_output;
  return at::mm(grad_output_2d, weight_contig).view(input_size);
}

static std::tuple<Tensor, Tensor> _mps_linear_backward_weights(const Tensor& grad_output,
                                                               const Tensor& input,
                                                               const Tensor& weight,
                                                               bool bias_defined) {
  TORCH_CHECK(grad_output.is_mps() && input.is_mps(),
              "_mps_linear_backward: grad_output and input needs to be mps layout");

  TORCH_CHECK(supportedFloatingOrComplexType(grad_output),
              "MPS device does not support linear backward for non-float inputs");

  // Guard before the reshapes below: for a 0-element input, reshape({-1, 0}) is
  // ambiguous and throws. The weight gradient is empty or zero here, but the bias
  // gradient is still the sum of grad_output over the leading dims.
  if (grad_output.numel() == 0 || input.numel() == 0) {
    auto grad_weight = at::zeros({grad_output.size(-1), input.size(-1)}, grad_output.options());
    Tensor grad_bias;
    if (bias_defined) {
      grad_bias = at::zeros({grad_output.size(-1)}, grad_output.options());
      if (grad_output.numel() != 0) {
        const auto grad_output_flat =
            grad_output.dim() != 2 ? grad_output.reshape({-1, grad_output.size(-1)}) : grad_output;
        grad_bias.copy_(grad_output_flat.sum(0));
      }
    }
    return {grad_weight, grad_bias};
  }

  const auto grad_output_2d = grad_output.dim() != 2 ? grad_output.reshape({-1, grad_output.size(-1)}) : grad_output;
  const auto input_2d = input.dim() != 2 ? input.reshape({-1, input.size(-1)}) : input;

  // Route through at::mm so the dispatcher can pick the Metal fallback for K-dim
  // overflow on Apple7/8 (M1/M2). See pytorch/pytorch#177116.
  auto grad_weight = at::mm(grad_output_2d.t(), input_2d.contiguous());
  // autocast promotes sum() to float32, but linear_backward's meta keeps grad_output's
  // dtype; cast back so inductor's baked-in dtype matches the runtime buffer.
  auto grad_bias = bias_defined ? grad_output_2d.sum(0).to(grad_output.scalar_type()) : Tensor();
  return {grad_weight, grad_bias};
}

std::tuple<Tensor, Tensor, Tensor> mps_linear_backward(const Tensor& input,
                                                       const Tensor& grad_output,
                                                       const Tensor& weight,
                                                       std::array<bool, 3> output_mask) {
  Tensor grad_input, grad_weight, grad_bias;
  if (output_mask[0]) {
    grad_input = _mps_linear_backward_input(input.sizes(), grad_output, weight);
  }
  if (output_mask[1] || output_mask[2]) {
    std::tie(grad_weight, grad_bias) = _mps_linear_backward_weights(grad_output, input, weight, output_mask[2]);
  }
  return std::tuple<Tensor, Tensor, Tensor>{grad_input, grad_weight, grad_bias};
}

} // namespace at::native
