//  Copyright © 2022 Apple Inc.
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/OpMathType.h>
#include <ATen/ceil_div.h>
#include <ATen/native/LinearAlgebraUtils.h>
#include <ATen/native/mps/OperationUtils.h>
#include <ATen/native/mps/kernels/ReduceOps.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/addmv_native.h>
#include <ATen/ops/dot_native.h>
#include <ATen/ops/mm.h>
#include <ATen/ops/vdot_native.h>
#endif

#ifdef __OBJC__
#include <MetalPerformanceShaders/MetalPerformanceShaders.h>
#endif

namespace at::native {

namespace mps {

#ifndef PYTORCH_JIT_COMPILE_SHADERS
static auto& lib = MetalShaderLibrary::getBundledLibrary();
#else
#include <ATen/native/mps/ReduceOps_metallib.h>
#endif

inline void dot_check(const Tensor& self, const Tensor& other) {
  TORCH_CHECK(self.dim() == 1 && other.dim() == 1,
              "1D tensors expected, but got ",
              self.dim(),
              "D and ",
              other.dim(),
              "D tensors");
  TORCH_CHECK(self.scalar_type() == other.scalar_type(),
              "dot : expected both vectors to have same dtype, but found ",
              self.scalar_type(),
              " and ",
              other.scalar_type());
  TORCH_CHECK(self.numel() == other.numel(),
              "inconsistent tensor size, expected tensor [",
              self.numel(),
              "] and src [",
              other.numel(),
              "] to have the same number of elements, but got ",
              self.numel(),
              " and ",
              other.numel(),
              " elements respectively");
  TORCH_CHECK(self.device() == other.device(),
              "Expected all tensors to be on the same device. Found: ",
              self.device(),
              ", ",
              other.device());
}
} // namespace mps

Tensor dot_mps(const Tensor& self, const Tensor& other) {
  using namespace mps;

  const auto dtype = self.scalar_type();
  TORCH_CHECK_NOT_IMPLEMENTED(supportedFloatingType(dtype) || dtype == kComplexFloat ||
                                  (isIntegralType(dtype, false) && !isBarebonesUnsignedType(dtype)),
                              "\"dot\" not implemented for '",
                              dtype,
                              "'");

  if (self.numel() == 0 & other.numel() == 0) {
    return zeros({}, self.options());
  }

  dot_check(self, other);

  using namespace std::string_view_literals;
  const auto acc_type = isIntegralType(dtype, false) ? (dtype == kLong ? kLong : kInt) : toOpMathType(dtype);
  const auto numel = self.numel();
  const auto num_partials =
      std::min<int64_t>(DOT_MAX_TGS, at::ceil_div<int64_t>(numel, MAX_THREADGROUP_SIZE * SUM_NCHAINS));
  const DotParams<> params{.numel = numel,
                           .num_tgs = num_partials,
                           .a_stride = self.stride(0),
                           .b_stride = other.stride(0),
                           .a_conj = self.is_conj(),
                           .b_conj = other.is_conj()};
  const auto partials = at::empty({num_partials}, self.options().dtype(acc_type));
  const auto layout = self.stride(0) == 1 && other.stride(0) == 1 ? "dense"sv : "strided"sv;
  const auto kernel = fmt::format("dot_reduce_{}_{}", layout, scalarToMetalTypeString(dtype));
  auto stream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      auto pso = lib.getPipelineStateForFunc(kernel);
      auto encoder = stream->commandEncoder();
      [encoder setComputePipelineState:pso];
      mtl_setArgs(encoder, self, other, partials, params);
      mtl_dispatch1DJob(encoder, pso, num_partials * [pso maxTotalThreadsPerThreadgroup]);
    }
  });
  return (num_partials == 1 ? partials.squeeze(0) : partials.sum(acc_type)).to(dtype);
}

Tensor vdot_mps(const Tensor& self, const Tensor& other) {
  // For real dtypes, vdot is identical to dot
  if (!self.is_complex()) {
    return dot_mps(self, other);
  }

  return dot_mps(self.conj(), other);
}

static Tensor& addmv_out_mps_impl(const Tensor& self,
                                  const Tensor& mat,
                                  const Tensor& vec,
                                  const Scalar& beta_,
                                  const Scalar& alpha_,
                                  Tensor& result) {
  using namespace mps;

  TORCH_CHECK(mat.is_mps());
  TORCH_CHECK(vec.is_mps());
  TORCH_CHECK(result.is_mps());
  TORCH_CHECK(self.is_mps());

  c10::MaybeOwned<Tensor> self_ = expand_size(self, {mat.size(0)});
  auto betaval = beta_.toComplexDouble();

  struct CachedGraph : public MPSCachedGraph {
    CachedGraph(MPSGraph* graph) : MPSCachedGraph(graph) {}
    MPSGraphTensor* selfTensor_ = nil;
    MPSGraphTensor* matMulVecTensor_ = nil;
    MPSGraphTensor* outputTensor_ = nil;
  };

  MPSStream* stream = at::mps::getCurrentMPSStream();
  if (result.numel() == 0) {
    return result;
  }
  Tensor matMulVec = at::mm(mat, vec.unsqueeze(1)).squeeze(1);

  @autoreleasepool {
    std::string key = "addmv_out_mps_impl" + getTensorsStringKey({self, matMulVec}) + ":" +
        std::to_string(beta_.toDouble()) + ":" + std::to_string(alpha_.toDouble());
    auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      MPSGraphTensor* matMulVecTensor = mpsGraphRankedPlaceHolder(mpsGraph, matMulVec);
      MPSGraphTensor* selfTensor = mpsGraphRankedPlaceHolder(mpsGraph, self);

      // Intermediates for beta and alpha
      MPSGraphTensor* alphaTensor = [mpsGraph constantWithScalar:alpha_.toDouble()
                                                        dataType:getMPSScalarType(mat.scalar_type())];

      // Intermediates for multiplying by beta and alpha
      MPSGraphTensor* productTimesAlphaTensor = [mpsGraph multiplicationWithPrimaryTensor:matMulVecTensor
                                                                          secondaryTensor:alphaTensor
                                                                                     name:@"MM/alpha*(mat@vec)"];
      newCachedGraph->outputTensor_ = productTimesAlphaTensor;

      if (betaval != 0.0) {
        MPSGraphTensor* betaTensor = [mpsGraph constantWithScalar:beta_.toDouble()
                                                         dataType:getMPSScalarType(self.scalar_type())];

        MPSGraphTensor* selfTimesBetaTensor = [mpsGraph multiplicationWithPrimaryTensor:selfTensor
                                                                        secondaryTensor:betaTensor
                                                                                   name:@"MM/beta*input"];

        MPSGraphTensor* outputTensor = [mpsGraph additionWithPrimaryTensor:productTimesAlphaTensor
                                                           secondaryTensor:selfTimesBetaTensor
                                                                      name:@"MM/beta*input + alpha*(mat@vec)"];

        newCachedGraph->outputTensor_ = outputTensor;
      }

      newCachedGraph->selfTensor_ = selfTensor;
      newCachedGraph->matMulVecTensor_ = matMulVecTensor;
    });

    Placeholder matMulVecPlaceholder = Placeholder(cachedGraph->matMulVecTensor_, matMulVec);
    Placeholder outputPlaceholder = Placeholder(cachedGraph->outputTensor_, result);

    NSMutableDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds = [NSMutableDictionary dictionary];
    feeds[matMulVecPlaceholder.getMPSGraphTensor()] = matMulVecPlaceholder.getMPSGraphTensorData();
    if (betaval != 0.0) {
      Placeholder selfPlaceholder = Placeholder(cachedGraph->selfTensor_, self);
      feeds[selfPlaceholder.getMPSGraphTensor()] = selfPlaceholder.getMPSGraphTensorData();
    }

    runMPSGraph(stream, cachedGraph->graph(), feeds, outputPlaceholder);
  }

  return result;
}

TORCH_IMPL_FUNC(addmv_out_mps)
(const Tensor& self,
 const Tensor& mat,
 const Tensor& vec,
 const Scalar& beta_,
 const Scalar& alpha_,
 const Tensor& result) {
  addmv_out_mps_impl(self, mat, vec, beta_, alpha_, const_cast<Tensor&>(result));
}

} // namespace at::native
