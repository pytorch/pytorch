#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/mps/MPSGraphVenturaOps.h>
#include <ATen/native/mps/OperationUtils.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/linalg_inv_ex.h>
#include <ATen/ops/linalg_inv_ex_native.h>
#endif

namespace at::native {

TORCH_IMPL_FUNC(linalg_inv_ex_out_mps)(const Tensor& A, bool check_errors, const Tensor& result, const Tensor& info) {
  TORCH_CHECK(result.is_mps(), "Output tensor is not MPS");
  if (!is_macos_13_or_newer(MacOSVersion::MACOS_VER_13_3_PLUS)) {
    TORCH_WARN_ONCE(
        "torch.linalg_inv_ex.inverse is supported by MPS on MacOS 13+, please upgrade. Falling back to CPU.");
    auto cpu_info = at::empty({0}, kInt, c10::nullopt, kCPU, c10::nullopt, c10::nullopt);
    auto cpu_result = result.clone().to("cpu");
    at::linalg_inv_ex_out(cpu_result, cpu_info, A.to("cpu"));
    info.copy_(cpu_info);
    result.copy_(cpu_result);
    return;
  }

  using namespace mps;
  using CachedGraph = MPSUnaryCachedGraph;

  MPSStream* stream = getCurrentMPSStream();
  info.zero_();

  if (A.numel() == 0) {
    return;
  }

  if (!result.is_contiguous()) {
    result.unsafeGetTensorImpl()->empty_tensor_restride(MemoryFormat::Contiguous);
  }

  @autoreleasepool {
    string key = "inv_out_mps" + getTensorsStringKey({A});
    auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      MPSGraphTensor* inputTensor = mpsGraphRankedPlaceHolder(mpsGraph, A);
      MPSGraphTensor* outputTensor = [mpsGraph inverseOfTensor:inputTensor name:nil];

      newCachedGraph->inputTensor_ = inputTensor;
      newCachedGraph->outputTensor_ = outputTensor;
    });

    Placeholder inputPlaceholder = Placeholder(cachedGraph->inputTensor_, A);
    Placeholder outputPlaceholder = Placeholder(cachedGraph->outputTensor_, result);

    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds =
        @{inputPlaceholder.getMPSGraphTensor() : inputPlaceholder.getMPSGraphTensorData()};

    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results =
        @{outputPlaceholder.getMPSGraphTensor() : outputPlaceholder.getMPSGraphTensorData()};

    runMPSGraph(stream, cachedGraph->graph(), feeds, results);
  }
}

} // namespace at::native
