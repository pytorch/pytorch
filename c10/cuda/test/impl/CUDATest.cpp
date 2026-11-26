#include <gtest/gtest.h>

#include <c10/core/Device.h>
#include <c10/core/DeviceType.h>
#include <c10/cuda/CUDAException.h>
#if !defined(USE_ROCM) && !defined(__HIP_PLATFORM_AMD__)
#include <c10/cuda/driver_api.h>
#endif
#include <c10/cuda/impl/CUDAGuardImpl.h>
#include <c10/cuda/impl/CUDATest.h>

using namespace c10::cuda::impl;

TEST(CUDATest, SmokeTest) {
  c10_cuda_test();
}

// Regression: the device-guard restore path is noexcept, but internally
// MaybeSetDevice() -> SetDevice() can throw -- via TORCH_CHECK(device >= 0) on
// the pre-CUDA-12 / HIP path (a pure host check), or via
// C10_CUDA_CHECK(cudaGetDevice) on a device carrying a sticky error. A leaked
// throw from a noexcept method calls std::terminate. A negative device index
// deterministically hits the TORCH_CHECK throw where MaybeSetDevice routes to
// SetDevice, so uncheckedSetDevice must swallow it (warn) instead of
// terminating -- if it regresses, this process aborts. On the CUDA-12 branch
// MaybeSetDevice short-circuits for an invalid index, so this is a benign
// no-op there.
TEST(CUDAGuardImplTest, UncheckedSetDeviceSwallowsErrorAndDoesNotTerminate) {
  CUDAGuardImpl impl;
  EXPECT_NO_THROW(
      impl.uncheckedSetDevice(c10::Device(c10::DeviceType::CUDA, -1)));
}

#if !defined(USE_ROCM) && !defined(__HIP_PLATFORM_AMD__) && \
    defined(CUDA_VERSION) && (CUDA_VERSION >= 12090)
TEST(CUDAErrorTest, IncludesDriverErrorLog) {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    cudaGetLastError();
    GTEST_SKIP() << "No CUDA device is available";
  }

  auto* log_api = c10::cuda::DriverAPI::get();
  if (!log_api->cuLogsRegisterCallback_) {
    GTEST_SKIP() << "The CUDA driver log APIs are unavailable";
  }

  int device = 0;
  ASSERT_EQ(cudaGetDevice(&device), cudaSuccess);

  cudaGraph_t graph = nullptr;
  ASSERT_EQ(cudaGraphCreate(&graph, 0), cudaSuccess);

  cudaMemAllocNodeParams params{};
  params.poolProps.allocType = cudaMemAllocationTypePinned;
  params.poolProps.location.type = cudaMemLocationTypeDevice;
  params.poolProps.location.id = device;
  params.bytesize = 4;
  cudaGraphNode_t node = nullptr;
  ASSERT_EQ(
      cudaGraphAddMemAllocNode(&node, graph, nullptr, 0, &params), cudaSuccess);

  bool threw = false;
  cudaGraphExec_t graph_exec = nullptr;
  try {
    C10_CUDA_CHECK(cudaGraphInstantiateWithFlags(
        &graph_exec, graph, cudaGraphInstantiateFlagDeviceLaunch));
  } catch (const c10::AcceleratorError& error) {
    threw = true;
    EXPECT_NE(
        std::string(error.what_without_backtrace())
            .find(
                "The CUDA driver logged these messages, which may provide useful details:"),
        std::string::npos);
    EXPECT_NE(
        std::string(error.what_without_backtrace())
            .find(
                "Device-launched graphs and conditional nodes do not support MEM_ALLOC nodes"),
        std::string::npos);
  }

  EXPECT_TRUE(threw);
  if (graph_exec) {
    EXPECT_EQ(cudaGraphExecDestroy(graph_exec), cudaSuccess);
  }
  EXPECT_EQ(cudaGraphDestroy(graph), cudaSuccess);
}
#endif
