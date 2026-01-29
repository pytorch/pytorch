#include <gtest/gtest.h>

#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/PeerToPeerAccess.h>
#include <c10/cuda/driver_api.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cstdlib>
#include <unistd.h>

// Compatibility definitions for CUDA versions < 12.3
#if CUDA_VERSION < 12030
#define CU_MEM_HANDLE_TYPE_FABRIC ((CUmemAllocationHandleType)0x8ULL)
#define CU_IPC_HANDLE_SIZE 64
typedef struct CUmemFabricHandle_st {
  unsigned char data[CU_IPC_HANDLE_SIZE];
} CUmemFabricHandle_v1;
typedef CUmemFabricHandle_v1 CUmemFabricHandle;
#define CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED \
  ((CUdevice_attribute)128)
#endif

#if !defined(USE_ROCM)

using namespace c10::cuda::CUDACachingAllocator;

class ExpandableSegmentsIPCTest : public ::testing::Test {
 protected:
  void SetUp() override {
    setenv("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True", 1);
    setenv("TORCH_CUDA_EXPANDABLE_SEGMENTS_IPC", "1", 1);

    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
      GTEST_SKIP() << "No CUDA devices available";
    }
    cudaSetDevice(0);

    c10::cuda::CUDACachingAllocator::init(deviceCount);
    c10::cuda::detail::init_p2p_access_cache(deviceCount);
  }
};

// Test that expandable segments allocations have valid IPC handle types
// (either FABRIC or POSIX_FD, but never NONE)
TEST_F(ExpandableSegmentsIPCTest, AllocationHasValidIPCHandleType) {
  auto allocator = get();
  if (allocator == nullptr) {
    GTEST_SKIP() << "CUDACachingAllocator not available";
  }

  ASSERT_EQ(CUDAAllocatorConfig::expandable_segments(), true);

  auto* driver = c10::cuda::DriverAPI::get();
  CUdevice device;
  CUresult deviceResult = driver->cuDeviceGet_(&device, 0);
  ASSERT_EQ(deviceResult, CUDA_SUCCESS) << "Failed to get CUDA device";

  int fabricSupported = 0;
#if CUDA_VERSION < 12030
  fabricSupported = 0;
#else
  CUresult attrResult = driver->cuDeviceGetAttribute_(
      &fabricSupported,
      CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED,
      device);
  // CUDA_ERROR_INVALID_VALUE means the driver doesn't recognize this attribute
  // (driver older than CUDA 12.3), so fabric is not supported
  if (attrResult == CUDA_ERROR_INVALID_VALUE) {
    fabricSupported = 0;
  } else {
    ASSERT_EQ(attrResult, CUDA_SUCCESS)
        << "Failed to query FABRIC support attribute";
  }
#endif

  CUmemAllocationHandleType expectedHandleType;
#if CUDA_VERSION > 12040
  if (fabricSupported) {
    expectedHandleType = CU_MEM_HANDLE_TYPE_FABRIC;
  } else {
    expectedHandleType = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
  }
#else
  expectedHandleType = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
#endif

  constexpr size_t allocSize = 1024 * 1024;  // 1MB
  void* ptr = allocator->raw_alloc(allocSize);
  ASSERT_NE(ptr, nullptr) << "Failed to allocate memory";

  CUmemGenericAllocationHandle handle;
  CUresult retainResult = driver->cuMemRetainAllocationHandle_(&handle, ptr);
  ASSERT_EQ(retainResult, CUDA_SUCCESS);

  CUmemAllocationProp prop = {};
  CUresult propResult =
      driver->cuMemGetAllocationPropertiesFromHandle_(&prop, handle);
  ASSERT_EQ(propResult, CUDA_SUCCESS)
      << "Failed to get allocation properties";

  EXPECT_EQ(prop.requestedHandleTypes, expectedHandleType)
      << "Handle type mismatch. Expected: " << expectedHandleType
      << " (POSIX_FILE_DESCRIPTOR=1, FABRIC=8), got: "
      << prop.requestedHandleTypes
      << ". CUDA_VERSION=" << CUDA_VERSION
      << ", fabricSupported=" << fabricSupported;

  allocator->raw_delete(ptr);
}

// Test that allocation can be exported to shareable handle
TEST_F(ExpandableSegmentsIPCTest, AllocationCanBeExported) {
  auto allocator = get();
  if (allocator == nullptr) {
    GTEST_SKIP() << "CUDACachingAllocator not available";
  }

  constexpr size_t allocSize = 1024 * 1024;  // 1MB

  void* ptr = allocator->raw_alloc(allocSize);
  ASSERT_NE(ptr, nullptr) << "Failed to allocate memory";

  auto* driver = c10::cuda::DriverAPI::get();
  CUmemGenericAllocationHandle handle;
  CUresult retainResult = driver->cuMemRetainAllocationHandle_(&handle, ptr);

  if (retainResult == CUDA_SUCCESS) {
    CUmemAllocationProp prop = {};
    CUresult propResult =
        driver->cuMemGetAllocationPropertiesFromHandle_(&prop, handle);
    ASSERT_EQ(propResult, CUDA_SUCCESS);

    if (prop.requestedHandleTypes != 0) {
      CUmemAllocationHandleType exportType =
          (prop.requestedHandleTypes & CU_MEM_HANDLE_TYPE_FABRIC)
              ? CU_MEM_HANDLE_TYPE_FABRIC
              : CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;

      // Try to export - this should succeed with the fix
      if (exportType == CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR) {
        int fd = -1;
        CUresult exportResult = driver->cuMemExportToShareableHandle_(
            &fd, handle, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0);
        EXPECT_EQ(exportResult, CUDA_SUCCESS)
            << "Failed to export to POSIX_FILE_DESCRIPTOR handle";
        if (exportResult == CUDA_SUCCESS && fd >= 0) {
          close(fd);
        }
      } else {
        CUmemFabricHandle fabricHandle = {};
        CUresult exportResult = driver->cuMemExportToShareableHandle_(
            &fabricHandle, handle, CU_MEM_HANDLE_TYPE_FABRIC, 0);
        EXPECT_EQ(exportResult, CUDA_SUCCESS)
            << "Failed to export to FABRIC handle";
      }
    }
  }

  allocator->raw_delete(ptr);
}

#endif // !defined(USE_ROCM)
