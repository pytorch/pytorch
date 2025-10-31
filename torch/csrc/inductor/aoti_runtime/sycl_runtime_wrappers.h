// NOLINT
#pragma once
#ifdef USE_XPU
#include <c10/xpu/XPUFunctions.h>
#include <level_zero/ze_api.h>
#include <sycl/sycl.hpp>
#include <fstream>
#include <iostream>
#include <string>

#define ZE_CHECK(status)                                                  \
  {                                                                       \
    if (status != ZE_RESULT_SUCCESS) {                                    \
      std::stringstream ss;                                               \
      ss << "L0 runtime error: " << std::hex << std::uppercase << status; \
      throw std::runtime_error(ss.str());                                 \
    }                                                                     \
  }

static ze_module_handle_t _createModule(
    const uint8_t* binaryPtr,
    size_t binarySize) {
  sycl::device& syclDevice =
      c10::xpu::get_raw_device(c10::xpu::current_device());
  auto& syclContext = c10::xpu::get_device_context();
  auto device =
      sycl::get_native<sycl::backend::ext_oneapi_level_zero>(syclDevice);
  auto context =
      sycl::get_native<sycl::backend::ext_oneapi_level_zero>(syclContext);

  const char* buildFlags = "";
  const ze_module_format_t format = ZE_MODULE_FORMAT_IL_SPIRV;
  ze_module_desc_t moduleDescription = {};
  moduleDescription.stype = ZE_STRUCTURE_TYPE_MODULE_DESC;
  moduleDescription.format = format;
  moduleDescription.inputSize = binarySize;
  moduleDescription.pInputModule = (uint8_t*)binaryPtr;
  moduleDescription.pBuildFlags = buildFlags;
  ze_module_build_log_handle_t buildLog = nullptr;
  ze_module_handle_t module = nullptr;
  auto error_no = ZE_RESULT_SUCCESS;
  error_no =
      zeModuleCreate(context, device, &moduleDescription, &module, &buildLog);

  if (error_no != ZE_RESULT_SUCCESS) {
    size_t szLog = 0;
    ZE_CHECK(zeModuleBuildLogGetString(buildLog, &szLog, nullptr));
    char* strLog = (char*)malloc(szLog);
    ZE_CHECK(zeModuleBuildLogGetString(buildLog, &szLog, strLog));
    std::cerr << "L0 build module failed. Log: " << strLog << std::endl;
    free(strLog);
  }
  if (buildLog) {
    ZE_CHECK(zeModuleBuildLogDestroy(buildLog));
  }
  ZE_CHECK(error_no);
  return module;
}

static std::unique_ptr<sycl::kernel> _createKernel(
    ze_module_handle_t module,
    const char* kernelName) {
  assert(module);
  assert(kernelName);
  ze_kernel_handle_t kernel = nullptr;
  ze_kernel_desc_t kernelDescription = {};
  kernelDescription.stype = ZE_STRUCTURE_TYPE_KERNEL_DESC;
  kernelDescription.pNext = nullptr;
  kernelDescription.flags = ZE_KERNEL_FLAG_FORCE_RESIDENCY;
  kernelDescription.pKernelName = kernelName;
  ZE_CHECK(zeKernelCreate(module, &kernelDescription, &kernel));

  auto& syclContext = c10::xpu::get_device_context();
  auto mod = sycl::make_kernel_bundle<
      sycl::backend::ext_oneapi_level_zero,
      sycl::bundle_state::executable>(
      {module, sycl::ext::oneapi::level_zero::ownership::transfer},
      syclContext);
  auto fun = sycl::make_kernel<sycl::backend::ext_oneapi_level_zero>(
      {mod, kernel, sycl::ext::oneapi::level_zero::ownership::transfer},
      syclContext);
  return std::make_unique<sycl::kernel>(fun);
}

// GPU Cpp Wrapper API
[[maybe_unused]] static std::unique_ptr<sycl::kernel> loadKernel(
    std::string filePath,
    const std::string& funcName,
    uint32_t sharedMemBytes,
    const std::optional<std::string>& binDir = std::nullopt) {
  if (binDir) {
    std::filesystem::path p1{*binDir};
    std::filesystem::path p2{filePath};
    filePath = (p1 / p2.filename()).string();
  }

  std::ifstream IFS(filePath.c_str(), std::ios::binary);
  std::ostringstream OSS;
  OSS << IFS.rdbuf();
  std::string data(OSS.str());

  auto mod = _createModule(
      reinterpret_cast<const uint8_t*>(data.c_str()), data.size());

  return _createKernel(mod, funcName.c_str());
}

// GPU Cpp Wrapper API
[[maybe_unused]] static std::unique_ptr<sycl::kernel> loadKernel(
    const void* start,
    const void* end,
    const std::string& funcName,
    uint32_t sharedMemBytes) {
  size_t size = reinterpret_cast<const uint8_t*>(end) -
      reinterpret_cast<const uint8_t*>(start);

  auto mod = _createModule(reinterpret_cast<const uint8_t*>(start), size);

  return _createKernel(mod, funcName.c_str());
}

// GPU Cpp Wrapper API
[[maybe_unused]] static void launchKernel(
    std::unique_ptr<sycl::kernel>& kernelPtr,
    uint32_t gridX,
    uint32_t gridY,
    uint32_t gridZ,
    uint32_t numWarps,
    uint32_t sharedMemory,
    void** params,
    sycl::queue* queuePtr,
    uint32_t threadsPerWarp) {
  std::string kernelName =
      kernelPtr->get_info<sycl::info::kernel::function_name>();
  uint32_t numParams = kernelPtr->get_info<sycl::info::kernel::num_args>();
  size_t globalRangeX = gridX * threadsPerWarp * numWarps;
  size_t globalRangeY = gridY;
  size_t globalRangeZ = gridZ;
  size_t localRangeX = numWarps * threadsPerWarp;
  size_t localRangeY = 1;
  size_t localRangeZ = 1;
  sycl::range<3> globalRange(globalRangeZ, globalRangeY, globalRangeX);
  sycl::range<3> localRange(localRangeZ, localRangeY, localRangeX);
  sycl::nd_range<3> parallelWorkSize(globalRange, localRange);
  if (sharedMemory) {
    // numParams from sycl info  = user provided args + sharedMemoryBuffer
    numParams -= 1;
  }
  // Submit the imported kernel.
  auto cgf = [&](sycl::handler& cgh) {
    for (uint32_t i = 0; i < numParams; ++i) {
      cgh.set_arg(i, *(static_cast<void**>(params[i])));
    }

    if (sharedMemory > 0) {
      constexpr int dimensions = 1;
      using share_mem_t = sycl::local_accessor<int8_t, dimensions>;
      share_mem_t localBuffer = share_mem_t(sharedMemory, cgh);
      cgh.set_arg(numParams, localBuffer);
      cgh.parallel_for(parallelWorkSize, *kernelPtr);
    } else {
      cgh.parallel_for(parallelWorkSize, *kernelPtr);
    }
  };
  auto event = queuePtr->submit(cgf);
}
#endif
