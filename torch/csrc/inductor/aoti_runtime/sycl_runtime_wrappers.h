// NOLINT
#pragma once

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

static ze_module_handle_t create_module(
    ze_context_handle_t context,
    ze_device_handle_t device,
    const uint8_t* binary_ptr,
    size_t binary_size) {
  const char* build_flags = "";
  const ze_module_format_t format = ZE_MODULE_FORMAT_IL_SPIRV;
  ze_module_desc_t module_description = {};
  module_description.stype = ZE_STRUCTURE_TYPE_MODULE_DESC;
  module_description.format = format;
  module_description.inputSize = binary_size;
  module_description.pInputModule = (uint8_t*)binary_ptr;
  module_description.pBuildFlags = build_flags;
  ze_module_build_log_handle_t buildlog;
  ze_module_handle_t module;
  auto context_initial = context;
  auto device_initial = device;
  auto error_no = ZE_RESULT_SUCCESS;
  error_no =
      zeModuleCreate(context, device, &module_description, &module, &buildlog);

  if (error_no != ZE_RESULT_SUCCESS) {
    size_t szLog = 0;
    ZE_CHECK(zeModuleBuildLogGetString(buildlog, &szLog, nullptr));
    char* strLog = (char*)malloc(szLog);
    ZE_CHECK(zeModuleBuildLogGetString(buildlog, &szLog, strLog));
    std::cerr << "L0 build module failed. Log: " << strLog << std::endl;
    free(strLog);
    ZE_CHECK(zeModuleBuildLogDestroy(buildlog));
  }
  ZE_CHECK(error_no);
  return module;
}

ze_kernel_handle_t create_function(
    ze_module_handle_t module,
    ze_kernel_flags_t flag,
    std::string func_name) {
  ze_kernel_handle_t kernel;
  ze_kernel_desc_t kernel_description = {};
  kernel_description.stype = ZE_STRUCTURE_TYPE_KERNEL_DESC;
  kernel_description.pNext = nullptr;
  kernel_description.flags = flag;
  kernel_description.pKernelName = func_name.c_str();
  assert(module);
  ZE_CHECK(zeKernelCreate(module, &kernel_description, &kernel));
  return kernel;
}

static ze_module_handle_t loadModule(
    sycl::queue& queue,
    std::string& spv_path) {
  auto l0_device = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(
      queue.get_device());
  auto l0_context = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(
      queue.get_context());
  std::ifstream IFS(spv_path.c_str(), std::ios::binary);
  std::ostringstream OSS;
  OSS << IFS.rdbuf();
  std::string data(OSS.str());

  return create_module(
      l0_context,
      l0_device,
      reinterpret_cast<const uint8_t*>(data.c_str()),
      data.size());
}

static sycl::kernel* getKernel(
    sycl::queue& queue,
    ze_module_handle_t l0_module,
    const char* kernel_name) {
  assert(l0_module);
  assert(kernel_name);
  auto l0_kernel =
      create_function(l0_module, ZE_KERNEL_FLAG_FORCE_RESIDENCY, kernel_name);
  auto sycl_device = queue.get_device();
  auto ctx = sycl_device.get_platform().ext_oneapi_get_default_context();

  auto mod = sycl::make_kernel_bundle<
      sycl::backend::ext_oneapi_level_zero,
      sycl::bundle_state::executable>(
      {l0_module, sycl::ext::oneapi::level_zero::ownership::transfer}, ctx);

  auto fun = sycl::make_kernel<sycl::backend::ext_oneapi_level_zero>(
      {mod, l0_kernel, sycl::ext::oneapi::level_zero::ownership::transfer},
      ctx);
  auto kernel_ptr = new sycl::kernel(fun);
  return kernel_ptr;
}

static sycl::kernel* loadKernel(
    sycl::queue& queue,
    std::string filePath,
    const std::string& funcName,
    uint32_t sharedMemBytes,
    const std::optional<std::string>& binDir = std::nullopt) {
  if (binDir) {
    std::filesystem::path p1{*binDir};
    std::filesystem::path p2{filePath};
    filePath = (p1 / p2.filename()).string();
  }
  auto mod = loadModule(queue, filePath);
  return getKernel(queue, mod, funcName.c_str());
}

static void launchKernel(
    sycl::kernel* kernel_ptr,
    uint32_t gridX,
    uint32_t gridY,
    uint32_t gridZ,
    int num_warps,
    int shared_memory,
    std::vector<void*>& params,
    sycl::queue& stream) {
  std::string kernel_name =
      kernel_ptr->get_info<sycl::info::kernel::function_name>();
  uint32_t num_params = params.size();
  int threads_per_warp = 32;
  uint32_t expected_num_params =
      kernel_ptr->get_info<sycl::info::kernel::num_args>();
  size_t global_range_x = gridX * threads_per_warp * num_warps;
  size_t global_range_y = gridY;
  size_t global_range_z = gridZ;
  size_t local_range_x = num_warps * threads_per_warp;
  size_t local_range_y = 1;
  size_t local_range_z = 1;
  sycl::range<3> global_range(global_range_z, global_range_y, global_range_x);
  sycl::range<3> local_range(local_range_z, local_range_y, local_range_x);
  sycl::nd_range<3> parallel_work_size(global_range, local_range);
  if (shared_memory) {
    {
      expected_num_params -= 1;
    }
  }
  assert(
      num_params == expected_num_params &&
      "number of kernel param not matched");
  // Submit the imported kernel.
  auto cgf = [&](sycl::handler& cgh) {
    for (uint32_t i = 0; i < num_params; ++i) {
      cgh.set_arg(i, *(static_cast<void**>(params[i])));
    }

    if (shared_memory) {
      using share_mem_t = sycl::local_accessor<int8_t, 1>;
      share_mem_t local_buffer = share_mem_t(shared_memory, cgh);
      cgh.set_arg(num_params, local_buffer);
      cgh.parallel_for(parallel_work_size, *kernel_ptr);
    } else {
      cgh.parallel_for(parallel_work_size, *kernel_ptr);
    }
  };
  auto event = stream.submit(cgf);
}
