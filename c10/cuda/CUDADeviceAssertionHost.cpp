#include <c10/cuda/CUDADeviceAssertionHost.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/util/env.h>
#include <c10/util/irange.h>
#include <cuda_runtime.h>

#include <memory>
#include <string>

#define C10_CUDA_CHECK_WO_DSA(EXPR)                                 \
  do {                                                              \
    const cudaError_t __err = EXPR;                                 \
    c10::cuda::c10_cuda_check_implementation(                       \
        static_cast<int32_t>(__err),                                \
        __FILE__,                                                   \
        __func__, /* Line number data type not well-defined between \
                      compilers, so we perform an explicit cast */  \
        static_cast<uint32_t>(__LINE__),                            \
        false);                                                     \
  } while (0)

namespace c10::cuda {

namespace {


/// Get the number of CUDA devices
/// We need our own implementation of this function to prevent
/// an infinite initialization loop for CUDAKernelLaunchRegistry
int dsa_get_device_count() {
  int device_count = -1;
  C10_CUDA_CHECK_WO_DSA(c10::cuda::GetDeviceCount(&device_count));
  return device_count;
}

bool dsa_check_if_all_devices_support_managed_memory() {
// It looks as though this'll work best on CUDA GPUs with Pascal
// architectures or newer, per
// https://developer.nvidia.com/blog/unified-memory-cuda-beginners/
  return false;
}

bool env_flag_set(const char* env_var_name) {
  const auto env_flag = c10::utils::check_env(env_var_name);
  return env_flag.has_value() && env_flag.value();
}

/// Deleter for UVM/managed memory pointers
void uvm_deleter(DeviceAssertionsData* uvm_assertions_ptr) {
  // Ignore error in destructor
  if (uvm_assertions_ptr) {
    C10_CUDA_IGNORE_ERROR(cudaFree(uvm_assertions_ptr));
  }
}

} // namespace

CUDAKernelLaunchRegistry::CUDAKernelLaunchRegistry()
    : do_all_devices_support_managed_memory(
          dsa_check_if_all_devices_support_managed_memory()),
      gather_launch_stacktrace(check_env_for_enable_launch_stacktracing()),
      enabled_at_runtime(check_env_for_dsa_enabled()) {
  for ([[maybe_unused]] const auto _ : c10::irange(dsa_get_device_count())) {
    uvm_assertions.emplace_back(nullptr, uvm_deleter);
  }

  kernel_launches.resize(max_kernel_launches);
}

bool CUDAKernelLaunchRegistry::check_env_for_enable_launch_stacktracing()
    const {
  return env_flag_set("PYTORCH_CUDA_DSA_STACKTRACING");
}

bool CUDAKernelLaunchRegistry::check_env_for_dsa_enabled() const {
  return false;
}

uint32_t CUDAKernelLaunchRegistry::insert(
    const char* launch_filename [[maybe_unused]],
    const char* launch_function [[maybe_unused]],
    const uint32_t launch_linenum [[maybe_unused]],
    const char* kernel_name [[maybe_unused]],
    const int32_t stream_id [[maybe_unused]]) {
  return 0;
}

std::pair<std::vector<DeviceAssertionsData>, std::vector<CUDAKernelLaunchInfo>>
CUDAKernelLaunchRegistry::snapshot() const {
  // This is likely to be the longest-lasting hold on the mutex, but
  // we only expect it to be called in cases where we're already failing
  // and speed is no longer important
  const std::lock_guard<std::mutex> lock(read_write_mutex);

  std::vector<DeviceAssertionsData> device_assertions_data;
  for (const auto& x : uvm_assertions) {
    if (x) {
      device_assertions_data.push_back(*x);
    } else {
      device_assertions_data.emplace_back();
    }
  }

  return std::make_pair(device_assertions_data, kernel_launches);
}

DeviceAssertionsData* CUDAKernelLaunchRegistry::
    get_uvm_assertions_ptr_for_current_device() {
  return nullptr;
}

CUDAKernelLaunchRegistry& CUDAKernelLaunchRegistry::get_singleton_ref() {
  static CUDAKernelLaunchRegistry launch_registry;
  return launch_registry;
}

bool CUDAKernelLaunchRegistry::has_failed() const {
  for (const auto& x : uvm_assertions) {
    if (x && x->assertion_count > 0) {
      return true;
    }
  }
  return false;
}

} // namespace c10::cuda
