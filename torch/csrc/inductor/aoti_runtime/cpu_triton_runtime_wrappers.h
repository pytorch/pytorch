// CPU AOTI Triton runtime helpers. CPU counterpart of the GPU AOTI Triton path
// (cf. `sycl_runtime_wrappers.h`, `static_launcher/cuda.cpp`): the generated
// wrapper dlopens the per-kernel `.so` and a launcher `.so` exporting
// `run_from_nativert`, then invokes the kernel through the launcher.
//
// Note: only Triton CPU builds that emit a NativeRT-compatible launcher are
// supported. OSS triton-cpu does not yet, and `loadCpuTritonLauncher` aborts
// at model load time -- expected and intentional.

#pragma once

#ifdef _WIN32
#error "CPU AOTI Triton runtime helpers are not supported on Windows"
#endif

#include <c10/util/Exception.h>
#include <dlfcn.h>
#include <filesystem>
#include <optional>
#include <string>

namespace torch::aot_inductor {

// Launcher ABI: (gridX, gridY, gridZ, num_cpu_threads, void** args, kernel_fn).
using CpuTritonLauncherFn =
    void (*)(uint32_t, uint32_t, uint32_t, int, void**, void*);

// Resolve `filePath` to an actual `.so` location: `soDir` override (GPU's
// `binDir` / `cubin_dir_` analog), else the absolute path, else the basename
// next to the AOTI .so via `dladdr`.
[[maybe_unused]] static std::string _resolve_cpu_triton_so_path(
    std::string filePath,
    const std::optional<std::string>& soDir) {
  if (soDir) {
    std::filesystem::path p1{*soDir};
    std::filesystem::path p2{filePath};
    return (p1 / p2.filename()).string();
  }
  std::error_code ec;
  if (std::filesystem::exists(filePath, ec)) {
    return filePath;
  }
  Dl_info info{};
  if (dladdr(reinterpret_cast<void*>(&_resolve_cpu_triton_so_path), &info) &&
      info.dli_fname != nullptr) {
    std::filesystem::path basename = std::filesystem::path(filePath).filename();
    std::filesystem::path candidate =
        std::filesystem::path(info.dli_fname).parent_path() / basename;
    if (std::filesystem::exists(candidate, ec)) {
      return candidate.string();
    }
  }
  return filePath;
}

// Load a kernel symbol from `filePath`. The handle is intentionally leaked --
// the AOTI process keeps the .so loaded for its lifetime, mirroring how cubin
// modules are leaked on the GPU side.
[[maybe_unused]] static void* loadCpuTritonKernel(
    std::string filePath,
    const std::string& funcName,
    const std::optional<std::string>& soDir = std::nullopt) {
  filePath = _resolve_cpu_triton_so_path(std::move(filePath), soDir);
  void* handle = dlopen(filePath.c_str(), RTLD_NOW | RTLD_LOCAL);
  TORCH_CHECK(handle, "dlopen ", filePath, ": ", dlerror());
  void* fn = dlsym(handle, funcName.c_str());
  TORCH_CHECK(fn, "dlsym ", funcName, " from ", filePath, ": ", dlerror());
  return fn;
}

// Load the launcher's `run_from_nativert` entry point. Same handle-leak policy
// as `loadCpuTritonKernel`. Aborts loudly if the symbol is missing.
[[maybe_unused]] static CpuTritonLauncherFn loadCpuTritonLauncher(
    std::string filePath,
    const std::optional<std::string>& soDir = std::nullopt) {
  filePath = _resolve_cpu_triton_so_path(std::move(filePath), soDir);
  void* handle = dlopen(filePath.c_str(), RTLD_NOW | RTLD_LOCAL);
  TORCH_CHECK(handle, "dlopen ", filePath, ": ", dlerror());
  void* fn = dlsym(handle, "run_from_nativert");
  TORCH_CHECK(
      fn,
      "Triton CPU launcher .so does not export 'run_from_nativert' (",
      filePath,
      "): ",
      dlerror(),
      ". The CPU AOTI Triton path requires a Triton CPU build that emits a "
      "NativeRT-compatible launcher.");
  return reinterpret_cast<CpuTritonLauncherFn>(fn);
}

} // namespace torch::aot_inductor
