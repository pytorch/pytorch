#include <ATen/Context.h>
#include <ATen/detail/MTIAHooksInterface.h>
#include <nlohmann/json.hpp>
#include <torch/csrc/mtia/profiler/MTIAMemoryProfiler.h>

using json = nlohmann::json;

namespace torch::mtia {

void MTIAMemoryProfiler::start() {
  at::detail::getMTIAHooks().recordMemoryHistory("all", "all", 150000);
}

void MTIAMemoryProfiler::export_memory_history(const std::string& path) {
  at::detail::getMTIAHooks().memorySnapshot(path);
  return;
}

void MTIAMemoryProfiler::stop() {
  at::detail::getMTIAHooks().recordMemoryHistory(std::nullopt, "all", 0);
}

std::unique_ptr<torch::profiler::impl::python_tracer::PythonMemoryTracerBase>
getMemoryTracer() {
  return std::make_unique<MTIAMemoryProfiler>();
}

void initMemoryProfiler() {
  // MTIA hooks are linked into builds that also run on hosts without MTIA
  // devices, where this tracer would replace the CUDA one.
  if (at::globalContext()
          .getAcceleratorHooksInterface(at::kMTIA)
          .isAvailable()) {
    fprintf(stderr, "Initializing MTIA Memory Tracer\n");
    torch::profiler::impl::python_tracer::registerMemoryTracer(
        &getMemoryTracer);
  }
}
} // namespace torch::mtia
