#include <cpuinfo.h>
#include <torch/extension.h>
#include <torch/script.h>

using torch::Tensor;

Tensor consume(Tensor a) {
  return a;
}

Tensor clear_cache() {
  static uint32_t* wipe_buffer = nullptr;
  static size_t wipe_size = 0;

  if (wipe_buffer == nullptr) {
    TORCH_CHECK(cpuinfo_initialize(), "failed to initialize cpuinfo");
    const cpuinfo_processor* processor = cpuinfo_get_processor(0);
    wipe_size = processor->cache.l3->size;
    wipe_buffer = static_cast<uint32_t*>(malloc(wipe_size));
    TORCH_CHECK(wipe_buffer != nullptr);
  }
  int64_t hash = 0;
  for (uint32_t i = 0; i * sizeof(uint32_t) < wipe_size; i += 8) {
    hash ^= wipe_buffer[i];
    wipe_buffer[i] = hash;
  }
  /* Make sure compiler doesn't optimize the loop away */
  Tensor ret = torch::from_blob(&hash, {1,});
  return ret;
}

// When JIT tracing is used on function with constant for loop,
// the for loop is optimized away because of dead code elimination.
// That caused an issue for our op benchmark which needs to run an op
// in a loop and report the execution time. This diff resolves that issue by
// registering this consume op with correct alias information which is DEFAULT.
auto reg = torch::jit::RegisterOperators()
  .op("operator_benchmark::_consume", &consume)
  .op("operator_benchmark::_clear_cache", &clear_cache);

PYBIND11_MODULE(cpp_extension, m) {
  m.def("_consume", &consume, "consume");
  m.def("_clear_cache", &clear_cache, "clear_cache");
}
