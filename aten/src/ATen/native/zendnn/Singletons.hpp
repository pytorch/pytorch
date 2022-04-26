#pragma once

#include <ATen/native/zendnn/AbstractTypes.hpp>
#include <ATen/native/zendnn/Tensor.hpp>

namespace zendnn {
namespace utils {

engine& engine::cpu_engine() {
  static engine cpu_engine(kind::cpu, 0);
  return cpu_engine;
}

struct RegisterEngineAllocator {
  RegisterEngineAllocator(
      engine& eng,
      const std::function<void*(size_t)>& malloc,
      const std::function<void(void*)>& free) {
    eng.set_allocator(malloc, free);
  }
};

} // namespace utils
} // namespace zendnn
