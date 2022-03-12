#include <torch/csrc/jit/mobile/model_tracer/KernelDTypeTracer.h>
#include <map>
#include <mutex>
#include <set>
#include <string>

namespace torch {
namespace jit {
namespace mobile {
KernelDTypeTracer::KernelDTypeTracer() {
  auto recorder_cb =
      [](const at::RecordFunction& fn) -> std::unique_ptr<at::ObserverContext> {
    std::string name = fn.name();
    size_t dollar_pos = name.find_first_of('$');
    std::string kernel_tag = name.substr(0, dollar_pos);
    std::string dtype = name.substr(dollar_pos + 1);

    std::lock_guard<std::mutex> guard(getMutex());
    getCalledKernelTags()[kernel_tag].insert(dtype);
    return nullptr;
  };

  handle_ = at::addGlobalCallback(
      at::RecordFunctionCallback(recorder_cb)
          .scopes({at::RecordScope::KERNEL_FUNCTION_DTYPE}));
}

KernelDTypeTracer::kernel_tags_type& KernelDTypeTracer::getCalledKernelTags() {
  static kernel_tags_type called_kernel_tags;
  return called_kernel_tags;
}

std::mutex& KernelDTypeTracer::getMutex() {
  static std::mutex m;
  return m;
}

} // namespace mobile
} // namespace jit
} // namespace torch
