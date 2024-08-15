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

    getCalledKernelTags().withLock([&](kernel_tags_type& kernel_tags) {
      kernel_tags[kernel_tag].insert(dtype);
    });
    return nullptr;
  };

  handle_ = at::addGlobalCallback(
      at::RecordFunctionCallback(recorder_cb)
          .scopes({at::RecordScope::KERNEL_FUNCTION_DTYPE}));
}

c10::Synchronized<KernelDTypeTracer::kernel_tags_type>& KernelDTypeTracer::
    getCalledKernelTags() {
  static c10::Synchronized<kernel_tags_type> called_kernel_tags;
  return called_kernel_tags;
}

} // namespace mobile
} // namespace jit
} // namespace torch
