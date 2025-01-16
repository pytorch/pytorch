#include <ittnotify.h>
#include <torch/csrc/itt_wrapper.h>
#include <torch/csrc/profiler/stubs/base.h>

namespace torch::profiler {
__itt_domain* _itt_domain = __itt_domain_create("PyTorch");

bool itt_is_available() {
  return torch::profiler::impl::ittStubs()->enabled();
}

void itt_range_push(const char* msg) {
  __itt_string_handle* hsMsg = __itt_string_handle_create(msg);
  __itt_task_begin(_itt_domain, __itt_null, __itt_null, hsMsg);
}

void itt_range_pop() {
  __itt_task_end(_itt_domain);
}

void itt_mark(const char* msg) {
  __itt_string_handle* hsMsg = __itt_string_handle_create(msg);
  __itt_task_begin(_itt_domain, __itt_null, __itt_null, hsMsg);
  __itt_task_end(_itt_domain);
}
} // namespace torch::profiler
