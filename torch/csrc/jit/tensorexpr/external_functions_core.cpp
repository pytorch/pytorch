#include <torch/csrc/jit/tensorexpr/external_functions_core.h>

namespace torch::jit::tensorexpr {

#ifdef C10_MOBILE
extern "C" {
#endif

using ParallelCallee = void (*)(int64_t, int8_t*);
void DispatchParallel(
    int8_t* func,
    int64_t start,
    int64_t stop,
    int8_t* packed_data) noexcept {
  // TODO: preserve the func type.
  try {
    ParallelCallee callee = reinterpret_cast<ParallelCallee>(func);
    at::parallel_for(start, stop, 1, [&](int64_t f_begin, int64_t f_end) {
      for (int64_t index = f_begin; index < f_end; index++) {
        callee(index, packed_data);
      }
    });
  } catch (...) {
  }
}

void nnc_aten_free(size_t bufs_num, void** ptrs) noexcept {
  for (const auto i : c10::irange(bufs_num)) {
    c10::raw::intrusive_ptr::decref((c10::TensorImpl*)ptrs[i]);
  }
}

#ifdef C10_MOBILE
} // extern "C"
#endif

} // namespace torch::jit::tensorexpr
