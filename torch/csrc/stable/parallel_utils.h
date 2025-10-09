#pragma once

#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/headeronly/util/shim_utils.h>

#include <memory>

namespace torch::stable {

using DeleterFnPtr = void (*)(void*);

namespace {
inline void delete_thread_id_guard(void* ptr) {
  TORCH_ERROR_CODE_CHECK(aoti_torch_delete_thread_id_guard(
      reinterpret_cast<ThreadIdGuardHandle>(ptr)));
}

inline void delete_parallel_guard(void* ptr) {
  TORCH_ERROR_CODE_CHECK(aoti_torch_delete_parallel_guard(
      reinterpret_cast<ParallelGuardHandle>(ptr)));
}
} // namespace

class ThreadIdGuard {
 public:
  explicit ThreadIdGuard() = delete;
  explicit ThreadIdGuard(int32_t thread_id)
      : guard_(nullptr, delete_thread_id_guard) {
    ThreadIdGuardHandle ptr = nullptr;
    TORCH_ERROR_CODE_CHECK(aoti_torch_create_thread_id_guard(thread_id, &ptr));
    guard_.reset(ptr);
  }

 private:
  std::unique_ptr<ThreadIdGuardOpaque, DeleterFnPtr> guard_;
};

class ParallelGuard {
 public:
  explicit ParallelGuard() = delete;
  explicit ParallelGuard(bool state) : guard_(nullptr, delete_parallel_guard) {
    ParallelGuardHandle ptr = nullptr;
    TORCH_ERROR_CODE_CHECK(aoti_torch_create_parallel_guard(state, &ptr));
    guard_.reset(ptr);
  }

  static bool is_enabled() {
    return aoti_torch_parallel_guard_is_enabled();
  }

 private:
  std::unique_ptr<ParallelGuardOpaque, DeleterFnPtr> guard_;
};

} // namespace torch::stable
