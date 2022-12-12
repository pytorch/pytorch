#pragma once

#include <c10/core/ModePyObjTrampoline.h>
#include <c10/macros/Macros.h>
#include <c10/util/ArrayRef.h>
#include <c10/util/Optional.h>

namespace c10 {
namespace impl {

struct C10_API TorchDispatchModeTLS {
  static void push_onto_stack(std::shared_ptr<c10::ModePyObjTrampoline> mode);
  static const std::shared_ptr<c10::ModePyObjTrampoline> pop_stack();
  static const std::shared_ptr<c10::ModePyObjTrampoline>& get_stack_at(int64_t idx);
  static int64_t stack_len();

  static const TorchDispatchModeTLS& get_state();
  static void set_state(const TorchDispatchModeTLS& state);

 private:
  std::vector<std::shared_ptr<c10::ModePyObjTrampoline>> stack_;
};

C10_API bool dispatch_mode_enabled();

} // namespace impl
} // namespace c10
