#pragma once

#include <ATen/core/dispatch/Dispatcher.h>

// TODO: this can probably live in c10

namespace at {
namespace impl {

class TORCH_API PythonOpRegistrationTrampoline final {
  static std::atomic<c10::impl::PyInterpreter*> interpreter_;

public:
  //  Returns true if you successfully registered yourself (that means
  //  you are in the hot seat for doing the operator registrations!)
  static bool registerInterpreter(c10::impl::PyInterpreter*);
};

} // namespace impl
} // namespace at
