#pragma once
#include <torch/csrc/profiler/api.h>

namespace torch::profiler::impl {

using CallBackFnPtr = void (*)(
    const ProfilerConfig& config,
    const std::unordered_set<at::RecordScope>& scopes);

struct PushPRIVATEUSE1CallbacksStub {
  PushPRIVATEUSE1CallbacksStub() = default;
  PushPRIVATEUSE1CallbacksStub(const PushPRIVATEUSE1CallbacksStub&) = delete;
  PushPRIVATEUSE1CallbacksStub& operator=(const PushPRIVATEUSE1CallbacksStub&) =
      delete;

  template <typename... ArgTypes>
  void operator()(ArgTypes&&... args) {
    return (*push_privateuse1_callbacks_fn)(std::forward<ArgTypes>(args)...);
  }

  void set_privateuse1_dispatch_ptr(CallBackFnPtr fn_ptr) {
    push_privateuse1_callbacks_fn = fn_ptr;
  }

 private:
  CallBackFnPtr push_privateuse1_callbacks_fn = nullptr;
};

extern TORCH_API struct PushPRIVATEUSE1CallbacksStub
    pushPRIVATEUSE1CallbacksStub;

struct RegisterPRIVATEUSE1Observer {
  RegisterPRIVATEUSE1Observer(
      PushPRIVATEUSE1CallbacksStub& stub,
      CallBackFnPtr value) {
    stub.set_privateuse1_dispatch_ptr(value);
  }
};

#define REGISTER_PRIVATEUSE1_OBSERVER(name, fn) \
  static RegisterPRIVATEUSE1Observer name##__register(name, fn);
} // namespace torch::profiler::impl
