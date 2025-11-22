#pragma once

#include <string>
#include <utility>

namespace torch::aot_inductor {

struct KernelContext {
  std::string kernel_name;
  std::string python_stack;

  KernelContext(std::string name, std::string stack)
      : kernel_name(std::move(name)), python_stack(std::move(stack)) {}
};

// Thread-local pointer
extern thread_local KernelContext* tls_kernel_context;

inline KernelContext* current_kernel_context() {
  return tls_kernel_context;
}

inline void set_kernel_context(KernelContext* ctx) {
  tls_kernel_context = ctx;
}

inline void clear_kernel_context() {
  tls_kernel_context = nullptr;
}

struct KernelContextGuard {
  KernelContextGuard(const std::string& name, const std::string& stack)
      : owned_context_(name, stack) {
    set_kernel_context(&owned_context_);
  }
  ~KernelContextGuard() {
    clear_kernel_context();
  }

  // Delete copy constructor and copy assignment operator
  KernelContextGuard(const KernelContextGuard&) = delete;
  KernelContextGuard& operator=(const KernelContextGuard&) = delete;

  KernelContextGuard(KernelContextGuard&&) = default;
  KernelContextGuard& operator=(KernelContextGuard&&) = delete;

 private:
  KernelContext owned_context_;
};

} // namespace torch::aot_inductor
