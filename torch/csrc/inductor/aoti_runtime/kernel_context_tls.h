#pragma once

#include <cstdio>
#include <filesystem>
#include <sstream>
#include <string>
#include <utility>

namespace torch::aot_inductor {

struct KernelContext {
  std::string kernel_name;
  std::string python_stack;
  std::string compressed_python_stack;

  KernelContext(std::string name, std::string stack)
      : kernel_name(std::move(name)), python_stack(std::move(stack)) {
    compressed_python_stack = compress_python_stack(python_stack);
  }

  KernelContext(const KernelContext&) = default;
  KernelContext& operator=(const KernelContext&) = default;
  KernelContext(KernelContext&&) = default;
  KernelContext& operator=(KernelContext&&) = default;

 private:
  static std::string compress_python_stack(const std::string& stack) {
    namespace fs = std::filesystem;
    char func[129];
    char path[1025];
    uint32_t line;
    int ret;
    std::string compressed_stack;
    std::stringstream stream{stack};
    std::string str;
    std::string fmt = "File \"%1024[^\"]\", line %u, in %128[^\n]\n";
    while (std::getline(stream, str)) {
      ret = sscanf(str.c_str(), fmt.c_str(), path, &line, func);
      if (ret == 3) {
        compressed_stack += func;
        compressed_stack += ' ';
        compressed_stack += fs::path{path}.filename();
        compressed_stack += ':';
        compressed_stack += std::to_string(line);
        compressed_stack += '\n';
      }
    }
    return compressed_stack;
  }
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
