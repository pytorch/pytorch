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
      : kernel_name(std::move(name)) {
    python_stack = trim_stack(stack);
    compressed_python_stack = compress_stack(python_stack);
  }

  KernelContext(const KernelContext&) = default;
  KernelContext& operator=(const KernelContext&) = default;
  KernelContext(KernelContext&&) = default;
  KernelContext& operator=(KernelContext&&) = default;

 private:
  // Strip leading and trailing newlines from stack:
  // - finds first and last non-newline characters
  // - outputs substring between them (inclusive)
  // - returns empty string if stack contains only newlines
  static std::string trim_stack(const std::string& stack) {
    std::string res_stack;
    auto beg = stack.find_first_not_of('\n');
    if (beg == std::string::npos) {
      return res_stack;
    }
    auto end = stack.find_last_not_of('\n');
    if (end == std::string::npos) {
      res_stack = stack.substr(beg);
    } else {
      res_stack = stack.substr(beg, end - beg + 1);
    }
    return res_stack;
  }

  // Compress stack into compact format:
  // - blank lines are treated as stack separators
  // - lines with 0 or 2 spaces of indentation are parsed as filename, fileline
  //   and function
  // - lines with 4 spaces of indentation are parsed as the code snippet
  // - outputs function[code], filename and fileline (at newline each)
  // - returns empty string on any parse error
  static std::string compress_stack(const std::string& stack) {
    std::string res_stack;
    namespace fs = std::filesystem;
    char function[1025];
    char filename[1025];
    uint32_t fileline;
    int ret, n, ws;
    const char* p;
    std::stringstream stream{stack};
    std::string line;
    std::string fmt = "File \"%1024[^\"]\", line %u, in %1024[^\n]\n%n";
    while (std::getline(stream, line)) {
      // check if new stack
      if (line.empty()) {
        res_stack += '\n';
        continue;
      }
      p = line.c_str();
      ws = 0;
      while (*p == ' ') {
        ++p;
        ++ws;
      }
      // check if new file
      if (ws != 0 && ws != 2) {
        return {};
      }
      ret = sscanf(p, fmt.c_str(), filename, &fileline, function, &n);
      if (ret != 3) {
        return {};
      }
      if (!std::getline(stream, line)) {
        return {};
      }
      p = line.c_str();
      ws = 0;
      while (*p == ' ') {
        ++p;
        ++ws;
      }
      // check if command
      if (ws != 4) {
        return {};
      }
      res_stack += std::string{function} + '[' + std::string{p} + ']';
      res_stack += '\n';
      res_stack += fs::path{filename}.filename().string();
      res_stack += '\n';
      res_stack += std::to_string(fileline);
      res_stack += '\n';
    }
    return res_stack;
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
