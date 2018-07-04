#pragma once

#include <exception>
#include <string>
#include "THP_export.h"

namespace torch {

struct assert_error final : public std::exception {
  const std::string msg;
  explicit assert_error(const std::string& msg) : msg(msg) {}
  virtual const char* what() const noexcept { return msg.c_str(); }
};

[[noreturn]]
THP_CLASS void barf(const char *fmt, ...);

} // namespace torch

#if defined(__GNUC__) || defined(__ICL) || defined(__clang__)
#define TORCH_EXPECT(x, y) (__builtin_expect((x), (y)))
#else
#define TORCH_EXPECT(x, y) (x)
#endif

#define TORCH_ASSERT(cond) \
  if (TORCH_EXPECT(!(cond), 0)) { \
    ::torch::barf("%s:%u: %s: Assertion `%s` failed.", __FILE__, __LINE__, __func__, #cond); \
  }

// The trailing ' ' argument is a hack to deal with the extra comma when ... is empty.
// Another way to solve this is ##__VA_ARGS__ in _TORCH_ASSERTM, but this is a non-portable
// extension we shouldn't use.
#define TORCH_ASSERTM(...) _TORCH_ASSERTM(__VA_ARGS__, " ")

// Note: msg must be a string literal
#define _TORCH_ASSERTM(cond, msg, ...) \
  if (TORCH_EXPECT(!(cond), 0)) { \
    ::torch::barf("%s:%u: %s: Assertion `%s` failed: " msg, __FILE__, __LINE__, __func__, #cond, __VA_ARGS__); \
  }

#define TORCH_EXPECTM(...) _TORCH_EXPECTM(__VA_ARGS__, " ")

// Note: msg must be a string literal
#define _TORCH_EXPECTM(cond, msg, ...) \
  if (TORCH_EXPECT(!(cond), 0)) { \
    ::torch::barf("%s:%u: %s: " msg, __FILE__, __LINE__, __func__, __VA_ARGS__); \
  }

#define JIT_ASSERT TORCH_ASSERT
#define JIT_ASSERTM TORCH_ASSERTM
#define JIT_EXPECTM TORCH_EXPECTM
