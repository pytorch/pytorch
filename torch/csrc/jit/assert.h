#pragma once

#include <exception>
#include <string>

// This header is just a little bit of indirection to make it easier to swap
// out the assert mechanism later.  Right now it just sets a Python error
// and throws.  If we want to call from not Python in the future we need
// a different approach.  Also, if we have really screwed up the heap, we
// may fail to actually get the message printed before we die.

namespace torch { namespace jit {

struct assert_error final : public std::exception {
  // THError
  const std::string msg;
  explicit assert_error(const std::string& msg) : msg(msg) {}
  virtual const char* what() const noexcept { return msg.c_str(); }
};

[[noreturn]]
void barf(const char *fmt, ...);

}} // namespace torch::jit

#define JIT_ASSERT(cond) \
  if (__builtin_expect(!(cond), 0)) { \
    ::torch::jit::barf("%s:%u: %s: Assertion `%s` failed.", __FILE__, __LINE__, __func__, #cond); \
  }

// The trailing ' ' argument is a hack to deal with the extra comma when ... is empty.
// Another way to solve this is ##__VA_ARGS__ in _JIT_ASSERTM, but this is a non-portable
// extension we shouldn't use.
#define JIT_ASSERTM(...) _JIT_ASSERTM(__VA_ARGS__, ' ')

// Note: msg must be a string literal
#define _JIT_ASSERTM(cond, msg, ...) \
  if (__builtin_expect(!(cond), 0)) { \
    ::torch::jit::barf("%s:%u: %s: Assertion `%s` failed: " msg, __FILE__, __LINE__, __func__, #cond, __VA_ARGS__); \
  }

#define JIT_EXPECTM(...) _JIT_EXPECTM(__VA_ARGS__, ' ')

// Note: msg must be a string literal
#define _JIT_EXPECTM(cond, msg, ...) \
  if (__builtin_expect(!(cond), 0)) { \
    ::torch::jit::barf("%s:%u: %s: " msg, __FILE__, __LINE__, __func__, __VA_ARGS__); \
  }
