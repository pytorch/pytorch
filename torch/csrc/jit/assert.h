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

void barf(const char *fmt, ...);

}} // namespace torch::jit

#define JIT_ASSERT(cond) \
  if (__builtin_expect(!(cond), 0)) { \
    ::torch::jit::barf("ASSERTION FAILED: file %s, line %u\n", __FILE__, __LINE__); \
  }

#define JIT_ASSERTM(cond, msg, ...) \
  if (__builtin_expect(!(cond), 0)) { \
    ::torch::jit::barf(msg, __VA_ARGS__); \
  }
