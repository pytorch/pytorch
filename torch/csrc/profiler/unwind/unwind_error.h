#pragma once
#include <stdexcept>

struct UnwindError : public std::runtime_error {
  using std::runtime_error::runtime_error;
};

#define ASSERT(condition, message)                          \
  if (!(condition)) {                                       \
    throw std::runtime_error(                               \
        "Assertion failed: " #condition " (" __FILE__ ":" + \
        std::to_string(__LINE__) + ") " + (message));       \
  }
