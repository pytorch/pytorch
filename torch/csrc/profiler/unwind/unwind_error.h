#pragma once
#include <stdexcept>

struct UnwindError : public std::runtime_error {
  using std::runtime_error::runtime_error;
};
