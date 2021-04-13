#pragma once

#include "lazy_tensors/status.h"

namespace lazy_tensors {

template <typename T>
class StatusOr {
 public:
  StatusOr(const T& value) : value_(value), status_(Status::OK()) {}

  const Status& status() const { return status_; }

  T ConsumeValueOrDie() { return std::move(value_); }

  const T& ValueOrDie() const { return value_; }

 private:
  const T value_;
  const Status status_;
};

}  // namespace lazy_tensors
