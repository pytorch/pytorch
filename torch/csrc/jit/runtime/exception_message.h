#pragma once
#include <c10/util/Exception.h>
#include <stdexcept>

namespace torch {
namespace jit {

struct ExceptionMessage {
  ExceptionMessage(const std::exception& e) : e_(e) {}

 private:
  const std::exception& e_;
  friend std::ostream& operator<<(
      std::ostream& out,
      const ExceptionMessage& msg);
};

inline std::ostream& operator<<(
    std::ostream& out,
    const ExceptionMessage& msg) {
  auto c10_error = dynamic_cast<const c10::Error*>(&msg.e_);
  if (c10_error) {
    out << c10_error->what_without_backtrace();
  } else {
    out << msg.e_.what();
  }
  return out;
}

} // namespace jit
} // namespace torch
