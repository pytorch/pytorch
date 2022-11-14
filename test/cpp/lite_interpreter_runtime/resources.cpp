#include "test/cpp/lite_interpreter_runtime/resources.h"

namespace torch {
namespace testing {

auto getResourcePath(std::string name) -> detail::Path {
  return Path(std::move(name));
}

namespace detail {

Path::Path(std::string rep) : rep_(std::move(rep)) {}

auto Path::string() const -> std::string { return rep_; }

}  // namespace detail

}  // namespace testing
}  // namespace torch
