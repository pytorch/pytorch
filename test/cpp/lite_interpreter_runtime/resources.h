#pragma once

#include <string>

namespace torch {
namespace testing {

namespace detail {
class Path;
}

/// Gets the path to the resource identified by name.
///
/// @param name identifies a resource, relative path starting from the
///             repo root
inline auto getResourcePath(std::string name) -> detail::Path;

// End interface: implementation details follow.

namespace detail {

class Path {
 public:
  explicit Path(std::string rep) : rep_(std::move(rep)) {}

  auto string() const -> std::string const& {
    return rep_;
  }

 private:
  std::string rep_;
};

} // namespace detail

inline auto getResourcePath(std::string name) -> detail::Path {
  return detail::Path(std::move(name));
}

} // namespace testing
} // namespace torch
