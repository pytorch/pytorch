#ifndef RESOURCES_H
#define RESOURCES_H

#include <string>

namespace torch {
namespace testing {

namespace detail { class Path; }

auto getResourcePath(std::string name) -> detail::Path;

namespace detail {
class Path {
 public:
  Path(std::string rep) : rep_(std::move(rep)) {}

  auto string() const -> std::string const& { return rep_; }

 private:
  std::string rep_;
};
}  // namespace detail

auto getResourcePath(std::string name) -> detail::Path {
  return detail::Path(std::move(name));
}

}  // namespace testing
}  // namespace torch

#endif
