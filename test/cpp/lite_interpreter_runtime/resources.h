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
  Path(std::string rep);

  auto string() const -> std::string const&;

 private:
  std::string rep_;
};
}  // namespace detail

}  // namespace testing
}  // namespace torch

#endif
