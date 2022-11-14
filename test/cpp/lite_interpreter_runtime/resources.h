#ifndef RESOURCES_H
#define RESOURCES_H

#include <experimental/filesystem>
#include <string>

namespace torch {
namespace testing {

/// Gets the path to the resource identified by name.
auto getResourcePath(std::string name) -> std::experimental::filesystem::path {
  return std::move(name);
}

} // namespace testing
} // namespace torch

#endif
