#pragma once

#include <experimental/filesystem>
#include <string>

namespace torch {
namespace testing {

/// Gets the path to the resource identified by name.
///
/// @param name identifies a resource, relative path starting from the
///             repo root
auto getResourcePath(std::string name) -> std::experimental::filesystem::path {
  return std::move(name);
}

} // namespace testing
} // namespace torch
