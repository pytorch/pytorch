#pragma once
#include <c10/macros/Export.h>
#include <iostream>
#include <mutex>
#include <string>
#include <unordered_map>

namespace torch {
namespace jit {

class UpgradersMap {
 public:
  void set_content(std::unordered_map<std::string, std::string>&& content);
  int count();
  const std::unordered_map<std::string, std::string>& get_content();
  // THESE METHODS ARE ONLY USED FOR TESTING PURPOSES
  void test_only_set_content(
      const std::unordered_map<std::string, std::string>& content);
  void test_only_remove_content(
      const std::unordered_map<std::string, std::string>& content);

 private:
  std::unordered_map<std::string, std::string> content_;
  std::mutex lock;
  bool isPopulated = false;
};

TORCH_API void populate_upgraders_map(
    std::unordered_map<std::string, std::string>&& content);

TORCH_API int get_upgraders_map_size();

TORCH_API const std::unordered_map<std::string, std::string>&
dump_upgraders_map();

// THESE TWO METHODS BELOW ARE ONLY USED FOR TESTING
TORCH_API void test_only_populate_upgraders(
    const std::unordered_map<std::string, std::string>& content);

TORCH_API void test_only_remove_upgraders(
    const std::unordered_map<std::string, std::string>& content);

} // namespace jit
} // namespace torch
