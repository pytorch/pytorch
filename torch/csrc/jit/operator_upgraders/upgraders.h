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
  void set_content(const std::unordered_map<std::string, std::string>& content);
  int count();
  const std::unordered_map<std::string, std::string>& get_content();
  // THESE METHODS ARE ONLY USED FOR TESTING PURPOSES
  void set_content_for_test(
      const std::unordered_map<std::string, std::string>& content);
  void remove_content_for_test(
      const std::unordered_map<std::string, std::string>& content);

 private:
  std::unordered_map<std::string, std::string> content_;
  std::mutex lock;
  bool isPopulated = false;
};

TORCH_API void populate_upgraders_map(const std::unordered_map<std::string, std::string>& content);

TORCH_API int get_upgraders_map_size();

TORCH_API const std::unordered_map<std::string, std::string>&
dump_upgraders_map();

// THESE TWO METHODS BELOW ARE ONLY USED FOR TESTING
TORCH_API void populate_test_upgraders(
    const std::unordered_map<std::string, std::string>& content);

TORCH_API void remove_test_upgraders(
    const std::unordered_map<std::string, std::string>& content);

static UpgradersMap upgradersMap;

} // namespace jit
} // namespace torch
