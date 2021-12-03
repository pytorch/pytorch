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
  std::unordered_map<std::string, std::string> get_content();

 private:
  std::unordered_map<std::string, std::string> content_;
  std::mutex lock;
  bool isPopulated = false;
};

TORCH_API void populate_upgraders_map(const std::unordered_map<std::string, std::string>& content);

TORCH_API int get_upgraders_map_size();

TORCH_API std::unordered_map<std::string, std::string> dump_upgraders_map();

static UpgradersMap upgradersMap;

} // namespace jit
} // namespace torch
