#include <torch/csrc/jit/operator_upgraders/upgraders.h>
#include <mutex>
#include <string>
#include <unordered_map>

namespace torch {
namespace jit {

void UpgradersMap::set_content(
    const std::unordered_map<std::string, std::string>& content) {
  // make sure we populate the map only once
  std::lock_guard<std::mutex> _(lock);
  if (isPopulated) {
    return;
  }

  for (const auto& entry : content) {
    content_.insert(entry);
  }

  isPopulated = true;
}

int UpgradersMap::count() {
  std::lock_guard<std::mutex> _(lock);
  return content_.size();
}

// this is used for testing, so copying is not a perf issue
std::unordered_map<std::string, std::string> UpgradersMap::get_content() {
  std::lock_guard<std::mutex> _(lock);
  return content_;
}

void populate_upgraders_map(
    const std::unordered_map<std::string, std::string>& content) {
  upgradersMap.set_content(content);
}

int get_upgraders_map_size() {
  return upgradersMap.count();
}

std::unordered_map<std::string, std::string> dump_upgraders_map() {
  return upgradersMap.get_content();
}

} // namespace jit
} // namespace torch
