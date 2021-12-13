#include <torch/csrc/jit/operator_upgraders/upgraders.h>
#include <mutex>
#include <string>
#include <unordered_map>

namespace torch {
namespace jit {

static UpgradersMap upgradersMap;

void UpgradersMap::set_content(
    std::unordered_map<std::string, std::string>&& content) {
  // make sure we populate the map only once
  std::lock_guard<std::mutex> _(lock);
  if (isPopulated) {
    return;
  }
  content_ = std::move(content);
  isPopulated = true;
}

int UpgradersMap::count() {
  std::lock_guard<std::mutex> _(lock);
  return content_.size();
}

const std::unordered_map<std::string, std::string>& UpgradersMap::
    get_content() {
  std::lock_guard<std::mutex> _(lock);
  return content_;
}

void UpgradersMap::test_only_set_content(
    const std::unordered_map<std::string, std::string>& content) {
  std::lock_guard<std::mutex> _(lock);
  for (const auto& entry : content) {
    content_.insert(entry);
  }
}
void UpgradersMap::test_only_remove_content(
    const std::unordered_map<std::string, std::string>& content) {
  std::lock_guard<std::mutex> _(lock);
  for (const auto& entry : content) {
    content_.erase(entry.first);
  }
}

void populate_upgraders_map(
    std::unordered_map<std::string, std::string>&& content) {
  upgradersMap.set_content(std::move(content));
}

int get_upgraders_map_size() {
  return upgradersMap.count();
}

const std::unordered_map<std::string, std::string>& dump_upgraders_map() {
  return upgradersMap.get_content();
}

void test_only_populate_upgraders(
    const std::unordered_map<std::string, std::string>& content) {
  upgradersMap.test_only_set_content(content);
}

void test_only_remove_upgraders(
    const std::unordered_map<std::string, std::string>& content) {
  upgradersMap.test_only_remove_content(content);
}

} // namespace jit
} // namespace torch
