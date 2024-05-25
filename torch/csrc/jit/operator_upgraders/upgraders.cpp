#include <torch/csrc/jit/operator_upgraders/upgraders.h>

#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/ir/irparser.h>
#include <mutex>
#include <string>
#include <unordered_map>

namespace torch::jit {

static UpgradersMap upgradersMap;

void UpgradersMap::set_content(
    std::unordered_map<std::string, std::shared_ptr<Graph>>&& content) {
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

bool UpgradersMap::is_populated() {
  std::lock_guard<std::mutex> _(lock);
  return isPopulated;
}

const std::unordered_map<std::string, std::shared_ptr<Graph>>& UpgradersMap::
    get_content() {
  std::lock_guard<std::mutex> _(lock);
  return content_;
}

void UpgradersMap::test_only_set_content(
    const std::unordered_map<std::string, std::string>& content) {
  std::lock_guard<std::mutex> _(lock);
  for (const auto& entry : content) {
    auto graph = std::make_shared<Graph>();
    torch::jit::parseIR(entry.second, graph.get());
    content_.insert(std::make_pair(entry.first, graph));
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
    std::unordered_map<std::string, std::shared_ptr<Graph>>&& content) {
  upgradersMap.set_content(std::move(content));
}

int get_upgraders_map_size() {
  return upgradersMap.count();
}

bool is_upgraders_map_populated() {
  return upgradersMap.is_populated();
}

const std::unordered_map<std::string, std::shared_ptr<Graph>>&
dump_upgraders_map() {
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

} // namespace torch::jit
