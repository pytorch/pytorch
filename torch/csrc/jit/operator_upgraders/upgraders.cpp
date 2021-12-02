#include <torch/csrc/jit/operator_upgraders/upgraders.h>
#include <mutex>
#include <string>
#include <unordered_map>

namespace torch {
namespace jit {

std::unordered_map<std::string, std::string> UpgradersMap::content = std::unordered_map<std::string, std::string>();
std::mutex UpgradersMap::lock;
bool UpgradersMap::isPopulated = false;

void populate_upgraders_map(const std::unordered_map<std::string, std::string>& content) {
    // make sure we populate the map only once
    std::lock_guard<std::mutex> lock(UpgradersMap::lock);
    if (UpgradersMap::isPopulated) {
        UpgradersMap::lock.unlock();
        return;
    }
    for (const auto& entry: content) {
        UpgradersMap::content.insert(entry);
    }
    UpgradersMap::isPopulated = true;
}

int get_upgraders_map_size() {
  std::lock_guard<std::mutex> lock(UpgradersMap::lock);
  return UpgradersMap::content.size();
}

// this is used for testing, so copying is not a perf issue
std::unordered_map<std::string, std::string> dump_upgraders_map() {
  std::lock_guard<std::mutex> lock(UpgradersMap::lock);
  return UpgradersMap::content;
}

} // namespace jit
} // namespace torch
