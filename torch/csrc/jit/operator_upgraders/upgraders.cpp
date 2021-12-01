#include <torch/csrc/jit/operator_upgraders/upgraders.h>
#include <mutex>
#include <string>
#include <unordered_map>

namespace torch {
namespace jit {

std::unordered_map<std::string, std::string> UpgradersMap::content = std::unordered_map<std::string, std::string>();
std::mutex* UpgradersMap::lock = new std::mutex();
bool UpgradersMap::isPopulated = false;

void populate_upgraders_map(const std::unordered_map<std::string, std::string>& content) {
    // make sure we populate the map only once
    UpgradersMap::lock->lock();
    if (UpgradersMap::isPopulated) {
        UpgradersMap::lock->unlock();
        return;
    }
    for (const auto& entry: content) {
        UpgradersMap::content.insert(entry);
    }
    UpgradersMap::isPopulated = true;
    UpgradersMap::lock->unlock();
}

int get_upgraders_map_size() {
    UpgradersMap::lock->lock();
    int out = UpgradersMap::content.size();
    UpgradersMap::lock->unlock();
    return out;
}

// this is used for testing, so copying is not a perf issue
std::unordered_map<std::string, std::string> dump_upgraders_map() {
    UpgradersMap::lock->lock();
    auto out = UpgradersMap::content;
    UpgradersMap::lock->unlock();
    return out;
}

} // namespace jit
} // namespace torch
