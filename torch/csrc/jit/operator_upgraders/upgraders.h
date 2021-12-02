#pragma once
#include <c10/macros/Export.h>
#include <iostream>
#include <mutex>
#include <string>
#include <unordered_map>

namespace torch {
namespace jit {

struct UpgradersMap {
    static std::unordered_map<std::string, std::string> content;
    static std::mutex lock;
    static bool isPopulated;
};

TORCH_API void populate_upgraders_map(const std::unordered_map<std::string, std::string>& content);

TORCH_API int get_upgraders_map_size();

// this is used for testing, so copying is not a perf issue
TORCH_API std::unordered_map<std::string, std::string> dump_upgraders_map();

} // namespace jit
} // namespace torch
