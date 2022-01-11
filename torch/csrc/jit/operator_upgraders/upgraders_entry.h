#pragma once
#include <c10/macros/Export.h>
#include <iostream>
#include <mutex>
#include <string>
#include <unordered_map>

namespace torch {
namespace jit {

TORCH_API void populate_upgraders_graph_map();

} // namespace jit
} // namespace torch
