#pragma once
#include <c10/macros/Export.h>
#include <c10/util/Optional.h>
#include <torch/csrc/jit/operator_upgraders/version_map.h>
#include <iostream>
#include <regex>
#include <string>
#include <vector>

namespace torch {
namespace jit {

TORCH_API c10::optional<UpgraderEntry> findUpgrader(
    const std::vector<UpgraderEntry>& upgraders_for_schema,
    size_t current_version);

TORCH_API bool isOpEntryCurrent(
    const std::vector<UpgraderEntry>& upgraders_for_schema,
    size_t current_version);

TORCH_API bool isOpSymbolCurrent(
    const std::string& name,
    size_t current_version);

} // namespace jit
} // namespace torch
