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

struct UpgraderRange {
  int min_version;
  int max_version;
};

// Given a list of upgrader entries for a single operator
// and the model version for that operator, find a valid
// upgrader.
TORCH_API c10::optional<UpgraderEntry> findUpgrader(
    const std::vector<UpgraderEntry>& upgraders_for_schema,
    size_t current_version);

// Utility methods to find if the operator is up-to-date
// based on all registered upgraders for this operator.
// This can be different from the current server version
// because the implementation of this operator could have
// been consistent for many later version bumps.
TORCH_API bool isOpCurrentBasedOnUpgraderEntries(
    const std::vector<UpgraderEntry>& upgraders_for_schema,
    size_t current_version);

TORCH_API bool isOpSymbolCurrent(
    const std::string& name,
    size_t current_version);

// Returns the possible old schemas for the operator that
// doesn't exist anymore. This can be true for deprecated
// operators. Since name is always a symbol name, there
// can be multiple schemas for different overloads.
TORCH_API std::vector<std::string> loadPossibleHistoricOps(
    const std::string& name,
    c10::optional<size_t> version);

TORCH_API uint64_t getMaxOperatorVersion();

// Returns the list of min and max version numbers of the operators
// that an upgrader `x` support for all upgraders for op `foo`
TORCH_API std::vector<UpgraderRange> getUpgradersRangeForOp(
    const std::string& name);

} // namespace jit
} // namespace torch
