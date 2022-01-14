#include <torch/csrc/jit/operator_upgraders/utils.h>

#include <c10/util/Optional.h>
#include <torch/csrc/jit/operator_upgraders/version_map.h>
#include <iostream>
#include <regex>
#include <string>
#include <vector>

namespace torch {
namespace jit {

c10::optional<UpgraderEntry> findUpgrader(
    const std::vector<UpgraderEntry>& upgraders_for_schema,
    size_t current_version) {
  // we want to find the entry which satisfies following two conditions:
  //    1. the version entry must be greater than current_version
  //    2. Among the version entries, we need to see if the current version
  //       is in the upgrader name range
  auto pos = std::find_if(
      upgraders_for_schema.begin(),
      upgraders_for_schema.end(),
      [current_version](const UpgraderEntry& entry) {
        return entry.bumped_at_version > current_version;
      });

  if (pos != upgraders_for_schema.end()) {
    return *pos;
  }
  return c10::nullopt;
}

bool isOpCurrentBasedOnUpgraderEntries(
    const std::vector<UpgraderEntry>& upgraders_for_schema,
    size_t current_version) {
  auto latest_update =
      upgraders_for_schema[upgraders_for_schema.size() - 1].bumped_at_version;
  if (latest_update > current_version) {
    return false;
  }
  return true;
}

bool isOpSymbolCurrent(const std::string& name, size_t current_version) {
  auto it = get_operator_version_map().find(name);
  if (it != get_operator_version_map().end()) {
    return isOpCurrentBasedOnUpgraderEntries(it->second, current_version);
  }
  return true;
}

std::vector<std::string> loadPossibleHistoricOps(
    const std::string& name,
    c10::optional<size_t> version) {
  std::vector<std::string> possibleSchemas;

  if (!version.has_value()) {
    return possibleSchemas;
  }

  for (const auto& entry : get_operator_version_map()) {
    auto old_symbol_name = entry.first;
    // strip off the overload name, if exist
    auto base_name = old_symbol_name.substr(0, old_symbol_name.find('.'));
    if (base_name == name) {
      auto possibleUpgrader = findUpgrader(entry.second, version.value());
      if (possibleUpgrader.has_value()) {
        possibleSchemas.push_back(possibleUpgrader.value().old_schema);
      }
    }
  }

  return possibleSchemas;
}

} // namespace jit
} // namespace torch
