#pragma once
#include <c10/util/Optional.h>
#include <torch/csrc/jit/operator_upgraders/version_map.h>
#include <iostream>
#include <regex>
#include <string>
#include <vector>

namespace torch {
namespace jit {

static c10::optional<UpgraderEntry> findUpgrader(
    std::vector<UpgraderEntry> upgraders_for_schema,
    size_t current_version) {
  // we want to find the entry which satisfies following two conditions:
  //    1. the version entry must be greater than current_version
  //    2. Among the version entries, we need to see if the current version
  //       is in the upgrader name range
  auto pos = std::find_if(
      upgraders_for_schema.begin(),
      upgraders_for_schema.end(),
      [current_version](const UpgraderEntry& entry) {
        return entry.version_bump > current_version;
      });

  if (pos != upgraders_for_schema.end()) {
    return *pos;
  }

  return c10::nullopt;
}

static bool isOpEntryCurrent(
    std::vector<UpgraderEntry> upgraders_for_schema,
    size_t current_version) {

  for (const auto& entry: upgraders_for_schema) {
    if (entry.version_bump > current_version) {
      return false;
    }

  }
  return true;
}

static bool isOpSymbolCurrent(const std::string& name, size_t current_version) {
  auto it = operator_version_map.find(name);
  if (it != operator_version_map.end()) {
    return isOpEntryCurrent(it->second, current_version);
  }
  return true;
}

} // namespace jit
} // namespace torch
