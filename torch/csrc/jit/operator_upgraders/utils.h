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
    int current_version) {
  // we want to find the entry which satisfies following two conditions:
  //    1. the version entry must be greater than current_version
  //    2. Among the version entries, we need to see if the current version
  //       is in the upgrader name range

  auto upgrader_entry_copy = upgraders_for_schema;
  std::sort(
      upgrader_entry_copy.begin(),
      upgrader_entry_copy.end(),
      [](const UpgraderEntry& lhs, const UpgraderEntry& rhs) {
        return lhs.version_bump < rhs.version_bump;
      });

  auto pos = std::find_if(
      upgrader_entry_copy.begin(),
      upgrader_entry_copy.end(),
      [current_version](const UpgraderEntry& entry) {
        return entry.version_bump > current_version;
      });

  if (pos != upgrader_entry_copy.end()) {
    return *pos;
  }

  return c10::nullopt;
}

static bool isOpUptoDate(
    std::vector<UpgraderEntry> upgraders_for_schema,
    int current_version) {

  for (const auto& entry: upgraders_for_schema) {
    if (entry.version_bump > current_version) {
      return false;
    }

  }
  return true;
}

} // namespace jit
} // namespace torch
