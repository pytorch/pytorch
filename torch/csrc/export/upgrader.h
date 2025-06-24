#pragma once

#include <nlohmann/json.hpp>
#include <functional>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace torch::_export {

/// Function type for upgrading JSON fields during schema version migration.
/// Takes a JSON field and returns the upgraded version of that field.
using UpgraderFunction = std::function<nlohmann::json(const nlohmann::json&)>;

/// Structure containing upgrader information for a specific keypath.
/// The version is stored as the map key in the registry, so it's not
/// duplicated here.
struct Upgrader {
  /// Path to the field that should be upgraded (e.g., {"graph_module", "graph",
  /// "nodes"}) Assuming top-level is a JSON object that represents
  /// ExportedProgram
  std::vector<std::string> keypath;

  /// Function that performs the actual upgrade transformation
  UpgraderFunction upgrade_func;

  /// Constructor for creating an upgrader with keypath and function
  Upgrader(std::vector<std::string> kp, UpgraderFunction func);

  /// Comparator for maintaining bottom-up ordering in the registry.
  /// Deeper keypaths are processed first to ensure safe upgrade application
  /// without conflicts between parent and child field modifications.
  bool operator<(const Upgrader& other) const;
};

/// Retrieve all upgraders registered for a specific schema version.
///
/// @param current_version The schema version to get upgraders for
/// @return Multiset of upgraders ordered for safe application (bottom-up)
const std::multiset<Upgrader>& getUpgrader(int current_version);

/// Extract a field value from JSON using a keypath.
///
/// Traverses the JSON object following the provided keypath and returns
/// the value at that location. Throws if any part of the path doesn't exist.
///
/// @param obj The JSON object to traverse
/// @param keypath Path components to follow (e.g., {"a", "b", "c"} for
/// obj.a.b.c)
/// @return The JSON value at the specified keypath
/// @throws std::runtime_error if keypath is not found
nlohmann::json getFieldByKeypath(
    const nlohmann::json& obj,
    const std::vector<std::string>& keypath);

/// Set a field value in JSON using a keypath.
///
/// Traverses the JSON object following the provided keypath and sets
/// the value at that location. Throws if any part of the path doesn't exist.
///
/// @param obj The JSON object to modify
/// @param keypath Path components to follow
/// @param value The new value to set at the keypath location
/// @throws std::runtime_error if keypath is not found
void setFieldByKeypath(
    nlohmann::json& obj,
    const std::vector<std::string>& keypath,
    const nlohmann::json& value);

void throwUpgraderError(
    const std::string& upgrader_name,
    int from_version,
    int to_version,
    const std::string& error_message,
    const nlohmann::json& problematic_object = nlohmann::json::object());

/// Notes [JSON based export schema upgrader]
/// Upgrade a JSON artifact to latest major version with all available
/// upgraders. It handles major version upgrade only. For minor version upgrade,
/// e.g. adding a new field with default value, it's automatically handled by
/// the default constructor in generated_serialization_types.h.
///
/// Iteratively applies schema version upgraders starting from the artifact's
/// current version until no more upgraders are available. Each version's
/// upgraders are applied in bottom-up order (deeper keypaths first) to
/// prevent conflicts between parent and child field modifications.
///
/// @param artifact The JSON artifact to upgrade
/// @return The upgraded JSON artifact with updated schema version
nlohmann::json upgrade(const nlohmann::json& artifact);

} // namespace torch::_export
