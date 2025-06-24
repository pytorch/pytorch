#include <torch/csrc/export/upgrader.h>
#include <sstream>
#include <stdexcept>

namespace torch::_export {

// Global upgrader registry organized by version.
// Using std::multiset to maintain automatic bottom-up ordering where
// deeper keypaths are processed before shallower ones.
static std::map<int, std::multiset<Upgrader>> upgrader_registry;

Upgrader::Upgrader(std::vector<std::string> kp, UpgraderFunction func)
    : keypath(std::move(kp)), upgrade_func(std::move(func)) {}

bool Upgrader::operator<(const Upgrader& other) const {
  // First compare by depth - deeper paths come first for bottom-up processing
  if (keypath.size() != other.keypath.size()) {
    return keypath.size() > other.keypath.size();
  }
  // If same depth, compare lexicographically for deterministic ordering
  return keypath < other.keypath;
}

static void registerUpgrader(
    int version,
    const std::vector<std::string>& keypath,
    const UpgraderFunction& upgrade_func) {
  upgrader_registry[version].emplace(keypath, upgrade_func);
}

const std::multiset<Upgrader>& getUpgrader(int current_version) {
  static const std::multiset<Upgrader> empty_upgraders;
  auto it = upgrader_registry.find(current_version);
  if (it != upgrader_registry.end()) {
    return it->second;
  }
  return empty_upgraders;
}

nlohmann::json getFieldByKeypath(
    const nlohmann::json& obj,
    const std::vector<std::string>& keypath) {
  nlohmann::json current = obj;
  for (const auto& key : keypath) {
    if (!current.contains(key)) {
      throw std::runtime_error("Keypath not found: " + key);
    }
    current = current[key];
  }
  return current;
}

void setFieldByKeypath(
    nlohmann::json& obj,
    const std::vector<std::string>& keypath,
    const nlohmann::json& value) {
  nlohmann::json* current = &obj;
  for (size_t i = 0; i < keypath.size() - 1; ++i) {
    const auto& key = keypath[i];
    if (!current->contains(key)) {
      throw std::runtime_error("Keypath not found: " + key);
    }
    current = &((*current)[key]);
  }
  if (!current->contains(keypath.back())) {
    throw std::runtime_error("Keypath not found: " + keypath.back());
  }
  (*current)[keypath.back()] = value;
}

void throwUpgraderError(
    const std::string& upgrader_name,
    int from_version,
    int to_version,
    const std::string& error_message,
    const nlohmann::json& problematic_object) {
  std::ostringstream error_stream;
  error_stream << "Error in upgrader '" << upgrader_name << "' "
               << "while upgrading from version " << from_version
               << " to version " << to_version << ": " << error_message;

  if (!problematic_object.empty()) {
    error_stream << "\nProblematic object: " << problematic_object.dump(2);
  }

  throw std::runtime_error(error_stream.str());
}

nlohmann::json upgrade(const nlohmann::json& artifact) {
  auto current_artifact = artifact;

  // Validate that the artifact contains required schema version information
  if (!current_artifact.contains("schema_version")) {
    throw std::runtime_error("Missing schema_version field in artifact");
  }

  int current_version = current_artifact["schema_version"]["major"];

  // Iteratively apply upgraders until no more are available
  while (true) {
    // Look up upgraders for the current version
    const auto& upgraders = getUpgrader(current_version);

    if (upgraders.empty()) {
      // No more upgraders available - upgrade process complete
      break;
    }

    // Apply all upgraders for this version in bottom-up order
    // (deeper keypaths first to prevent parent/child conflicts)
    for (const auto& upgrader : upgraders) {
      // Extract the field to be upgraded using its keypath
      auto field_to_upgrade =
          getFieldByKeypath(current_artifact, upgrader.keypath);

      // Apply the upgrade transformation
      auto upgraded_field = upgrader.upgrade_func(field_to_upgrade);

      // Update the artifact with the upgraded field
      setFieldByKeypath(current_artifact, upgrader.keypath, upgraded_field);
    }

    // Move to the next version for potential additional upgrades
    current_version++;
  }

  // Update schema version to reflect the final upgraded version
  if (current_artifact["schema_version"]["major"] != current_version) {
    current_artifact["schema_version"]["major"] = current_version;
    // Reset minor version to 0 - the correct minor version should be set
    // when converting the json to in memory representation of ExportedProgram
    current_artifact["schema_version"]["minor"] = 0;
  }
  return current_artifact;
}

// NOTE: The following version_0 and version_1 upgraders are for testing
// purposes only. They demonstrate the upgrader system functionality and are
// used in test/export/test_upgrader.py.
//
// We use the static bool lambda pattern (static bool var = [](){...}()) to
// ensure upgraders are registered during static initialization when the module
// loads. The lambda executes once, calls registerUpgrader() as a side effect,
// and returns true to initialize the static bool variable. This guarantees
// registration happens before any upgrade() calls, without requiring explicit
// initialization functions.

static bool version_0_upgrader_registered = []() {
  registerUpgrader(
      0,
      {"graph_module", "graph", "nodes"},
      [](const nlohmann::json& nodes_array) -> nlohmann::json {
        nlohmann::json upgraded_nodes = nodes_array;

        // Process each node in the nodes array
        for (auto& node : upgraded_nodes) {
          if (node.contains("metadata") && node["metadata"].is_object()) {
            // Process each metadata key-value pair
            for (auto& [key, value] : node["metadata"].items()) {
              if (key == "nn_module_stack") {
                // Transform nn_module_stack values by prepending prefix
                if (value.is_string()) {
                  std::string stack_str = value.get<std::string>();
                  value = "test_upgrader_" + stack_str;
                } else {
                  throwUpgraderError(
                      "version_0_upgrader_registered",
                      0,
                      1,
                      "nn_module_stack metadata value must be a string, got: " +
                          std::string(value.type_name()),
                      node);
                }
              }
              // Other metadata keys remain unchanged
            }
          }
        }

        return upgraded_nodes;
      });
  return true;
}();

static bool version_0_graph_field_upgrader_registered = []() {
  registerUpgrader(
      0,
      {"graph_module", "graph"},
      [](const nlohmann::json& graph_obj) -> nlohmann::json {
        nlohmann::json upgraded_graph = graph_obj;

        // Rename field if it exists in the graph object
        if (upgraded_graph.contains("old_test_field")) {
          upgraded_graph["new_test_field"] = upgraded_graph["old_test_field"];
          upgraded_graph.erase("old_test_field");
        }

        return upgraded_graph;
      });
  return true;
}();

static bool version_1_graph_field_upgrader_registered = []() {
  registerUpgrader(
      1,
      {"graph_module", "graph"},
      [](const nlohmann::json& graph_obj) -> nlohmann::json {
        nlohmann::json upgraded_graph = graph_obj;

        // Continue the field renaming chain from version 0
        if (upgraded_graph.contains("new_test_field")) {
          upgraded_graph["new_test_field2"] = upgraded_graph["new_test_field"];
          upgraded_graph.erase("new_test_field");
        }

        return upgraded_graph;
      });
  return true;
}();

} // namespace torch::_export
