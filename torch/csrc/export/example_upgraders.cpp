#include <torch/csrc/export/example_upgraders.h>
#include <torch/csrc/export/upgrader.h>

namespace torch::_export {

/// Register test upgraders for the upgrader system.
/// and shows some common upgrade patterns.
static bool test_upgraders_registered = false;

void registerExampleUpgraders() {
  if (test_upgraders_registered) {
    return;
  }

  registerUpgrader(
      0,
      "graph_module.graph.nodes",
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

  registerUpgrader(
      0,
      "graph_module.graph",
      [](const nlohmann::json& graph_obj) -> nlohmann::json {
        nlohmann::json upgraded_graph = graph_obj;

        // Rename field if it exists in the graph object
        if (upgraded_graph.contains("old_test_field")) {
          upgraded_graph["new_test_field"] = upgraded_graph["old_test_field"];
          upgraded_graph.erase("old_test_field");
        }

        return upgraded_graph;
      });

  registerUpgrader(
      1,
      std::vector<std::string>{"graph_module", "graph"},
      [](const nlohmann::json& graph_obj) -> nlohmann::json {
        nlohmann::json upgraded_graph = graph_obj;

        // Continue the field renaming chain from version 0
        if (upgraded_graph.contains("new_test_field")) {
          upgraded_graph["new_test_field2"] = upgraded_graph["new_test_field"];
          upgraded_graph.erase("new_test_field");
        }

        return upgraded_graph;
      });

  test_upgraders_registered = true;
}

/// Deregister test upgraders for the upgrader system.
void deregisterExampleUpgraders() {
  deregisterUpgrader(0, "graph_module.graph.nodes");
  deregisterUpgrader(0, "graph_module.graph");
  deregisterUpgrader(1, std::vector<std::string>{"graph_module", "graph"});
  test_upgraders_registered = false;
}

} // namespace torch::_export
