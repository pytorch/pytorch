#include <torch/csrc/jit/passes/replacement_of_old_operators.h>

#include <c10/util/Exception.h>
#include <caffe2/serialize/versions.h>
#include <torch/csrc/jit/frontend/schema_matching.h>
#include <torch/csrc/jit/ir/irparser.h>
#include <torch/csrc/jit/operator_upgraders/upgraders.h>
#include <torch/csrc/jit/operator_upgraders/utils.h>
#include <torch/csrc/jit/operator_upgraders/version_map.h>
#include <torch/csrc/jit/runtime/graph_iterator.h>
#include <limits>
#include <string>
#include <unordered_map>
#include <utility>

namespace torch {
namespace jit {

struct OldOpsReplacerWithUpgraders {
  OldOpsReplacerWithUpgraders(std::shared_ptr<Graph> graph)
      : graph_(std::move(graph)) {}

  void run() {
    if (!graph_->get_op_version().has_value()) {
      return;
    }

    auto current_version = graph_->get_op_version().value();
    DepthFirstGraphNodeIterator graph_it(graph_);
    Node* node = graph_it.next();
    while (node) {
      // load the schema name for this op
      c10::optional<std::string> schema_name = c10::nullopt;
      if (auto op_schema = node->maybeSchema()) {
        schema_name = getFullSchemaName(*op_schema);
      } else {
        schema_name = node->getHistoricSchemaName();
      }

      if (schema_name.has_value()) {
        // this implies there was a version bump because of this operator
        auto version_entry =
            get_operator_version_map().find(schema_name.value());
        if (version_entry != get_operator_version_map().end()) {
          const auto& entry = version_entry->second;
          auto upgrader_entry = findUpgrader(entry, current_version);
          if (!upgrader_entry.has_value()) {
            if (!isOpSymbolCurrent(schema_name.value(), current_version)) {
              TORCH_INTERNAL_ASSERT(
                  false,
                  "Upgrader must be present for ",
                  schema_name.value(),
                  ". The upgrader might have deprecated");
            }
            node = graph_it.next();
            continue;
          }
          auto upgrader_entry_val = upgrader_entry.value();
          auto upgrader_name = upgrader_entry_val.upgrader_name;
          auto upgrader_graph_entry = dump_upgraders_map().find(upgrader_name);
          TORCH_INTERNAL_ASSERT(
              upgrader_graph_entry != dump_upgraders_map().end(),
              "Corresponding upgrader graph for ",
              upgrader_name,
              " must exist.",
              " This upgrader"
              " might be deprecated.");

          auto upgrader_graph = upgrader_graph_entry->second;
          // inline the upgrader function body
          WithInsertPoint guard(node);
          auto new_outputs = insertGraph(
              *node->owningGraph(), *upgrader_graph, node->inputs());
          const auto& old_outputs = node->outputs();
          TORCH_INTERNAL_ASSERT(new_outputs.size() == old_outputs.size());
          for (const auto i : c10::irange(old_outputs.size())) {
            TORCH_INTERNAL_ASSERT(
                new_outputs[i]->type() == old_outputs[i]->type())
            old_outputs[i]->replaceAllUsesWith(new_outputs[i]);
          }
          node->removeAllInputs();
          node->destroy();
        }
      }
      node = graph_it.next();
    }

    // now that we updated the graph, we want to bump the
    // graph version too.
    graph_->set_op_version(caffe2::serialize::kProducedFileFormatVersion);
  }

  std::shared_ptr<Graph> graph_;
};

TORCH_API void ReplaceOldOperatorsWithUpgraders(std::shared_ptr<Graph> graph) {
  OldOpsReplacerWithUpgraders(std::move(graph)).run();
}

} // namespace jit
} // namespace torch
