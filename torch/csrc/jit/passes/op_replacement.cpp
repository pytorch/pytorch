#include <torch/csrc/jit/passes/op_replacement.h>
#include <c10/util/Exception.h>
#include <limits>
#include <regex>
#include <string>
#include <torch/csrc/jit/ir/irparser.h>
#include <torch/csrc/jit/operator_upgraders/upgraders.h>
#include <torch/csrc/jit/operator_upgraders/utils.h>
#include <torch/csrc/jit/operator_upgraders/version_map.h>
#include <torch/csrc/jit/runtime/graph_iterator.h>

namespace torch {
namespace jit {

// TODO will be somehow passed from the pt file
int getCurrentGlobalVersion() {
    return 0;
}

struct OpsReplacer {
  OpsReplacer(std::shared_ptr<Graph> graph)
      : graph_(std::move(graph)) {}

  void run() {
    DepthFirstGraphNodeIterator graph_it(graph_);
    Node* node = graph_it.next();
    while(node) {
        if (auto schema = node->maybeSchema()) {
            auto schema_name = schema->name() + "." + schema->overload_name();
            // this implies there was a version bump because of this operator
            auto version_entry = operator_version_map.find(schema_name);
            if (version_entry != operator_version_map.end()){
                auto current_version = getCurrentGlobalVersion();
                auto upgrader_name = findUpgrader(version_entry->second, current_version);
                auto upgrader_graph_entry = upgraders_graph.find(upgrader_name.upgrader_name);
                assert(upgrader_graph_entry != upgraders_graph.end());
                torch::jit::Graph upgrader_graph;
                parseIR(upgrader_graph_entry->second, &upgrader_graph);
                // inline the upgrader function body
                WithInsertPoint guard(node);
                auto new_outputs = insertGraph(*node->owningGraph(), upgrader_graph, node->inputs());
                const auto& old_outputs = node->outputs();
                TORCH_INTERNAL_ASSERT(new_outputs.size() == old_outputs.size());
                for (const auto i : c10::irange(old_outputs.size())) {
                    TORCH_INTERNAL_ASSERT(new_outputs[i]->type() == old_outputs[i]->type())
                    old_outputs[i]->replaceAllUsesWith(new_outputs[i]);
                }
                node->removeAllInputs();
                node->destroy();
            }
        }
        node = graph_it.next();
    }
  }

  std::shared_ptr<Graph> graph_;
};

TORCH_API void ReplaceOpsWithUpgraders(std::shared_ptr<Graph> graph) {
    OpsReplacer(graph).run();
}

} // namespace jit
} // namespace torch
