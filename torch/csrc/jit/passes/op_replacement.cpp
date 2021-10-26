#include <torch/csrc/jit/passes/op_replacement.h>
#include <limits>
#include <torch/csrc/jit/ir/irparser.h>
#include <torch/csrc/jit/upgraders.h>
#include <torch/csrc/jit/runtime/graph_iterator.h>

namespace torch {
namespace jit {

// TODO Is it just FileFormatVersion?
int getCurrentGlobalVersion() {
    return 0;
}

std::string findUpgrader(UpgraderDB upgraders_for_schema, int current_version) {
    std::string upgrader_name = "";
    // we want to find the entry which satisfies following two conditions:
    //    1. the version entry must be greater than current_version
    //    2. Among the version entries that are greater than current_version,
    //       we choose one that is the closest to the current_version.
    int closest_difference = std::numeric_limits<int>::max();
    for (const auto& upgrader_entry: upgraders_for_schema) {
        if (upgrader_entry.first > current_version) {
            int current_diff = upgrader_entry.first - current_version;
            if(current_diff < closest_difference) {
                closest_difference = current_diff;
                upgrader_name = upgrader_entry.second;
            }
        }
    }
    return upgrader_name;
}

struct OpsReplacer {
  OpsReplacer(std::shared_ptr<Graph> graph)
      : graph_(std::move(graph)) {}

//   void run() {
//     auto graph_block = graph_->block();
//     for(auto node_it = graph_block->nodes().begin(), end = graph_block->nodes().end();  node_it != end;
//        ++node_it) {
//         if (auto schema = node_it->maybeSchema()) {
//             auto schema_name = schema->name() + "." + schema->overload_name();
//             // this implies there was a version bump because of this operator
//             auto version_entry = operator_version_map.find(schema_name);
//             if (version_entry != operator_version_map.end()){
//                 auto current_version = getCurrentGlobalVersion();
//                 auto upgrader_name = findUpgrader(version_entry->second, current_version);
//                 auto upgrader_graph_entry = upgraders_graph.find(upgrader_name);
//                 assert(upgrader_graph_entry != upgraders_graph.end());
//                 auto upgrader_graph = std::make_shared<torch::jit::Graph>();
//                 parseIR(upgrader_graph_entry->second, upgrader_graph.get());
//                 // inline the upgrader function body
//                 auto new_outputs = insertGraph(*node_it->owningGraph(), *upgrader_graph, node_it->inputs());
//                 const auto& old_outputs = node_it->outputs();
//                 AT_ASSERT(new_outputs.size() == old_outputs.size());
//                 for (const auto i : c10::irange(old_outputs.size())) {
//                     new_outputs[i]->setType(old_outputs[i]->type());
//                     old_outputs[i]->replaceAllUsesWith(new_outputs[i]);
//                 }
//                 node_it->removeAllInputs();
//                 node_it.destroyCurrent();
//             }
//         }
//     }
//   }
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
                auto upgrader_graph_entry = upgraders_graph.find(upgrader_name);
                assert(upgrader_graph_entry != upgraders_graph.end());
                auto upgrader_graph = std::make_shared<torch::jit::Graph>();
                parseIR(upgrader_graph_entry->second, upgrader_graph.get());
                // inline the upgrader function body
                auto new_outputs = insertGraph(*node->owningGraph(), *upgrader_graph, node->inputs());
                const auto& old_outputs = node->outputs();
                AT_ASSERT(new_outputs.size() == old_outputs.size());
                for (const auto i : c10::irange(old_outputs.size())) {
                    new_outputs[i]->setType(old_outputs[i]->type());
                    old_outputs[i]->replaceAllUsesWith(new_outputs[i]);
                }
                node->removeAllInputs();
                node->destroy();
                //graph_it.next();
            }
        }
        node = graph_it.next();
    }
  }

  std::shared_ptr<Graph> graph_;
};

TORCH_API void ReplaceOpsWithUpgraders(const std::shared_ptr<Graph>& graph) {
    OpsReplacer(graph).run();
}

} // namespace jit
} // namespace torch
