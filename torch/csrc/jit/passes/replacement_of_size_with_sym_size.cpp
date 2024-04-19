#include <torch/csrc/jit/passes/replacement_of_size_with_sym_size.h>

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
#include <iostream>

namespace torch {
namespace jit {

struct SizeWithSymSizeReplacer {
  SizeWithSymSizeReplacer(std::shared_ptr<Graph> graph)
      : graph_(std::move(graph)) {}

  void run() {
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
        if (schema_name.value() == "aten::size.int") {
            WithInsertPoint guard(node);
            auto* sym_size = graph_->insertNode(graph_->create(aten::sym_size, /*num_outputs =*/1));
            for (auto* v : node->inputs()) {
                sym_size->addInput(v);
            }
            const auto& old_outputs = node->outputs();
            const auto& new_outputs = sym_size->outputs();
            for (const auto i : c10::irange(old_outputs.size())) {
                new_outputs[i]->setType(SymIntType::get());
                old_outputs[i]->replaceAllUsesWith(new_outputs[i]);
            }
            node->removeAllInputs();
            node->destroy();
        }
      }
      node = graph_it.next();
    }

    std::vector<Node*> deleted_nodes;
    DepthFirstGraphNodeIterator graph_it_v2(graph_);
    Node* node_v2 = graph_it_v2.next();

    while (node_v2) {
      // load the schema name for this op
      c10::optional<std::string> schema_name = c10::nullopt;
      if (auto op_schema = node_v2->maybeSchema()) {
        schema_name = getFullSchemaName(*op_schema);
      } else {
        schema_name = node_v2->getHistoricSchemaName();
      }

      if (schema_name.has_value()) {
        if (schema_name.value() == "prim::NumToTensor.Scalar") {
            Value* input_value = node_v2->inputs()[0];
            Value* output_value = node_v2->outputs()[0];
            output_value->replaceAllUsesWith(input_value);
            output_value->setType(SymIntType::get());
            deleted_nodes.push_back(node_v2);;
        }
      }
      node_v2 = graph_it_v2.next();
    }

    for (auto del_node : deleted_nodes) {
        del_node->destroy();
    }
  }

  std::shared_ptr<Graph> graph_;
};

TORCH_API void ReplaceSizeWithSymSize(std::shared_ptr<Graph> graph) {
  SizeWithSymSizeReplacer(std::move(graph)).run();
}

} // namespace jit
} // namespace torch
