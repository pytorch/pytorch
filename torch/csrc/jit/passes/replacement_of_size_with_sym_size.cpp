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
#include "jit/ir/ir.h"

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
            input_value->setType(SymIntType::get());
            output_value->setType(SymIntType::get());
            deleted_nodes.push_back(node_v2);;
        }
      }
      node_v2 = graph_it_v2.next();
    }

    for (auto del_node : deleted_nodes) {
        del_node->destroy();
    }

    std::vector<Node*> deleted_nodes_v2;
    DepthFirstGraphNodeIterator graph_it_v3(graph_);
    Node* node_v3 = graph_it_v3.next();

    while (node_v3) {
      // load the schema name for this op
      c10::optional<std::string> schema_name = c10::nullopt;
      if (auto op_schema = node_v3->maybeSchema()) {
        schema_name = getFullSchemaName(*op_schema);
      } else {
        schema_name = node_v3->getHistoricSchemaName();
      }

      if (schema_name.has_value()) {
        std::vector<NamedValue> named_values;
        for (const auto i : c10::irange(node_v3->inputs().size())) {
          auto value = node_v3->inputs()[i];
          named_values.push_back(NamedValue(node_v3->sourceRange(), value->debugName(), value));
        }
        auto symbol_name = node_v3->maybeSchema()->operator_name().name;
        WithInsertPoint guard(node_v3);
        auto out = emitBuiltinCall(node_v3->sourceRange(), *graph_, Symbol::fromQualString(symbol_name), named_values, {});
        auto new_outputs = out->node()->outputs();
        auto old_outputs = node_v3->outputs();
        for (const auto i : c10::irange(old_outputs.size())) {
            old_outputs[i]->replaceAllUsesWith(new_outputs[i]);
        }
        deleted_nodes_v2.push_back(node_v3);

      }
      node_v3 = graph_it_v3.next();
    }

    for (auto del_node : deleted_nodes_v2) {
        del_node->destroy();
    }

    for (auto* node: graph_->nodes()) {
      c10::optional<std::string> schema_name = c10::nullopt;
      if (auto op_schema = node->maybeSchema()) {
        schema_name = getFullSchemaName(*op_schema);
      } else {
        schema_name = node->getHistoricSchemaName();
      }
      if (schema_name.has_value()) {
        std::cout << "NAME: " << schema_name.value() << "\n";
      }
    }


  }

  std::shared_ptr<Graph> graph_;
};

TORCH_API void ReplaceSizeWithSymSize(std::shared_ptr<Graph> graph) {
  SizeWithSymSizeReplacer(std::move(graph)).run();
}

} // namespace jit
} // namespace torch
