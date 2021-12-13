#include <torch/csrc/jit/passes/dbr_quantization/annotate_with_order_logger.h>

#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/frontend/schema_matching.h>
#include <torch/csrc/jit/passes/graph_rewrite_helper.h>
#include <torch/csrc/jit/passes/quantization/helper.h>
#include <iostream>


namespace torch {
namespace jit {

namespace {

// TODO(future PR): add all the quantizeable functions
// TODO(future PR): figure out if this can work with user defined functions
std::vector<std::string> _static_quantizable_call_funcs = {};
std::vector<std::string> _static_quantizable_aten_funcs = {
  "add",
};

using ModuleMethodVector = std::vector<std::pair<Module, std::string>>;

// TODO(before landing): refactor from insert_quant_dequant.cpp instead of copy-pasta
ModuleMethodVector getInvokedMethods(
    Module& module,
    const std::string& method_name) {
  auto graph = module.get_method(method_name).graph();

  ModuleMethodVector invoked_methods;
  std::stack<Block*> blocks_to_visit;
  blocks_to_visit.push(graph->block());
  while (!blocks_to_visit.empty()) {
    Block* b = blocks_to_visit.top();
    blocks_to_visit.pop();
    for (Node* n : b->nodes()) {
      if (n->kind() == prim::CallMethod) {
        auto module_instance = n->inputs()[0];
        auto module_method_name = n->s(attr::name);
        c10::optional<Module> m;
        // calling method on self
        if (module_instance == graph->inputs()[0]) {
          m = module;
        } else if (
            module_instance->node()->kind() == prim::GetAttr &&
            module_instance->node()->s(attr::name).find("_observer_") ==
                std::string::npos) {
          m = getInvokedModuleOpt(module, n, graph->inputs()[0]);
        }
        if (m) {
          invoked_methods.push_back({*m, module_method_name});
        }
      }

      for (Block* subblock : n->blocks()) {
        blocks_to_visit.push(subblock);
      }
    }
  }
  return invoked_methods;
}

} // namespace

Module DBRQuantAnnotateWithOrderLogger(Module& module, const Module& logger) {

  // Call the function recursively on child methods
  for (auto& invoked_methods : getInvokedMethods(module, "forward")) {
    auto& invoked_module = std::get<0>(invoked_methods);
    DBRQuantAnnotateWithOrderLogger(invoked_module, logger);
  }

  // After this point, there is no more recursion. We will transform
  //
  //   def forward(self, x):
  //     x = x + x
  //     x = x + x
  //     return x
  //
  // into
  //
  //   def forward(self, x):
  //     _logger_0 = self._logger_0
  //     x = x + x
  //     # record the order of execution of the previous op with ID 0
  //     x = _logger_0(0, x)
  //     x = x + x
  //     # record the order of execution of the previous op with ID 1
  //     x = _logger_0(1, x)
  //     return x

  // Attach a copy of the logger instance to the current module
  Module logger_copy = logger.deepcopy();
  int counter = 0;
  std::string logger_name = "_logger_" + c10::to_string(counter);
  while (module.hasattr(logger_name)) {
    logger_name = "_logger_" + c10::to_string(counter++);
  }
  module.register_module(logger_name, logger_copy);
  // std::cout << module.dump_to_str(false, false, false) << std::endl;

  for (auto& method : module.get_methods()) {
    GRAPH_DUMP(
        module.type()->name()->name() + "::" + method.name() +
            "() before dbr_quant_annotate_with_order_logger",
        method.graph());

    auto g = method.graph();
    std::vector<Node*> orig_nodes;
    std::stack<Block*> blocks_to_visit;
    blocks_to_visit.push(g->block());
    while (!blocks_to_visit.empty()) {
      Block* b = blocks_to_visit.top();
      blocks_to_visit.pop();
      for (auto* node : b->nodes()) {
        orig_nodes.push_back(node);
        for (auto* subblock : node->blocks()) {
          blocks_to_visit.push(subblock);
        }
      }
    }

    // insert a single get_attr call to the logger, it will be shared
    // among the ops that need to log
    Node* logger_instance =
     g->createGetAttr(g->inputs()[0], logger_name)
      ->insertBefore(g->nodes().front());

    int node_idx = 0;

    // for each op we care about, pass its output through the logger
    for (auto* node : orig_nodes) {

      // `needs_logger` is true for nodes which are quantizeable
      // and need to be associated with Python state. For example,
      bool needs_logger = false;
      if (isFunctionNode(
            node,
            _static_quantizable_call_funcs,
            _static_quantizable_aten_funcs)) {
        needs_logger = true;
      }
      // TODO(future PR): handle modules, methods, user functions

      if (!needs_logger) {
        continue;
      }

      // Pass this node's output through the logger and tag it with
      // this op's node_idx. For example,
      //
      // before:
      //
      //   %_logger_0 = prim::GetAttr[name="_logger_0"](%self)
      //   %x.7 : Tensor = aten::add(%x.1, %x.1, %4)
      //
      // after:
      //
      //   %_logger_0 = prim::GetAttr[name="_logger_0"](%self)
      //   %x.13 : Tensor = aten::add(%x.1, %x.1, %4)
      //   %22 : int = prim::Constant[value=0]()
      //   %x.7 : Tensor = prim::CallMethod[name="forward"](%_logger_0, %22, %x.13)
      {
        WithInsertPoint guard(node->next());
        Value* output_value = node->outputs()[0];
        // Match arguments to types of observer's arguments
        MatchedSchema forward_matched_schema = matchSchema(
          logger_copy.get_method("forward").function().getSchema(),
          node->sourceRange(),
          *g,
          // the first arg is self
          {logger_instance->output(), node_idx, output_value},
          {});
        node_idx++;

        Node* call = g->insertMethodCall("forward", forward_matched_schema)
          ->node();
        call->output()->copyMetadata(output_value);

        // replace node_output with the output of logger
        output_value->replaceAllUsesWith(call->output());
        // the above also replaces the input to `call`, switch it back
        call->replaceInput(2, output_value);
      }
    }

    GRAPH_DUMP(
        module.type()->name()->name() + "::" + method.name() +
            "() after dbr_quant_annotate_with_order_logger",
        method.graph());

  }

  return module;
}


} // namespace jit
} // namespace torch
