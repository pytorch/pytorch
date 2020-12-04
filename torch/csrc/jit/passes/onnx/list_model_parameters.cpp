#include <torch/csrc/jit/passes/onnx/list_model_parameters.h>
#include <torch/csrc/jit/frontend/error_report.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>

namespace torch {
namespace jit {

// findSubModuleAttr function chases getAttr chains backwards to locate the
// submodules. For example: module M {
//   attributes {
//     A = <SubModule at ...>
//   }
//   ...
//   %A = prim::GetAttr[name="A"](%self)
//   ...
//   %B = prim::GetAttr[name="B"](%A)
//   ...
//   %weight = prim::GetAttr[name="scale"](%B)
//   ...

std::deque<std::string> findSubModuleAttr(
    Value* input,
    std::string& name,
    Module& attrModule,
    std::shared_ptr<Graph>& graph) {
  Node* node = input->node();
  std::deque<std::string> moduleNames;

  // Loop starts from inner submodule and follows the chain until reaches the
  // top module.
  while (node->outputs().at(0)->type() != graph->inputs().at(0)->type()) {
    if (node->kind() == prim::GetAttr) {
      moduleNames.push_front(node->s(attr::name));
      node = node->inputs()[0]->node();
    }
  }

  // Assign the inner module to attrModule.
  for (auto& moduleName : moduleNames) {
    attrModule = attrModule.attr(moduleName).toModule();
  }
  return moduleNames;
}

Value* addParamAsArgument(Function* function, std::string& name, IValue& attr) {
  auto schema = function->getSchema();
  auto args = schema.arguments();
  args.emplace_back(Argument(name, nullptr, c10::nullopt, attr));
  auto new_schema = FunctionSchema(
      schema.name(),
      schema.overload_name(),
      args,
      schema.returns(),
      schema.is_vararg(),
      schema.is_varret());
  function->setSchema(new_schema);
  return function->graph()->addInput(name)->setType(attr.type());
}

std::vector<IValue> getParamAttributes(
    std::shared_ptr<Graph>& graph,
    const Module& module_,
    Function* function_) {
  std::vector<IValue> attrValues;
  auto isEval = !module_.hasattr("training") || !module_.is_training();
  auto block = graph->block();
  std::vector<Block*> blocks({block});

  Node* m = *block->nodes().begin();
  WithInsertPoint guard(m);

  while (!blocks.empty()) {
    Block* block = blocks.back();
    blocks.pop_back();
    for (auto it = block->nodes().begin(); it != block->nodes().end();) {
      Node* n = *it;
      it++; // node n can be destroyed

      for (Block* sub_block : n->blocks()) {
        blocks.emplace_back(sub_block);
      }
      if (n->kind() == prim::SetAttr &&
          n->s(attr::name) == "num_batches_tracked") {
        n->destroy();
      } else if (n->kind() == prim::GetAttr) {
        for (auto use : n->output()->uses()) {
          if (use.user->kind() == prim::PythonOp)
            throw ErrorReport(n->sourceRange())
                << "Couldn't export Python method.";
        }

        auto name = n->s(attr::name);
        auto attrModule = module_;
        auto input = n->inputs()[0];

        auto moduleNames = findSubModuleAttr(input, name, attrModule, graph);
        if (!attrModule.hasattr(name)) {
          continue;
        }
        Value* paramConst = nullptr;

        auto attr = attrModule.attr(name);

        std::string fullName("");
        for (auto& name : moduleNames) {
          fullName += name + '.';
        }
        fullName += name;

        auto type = attrModule.type();
        auto slot = *type->findAttributeSlot(name);

        if (type->is_parameter(slot) || type->is_buffer(slot) ||
            (attr.isObject() && !attr.toObjectRef().type()->is_module()) ||
            name == "training") {
          if (attr.isTensor()) {
            TORCH_INTERNAL_ASSERT(attr.isTensor());
            auto tensor_ = attr.toTensor();
            if (isEval && tensor_.requires_grad()) {
              tensor_ = tensor_.detach();
              tensor_.set_requires_grad(false);
              attr = IValue(tensor_);
            }
            attrValues.emplace_back(attr.toTensor());
            paramConst = addParamAsArgument(function_, fullName, attr);
          } else if (
              attr.isObject() && !attr.toObjectRef().type()->is_module()) {
            // Only below registered torch classes are supported.
            auto type = attr.type();
            TORCH_CHECK(
                (type ==
                 getCustomClass(
                     "__torch__.torch.classes.quantized.Conv2dPackedParamsBase")) ||
                    (type ==
                     getCustomClass(
                         "__torch__.torch.classes.quantized.Conv3dPackedParamsBase")) ||
                    (type ==
                     getCustomClass(
                         "__torch__.torch.classes.quantized.LinearPackedParamsBase")),
                "Unknown type ",
                type->repr_str(),
                " encountered in handling model params. This type is not supported in ONNX export.");
            attrValues.emplace_back(
                script::Object(attr.toObject()).run_method("__getstate__"));
            paramConst = addParamAsArgument(function_, fullName, attr);
          } else if (attr.isNone() || name == "training") {
            auto attrVal = tryInsertConstant(*graph, attr);
            paramConst = *attrVal;
          }
          n->output()->replaceAllUsesWith(paramConst);
          n->removeAllInputs();

          GRAPH_UPDATE("Folding GetAttr %", n->outputs()[0]->debugName());
        }
      }
    }
  }
  return attrValues;
}

std::pair<Module, std::vector<IValue>> list_module_parameters(
    const Module& module) {
  Module moduleClone = module.clone(true);
  Method method = moduleClone.get_method("forward");
  auto function = &method.function();
  std::vector<IValue> modelParams;

  GRAPH_DEBUG("List attributes for function: " + function->name());
  auto graph = function->graph();
  // Add model_parameters and model_buffers as model inputs. Order is based on
  // the appearance in the graph.
  auto attributes = getParamAttributes(graph, moduleClone, function);

  modelParams.reserve(attributes.size());
  for (auto& attr_ : attributes) {
    modelParams.push_back(attr_);
  }
  GRAPH_DEBUG("Cleaning up module");
  EliminateDeadCode(graph->block());

  return std::make_pair(moduleClone, modelParams);
}

} // namespace jit
} // namespace torch
