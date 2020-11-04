#include <torch/csrc/jit/passes/onnx/list_model_parameters.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>

namespace torch {
namespace jit {

std::deque<std::string> names_;

// findSubModuleAttr function chases getAttr chains to locate the submodules.
// For example:
// module M {
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
void findSubModuleAttr(
    Value* input,
    std::string& name,
    Module& attrModule,
    std::shared_ptr<Graph>& graph) {
  Node* node = input->node();
  names_.clear();
  while (!(node->outputs()[0]->type() == graph->inputs()[0]->type())) {
    if (node->kind() == prim::GetAttr) {
      names_.push_front(node->s(attr::name));
      node = node->inputs()[0]->node();
    }
  }

  for (auto& moduleName : names_) {
    attrModule = attrModule.attr(moduleName).toModule();
  }
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
    Module module_,
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
      if (n->kind() == prim::SetAttr) {
        if (!(n->outputs().size())) {
          n->destroy();
        }
      } else if (n->kind() == prim::GetAttr) {
        auto name = n->s(attr::name);
        auto attrModule = module_;
        auto input = n->inputs()[0];

        findSubModuleAttr(input, name, attrModule, graph);

        TORCH_INTERNAL_ASSERT(attrModule.hasattr(name));
        Value* paramConst = nullptr;

        auto attr = attrModule.attr(name);

        std::string fullName("self_");
        for (auto& name : names_) {
          fullName += name + '_';
        }
        fullName += name;

        auto type = attrModule.type();
        auto slot = *type->findAttributeSlot(name);

        if (type->is_parameter(slot) || type->is_buffer(slot)) {
          if (type->is_parameter(slot) || type->is_buffer(slot)) {
            if (attr.isTensor()) {
              TORCH_INTERNAL_ASSERT(attr.isTensor());
              auto tensor_ = attr.toTensor();
              if (isEval && tensor_.requires_grad()) {
                tensor_ = tensor_.detach();
                tensor_.set_requires_grad(false);
                attr = IValue(tensor_);
              }
              attrValues.push_back(attr.toTensor());
              paramConst = addParamAsArgument(function_, fullName, attr);
            } else if (attr.isNone()) {
              auto attrVal = tryInsertConstant(*graph, attr);
              paramConst = *attrVal;
            }
            n->output()->replaceAllUsesWith(paramConst);
            n->removeAllInputs();

            GRAPH_UPDATE(
                "Folding GetAttr %",
                n->outputs()[0]->debugName(),
                " with ",
                paramConst->debugName());
          }
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
  std::unordered_set<Function*> preservedMethods_;
  preservedMethods_.insert(&method.function());

  std::vector<IValue> modelParams;
  for (auto function : preservedMethods_) {
    GRAPH_DEBUG("List attributes for function: " + function->name());
    auto graph = function->graph();
    auto attributes = getParamAttributes(graph, moduleClone, function);
    for (auto attr_ : attributes) {
      modelParams.push_back(attr_);
    }
    GRAPH_DEBUG("Cleaning up module");
    EliminateDeadCode(graph->block());
  }

  return std::make_pair(moduleClone, modelParams);
}

} // namespace jit
} // namespace torch
