#include <torch/csrc/jit/passes/onnx/list_model_parameters.h>
#include <torch/csrc/jit/frontend/error_report.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/onnx/helper.h>

namespace torch {
namespace jit {

namespace onnx {
using namespace ::c10::onnx;
}

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

std::unordered_map<std::string, Value*> attrValues = {};
std::unordered_map<std::string, Value*> setAttrValues = {};

std::vector<IValue> parameterIValues = {};
std::vector<Node*> toDestory = {};

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
    } else {
      return moduleNames;
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

Node* insertCloneBeforeNode(
    std::shared_ptr<Graph> graph,
    Value* orig_data,
    Node* node) {
  auto* noneNode = graph->create(prim::Constant);
  noneNode->output()->setType(NoneType::get());
  auto cloneNode = graph->create(aten::clone, /*num_outputs =*/1);
  cloneNode->addInput(orig_data);

  cloneNode->addInput(noneNode->output());
  cloneNode->output()->copyMetadata(orig_data);

  cloneNode->insertBefore(node);
  noneNode->insertBefore(cloneNode);
  return cloneNode;
}

Value* registerSetAttrInIfBlocks(std::shared_ptr<Graph> graph, Node* n) {
  if (!n->owningBlock()->owningNode() ||
      n->owningBlock()->owningNode()->kind() != prim::If)
    return nullptr;

  WithInsertPoint guard(n);

  auto orig_data = n->inputs().at(1);
  auto cloneNode = insertCloneBeforeNode(graph, orig_data, n);
  auto outer_block = cloneNode->owningBlock();
  auto next_node = outer_block->owningNode();

  RegisterInplaceNodeInIfBlocks(
      orig_data, cloneNode->output(), cloneNode, outer_block, next_node);
  return outer_block->owningNode()->output();
}

void getParamAttributes(
    Block* block,
    std::shared_ptr<Graph>& graph,
    const Module& module_,
    Function* function_) {
  auto isEval = !module_.hasattr("training") || !module_.is_training();

  Node* m = *block->nodes().begin();
  WithInsertPoint guard(m);

  for (auto it = block->nodes().begin(); it != block->nodes().end();) {
    Node* n = *it;
    it++; // node n can be destroyed

    if (n->kind() == prim::GetAttr || n->kind() == prim::SetAttr) {
      if (n->kind() == prim::GetAttr) {
        for (auto use : n->output()->uses()) {
          if (use.user->kind() == prim::PythonOp)
            throw ErrorReport(n->sourceRange())
                << "Couldn't export Python method.";
        }
      }

      auto name = n->s(attr::name);
      auto attrModule = module_;
      auto input = n->inputs()[0];

      auto moduleNames = findSubModuleAttr(input, name, attrModule, graph);
      if (!attrModule.hasattr(name))
        continue;
      auto attr = attrModule.attr(name);
      Value* paramConst = nullptr;

      std::string fullName("");
      for (auto& name : moduleNames) {
        fullName += name + '.';
      }
      fullName += name;

      auto type = attrModule.type();
      auto slot = *type->findAttributeSlot(name);

      // Add model_parameters and model_buffers as model inputs. Order is based
      // on
      // the appearance in the graph.
      if (type->is_parameter(slot) || type->is_buffer(slot) ||
          (attr.isObject() && !attr.toObjectRef().type()->is_module()) ||
          attr.isBool()) {
        if (attrValues.find(fullName) != attrValues.end()) {
          paramConst = attrValues[fullName];
        } else if (attr.isTensor()) {
          TORCH_INTERNAL_ASSERT(attr.isTensor());
          auto tensor_ = attr.toTensor();
          if (isEval && tensor_.requires_grad()) {
            tensor_ = tensor_.detach();
            tensor_.set_requires_grad(false);
            attr = IValue(tensor_);
          }
          parameterIValues.emplace_back(attr.toTensor());
          paramConst = addParamAsArgument(function_, fullName, attr);
        } else if (attr.isObject() && !attr.toObjectRef().type()->is_module()) {
          // Only below registered torch classes are supported.
          try {
            parameterIValues.emplace_back(
                script::Object(attr.toObject()).run_method("__getstate__"));
            paramConst = addParamAsArgument(function_, fullName, attr);
          } catch (const std::exception&) {
            throw ErrorReport(n->sourceRange())
                << "Unknown type " << attr.type()->repr_str()
                << " encountered in handling model params. This class type does not extend __getstate__ method.";
          }
        } else if (attr.isNone() || attr.isBool()) { // TODO: Handle float/int
                                                     // attributes
          auto attrVal = tryInsertConstant(*graph, attr);
          paramConst = *attrVal;
        }

        attrValues.insert({fullName, paramConst});
      }

      if (n->kind() == prim::SetAttr) { // Handle SetAttr node
        if (n->s(attr::name) ==
            "num_batches_tracked") { // This attr is not used in ONNX
          toDestory.emplace_back(n);
          return;
        }
        if (attrModule.hasattr(name)) {
          // SetAttr writes a value to an attr. Keep this in the setAttrValues
          // map.
          setAttrValues[fullName] = n->inputs().at(1);
          if (auto block_output = registerSetAttrInIfBlocks(graph, n)) {
            setAttrValues[fullName] = block_output;
          }
          toDestory.emplace_back(n);
          GRAPH_UPDATE("Folding SetAttr node %", n->outputs()[0]->debugName());
        }
      } else if (n->kind() == prim::GetAttr) { // Handle GetAttr node

        if (setAttrValues.find(fullName) != setAttrValues.end()) {
          // Attr has been set earlier in the graph. Read its value from
          // setAttrValues map.
          auto set_attr_node_input = setAttrValues[fullName];

          if (set_attr_node_input->type()->kind() == TypeKind::ListType) {
            // Create an aten::list to clone the list in graph inputs
            auto newNode = graph->create(aten::list, /*num_outputs =*/1);
            newNode->addInput(set_attr_node_input);
            newNode->output()->copyMetadata(set_attr_node_input);
            newNode->insertBefore(n);
            n->output()->replaceAllUsesAfterNodeWith(n, newNode->output());
          } else if (
              set_attr_node_input->type()->kind() == TypeKind::TensorType) {
            // Create an aten::clone to clone the list in graph inputs
            auto cloneNode =
                insertCloneBeforeNode(graph, set_attr_node_input, n);
            n->output()->replaceAllUsesAfterNodeWith(n, cloneNode->output());
          }
          // TODO: Handle float/int attributes
          toDestory.emplace_back(n);

        } else if (paramConst) {
          // Attr has not been set earlier in the graph. Replace it with the
          // graph parameter if exists.
          n->output()->replaceAllUsesWith(paramConst);
          n->removeAllInputs();
          toDestory.emplace_back(n);
        }
        GRAPH_UPDATE("Folding GetAttr node %", n->outputs()[0]->debugName());
      }
    }
    for (Block* sub_block : n->blocks()) {
      getParamAttributes(sub_block, graph, module_, function_);
    }
  }
}

std::pair<Module, std::vector<IValue>> list_module_parameters(
    const Module& module) {
  Module moduleClone = module.clone(true);
  Method method = moduleClone.get_method("forward");
  auto function = &method.function();
  auto graph = function->graph();

  parameterIValues.clear();
  attrValues.clear();
  setAttrValues.clear();
  toDestory.clear();

  GRAPH_DEBUG("Fetch attributes for function: " + function->name());
  getParamAttributes(graph->block(), graph, moduleClone, function);

  for (Node* n : toDestory) {
    n->destroy();
  }

  GRAPH_DEBUG("Cleaning up module");
  EliminateDeadCode(graph->block());

  return std::make_pair(moduleClone, parameterIValues);
}

} // namespace jit
} // namespace torch
