#include <ATen/core/function_schema.h>
#include <ATen/core/jit_type.h>
#include <c10/core/ScalarType.h>
#include <c10/util/Optional.h>
#include <torch/csrc/jit/ir/alias_analysis.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/tensor_property_propagation.h>
#include <torch/library.h>
#include <torch/types.h>

namespace torch {
namespace jit {

namespace {

#define TRACE_EXEC(msg, ...) \
  std::cout << "TRACE_EXEC:" << msg << c10::str(__VA_ARGS__) << std::endl;

// TensorPropertyPropagationPass is an analysis pass that walks through a graph
// in topological order and forward propagate TensorProperties from graph inputs
// (expressed in input_descriptors) to all output tensor nodes in the graph.
// The inferred TensorProperties of an output tensor will be checked against
// the original TensorProperties of the tensor node:
//  - if inferred property is incongruent with the original property, an error
//  is issued
//  - otherwise if the inferred property is more precise, original property of
//  the output tensor will be updated
struct TensorPropertyPropagationPass {
  TensorPropertyPropagationPass(std::shared_ptr<Graph> graph)
      : graph_(std::move(graph)) {}

  // returns true if at least one node has its scalar type set on a tensor node
  bool run() {
    return processBlocks(graph_->block());
  }

 private:
  c10::optional<at::ScalarType> getScalarType(const Value* value) const {
    c10::optional<at::ScalarType> stype;
    auto type = value->type();
    if (type->kind() == TypeKind::TensorType) {
      stype = type->castRaw<TensorType>()->scalarType();
    } else {
      stype = tryScalarTypeFromJitType(type);
    }
    if (stype.has_value()) {
      TRACE_EXEC("getScalarType = ", stype.value());
    }
    return stype;
  }

  // Set scalar type for value (of Tensor type) if its scalar type
  // is not yet specified; otherwise report error if scalarType
  // differs from value's scalar type
  bool setTensorScalarType(Value* value, ScalarType scalarType) {
    bool changed = false;
    auto tensor_type = value->type()->cast<TensorType>();
    TORCH_INTERNAL_ASSERT(tensor_type, "Expecting a tensor type");
    if (!tensor_type->scalarType().has_value()) {
      value->setType(tensor_type->withScalarType(scalarType));
      changed = true;
    } else if (tensor_type->scalarType().value() != scalarType) {
      TORCH_INTERNAL_ASSERT(
          false,
          "scalar type mismatch: t1 = ",
          scalarType,
          " t2=",
          tensor_type->scalarType().value());
    }
    return changed;
  }

  bool processBlocks(at::ArrayRef<Block*> blocks) {
    TRACE_EXEC("processBlocks");
    bool changed = false;
    for (auto block : blocks) {
      changed |= processBlock(block);
    }
    return changed;
  }

  bool processBlock(Block* block) {
    TRACE_EXEC("processBlock");
    bool changed = false;
    for (auto it = block->nodes().begin(); it != block->nodes().end(); it++) {
      changed |= processNode(*it);
    }
    return changed;
  }

  bool processNode(Node* n) {
    TRACE_EXEC("processNode");
    // process block nodes
    switch (n->kind()) {
      case prim::If:
        return processIf(n);
      case prim::Loop:
        return processLoop(n);
      case prim::CallMethod:
      case prim::CallFunction:
        return processCall(n);
      default:
        break;
    }

    TORCH_INTERNAL_ASSERT(
        n->blocks().empty(), "Do not handle this block structure");

    // process non-block nodes
    bool has_tensor_output = false;
    for (size_t i = 0; i < n->outputs().size(); i++) {
      auto type = n->output(i)->type();
      if (auto tt = type->castRaw<TensorType>()) {
        has_tensor_output = true;
        break;
      }
    }
    // if output contains no tensor, nothing to propagate
    if (!has_tensor_output) {
      return false;
    }

    bool changed = false;
    // Main dispatch loop for the abstract interpreter
    switch (n->kind()) {
      case prim::Constant:
        changed = processConstant(n);
        break;
      case aten::add:
      case aten::add_:
      case aten::conv2d:
      case aten::append:
      case aten::view:
        changed = processAtenOps(n);
        break;
      case prim::ListConstruct:
        changed = processListConstruct(n);
        break;
      default:
        TORCH_INTERNAL_ASSERT(false, "not supported IR");
    }
    return changed;
  }

  bool processConstant(Node* n) {
    TORCH_INTERNAL_ASSERT(false, "Constant not supported yet");
    return false;
  }

  // TODO: implement the merge function
  bool mergeTensorProperties(
      const at::ArrayRef<Value*>& oneList,
      const at::ArrayRef<Value*>& anotherList) {
    TORCH_INTERNAL_ASSERT(false, "Not implemented yet");
  }

  bool processIf(Node* node) {
    TRACE_EXEC("processIf");
    bool changed = false;
    auto blocks = node->blocks();
    auto true_block = blocks.at(0);
    auto false_block = blocks.at(1);

    changed |= processBlock(true_block);
    changed |= processBlock(false_block);

    changed |=
        mergeTensorProperties(true_block->outputs(), false_block->outputs());

    return changed;
  }

  bool processLoop(Node* n) {
    TORCH_INTERNAL_ASSERT(false, "Loop not supported yet");
    return false;
  }

  bool processCall(Node* n) {
    TORCH_INTERNAL_ASSERT(false, "Call not supported yet");
    return false;
  }

  bool processListConstruct(Node* n) {
    TORCH_INTERNAL_ASSERT(false, "ListConstruct not supported yet");
    return false;
  }

  // simpleTypeTransferFunction returns the type that is common among all input
  // scalar type if promoteToCommonType is true, will promote differing types to
  // a common type and return
  //
  // Examples:
  //    input types = (float, int, int)
  //    when promoteToCommonType == true: output type = float
  //    when promoteToCommonType == false: output type = nullopt
  c10::optional<ScalarType> simpleTypeTransferFunction(
      Node* n,
      bool promoteType = false) {
    auto scalarType = getScalarType(n->inputs().at(0));
    if (!scalarType.has_value()) {
      return nullopt;
    }

    auto stype = scalarType.value();
    for (size_t i = 1; i < n->inputs().size(); i++) {
      auto input = n->inputs().at(i);
      auto t = getScalarType(input);
      if (!t.has_value()) {
        return nullopt;
      }
      auto ttype = t.value();
      if (ttype != stype) {
        if (promoteType) {
          stype = c10::promoteTypes(stype, ttype);
          if (stype == ScalarType::Undefined) {
            return nullopt;
          }
        } else {
          return nullopt;
        }
      }
    }
    TRACE_EXEC("SimpleTypeTransferFunction: result = ", stype);
    return stype;
  }

  // This transfer function uses the scalar type of one input as that of the
  // output slected_input: the index of the input Value* whose scalar type will
  // be used for the output
  c10::optional<ScalarType> useInputScalarTypeTransferFunction(
      Node* n,
      int selected_input) {
    TORCH_INTERNAL_ASSERT(false, "not implemented");
  }

  bool checkSchemaReturnsTensors(const c10::FunctionSchema* schema) {
    const std::vector<Argument>& return_args = schema->returns();
    bool has_tensor_output = false;
    for (size_t i = 0; i < return_args.size(); i++) {
      auto arg = return_args[i];
      if (auto tt = arg.type()->castRaw<TensorType>()) {
        has_tensor_output = true;
        break;
      }
    }
    return has_tensor_output;
  }

  // TODO: keep a cache of valid schema to avoid repeated schema checking
  // for efficiency
  bool processAtenOps(Node* n) {
    TRACE_EXEC("processAtenOps");

    auto schema_opt = n->maybeSchema();
    if (!schema_opt || !checkSchemaReturnsTensors(schema_opt)) {
      TRACE_EXEC("schema not found or op does not return tensors");
      return false;
    }

    bool changed = false;
    c10::optional<ScalarType> scalarType;
    TRACE_EXEC("case = ", n->kind(), " ", *n);
    switch (n->kind()) {
      case aten::add: {
        if (n->matches(
                "aten::add(Tensor self, Tensor other, *, Scalar alpha) -> Tensor")) {
          scalarType = simpleTypeTransferFunction(n, true);
        } else {
          TORCH_INTERNAL_ASSERT(false, "add other schema not supported yet");
        }
        break;
      }
      case aten::add_:
        TORCH_INTERNAL_ASSERT(false, "add_ not supported yet");
        break;
      case aten::conv2d:
        TORCH_INTERNAL_ASSERT(false, "conv2d not supported yet");
        break;
      default:
        TORCH_INTERNAL_ASSERT(false, "AtenOps not supported yet");
    }

    if (scalarType.has_value()) {
      TORCH_INTERNAL_ASSERT(
          n->outputs().size() == 1, "Only handle a single output");
      changed = setTensorScalarType(n->output(0), scalarType.value());
    }
    return changed;
  }

  AliasDb* getOrCreateAliasDb() {
    if (!aliasDb_) {
      aliasDb_ = std::make_unique<AliasDb>(graph_);
    }
    return aliasDb_.get();
  }

  std::shared_ptr<Graph> graph_;
  // lazily initialized if using aliasing_types, otherwise not initialized
  std::unique_ptr<AliasDb> aliasDb_ = nullptr;

  // stores inferred tensor properties
  // SymbolicTensorPropertiesMap value_to_tensor_properties_map_;
  // const SymbolicTensorPropertiesMap & input_value_to_tensor_properties_map_;
};

} // anonymous namespace

// This analysis propagates input tensor properties (if any) throughout the
// graph. Currently only support dtype propagation.
void TensorPropertyPropagation(std::shared_ptr<Graph>& graph) {
  TensorPropertyPropagationPass tp = TensorPropertyPropagationPass(graph);
  bool changed = tp.run();
  if (changed) {
    GRAPH_DUMP("After TensorPropertyPropagation pass:", graph);
  }
}

} // namespace jit
} // namespace torch
