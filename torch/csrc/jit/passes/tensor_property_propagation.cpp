#include <ATen/core/function_schema.h>
#include <ATen/core/interned_strings.h>
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

using dtype_func_t = std::function<c10::optional<ScalarType>(Node*)>;
static std::vector<std::pair<OperatorSet, dtype_func_t>>
    dtype_transfer_functions;

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
      : graph_(std::move(graph)) {
    // FIXME (penguin): multi-threading? how do I initialize things once for
    // all?
    if (dtype_transfer_functions.empty()) {
      initializeTransferFunctions();
    }
  }

  // returns true if at least one node has its scalar type set on a tensor node
  bool run() {
    return processBlocks(graph_->block());
  }

 private:
  static c10::optional<at::ScalarType> getScalarType(const Value* value) {
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
    switch (n->kind()) {
      case prim::If:
        return processIf(n);
      case prim::RaiseException:
        return processRaiseException(n);
      case prim::Loop:
      case prim::CallMethod:
      case prim::CallFunction:
        TORCH_INTERNAL_ASSERT(false, "Loop/Call not handled now");
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
      case prim::ListConstruct:
        changed = processListConstruct(n);
        break;
      case prim::ListUnpack:
        changed = processListUnpack(n);
        break;
      default:
        if (n->kind().is_aten()) {
          changed = processAtenOps(n);
        } else {
          TORCH_INTERNAL_ASSERT(false, "not supported IR");
        }
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

  bool processRaiseException(Node* n) {
    TORCH_INTERNAL_ASSERT(false, "RaiseException not supported yet");
    return false;
  }

  bool processListConstruct(Node* n) {
    TORCH_INTERNAL_ASSERT(false, "ListConstruct not supported yet");
    return false;
  }

  bool processListUnpack(Node* n) {
    TORCH_INTERNAL_ASSERT(false, "ListUnpack not supported yet");
    return false;
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

    TRACE_EXEC("case = ", n->kind(), " ", *n);

    bool changed = false;
    bool found = false;
    c10::optional<ScalarType> scalarType;
    switch (n->kind()) {
      case aten::eq:
      case aten::lt:
      case aten::gt:
      case aten::ne:
        scalarType = kBool;
        found = true;
        break;
      case aten::dim:
        scalarType = kInt64;
        found = true;
        break;
      case aten::size:
      case aten::len:
        scalarType = kInt;
        found = true;
        break;
      case aten::format: // note: returns string but ScalarType does not include
                         // string
        break;
      case aten::append:
        // TODO: add
        break;
      default:
        for (auto& entry : dtype_transfer_functions) {
          if (n->isMemberOf(entry.first)) {
            scalarType = entry.second(n);
            found = true;
            break;
          }
        }
    }
    if (!found) {
      TORCH_INTERNAL_ASSERT(false, "schema not supported yet: ", schema_opt);
    } else {
      if (scalarType.has_value()) {
        TORCH_INTERNAL_ASSERT(
            n->outputs().size() == 1, "Only handle a single output");
        changed = setTensorScalarType(n->output(0), scalarType.value());
      }
    }
    return changed;
  }

  AliasDb* getOrCreateAliasDb() {
    if (!aliasDb_) {
      aliasDb_ = std::make_unique<AliasDb>(graph_);
    }
    return aliasDb_.get();
  }

  // simpleTypeTransferFunction returns the type that is common among all input
  // scalar type if promoteToCommonType is true, will promote differing types to
  // a common type and return
  //
  // Examples:
  //    input types = (float, int, int)
  //    when promoteToCommonType == true: output type = float
  //    when promoteToCommonType == false: output type = nullopt
  static c10::optional<ScalarType> simpleTypeTransferFunction(
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

  // This transfer function returns the dtype of the <idx>th operand
  static c10::optional<ScalarType> typeOfNthOperand(Node* n, int idx) {
    auto stype = getScalarType(n->inputs().at(idx));
    if (stype.has_value()) {
      TRACE_EXEC("typeOfNthOperand: result = ", stype.value());
    }
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

  // Initialize all transfer functions for OperatorSet
  static void initializeTransferFunctions() {
    // transfer functions take a node as input and propagate properties to the
    // outputs of the node, return true if output node properties are updated.
    struct register_transfer_func_for {
      register_transfer_func_for(OperatorSet operators, dtype_func_t tfunc) {
        dtype_transfer_functions.emplace_back(
            std::move(operators), std::move(tfunc));
      }
    };

    static const register_transfer_func_for simple_ops_with_common_type_promotion{
        {
            "aten::add(Tensor self, Tensor other, *, Scalar alpha) -> Tensor",
            // TODO: use mutation variant rules
            "aten::add_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha) -> Tensor(a!)",
            "aten::sub(Tensor self, Tensor other, *, Scalar alpha) -> Tensor",
            "aten::div(Tensor self, Tensor other) -> Tensor",
            "aten::mul(Tensor self, Tensor other) -> Tensor",
            "aten::floor_divide(Tensor self, Tensor other) -> Tensor",
            // TODO: use meta-tensor rules
            "aten::addmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta, Scalar alpha) -> Tensor",
        },
        [](Node* node) -> c10::optional<ScalarType> {
          return simpleTypeTransferFunction(node, true);
        }};

    // simply return the dtype of the 1st operand
    static const register_transfer_func_for ops_return_with_first_operand_type{
        {
            "aten::hardsigmoid(Tensor self) -> Tensor",
            "aten::hardswish(Tensor self) -> Tensor",
            "aten::hardtanh(Tensor self, Scalar min_val, Scalar max_val) -> Tensor",
            "aten::relu(Tensor self) -> Tensor",
            "aten::transpose.int(Tensor(a) self, int dim0, int dim1) -> Tensor(a)",
            "aten::transpose.Dimname(Tensor(a) self, Dimname dim0, Dimname dim1) -> Tensor(a)",
            "aten::view(Tensor(a) self, int[] size) -> Tensor(a)",
            "aten::flatten.using_ints(Tensor(a) self, int start_dim, int end_dim) -> Tensor(a)",
            "aten::flatten.named_out_dim(Tensor(a) self, int start_dim, int end_dim, Dimname out_dim) -> Tensor(a)",
            "aten::flatten.using_names(Tensor(a) self, Dimname start_dim, Dimname end_dim, Dimname out_dim) -> Tensor(a)",
            "aten::flatten.DimnameList(Tensor(a) self, Dimname[] dims, Dimname out_dim) -> Tensor(a)",
            "aten::max_pool2d(Tensor self, int[2] kernel_size, int[2] stride, int[2] padding, int[2] dilation, bool ceil_mode) -> Tensor",
            "aten::chunk(Tensor(a) self, int chunks, int dim) -> Tensor(a)[]",
            "aten::contiguous(Tensor(a) self, *, MemoryFormat memory_format) -> Tensor(a)",
            "aten::adaptive_avg_pool2d(Tensor self, int[2] output_size) -> Tensor",
            "aten::avg_pool2d(Tensor self, int[2] kernel_size, int[2] stride, int[2] padding, bool ceil_mode, bool count_include_pad, int? divisor_override) -> Tensor",
            "aten::cat(Tensor[] tensors, int dim) -> Tensor",
            "aten::expand_as(Tensor(a) self, Tensor other) -> Tensor(a)",
            // TODO: need validation
            "aten::batch_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float momentum, float eps, bool cudnn_enabled) -> Tensor",
            "aten::linear(Tensor input, Tensor weight, Tensor? bias) -> Tensor",
            "aten::matmul(Tensor self, Tensor other) -> Tensor",
            "aten::conv2d(Tensor input, Tensor weight, Tensor? bias, int[2] stride, int[2] padding, int[2] dilation, int groups) -> Tensor",
            "aten::conv2d.padding(Tensor input, Tensor weight, Tensor? bias, int[2] stride, str padding, int[2] dilation, int groups) -> Tensor",
            // TODO: use out variant rules
            "aten::adaptive_avg_pool2d.out(Tensor self, int[2] output_size, *, Tensor(a!) out) -> Tensor(a!)",
            "aten::avg_pool2d.out(Tensor self, int[2] kernel_size, int[2] stride, int[2] padding, bool ceil_mode, bool count_include_pad, int? divisor_override, *, Tensor(a!) out) -> Tensor(a!)",
            "aten::addmm.out(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta, Scalar alpha, Tensor(a!) out) -> Tensor(a!)",
            "aten::matmul.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)",
            "aten::cat.out(Tensor[] tensors, int dim, *, Tensor(a!) out) -> Tensor(a!)",
        },
        [](Node* node) -> c10::optional<ScalarType> {
          return typeOfNthOperand(node, 0);
        }};
  }

  /*
  // This one is a special rule -- mean take the ScalarType if specified,
  otherwise Tensor type
  // This is an example where the dtype rule has to be extracted from the
  implementation
  "aten::mean(Tensor self, *, ScalarType? dtype) -> Tensor",
  */

  std::shared_ptr<Graph> graph_;
  // lazily initialized if using aliasing_types, otherwise not initialized
  std::unique_ptr<AliasDb> aliasDb_ = nullptr;
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
