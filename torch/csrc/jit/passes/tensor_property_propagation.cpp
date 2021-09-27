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
#include <stdexcept>
#include "jit/runtime/operator.h"

namespace torch {
namespace jit {

namespace {

using ArgumentCreator =  std::function<c10::optional<Stack>(Node*)>;

static OperatorMap<ArgumentCreator>&  getArgumentCreatorMap() {
  static ArgumentCreator defaultArgumentCreator = [](Node* n) -> c10::optional<Stack> {

    std::vector<IValue> stack;
    for (auto inp : n->inputs()) {      
        if (auto tp = inp->type()->cast<TensorType>()) {
          stack.push_back(at::empty({2}, at::TensorOptions(at::kMeta).dtype(*tp->scalarType())));
        } else if (inp->type() == FloatType::get()) {
          stack.push_back(0.);
        } else if (inp->type() == IntType::get()) {
          stack.push_back(0);
        } else if (inp->type() == BoolType::get()) {
          stack.push_back(false);
        }
    }
    return stack;
  };

  static OperatorMap<ArgumentCreator> ops {
    {"aten::add(Tensor self, Tensor other, *, Scalar alpha) -> Tensor", defaultArgumentCreator},
    {"aten::div(Tensor self, Tensor other) -> Tensor", defaultArgumentCreator},
    {"aten::mul(Tensor self, Tensor other) -> Tensor", defaultArgumentCreator},
    {"aten::add(Tensor self, Tensor other, *, Scalar alpha) -> Tensor", defaultArgumentCreator},
    // TODO: use mutation variant rules
    {"aten::add_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha) -> Tensor(a!)", defaultArgumentCreator},
    {"aten::sub(Tensor self, Tensor other, *, Scalar alpha) -> Tensor", defaultArgumentCreator},
    {"aten::div(Tensor self, Tensor other) -> Tensor", defaultArgumentCreator},
    {"aten::mul(Tensor self, Tensor other) -> Tensor", defaultArgumentCreator},
    {"aten::floor_divide(Tensor self, Tensor other) -> Tensor", defaultArgumentCreator},
    // TODO: use meta-tensor rules
    {"aten::addmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta, Scalar alpha) -> Tensor", defaultArgumentCreator},
    //
    {"aten::hardsigmoid(Tensor self) -> Tensor", defaultArgumentCreator},
    {"aten::hardswish(Tensor self) -> Tensor", defaultArgumentCreator},
    {"aten::hardtanh(Tensor self, Scalar min_val, Scalar max_val) -> Tensor", defaultArgumentCreator},
    {"aten::relu(Tensor self) -> Tensor", defaultArgumentCreator},
    {"aten::transpose.int(Tensor(a) self, int dim0, int dim1) -> Tensor(a)", defaultArgumentCreator},
    {"aten::transpose.Dimname(Tensor(a) self, Dimname dim0, Dimname dim1) -> Tensor(a)", defaultArgumentCreator},
    {"aten::view(Tensor(a) self, int[] size) -> Tensor(a)", defaultArgumentCreator},
    {"aten::flatten.using_ints(Tensor(a) self, int start_dim, int end_dim) -> Tensor(a)", defaultArgumentCreator},
    {"aten::flatten.named_out_dim(Tensor(a) self, int start_dim, int end_dim, Dimname out_dim) -> Tensor(a)", defaultArgumentCreator},
    {"aten::flatten.using_names(Tensor(a) self, Dimname start_dim, Dimname end_dim, Dimname out_dim) -> Tensor(a)", defaultArgumentCreator},
    {"aten::flatten.DimnameList(Tensor(a) self, Dimname[] dims, Dimname out_dim) -> Tensor(a)", defaultArgumentCreator},
    {"aten::max_pool2d(Tensor self, int[2] kernel_size, int[2] stride, int[2] padding, int[2] dilation, bool ceil_mode) -> Tensor", defaultArgumentCreator},
    {"aten::chunk(Tensor(a) self, int chunks, int dim) -> Tensor(a)[]", defaultArgumentCreator},
    {"aten::contiguous(Tensor(a) self, *, MemoryFormat memory_format) -> Tensor(a)", defaultArgumentCreator},
    {"aten::adaptive_avg_pool2d(Tensor self, int[2] output_size) -> Tensor", defaultArgumentCreator},
    {"aten::avg_pool2d(Tensor self, int[2] kernel_size, int[2] stride, int[2] padding, bool ceil_mode, bool count_include_pad, int? divisor_override) -> Tensor", defaultArgumentCreator},
    {"aten::cat(Tensor[] tensors, int dim) -> Tensor", defaultArgumentCreator},
    {"aten::expand_as(Tensor(a) self, Tensor other) -> Tensor(a)", defaultArgumentCreator},
    // TODO: need validation
    {"aten::batch_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float momentum, float eps, bool cudnn_enabled) -> Tensor", defaultArgumentCreator},
    {"aten::linear(Tensor input, Tensor weight, Tensor? bias) -> Tensor", defaultArgumentCreator},
    {"aten::matmul(Tensor self, Tensor other) -> Tensor", defaultArgumentCreator},
    {"aten::conv2d(Tensor input, Tensor weight, Tensor? bias, int[2] stride, int[2] padding, int[2] dilation, int groups) -> Tensor", defaultArgumentCreator},
    {"aten::conv2d.padding(Tensor input, Tensor weight, Tensor? bias, int[2] stride, str padding, int[2] dilation, int groups) -> Tensor", defaultArgumentCreator},
    // TODO: use out variant rules
    {"aten::adaptive_avg_pool2d.out(Tensor self, int[2] output_size, *, Tensor(a!) out) -> Tensor(a!)", defaultArgumentCreator},
    {"aten::avg_pool2d.out(Tensor self, int[2] kernel_size, int[2] stride, int[2] padding, bool ceil_mode, bool count_include_pad, int? divisor_override, *, Tensor(a!) out) -> Tensor(a!)", defaultArgumentCreator},
    {"aten::addmm.out(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta, Scalar alpha, Tensor(a!) out) -> Tensor(a!)", defaultArgumentCreator},
    {"aten::matmul.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)", defaultArgumentCreator},
    {"aten::cat.out(Tensor[] tensors, int dim, *, Tensor(a!) out) -> Tensor(a!)", defaultArgumentCreator},
    //
    {"aten::neg(Tensor self) -> Tensor", defaultArgumentCreator}
  };

  return ops;
}

bool TryRegisterArgumentCreatorFor(const char* schema, ArgumentCreator creator) {
  static std::mutex mut;
  std::lock_guard<std::mutex> lock(mut);
  auto op = getOperatorForLiteral(schema);
  if (!op) {
    return false;
  }

  auto& map = getArgumentCreatorMap();
  if (map.contains(*op)) {
    return false;
  }
  getArgumentCreatorMap().insert(op, creator);
  return true;
}

void RegisterArgumentCreatorFor(const char* schema, ArgumentCreator creator) {

  if (!TryRegisterArgumentCreatorFor(schema, creator)) {
    throw std::runtime_error(std::string("Couldn't register ArgumentCreator for ") + schema);
  }
}

static bool canBeInferredWithMetaTensor(Node* n) {
  auto opt_op = n->maybeOperator();
  GRAPH_DEBUG("Checking ", getHeader(n));
  if (!opt_op) {
    GRAPH_DEBUG("not registered with Meta");
    return false;
  }
  GRAPH_DEBUG("not registered with Meta");
  return getArgumentCreatorMap().contains(*opt_op);
}

c10::optional<c10::ScalarType> inferWithMetaTensor(Node* n) {

    GRAPH_DEBUG("inferWithMetaTensor");
    auto argument_creator = *getArgumentCreatorMap().find(n->getOperator());
    bool args_have_dtypes = std::all_of(n->inputs().begin(), n->inputs().end(), [](Value* v) { return !v->type()->cast<TensorType>() || v->type()->expect<TensorType>()->scalarType().has_value(); });
    if (!args_have_dtypes) {
      return c10::nullopt;
    }
    Stack stack = *argument_creator(n);
    auto op = n->getOperation();
    try {
      GRAPH_DEBUG("Running op for ", getHeader(n));
      op(&stack);
      GRAPH_DEBUG("op run successfully", getHeader(n));
      GRAPH_DEBUG("Received ", toString(stack.back().toTensor().scalar_type()));
      GRAPH_DEBUG("After receive!");
      return stack.back().toTensor().scalar_type();

    }
    catch(...) {
      GRAPH_DEBUG("caught exception!");
    };
    return c10::nullopt;
}


class TensorPropertyInferrer {
  public:
  
  TensorPropertyInferrer(OperatorMap<ArgumentCreator> om): ops_to_args_(om) {}
  bool virtual hasProperty(Value* v) = 0;
  void virtual setType(Value* v, const IValue& ival) { throw std::runtime_error("setType(Value*, IValue&) not yet implemented "); }
  void virtual setType(Value* dst, const Value* src) { throw std::runtime_error("setType(Value*, Value*) not yet implemented "); }
  bool supportsNodeWithMeta(Node* n) {
    auto opt_op = n->maybeOperator();
    GRAPH_DEBUG("Checking ", getHeader(n));
    if (!opt_op) {
      GRAPH_DEBUG("not registered with Meta");
      return false;
    }
    GRAPH_DEBUG("not registered with Meta");
    return ops_to_args_.contains(*opt_op);
  }

  bool virtual processNodeWithOther(Node*) {return false; };
  bool virtual processValueWithCopy(Value* src, Value* dst) {
    if (!hasProperty(src)) {
      return false;
    }

    setType(dst, src);
    return true;
  }

  bool virtual processNodeWithMeta(Node* n) {
    
    auto argument_creator = *ops_to_args_.find(n->getOperator());
    bool args_have_dtypes = std::all_of(n->inputs().begin(), n->inputs().end(), [this](Value* v) { return this->hasProperty(v); });
    if (!args_have_dtypes) {
      return false;
    }

    Stack stack = *argument_creator(n);
    auto op = n->getOperation();
    try {
      op(&stack);
      const auto num_inputs = n->outputs().size();
      for (auto i: c10::irange(num_inputs)) {
        setType(n->output(i), peek(stack, i, num_inputs));
      }
      return true;
    }
    catch(...) {
      GRAPH_DEBUG("caught exception!");
      return false;
    };

  }
  OperatorMap<ArgumentCreator> ops_to_args_;
};

class ScalarTypeInferrer: public TensorPropertyInferrer  {

public:

  using TensorPropertyInferrer::TensorPropertyInferrer;

  bool hasProperty(Value* v) override {
    return !v->type()->cast<TensorType>() || v->type()->expect<TensorType>()->scalarType().has_value();
  }

  void setType(Value* value, const IValue& ival) override {
    auto scalarType = ival.toTensor().scalar_type();
    _setType(value, scalarType);
  }

  void setType(Value* dst, const Value* src) override {
    auto scalarType = *src->type()->expect<TensorType>()->scalarType();
    _setType(dst, scalarType);
  }

private:

  void _setType(Value* value, ScalarType scalarType) {
    auto tensor_type = value->type()->cast<TensorType>();
    TORCH_INTERNAL_ASSERT(tensor_type, "Expecting a tensor type");
    if (!tensor_type->scalarType().has_value()) {
      value->setType(tensor_type->withScalarType(scalarType));
    } else if (tensor_type->scalarType().value() != scalarType) {
      tensor_type->withScalarType({});
    }
  }


};

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
  TensorPropertyPropagationPass(std::shared_ptr<Graph> graph, TensorPropertyInferrer* tpi)
      : graph_(std::move(graph)),
      node_type_inferrer_(tpi) {
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
      GRAPH_DEBUG("getScalarType = ", stype.value());
    }
    return stype;
  }

  bool processBlocks(at::ArrayRef<Block*> blocks) {
    GRAPH_DEBUG("processBlocks");
    bool changed = false;
    for (auto block : blocks) {
      changed |= processBlock(block);
    }
    return changed;
  }

  bool processBlock(Block* block) {
    GRAPH_DEBUG("processBlock");
    bool changed = false;
    for (auto it = block->nodes().begin(); it != block->nodes().end(); it++) {
      changed |= processNode(*it);
    }
    return changed;
  }

  bool processNode(Node* n) {
    GRAPH_DEBUG("processNode");
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
    TORCH_INTERNAL_ASSERT(oneList.size() == anotherList.size());
    for (auto i: c10::irange(oneList.size())) {
      
    }

    return false;
  }

  bool processIf(Node* node) {
    GRAPH_DEBUG("processIf");
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
    GRAPH_DEBUG("processAtenOps");

    auto schema_opt = n->maybeSchema();
    if (!schema_opt || !checkSchemaReturnsTensors(schema_opt)) {
      GRAPH_DEBUG("schema not found or op does not return tensors");
      return false;
    }

    GRAPH_DEBUG("case = ", n->kind(), " ", *n);
    bool changed = false;
    bool found = false;
    if (node_type_inferrer_->supportsNodeWithMeta(n)) {
      changed = node_type_inferrer_->processNodeWithMeta(n);
      TORCH_INTERNAL_ASSERT(changed);
      found = true;
    } else {
      switch (n->kind()) {
        case aten::append:
          // auto elementType = n->input()->type()->expect<ListType>()->containedTypes()[0];
          // if (auto itp  = elementType->cast<TensorType>()) {
          //   //itp->
          //   auto otp = elementType->cast<TensorType>()->containedTypes()[0];
          //   ListType::create(otp->withScalarType(i))
          // }
          break;
        default:
            TORCH_INTERNAL_ASSERT(false);
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
  std::shared_ptr<TensorPropertyInferrer> node_type_inferrer_ = nullptr;
};

} // anonymous namespace

// This analysis propagates input tensor properties (if any) throughout the
// graph. Currently only support dtype propagation.
void TensorPropertyPropagation(std::shared_ptr<Graph>& graph) {
  ScalarTypeInferrer* mtp = new ScalarTypeInferrer(getArgumentCreatorMap());
  TensorPropertyPropagationPass tp = TensorPropertyPropagationPass(graph, mtp);
  bool changed = tp.run();
  if (changed) {
    GRAPH_DUMP("After TensorPropertyPropagation pass:", graph);
  }
}

} // namespace jit
} // namespace torch
