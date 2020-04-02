#include <torch/csrc/jit/codegen/cuda/parser.h>

#include <torch/csrc/jit/codegen/cuda/arith.h>
#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>
#include <torch/csrc/jit/codegen/cuda/tensor.h>
#include <torch/csrc/jit/frontend/function_schema_parser.h>
#include <torch/csrc/jit/ir/constants.h>

#include <unordered_map>

namespace torch {
namespace jit {

typedef Value JitValue;
typedef Node JitOp;

namespace fuser {
namespace cuda {

namespace {

typedef Val CgValue;
typedef Expr CgOp;

typedef void (
    *ParseFuncPtr)(const Node* const, std::unordered_map<size_t, CgValue*>&);

// TODO: add a mutex to make it thread safe.
class IrParser {
 public:
  IrParser(std::shared_ptr<Graph> graph, Fusion& fusion)
      : graph_(std::move(graph)), fusion_(&fusion) {
    if (init_registry_) {
      registerJitOperator();
      init_registry_ = false;
    }
  }

  void parse() {
    FusionGuard fg(fusion_);
    auto block = graph_->block();

    // register all inputs;
    // shape propagation during parsing is effctively done in parsing rules, as
    // we only explicitly register inputs in the graph.
    for (auto val : block->inputs()) {
      TORCH_CHECK(registerValue(val));
      fusion_->addInput(value_maps_[val->unique()]);
    }

    // compose nodes in topo order;
    for (const JitOp* node : block->nodes()) {
      processJitNode(node);
    }

    // mark output;
    for (auto jit_output : block->outputs()) {
      TensorView* out =
          static_cast<TensorView*>(value_maps_[jit_output->unique()]);
      fusion_->addOutput(out);

      // Merge all dimensions because we're only supporting pointwise
      while (out->nDims() > 1)
        out->merge(0);
      // Split into 128 so we can map blocks/threads
      out->split(0, 128);

      // Map blocks/threads
      out->axis(0)->parallelize(ParallelType::BIDx);
      out->axis(-1)->parallelize(ParallelType::TIDx);
    }

    for (auto jit_input : block->inputs()) {
      TensorView* inp =
          static_cast<TensorView*>(value_maps_[jit_input->unique()]);
      for (auto jit_output : block->outputs()) {
        TensorView* out =
            static_cast<TensorView*>(value_maps_[jit_output->unique()]);
        if (DependencyCheck::isDependencyOf(inp, out)) {
          inp->computeAt(out, -1);
          break;
        }
      }
    }
  }

  static bool canParseNode(const Node* const node) {
    if (init_registry_) {
      // TODO: mutex this guy;
      registerJitOperator();
      init_registry_ = false;
    }

    // match signature.
    auto iter = jit_operator_registry_.find(node->kind());
    if (iter == jit_operator_registry_.end()) {
      return false;
    }
    for (auto& pair_op_func : iter->second) {
      if (node->matches(pair_op_func.first->schema())) {
        return true;
      }
    }
    return false;
  }

  static void registerParseRule(
      std::shared_ptr<Operator>& op,
      ParseFuncPtr fn) {
    jit_operator_registry_[Symbol::fromQualString(op->schema().name())]
        .push_back(std::make_pair(op, fn));
  }

 protected:
  static void parseBinaryOpWithAlpha(
      const Node* const node,
      std::unordered_map<size_t, CgValue*>& value_maps) {
    static std::unordered_map<Symbol, BinaryOpType> op_mapping({
        {aten::add, BinaryOpType::Add},
        {aten::sub, BinaryOpType::Sub},
    });
    auto lhs = value_maps[node->inputs()[0]->unique()];
    auto rhs = value_maps[node->inputs()[1]->unique()];

    auto out = binaryOp(op_mapping[node->kind()], lhs, rhs);
    value_maps.emplace(node->output()->unique(), out);
  }

  static void parseBinaryOp(
      const Node* const node,
      std::unordered_map<size_t, CgValue*>& value_maps) {
    static std::unordered_map<Symbol, BinaryOpType> op_mapping({
        {aten::mul, BinaryOpType::Mul},
        {aten::div, BinaryOpType::Div},
    });
    auto lhs = value_maps[node->inputs()[0]->unique()];
    auto rhs = value_maps[node->inputs()[1]->unique()];

    auto out = binaryOp(op_mapping[node->kind()], lhs, rhs);
    value_maps.emplace(node->output()->unique(), out);
  }

  static void registerJitOperator() {
    // Register parse-function for each JIT operator;
    // This is a one-time look up, our hash registry indexes on the pointer in
    // OperatorRegistry.

    std::array<const char*, 4> BinaryOpWithAlpha = {
        "aten::add(Tensor self, Tensor other, *, Scalar alpha) -> Tensor",
        "aten::add(Tensor self, Scalar other, Scalar alpha) -> Tensor",
        "aten::sub(Tensor self, Tensor other, *, Scalar alpha) -> Tensor",
        "aten::sub(Tensor self, Scalar other, Scalar alpha) -> Tensor"};
    for (auto signature : BinaryOpWithAlpha) {
      auto ptr_op = getOperatorForLiteral(signature);
      registerParseRule(ptr_op, &parseBinaryOpWithAlpha);
    }

    std::array<const char*, 4> BinaryOp = {
        "aten::div(Tensor self, Tensor other) -> Tensor",
        "aten::div(Tensor self, Scalar other) -> Tensor",
        "aten::mul(Tensor self, Tensor other) -> Tensor",
        "aten::mul(Tensor self, Scalar other) -> Tensor"};
    for (auto signature : BinaryOp) {
      auto ptr_op = getOperatorForLiteral(signature);
      registerParseRule(ptr_op, &parseBinaryOp);
    }
  }

  void processJitNode(const JitOp* node) {
    if (node->kind() == prim::Constant) {
      // partition doesn't take constant node explicitly, but it does and copy
      // constant into subgraph. So we need to register constants in codegen IR;
      for (auto output : node->outputs()) {
        TORCH_CHECK(registerScalar(output));
      }
    } else {
      auto iter = IrParser::jit_operator_registry_.find(node->kind());
      // make sure we have a parser for the op;
      TORCH_CHECK(
          iter != IrParser::jit_operator_registry_.end(),
          "CudaFusionGroup Parser doesn't handle operator kind(): ",
          node->kind().toDisplayString());
      for (auto& pair_op_func : iter->second) {
        if (node->matches(pair_op_func.first->schema())) {
          pair_op_func.second(node, value_maps_);
          return;
        }
      }
      TORCH_CHECK(
          false,
          "CudaFusionGroup Parser doesn't recognize operator overload:",
          canonicalSchemaString(node->schema()));
    }
  }

  bool registerValue(const JitValue* val) {
    return registerTensor(val) || registerScalar(val);
  }

  bool registerScalar(const JitValue* val) {
    if (val->type()->isSubtypeOf(static_cast<c10::TypePtr>(FloatType::get()))) {
      CgValue* cg_val;
      if (auto ival = constant_as<float>(val)) {
        cg_val = new Float(ival.value());
      } else {
        cg_val = new Float();
      }
      value_maps_.emplace(val->unique(), cg_val);
      return true;
    } else if (val->type()->isSubtypeOf(
                   static_cast<c10::TypePtr>(IntType::get()))) {
      CgValue* cg_val;
      if (auto ival = constant_as<int>(val)) {
        cg_val = new Float(ival.value());
      } else {
        cg_val = new Int();
      }
      value_maps_.emplace(val->unique(), cg_val);
      return true;
    }
    return false;
  }

  bool registerTensor(const JitValue* val) {
    CgValue* cg_val;
    if (val->isCompleteTensor()) {
      // TODO: make this a static function in Tensor class;
      // create tensor;
      cg_val = new TensorView(val->type()->cast<TensorType>());
      value_maps_.emplace(val->unique(), cg_val);
      return true;
    }
    return false;
  }

  std::shared_ptr<Graph> graph_;
  Fusion* fusion_;

  // maps from JitValue::unique() to fusion Val;
  std::unordered_map<size_t, CgValue*> value_maps_;
  // parsing rule registry.
  static std::unordered_map<
      Symbol,
      std::vector<std::pair<std::shared_ptr<Operator>, ParseFuncPtr>>>
      jit_operator_registry_;
  static bool init_registry_;
};

std::unordered_map<
    Symbol,
    std::vector<std::pair<std::shared_ptr<Operator>, ParseFuncPtr>>>
    IrParser::jit_operator_registry_;
bool IrParser::init_registry_ = true;

} // namespace

bool isNodeParsible(const Node* const node) {
  return IrParser::canParseNode(node);
}

void parseJitIR(std::shared_ptr<Graph>& graph, Fusion& fusion) {
  IrParser parser(graph, fusion);
  parser.parse();
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
