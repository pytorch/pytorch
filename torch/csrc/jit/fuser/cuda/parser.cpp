#include <torch/csrc/jit/fuser/cuda/parser.h>
#include <torch/csrc/jit/constants.h>

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

class IrParser {
public:
  IrParser(std::shared_ptr<Graph> graph, Fusion& fusion)
  : graph_(std::move(graph)),
    fusion_(&fusion) {}

  void parse() {
    FusionGuard fg(fusion_);
    auto block = graph_->block();

    // register all inputs;
    for (auto val : block->inputs()) {
      registerValue(val);
    }

    // compose nodes in topo order;
    for (const JitOp* node : block->nodes()) {
      processJitNode(node);
    }

    // mark output;
    for (auto jit_output : block->outputs()) {
      printf("add output!\n");
      fusion_->addOutput(value_maps_[jit_output->unique()]);
    }
    std::cout << fusion_ << std::endl;
  }

protected:

  void processJitNode(const JitOp* node) {
    // register outputs;
    for (auto val : node->outputs()) {
      registerValue(val);
    }

    CgOp* cg_op;

    static std::unordered_map<Symbol, BinaryOpType> binary_op_mapping({
      {aten::add, BinaryOpType::Add},
      {aten::sub, BinaryOpType::Sub},
      {aten::mul, BinaryOpType::Mul},
      {aten::div, BinaryOpType::Div},
    });
    if (binary_op_mapping.count(node->kind()) != 0) {
      auto lhs = value_maps_[node->inputs()[0]->unique()];
      auto rhs = value_maps_[node->inputs()[1]->unique()];
      auto out = value_maps_[node->output()->unique()];
      cg_op = new BinaryOp(binary_op_mapping[node->kind()], out, lhs, rhs);
    } else if (node->kind() == prim::Constant) {
      // we should just ignore constant node;
    } else {
      assert(false);
    }
  }

  void registerValue(const JitValue* val) {
    CgValue* cg_val;
    if (val->isCompleteTensor()) {
      // TODO: make this a static function in Tensor class;
      // create tensor;
      printf("register tensor\n");
      cg_val = new Tensor(val->type()->cast<TensorType>());
    } else if (val->type()->isSubtypeOf(FloatType::get())) {
      printf("register float\n");
      if (auto ival = toIValue(val)) {
        cg_val = new Float(ival.value().to<float>());
      } else {
        cg_val = new Float();
      }
    } else if (val->type()->isSubtypeOf(IntType::get())) {
      printf("register int\n");
      if (auto ival = toIValue(val)) {
        cg_val = new Int(ival.value().to<int>());
      } else {
        cg_val = new Int();
      }
    } else {
      // error out!
      assert(false);
    }
    value_maps_.emplace(val->unique(), cg_val);
  }

  std::shared_ptr<Graph> graph_;
  Fusion* fusion_;

  // maps from JitValue::unique() to fusion Val;
  std::unordered_map<size_t, CgValue*> value_maps_;
};

} // namespace

void parseJitIR(std::shared_ptr<Graph>& graph, Fusion& fusion) {
  IrParser parser(graph, fusion);
  parser.parse();
}

}}}} // namespace torch::jit::fuser::cuda
