#include <torch/csrc/jit/fuser/cuda/parser.h>
#include <torch/csrc/jit/constants.h>

#include <torch/csrc/jit/fuser/common/arith.h>
#include <torch/csrc/jit/fuser/common/tensor.h>
#include <torch/csrc/jit/fuser/common/iriostream.h>

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
      assert(registerValue(val));
      fusion_->addInput(value_maps_[val->unique()]);
      std::cout << "==== add fusion input: " << value_maps_[val->unique()] << std::endl;
    }

    // compose nodes in topo order;
    for (const JitOp* node : block->nodes()) {
      processJitNode(node);
    }

    // mark output;
    for (auto jit_output : block->outputs()) {
      TensorView* out = static_cast<TensorView*>(value_maps_[jit_output->unique()]);
      fusion_->addOutput(out);
      std::cout << "==== add fusion output: " << out << std::endl;
      
      //Merge all dimensions because we're only supporting pointwise
      while(out->domain()->size() > 1)
        merge(out, 0);
      //Split into 128 so we can map blocks/threads
      split(out, 0, 128);

      //Map blocks/threads
      out->domain()->axis(0)->parallelize(ParallelType::BIDx);
      out->domain()->axis(-1)->parallelize(ParallelType::TIDx);
      
    }

    for (auto jit_input : block->inputs()) {
      TensorView* inp = static_cast<TensorView*>(value_maps_[jit_input->unique()]);
      for (auto jit_output : block->outputs()) {
        TensorView* out = static_cast<TensorView*>(value_maps_[jit_output->unique()]);
        if(DependencyCheck::isDependencyOf(inp, out)){
          inp->computeAt(out, -1);
          break;
        }
      }
    }

  }

protected:

  void processJitNode(const JitOp* node) {

    static std::unordered_map<Symbol, BinaryOpType> binary_op_mapping({
      {aten::add, BinaryOpType::Add},
      {aten::sub, BinaryOpType::Sub},
      {aten::mul, BinaryOpType::Mul},
      {aten::div, BinaryOpType::Div},
    });
    if (binary_op_mapping.count(node->kind()) != 0) {
      auto lhs = value_maps_[node->inputs()[0]->unique()];
      auto rhs = value_maps_[node->inputs()[1]->unique()];

      auto out = binaryOp(binary_op_mapping[node->kind()], lhs, rhs);
      value_maps_.emplace(node->output()->unique(), out);

    } else if (node->kind() == prim::Constant) {
      // we should just ignore constant node;
      for (auto output : node->outputs()) {
        assert(registerScalar(output));
      }
    } else {
      assert(false);
    }
  }

  bool registerValue(const JitValue* val) {
    return registerTensor(val) || registerScalar(val);
  }

  bool registerScalar(const JitValue* val) {
    CgValue* cg_val;
    if (val->type()->isSubtypeOf(static_cast<c10::TypePtr>(FloatType::get()))) {
      if (auto ival = constant_as<float>(val)) {
        cg_val = new Float(ival.value());
      } else {
        cg_val = new Float();
      }
    } else if (val->type()->isSubtypeOf(static_cast<c10::TypePtr>(IntType::get()))) {
      if (auto ival = constant_as<int>(val)) {
        cg_val = new Float(ival.value());
      } else {
        cg_val = new Int();
      }
    } else {
      return false;
    }
    value_maps_.emplace(val->unique(), cg_val);
    return true;
  }

  bool registerTensor(const JitValue* val) {
    CgValue* cg_val;
    if (val->isCompleteTensor()) {
      // TODO: make this a static function in Tensor class;
      // create tensor;
      cg_val = new TensorView(new Tensor(val->type()->cast<TensorType>()));
    } else {
      return false;
    }
    value_maps_.emplace(val->unique(), cg_val);
    return true;
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
