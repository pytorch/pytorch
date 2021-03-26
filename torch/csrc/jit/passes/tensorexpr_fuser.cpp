#include <torch/csrc/jit/passes/tensorexpr_fuser.h>

#include <ATen/record_function.h>
#include <c10/util/FunctionRef.h>
#include <torch/csrc/jit/codegen/fuser/interface.h>
#include <torch/csrc/jit/ir/alias_analysis.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/jit_opt_limit.h>
#include <torch/csrc/jit/passes/common_subexpression_elimination.h>
#include <torch/csrc/jit/passes/constant_pooling.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/pass_manager.h>
#include <torch/csrc/jit/passes/remove_redundant_profiles.h>
#include <torch/csrc/jit/passes/utils/subgraph_utils.h>
#include <torch/csrc/jit/runtime/custom_operator.h>
#include <torch/csrc/jit/runtime/graph_executor.h>
#include <torch/csrc/jit/runtime/operator_options.h>
#include <torch/csrc/jit/tensorexpr/kernel.h>
#include <torch/csrc/utils/memory.h>

// NOLINTNEXTLINE
C10_DEFINE_bool(
    torch_jit_disable_cat,
    false,
    "disable aten::cat in TE fusion groups");
namespace torch {
namespace jit {

static bool texpr_reductions_enabled = false;

bool isSupportedForBlock(Node* node) {
  switch (node->kind()) {
    case aten::add:
    case aten::mul:
      return true;
    default:
      return false;
  }
}

bool usedOnlyInSize(Value* v) {
  const auto& uses = v->uses();
  return std::all_of(uses.begin(), uses.end(), [](const Use& u) {
    return u.user->matches("aten::size(Tensor self) -> int[]");
  });
}

Value* broadcastSizes(at::ArrayRef<Value*> sizes, AliasDb* db) {
  AT_ASSERT(!sizes.empty());
  Graph* graph = sizes[0]->owningGraph();
  Node* broadcast_n =
      graph->insertNode(graph->create(prim::BroadcastSizes, sizes));
  broadcast_n->output()->setType(ListType::ofInts());
  db->createValue(broadcast_n->output());
  return broadcast_n->output();
}

namespace tensorexpr {

static const OperatorSet& supported_eltwise_set() {
  // clang-format off
  // breaks up the schema strings so they are no longer discoverable with ctrl-F
    static const OperatorSet supported_eltwise_set{
      "aten::add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor",
      "aten::add.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> Tensor",
      "aten::_cast_Float(Tensor self, bool non_blocking) -> Tensor",
      "aten::type_as(Tensor self, Tensor other) -> Tensor",
      "aten::sub.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor",
      "aten::sub.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> Tensor",
      "aten::mul.Tensor(Tensor self, Tensor other) -> Tensor",
      "aten::mul.Scalar(Tensor self, Scalar other) -> Tensor",
      "aten::div.Tensor(Tensor self, Tensor other) -> Tensor",
      "aten::div.Scalar(Tensor self, Scalar other) -> Tensor",
      "aten::eq.Tensor(Tensor self, Tensor other) -> Tensor",
      "aten::eq.Scalar(Tensor self, Scalar other) -> Tensor",
      "aten::ne.Tensor(Tensor self, Tensor other) -> Tensor",
      "aten::ne.Scalar(Tensor self, Scalar other) -> Tensor",
      "aten::ge.Tensor(Tensor self, Tensor other) -> Tensor",
      "aten::ge.Scalar(Tensor self, Scalar other) -> Tensor",
      "aten::gt.Tensor(Tensor self, Tensor other) -> Tensor",
      "aten::gt.Scalar(Tensor self, Scalar other) -> Tensor",
      "aten::le.Tensor(Tensor self, Tensor other) -> Tensor",
      "aten::le.Scalar(Tensor self, Scalar other) -> Tensor",
      "aten::lt.Tensor(Tensor self, Tensor other) -> Tensor",
      "aten::lt.Scalar(Tensor self, Scalar other) -> Tensor",
      "aten::pow.Tensor_Scalar(Tensor self, Scalar exponent) -> Tensor",
      // TODO: uncomment when we properly support pow
      // "aten::pow.Tensor_Tensor(Tensor self, Tensor exponent) -> Tensor",
      // "aten::pow.Scalar(Scalar self, Tensor exponent) -> Tensor",
      // TODO: support clamp_min, clamp_max
      "aten::clamp(Tensor self, Scalar? min=None, Scalar? max=None) -> Tensor",
      "aten::lerp.Scalar(Tensor self, Tensor end, Scalar weight) -> Tensor",
      "aten::lerp.Tensor(Tensor self, Tensor end, Tensor weight) -> Tensor",
      "aten::to.dtype(Tensor self, ScalarType dtype, bool non_blocking=False, bool copy=False, MemoryFormat? memory_format=None) -> Tensor",
      "aten::to.device(Tensor self, Device device, ScalarType dtype, bool non_blocking=False, bool copy=False, MemoryFormat? memory_format=None) -> Tensor",
      "aten::to.dtype_layout(Tensor self, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None"
      ", bool? pin_memory=None, bool non_blocking=False, bool copy=False, MemoryFormat? memory_format=None) -> Tensor",
      "aten::to.prim_Device(Tensor(a) self, Device? device, int? dtype=None, bool non_blocking=False, bool copy=False) -> Tensor(a|b)",
      "aten::to.prim_dtype(Tensor(a) self, int? dtype=None, bool non_blocking=False, bool copy=False) -> Tensor(a|b)",
      "aten::isnan(Tensor self) -> Tensor",
      "aten::lgamma(Tensor self) -> Tensor",
      "aten::log10(Tensor self) -> Tensor",
      "aten::log(Tensor self) -> Tensor",
      "aten::log2(Tensor self) -> Tensor",
      "aten::log1p(Tensor self) -> Tensor",
      "aten::exp(Tensor self) -> Tensor",
      "aten::erf(Tensor self) -> Tensor",
      "aten::erfc(Tensor self) -> Tensor",
      "aten::fmod.Scalar(Tensor self, Scalar other) -> Tensor",
      "aten::fmod.Tensor(Tensor self, Tensor other) -> Tensor",
      "aten::cos(Tensor self) -> Tensor",
      "aten::sin(Tensor self) -> Tensor",
      "aten::tan(Tensor self) -> Tensor",
      "aten::acos(Tensor self) -> Tensor",
      "aten::asin(Tensor self) -> Tensor",
      "aten::atan(Tensor self) -> Tensor",
      "aten::atan2(Tensor self, Tensor other) -> Tensor",
      "aten::cosh(Tensor self) -> Tensor",
      "aten::sinh(Tensor self) -> Tensor",
      "aten::tanh(Tensor self) -> Tensor",
      "aten::hardtanh(Tensor self, Scalar min_val=-1, Scalar max_val=1) -> Tensor",
      "aten::sqrt(Tensor self) -> Tensor",
      "aten::rsqrt(Tensor self) -> Tensor",
      "aten::abs(Tensor self) -> Tensor",
      "aten::floor(Tensor self) -> Tensor",
      "aten::ceil(Tensor self) -> Tensor",
      "aten::round(Tensor self) -> Tensor",
      "aten::trunc(Tensor self) -> Tensor",
      "aten::threshold(Tensor self, Scalar threshold, Scalar value) -> Tensor",
      // "aten::masked_fill.Scalar(Tensor self, Tensor mask, Scalar value) -> Tensor",
      // "aten::masked_fill.Tensor(Tensor self, Tensor mask, Tensor value) -> Tensor", TODO: requires 0-dim Tensor
      "aten::remainder.Scalar(Tensor self, Scalar other) -> Tensor",
      "aten::remainder.Tensor(Tensor self, Tensor other) -> Tensor",
      "aten::sigmoid(Tensor self) -> Tensor",
      "aten::relu(Tensor self) -> Tensor",
      "aten::addcmul(Tensor self, Tensor tensor1, Tensor tensor2, *, Scalar value=1) -> Tensor",
      "aten::neg(Tensor self) -> Tensor",
      "aten::reciprocal(Tensor self) -> Tensor",
      "aten::expm1(Tensor self) -> Tensor",
      "aten::frac(Tensor self) -> Tensor",
      // TODO: uncomment once we can handle rand+broadcasts
      // "aten::rand_like(Tensor self, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor",
      "aten::__and__.Scalar(Tensor self, Scalar other) -> Tensor",
      "aten::__and__.Tensor(Tensor self, Tensor other) -> Tensor",
      "aten::__or__.Scalar(Tensor self, Scalar other) -> Tensor",
      "aten::__or__.Tensor(Tensor self, Tensor other) -> Tensor",
      "aten::__xor__.Scalar(Tensor self, Scalar other) -> Tensor",
      "aten::__xor__.Tensor(Tensor self, Tensor other) -> Tensor",
      "aten::__lshift__.Scalar(Tensor self, Scalar other) -> Tensor",
      "aten::__lshift__.Tensor(Tensor self, Tensor other) -> Tensor",
      "aten::__rshift__.Scalar(Tensor self, Scalar other) -> Tensor",
      "aten::__rshift__.Tensor(Tensor self, Tensor other) -> Tensor",
      "aten::where.self(Tensor condition, Tensor self, Tensor other) -> Tensor",
      "aten::where.ScalarSelf(Tensor condition, Scalar self, Tensor other) -> Tensor",
      "aten::where.ScalarOther(Tensor condition, Tensor self, Scalar other) -> Tensor",
      "aten::where.Scalar(Tensor condition, Scalar self, Scalar other) -> Tensor",
      "aten::batch_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float momentum, float eps, bool cudnn_enabled) -> Tensor",
      // TODO: enable other min/max variants, operators that can be both
      // elementwise or reductions:
      "aten::min.other(Tensor self, Tensor other) -> Tensor",
      "aten::max.other(Tensor self, Tensor other) -> Tensor",
      // TODO: enable slice, shape inference is not implemented for this op yet

      // TODO: enable once we have an out variant for conv2d to use in the NNC's
      // external call or when we have a performant TE representation for conv
      // "aten::conv2d(Tensor input, Tensor weight, Tensor? bias=None, int[2] stride=1, int[2] padding=0, int[2] dilation=1, int groups=1) -> Tensor",
  };
  // clang-format on

  return supported_eltwise_set;
}

bool isSupported(Node* node) {
  // For Block codegen we allow limited ops.
  if (tensorexpr::getTEGenerateBlockCode()) {
    return isSupportedForBlock(node);
  }

  static const OperatorSet cuda_only_operator_set{
      "aten::pow.Tensor_Scalar(Tensor self, Scalar exponent) -> Tensor",
  };
  static const OperatorSet cpu_only_operator_set{
      "aten::conv2d(Tensor input, Tensor weight, Tensor? bias=None, int[2] stride=1, int[2] padding=0, int[2] dilation=1, int groups=1) -> Tensor"};
  static const OperatorSet supported_reduction_set{
      "aten::sum(Tensor self, *, ScalarType? dtype=None) -> Tensor",
      "aten::sum.dim_IntList(Tensor self, int[1] dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor",
      "aten::softmax.int(Tensor self, int dim , ScalarType? dtype=None) -> Tensor",
      "aten::log_softmax.int(Tensor self, int dim, ScalarType? dtype=None) -> Tensor",
  };
  static const OperatorSet supported_misc_set{
      "aten::cat(Tensor[] tensors, int dim=0) -> Tensor",
      "aten::unsqueeze(Tensor(a) self, int dim) -> Tensor(a)",
  };
  // clang-format on

  if (node->isMemberOf(supported_eltwise_set()) ||
      node->isMemberOf(supported_misc_set) ||
      (texpr_reductions_enabled && node->isMemberOf(supported_reduction_set))) {
    // We only insert guards on Tensor types, so we rely on the output
    // of a node being uniquely determined by its input types.
    // bail if any non-Tensor input affects the output type
    // and cannot be reasoned about statically

    // Value is either an int or a float (can occur from .item())
    for (Value* v : node->inputs()) {
      if (v->type()->cast<NumberType>()) {
        return false;
      }
    }

    // Operator is only supported on CUDA.
    if (node->isMemberOf(cuda_only_operator_set)) {
      auto device = tensorexpr::pickDeviceType(node->inputs());
      if (!device) {
        device = tensorexpr::pickDeviceType(node->outputs());
      }
      if (!device || device->is_cpu()) {
        return false;
      }
    }

    // Operator is only supported on CPU.
    if (node->isMemberOf(cpu_only_operator_set)) {
      auto device = tensorexpr::pickDeviceType(node->inputs());
      if (!device) {
        device = tensorexpr::pickDeviceType(node->outputs());
      }
      if (!device || !device->is_cpu()) {
        return false;
      }
    }

    // non-const dtype / device
    for (auto arg_name : {"dtype", "device"}) {
      if (auto index = node->schema().argumentIndexWithName(arg_name)) {
        if (!toIValue(node->input(*index))) {
          return false;
        }
      }
    }

    if (FLAGS_torch_jit_disable_cat && node->kind() == aten::cat) {
      return false;
    }

    // We only support conv2d with constant strides/padding/dilation/groups
    if (node->kind() == aten::conv2d) {
      // Strides, padding and dilation are inputs 3, 4, and 5. They all must be
      // an IntList of size 2.
      for (size_t idx = 3; idx <= 5; idx++) {
        if (node->input(idx)->node()->kind() != prim::Constant ||
            !toIValue(node->input(idx))->isIntList() ||
            toIValue(node->input(idx))->toIntList().size() != 2) {
          return false;
        }
      }
      // Groups is input 6. It must be a constant int.
      if (node->input(6)->node()->kind() != prim::Constant ||
          !toIValue(node->input(6))->isInt()) {
        return false;
      }
    }

    return true;
  }

  // unschematized ops
  switch (node->kind()) {
    case prim::ConstantChunk:
    case prim::ListConstruct:
    case prim::TensorExprGroup:
      return true;
  }

  return false;
}

} // namespace tensorexpr

static bool texpr_fuser_enabled_ = true;

void setTensorExprFuserEnabled(bool val) {
  texpr_fuser_enabled_ = val;
}

bool tensorExprFuserEnabled() {
  static const char* enable_c_str = std::getenv("PYTORCH_TENSOREXPR");
  if (!enable_c_str) {
    return texpr_fuser_enabled_;
  }
  if (std::string(enable_c_str) == "0") {
    return false;
  }
  return true;
}

bool setTexprReductionsEnabled(bool value) {
  bool old_value = texpr_reductions_enabled;
  texpr_reductions_enabled = value;
  return old_value;
}

bool texprReductionsEnabled() {
  return texpr_reductions_enabled;
}

void removeProfileNodesAndSpecializeTypes(Block* b) {
  for (auto it = b->nodes().begin(); it != b->nodes().end(); it++) {
    if (it->kind() == prim::profile) {
      GRAPH_DEBUG("Removing prim::profile: %", it->output()->debugName());
      it->output()->replaceAllUsesWith(it->input());
      auto profiled_type = it->ty(attr::profiled_type)->expect<TensorType>();

      // A value can be profiled with differently typed uses.
      // This can occur from:
      // - having a use which is not executed, so the type will be
      // TensorType::get()
      // - control-flow that depends on tensor type:
      //   if x.size() == 2 op(x) else op(x)
      // - mutation of the value on a field represented in the tensor type
      //   op(x); x.resize_([...]); op(x)

      // The most common case today with num_profiles = 1 is from the first
      // case. Here we can just ignore non-profiled uses, and choose any of the
      // profiled uses. Because we guard all tensor types in the runtime, even
      // if we set a Value to have a profiled type from one use and then execute
      // a use with a different profiled type, we will still be correct.
      // In the future we could consider unifying the types of uses, or adding a
      // type refinement node so uses can have the correct corresponding type.
      if (profiled_type == TensorType::get()) {
        continue;
      }
      it->input()->setType(it->ty(attr::profiled_type));
      it.destroyCurrent();

    } else {
      for (Block* ib : it->blocks()) {
        removeProfileNodesAndSpecializeTypes(ib);
      }
    }
  }
}

void RemoveProfileNodesAndSpecializeTypes(std::shared_ptr<Graph>& graph) {
  GRAPH_DEBUG("Before removeProfileNodesAndSpecializeTypes:\n", *graph);
  removeProfileNodesAndSpecializeTypes(graph->block());
  GRAPH_DEBUG("After removeProfileNodesAndSpecializeTypes:\n", *graph);
}

void removeTensorTypeSpecialization(Value* v) {
  if (!v->type()->cast<TensorType>()) {
    return;
  }
  // Constants & TensorExprGroup will always produce specialized tensor type,
  // TypeCheck are inserted by this pass and only used by fusion groups that
  // insert proper guards
  if (v->node()->kind() == prim::Constant ||
      v->node()->kind() == prim::TypeCheck ||
      v->node()->kind() == prim::TensorExprGroup) {
    return;
  }
  v->setType(TensorType::get());
}

void removeTensorTypeSpecializations(Block* block) {
  for (Value* v : block->inputs()) {
    removeTensorTypeSpecialization(v);
  }
  for (Node* n : block->nodes()) {
    for (Block* b : n->blocks()) {
      removeTensorTypeSpecializations(b);
    }
    for (Value* v : n->outputs()) {
      removeTensorTypeSpecialization(v);
    }
  }
}

void RemoveTensorTypeSpecializations(std::shared_ptr<Graph>& graph) {
  removeTensorTypeSpecializations(graph->block());
}

void insertTypeGuard(
    Node* guarded_node,
    tensor_type_converter_t type_converter,
    Symbol kind) {
  GRAPH_DEBUG("Inserting a typecheck guard for a node", *guarded_node);
  auto subgraph = SubgraphUtils::getSubgraph(guarded_node);

  // Fixup types of the subgraph inputs
  std::vector<Value*> inputs_to_check;
  std::vector<TypePtr> guard_types;
  for (Value* input : guarded_node->inputs()) {
    // We only check inputs of the guarded nodes and expect user to infer
    // intermediates and outputs shapes
    if (!input->type()->cast<TensorType>()) {
      continue;
    }

    // fusion outputs are already guarded
    if (input->node()->kind() == prim::Constant ||
        input->node()->kind() == prim::FusionGroup) {
      continue;
    }
    inputs_to_check.push_back(input);
    guard_types.push_back(type_converter(input->type()->expect<TensorType>()));
  }
  if (!inputs_to_check.size()) {
    return;
  }

  // Add prim::TypeCheck node
  //
  // TypeCheck nodes  look like the following:
  //   %out1 : Float(2, 3), %out2 : Int(10, 30), %types_match : bool =
  //   prim::TypeCheck(%inp1 : Tensor, %inp2 : Tensor)
  //
  // They have N inputs whose types we are going to check and N+1 outputs. The
  // first N outputs specify expected types and N+1-th output holds the result
  // of the check (bool).
  Node* typecheck_node =
      guarded_node->owningGraph()
          ->create(kind, inputs_to_check, inputs_to_check.size() + 1)
          ->insertBefore(guarded_node);
  typecheck_node->tys_(attr::types, guard_types);
  Value* typecheck_result = typecheck_node->output(inputs_to_check.size());

  std::unordered_map<Value*, Value*> typechecked_inputs;
  for (size_t i = 0; i < typecheck_node->inputs().size(); ++i) {
    typechecked_inputs[typecheck_node->input(i)] = typecheck_node->output(i);
  }

  // Fixup types of the typecheck node outputs, which are used by the op in
  // execution
  typecheck_node->output(inputs_to_check.size())->setType(BoolType::get());
  for (size_t i = 0; i < typecheck_node->inputs().size(); ++i) {
    typecheck_node->output(i)->setType(typecheck_node->input(i)->type());
  }

  // Insert if
  auto versioning_if =
      guarded_node->owningGraph()
          ->create(prim::If, {typecheck_result}, guarded_node->outputs().size())
          ->insertAfter(typecheck_node);
  for (size_t idx = 0; idx < guarded_node->outputs().size(); ++idx) {
    versioning_if->output(idx)->setType(guarded_node->output(idx)->type());
    guarded_node->output(idx)->replaceAllUsesWith(versioning_if->output(idx));
  }
  auto true_block = versioning_if->addBlock();
  auto false_block = versioning_if->addBlock();

  // Fill in the false block. It should contain the unoptimized
  // copy of the fused subgraph.
  WithInsertPoint guard(false_block->return_node());
  const auto subgraph_outputs = insertGraph(
      *guarded_node->owningGraph(), *subgraph, guarded_node->inputs());
  for (Value* output : subgraph_outputs) {
    false_block->registerOutput(output);
  }

  // types get copied to the fallback graph, so remove specializations before
  // replacing
  removeTensorTypeSpecializations(false_block);
  replaceBlockWithFallbackGraph(false_block, guarded_node->inputs());

  // Fill in the true block. It has all inputs type-checked and its
  // body should be the fusion group node.
  guarded_node->moveBefore(true_block->return_node());
  for (size_t idx = 0; idx < guarded_node->inputs().size(); ++idx) {
    if (typechecked_inputs.count(guarded_node->input(idx))) {
      guarded_node->replaceInput(
          idx, typechecked_inputs.at(guarded_node->input(idx)));
    }
  }
  for (Value* output : guarded_node->outputs()) {
    true_block->registerOutput(output);
  }
}

class TensorExprFuser {
 public:
  TensorExprFuser(
      std::shared_ptr<Graph> graph,
      size_t min_group_size,
      bool disable_shape_checks)
      : graph_(std::move(graph)),
        min_group_size_(min_group_size),
        disable_shape_checks_(disable_shape_checks) {}

  // Builds up expressions that compute shapes of all intermediates (and
  // outputs) of the fusion group, based on the sizes of inputs. You should run
  // DCE to remove those that you end up not using.
  std::unordered_map<Value*, Value*> buildShapeExpressions(Node* fusion_group) {
    GRAPH_DUMP("buildShapeExpressions for ", fusion_group->g(attr::Subgraph));
    WithInsertPoint insert_guard{fusion_group->next()};
    std::unordered_map<Value*, Value*> shape_of;

    Graph* graph = fusion_group->owningGraph();
    auto subgraph = fusion_group->g(attr::Subgraph);

    auto inputs = fusion_group->inputs();
    auto sinputs = subgraph->inputs();
    AT_ASSERT(inputs.size() == sinputs.size());
    for (size_t i = 0; i < inputs.size(); ++i) {
      if (inputs[i]->type()->isSubtypeOf(TensorType::get())) {
        Value* soutput = graph->insert(aten::size, {inputs[i]});
        aliasDb_->createValue(soutput);
        GRAPH_DEBUG(
            "Adding a mapping for %",
            sinputs[i]->debugName(),
            " ",
            getHeader(soutput->node()));
        shape_of[sinputs[i]] = soutput;
      }
    }

    // When we have a guarantee that an output won't be removed, because it's
    // used in expressions that don't involve size checks, we can use its size
    // instead of computing a long chain of broadcasts, starting from the
    // beginning of the kernel.
    auto outputs = fusion_group->outputs();
    auto soutputs = subgraph->outputs();
    AT_ASSERT(outputs.size() == soutputs.size());
    for (size_t i = 0; i < outputs.size(); ++i) {
      if (usedOnlyInSize(outputs[i]))
        continue;
      Value* soutput = graph->insert(aten::size, {outputs[i]});
      aliasDb_->createValue(soutput);
      shape_of[soutputs[i]] = soutput;
    }

    for (Node* n : subgraph->nodes()) {
      if (n->kind() == prim::ConstantChunk) {
        Node* sizes_node = graph->insertNode(
            graph->create(prim::ChunkSizes, shape_of.at(n->input()), 2));
        sizes_node->i_(attr::dim, n->i(attr::dim));
        sizes_node->i_(attr::chunks, n->i(attr::chunks));
        for (Value* output : sizes_node->outputs()) {
          aliasDb_->createValue(output);
        }
        Value* regular_size = sizes_node->outputs().at(0);
        Value* last_size = sizes_node->outputs().at(1);
        regular_size->setType(ListType::ofInts());
        last_size->setType(ListType::ofInts());
        auto outputs = n->outputs();
        for (Value* o : outputs.slice(0, outputs.size() - 1)) {
          shape_of.emplace(o, regular_size);
        }
        shape_of.emplace(outputs.at(outputs.size() - 1), last_size);
        continue;
      }

      // we only support shape calculations for elementwise and
      // a few exceptions (e.g. prim::ConstantChunk, etc) listed above
      if (!n->isMemberOf(tensorexpr::supported_eltwise_set())) {
        continue;
      }

      auto tensor_inputs = filter(n->inputs(), [](Value* v) {
        return v->type()->isSubtypeOf(TensorType::get());
      });
      GRAPH_DEBUG("Building sizes for ", getHeader(n));
      bool all_inputs_have_sizes = true;
      auto shapes = fmap(tensor_inputs, [&](Value* v) {
        GRAPH_DEBUG("Getting aten::size for %", v->debugName());
        all_inputs_have_sizes &= shape_of.count(v);
        return shape_of.count(v) != 0 ? shape_of.at(v) : nullptr;
      });

      if (!all_inputs_have_sizes) {
        GRAPH_DEBUG(
            "Not all tensor arguments have sizes available to compute the broadcasted size",
            getHeader(n));
        continue;
      }
      shape_of.emplace(
          n->output(),
          shapes.size() == 1 ? shapes[0]
                             : broadcastSizes(shapes, aliasDb_.get()));
    }
    return shape_of;
  }

  void removeOutputsUsedOnlyInSize(Node* fusion_group) {
    if (fusion_group->kind() != prim::TensorExprGroup)
      return;
    auto subgraph = fusion_group->g(attr::Subgraph);

    auto shape_of = buildShapeExpressions(fusion_group);
    auto outputs = fusion_group->outputs().vec();
    auto soutputs = subgraph->outputs().vec();
    // XXX: Iterating in this order is not only good for performance reasons!
    // It is also crucial for correctness (i has to reflect the current true
    // index of outputs[i])!
    for (int64_t i = static_cast<int64_t>(outputs.size()) - 1; i >= 0; --i) {
      auto output = outputs[i];
      auto soutput = soutputs[i];
      if (usedOnlyInSize(output) && shape_of.count(soutput) > 0) {
        auto uses = output->uses();
        for (Use u : uses) {
          AT_ASSERT(u.user->matches("aten::size(Tensor self) -> int[]"));
          u.user->output()->replaceAllUsesWith(shape_of.at(soutput));
          u.user->destroy();
        }
        fusion_group->eraseOutput(i);
        subgraph->eraseOutput(i);
      }
    }
  }

  void run() {
    aliasDb_ = torch::make_unique<AliasDb>(graph_);
    RemoveRedundantProfiles(graph_);
    GRAPH_DUMP("After removing redundant profile nodes: ", graph_);
    createFusionGroups(graph_->block());
    GRAPH_DUMP("After creating fusion groups: ", graph_);
    // we maintain alias db correctness during initial fusion, but it is
    // difficult to maintain correctness after inlining so inline only after
    // fusion is done.
    inlineSmallFusionGroups(graph_->block());
    GRAPH_DUMP("After inlining small fusion groups: ", graph_);
    prepareFusionGroupAndGuardOutputs(graph_->block());
    GRAPH_DUMP("After guarding fusion groups: ", graph_);
    removeTensorTypeSpecializations(graph_->block());
    GRAPH_DUMP("After removing tensor type specializations: ", graph_);
  }

 private:
  Node* getOrCreateTensorExprSubgraph(Node* n) {
    if (n->hasAttribute(attr::Subgraph) && n->kind() == prim::TensorExprGroup) {
      return n;
    }
    GRAPH_UPDATE("Creating a tensorexpr::Group node from: ", *n);
    return SubgraphUtils::createSingletonSubgraphAndUpdateAliasing(
        n, prim::TensorExprGroup, *aliasDb_);
  }

  value_list sortReverseTopological(ArrayRef<Value*> inputs, Block* b) {
    value_list result;
    for (auto i : inputs) {
      if (i->node()->owningBlock() == b) {
        result.push_back(i);
      }
    }
    // Sort in reverse topological order
    std::sort(result.begin(), result.end(), [&](Value* a, Value* b) {
      return a->node()->isAfter(b->node());
    });
    return result;
  }

  // Create a fusion group starting from the node N.
  // We then try to pull inputs into the fusion group and repeat that process
  // until there is nothing we can pull in.
  std::pair<graph_node_list::iterator, bool> createFusionGroup(
      Node* fusion_node) {
    if (min_group_size_ == 1) {
      fusion_node = getOrCreateTensorExprSubgraph(fusion_node);
    }

    GRAPH_DEBUG("Iteratively pull input nodes into the fusion group...\n");
    auto inputs = sortReverseTopological(
        fusion_node->inputs(), fusion_node->owningBlock());
    for (auto input : inputs) {
      debugDumpFusionGroup("Current fusion group: ", fusion_node);
      GRAPH_DEBUG("Trying to merge: ", *input->node());
      if (auto maybe_fusion_group = tryMerge(fusion_node, input->node())) {
        // we successfully merged, so the new group's `inputs` may have
        // changed. So rescan the new group for more merging opportunities.
        return std::make_pair(
            maybe_fusion_group.value()->reverseIterator(), true);
      }
    }

    return std::make_pair(++fusion_node->reverseIterator(), false);
  }

  static void debugDumpFusionGroup(const std::string& msg, Node* n) {
    GRAPH_DEBUG(msg, *n);
    if (n->kind() == prim::TensorExprGroup) {
      GRAPH_DEBUG(*n->g(attr::Subgraph));
    }
  }

  // No Ops in eager shouldn't be outputs of Fusion Groups because it
  // will degrade perf and change aliasing relationships
  static bool unexecutedEagerOp(Node* n) {
    if (n->kind() != aten::to) {
      return false;
    }

    return *n->input(0)->type()->expect<TensorType>() ==
        *n->output()->type()->expect<TensorType>();
  }

  std::pair<graph_node_list::iterator, bool> scanNode(Node* n) {
    GRAPH_DEBUG("Considering node:", *n)

    if (!canHandle(n)) {
      return std::make_pair(++n->reverseIterator(), false);
    }
    // There are some nodes that we can support, but we don't want to start a
    // fusion group from - skip them.
    if (n->kind() == prim::ListConstruct || n->kind() == aten::slice ||
        n->kind() == aten::unsqueeze || n->kind() == prim::ConstantChunk ||
        n->kind() == prim::Constant || unexecutedEagerOp(n)) {
      return std::make_pair(++n->reverseIterator(), false);
    }
    return createFusionGroup(n);
  }

  // Merge fusible nodes into subgraphs in prim::TensorExprGroup nodes.
  void createFusionGroups(Block* block) {
    bool any_changed = true;
    while (any_changed) {
      any_changed = false;
      for (auto it = block->nodes().rbegin(); it != block->nodes().rend();) {
        bool changed;
        std::tie(it, changed) = scanNode(*it);
        any_changed |= changed;
      }
    }

    for (Node* n : block->nodes()) {
      for (Block* b : n->blocks()) {
        createFusionGroups(b);
      }
    }

    // Try to merge adjacent fusion groups together. Because we have only merged
    // by looking at graph inputs, without this we would not attempt to merge
    // adjacent fusion groups that don't have a depdency on each other

    std::vector<Node*> initial_fusion_groups;
    for (Node* n : block->nodes()) {
      if (n->kind() == prim::TensorExprGroup) {
        initial_fusion_groups.push_back(n);
      }
    }

    Node* prev_fusion_group =
        initial_fusion_groups.size() ? initial_fusion_groups[0] : nullptr;

    for (size_t i = 1; i < initial_fusion_groups.size(); ++i) {
      // Try merging the just created fusion group into the previous one.
      // If it did not work, then put the previous fusion group into
      // fusion_groups vector - we will not touch it anymore in this loop.
      // If merging suceeded, save the merged group as the "previous" fusion
      // group so that we can try to merge the next one into it.

      Node* fusion_group = initial_fusion_groups[i];
      debugDumpFusionGroup(
          "Trying to merge into the previous fusion group: ",
          prev_fusion_group);
      if (auto merged_fusion_group =
              tryMerge(prev_fusion_group, fusion_group)) {
        prev_fusion_group = *merged_fusion_group;
        debugDumpFusionGroup(
            "Successfully merged into the previous fusion group: ",
            prev_fusion_group);
      } else {
        GRAPH_DEBUG("Cannot merge into the previous fusion group");
        prev_fusion_group = fusion_group;
      }
    }
  }

  size_t blockSize(Block* block) {
    size_t num = 0;
    for (Node* n : block->nodes()) {
      // Don't count prim::Constants and prim::ListConstructs as these are nodes
      // we only pull in along with another, "main", node. E.g. the
      // ListConstruct nodes would also be pulled into a fusion group if they
      // are inputs of an aten::cat node.
      if (n->kind() == prim::Constant || n->kind() == prim::ListConstruct) {
        continue;
      }
      for (Block* b : n->blocks()) {
        num += blockSize(b);
      }
      num++;
    }
    return num;
  }

  bool inlineIfTooSmall(Node* n) {
    if (n->kind() != prim::TensorExprGroup) {
      return false;
    }
    auto subgraph = SubgraphUtils::getSubgraph(n);
    size_t num_modes = blockSize(subgraph->block());
    if (num_modes < min_group_size_) {
      GRAPH_UPDATE("Fusion group is too small, unmerging: ", *n);
      SubgraphUtils::unmergeSubgraph(n);
      return true;
    }
    // Cleanup the subgraph from duplicated constants while we're at it.
    ConstantPooling(subgraph);
    return false;
  }

  void inlineSmallFusionGroups(Block* block) {
    for (auto it = block->nodes().begin(); it != block->nodes().end();) {
      Node* n = *it;
      it++;

      for (Block* b : n->blocks()) {
        inlineSmallFusionGroups(b);
      }
      inlineIfTooSmall(n);
    }
  }

  c10::optional<Node*> tryMerge(Node* fusion_group, Node* to_merge) {
    if (!canMerge(fusion_group, to_merge)) {
      return c10::nullopt;
    }

    std::vector<Node*> nodes_to_merge = {to_merge};

    if (to_merge->kind() == aten::cat) {
      Node* listconstruct = to_merge->input(0)->node();
      nodes_to_merge.push_back(listconstruct);
    }

    // First, try to move all the nodes we want to fuse next to the fusion
    // group.
    Node* move_point = fusion_group;
    for (auto n : nodes_to_merge) {
      GRAPH_UPDATE("Trying to move node next to fusion group: ", getHeader(n));
      if (!aliasDb_->moveBeforeTopologicallyValid(n, move_point)) {
        GRAPH_UPDATE("Failed to move because of AliasDB checks!");
        return c10::nullopt;
      }
      move_point = n;
    }

    // Now all the nodes that we're going to fuse are moved next to the fusion
    // group, so we can safely merge them into the fusion group subgraph.
    fusion_group = getOrCreateTensorExprSubgraph(fusion_group);

    for (auto n : nodes_to_merge) {
      GRAPH_UPDATE("Merging ", getHeader(n));
      SubgraphUtils::mergeNodeIntoSubgraphAndUpdateAliasing(
          n, fusion_group, *aliasDb_);
    }
    return fusion_group;
  }

  bool shapeIsKnown(Value* v) {
    if (v->type()->cast<TensorType>()) {
      if (!v->isCompleteTensor()) {
        return false;
      }
      if (*v->type()->castRaw<TensorType>()->dim() == 0) {
        return false;
      }
    }
    return true;
  }
  bool allShapesAreKnown(Node* node) {
    // TODO: Relax the checks to support dynamic shapes
    for (Value* input : node->inputs()) {
      if (!shapeIsKnown(input)) {
        return false;
      }
    }
    for (Value* output : node->outputs()) {
      if (!shapeIsKnown(output)) {
        return false;
      }
    }
    return true;
  }

  bool canFuseOnDevice(Value* v) {
    auto type = v->type()->cast<TensorType>();
    if (!type) {
      return true;
    }
    auto device = type->device();
    if (!device) {
      return false;
    }
    if (device->is_cpu()) {
      return canFuseOnCPU();
    } else if (device->is_cuda()) {
      return canFuseOnGPU();
    } else if (device->is_xpu()) {
      return false;
    } else {
      TORCH_CHECK_NOT_IMPLEMENTED(false, "Unknown device for tensorexpr fuser")
    }
  }

  bool isFusableOnDevice(Node* node) {
    for (const auto& input : node->inputs()) {
      if (!canFuseOnDevice(input)) {
        return false;
      }
    }
    return true;
  }

  bool typesAreSupported(Node* node) {
    // clang-format off
    // breaks up the schema strings so they are no longer discoverable with ctrl-F
    static const OperatorSet float_only_operator_set{
      "aten::fmod.Scalar(Tensor self, Scalar other) -> Tensor",
      "aten::fmod.Tensor(Tensor self, Tensor other) -> Tensor",
      "aten::remainder.Scalar(Tensor self, Scalar other) -> Tensor",
      "aten::remainder.Tensor(Tensor self, Tensor other) -> Tensor",
    };
    static const OperatorSet int_only_operator_set{
      "aten::__lshift__.Scalar(Tensor self, Scalar other) -> Tensor",
      "aten::__lshift__.Tensor(Tensor self, Tensor other) -> Tensor",
      "aten::__rshift__.Scalar(Tensor self, Scalar other) -> Tensor",
      "aten::__rshift__.Tensor(Tensor self, Tensor other) -> Tensor",
    };
    // clang-format on

    for (const Value* v : node->inputs()) {
      if (auto const& tt = v->type()->cast<TensorType>()) {
        auto const& st = tt->scalarType();

        // All tensors must be typed.
        if (!st) {
          return false;
        }

        // Byte tensors introduce too many corner cases in type promotion.
        // Better not to try to handle them.
        if (*st == c10::ScalarType::Byte) {
          return false;
        }

        // These operators only support floats, because integer divisors need to
        // raise ZeroDivisionError.
        if (node->isMemberOf(float_only_operator_set) && !isFloatingType(*st)) {
          return false;
        }

        // These operators have complicated casting rules for floats.
        if (node->isMemberOf(int_only_operator_set) && isFloatingType(*st)) {
          return false;
        }
      } else if (node->isMemberOf(float_only_operator_set)) {
        // Check scalar operands of float-only ops.
        if (!v->type()->cast<FloatType>()) {
          return false;
        }
      } else if (node->isMemberOf(int_only_operator_set)) {
        if (!v->type()->cast<IntType>()) {
          return false;
        }
      }
    }
    if (node->kind() == aten::to) {
      // only support same-device conversion
      auto device = tensorexpr::pickDeviceType(node->inputs());
      auto output_device = tensorexpr::pickDeviceType(node->outputs());
      if (!device || !output_device || *device != *output_device) {
        return false;
      }
      // non_blocking only applies in cross-device conversion, which we bail on
      // copy arg only applies if op is a no-op, which we dont start fusion
      // group from memory format is separately handled in NNC output

      // all non-Tensor arguments must be constant
      for (size_t i = 1; i < node->inputs().size(); i++) {
        if (node->inputs().at(i)->node()->kind() != prim::Constant) {
          return false;
        }
      }
      // cant support non-constant pin_memory or pin_memory = True
      if (auto maybe_index =
              node->schema().argumentIndexWithName("pin_memory")) {
        int index = *maybe_index;
        auto inp = node->input(index);
        if (inp->type() != NoneType::get() &&
            constant_as<bool>(inp).value_or(true)) {
          return false;
        }
      }
    }

    return true;
  }

#define REQ(cond)                           \
  if (!(cond)) {                            \
    GRAPH_DEBUG("Failed cond " #cond "\n"); \
    return false;                           \
  }

  bool canHandle(Node* node) {
    REQ(disable_shape_checks_ || allShapesAreKnown(node));
    REQ(isFusableOnDevice(node));

    for (Value* input : node->inputs()) {
      if (auto const& tt = input->type()->cast<TensorType>()) {
        auto st = tt->scalarType();
        if (!st) {
          // All tensor types should be known.
          return false;
        }
        if (c10::isComplexType(*st) || c10::isQIntType(*st) ||
            *st == c10::ScalarType::BFloat16) {
          return false;
        }
      }
    }
    if (node->kind() == aten::cat) {
      REQ(node->input(0)->node()->kind() == prim::ListConstruct);
      REQ(node->input(0)->uses().size() == 1);
      REQ(node->input(1)->node()->kind() == prim::Constant);
      auto const& listconstruct = node->input(0)->node();
      REQ(tensorexpr::pickDeviceType(listconstruct->inputs()));
    } else {
      REQ(tensorexpr::pickDeviceType(node->inputs()));
    }

    // Only fuse aten::batch_norm when the parameter 'training' is false
    if (node->kind() == aten::batch_norm) {
      REQ(node->input(5)->node()->kind() == prim::Constant);
      REQ(!toIValue(node->input(5)).value().toBool());
    }

    REQ(tensorexpr::isSupported(node));
    REQ(typesAreSupported(node));

    // A hook to optimizations limitter to allow bisecting the pass
    if (!JIT_OPT_ALLOWED) {
      return false;
    }

    return true;
  }

  bool canMerge(Node* consumer, Node* producer) {
    // Only fuse within a block
    REQ(consumer->owningBlock() == producer->owningBlock());

    // Symbolic checks
    REQ(canHandle(producer) || producer->kind() == prim::TensorExprGroup);
    TORCH_INTERNAL_ASSERT(
        consumer->kind() == prim::TensorExprGroup || canHandle(consumer));

    // Device checks
    if (consumer->kind() != aten::cat && producer->kind() != aten::cat) {
      // aten::cat needs a special handling because it takes a Tensor[] as its
      // input We deal with that in the code below.
      auto consumer_device = tensorexpr::pickDeviceType(consumer->inputs());
      REQ(consumer_device);
      auto producer_device = tensorexpr::pickDeviceType(producer->inputs());
      REQ(producer_device);
      REQ(*consumer_device == *producer_device);
    }

    // Alias checks
    REQ(aliasDb_->couldMoveBeforeTopologically(producer, consumer));

    // Ops that return aliases can only be folded if this is the only use.
    if (producer->kind() == aten::slice ||
        producer->kind() == aten::unsqueeze ||
        producer->kind() == prim::ConstantChunk) {
      for (auto& use : producer->output(0)->uses()) {
        REQ(use.user == consumer);
      }
    }

    if (!consumer->hasAttribute(attr::Subgraph) &&
        consumer->kind() != prim::TensorExprGroup) {
      // Don't initiate a fusion group from prim::ListConstruct
      REQ(consumer->kind() != prim::ListConstruct);
      REQ(consumer->kind() != aten::slice);
      REQ(consumer->kind() != aten::unsqueeze);
      REQ(consumer->kind() != prim::ConstantChunk);

      // Don't initiate a fusion group just for a constant operand
      REQ(producer->kind() != prim::Constant);
    }

    if (producer->kind() == aten::cat) {
      REQ(producer->input(0)->node()->kind() == prim::ListConstruct);
      REQ(producer->input(0)->uses().size() == 1);
      REQ(producer->input(1)->node()->kind() == prim::Constant);
      auto const& listConstruct = producer->input(0)->node();
      // We're merging listconstruct->cat->consumer. cat is the producer here
      // and we cannot determine its device type - we should use device of the
      // listconstruct instead
      auto listconstruct_device =
          tensorexpr::pickDeviceType(listConstruct->inputs());
      auto consumer_device = tensorexpr::pickDeviceType(consumer->inputs());
      REQ(listconstruct_device);
      REQ(consumer_device);
      REQ(*listconstruct_device == *consumer_device);
      for (auto const& input : listConstruct->inputs()) {
        REQ(isFusableOnDevice(input->node()));
      }
    } else if (consumer->kind() == aten::cat) {
      REQ(consumer->input(0)->node()->kind() == prim::ListConstruct);
      REQ(consumer->input(0)->uses().size() == 1);
      REQ(consumer->input(1)->node()->kind() == prim::Constant);
      auto const& listConstruct = consumer->input(0)->node();
      // We're merging listconstruct->cat. cat is the consumer and listconstruct
      // is the producer. cat doesn't have its device type and thus the only
      // thing we should check is that listconstruct's device is well defined
      // (e.g. all its inputs has the same device).
      auto listconstruct_device =
          tensorexpr::pickDeviceType(listConstruct->inputs());
      REQ(listconstruct_device);
    } else {
      REQ(isFusableOnDevice(producer));
    }

    return true;
  }
#undef REQ

  // TODO: support constant tensors instead of setting them as input
  void liftTensorConstantsFromFusionGroups(Node* fusion_group) {
    auto subgraph = SubgraphUtils::getSubgraph(fusion_group);
    WithInsertPoint guard(fusion_group);
    for (auto it = subgraph->block()->nodes().begin();
         it != subgraph->block()->nodes().end();
         ++it) {
      auto n = *it;
      if (n->kind() == prim::Constant &&
          n->output()->type()->cast<TensorType>()) {
        auto constant =
            fusion_group->owningGraph()->insertConstant(*toIValue(n->output()));
        fusion_group->addInput(constant);
        auto inputToGraph = subgraph->addInput();
        inputToGraph->setType(n->output()->type());
        n->output()->replaceAllUsesWith(inputToGraph);
        it.destroyCurrent();
      }
    }
  }

  void prepareFusionGroupAndGuardOutputs(Block* block) {
    std::vector<Node*> fusion_groups;
    for (Node* n : block->nodes()) {
      for (Block* b : n->blocks()) {
        prepareFusionGroupAndGuardOutputs(b);
      }
      if (n->kind() == prim::TensorExprGroup) {
        fusion_groups.push_back(n);
      }
    }
    for (Node* fusion_group : fusion_groups) {
      removeOutputsUsedOnlyInSize(fusion_group);
      liftTensorConstantsFromFusionGroups(fusion_group);
      insertTypeGuard(
          fusion_group,
          [](const TensorTypePtr& t) { return t; },
          prim::TypeCheck);
    }
  }

  std::shared_ptr<Graph> graph_;
  std::unique_ptr<AliasDb> aliasDb_ = nullptr;

  // Minimal size of a fusion group
  size_t min_group_size_;
  // If true, shapes are ignored
  bool disable_shape_checks_;
};

void FuseTensorExprs(
    std::shared_ptr<Graph>& graph,
    size_t min_group_size,
    bool disable_shape_checks) {
  GRAPH_DUMP("Before TExprFuser: ", graph);

  // Temporary change for Block code generation.
  if (tensorexpr::getTEGenerateBlockCode()) {
    min_group_size = 1;
  }

  // Get rid of dead code so that we don't waste effort fusing it.
  EliminateDeadCode(graph);

  TensorExprFuser fuser(graph, min_group_size, disable_shape_checks);
  fuser.run();

  EliminateCommonSubexpression(graph);
  EliminateDeadCode(graph);

  GRAPH_DUMP("After TExprFuser: ", graph);
}

Operation createTensorExprOp(const Node* node) {
  auto kernel =
      std::make_shared<tensorexpr::TensorExprKernel>(node->g(attr::Subgraph));
  return [kernel](Stack* stack) {
    RECORD_FUNCTION("TensorExpr", std::vector<c10::IValue>());
    kernel->run(*stack);
    return 0;
  };
}

RegisterOperators TensorExprOps({
    torch::jit::Operator(
        prim::TensorExprGroup,
        createTensorExprOp,
        AliasAnalysisKind::INTERNAL_SPECIAL_CASE),
});

} // namespace jit
} // namespace torch
