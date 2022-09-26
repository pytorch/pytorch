
#include <torch/csrc/jit/passes/autocast.h>

#include <ATen/autocast_mode.h>
#include <c10/core/ScalarType.h>
#include <c10/util/Exception.h>
#include <c10/util/Optional.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/cuda_graph_fuser.h>
#include <torch/csrc/jit/passes/quantization/helper.h>

#include <stack>
#include <unordered_set>
#include <vector>

namespace torch {
namespace jit {

namespace {

bool autocast_enabled = true;

struct AutocastContext {
  bool gpu_enabled = false;
  bool cpu_enabled = false;
  c10::ScalarType gpu_scalar_type = c10::ScalarType::Undefined;
  c10::ScalarType cpu_scalar_type = c10::ScalarType::Undefined;

  operator bool() const {
    return gpu_enabled || cpu_enabled;
  }
};

struct AutocastScope {
  Value* instance = nullptr;
  AutocastContext context;
  void stack(const AutocastContext& parent_context) {}
};

bool isAutocastNode(Value* value) {
  const auto class_name = getModuleName(value);
  return class_name.has_value() &&
      (*class_name == "__torch__.torch.cuda.amp.autocast_mode.autocast" ||
       *class_name == "__torch__.torch.cpu.amp.autocast_mode.autocast" ||
       *class_name == "__torch__.torch.amp.autocast_mode.autocast");
}

// If we have an autocast instance, return it
//
// This is the pattern we're looking for (this is done after
//  autocast.__init__() has been inlined)
//
// %4 : bool = prim::Constant[value=1]()
// %5 : __torch__.torch.cuda.amp.autocast_mode.autocast = prim::CreateObject()
//  = prim::SetAttr[name="_enabled"](%5, %4)
//
// Notes:
//  1. There's no guarantee that the autocast instance is in the same block
//    as the prim::Enter() node
//  2. `prim::SetAttr` must follow `prim::CreateObject()` in the same block,
//    but there might be other nodes in between
//
c10::optional<AutocastScope> parseAutocast(
    Value* value,
    const AutocastContext& context) {
  if (!isAutocastNode(value)) {
    // Not an autocast...
    return c10::nullopt;
  }
  if (value->node()->kind() == prim::CreateObject) {
    AutocastScope scope;
    scope.instance = value;
    scope.context = context;
    c10::optional<bool> enabled;
    std::string device;
    c10::ScalarType dtype = c10::ScalarType::Undefined;
    for (Use use : value->uses()) {
      // TODO: support runtime flag
      if (use.user->kind() == prim::SetAttr &&
          use.user->s(attr::name) == "_enabled") {
        // Search for `prim::SetAttr[name="_enabled"]`
        auto ret = constant_as<bool>(use.user->input(1));
        TORCH_CHECK(
            ret.has_value(), "Autocast _enabled argument must be a constant");
        enabled = ret.value();
      } else if (
          use.user->kind() == prim::SetAttr &&
          use.user->s(attr::name) == "device") {
        // Search for `prim::SetAttr[name="device"]`
        auto ret = constant_as<std::string>(use.user->input(1));
        TORCH_CHECK(
            ret.has_value(), "Autocast device argument must be a constant");
        device = ret.value();
      } else if (
          use.user->kind() == prim::SetAttr &&
          use.user->s(attr::name) == "fast_dtype") {
        // Search for `prim::SetAttr[name="fast_dtype"]`
        auto ret = constant_as<c10::ScalarType>(use.user->input(1));
        TORCH_CHECK(
            ret.has_value() && ret.value() != c10::ScalarType::Undefined,
            "Autocast dtype argument must be a constant and defined");
        dtype = ret.value();
      }
    }
    TORCH_CHECK(enabled.has_value(), "Autocast missing _enabled attribute");
    TORCH_CHECK(
        dtype != c10::ScalarType::Undefined,
        "Autocast missing fast_dtype attribute");
    TORCH_CHECK(!device.empty(), "Autocast missing device attribute");
    if (device == "cuda") {
      scope.context.gpu_enabled = enabled.value();
      scope.context.gpu_scalar_type = dtype;
    } else if (device == "cpu") {
      scope.context.cpu_enabled = enabled.value();
      scope.context.cpu_scalar_type = dtype;
    } else {
      TORCH_INTERNAL_ASSERT(
          false, "unrecognized device for autocast pass: ", device);
    }
    return scope;
  } else {
    // We only support simple and static autocast expressions. For example,
    // the following should report an error (since the autocast would not
    // work as expected)
    //
    //    autocast_on = autocast(enabled=True)
    //    autocast_off = autocast(enabled=False)
    //    with autocast_on if condition else autocast_off:
    //        ...
    //
    // TODO: better error message
    //
    AT_ERROR("Unsupported autocast syntax");
  }

  return c10::nullopt;
}

void castTensorInputs(
    Node* node,
    Symbol cast_op,
    const AutocastContext& context) {
  if (!context) {
    return;
  }

  const auto graph = node->owningGraph();

  std::unordered_set<Value*> casted_inputs;
  // need to also keep the inputs in order, otherwise tracing fails
  // sanity checks because casting ops are inserted in random order
  std::vector<Value*> casted_inputs_ordered;
  for (auto input : node->inputs()) {
    // TODO: update cast_op signature to take dynamic context flags
    auto input_tensor_type = input->type()->cast<TensorType>();
    if (input_tensor_type && input->node()->kind() != cast_op) {
      auto has_inserted = casted_inputs.insert(input);
      if (has_inserted.second) {
        casted_inputs_ordered.push_back(input);
      }
    }
  }

  WithInsertPoint insert_point(node);

  for (auto input : casted_inputs_ordered) {
    if (cast_op == aten::_autocast_to_full_precision) {
      const auto new_input = graph->insert(
          cast_op,
          {input,
           graph->insertConstant(IValue(context.gpu_enabled)),
           graph->insertConstant(IValue(context.cpu_enabled))});
      node->replaceInputWith(input, new_input);
    } else if (cast_op == aten::_autocast_to_reduced_precision) {
      const auto new_input = graph->insert(
          cast_op,
          {input,
           graph->insertConstant(IValue(context.gpu_enabled)),
           graph->insertConstant(IValue(context.cpu_enabled)),
           graph->insertConstant(IValue(context.gpu_scalar_type)),
           graph->insertConstant(IValue(context.cpu_scalar_type))});
      node->replaceInputWith(input, new_input);
    } else {
      TORCH_INTERNAL_ASSERT(
          false, "unrecognized cast_op symbol: ", cast_op.toQualString());
    }
  }
}

bool hasExplicitDtypeArgument(Node* node) {
  if (node->hasNamedInput("dtype")) {
    Value* dtype_arg = node->namedInput("dtype");
    return dtype_arg->type()->kind() != TypeKind::NoneType;
  }
  return false;
}

void castInputsToWidestType(Node* node, const AutocastContext& context) {
  if (!context) {
    return;
  }
  // Figure out the widest type
  // (really, just looking for any float32 inputs)
  //
  // TODO: revisit this (do we need to consider float64 types?)
  //
  for (auto input : node->inputs()) {
    if (auto tensor_type = input->type()->cast<TensorType>()) {
      const auto dtype = tensor_type->scalarType();
      if (!dtype.has_value() || *dtype == at::ScalarType::Float) {
        castTensorInputs(node, aten::_autocast_to_full_precision, context);
        return;
      }
    }
  }
}

// Users can call torch.is_autocast_enabled() or is_autocast_cpu_enabled() to
// determine whether autocasting is enabled. With JIT-scripted functions, we
// actually need to return true if eager autocast OR jit autocast are enabled.
//
// In the case where JIT autocast is enabled, we replace
//    %x : bool = aten::is_autocast_enabled()
// with a constant "True".
//
// More context on eager vs JIT autocasting:
//
// Autocasting actually has two settings: eager autocasting, and JIT
// autocasting. Eager autocasting is the thread-local setting that turns on
// the relevant bit in the dispatcher settings. JIT autocasting is the pass
// implemented in this file, which makes changes to the graph to insert casting
// ops in order to achieve the same behavior as eager autocasting.
//
// If eager autocasting is enabled at the time when a JIT-scripted function is
// invoked, then autocasting will occur regardless of what the JIT-autocasting
// settings are.
void updateAutocastEnabledCheck(Node* node, bool is_jit_enabled) {
  if (!is_jit_enabled) {
    return;
  }

  auto graph = node->owningGraph();

  WithInsertPoint insert_point(node);

  Value* true_constant = graph->insertConstant(IValue(true));
  node->output()->replaceAllUsesWith(true_constant);
  node->destroy();
}

// [Note: implicit type promotion in Autocast]
//
// Casting policy below mostly follows pytorch/aten/src/ATen/autocast.cpp, with
// a few exceptions, e.g. `aten::add`, which is needed to be put to promotion
// list for JIT autocast.
// The reason is that in eager amp, some binary ops promote inputs implicitly
// inside the operation, e.g. `aten::add` with fp16 & fp32 inputs would both be
// casted to fp32. In backward, autograd would cast dgrad to match their
// scalar_type in forward graph. So inputs with mismatched scalar_type would
// get the different dgrad.
// While in JIT, autodiff doesn't do this, so implicit cast is not visible to
// autodiff and backward dgrad for mismatched inputs would ended up with dgrads
// in the same scalar_type. This has caused downstream operations, which
// expects dgrad to be the same scalar type to throw mismatch error.
//
// TODO: Use the list from AMP eager directly
void handleBlock(Block* block, AutocastContext initial_state) {
  std::stack<AutocastScope> autocast_stack;

  c10::optional<bool> incompatible_amp = c10::nullopt;

  // The current autocast enabled/disabled state
  auto current_state = [&] {
    return autocast_stack.empty() ? initial_state
                                  : autocast_stack.top().context;
  };

  for (Node* node : block->nodes()) {
    switch (node->kind()) {
      case prim::CallFunction:
        // TODO: limit it only to amp related node;
        if (current_state() == initial_state) {
          // if the current autocasting state is the same as the global state,
          // then autocasting will be done correctly on subsequent method and
          // function calls
          if (current_state()) {
            castTensorInputs(
                node, aten::_autocast_to_full_precision, current_state());
          }
          break;
        }
        TORCH_INTERNAL_ASSERT(
            !incompatible_amp.has_value() || incompatible_amp.value(),
            "Calls are not expected with AMP & JIT");
        incompatible_amp = true;
        break;

      case prim::CallMethod:
        // TODO: limit it only to amp related node;
        if (current_state() == initial_state) {
          // if the current autocasting state is the same as the global state,
          // then autocasting will be done correctly on subsequent method and
          // function calls
          if (current_state()) {
            castTensorInputs(
                node, aten::_autocast_to_full_precision, current_state());
          }
          break;
        }
        if (auto class_type = node->input(0)->type()->cast<ClassType>()) {
          const auto& name = node->s(attr::name);
          const auto& function = class_type->getMethod(name);
          if (!function.isGraphFunction()) {
            TORCH_INTERNAL_ASSERT(
                !incompatible_amp.has_value() || incompatible_amp.value(),
                "Calls are not expected with AMP & JIT");
            incompatible_amp = true;
          }
        } else {
          TORCH_INTERNAL_ASSERT(
              !incompatible_amp.has_value() || incompatible_amp.value(),
              "Unexpected prim::CallMethod form with AMP & JIT");
          incompatible_amp = true;
        }
        break;

      case prim::Enter:
        if (auto autocast_scope =
                parseAutocast(node->input(), current_state())) {
          if (node->hasUses()) {
            // TODO: better error message
            AT_ERROR("`with autocast() as ...` is not supported");
          }
          TORCH_INTERNAL_ASSERT(
              !incompatible_amp.has_value() || !incompatible_amp.value(),
              "Unsupported case by AMP & JIT");
          incompatible_amp = false;
          autocast_stack.push(*autocast_scope);
        }
        break;

      case prim::Exit:
        if (isAutocastNode(node->input(0))) {
          TORCH_INTERNAL_ASSERT(!autocast_stack.empty());
          TORCH_INTERNAL_ASSERT(autocast_stack.top().instance == node->input());
          TORCH_INTERNAL_ASSERT(
              !incompatible_amp.has_value() || !incompatible_amp.value(),
              "Unsupported case by AMP & JIT");
          incompatible_amp = false;
          autocast_stack.pop();
        }
        break;

      case aten::is_autocast_enabled:
        updateAutocastEnabledCheck(node, current_state().gpu_enabled);
        break;

      case aten::is_autocast_cpu_enabled:
        updateAutocastEnabledCheck(node, current_state().cpu_enabled);
        break;

      // CastPolicy::fp16 (cast all inputs to float16)
      case aten::_convolution:
      case aten::conv1d:
      case aten::conv2d:
      case aten::conv3d:
      case aten::conv_tbc:
      case aten::conv_transpose1d:
      case aten::convolution:
      case aten::cudnn_convolution:
      case aten::cudnn_convolution_transpose:
      case aten::prelu:
      case aten::addmm:
      case aten::addmv:
      case aten::addr:
      case aten::matmul:
      case aten::mm:
      case aten::mv:
      case aten::linear:
      case aten::addbmm:
      case aten::baddbmm:
      case aten::bmm:
      case aten::chain_matmul:
      case aten::_thnn_fused_lstm_cell:
      case aten::_thnn_fused_gru_cell:
      case aten::lstm_cell:
      case aten::gru_cell:
      case aten::rnn_tanh_cell:
      case aten::rnn_relu_cell:
        if (!node->schema().is_mutable()) {
          castTensorInputs(
              node, aten::_autocast_to_reduced_precision, current_state());
        }
        break;

      // CastPolicy::fp32 (cast all inputs to float32)
      case aten::native_layer_norm:
      case aten::acos:
      case aten::asin:
      case aten::cosh:
      case aten::erfinv:
      case aten::exp:
      case aten::expm1:
      case aten::log:
      case aten::log10:
      case aten::log2:
      case aten::log1p:
      case aten::reciprocal:
      case aten::rsqrt:
      case aten::sinh:
      case aten::tan:
      case aten::pow:
      case aten::softplus:
      case aten::gelu:
      case aten::layer_norm:
      case aten::group_norm:
      case aten::frobenius_norm:
      case aten::nuclear_norm:
      case aten::cosine_similarity:
      case aten::cosine_embedding_loss:
      case aten::nll_loss:
      case aten::nll_loss2d:
      case aten::hinge_embedding_loss:
      case aten::kl_div:
      case aten::l1_loss:
      case aten::smooth_l1_loss:
      case aten::mse_loss:
      case aten::margin_ranking_loss:
      case aten::multilabel_margin_loss:
      case aten::soft_margin_loss:
      case aten::triplet_margin_loss:
      case aten::multi_margin_loss:
      case aten::binary_cross_entropy_with_logits:
      case aten::dist:
      case aten::pdist:
      case aten::cdist:
      case aten::renorm:
      case aten::logsumexp:
        if (!node->schema().is_mutable()) {
          castTensorInputs(
              node, aten::_autocast_to_full_precision, current_state());
        }
        break;

      // CastPolicy::fp32_set_opt_dtype
      case aten::prod:
      case aten::log_softmax:
      case aten::cumprod:
      case aten::cumsum:
      case aten::sum:
        if (!node->schema().is_mutable() && !hasExplicitDtypeArgument(node)) {
          castTensorInputs(
              node, aten::_autocast_to_full_precision, current_state());
        }
        break;

      // cast softmax to fp32 only on GPU
      case aten::softmax:
        if (!node->schema().is_mutable() && !hasExplicitDtypeArgument(node)) {
          auto context = current_state();
          context.cpu_enabled = false;
          castTensorInputs(node, aten::_autocast_to_full_precision, context);
        }
        break;

      // CastPolicy::promote (promote inputs to the widest type)
      case aten::addcdiv:
      case aten::addcmul:
      case aten::atan2:
      case aten::bilinear:
      case aten::cat:
      case aten::cross:
      case aten::dot:
      case aten::equal:
      case aten::index_put:
      case aten::stack:
      case aten::tensordot:
      // add, sub, mul, div were added to autocast jit, because aten implicit
      // type promotion is not visible to JIT and could cause dtype mismatch on
      // backward
      // see [Note: implicit type promotion in Autocast]
      case aten::add:
      case aten::sub:
      case aten::mul:
      case aten::div:
        if (!node->schema().is_mutable()) {
          castInputsToWidestType(node, current_state());
        }
        break;

      // Banned in autocast, see binary_cross_entropy_banned()
      case aten::binary_cross_entropy:
        if (current_state()) {
          AT_ERROR("Unsafe to autocast");
        }
    }

    // process sub-blocks, if any
    for (Block* sub_block : node->blocks()) {
      handleBlock(sub_block, current_state());
    }
  }

  // Sanity check: make sure there's no unbalanced transition
  TORCH_INTERNAL_ASSERT(autocast_stack.empty());
}

} // namespace

bool setAutocastMode(bool value) {
  auto old_value = autocast_enabled;
  autocast_enabled = value;
  return old_value;
}

bool autocastEnabled() {
  return autocast_enabled;
}

void Autocast(const std::shared_ptr<Graph>& graph) {
  GRAPH_DUMP("\nBefore Autocast: ", graph);
  if (autocastEnabled()) {
    AutocastContext init = {
        at::autocast::is_enabled(),
        at::autocast::is_cpu_enabled(),
        at::autocast::get_autocast_gpu_dtype(),
        at::autocast::get_autocast_cpu_dtype()};
    handleBlock(graph->block(), init);
  }
  GRAPH_DUMP("\nAfter Autocast: ", graph);
}

} // namespace jit
} // namespace torch
