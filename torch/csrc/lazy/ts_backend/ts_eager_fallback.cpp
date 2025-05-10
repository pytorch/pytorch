#include <torch/csrc/lazy/ts_backend/ts_eager_fallback.h>

#include <ATen/FunctionalTensorWrapper.h>
#include <ATen/Functions.h>
#include <ATen/core/boxing/KernelFunction.h>
#include <ATen/native/CPUFallback.h>
#include <torch/csrc/lazy/backend/backend_interface.h>
#include <torch/csrc/lazy/core/config.h>
#include <torch/csrc/lazy/core/metrics.h>
#include <torch/csrc/lazy/core/tensor.h>
#include <torch/library.h>
#include <sstream>
#include <unordered_map>

namespace torch::lazy {
namespace {

std::vector<at::Tensor> _to_eager(
    at::TensorList tensors,
    c10::DeviceType device_type) {
  switch (device_type) {
    case at::kCPU: {
      return at::_to_cpu(tensors);
    }
    default: {
      std::vector<at::Tensor> eager_tensors;
      for (const auto& t : tensors) {
        c10::TensorOptions options = t.options().device(device_type);
        at::Tensor eager_tensor = t.to(
            options,
            /*non_blocking*/ false,
            /*copy*/ false);
        eager_tensors.push_back(eager_tensor);
      }
      return eager_tensors;
    }
  }
}

// convenience helper for converting tensors to cpu

std::vector<at::Tensor> to_eager(
    const at::TensorList& tensors,
    c10::DeviceType device_type) {
  // We can't just call _to_eager() on the entire list of Tensors because it
  // will break on undefined tensors. Separate out undefined tensors first.
  std::vector<at::Tensor> eager_tensors(tensors.size());
  std::vector<at::Tensor> valid_tensors;
  std::vector<bool> to_translate(tensors.size());
  for (size_t i = 0; i < tensors.size(); ++i) {
    const at::Tensor& tensor = tensors[i];
    // Explicitly handling undefined tensors here instead of letting `_to_eager`
    // handle it. Otherwise, we'd need to require all backends with their own
    // implementation of _to_eager to properly handle undefined tensors.
    if (tensor.defined()) {
      to_translate[i] = true;
      valid_tensors.push_back(tensor);
    } else {
      eager_tensors[i] = tensor;
    }
  }
  auto eager_valid_tensors = _to_eager(valid_tensors, device_type);
  for (size_t i = 0, defined_pos = 0; i < tensors.size(); ++i) {
    if (to_translate[i]) {
      eager_tensors[i] = std::move(eager_valid_tensors[defined_pos++]);
    }
  }
  return eager_tensors;
}

std::vector<std::optional<at::Tensor>> to_eager(
    const std::vector<std::optional<at::Tensor>>& tensors,
    c10::DeviceType device_type) {
  // We can't just call _to_eager() on the entire list of Tensors because it
  // will break on undefined tensors. Separate out undefined tensors first.
  std::vector<std::optional<at::Tensor>> eager_tensors(tensors.size());
  std::vector<at::Tensor> valid_tensors;
  std::vector<bool> to_translate(tensors.size());
  for (size_t i = 0; i < tensors.size(); ++i) {
    const std::optional<at::Tensor>& tensor = tensors[i];
    // Explicitly handling undefined tensors here instead of letting `_to_eager`
    // handle it. Otherwise, we'd need to require all backends with their own
    // implementation of _to_eager to properly handle undefined tensors.
    if (tensor.has_value() && tensor->defined()) {
      to_translate[i] = true;
      valid_tensors.push_back(*tensor);
    } else {
      eager_tensors[i] = tensor;
    }
  }
  auto eager_valid_tensors = _to_eager(valid_tensors, device_type);
  for (size_t i = 0, defined_pos = 0; i < tensors.size(); ++i) {
    if (to_translate[i]) {
      eager_tensors[i] = std::move(eager_valid_tensors[defined_pos++]);
    }
  }
  return eager_tensors;
}

c10::DispatchKey dispatch_key(c10::DeviceType device_type) {
  switch (device_type) {
    case at::kCPU: {
      return c10::DispatchKey::CPU;
    }
    case at::kCUDA: {
      return c10::DispatchKey::CUDA;
    }
    default: {
      TORCH_CHECK(false, "Unsupported device type: ", device_type);
    }
  }
}

std::optional<c10::Device> compute_target_device(
    std::vector<at::Tensor>& t_args,
    const std::vector<c10::List<at::Tensor>>& tlist_args,
    const std::vector<c10::List<std::optional<at::Tensor>>>& opt_tlist_args) {
  // Decide what device to move the output tensor(s) to.
  // The current convention is that we use the first tensor arg to pick the
  // device Barring that, we take the first tensor from a TensorList arg.
  if (!t_args.empty()) {
    return t_args[0].device();
  } else {
    // We need to loop through all of the (potentially multiple) TensorList
    // arguments In case, e.g. the first one is empty but the second is not.
    for (auto& tens_list : tlist_args) {
      for (const auto i : c10::irange(tens_list.size())) {
        return tens_list.get(i).device();
      }
    }
    for (auto& tens_list : opt_tlist_args) {
      for (const auto i : c10::irange(tens_list.size())) {
        auto const& e = tens_list.get(i);
        if (e.has_value()) {
          return e->device();
        }
      }
    }
  }
  return std::nullopt;
}

} // namespace

static std::unordered_map<std::string, ::torch::lazy::Counter*>
    _eager_fallback_counters;

bool force_eager_fallback(c10::Symbol op) {
  auto force_str = getLTCForceFallback();
  if (!force_str.empty()) {
    static auto force_sym = c10::Symbol::fromQualString(std::string(force_str));
    if (op == force_sym) {
      return true;
    }
  }
  if (op == at::aten::nonzero) {
    // When symbolic shape mode is not enabled, the nonzero shape function
    // returns an incorrect result.
    return !symbolicShapeEnabled();
  }

  return false;
}

void ltc_eager_fallback(
    const c10::OperatorHandle& op,
    torch::jit::Stack* stack) {
  // TODO(whc) this FN_TRACK thing hasn't been used so far in LTC iirc but could
  // land/re-enable it LTC_FN_TRACK(3);;
  const auto name = c10::toString(op.operator_name());

  // Manually applying the TORCH_LAZY_COUNTER macro.
  // We need to do it ourselves and explicitly keep a mapping of counters
  // because this boxed fallback kernel is used by multiple operators,
  // and the macro stamps out a static Counter object with a fixed name
  // at the code location that it was called.
  if (_eager_fallback_counters.find(name) == _eager_fallback_counters.end()) {
    _eager_fallback_counters[name] = new ::torch::lazy::Counter(name);
  }
  _eager_fallback_counters[name]->AddValue(1);

  auto& args = op.schema().arguments();
  auto arguments = torch::jit::last(stack, args.size());

  // Log each tensor argument.
  for (const auto& ivalue : arguments) {
    if (ivalue.isTensor()) {
      VLOG(3) << ivalue.toTensor().toString();
    }
  }

  // Call the actual boxed CPU fallback.
  ts_eager_fallback(
      op, stack, torch::lazy::getBackend()->EagerFallbackDeviceType());
}

void register_ts_ltc_eager_fallback() {
  static auto m = MAKE_TORCH_LIBRARY_IMPL(_, Lazy);
  // Most backends use TORCH_LIBRARY_* macros which perform their dispatcher
  // registrations at static library init time, but the lazy Torchscript backend
  // does not since it is built in the main torch lib but not always used.
  // In particular, if another external backend wants to register itself to the
  // same key (Lazy), Torchscript backend must not be initialized.
  m.fallback(torch::CppFunction::makeFromBoxedFunction<&ltc_eager_fallback>());
}

void ts_eager_fallback(
    const c10::OperatorHandle& op,
    torch::jit::Stack* stack,
    c10::DeviceType device_type) {
  auto& schema_args = op.schema().arguments();
  const auto num_arguments = schema_args.size();
  auto arguments = torch::jit::last(stack, num_arguments);
  const auto arguments_begin = stack->size() - num_arguments;

  std::vector<at::Tensor> tensor_args;
  std::vector<size_t> tensor_args_indices;

  std::vector<c10::List<at::Tensor>> tensorlist_args;
  std::vector<c10::List<std::optional<at::Tensor>>> opt_tensorlist_args;

  // Step 1: Convert all non-eager tensor inputs into eager tensors and put them
  // on the stack at the correct indices.
  for (size_t idx = 0; idx < arguments.size(); ++idx) {
    const auto& ivalue = arguments[idx];
    if (ivalue.isTensor()) {
      tensor_args.push_back(ivalue.toTensor());
      tensor_args_indices.push_back(idx);
    } else if (ivalue.isTensorList()) {
      // Note: we copy each TensorList argument to eager individually out of
      // convenience, but XLA would benefit from materializing all tensor and
      // TensorList args onto the CPU at the same time. We can improve this if
      // we need better perf for XLA's CPU fallbacks.
      auto eager_ivalue = c10::IValue(c10::List<at::Tensor>(
          to_eager(ivalue.toTensorVector(), device_type)));
      (*stack)[arguments_begin + idx] = std::move(eager_ivalue);
      tensorlist_args.push_back(ivalue.toTensorList());
    } else if (ivalue.isOptionalTensorList()) {
      auto eager_ivalue = c10::IValue(c10::List<std::optional<at::Tensor>>(
          to_eager(ivalue.toOptionalTensorVector(), device_type)));
      (*stack)[arguments_begin + idx] = std::move(eager_ivalue);
      opt_tensorlist_args.push_back(ivalue.toOptionalTensorList());
    }
  }
  // XLA requires all of the tensor arguments to be gathered up and converted to
  // CPU together.
  auto eager_tensors = to_eager(tensor_args, device_type);

  for (const auto i : c10::irange(tensor_args_indices.size())) {
    auto idx = tensor_args_indices[i];
    (*stack)[arguments_begin + idx] = c10::IValue(eager_tensors[i]);
  }

  // Step 2: Call the underlying eager implementation of the operator
  op.redispatchBoxed(c10::DispatchKeySet(dispatch_key(device_type)), stack);

  // Step 3: We need to take special care to handle mutable aliases properly:
  // If any input tensors are mutable aliases, we need to directly copy the
  // updated data on the eager tensors back to the original inputs.
  for (const auto i : c10::irange(tensor_args_indices.size())) {
    auto tensor_idx = tensor_args_indices[i];
    const auto alias_info = schema_args[tensor_idx].alias_info();
    if (alias_info != nullptr && alias_info->isWrite()) {
      at::_copy_from_and_resize(eager_tensors[i], tensor_args[i]);
    }
  }

  // Step 4: Convert any eager output tensors back to the original input device.
  // For mutable alias'd outputs, we also need to take special care
  // to move the ORIGINAL input tensor back onto the stack, in place of
  // the temporary eager output tensor that we created.
  //
  // Note [Eager Fallback Does Not Handle View Operators]
  // Also note that we are incapable of handling immutable alises properly.
  // Why?
  // Schemas with an immutable alias'd tensor outputs correspond to view
  // operators. For example, the `view_as` schema from native_functions.yaml:
  // `view_as(Tensor(a) self, Tensor other) -> Tensor(a)`
  // We can't handle these ops properly, because view ops are supposed to return
  // a NEW tensor that shares the SAME storage as the original tensor.
  // However, the new tensor that we created cannot share the same storage,
  // since it lives on the eager CPU / CUDA device and the original tensor lives
  // on a different device. Because of that, we warn if someone attempts to call
  // the eager fallback on a view operator (this is to maintain BC for view ops
  // for XLA that fall back to CPU).
  const auto& schema_returns = op.schema().returns();
  const auto& num_returns = schema_returns.size();
  auto returns = torch::jit::last(stack, num_returns);
  const auto returns_begin = stack->size() - num_returns;

  for (const auto idx : c10::irange(returns.size())) {
    if (returns[idx].isTensor()) {
      const auto& return_tens = returns[idx].toTensor();
      if (return_tens.defined()) {
        const auto alias_info = schema_returns[idx].alias_info();
        if (alias_info != nullptr && alias_info->isWrite()) {
          // Case (1): mutable alias case. Move the input ivalue directly onto
          // the stack in place of the existing eager output tensor.
          bool found_alias = false;
          // We could store some extra metadata on the function schema to avoid
          // the loop here if we need to improve perf.
          for (const auto i : c10::irange(tensor_args_indices.size())) {
            auto input_tensor_idx = tensor_args_indices[i];
            const auto& input_tensor = eager_tensors[i];
            const auto input_alias_info =
                schema_args[input_tensor_idx].alias_info();
            if (input_tensor.defined() && input_alias_info != nullptr &&
                *alias_info == *input_alias_info) {
              // We've found the original input tensor that aliases with the
              // current output. Wrap it in an IValue and put it directly on the
              // stack.
              (*stack)[returns_begin + idx] = c10::IValue(tensor_args[i]);
              found_alias = true;
              break;
            }
          }
          TORCH_CHECK(
              found_alias,
              "The operator ",
              op.schema().operator_name(),
              " appears to have invalid alias information. ",
              "Found a return tensor argument with a mismatched "
              "mutable alias: ",
              schema_returns[idx]);
        } else {
          std::optional<c10::Device> tgt_device = compute_target_device(
              tensor_args, tensorlist_args, opt_tensorlist_args);
          if (alias_info != nullptr && !alias_info->isWrite()) {
            // immutable alias (view) case: Warn here, since we're copying and
            // not creating a view.
            // If this operator is needed, the backend should provide a kernel
            // for it.
            // See Note [Eager Fallback Does Not Handle View Operators]
            std::stringstream dev_str;
            if (tgt_device) {
              dev_str << *tgt_device;
            } else {
              dev_str << "<none>";
            }
            // We should never hit this for a view op,
            // because LazyTensor should provide a lowering for the
            // corresponding view_copy operator. The functionalization pass will
            // take care of calling the view_copy operator intead of the view.
            TORCH_CHECK(
                false,
                "The operator ",
                op.schema().operator_name(),
                " appears to be a view operator, ",
                "but it has no implementation for the backend \"",
                dev_str.str(),
                "\". View operators don't support ",
                "falling back to run on the eager, since the tensor's "
                "storage cannot be shared across devices.");
          }
          // Case (2): copy case. Copy the eager output tensor to the original
          // device.

          // We technically  might not have a target device, e.g. if you call
          // torch.cat() with an empty list In that case, we shouldn't have any
          // tensors to schlep across devices anyway.
          if (tgt_device) {
            (*stack)[returns_begin + idx] =
                c10::IValue(returns[idx].toTensor().to(*tgt_device));
          }
        }
      }
    }
  }
}

} // namespace torch::lazy
