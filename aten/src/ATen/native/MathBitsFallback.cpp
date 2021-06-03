#include <ATen/ATen.h>
#include <ATen/core/op_registration/op_registration.h>
#include <torch/library.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/native/UnaryOps.h>
#include <ATen/NativeFunctions.h>

namespace at {

struct MathOpFallback {
  MathOpFallback(DispatchKey key_, string op_name_) : key(key_), op_name(op_name_) {}
  virtual bool is_bit_set(const Tensor&) = 0;
  virtual void set_bit(const Tensor&, bool) = 0;
  virtual Tensor resolve_bit(const Tensor&) = 0;
  virtual Tensor& math_op_(Tensor&) = 0;
  void linalg_fallback(const c10::OperatorHandle& op, DispatchKeySet dispatch_keys, torch::jit::Stack* stack) {
    std::cout<<"Entering linalg_fallback";
    const auto& arguments = op.schema().arguments();
    const auto num_arguments = arguments.size();
    const auto stack_start = stack->size() - num_arguments;

    // Mutable inputs to be tracked separately
    std::vector<Tensor> mutable_inputs;

    for (int64_t i = 0; i < num_arguments; ++i) {
      auto& ivalue = (*stack)[stack_start + i];
      if (!ivalue.isTensor() || !ivalue.isTensorList()) {
        continue;
      }
      const auto& argument = arguments[i];
      bool mut_arg = false;
      if (argument.alias_info()) {
        // Was already tested by is_write loop above
        TORCH_INTERNAL_ASSERT_DEBUG_ONLY(argument.alias_info()->isWrite());
        mut_arg = true;
      } else {
        continue;
      }

      if (ivalue.isTensor()) {
        if (!is_bit_set(ivalue.toTensor())) {
          continue;
        }

        auto tensor = std::move(ivalue).toTensor();
        TORCH_CHECK_NOT_IMPLEMENTED(!tensor.is_meta(), op_name, " fallback does not support meta tensors.");
        if (mut_arg) {
          // TODO: This is a waste if the argument is write only
          set_bit(tensor, false);
          math_op_(tensor);
          mutable_inputs.emplace_back(tensor);
        }
        (*stack)[stack_start + i] = std::move(tensor);
      }
    }

    op.redispatchBoxed(dispatch_keys & c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, key), stack);

    for (auto& mutable_input : mutable_inputs) {
      math_op_(mutable_input);
      set_bit(mutable_input, true);
    }
  }
  void fallback_impl(const c10::OperatorHandle& op, DispatchKeySet dispatch_keys, torch::jit::Stack* stack) {
    // This fallback can be used for lazy math operations for which tensors maintain a corresponding dispatch key.
    // At the time of writing, there are two bits that can be set on a tensor: conj and neg. The explanation below uses
    // tensors with conj bit set as an example, but this is also applicable for tensors with neg bit.
    // Situations to handle:
    //  1. Purely functional situation.  Easy: materialize all inputs and
    //     call it a day.
    //  2. Inplace operation.  Desugar x.add_(2) into x.conj_().add_(2).conj_().
    //     Materialize other inputs as in (1).
    //  3. Out-of-place operation.  Desugar add(x, 2, out=y) into y.copy_(add(x, 2))
    //  Materialize other inputs as in (1).
    //
    //  It is important to be able to tell if we READ from an argument and if we
    //  WRITE from an argument.  Conservative approach is to assume that we always
    //  READ from an argument, but in out-of-place operations you can skip
    //  negating inputs on entry that never get used.  In current schema we
    //  can't easily tell if inplace situation has happened, so don't do it.

    //  std::cerr << "conj fallback " << op.schema().name() << "\n";
    const auto& arguments = op.schema().arguments();
    const auto num_arguments = arguments.size();

    c10::optional<bool> is_write;
    for (int64_t i = 0; i < num_arguments; ++i) {
      const auto& alias_info = arguments[i].alias_info();
      if (alias_info.has_value()) {
        if (is_write.has_value()) {
          TORCH_CHECK(*is_write == alias_info->isWrite(),
            "Unsupported operator for ", op_name, " fallback: ", op.schema().name(),
            op_name, " fallback doesn't work for operators with a mix "
            "mutable and non-mutable inputs that alias with outputs, "
            "this must be implemented manually.  "
            "If you got this error on a core op, please report a bug to PyTorch.");
        } else {
          is_write = alias_info->isWrite();
        }
      }
    }

    if (is_write.has_value() && !*is_write) {
      // We assume that view operators automatically handle the math bit
      // correctly by propagating the dispatch key in key_set.
      // This is not necessarily always right, so you should test these cases.
      op.redispatchBoxed(dispatch_keys & c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, key), stack);
      return;
    }

    // Mutable inputs to be tracked separately
    std::vector<Tensor> mutable_inputs;

    for (int64_t i = 0; i < num_arguments; ++i) {
      auto& ivalue = (*stack)[stack_start + i];
      if (!ivalue.isTensor()) {
        continue;
      }
      const auto& argument = arguments[i];
      bool mut_arg = false;
      if (argument.alias_info()) {
        // Was already tested by is_write loop above
        TORCH_INTERNAL_ASSERT_DEBUG_ONLY(argument.alias_info()->isWrite());
        mut_arg = true;
      }
      if (ivalue.isTensor()) {
        if (!is_bit_set(ivalue.toTensor())) {
          continue;
        }

        auto tensor = std::move(ivalue).toTensor();
        TORCH_CHECK_NOT_IMPLEMENTED(!tensor.is_meta(), op_name, " fallback does not support meta tensors.");
        if (mut_arg) {
          // TODO: This is a waste if the argument is write only
          set_bit(tensor, false);
          math_op_(tensor);
          mutable_inputs.emplace_back(tensor);
        } else {
          tensor = resolve_bit(tensor);
        }
        (*stack)[stack_start + i] = std::move(tensor);
      } else if (ivalue.isTensorList()) {
          auto tensors = std::move(ivalue).toTensorList();
          for(const auto j : c10::irange(tensors.size())) {
            // At the time of writing this, no operators use tensorlists with mutable tensors.
            // We could add additional code logic in the future if this changes.
            TORCH_CHECK(!mut_arg, op_name, " fallback doesn't work for mutable TensorLists.");
            tensors[j] = resolve_bit(tensors[j]);
          }
          (*stack)[stack_start + i] = std::move(tensors);
        }
      }

    op.redispatchBoxed(dispatch_keys & c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, key), stack);

    for (auto& mutable_input : mutable_inputs) {
      math_op_(mutable_input);
      set_bit(mutable_input, true);
    }
  }
  DispatchKey key;
  string op_name;
};

} // namespace at
