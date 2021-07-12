#include <ATen/ATen.h>
#include <ATen/core/op_registration/op_registration.h>
#include <torch/library.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/native/UnaryOps.h>
#include <ATen/NativeFunctions.h>

namespace at {

// This fallback should only be used for operations that are self inverse and have a corresponding tensor
// bit (internally implemented using DispatchKey) to maintain the state on tensor using tensor bit.
// Currently there are two tensor bits that trigger this fallback: conjugate bit and negative bit.
// Conjugate bit is set on a tensor when `.conj()` is called and neg bit is set on a tensor when `.conj().imag` is called.

struct MathOpFallback {
  MathOpFallback(DispatchKey key_, string op_name_) : key(key_), op_name(op_name_) {}
  virtual bool is_bit_set(const Tensor&) = 0;
  virtual void _set_bit(const Tensor&, bool) = 0;
  // materializes the bit, i.e., returns a new tensor tensor containing the true output
  // (after performing the math operation corresponding to the tensor bit) if the bit is set to 1
  // else returns self.
  virtual Tensor resolve_bit(const Tensor&) = 0;
  // in-place operation corresponding to the math op represented by the bit. Im the future if this class
  // is generalized for ops that are not self inverse, then this must be replaced by op_inverse_inplace
  virtual Tensor& math_op_(Tensor&) = 0;
  void fallback_impl(const c10::OperatorHandle& op, DispatchKeySet dispatch_keys, torch::jit::Stack* stack) {
    // Situations to handle:
    //  1. Out-of-place operation.  Easy: materialize all inputs and
    //     call it a day.
    //  2. Inplace operation.  Desugar x.add_(2) into x.conj_().add_(2).conj_().
    //     Materialize other inputs as in (1).
    //  3. out= operation.  Desugar add(x, 2, out=y) into y.copy_(add(x, 2))
    //  Materialize other inputs as in (1).
    //
    //  It is important to be able to tell if we READ from an argument and if we
    //  WRITE from an argument.  Conservative approach is to assume that we always
    //  READ from an argument, but in out-of-place operations you can skip
    //  conjugating inputs on entry that never get used.  In current schema we
    //  can't easily tell if inplace situation has happened, so don't do it.

    const auto& arguments = op.schema().arguments();
    const auto num_arguments = arguments.size();
    const auto stack_start = stack->size() - num_arguments;

    c10::optional<bool> is_write;
    for (int64_t i = 0; i < num_arguments; ++i) {
      // Three possible states:
      // 1. alias_info has no value --> out-of-place operation
      // 2. alias_info does have a value, alias_info->is_write=True --> in-place or out= operation
      // 3. alias_info does have a value, alias_info->is_write=False --> view operation
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
      if (!(ivalue.isTensor() || ivalue.isTensorList())) {
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
          _set_bit(tensor, false);
          math_op_(tensor);
          mutable_inputs.emplace_back(tensor);
        } else {
          tensor = resolve_bit(tensor);
        }
        (*stack)[stack_start + i] = std::move(tensor);
      } else if (ivalue.isTensorList()) {
        auto tensors = std::move(ivalue).toTensorList();
        if (mut_arg) {
          for(const auto j : c10::irange(tensors.size())) {
            Tensor t = tensors[j];
            _set_bit(t, false);
            math_op_(t);
            mutable_inputs.emplace_back(t);
          }
        } else {
          for(const auto j : c10::irange(tensors.size())) {
            tensors[j] = resolve_bit(tensors[j]);
          }
        }
        (*stack)[stack_start + i] = std::move(tensors);
      }
    }

    op.redispatchBoxed(dispatch_keys & c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, key), stack);

    for (auto& mutable_input : mutable_inputs) {
      math_op_(mutable_input);
      _set_bit(mutable_input, true);
    }
  }

  virtual ~MathOpFallback() = default;

  DispatchKey key;
  string op_name;
};

} // namespace at
