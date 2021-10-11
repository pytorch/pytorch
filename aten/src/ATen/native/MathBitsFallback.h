#include <ATen/ATen.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/core/op_registration/op_registration.h>
#include <ATen/native/UnaryOps.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/Resize.h>
#include <c10/util/irange.h>
#include <torch/library.h>

namespace at {
namespace native {
// This fallback should only be used for operations that are self inverse and have a corresponding tensor
// bit (internally implemented using DispatchKey) to maintain the state on tensor using tensor bit.
// Currently there are two tensor bits that trigger this fallback: conjugate bit and negative bit.
// Conjugate bit is set on a tensor when `.conj()` is called and neg bit is set on a tensor when `.conj().imag` is called.

// NOTE: To use this fallback, `clone` and `copy_` should fully understand and be able to correctly handle the semantic of your math bit.
struct MathOpFallback {
  MathOpFallback(DispatchKey key_, string op_name_) : key(key_), op_name(op_name_) {}
  virtual bool is_bit_set(const Tensor&) = 0;
  void fallback_impl(const c10::OperatorHandle& op, DispatchKeySet dispatch_keys, torch::jit::Stack* stack) {
    /*
      Situations to handle:
        1. Out-of-place operation.  Easy: materialize all inputs and
          call it a day.
        2. Inplace operation.  Desugar x.add_(2) into x.conj_().add_(2).conj_().
          Materialize other inputs as in (1).
        3. out= operation.  Desugar add(x, 2, out=y) into y.copy_(add(x, 2))
        Materialize other inputs as in (1).

        It is important to be able to tell if we READ from an argument and if we
        WRITE to an argument.  Conservative approach is to assume that we always
        READ from an argument, but in out= operations you can skip
        conjugating inputs on entry that never get used. In the current schema we
        can't easily tell if the operation is in in-place or out= operation.

        Note:
        1. Mutable tensorlists containing tensors whose math bit set to true are disallowed.
        2. Mutable tensors with math bit set to true are unconditionally cloned to ensure
           correct behavior in the case when the mutable tensor shares memory with non mutable arguments. 
           
           If we were to in-place resolve the math bit for mutable inputs, then the non-mutable inputs sharing partial or full memory
           with these mutable inputs would read into wrong values in the following cases:
           1. Non mutable inputs have their math bit set to false.
           2. Math bit for mutable input(s) is resolved before the non mutable inputs (with bit set to true and sharing memory with one or more mutable arg(s)) are cloned.
           
           At the end, the final value of the mutable arguments from the stack are copied into the original input mutable tensor inputs.
    */
    const auto& arguments = op.schema().arguments();
    const auto num_arguments = arguments.size();
    const auto stack_start = stack->size() - num_arguments;

    c10::optional<bool> is_write;
    for (const auto i : c10::irange(num_arguments)) {
      // Three possible states:
      // 1. alias_info has no value --> out-of-place operation
      // 2. alias_info does have a value, alias_info->is_write=True --> in-place or out= operation
      // 3. alias_info does have a value, alias_info->is_write=False --> view operation
      const AliasInfo* alias_info = arguments[i].alias_info();
      if (alias_info != nullptr) {
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

    // Mutable inputs with math bit set to True to be tracked separately
    std::vector<std::pair<int, Tensor>> mutable_inputs;
    for (const auto i : c10::irange(num_arguments)) {
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
          mutable_inputs.emplace_back(std::make_pair(i, tensor));
        } 
        tensor = at::clone(tensor);
        (*stack)[stack_start + i] = std::move(tensor);
      } else if (ivalue.isTensorList()) {
        auto tensors = std::move(ivalue).toTensorList();
        bool not_a_mut_arg = !mut_arg;
        for(const auto j : c10::irange(tensors.size())) {
          const auto& tensor = tensors[j];
          if (!is_bit_set(tensor)) {
            continue;
          }
          TORCH_CHECK(not_a_mut_arg, " fallback doesn't currently support mutable TensorLists with ",
              op_name, " inputs. Please materialize all the ", op_name, " input tensor(s) in the mutable TensorList inputs before calling ", op.schema().name());
          tensors[j] = at::clone(tensor);
        }
        (*stack)[stack_start + i] = std::move(tensors);
      }
    }

    op.redispatchBoxed(dispatch_keys & c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, key), stack);

    for (std::pair<int, Tensor> &index_and_mutable_input : mutable_inputs) {
      int i = index_and_mutable_input.first;
      auto& mut_arg = index_and_mutable_input.second;
      auto& ivalue = (*stack)[stack_start + i];
      auto tensor = std::move(ivalue).toTensor();
      at::native::resize_output(mut_arg, tensor.sizes());
      mut_arg.copy_(tensor);  
      (*stack)[stack_start + i] = std::move(mut_arg);
    }
  }

  virtual ~MathOpFallback() = default;

  DispatchKey key;
  string op_name;
};
}
}// namespace at
