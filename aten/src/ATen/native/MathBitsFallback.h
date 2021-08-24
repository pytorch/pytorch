#include <ATen/ATen.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/core/op_registration/op_registration.h>
#include <ATen/native/UnaryOps.h>
#include <ATen/NativeFunctions.h>
#include <c10/util/irange.h>
#include <torch/library.h>

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
    /*
      Situations to handle:
        1. Out-of-place operation.  Easy: materialize all inputs and
          call it a day.
        2. Inplace operation.  Desugar x.add_(2) into x.conj_().add_(2).conj_().
          Materialize other inputs as in (1).
        3. out= operation.  Desugar add(x, 2, out=y) into y.copy_(add(x, 2))
        Materialize other inputs as in (1).

        It is important to be able to tell if we READ from an argument and if we
        WRITE from an argument.  Conservative approach is to assume that we always
        READ from an argument, but in out-of-place operations you can skip
        conjugating inputs on entry that never get used.  In current schema we
        can't easily tell if inplace situation has happened, so don't do it.

      Algorithm used:

      1. Scan through the input arguments
          a. Create a vector of mutable arguments that have the math bit set to True.
          a. Additionally, if input argument has alias_info and it's mutable, set the optional variable is_write to True, else False.
          b. is_write is only ever set for view, in-place or out= operations.
      2. If is_write has value, and is set to False, then the operation is a view operation.
          a. Perform the operation and return.
      3. Iterate through all the input arguments again.
          a. At this time, only non-mutable arguments are considered.
          b. If the size of mutable_arguments vector is non-zero, then we iterate through mutable_inputs to check
            if the current arg tensor in question aliases any of the entries in mutable_inputs.
          c. If it does and the non-mutable arg's math bit is set to False, then we clone the non-mutable tensor arg,
             else it is a no op.
          d. If it doesn't then just resolve the non-mutable arg as usual.
      4. Now we look through the mutable_inputs vector (which contains only mutable input tensors with math bit set to True).
          a. In-place materialize the bit (math_op_) for each of the entries in the vector.
      5. Perform the actual operation.
      6. Perform math_op_ for each of the mutable argument tensors.

    */

    /*
      Different possible cases for shared memory:
      1. Functions with no tensorlist inputs
        a. no mutable args
        b. one or more mutable args
          b.1. shared memory between mutable and non-mutable args (--handled)
          b.2. shared memory between two or more mutable args (incorrect result but this is bad
                and users shouldn't do it anyway)
          b.3. shared memory between two or more non-mutable args (works fine since we never modify the memory)
          b.4. no shared memory between args (-- works fine)
      2. Functions with tensorlist inputs
        a. no mutable args (-- works fine)
        b. Mutable tensor arg(s) but non-mutable tensorlist arg
          b.1. All the possible cases listed in 1.b (-- works fine)
          b.2. shared memory between a tensor arg and a tensorlist arg ( -- Not currently supported)
        c. Mutable tensorlist arg(s) ( -- Not currently supported)
          c.1. shared memory between a mutable and non-mutable tensorlist
          c.2. shared memory between two or more mutable tensorlist args
          c.3. shared memory between two or more non-mutable tensorlist args
          c.4. shared memory between two or more non-mutable tensor args
          c.5. shared memory between a tensor and tensorlist arg
          ...
          c.n. no shared memory between args
    */
    const auto& arguments = op.schema().arguments();
    const auto num_arguments = arguments.size();
    const auto stack_start = stack->size() - num_arguments;

    // set to True if there's one or mutable inputs
    c10::optional<bool> is_write;

    // Mutable inputs to be tracked separately
    std::vector<Tensor> mutable_inputs;
    for (const auto i : c10::irange(num_arguments)) {
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
          auto& ivalue = (*stack)[stack_start + i];
          //TODO: add a table listing all possible cases and clearly state what works/is allowed
          if (ivalue.isTensor()) {
            const auto& mut_arg_tensor = ivalue.toTensor();
            if (is_bit_set(mut_arg_tensor)) {
              mutable_inputs.emplace_back(mut_arg_tensor);
            }
          } else {
            TORCH_CHECK(false, op_name, " fallback doesn't currently support mutable TensorLists.",
              "Please materialize the ", op_name, " tensor(s) before calling ", op.schema().name());
          }
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

    // updates for non-mutable inputs
    bool check_for_alias_with_mut_arg = mutable_inputs.size() != 0;
    for (const auto i : c10::irange(num_arguments)) {
      auto& ivalue = (*stack)[stack_start + i];
      if (!(ivalue.isTensor() || ivalue.isTensorList())) {
        continue;
      }
      const auto& argument = arguments[i];
      if (argument.alias_info()) {
        // Was already tested by is_write loop above
        TORCH_INTERNAL_ASSERT_DEBUG_ONLY(argument.alias_info()->isWrite());
        continue;
      }
      if (ivalue.isTensor()) {
        auto tensor = std::move(ivalue).toTensor();
        TORCH_CHECK_NOT_IMPLEMENTED(!tensor.is_meta(), op_name, " fallback does not support meta tensors.");
        bool resolve_needed = true;
        if (check_for_alias_with_mut_arg) {
          for (const auto& mutable_input : mutable_inputs) {
            // check if tensor shares memory with one of the mutable tensors
            // with math bit set to True.
            // This check is crucial since we in-place materialize the bit for the mutable tensor(s) which
            // might be sharing memory with one of the non-mutable tensors that could be reading into the wrong
            // values if their math bit is not set to True.
            // no op if tensor and mutable arg share memory and have math bit set to True
            if (tensor.is_alias_of(mutable_input)) {
              if (!is_bit_set(tensor)) {
                tensor = tensor.clone();
              }
              resolve_needed = false;
              break;
            }
          }
        }
        if (resolve_needed) {
          tensor = resolve_bit(tensor);
        }
        (*stack)[stack_start + i] = std::move(tensor);
      } else {
        TORCH_CHECK(!is_write.has_value(), op_name, " fallback doesn't currently support operators with TensorLists and mutable inputs.",
              "Please materialize the ", op_name, " tensor(s) before calling ", op.schema().name());
        auto tensors = std::move(ivalue).toTensorList();
        for(const auto j : c10::irange(tensors.size())) {
          tensors[j] = resolve_bit(tensors[j]);
        }
        (*stack)[stack_start + i] = std::move(tensors);
      }
    }

    // updates for mutable inputs
    for (auto& mutable_input : mutable_inputs) {
      _set_bit(mutable_input, false);
      math_op_(mutable_input);
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
