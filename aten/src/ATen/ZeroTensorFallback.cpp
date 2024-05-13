#include <ATen/ATen.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/core/op_registration/op_registration.h>
#include <ATen/native/UnaryOps.h>
#include <ATen/NativeFunctions.h>
#include <c10/util/irange.h>
#include <torch/library.h>
#include <ATen/native/MathBitFallThroughLists.h>

namespace at {

  // TODO: add a note explaining the design decisions
  // ZeroTensors are designed to be immutable. Thus, we error out when an in-place operation is performed on ZeroTensors
  static void zeroTensorFallback(const c10::OperatorHandle& op, DispatchKeySet dispatch_keys, torch::jit::Stack* stack) {
    const auto& arguments = op.schema().arguments();
    const auto num_arguments = arguments.size();
    const auto stack_start = stack->size() - num_arguments;

    c10::optional<bool> is_write;
    for (const auto i : c10::irange(num_arguments)) {
      const auto& alias_info = arguments[i].alias_info();
      if (alias_info != nullptr) {
        if (is_write.has_value()) {
          TORCH_CHECK(*is_write == alias_info->isWrite(),
            "Unsupported operator for ", "ZeroTensorFallback: ", op.schema().name(),
            "ZeroTensor fallback doesn't work for operators with a mix "
            "mutable and non-mutable inputs that alias with outputs, "
            "this must be implemented manually.  "
            "If you got this error on a core op, please report a bug to PyTorch.");
        } else {
          is_write = alias_info->isWrite();
        }
      }
    }

    if (is_write.has_value() && !*is_write) {
      // We assume that view operators automatically handle the ZeroTensor bit
      // correctly by propagating the dispatch key in key_set.
      // This is not necessarily always right, so you should test these cases.
      op.redispatchBoxed(dispatch_keys & c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::ZeroTensor), stack);
      return;
    }

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
        auto tensor = std::move(ivalue).toTensor();
        if (tensor._is_zerotensor()) {
          TORCH_CHECK(!mut_arg, "ZeroTensors are immutable. Please use the materialized zero tensor ",
                    "obtained using .clone() if you want a mutable tensor.");
          tensor = at::zeros({}, tensor.options()).expand(tensor.sizes());
        }
        (*stack)[stack_start + i] = std::move(tensor);
      } else if (ivalue.isTensorList()) {
        auto tensors = std::move(ivalue).toTensorList();
        for(const auto j : c10::irange(tensors.size())) {
          const Tensor& tensor = tensors[j];
          if (tensor._is_zerotensor()) {
            // TODO: assert requires_grad=False
            //_like should not propagate zerotensor dispatch key
            TORCH_CHECK(!mut_arg, "ZeroTensors are immutable. Please use the materialized zero tensor ",
                    "obtained using .clone() if you want a mutable tensor.");
            tensors[j] = at::zeros({}, tensor.options()).expand(tensor.sizes());
          }
        }
        (*stack)[stack_start + i] = std::move(tensors);
      }
    }

    op.redispatchBoxed(dispatch_keys & c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::ZeroTensor), stack);
  }


  TORCH_LIBRARY_IMPL(_, ZeroTensor, m) {
    m.fallback(torch::CppFunction::makeFromBoxedFunction<&zeroTensorFallback>());
  }

  TORCH_LIBRARY_IMPL(aten, ZeroTensor, m) {
    m.impl("zeros_like", torch::CppFunction::makeFallthrough());
    m.impl("mul.Scalar", torch::CppFunction::makeFallthrough());
    m.impl("add.Scalar", torch::CppFunction::makeFallthrough());
    m.impl("copy_", torch::CppFunction::makeFallthrough());
    m.impl("clone", torch::CppFunction::makeFallthrough());
    m.impl("dot", torch::CppFunction::makeFallthrough());
    m.impl("vdot", torch::CppFunction::makeFallthrough());
    // The functions in the list below have a specific registeration in native_functions.yaml and
    // do not use the fallback.
    // m.impl("mul.Tensor", torch::CppFunction::makeFallthrough());
    // m.impl("add.Tensor", torch::CppFunction::makeFallthrough());
    // m.impl("linalg_cross", torch::CppFunction::makeFallthrough());

    TORCH_VIEW_FNS(m)
    TENSOR_UTILITIES_AND_CONSTRUCTORS(m)
  }
} // namespace at
