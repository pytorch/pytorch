#include <ATen/ATen.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/core/op_registration/op_registration.h>
#include <ATen/native/UnaryOps.h>
#include <ATen/NativeFunctions.h>
#include <c10/util/irange.h>
#include <torch/library.h>
#include <ATen/native/MathBitFallThroughLists.h>

namespace at {

  /*
  * The goal of ZeroTensors is to replace the undefined tensors used in the forward
  * and backward mode AD which are currently materialized to a tensor full of zeros
  * with a new type of zero tensor that use O(1) memory (tensors with a storage
  * pointing to nullptr). Having undefined tensors is a common occurrence in the
  * forward mode AD. Currently these undefined tensors are materialized to zero
  * tensors. Not only the materialization is wasteful, but the cases where
  * computationally heavy operations like matrix multiplies are called for these
  * tensors along with some non-zero tensors provide opportunities for perf
  * improvement. With zero tensors, expensive operations like matmul on zero
  * tensors will become O(1).
  *
  * Note that as of writing this, ZeroTensors do not propagate nans or other
  * extreme values. This is rooted in how the kernels are implemented for
  * ZeroTensors. For example, ZT * (non-ZT) = ZT, i.e., we don't check whether the
  * non-ZT has nans for example.
  *
  * Design:
  *
  * 1. ZeroTensors are regular tensors with TensorOptions (as usual), and a storage
  *    pointing to nullptr (aliasing semantic is correctly maintained), and a
  *    ZeroTensor dispatch key set.
  *
  * 2. Zero tensors are immutable.
  *     - As a result, in-place operations on these tensors will be disallowed. This is
  *       done to prevent data race in the case of multithreading (when two threads try
  *       to read the same zero tensor and materialize it in-place).
  *     - Zero tensors shouldn’t be passed as an output tensor in out= operations.
  *
  * 3. Similar to Conjugate and Negative dispatch keys, ZeroTensor keys also has a
  *    corresponding boxed fallback for dispatch key resolution. Most functions will
  *    use this fallback if the ZeroTensor key is set on one or more inputs. A select
  *    few functions that are registered as fallthrough kernels or have a special
  *    kernel registered in native_functions.yaml will bypass this fallback (mul,
  *    matmul, tensor views, etc.).
  *     - This fallback materializes each ZeroTensor in the following way:
  *       at::zeros({}, tensor.options()).expand(tensor.sizes()); where tensor is an
  *       efficient zero tensor (tensor with ZeroTensor dispatch key set)
  *     - If there’s a mutable ZeroTensor , then we simply error out to avoid the race
  *       condition (as mentioned above).
  *
  * 4. ZeroTensors are handled below autograd. This is necessary because we want the
  *    materialization to happen below the autograd to ensure gradients are populated
  *    on the correct tensors.
  *     - Example: Consider add in the case it was using the fallback: zerotensor_a + b.
  *       zerotensor_a would be materialized to c=torch.zeros_like(zerotensor_a) after
  *       passing through the fallback. If this happens above the autograd, then the
  *       gradients would be populated on c instead of zerotensor_a.
  *
  * 5. The grad field is always populated with an honest to goodness tensor. This
  *    materialization of zero tensors will happen in:
  *     - AcccumulateGrad for Backward Mode AD.
  *     - will never be required for ForwardMode AD.
  *       - This is because if all the tangents were undefined (efficient zero tensors),
  *         no computation will be performed (this is ensured via an existing pre-check).
  *       - If one or more tangent is non-zero (undefined), then even if the Jacobian is
  *         zero, we never get an efficient zero tensor as an output (since we don’t
  *         compute efficient zero tensors in Jacobian computation, i.e., Jacobian can
  *         never be a ZeroTensor). This will not be true anymore if ZeroTensors were
  *         exposed in the Python API.
  *
  * 6. torch.mul(efficient_zero_tensor, other) will return an efficient_zero_tensor.
  *    We aim to add similar specialized handling for other relevant functions based
  *    on: https://fb.quip.com/Dv5hARK0xyTo
  *
  * 7. All instance of zeros_like should be replaced with a call to
  *    _efficientzerotensor. This would ensure that we are taking maximally
  *    optimizing the performance wherever we can (combined with not using ZeroTensor
  *    fallback for functions where the function computation can be short circuited
  *    with the added information that one of the inputs is a zero tensor).
  */
  // ZeroTensors are designed to be immutable. Thus, we error out when an in-place operation is performed on ZeroTensors
  static void zeroTensorFallback(const c10::OperatorHandle& op, DispatchKeySet dispatch_keys, torch::jit::Stack* stack) {
    const auto& arguments = op.schema().arguments();
    const auto num_arguments = arguments.size();
    const auto stack_start = stack->size() - num_arguments;

    std::optional<bool> is_write;
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
    // The functions in the list below have a specific registration in native_functions.yaml and
    // do not use the fallback.
    // m.impl("mul.Tensor", torch::CppFunction::makeFallthrough());
    // m.impl("add.Tensor", torch::CppFunction::makeFallthrough());
    // m.impl("linalg_cross", torch::CppFunction::makeFallthrough());

    TORCH_VIEW_FNS(m)
    TENSOR_UTILITIES_AND_CONSTRUCTORS(m)
  }
} // namespace at
