#include <ATen/FakeTensor.h>
#include <ATen/FakeTensorDispatchTables.h>
#include <ATen/SparseCsrTensorUtils.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <c10/core/impl/FakeTensorModeTLS.h>
#include <c10/core/impl/LocalDispatchKeySet.h>
#include <c10/util/irange.h>
#include <torch/library.h>

#include <algorithm>
#include <optional>
#include <string_view>

// The C++ counterpart of FakeTensorMode.__torch_dispatch__ for C++ fakes.
// It runs an op whose tensor inputs are fakes of one mode on one device: the
// op runs on the inputs' meta storage through its Meta kernel, and the new
// outputs become fakes of that mode on that device, as Python's
// wrap_meta_outputs_with_default_device_logic does. Other ops raise for now.
// Ops that Python runs through a Python implementation (see fake_try_* in
// c10/core/impl/PyInterpreter.h) raise too. Python decompositions are not
// applied yet: an op that Python decomposes runs its Meta kernel here.

namespace {

template <typename Fn>
void for_each_tensor(
    torch::jit::Stack* stack,
    size_t begin,
    size_t count,
    const Fn& fn) {
  for (const auto idx : c10::irange(count)) {
    const auto& ivalue = (*stack)[begin + idx];
    if (ivalue.isTensor()) {
      fn(ivalue.toTensor());
    } else if (ivalue.isTensorList()) {
      for (const at::Tensor& t : ivalue.toTensorList()) {
        fn(t);
      }
    } else if (ivalue.isOptionalTensorList()) {
      for (const std::optional<at::Tensor>& t : ivalue.toOptionalTensorList()) {
        if (t.has_value()) {
          fn(*t);
        }
      }
    }
  }
}

[[noreturn]] void unsupported(
    const c10::OperatorHandle& op,
    std::string_view what) {
  TORCH_CHECK_NOT_IMPLEMENTED(
      false,
      "The C++ fake tensor fallback does not support ",
      what,
      " yet, found in ",
      op.operator_name());
}

bool contains_tensor_types(const c10::TypePtr& type) {
  if (type->isSubtypeOf(*c10::TensorType::get())) {
    return true;
  }
  const auto contained = type->containedTypes();
  return std::any_of(contained.begin(), contained.end(), contains_tensor_types);
}

// Same as Python's _is_tensor_constructor in fake_impls.py
bool is_tensor_constructor(const c10::FunctionSchema& schema) {
  const auto& arguments = schema.arguments();
  return std::none_of(
             arguments.begin(),
             arguments.end(),
             [](const c10::Argument& a) {
               return contains_tensor_types(a.type());
             }) &&
      schema.returns().size() == 1 &&
      schema.returns()[0].type()->kind() == c10::TypeKind::TensorType;
}

// Whether Python's FakeTensorMode runs op through a Python implementation: a
// registry mirrored in the fake dispatch tables, or an op_implementations
// handler matched by a predicate
bool has_python_fake_impl(const c10::OperatorHandle& op) {
  using at::impl::FakeDispatchCategory;
  const auto& name = op.operator_name();
  for (const auto category :
       {FakeDispatchCategory::OpImpl,
        FakeDispatchCategory::CustomOpImpl,
        FakeDispatchCategory::PrimMeta,
        FakeDispatchCategory::PythonCIA}) {
    if (at::impl::fakeDispatchTableContains(category, name)) {
      return true;
    }
  }
  return op.hasTag(at::Tag::dynamic_output_shape) ||
      op.hasTag(at::Tag::data_dependent_output) ||
      is_tensor_constructor(op.schema()) ||
      name.name.rfind("aten::_foreach_", 0) == 0;
}

void fakeFallback(
    const c10::OperatorHandle& op,
    c10::DispatchKeySet /*dispatchKeySet*/,
    torch::jit::Stack* stack) {
  const auto& schema = op.schema();
  const auto num_arguments = schema.arguments().size();
  const auto arguments_begin = stack->size() - num_arguments;

  // The active mode, or, as Python's FakeTensor dispatch enters a fake's
  // mode, the mode of the first fake input
  auto mode = c10::impl::FakeTensorModeTLS::get_state();
  if (mode == nullptr) {
    for_each_tensor(
        stack, arguments_begin, num_arguments, [&](const at::Tensor& t) {
          if (mode == nullptr && t.defined() && t.is_fake()) {
            mode = t.unsafeGetTensorImpl()->fake_tensor_mode();
          }
        });
  }
  if (mode == nullptr) {
    unsupported(op, "fake tensors without a mode");
  }
  if (has_python_fake_impl(op)) {
    unsupported(op, "ops with a Python fake implementation");
  }

  for (const auto idx : c10::irange(num_arguments)) {
    if ((*stack)[arguments_begin + idx].isDevice()) {
      unsupported(op, "device arguments");
    }
  }

  // The op's tensor inputs, so output wrapping can tell an input returned as
  // an output (by in-place and out= ops) from a new tensor
  c10::SmallVector<const c10::TensorImpl*, 8> input_impls;
  std::optional<c10::Device> device;
  c10::DispatchKeySet meta_keyset;
  for_each_tensor(
      stack, arguments_begin, num_arguments, [&](const at::Tensor& t) {
        if (!t.defined()) {
          return;
        }
        auto* impl = t.unsafeGetTensorImpl();
        // A Python number passed for a Tensor argument, which the Meta kernel
        // takes as is, as Python's FakeTensorMode does
        if (impl->is_wrapped_number()) {
          return;
        }
        if (!t.is_fake()) {
          unsupported(op, "non-fake tensor inputs");
        }
        if (impl->fake_tensor_mode() != mode) {
          unsupported(op, "fake tensors of another mode");
        }
        if (impl->real_tensor() != nullptr) {
          unsupported(op, "propagate_real_tensors");
        }
        if (mode->get_constant(impl) != nullptr) {
          unsupported(op, "constant inputs");
        }
        input_impls.push_back(impl);
        const auto fake_device = impl->fake_device();
        TORCH_INTERNAL_ASSERT(fake_device.has_value());
        if (!device.has_value()) {
          device = fake_device;
        } else if (*device != *fake_device) {
          unsupported(op, "inputs on different devices");
        }
        switch (t.layout()) {
          case c10::Layout::Sparse:
            meta_keyset = meta_keyset.add(c10::DispatchKey::SparseMeta);
            break;
          case c10::Layout::SparseCsr:
          case c10::Layout::SparseCsc:
          case c10::Layout::SparseBsr:
          case c10::Layout::SparseBsc:
            meta_keyset = meta_keyset.add(c10::DispatchKey::SparseCsrMeta);
            break;
          default:
            break;
        }
      });
  if (!device.has_value()) {
    unsupported(op, "ops without fake tensor inputs");
  }
  if (meta_keyset.empty()) {
    meta_keyset = c10::DispatchKeySet(c10::DispatchKey::Meta);
  }

  // As in Python, the outputs of alias and detach of a fake keep its recorded
  // dispatch keys (see Note [C++ fake dispatch keys]), and other new outputs
  // have none
  std::optional<c10::DispatchKeySet> output_dispatch_keys;
  if (schema.overload_name().empty() &&
      (schema.name() == "aten::alias" || schema.name() == "aten::detach")) {
    const auto& self = (*stack)[arguments_begin].toTensor();
    output_dispatch_keys = self.unsafeGetTensorImpl()->fake_dispatch_keys();
  }

  // Run the Meta kernel with the inputs reporting the meta device (see Note
  // [C++ in_kernel_invocation]), as Python's in_kernel_invocation_manager
  // does, and with the ops it runs, on meta tensors, kept out of fake dispatch
  c10::impl::ExcludeDispatchKeyGuard exclude_fake(c10::DispatchKey::Fake);
  c10::impl::IncludeDispatchKeyGuard include_meta(c10::DispatchKey::Meta);
  c10::impl::InKernelInvocationGuard in_kernel;
  {
    // Sparse invariant checks read index data, which meta tensors lack
    std::optional<at::sparse_csr::CheckSparseTensorInvariants> invariants_guard;
    if (!meta_keyset.has(c10::DispatchKey::Meta) &&
        at::globalContext().checkSparseTensorInvariants().value_or(false)) {
      invariants_guard.emplace(false);
    }
    op.redispatchBoxed(meta_keyset, stack);
  }

  // Returns the fake that replaces t on the stack, if t is not itself made one
  auto wrap_output = [&](const at::Tensor& t,
                         bool is_alias) -> std::optional<at::Tensor> {
    if (!t.defined()) {
      return std::nullopt;
    }
    auto* impl = t.unsafeGetTensorImpl();
    // An input returned as an output (by in-place and out= ops) is kept as
    // is. Its stored fake device is read, as device() reports meta here.
    if (is_alias &&
        std::find(input_impls.begin(), input_impls.end(), impl) !=
            input_impls.end()) {
      TORCH_INTERNAL_ASSERT(impl->fake_device() == device);
      return std::nullopt;
    }
    std::optional<at::Tensor> out;
    if (c10::raw::intrusive_ptr::use_count(impl) > 1) {
      // A tensor held elsewhere, such as the values _values returns from a
      // sparse tensor, is not changed: as Python wraps each output in a new
      // FakeTensor, a shallow copy becomes the fake
      out = at::Tensor(impl->shallow_copy_and_detach(
          impl->version_counter(), impl->allow_tensor_metadata_change()));
      impl = out->unsafeGetTensorImpl();
    } else if (!is_alias && t.requires_grad()) {
      // Backend kernels run below autograd, but a Meta kernel can copy
      // requires_grad from an input
      t.set_requires_grad(false);
    }
    // A copy a Meta kernel made of a fake input inherited the input's fake
    // state (see init_cpp_fake), where Python would see a plain meta tensor
    at::init_cpp_fake(impl, *device, mode);
    impl->set_real_tensor(nullptr);
    impl->set_fake_dispatch_keys(output_dispatch_keys);
    return out;
  };
  const auto num_returns = schema.returns().size();
  const auto returns_begin = stack->size() - num_returns;
  for (const auto i : c10::irange(num_returns)) {
    const bool is_alias = schema.returns()[i].alias_info() != nullptr;
    auto& ivalue = (*stack)[returns_begin + i];
    if (ivalue.isTensor()) {
      if (auto out = wrap_output(ivalue.toTensor(), is_alias)) {
        ivalue = std::move(*out);
      }
    } else if (ivalue.isList()) {
      auto list = ivalue.toList();
      // Read through toListRef, as reading the list copies its elements
      const auto elements = ivalue.toListRef();
      for (const auto j : c10::irange(elements.size())) {
        if (!elements[j].isTensor()) {
          continue;
        }
        if (auto out = wrap_output(elements[j].toTensor(), is_alias)) {
          list.set(j, std::move(*out));
        }
      }
    }
  }
}

TORCH_LIBRARY_IMPL(_, Fake, m) {
  m.fallback(torch::CppFunction::makeFromBoxedFunction<&fakeFallback>());
}

} // namespace
