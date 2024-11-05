// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <ATen/FunctionalTensorWrapper.h>
#include <ATen/WrapDimUtils.h>
#include <torch/csrc/utils/python_raii.h>
#include <torch/python.h>

#include <ATen/functorch/BatchRulesHelper.h>
#include <ATen/functorch/BatchedFallback.h>
#include <ATen/functorch/BatchedTensorImpl.h>
#include <ATen/functorch/DynamicLayer.h>
#include <ATen/functorch/Interpreter.h>
#include <ATen/functorch/LegacyVmapTransforms.h>
#include <ATen/functorch/PlumbingHelper.h>
#include <ATen/functorch/TensorWrapper.h>
#include <c10/core/AutogradState.h>

#include <iostream>

// This file contains functorch's Python bindings.

namespace torch::functorch::impl {

using namespace at::functorch;

static bool has_level(const Tensor& self, int64_t level) {
  const auto* batched = maybeGetBatchedImpl(self);
  if (!batched) {
    return false;
  }
  return batched->level() >= level;
}

Tensor _add_batch_dim(const Tensor& self, int64_t batch_dim, int64_t level) {
  return addBatchDim(self, batch_dim, level);
}

Tensor _wrap_functional_tensor(const Tensor& self, int64_t level) {
  auto t = at::functionalization::impl::to_functional_tensor(self);
  at::functionalization::impl::unsafeGetFunctionalWrapper(t)->set_level(level);
  return t;
}

void _assert_wrapped_functional(
    const Tensor& unwrapped,
    const Tensor& wrapped) {
  TORCH_INTERNAL_ASSERT(
      at::functionalization::impl::isFunctionalTensor(wrapped));
  TORCH_INTERNAL_ASSERT(
      !at::functionalization::impl::isFunctionalTensor(unwrapped));
  auto wrapped_impl =
      at::functionalization::impl::unsafeGetFunctionalWrapper(wrapped);
  auto& wrapped_inner = wrapped_impl->value();
  TORCH_INTERNAL_ASSERT(
      unwrapped.unsafeGetTensorImpl() == wrapped_inner.unsafeGetTensorImpl())
}

void _propagate_functional_input_mutation(
    const Tensor& unwrapped,
    const Tensor& wrapped) {
  TORCH_INTERNAL_ASSERT(
      at::functionalization::impl::isFunctionalTensor(wrapped));
  TORCH_INTERNAL_ASSERT(
      !at::functionalization::impl::isFunctionalTensor(unwrapped));
  auto wrapped_impl =
      at::functionalization::impl::unsafeGetFunctionalWrapper(wrapped);
  // Ensure that the input is up to date by committing any pending updates to
  // the alias.
  wrapped_impl->sync_();
  auto& wrapped_inner = wrapped_impl->value();
  // It would probably be more reasonable to check that the two tensors are
  // aliased, but we can't do that unless we give BatchedTensorImpl a notion of
  // storage.
  if (unwrapped.unsafeGetTensorImpl() == wrapped_inner.unsafeGetTensorImpl()) {
  } else {
    if (unwrapped.sym_nbytes() != wrapped_inner.sym_nbytes()) {
      // Functions might resize zero-sized inputs, which we need to reflect
      // ehre.
      unwrapped.resize__symint(wrapped_inner.sym_sizes());
    }
    // If the input tensor's metadata was mutated, then use as_strided_()
    // to propagate the metadata change.
    if (unwrapped.sym_sizes() != wrapped_inner.sym_sizes()) {
      unwrapped.as_strided__symint(
          wrapped_inner.sym_sizes(), wrapped_inner.sym_strides());
    }
    unwrapped.copy_(wrapped_inner);
  }
}

static std::pair<Tensor, int64_t> remove_existing_batch_dim(
    const BatchedTensorImpl* batched,
    int64_t level) {
  TORCH_INTERNAL_ASSERT(batched->level() == level);
  return std::make_pair(batched->value(), batched->bdim());
}

// Poor man's version of np.moveaxis. Moves the dimension at `dst` to `src`
// while preserving the order of other existing dimensions.
// We should probably add np.moveaxis (it is more general) to PyTorch. (#36048)
// When we do, replace the following with it.
static Tensor _movedim(const Tensor& self, int64_t src, int64_t dst) {
  auto logical_dim = self.dim();
  src = at::maybe_wrap_dim(src, logical_dim);
  dst = at::maybe_wrap_dim(dst, logical_dim);
  if (src == dst) {
    return self;
  }
  VmapDimVector permutation;
  permutation.reserve(logical_dim);
  for (int64_t dim = 0; dim < logical_dim; dim++) {
    if (dim == src) {
      continue;
    }
    permutation.push_back(dim);
  }
  permutation.insert(permutation.begin() + dst, src);
  return self.permute(permutation);
}

// Removes the batch dim with level `level` from `self`. If this causes the
// last batch dim to be removed from a BatchedTensor, then this returns a
// regular Tensor.
//
// If the `level` of the batch dim to remove does not exist in `self`, then we
// add the batch dim in. This can happen if `self` didn't interact with a tensor
// inside the vmap level, for example,
//     self = torch.randn(3)
//     y = torch.randn(5)
//     out = vmap(lambda x: vmap(lambda y: x)(y))(self)
//     assert out.shape == (3, 5)
// Inside the inner vmap, `x` is a BatchedTensor with a single batch dimension
// corresponding to the *outer* vmap level and it doesn't have any dimensions
// that correspond to the inner vmap level so we need to create one for the
// user.
//
// `out_dim` controls where we should put the batch dimension in the output
// tensor.
Tensor _remove_batch_dim(
    const Tensor& self,
    int64_t level,
    const c10::SymInt& batch_size,
    int64_t out_dim) {
  TORCH_CHECK(
      out_dim == 0 || !self.key_set().has(DispatchKey::BatchedNestedTensor),
      "Nested tensors can only be vmapped over dim=0, but got dim=",
      out_dim);
  if (!has_level(self, level)) {
    auto self_sizes = self.sym_sizes();
    VmapSymDimVector expanded_sizes(self_sizes.begin(), self_sizes.end());
    expanded_sizes.insert(expanded_sizes.begin() + out_dim, batch_size);
    auto result = self.expand_symint(expanded_sizes);
    return result;
  }

  // Must be batched if has_level(self, /*any_level*/)
  const auto* batched = maybeGetBatchedImpl(self);
  TORCH_INTERNAL_ASSERT(batched != nullptr);

  auto [self_without_bdim, newly_exposed_logical_dim] =
      remove_existing_batch_dim(batched, level);
  auto result = _movedim(self_without_bdim, newly_exposed_logical_dim, out_dim);
  return result;
}

Tensor _unwrap_functional_tensor(const Tensor& self, bool add_back_views) {
  // We only ever call that after popping out of a functionalize() call, in
  // which case the current tensors should always be wrapped in a
  // FunctionalTensorWrapper.
  TORCH_INTERNAL_ASSERT(at::functionalization::impl::isFunctionalTensor(self));
  auto functional =
      at::functionalization::impl::unsafeGetFunctionalWrapper(self);

  // when regenerating the (potentially mutated) input tensors, the
  // functionalization pass regenerates them through a series of view_copy() op
  // calls. Functorch wants to turn those back into view ops though. Ensure that
  // the input is up to date by committing any pending updates to the alias.
  at::functionalization::impl::FunctionalizationReapplyViewsGuard guard(
      add_back_views);
  bool any_updates = functional->apply_updates();
  if (any_updates) {
    functional->regenerate_from_base();
  }
  return functional->value();
}

Tensor _wrap_for_grad(const Tensor& self, int64_t level) {
  // NB: different behavior inside??
  // return self;
  // TORCH_INTERNAL_ASSERT(!maybeGetTensorWrapper(self));
  // TORCH_INTERNAL_ASSERT(self.has_storage());
  return makeTensorWrapper(self, level);
}

Tensor _unwrap_for_grad(const Tensor& self, int64_t level) {
  auto* result = maybeGetTensorWrapper(self);
  if (!result) {
    return self;
  }
  TORCH_INTERNAL_ASSERT(result->level().has_value());
  if (result->level() == level) {
    return result->value();
  }
  return self;
}

int64_t dlevel(const Tensor& tensor) {
  auto* wrapped = maybeGetTensorWrapper(tensor);
  if (!wrapped) {
    return 0;
  }
  if (!wrapped->is_alive()) {
    return -1;
  }
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
  return wrapped->level().value();
}

bool dump_tensor(const Tensor& self) {
  dumpTensorCout(self);
  return true;
}

RandomnessType get_randomness_enum(const std::string& randomness) {
  if (randomness == "error") {
    return RandomnessType::Error;
  } else if (randomness == "same") {
    return RandomnessType::Same;
  } else if (randomness == "different") {
    return RandomnessType::Different;
  } else {
    TORCH_CHECK(
        false, "randomness argument must be error, same, or different.");
  }
}

int64_t _grad_increment_nesting() {
  // See NOTE [grad and vjp interaction with no_grad]
  bool prev_grad_mode = c10::GradMode::is_enabled();
  return initAndPushDynamicLayer(
      TransformType::Grad, std::nullopt, std::nullopt, prev_grad_mode);
}

int64_t _grad_decrement_nesting() {
  auto layer = popDynamicLayerAndDeleteMetadata();
  TORCH_INTERNAL_ASSERT(layer.key() == TransformType::Grad);
  return layer.layerId();
}

int64_t _jvp_increment_nesting() {
  // See NOTE [grad and vjp interaction with no_grad]
  bool prev_fwd_grad_mode =
      c10::AutogradState::get_tls_state().get_fw_grad_mode();
  return initAndPushDynamicLayer(
      TransformType::Jvp,
      std::nullopt,
      std::nullopt,
      std::nullopt,
      prev_fwd_grad_mode);
}

int64_t _jvp_decrement_nesting() {
  auto layer = popDynamicLayerAndDeleteMetadata();
  TORCH_INTERNAL_ASSERT(layer.key() == TransformType::Jvp);
  return layer.layerId();
}

int64_t _vmap_increment_nesting(
    c10::SymInt batch_size,
    const std::string& randomness) {
  return initAndPushDynamicLayer(
      TransformType::Vmap,
      std::move(batch_size),
      get_randomness_enum(randomness));
}

int64_t _vmap_decrement_nesting() {
  auto layer = popDynamicLayerAndDeleteMetadata();
  TORCH_INTERNAL_ASSERT(layer.key() == TransformType::Vmap);
  return layer.layerId();
}

int64_t _func_increment_nesting(bool reapply_views) {
  return initAndPushDynamicLayer(
      TransformType::Functionalize,
      std::nullopt,
      std::nullopt,
      std::nullopt,
      std::nullopt,
      /*functionalize_add_back_views=*/reapply_views);
}

int64_t _func_decrement_nesting() {
  auto layer = popDynamicLayerAndDeleteMetadata();
  TORCH_INTERNAL_ASSERT(layer.key() == TransformType::Functionalize);
  return layer.layerId();
}

static bool is_batchedtensor(const Tensor& tensor) {
  auto* batched = maybeGetBatchedImpl(tensor);
  return batched != nullptr;
}

static bool is_legacy_batchedtensor(const Tensor& tensor) {
  return tensor.unsafeGetTensorImpl()->key_set().has(DispatchKey::Batched);
}

static bool is_gradtrackingtensor(const Tensor& tensor) {
  auto* wrapped = maybeGetTensorWrapper(tensor);
  return wrapped != nullptr;
}

static bool is_functionaltensor(const Tensor& tensor) {
  return tensor.unsafeGetTensorImpl()->key_set().has(
      c10::DispatchKey::Functionalize);
}

static Tensor get_unwrapped(const Tensor& tensor) {
  auto* batched = maybeGetBatchedImpl(tensor);
  if (batched) {
    return batched->value();
  }
  auto* wrapped = maybeGetTensorWrapper(tensor);
  if (wrapped) {
    return wrapped->value();
  }
  if (at::functionalization::impl::isFunctionalTensor(tensor)) {
    auto* functional =
        at::functionalization::impl::unsafeGetFunctionalWrapper(tensor);
    return functional->value();
  }
  TORCH_CHECK(false, "No wrappers present!");
}

static int64_t maybe_get_level(const Tensor& tensor) {
  auto* batched = maybeGetBatchedImpl(tensor);
  if (batched) {
    return batched->level();
  }
  auto* wrapped = maybeGetTensorWrapper(tensor);
  if (wrapped) {
    if (wrapped->level()) {
      // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
      return *wrapped->level();
    }
    // TODO: this is a weird special case...
    return -2;
  }
  if (at::functionalization::impl::isFunctionalTensor(tensor)) {
    auto* functional =
        at::functionalization::impl::unsafeGetFunctionalWrapper(tensor);
    return functional->level();
  }
  return -1;
}

static int64_t maybe_get_bdim(const Tensor& tensor) {
  auto* batched = maybeGetBatchedImpl(tensor);
  if (batched) {
    return batched->bdim();
  }
  return -1;
}

static int64_t currentLevel() {
  auto maybe_layer = maybeCurrentDynamicLayer();
  TORCH_INTERNAL_ASSERT(maybe_layer.has_value());
  int64_t current_level = maybe_layer->layerId();
  return current_level;
}

static std::optional<int64_t> maybe_current_level() {
  auto maybe_layer = maybeCurrentDynamicLayer();
  if (maybe_layer.has_value()) {
    int64_t current_level = maybe_layer->layerId();
    return current_level;
  }
  return std::nullopt;
}

static void tls_set_vmap_excluded(bool excluded) {
  c10::impl::tls_set_dispatch_key_excluded(
      c10::DispatchKey::FuncTorchBatched, excluded);
}

static void _set_dynamic_layer_keys_included(bool value) {
  return setDynamicLayerFrontBackKeysIncluded(value);
}

static void dump_dls() {
  std::cout << getDynamicLayerStack() << '\n';
}

static void dump_local_tls() {
  auto tls = c10::impl::tls_local_dispatch_key_set();
  std::cout << "[Local Include] " << tls.included_ << '\n';
  std::cout << "[Local Exclude] " << tls.excluded_ << '\n';
}

namespace {

// Pop the DynamicLayer stack until it's at the given depth.
void popDynamicLayerStackToDepth(size_t depth) {
  while (at::functorch::getDynamicLayerStack().size() > depth) {
    const auto top = popDynamicLayer();
    switch (top.key()) {
      case at::functorch::TransformType::Vmap:
        _vmap_decrement_nesting();
        break;
      case at::functorch::TransformType::Grad:
        _grad_decrement_nesting();
        break;
      case at::functorch::TransformType::Jvp:
        _jvp_decrement_nesting();
        break;
      case at::functorch::TransformType::Functionalize:
        _func_decrement_nesting();
        break;
      case at::functorch::TransformType::Torch:
        popDynamicLayerAndDeleteMetadata();
        break;
    }
  }
}

} // anonymous namespace

static std::tuple<Tensor, std::optional<int64_t>> unwrapBatched(
    const Tensor& tensor,
    int64_t level) {
  auto* batched = maybeGetBatchedImpl(tensor);
  if (!batched) {
    return std::make_tuple(tensor, std::nullopt);
  }
  if (batched->level() == level) {
    return std::make_tuple(batched->value(), batched->bdim());
  }
  return std::make_tuple(tensor, std::nullopt);
}

void initFuncTorchBindings(PyObject* module) {
  auto _C = py::handle(module).cast<py::module>();
  auto m = _C.def_submodule("_functorch");

  m.def("_add_batch_dim", &_add_batch_dim, "add batch dim");
  m.def("_remove_batch_dim", &_remove_batch_dim, "remove batch dim");
  m.def("_unwrap_batched", &unwrapBatched);
  m.def(
      "_wrap_functional_tensor",
      &_wrap_functional_tensor,
      "add functional tensor");
  m.def(
      "_assert_wrapped_functional",
      &_assert_wrapped_functional,
      "assert wrapped functional");
  m.def(
      "_propagate_functional_input_mutation",
      &_propagate_functional_input_mutation,
      "propagate functional input mutations");
  m.def(
      "_unwrap_functional_tensor",
      &_unwrap_functional_tensor,
      "remove functional tensor");
  m.def("_vmap_increment_nesting", &_vmap_increment_nesting);
  m.def("_vmap_decrement_nesting", &_vmap_decrement_nesting);
  m.def(
      "_func_increment_nesting",
      &_func_increment_nesting,
      "functionalization start");
  m.def(
      "_func_decrement_nesting",
      &_func_decrement_nesting,
      "functionalization end");
  m.def("_grad_increment_nesting", &_grad_increment_nesting);
  m.def("_grad_decrement_nesting", &_grad_decrement_nesting);
  m.def("_jvp_increment_nesting", &_jvp_increment_nesting);
  m.def("_jvp_decrement_nesting", &_jvp_decrement_nesting);
  m.def("_wrap_for_grad", &_wrap_for_grad, "wrap as gradtrackingtensor");
  m.def(
      "_unwrap_for_grad", &_unwrap_for_grad, "unwrap from gradtrackingtensor");
  m.def(
      "_set_vmap_fallback_warning_enabled",
      &at::functorch::setVmapFallbackWarningEnabled,
      "Set vmap fallback warnings");
  m.def("_set_vmap_fallback_enabled", &at::functorch::setVmapFallbackEnabled);
  m.def("_is_vmap_fallback_enabled", &at::functorch::isVmapFallbackEnabled);
  m.def(
      "set_inplace_requires_grad_allowed",
      &at::functorch::setInplaceRequiresGradAllowed);
  m.def(
      "get_inplace_requires_grad_allowed",
      &at::functorch::getInplaceRequiresGradAllowed);
  m.def(
      "set_single_level_autograd_function_allowed",
      &at::functorch::setSingleLevelAutogradFunctionAllowed);
  m.def(
      "get_single_level_autograd_function_allowed",
      &at::functorch::getSingleLevelAutogradFunctionAllowed);
  m.def("unwrap_if_dead", &unwrapIfDead);
  m.def("is_dead_tensor_wrapper", &isDeadTensorWrapper);
  m.def("dlevel", &dlevel, "dlevel");
  m.def("dump_tensor", &dump_tensor, "dump_tensor");
  m.def("reshape_dim_into", &at::functorch::reshape_dim_into);
  m.def("reshape_dim_outof", &at::functorch::reshape_dim_outof);
  // various debugging things. Maybe we should offer these as first-class APIs
  // on Tensors?
  m.def("is_batchedtensor", &is_batchedtensor);
  m.def("is_legacy_batchedtensor", &is_legacy_batchedtensor);
  m.def("is_gradtrackingtensor", &is_gradtrackingtensor);
  m.def("is_functionaltensor", &is_functionaltensor);
  m.def("get_unwrapped", &get_unwrapped);
  m.def("maybe_get_level", &maybe_get_level);
  m.def("maybe_get_bdim", &maybe_get_bdim);
  m.def("maybe_current_level", &maybe_current_level);
  m.def("current_level", &currentLevel);
  m.def("tls_set_vmap_excluded", &tls_set_vmap_excluded);
  m.def("_set_dynamic_layer_keys_included", &_set_dynamic_layer_keys_included);
  m.def("dump_dls", &dump_dls);
  m.def("dump_local_tls", &dump_local_tls);
  m.def("is_functorch_wrapped_tensor", [](const Tensor& tensor) {
    return maybe_get_level(tensor) != -1;
  });
  m.def(
      "get_interpreter_stack", []() -> std::optional<std::vector<Interpreter>> {
        const auto& stack = getDynamicLayerStack();
        if (stack.empty()) {
          return std::nullopt;
        }
        std::vector<Interpreter> result;
        result.reserve(stack.size());
        for (auto i : stack) {
          result.push_back(i.interpreter());
        }
        return result;
      });
  m.def("peek_interpreter_stack", []() -> std::optional<Interpreter> {
    const auto& stack = getDynamicLayerStack();
    if (stack.empty()) {
      return std::nullopt;
    }
    auto result = stack.back().interpreter();
    return result;
  });
  m.def("get_dynamic_layer_stack_depth", []() -> size_t {
    return getDynamicLayerStack().size();
  });
  m.def(
      "pop_dynamic_layer_stack_and_undo_to_depth",
      &popDynamicLayerStackToDepth);
  m.def("pop_dynamic_layer_stack", &popDynamicLayer);
  m.def("push_dynamic_layer_stack", [](DynamicLayer layer) -> int64_t {
    return pushDynamicLayer(std::move(layer));
  });
  // NOLINTNEXTLINE(bugprone-unused-raii)
  py::class_<DynamicLayer>(m, "DynamicLayer");

  py::enum_<TransformType>(m, "TransformType")
      .value("Torch", TransformType::Torch)
      .value("Grad", TransformType::Grad)
      .value("Jvp", TransformType::Jvp)
      .value("Functionalize", TransformType::Functionalize)
      .value("Vmap", TransformType::Vmap);
  py::enum_<RandomnessType>(m, "RandomnessType")
      .value("Error", RandomnessType::Error)
      .value("Same", RandomnessType::Same)
      .value("Different", RandomnessType::Different);
  py::class_<Interpreter>(m, "CInterpreter")
      .def("key", &Interpreter::key)
      .def("level", &Interpreter::level);
  py::class_<GradInterpreterPtr>(m, "CGradInterpreterPtr")
      .def(py::init<const Interpreter*>())
      .def("key", &GradInterpreterPtr::key)
      .def("level", &GradInterpreterPtr::level)
      .def("lift", &GradInterpreterPtr::lift)
      .def("prevGradMode", &GradInterpreterPtr::prevGradMode);
  py::class_<JvpInterpreterPtr>(m, "CJvpInterpreterPtr")
      .def(py::init<const Interpreter*>())
      .def("key", &JvpInterpreterPtr::key)
      .def("level", &JvpInterpreterPtr::level)
      .def("lift", &JvpInterpreterPtr::lift)
      .def("prevFwdGradMode", &JvpInterpreterPtr::prevFwdGradMode);
  py::class_<VmapInterpreterPtr>(m, "CVmapInterpreterPtr")
      .def(py::init<const Interpreter*>())
      .def("key", &VmapInterpreterPtr::key)
      .def("level", &VmapInterpreterPtr::level)
      .def("batchSize", &VmapInterpreterPtr::batchSize)
      .def("randomness", &VmapInterpreterPtr::randomness);
  py::class_<FunctionalizeInterpreterPtr>(m, "CFunctionalizeInterpreterPtr")
      .def(py::init<const Interpreter*>())
      .def("key", &FunctionalizeInterpreterPtr::key)
      .def("level", &FunctionalizeInterpreterPtr::level)
      .def(
          "functionalizeAddBackViews",
          &FunctionalizeInterpreterPtr::functionalizeAddBackViews);
}

} // namespace torch::functorch::impl
