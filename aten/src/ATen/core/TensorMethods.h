#pragma once

#include <c10/core/Scalar.h>
#include <c10/core/MemoryFormat.h>
#include <c10/core/QScheme.h>
#include <c10/macros/Macros.h>
#include <c10/core/TensorOptions.h>
#include <c10/util/intrusive_ptr.h>
#include <ATen/core/DeprecatedTypeProperties.h>
#include <ATen/core/ATenDispatch.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/core/Variadic.h>
#include <ATen/core/NamedTensor.h>
#include <ATen/core/EnableNamedTensor.h>

#ifdef USE_STATIC_DISPATCH
#include <ATen/TypeDefault.h>
#include <ATen/CPUType.h>
#include <ATen/QuantizedCPUType.h>
#include <ATen/SparseCPUType.h>
#endif

namespace at {

namespace detail {

struct MultiDispatchTensorTypeSet : IterArgs<MultiDispatchTensorTypeSet> {
  TensorTypeSet ts;
  void operator()(const at::Tensor& x) {
    ts = ts | x.type_set();
  }
  void operator()(TensorOptions x) {
    ts = ts | x.type_set();
  }
  void operator()(at::ArrayRef<at::Tensor> xs) {
    for (const auto& x : xs) {
      ts = ts | x.type_set();
    }
  }
};

template <typename... Args>
TensorTypeSet multi_dispatch_tensor_type_set(Args&&... args) {
  return MultiDispatchTensorTypeSet().apply(std::forward<Args>(args)...).ts;
}

}

struct Quantizer;
// This is temporary typedef to enable Quantizer in aten native function API
// we'll remove them when we are actually exposing Quantizer class
// to frontend
using ConstQuantizerPtr = const c10::intrusive_ptr<Quantizer>&;

inline Tensor Tensor::cpu() const {
  return to(options().device(DeviceType::CPU), /*non_blocking*/ false, /*copy*/ false);
}

// TODO: The Python version also accepts arguments
inline Tensor Tensor::cuda() const {
  return to(options().device(DeviceType::CUDA), /*non_blocking*/ false, /*copy*/ false);
}

inline Tensor Tensor::hip() const {
  return to(options().device(DeviceType::HIP), /*non_blocking*/ false, /*copy*/ false);
}

inline Tensor Tensor::toType(ScalarType t) const {
  return to(options().dtype(t), /*non_blocking*/ false, /*copy*/ false);
}

// TODO: Deprecate me
inline Tensor Tensor::toBackend(Backend b) const {
  return to(options().device(backendToDeviceType(b)).layout(layout_from_backend(b)), /*non_blocking*/ false, /*copy*/ false);
}

inline TensorOptions Tensor::options() const {
  return TensorOptions().dtype(dtype())
                        .device(device())
                        .layout(layout())
                        .is_variable(is_variable());
}

// all static inline to allow for inlining of the non-dynamic part of dispatch
inline void Tensor::backward(const Tensor & gradient, bool keep_graph, bool create_graph) const {
#ifdef USE_STATIC_DISPATCH
     TypeDefault::backward(const_cast<Tensor&>(*this), gradient, keep_graph, create_graph);
#else
    static auto table = globalATenDispatch().getOpTable("aten::backward(Tensor self, Tensor? gradient=None, bool keep_graph=False, bool create_graph=False) -> void");
    return table->getOp<void (const Tensor &, const Tensor &, bool, bool)>(at::detail::multi_dispatch_tensor_type_set(*this, gradient))(const_cast<Tensor&>(*this), gradient, keep_graph, create_graph);
#endif
}
inline void Tensor::set_data(const Tensor & new_data) const {
#ifdef USE_STATIC_DISPATCH
     TypeDefault::set_data(const_cast<Tensor&>(*this), new_data);
#else
    static auto table = globalATenDispatch().getOpTable("aten::set_data(Tensor(a!) self, Tensor new_data) -> void");
    return table->getOp<void (const Tensor &, const Tensor &)>(at::detail::multi_dispatch_tensor_type_set(*this, new_data))(const_cast<Tensor&>(*this), new_data);
#endif
}
inline Tensor Tensor::data() const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::data(const_cast<Tensor&>(*this));
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::data", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}
#ifdef BUILD_NAMEDTENSOR
inline Tensor & Tensor::names_(c10::optional<DimnameList> names) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::names_(const_cast<Tensor&>(*this), names);
#else
    static auto table = globalATenDispatch().getOpTable("aten::names_(Tensor(a!) self, Dimname[]? names) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, c10::optional<DimnameList>)>(at::detail::multi_dispatch_tensor_type_set(*this))(const_cast<Tensor&>(*this), names);
#endif
}
#endif
#ifdef BUILD_NAMEDTENSOR
inline Tensor Tensor::renamed(c10::optional<DimnameList> names) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::renamed(const_cast<Tensor&>(*this), names);
#else
    static auto table = globalATenDispatch().getOpTable("aten::renamed(Tensor(a) self, Dimname[]? names) -> Tensor(a)");
    return table->getOp<Tensor (const Tensor &, c10::optional<DimnameList>)>(at::detail::multi_dispatch_tensor_type_set(*this))(const_cast<Tensor&>(*this), names);
#endif
}
#endif
#ifdef BUILD_NAMEDTENSOR
inline Tensor Tensor::align_to(DimnameList names) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::align_to(const_cast<Tensor&>(*this), names);
#else
    static auto table = globalATenDispatch().getOpTable("aten::align_to(Tensor(a) self, DimnameList names) -> Tensor(a)");
    return table->getOp<Tensor (const Tensor &, DimnameList)>(at::detail::multi_dispatch_tensor_type_set(*this))(const_cast<Tensor&>(*this), names);
#endif
}
#endif
#ifdef BUILD_NAMEDTENSOR
inline Tensor Tensor::align_as(const Tensor & other) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::align_as(const_cast<Tensor&>(*this), other);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::align_as", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this, other)))
        .callUnboxed<Tensor, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), other);
#endif
}
#endif
#ifdef BUILD_NAMEDTENSOR
inline Tensor Tensor::refine_names(DimnameList names) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::refine_names(const_cast<Tensor&>(*this), names);
#else
    static auto table = globalATenDispatch().getOpTable("aten::refine_names(Tensor(a) self, DimnameList names) -> Tensor(a)");
    return table->getOp<Tensor (const Tensor &, DimnameList)>(at::detail::multi_dispatch_tensor_type_set(*this))(const_cast<Tensor&>(*this), names);
#endif
}
#endif
#ifdef BUILD_NAMEDTENSOR
inline Tensor Tensor::unflatten(Dimname dim, IntArrayRef sizes, DimnameList names) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::unflatten(const_cast<Tensor&>(*this), dim, sizes, names);
#else
    static auto table = globalATenDispatch().getOpTable("aten::unflatten(Tensor self, Dimname dim, int[] sizes, DimnameList names) -> Tensor");
    return table->getOp<Tensor (const Tensor &, Dimname, IntArrayRef, DimnameList)>(at::detail::multi_dispatch_tensor_type_set(*this))(const_cast<Tensor&>(*this), dim, sizes, names);
#endif
}
#endif
#ifdef BUILD_NAMEDTENSOR
inline Tensor Tensor::unflatten(int64_t dim, IntArrayRef sizes, DimnameList names) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::unflatten(const_cast<Tensor&>(*this), dim, sizes, names);
#else
    static auto table = globalATenDispatch().getOpTable("aten::unflatten(Tensor self, int dim, int[] sizes, DimnameList names) -> Tensor");
    return table->getOp<Tensor (const Tensor &, int64_t, IntArrayRef, DimnameList)>(at::detail::multi_dispatch_tensor_type_set(*this))(const_cast<Tensor&>(*this), dim, sizes, names);
#endif
}
#endif
inline Tensor Tensor::abs() const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::abs(const_cast<Tensor&>(*this));
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::abs", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}
inline Tensor & Tensor::abs_() const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::abs_(const_cast<Tensor&>(*this));
            break;
        default:
            AT_ERROR("abs_ not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::abs_", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor &, Tensor &>(const_cast<Tensor&>(*this));
#endif
}
inline Tensor Tensor::acos() const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::acos(const_cast<Tensor&>(*this));
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::acos", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}
inline Tensor & Tensor::acos_() const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::acos_(const_cast<Tensor&>(*this));
            break;
        default:
            AT_ERROR("acos_ not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::acos_", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor &, Tensor &>(const_cast<Tensor&>(*this));
#endif
}
inline Tensor Tensor::add(const Tensor & other, Scalar alpha) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::add(const_cast<Tensor&>(*this), other, alpha);
            break;
        case Backend::SparseCPU:
            return SparseCPUType::add(const_cast<Tensor&>(*this), other, alpha);
            break;
        default:
            AT_ERROR("add not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::add", "Tensor"}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this, other)))
        .callUnboxed<Tensor, const Tensor &, const Tensor &, Scalar>(const_cast<Tensor&>(*this), other, alpha);
#endif
}
inline Tensor & Tensor::add_(const Tensor & other, Scalar alpha) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::add_(const_cast<Tensor&>(*this), other, alpha);
            break;
        case Backend::SparseCPU:
            return SparseCPUType::add_(const_cast<Tensor&>(*this), other, alpha);
            break;
        default:
            AT_ERROR("add_ not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::add_", "Tensor"}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this, other)))
        .callUnboxed<Tensor &, Tensor &, const Tensor &, Scalar>(const_cast<Tensor&>(*this), other, alpha);
#endif
}
inline Tensor Tensor::add(Scalar other, Scalar alpha) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::add(const_cast<Tensor&>(*this), other, alpha);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::add", "Scalar"}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor, const Tensor &, Scalar, Scalar>(const_cast<Tensor&>(*this), other, alpha);
#endif
}
inline Tensor & Tensor::add_(Scalar other, Scalar alpha) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::add_(const_cast<Tensor&>(*this), other, alpha);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::add_", "Scalar"}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor &, Tensor &, Scalar, Scalar>(const_cast<Tensor&>(*this), other, alpha);
#endif
}
inline Tensor Tensor::addmv(const Tensor & mat, const Tensor & vec, Scalar beta, Scalar alpha) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::addmv(const_cast<Tensor&>(*this), mat, vec, beta, alpha);
            break;
        default:
            AT_ERROR("addmv not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::addmv", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this, mat, vec)))
        .callUnboxed<Tensor, const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar>(const_cast<Tensor&>(*this), mat, vec, beta, alpha);
#endif
}
inline Tensor & Tensor::addmv_(const Tensor & mat, const Tensor & vec, Scalar beta, Scalar alpha) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::addmv_(const_cast<Tensor&>(*this), mat, vec, beta, alpha);
            break;
        default:
            AT_ERROR("addmv_ not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::addmv_", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this, mat, vec)))
        .callUnboxed<Tensor &, Tensor &, const Tensor &, const Tensor &, Scalar, Scalar>(const_cast<Tensor&>(*this), mat, vec, beta, alpha);
#endif
}
inline Tensor Tensor::addr(const Tensor & vec1, const Tensor & vec2, Scalar beta, Scalar alpha) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::addr(const_cast<Tensor&>(*this), vec1, vec2, beta, alpha);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::addr", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this, vec1, vec2)))
        .callUnboxed<Tensor, const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar>(const_cast<Tensor&>(*this), vec1, vec2, beta, alpha);
#endif
}
inline Tensor & Tensor::addr_(const Tensor & vec1, const Tensor & vec2, Scalar beta, Scalar alpha) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::addr_(const_cast<Tensor&>(*this), vec1, vec2, beta, alpha);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::addr_", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this, vec1, vec2)))
        .callUnboxed<Tensor &, Tensor &, const Tensor &, const Tensor &, Scalar, Scalar>(const_cast<Tensor&>(*this), vec1, vec2, beta, alpha);
#endif
}
inline Tensor Tensor::all(int64_t dim, bool keepdim) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::all(const_cast<Tensor&>(*this), dim, keepdim);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::all", "dim"}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor, const Tensor &, int64_t, bool>(const_cast<Tensor&>(*this), dim, keepdim);
#endif
}
inline bool Tensor::allclose(const Tensor & other, double rtol, double atol, bool equal_nan) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::allclose(const_cast<Tensor&>(*this), other, rtol, atol, equal_nan);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::allclose", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this, other)))
        .callUnboxed<bool, const Tensor &, const Tensor &, double, double, bool>(const_cast<Tensor&>(*this), other, rtol, atol, equal_nan);
#endif
}
inline Tensor Tensor::any(int64_t dim, bool keepdim) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::any(const_cast<Tensor&>(*this), dim, keepdim);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::any", "dim"}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor, const Tensor &, int64_t, bool>(const_cast<Tensor&>(*this), dim, keepdim);
#endif
}
inline Tensor Tensor::argmax(c10::optional<int64_t> dim, bool keepdim) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::argmax(const_cast<Tensor&>(*this), dim, keepdim);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::argmax", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor, const Tensor &, c10::optional<int64_t>, bool>(const_cast<Tensor&>(*this), dim, keepdim);
#endif
}
inline Tensor Tensor::argmin(c10::optional<int64_t> dim, bool keepdim) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::argmin(const_cast<Tensor&>(*this), dim, keepdim);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::argmin", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor, const Tensor &, c10::optional<int64_t>, bool>(const_cast<Tensor&>(*this), dim, keepdim);
#endif
}
inline Tensor Tensor::as_strided(IntArrayRef size, IntArrayRef stride, c10::optional<int64_t> storage_offset) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::as_strided(const_cast<Tensor&>(*this), size, stride, storage_offset);
            break;
        case Backend::QuantizedCPU:
            return QuantizedCPUType::as_strided(const_cast<Tensor&>(*this), size, stride, storage_offset);
            break;
        default:
            AT_ERROR("as_strided not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::as_strided", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor, const Tensor &, IntArrayRef, IntArrayRef, c10::optional<int64_t>>(const_cast<Tensor&>(*this), size, stride, storage_offset);
#endif
}
inline Tensor & Tensor::as_strided_(IntArrayRef size, IntArrayRef stride, c10::optional<int64_t> storage_offset) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::as_strided_(const_cast<Tensor&>(*this), size, stride, storage_offset);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::as_strided_", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor &, Tensor &, IntArrayRef, IntArrayRef, c10::optional<int64_t>>(const_cast<Tensor&>(*this), size, stride, storage_offset);
#endif
}
inline Tensor Tensor::asin() const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::asin(const_cast<Tensor&>(*this));
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::asin", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}
inline Tensor & Tensor::asin_() const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::asin_(const_cast<Tensor&>(*this));
            break;
        default:
            AT_ERROR("asin_ not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::asin_", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor &, Tensor &>(const_cast<Tensor&>(*this));
#endif
}
inline Tensor Tensor::atan() const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::atan(const_cast<Tensor&>(*this));
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::atan", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}
inline Tensor & Tensor::atan_() const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::atan_(const_cast<Tensor&>(*this));
            break;
        default:
            AT_ERROR("atan_ not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::atan_", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor &, Tensor &>(const_cast<Tensor&>(*this));
#endif
}
inline Tensor Tensor::baddbmm(const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::baddbmm(const_cast<Tensor&>(*this), batch1, batch2, beta, alpha);
            break;
        default:
            AT_ERROR("baddbmm not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::baddbmm", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this, batch1, batch2)))
        .callUnboxed<Tensor, const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar>(const_cast<Tensor&>(*this), batch1, batch2, beta, alpha);
#endif
}
inline Tensor & Tensor::baddbmm_(const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::baddbmm_(const_cast<Tensor&>(*this), batch1, batch2, beta, alpha);
            break;
        default:
            AT_ERROR("baddbmm_ not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::baddbmm_", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this, batch1, batch2)))
        .callUnboxed<Tensor &, Tensor &, const Tensor &, const Tensor &, Scalar, Scalar>(const_cast<Tensor&>(*this), batch1, batch2, beta, alpha);
#endif
}
inline Tensor Tensor::bernoulli(Generator * generator) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::bernoulli(const_cast<Tensor&>(*this), generator);
#else
    static auto table = globalATenDispatch().getOpTable("aten::bernoulli(Tensor self, *, Generator? generator=None) -> Tensor");
    return table->getOp<Tensor (const Tensor &, Generator *)>(at::detail::multi_dispatch_tensor_type_set(*this))(const_cast<Tensor&>(*this), generator);
#endif
}
inline Tensor & Tensor::bernoulli_(const Tensor & p, Generator * generator) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::bernoulli_(const_cast<Tensor&>(*this), p, generator);
            break;
        default:
            AT_ERROR("bernoulli_ not implemented for ", at::toString(type_set()));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::bernoulli_.Tensor(Tensor(a!) self, Tensor p, *, Generator? generator=None) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, const Tensor &, Generator *)>(at::detail::multi_dispatch_tensor_type_set(*this, p))(const_cast<Tensor&>(*this), p, generator);
#endif
}
inline Tensor & Tensor::bernoulli_(double p, Generator * generator) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::bernoulli_(const_cast<Tensor&>(*this), p, generator);
            break;
        default:
            AT_ERROR("bernoulli_ not implemented for ", at::toString(type_set()));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::bernoulli_.float(Tensor(a!) self, float p=0.5, *, Generator? generator=None) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, double, Generator *)>(at::detail::multi_dispatch_tensor_type_set(*this))(const_cast<Tensor&>(*this), p, generator);
#endif
}
inline Tensor Tensor::bernoulli(double p, Generator * generator) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::bernoulli(const_cast<Tensor&>(*this), p, generator);
#else
    static auto table = globalATenDispatch().getOpTable("aten::bernoulli.p(Tensor self, float p, *, Generator? generator=None) -> Tensor");
    return table->getOp<Tensor (const Tensor &, double, Generator *)>(at::detail::multi_dispatch_tensor_type_set(*this))(const_cast<Tensor&>(*this), p, generator);
#endif
}
inline Tensor Tensor::bincount(const Tensor & weights, int64_t minlength) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::bincount(const_cast<Tensor&>(*this), weights, minlength);
            break;
        default:
            AT_ERROR("bincount not implemented for ", at::toString(type_set()));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::bincount(Tensor self, Tensor? weights=None, int minlength=0) -> Tensor");
    return table->getOp<Tensor (const Tensor &, const Tensor &, int64_t)>(at::detail::multi_dispatch_tensor_type_set(*this, weights))(const_cast<Tensor&>(*this), weights, minlength);
#endif
}
inline Tensor Tensor::bitwise_not() const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::bitwise_not(const_cast<Tensor&>(*this));
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::bitwise_not", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}
inline Tensor & Tensor::bitwise_not_() const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::bitwise_not_(const_cast<Tensor&>(*this));
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::bitwise_not_", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor &, Tensor &>(const_cast<Tensor&>(*this));
#endif
}
inline Tensor Tensor::logical_not() const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::logical_not(const_cast<Tensor&>(*this));
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::logical_not", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}
inline Tensor & Tensor::logical_not_() const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::logical_not_(const_cast<Tensor&>(*this));
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::logical_not_", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor &, Tensor &>(const_cast<Tensor&>(*this));
#endif
}
inline Tensor Tensor::logical_xor(const Tensor & other) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::logical_xor(const_cast<Tensor&>(*this), other);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::logical_xor", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this, other)))
        .callUnboxed<Tensor, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), other);
#endif
}
inline Tensor & Tensor::logical_xor_(const Tensor & other) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::logical_xor_(const_cast<Tensor&>(*this), other);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::logical_xor_", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this, other)))
        .callUnboxed<Tensor &, Tensor &, const Tensor &>(const_cast<Tensor&>(*this), other);
#endif
}
inline Tensor Tensor::bmm(const Tensor & mat2) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::bmm(const_cast<Tensor&>(*this), mat2);
            break;
        default:
            AT_ERROR("bmm not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::bmm", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this, mat2)))
        .callUnboxed<Tensor, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), mat2);
#endif
}
inline Tensor Tensor::ceil() const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::ceil(const_cast<Tensor&>(*this));
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::ceil", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}
inline Tensor & Tensor::ceil_() const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::ceil_(const_cast<Tensor&>(*this));
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::ceil_", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor &, Tensor &>(const_cast<Tensor&>(*this));
#endif
}
inline std::vector<Tensor> Tensor::chunk(int64_t chunks, int64_t dim) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::chunk(const_cast<Tensor&>(*this), chunks, dim);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::chunk", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<std::vector<Tensor>, const Tensor &, int64_t, int64_t>(const_cast<Tensor&>(*this), chunks, dim);
#endif
}
inline Tensor Tensor::clamp(c10::optional<Scalar> min, c10::optional<Scalar> max) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::clamp(const_cast<Tensor&>(*this), min, max);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::clamp", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor, const Tensor &, c10::optional<Scalar>, c10::optional<Scalar>>(const_cast<Tensor&>(*this), min, max);
#endif
}
inline Tensor & Tensor::clamp_(c10::optional<Scalar> min, c10::optional<Scalar> max) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::clamp_(const_cast<Tensor&>(*this), min, max);
            break;
        default:
            AT_ERROR("clamp_ not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::clamp_", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor &, Tensor &, c10::optional<Scalar>, c10::optional<Scalar>>(const_cast<Tensor&>(*this), min, max);
#endif
}
inline Tensor Tensor::clamp_max(Scalar max) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::clamp_max(const_cast<Tensor&>(*this), max);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::clamp_max", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor, const Tensor &, Scalar>(const_cast<Tensor&>(*this), max);
#endif
}
inline Tensor & Tensor::clamp_max_(Scalar max) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::clamp_max_(const_cast<Tensor&>(*this), max);
            break;
        default:
            AT_ERROR("clamp_max_ not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::clamp_max_", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor &, Tensor &, Scalar>(const_cast<Tensor&>(*this), max);
#endif
}
inline Tensor Tensor::clamp_min(Scalar min) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::clamp_min(const_cast<Tensor&>(*this), min);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::clamp_min", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor, const Tensor &, Scalar>(const_cast<Tensor&>(*this), min);
#endif
}
inline Tensor & Tensor::clamp_min_(Scalar min) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::clamp_min_(const_cast<Tensor&>(*this), min);
            break;
        default:
            AT_ERROR("clamp_min_ not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::clamp_min_", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor &, Tensor &, Scalar>(const_cast<Tensor&>(*this), min);
#endif
}
inline Tensor Tensor::contiguous(MemoryFormat memory_format) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::contiguous(const_cast<Tensor&>(*this), memory_format);
#else
    static auto table = globalATenDispatch().getOpTable("aten::contiguous(Tensor self, *, MemoryFormat memory_format=contiguous_format) -> Tensor");
    return table->getOp<Tensor (const Tensor &, MemoryFormat)>(at::detail::multi_dispatch_tensor_type_set(*this))(const_cast<Tensor&>(*this), memory_format);
#endif
}
inline Tensor & Tensor::copy_(const Tensor & src, bool non_blocking) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::copy_(const_cast<Tensor&>(*this), src, non_blocking);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::copy_", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this, src)))
        .callUnboxed<Tensor &, Tensor &, const Tensor &, bool>(const_cast<Tensor&>(*this), src, non_blocking);
#endif
}
inline Tensor Tensor::cos() const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::cos(const_cast<Tensor&>(*this));
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::cos", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}
inline Tensor & Tensor::cos_() const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::cos_(const_cast<Tensor&>(*this));
            break;
        default:
            AT_ERROR("cos_ not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::cos_", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor &, Tensor &>(const_cast<Tensor&>(*this));
#endif
}
inline Tensor Tensor::cosh() const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::cosh(const_cast<Tensor&>(*this));
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::cosh", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}
inline Tensor & Tensor::cosh_() const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::cosh_(const_cast<Tensor&>(*this));
            break;
        default:
            AT_ERROR("cosh_ not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::cosh_", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor &, Tensor &>(const_cast<Tensor&>(*this));
#endif
}
inline Tensor Tensor::cumsum(int64_t dim, c10::optional<ScalarType> dtype) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::cumsum(const_cast<Tensor&>(*this), dim, dtype);
#else
    static auto table = globalATenDispatch().getOpTable("aten::cumsum(Tensor self, int dim, *, ScalarType? dtype=None) -> Tensor");
    return table->getOp<Tensor (const Tensor &, int64_t, c10::optional<ScalarType>)>(at::detail::multi_dispatch_tensor_type_set(*this))(const_cast<Tensor&>(*this), dim, dtype);
#endif
}
inline Tensor Tensor::cumprod(int64_t dim, c10::optional<ScalarType> dtype) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::cumprod(const_cast<Tensor&>(*this), dim, dtype);
#else
    static auto table = globalATenDispatch().getOpTable("aten::cumprod(Tensor self, int dim, *, ScalarType? dtype=None) -> Tensor");
    return table->getOp<Tensor (const Tensor &, int64_t, c10::optional<ScalarType>)>(at::detail::multi_dispatch_tensor_type_set(*this))(const_cast<Tensor&>(*this), dim, dtype);
#endif
}
inline Tensor Tensor::det() const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::det(const_cast<Tensor&>(*this));
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::det", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}
inline Tensor Tensor::diag_embed(int64_t offset, int64_t dim1, int64_t dim2) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::diag_embed(const_cast<Tensor&>(*this), offset, dim1, dim2);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::diag_embed", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor, const Tensor &, int64_t, int64_t, int64_t>(const_cast<Tensor&>(*this), offset, dim1, dim2);
#endif
}
inline Tensor Tensor::diagflat(int64_t offset) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::diagflat(const_cast<Tensor&>(*this), offset);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::diagflat", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor, const Tensor &, int64_t>(const_cast<Tensor&>(*this), offset);
#endif
}
inline Tensor Tensor::diagonal(int64_t offset, int64_t dim1, int64_t dim2) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::diagonal(const_cast<Tensor&>(*this), offset, dim1, dim2);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::diagonal", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor, const Tensor &, int64_t, int64_t, int64_t>(const_cast<Tensor&>(*this), offset, dim1, dim2);
#endif
}
inline Tensor & Tensor::fill_diagonal_(Scalar fill_value, bool wrap) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::fill_diagonal_(const_cast<Tensor&>(*this), fill_value, wrap);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::fill_diagonal_", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor &, Tensor &, Scalar, bool>(const_cast<Tensor&>(*this), fill_value, wrap);
#endif
}
inline Tensor Tensor::div(const Tensor & other) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::div(const_cast<Tensor&>(*this), other);
            break;
        case Backend::SparseCPU:
            return SparseCPUType::div(const_cast<Tensor&>(*this), other);
            break;
        default:
            AT_ERROR("div not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::div", "Tensor"}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this, other)))
        .callUnboxed<Tensor, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), other);
#endif
}
inline Tensor & Tensor::div_(const Tensor & other) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::div_(const_cast<Tensor&>(*this), other);
            break;
        case Backend::SparseCPU:
            return SparseCPUType::div_(const_cast<Tensor&>(*this), other);
            break;
        default:
            AT_ERROR("div_ not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::div_", "Tensor"}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this, other)))
        .callUnboxed<Tensor &, Tensor &, const Tensor &>(const_cast<Tensor&>(*this), other);
#endif
}
inline Tensor Tensor::div(Scalar other) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::div(const_cast<Tensor&>(*this), other);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::div", "Scalar"}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor, const Tensor &, Scalar>(const_cast<Tensor&>(*this), other);
#endif
}
inline Tensor & Tensor::div_(Scalar other) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::div_(const_cast<Tensor&>(*this), other);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::div_", "Scalar"}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor &, Tensor &, Scalar>(const_cast<Tensor&>(*this), other);
#endif
}
inline Tensor Tensor::dot(const Tensor & tensor) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::dot(const_cast<Tensor&>(*this), tensor);
            break;
        default:
            AT_ERROR("dot not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::dot", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this, tensor)))
        .callUnboxed<Tensor, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), tensor);
#endif
}
inline Tensor Tensor::new_empty(IntArrayRef size, const TensorOptions & options) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::new_empty(const_cast<Tensor&>(*this), size, options);
#else
    static auto table = globalATenDispatch().getOpTable("aten::new_empty(Tensor self, int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor");
    return table->getOp<Tensor (const Tensor &, IntArrayRef, const TensorOptions &)>(at::detail::multi_dispatch_tensor_type_set(*this, options))(const_cast<Tensor&>(*this), size, options);
#endif
}
inline Tensor Tensor::new_full(IntArrayRef size, Scalar fill_value, const TensorOptions & options) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::new_full(const_cast<Tensor&>(*this), size, fill_value, options);
#else
    static auto table = globalATenDispatch().getOpTable("aten::new_full(Tensor self, int[] size, Scalar fill_value, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor");
    return table->getOp<Tensor (const Tensor &, IntArrayRef, Scalar, const TensorOptions &)>(at::detail::multi_dispatch_tensor_type_set(*this, options))(const_cast<Tensor&>(*this), size, fill_value, options);
#endif
}
inline Tensor & Tensor::resize_(IntArrayRef size) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::resize_(const_cast<Tensor&>(*this), size);
            break;
        default:
            AT_ERROR("resize_ not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::resize_", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor &, Tensor &, IntArrayRef>(const_cast<Tensor&>(*this), size);
#endif
}
inline Tensor Tensor::erf() const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::erf(const_cast<Tensor&>(*this));
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::erf", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}
inline Tensor & Tensor::erf_() const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::erf_(const_cast<Tensor&>(*this));
            break;
        default:
            AT_ERROR("erf_ not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::erf_", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor &, Tensor &>(const_cast<Tensor&>(*this));
#endif
}
inline Tensor Tensor::erfc() const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::erfc(const_cast<Tensor&>(*this));
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::erfc", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}
inline Tensor & Tensor::erfc_() const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::erfc_(const_cast<Tensor&>(*this));
            break;
        default:
            AT_ERROR("erfc_ not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::erfc_", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor &, Tensor &>(const_cast<Tensor&>(*this));
#endif
}
inline Tensor Tensor::exp() const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::exp(const_cast<Tensor&>(*this));
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::exp", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}
inline Tensor & Tensor::exp_() const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::exp_(const_cast<Tensor&>(*this));
            break;
        default:
            AT_ERROR("exp_ not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::exp_", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor &, Tensor &>(const_cast<Tensor&>(*this));
#endif
}
inline Tensor Tensor::expm1() const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::expm1(const_cast<Tensor&>(*this));
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::expm1", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}
inline Tensor & Tensor::expm1_() const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::expm1_(const_cast<Tensor&>(*this));
            break;
        default:
            AT_ERROR("expm1_ not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::expm1_", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor &, Tensor &>(const_cast<Tensor&>(*this));
#endif
}
inline Tensor Tensor::expand(IntArrayRef size, bool implicit) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::expand(const_cast<Tensor&>(*this), size, implicit);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::expand", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor, const Tensor &, IntArrayRef, bool>(const_cast<Tensor&>(*this), size, implicit);
#endif
}
inline Tensor Tensor::expand_as(const Tensor & other) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::expand_as(const_cast<Tensor&>(*this), other);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::expand_as", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this, other)))
        .callUnboxed<Tensor, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), other);
#endif
}
inline Tensor Tensor::flatten(int64_t start_dim, int64_t end_dim) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::flatten(const_cast<Tensor&>(*this), start_dim, end_dim);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::flatten", "using_ints"}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor, const Tensor &, int64_t, int64_t>(const_cast<Tensor&>(*this), start_dim, end_dim);
#endif
}
#ifdef BUILD_NAMEDTENSOR
inline Tensor Tensor::flatten(int64_t start_dim, int64_t end_dim, Dimname out_dim) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::flatten(const_cast<Tensor&>(*this), start_dim, end_dim, out_dim);
#else
    static auto table = globalATenDispatch().getOpTable("aten::flatten.named_out_dim(Tensor self, int start_dim, int end_dim, Dimname out_dim) -> Tensor");
    return table->getOp<Tensor (const Tensor &, int64_t, int64_t, Dimname)>(at::detail::multi_dispatch_tensor_type_set(*this))(const_cast<Tensor&>(*this), start_dim, end_dim, out_dim);
#endif
}
#endif
#ifdef BUILD_NAMEDTENSOR
inline Tensor Tensor::flatten(Dimname start_dim, Dimname end_dim, Dimname out_dim) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::flatten(const_cast<Tensor&>(*this), start_dim, end_dim, out_dim);
#else
    static auto table = globalATenDispatch().getOpTable("aten::flatten.using_names(Tensor self, Dimname start_dim, Dimname end_dim, Dimname out_dim) -> Tensor");
    return table->getOp<Tensor (const Tensor &, Dimname, Dimname, Dimname)>(at::detail::multi_dispatch_tensor_type_set(*this))(const_cast<Tensor&>(*this), start_dim, end_dim, out_dim);
#endif
}
#endif
#ifdef BUILD_NAMEDTENSOR
inline Tensor Tensor::flatten(DimnameList dims, Dimname out_dim) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::flatten(const_cast<Tensor&>(*this), dims, out_dim);
#else
    static auto table = globalATenDispatch().getOpTable("aten::flatten.DimnameList(Tensor self, DimnameList dims, Dimname out_dim) -> Tensor");
    return table->getOp<Tensor (const Tensor &, DimnameList, Dimname)>(at::detail::multi_dispatch_tensor_type_set(*this))(const_cast<Tensor&>(*this), dims, out_dim);
#endif
}
#endif
inline Tensor & Tensor::fill_(Scalar value) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::fill_(const_cast<Tensor&>(*this), value);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::fill_", "Scalar"}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor &, Tensor &, Scalar>(const_cast<Tensor&>(*this), value);
#endif
}
inline Tensor & Tensor::fill_(const Tensor & value) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::fill_(const_cast<Tensor&>(*this), value);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::fill_", "Tensor"}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this, value)))
        .callUnboxed<Tensor &, Tensor &, const Tensor &>(const_cast<Tensor&>(*this), value);
#endif
}
inline Tensor Tensor::floor() const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::floor(const_cast<Tensor&>(*this));
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::floor", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}
inline Tensor & Tensor::floor_() const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::floor_(const_cast<Tensor&>(*this));
            break;
        default:
            AT_ERROR("floor_ not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::floor_", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor &, Tensor &>(const_cast<Tensor&>(*this));
#endif
}
inline Tensor Tensor::frac() const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::frac(const_cast<Tensor&>(*this));
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::frac", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}
inline Tensor & Tensor::frac_() const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::frac_(const_cast<Tensor&>(*this));
            break;
        default:
            AT_ERROR("frac_ not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::frac_", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor &, Tensor &>(const_cast<Tensor&>(*this));
#endif
}
inline Tensor Tensor::ger(const Tensor & vec2) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::ger(const_cast<Tensor&>(*this), vec2);
            break;
        default:
            AT_ERROR("ger not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::ger", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this, vec2)))
        .callUnboxed<Tensor, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), vec2);
#endif
}
inline Tensor Tensor::fft(int64_t signal_ndim, bool normalized) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::fft(const_cast<Tensor&>(*this), signal_ndim, normalized);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::fft", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor, const Tensor &, int64_t, bool>(const_cast<Tensor&>(*this), signal_ndim, normalized);
#endif
}
inline Tensor Tensor::ifft(int64_t signal_ndim, bool normalized) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::ifft(const_cast<Tensor&>(*this), signal_ndim, normalized);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::ifft", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor, const Tensor &, int64_t, bool>(const_cast<Tensor&>(*this), signal_ndim, normalized);
#endif
}
inline Tensor Tensor::rfft(int64_t signal_ndim, bool normalized, bool onesided) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::rfft(const_cast<Tensor&>(*this), signal_ndim, normalized, onesided);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::rfft", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor, const Tensor &, int64_t, bool, bool>(const_cast<Tensor&>(*this), signal_ndim, normalized, onesided);
#endif
}
inline Tensor Tensor::irfft(int64_t signal_ndim, bool normalized, bool onesided, IntArrayRef signal_sizes) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::irfft(const_cast<Tensor&>(*this), signal_ndim, normalized, onesided, signal_sizes);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::irfft", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor, const Tensor &, int64_t, bool, bool, IntArrayRef>(const_cast<Tensor&>(*this), signal_ndim, normalized, onesided, signal_sizes);
#endif
}
inline Tensor Tensor::index(TensorList indices) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::index(const_cast<Tensor&>(*this), indices);
#else
    static auto table = globalATenDispatch().getOpTable("aten::index.Tensor(Tensor self, Tensor?[] indices) -> Tensor");
    return table->getOp<Tensor (const Tensor &, TensorList)>(at::detail::multi_dispatch_tensor_type_set(*this, indices))(const_cast<Tensor&>(*this), indices);
#endif
}
inline Tensor & Tensor::index_copy_(int64_t dim, const Tensor & index, const Tensor & source) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::index_copy_(const_cast<Tensor&>(*this), dim, index, source);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::index_copy_", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this, index, source)))
        .callUnboxed<Tensor &, Tensor &, int64_t, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), dim, index, source);
#endif
}
inline Tensor Tensor::index_copy(int64_t dim, const Tensor & index, const Tensor & source) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::index_copy(const_cast<Tensor&>(*this), dim, index, source);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::index_copy", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this, index, source)))
        .callUnboxed<Tensor, const Tensor &, int64_t, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), dim, index, source);
#endif
}
inline Tensor & Tensor::index_put_(TensorList indices, const Tensor & values, bool accumulate) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::index_put_(const_cast<Tensor&>(*this), indices, values, accumulate);
#else
    static auto table = globalATenDispatch().getOpTable("aten::index_put_(Tensor(a!) self, Tensor?[] indices, Tensor values, bool accumulate=False) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, TensorList, const Tensor &, bool)>(at::detail::multi_dispatch_tensor_type_set(*this, indices, values))(const_cast<Tensor&>(*this), indices, values, accumulate);
#endif
}
inline Tensor Tensor::index_put(TensorList indices, const Tensor & values, bool accumulate) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::index_put(const_cast<Tensor&>(*this), indices, values, accumulate);
#else
    static auto table = globalATenDispatch().getOpTable("aten::index_put(Tensor self, Tensor?[] indices, Tensor values, bool accumulate=False) -> Tensor");
    return table->getOp<Tensor (const Tensor &, TensorList, const Tensor &, bool)>(at::detail::multi_dispatch_tensor_type_set(*this, indices, values))(const_cast<Tensor&>(*this), indices, values, accumulate);
#endif
}
inline Tensor Tensor::inverse() const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::inverse(const_cast<Tensor&>(*this));
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::inverse", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}
inline Tensor Tensor::isclose(const Tensor & other, double rtol, double atol, bool equal_nan) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::isclose(const_cast<Tensor&>(*this), other, rtol, atol, equal_nan);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::isclose", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this, other)))
        .callUnboxed<Tensor, const Tensor &, const Tensor &, double, double, bool>(const_cast<Tensor&>(*this), other, rtol, atol, equal_nan);
#endif
}
inline bool Tensor::is_distributed() const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::is_distributed(const_cast<Tensor&>(*this));
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::is_distributed", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<bool, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}
inline bool Tensor::is_floating_point() const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::is_floating_point(const_cast<Tensor&>(*this));
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::is_floating_point", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<bool, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}
inline bool Tensor::is_complex() const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::is_complex(const_cast<Tensor&>(*this));
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::is_complex", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<bool, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}
inline bool Tensor::is_nonzero() const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::is_nonzero(const_cast<Tensor&>(*this));
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::is_nonzero", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<bool, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}
inline bool Tensor::is_same_size(const Tensor & other) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::is_same_size(const_cast<Tensor&>(*this), other);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::is_same_size", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this, other)))
        .callUnboxed<bool, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), other);
#endif
}
inline bool Tensor::is_signed() const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::is_signed(const_cast<Tensor&>(*this));
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::is_signed", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<bool, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}
inline std::tuple<Tensor,Tensor> Tensor::kthvalue(int64_t k, int64_t dim, bool keepdim) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::kthvalue(const_cast<Tensor&>(*this), k, dim, keepdim);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::kthvalue", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<std::tuple<Tensor,Tensor>, const Tensor &, int64_t, int64_t, bool>(const_cast<Tensor&>(*this), k, dim, keepdim);
#endif
}
inline Tensor Tensor::log() const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::log(const_cast<Tensor&>(*this));
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::log", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}
inline Tensor & Tensor::log_() const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::log_(const_cast<Tensor&>(*this));
            break;
        default:
            AT_ERROR("log_ not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::log_", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor &, Tensor &>(const_cast<Tensor&>(*this));
#endif
}
inline Tensor Tensor::log10() const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::log10(const_cast<Tensor&>(*this));
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::log10", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}
inline Tensor & Tensor::log10_() const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::log10_(const_cast<Tensor&>(*this));
            break;
        default:
            AT_ERROR("log10_ not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::log10_", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor &, Tensor &>(const_cast<Tensor&>(*this));
#endif
}
inline Tensor Tensor::log1p() const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::log1p(const_cast<Tensor&>(*this));
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::log1p", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}
inline Tensor & Tensor::log1p_() const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::log1p_(const_cast<Tensor&>(*this));
            break;
        case Backend::SparseCPU:
            return SparseCPUType::log1p_(const_cast<Tensor&>(*this));
            break;
        default:
            AT_ERROR("log1p_ not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::log1p_", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor &, Tensor &>(const_cast<Tensor&>(*this));
#endif
}
inline Tensor Tensor::log2() const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::log2(const_cast<Tensor&>(*this));
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::log2", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}
inline Tensor & Tensor::log2_() const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::log2_(const_cast<Tensor&>(*this));
            break;
        default:
            AT_ERROR("log2_ not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::log2_", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor &, Tensor &>(const_cast<Tensor&>(*this));
#endif
}
inline Tensor Tensor::logdet() const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::logdet(const_cast<Tensor&>(*this));
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::logdet", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}
inline Tensor Tensor::log_softmax(int64_t dim, c10::optional<ScalarType> dtype) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::log_softmax(const_cast<Tensor&>(*this), dim, dtype);
#else
    static auto table = globalATenDispatch().getOpTable("aten::log_softmax(Tensor self, int dim, ScalarType? dtype=None) -> Tensor");
    return table->getOp<Tensor (const Tensor &, int64_t, c10::optional<ScalarType>)>(at::detail::multi_dispatch_tensor_type_set(*this))(const_cast<Tensor&>(*this), dim, dtype);
#endif
}
#ifdef BUILD_NAMEDTENSOR
inline Tensor Tensor::log_softmax(Dimname dim, c10::optional<ScalarType> dtype) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::log_softmax(const_cast<Tensor&>(*this), dim, dtype);
#else
    static auto table = globalATenDispatch().getOpTable("aten::log_softmax(Tensor self, Dimname dim, *, ScalarType? dtype=None) -> Tensor");
    return table->getOp<Tensor (const Tensor &, Dimname, c10::optional<ScalarType>)>(at::detail::multi_dispatch_tensor_type_set(*this))(const_cast<Tensor&>(*this), dim, dtype);
#endif
}
#endif
inline Tensor Tensor::logsumexp(IntArrayRef dim, bool keepdim) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::logsumexp(const_cast<Tensor&>(*this), dim, keepdim);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::logsumexp", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor, const Tensor &, IntArrayRef, bool>(const_cast<Tensor&>(*this), dim, keepdim);
#endif
}
#ifdef BUILD_NAMEDTENSOR
inline Tensor Tensor::logsumexp(DimnameList dim, bool keepdim) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::logsumexp(const_cast<Tensor&>(*this), dim, keepdim);
#else
    static auto table = globalATenDispatch().getOpTable("aten::logsumexp.names(Tensor self, Dimname[1] dim, bool keepdim=False) -> Tensor");
    return table->getOp<Tensor (const Tensor &, DimnameList, bool)>(at::detail::multi_dispatch_tensor_type_set(*this))(const_cast<Tensor&>(*this), dim, keepdim);
#endif
}
#endif
inline Tensor Tensor::matmul(const Tensor & other) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::matmul(const_cast<Tensor&>(*this), other);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::matmul", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this, other)))
        .callUnboxed<Tensor, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), other);
#endif
}
inline Tensor Tensor::matrix_power(int64_t n) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::matrix_power(const_cast<Tensor&>(*this), n);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::matrix_power", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor, const Tensor &, int64_t>(const_cast<Tensor&>(*this), n);
#endif
}
inline std::tuple<Tensor,Tensor> Tensor::max(int64_t dim, bool keepdim) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::max(const_cast<Tensor&>(*this), dim, keepdim);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::max", "dim"}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<std::tuple<Tensor,Tensor>, const Tensor &, int64_t, bool>(const_cast<Tensor&>(*this), dim, keepdim);
#endif
}
inline Tensor Tensor::max_values(IntArrayRef dim, bool keepdim) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::max_values(const_cast<Tensor&>(*this), dim, keepdim);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::max_values", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor, const Tensor &, IntArrayRef, bool>(const_cast<Tensor&>(*this), dim, keepdim);
#endif
}
#ifdef BUILD_NAMEDTENSOR
inline std::tuple<Tensor,Tensor> Tensor::max(Dimname dim, bool keepdim) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::max(const_cast<Tensor&>(*this), dim, keepdim);
#else
    static auto table = globalATenDispatch().getOpTable("aten::max.names_dim(Tensor self, Dimname dim, bool keepdim=False) -> (Tensor values, Tensor indices)");
    return table->getOp<std::tuple<Tensor,Tensor> (const Tensor &, Dimname, bool)>(at::detail::multi_dispatch_tensor_type_set(*this))(const_cast<Tensor&>(*this), dim, keepdim);
#endif
}
#endif
#ifdef BUILD_NAMEDTENSOR
inline Tensor Tensor::max_values(DimnameList dim, bool keepdim) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::max_values(const_cast<Tensor&>(*this), dim, keepdim);
#else
    static auto table = globalATenDispatch().getOpTable("aten::max_values.names(Tensor self, Dimname[1] dim, bool keepdim=False) -> Tensor");
    return table->getOp<Tensor (const Tensor &, DimnameList, bool)>(at::detail::multi_dispatch_tensor_type_set(*this))(const_cast<Tensor&>(*this), dim, keepdim);
#endif
}
#endif
inline Tensor Tensor::mean(c10::optional<ScalarType> dtype) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::mean(const_cast<Tensor&>(*this), dtype);
#else
    static auto table = globalATenDispatch().getOpTable("aten::mean(Tensor self, *, ScalarType? dtype=None) -> Tensor");
    return table->getOp<Tensor (const Tensor &, c10::optional<ScalarType>)>(at::detail::multi_dispatch_tensor_type_set(*this))(const_cast<Tensor&>(*this), dtype);
#endif
}
inline Tensor Tensor::mean(IntArrayRef dim, bool keepdim, c10::optional<ScalarType> dtype) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::mean(const_cast<Tensor&>(*this), dim, keepdim, dtype);
#else
    static auto table = globalATenDispatch().getOpTable("aten::mean.dim(Tensor self, int[1] dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor");
    return table->getOp<Tensor (const Tensor &, IntArrayRef, bool, c10::optional<ScalarType>)>(at::detail::multi_dispatch_tensor_type_set(*this))(const_cast<Tensor&>(*this), dim, keepdim, dtype);
#endif
}
#ifdef BUILD_NAMEDTENSOR
inline Tensor Tensor::mean(DimnameList dim, bool keepdim, c10::optional<ScalarType> dtype) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::mean(const_cast<Tensor&>(*this), dim, keepdim, dtype);
#else
    static auto table = globalATenDispatch().getOpTable("aten::mean.names_dim(Tensor self, Dimname[1] dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor");
    return table->getOp<Tensor (const Tensor &, DimnameList, bool, c10::optional<ScalarType>)>(at::detail::multi_dispatch_tensor_type_set(*this))(const_cast<Tensor&>(*this), dim, keepdim, dtype);
#endif
}
#endif
inline std::tuple<Tensor,Tensor> Tensor::median(int64_t dim, bool keepdim) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::median(const_cast<Tensor&>(*this), dim, keepdim);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::median", "dim"}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<std::tuple<Tensor,Tensor>, const Tensor &, int64_t, bool>(const_cast<Tensor&>(*this), dim, keepdim);
#endif
}
#ifdef BUILD_NAMEDTENSOR
inline std::tuple<Tensor,Tensor> Tensor::median(Dimname dim, bool keepdim) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::median(const_cast<Tensor&>(*this), dim, keepdim);
#else
    static auto table = globalATenDispatch().getOpTable("aten::median.names_dim(Tensor self, Dimname dim, bool keepdim=False) -> (Tensor values, Tensor indices)");
    return table->getOp<std::tuple<Tensor,Tensor> (const Tensor &, Dimname, bool)>(at::detail::multi_dispatch_tensor_type_set(*this))(const_cast<Tensor&>(*this), dim, keepdim);
#endif
}
#endif
inline std::tuple<Tensor,Tensor> Tensor::min(int64_t dim, bool keepdim) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::min(const_cast<Tensor&>(*this), dim, keepdim);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::min", "dim"}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<std::tuple<Tensor,Tensor>, const Tensor &, int64_t, bool>(const_cast<Tensor&>(*this), dim, keepdim);
#endif
}
inline Tensor Tensor::min_values(IntArrayRef dim, bool keepdim) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::min_values(const_cast<Tensor&>(*this), dim, keepdim);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::min_values", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor, const Tensor &, IntArrayRef, bool>(const_cast<Tensor&>(*this), dim, keepdim);
#endif
}
#ifdef BUILD_NAMEDTENSOR
inline std::tuple<Tensor,Tensor> Tensor::min(Dimname dim, bool keepdim) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::min(const_cast<Tensor&>(*this), dim, keepdim);
#else
    static auto table = globalATenDispatch().getOpTable("aten::min.names_dim(Tensor self, Dimname dim, bool keepdim=False) -> (Tensor values, Tensor indices)");
    return table->getOp<std::tuple<Tensor,Tensor> (const Tensor &, Dimname, bool)>(at::detail::multi_dispatch_tensor_type_set(*this))(const_cast<Tensor&>(*this), dim, keepdim);
#endif
}
#endif
#ifdef BUILD_NAMEDTENSOR
inline Tensor Tensor::min_values(DimnameList dim, bool keepdim) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::min_values(const_cast<Tensor&>(*this), dim, keepdim);
#else
    static auto table = globalATenDispatch().getOpTable("aten::min_values.names(Tensor self, Dimname[1] dim, bool keepdim=False) -> Tensor");
    return table->getOp<Tensor (const Tensor &, DimnameList, bool)>(at::detail::multi_dispatch_tensor_type_set(*this))(const_cast<Tensor&>(*this), dim, keepdim);
#endif
}
#endif
inline Tensor Tensor::mm(const Tensor & mat2) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::mm(const_cast<Tensor&>(*this), mat2);
            break;
        case Backend::SparseCPU:
            return SparseCPUType::mm(const_cast<Tensor&>(*this), mat2);
            break;
        default:
            AT_ERROR("mm not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::mm", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this, mat2)))
        .callUnboxed<Tensor, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), mat2);
#endif
}
inline std::tuple<Tensor,Tensor> Tensor::mode(int64_t dim, bool keepdim) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::mode(const_cast<Tensor&>(*this), dim, keepdim);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::mode", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<std::tuple<Tensor,Tensor>, const Tensor &, int64_t, bool>(const_cast<Tensor&>(*this), dim, keepdim);
#endif
}
inline Tensor Tensor::mul(const Tensor & other) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::mul(const_cast<Tensor&>(*this), other);
            break;
        case Backend::SparseCPU:
            return SparseCPUType::mul(const_cast<Tensor&>(*this), other);
            break;
        default:
            AT_ERROR("mul not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::mul", "Tensor"}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this, other)))
        .callUnboxed<Tensor, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), other);
#endif
}
inline Tensor & Tensor::mul_(const Tensor & other) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::mul_(const_cast<Tensor&>(*this), other);
            break;
        case Backend::SparseCPU:
            return SparseCPUType::mul_(const_cast<Tensor&>(*this), other);
            break;
        default:
            AT_ERROR("mul_ not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::mul_", "Tensor"}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this, other)))
        .callUnboxed<Tensor &, Tensor &, const Tensor &>(const_cast<Tensor&>(*this), other);
#endif
}
inline Tensor Tensor::mul(Scalar other) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::mul(const_cast<Tensor&>(*this), other);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::mul", "Scalar"}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor, const Tensor &, Scalar>(const_cast<Tensor&>(*this), other);
#endif
}
inline Tensor & Tensor::mul_(Scalar other) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::mul_(const_cast<Tensor&>(*this), other);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::mul_", "Scalar"}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor &, Tensor &, Scalar>(const_cast<Tensor&>(*this), other);
#endif
}
inline Tensor Tensor::mv(const Tensor & vec) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::mv(const_cast<Tensor&>(*this), vec);
            break;
        default:
            AT_ERROR("mv not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::mv", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this, vec)))
        .callUnboxed<Tensor, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), vec);
#endif
}
inline Tensor Tensor::mvlgamma(int64_t p) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::mvlgamma(const_cast<Tensor&>(*this), p);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::mvlgamma", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor, const Tensor &, int64_t>(const_cast<Tensor&>(*this), p);
#endif
}
inline Tensor & Tensor::mvlgamma_(int64_t p) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::mvlgamma_(const_cast<Tensor&>(*this), p);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::mvlgamma_", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor &, Tensor &, int64_t>(const_cast<Tensor&>(*this), p);
#endif
}
inline Tensor Tensor::narrow_copy(int64_t dim, int64_t start, int64_t length) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::narrow_copy(const_cast<Tensor&>(*this), dim, start, length);
            break;
        case Backend::SparseCPU:
            return SparseCPUType::narrow_copy(const_cast<Tensor&>(*this), dim, start, length);
            break;
        default:
            AT_ERROR("narrow_copy not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::narrow_copy", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor, const Tensor &, int64_t, int64_t, int64_t>(const_cast<Tensor&>(*this), dim, start, length);
#endif
}
inline Tensor Tensor::narrow(int64_t dim, int64_t start, int64_t length) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::narrow(const_cast<Tensor&>(*this), dim, start, length);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::narrow", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor, const Tensor &, int64_t, int64_t, int64_t>(const_cast<Tensor&>(*this), dim, start, length);
#endif
}
inline Tensor Tensor::permute(IntArrayRef dims) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::permute(const_cast<Tensor&>(*this), dims);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::permute", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor, const Tensor &, IntArrayRef>(const_cast<Tensor&>(*this), dims);
#endif
}
inline Tensor Tensor::numpy_T() const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::numpy_T(const_cast<Tensor&>(*this));
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::numpy_T", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}
inline bool Tensor::is_pinned() const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::is_pinned(const_cast<Tensor&>(*this));
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::is_pinned", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<bool, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}
inline Tensor Tensor::pin_memory() const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::pin_memory(const_cast<Tensor&>(*this));
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::pin_memory", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}
inline Tensor Tensor::pinverse(double rcond) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::pinverse(const_cast<Tensor&>(*this), rcond);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::pinverse", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor, const Tensor &, double>(const_cast<Tensor&>(*this), rcond);
#endif
}
inline Tensor Tensor::reciprocal() const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::reciprocal(const_cast<Tensor&>(*this));
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::reciprocal", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}
inline Tensor & Tensor::reciprocal_() const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::reciprocal_(const_cast<Tensor&>(*this));
            break;
        default:
            AT_ERROR("reciprocal_ not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::reciprocal_", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor &, Tensor &>(const_cast<Tensor&>(*this));
#endif
}
inline Tensor Tensor::neg() const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::neg(const_cast<Tensor&>(*this));
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::neg", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}
inline Tensor & Tensor::neg_() const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::neg_(const_cast<Tensor&>(*this));
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::neg_", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor &, Tensor &>(const_cast<Tensor&>(*this));
#endif
}
inline Tensor Tensor::repeat(IntArrayRef repeats) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::repeat(const_cast<Tensor&>(*this), repeats);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::repeat", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor, const Tensor &, IntArrayRef>(const_cast<Tensor&>(*this), repeats);
#endif
}
inline Tensor Tensor::repeat_interleave(const Tensor & repeats, c10::optional<int64_t> dim) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::repeat_interleave(const_cast<Tensor&>(*this), repeats, dim);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::repeat_interleave", "self_Tensor"}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this, repeats)))
        .callUnboxed<Tensor, const Tensor &, const Tensor &, c10::optional<int64_t>>(const_cast<Tensor&>(*this), repeats, dim);
#endif
}
inline Tensor Tensor::repeat_interleave(int64_t repeats, c10::optional<int64_t> dim) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::repeat_interleave(const_cast<Tensor&>(*this), repeats, dim);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::repeat_interleave", "self_int"}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor, const Tensor &, int64_t, c10::optional<int64_t>>(const_cast<Tensor&>(*this), repeats, dim);
#endif
}
inline Tensor Tensor::reshape(IntArrayRef shape) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::reshape(const_cast<Tensor&>(*this), shape);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::reshape", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor, const Tensor &, IntArrayRef>(const_cast<Tensor&>(*this), shape);
#endif
}
inline Tensor Tensor::reshape_as(const Tensor & other) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::reshape_as(const_cast<Tensor&>(*this), other);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::reshape_as", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this, other)))
        .callUnboxed<Tensor, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), other);
#endif
}
inline Tensor Tensor::round() const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::round(const_cast<Tensor&>(*this));
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::round", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}
inline Tensor & Tensor::round_() const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::round_(const_cast<Tensor&>(*this));
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::round_", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor &, Tensor &>(const_cast<Tensor&>(*this));
#endif
}
inline Tensor Tensor::relu() const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::relu(const_cast<Tensor&>(*this));
            break;
        case Backend::QuantizedCPU:
            return QuantizedCPUType::relu(const_cast<Tensor&>(*this));
            break;
        default:
            AT_ERROR("relu not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::relu", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}
inline Tensor & Tensor::relu_() const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::relu_(const_cast<Tensor&>(*this));
            break;
        case Backend::QuantizedCPU:
            return QuantizedCPUType::relu_(const_cast<Tensor&>(*this));
            break;
        default:
            AT_ERROR("relu_ not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::relu_", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor &, Tensor &>(const_cast<Tensor&>(*this));
#endif
}
inline Tensor Tensor::prelu(const Tensor & weight) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::prelu(const_cast<Tensor&>(*this), weight);
            break;
        default:
            AT_ERROR("prelu not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::prelu", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this, weight)))
        .callUnboxed<Tensor, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), weight);
#endif
}
inline std::tuple<Tensor,Tensor> Tensor::prelu_backward(const Tensor & grad_output, const Tensor & weight) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::prelu_backward(grad_output, const_cast<Tensor&>(*this), weight);
            break;
        default:
            AT_ERROR("prelu_backward not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::prelu_backward", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(grad_output, *this, weight)))
        .callUnboxed<std::tuple<Tensor,Tensor>, const Tensor &, const Tensor &, const Tensor &>(grad_output, const_cast<Tensor&>(*this), weight);
#endif
}
inline Tensor Tensor::hardshrink(Scalar lambd) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::hardshrink(const_cast<Tensor&>(*this), lambd);
            break;
        default:
            AT_ERROR("hardshrink not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::hardshrink", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor, const Tensor &, Scalar>(const_cast<Tensor&>(*this), lambd);
#endif
}
inline Tensor Tensor::hardshrink_backward(const Tensor & grad_out, Scalar lambd) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::hardshrink_backward(grad_out, const_cast<Tensor&>(*this), lambd);
            break;
        default:
            AT_ERROR("hardshrink_backward not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::hardshrink_backward", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(grad_out, *this)))
        .callUnboxed<Tensor, const Tensor &, const Tensor &, Scalar>(grad_out, const_cast<Tensor&>(*this), lambd);
#endif
}
inline Tensor Tensor::rsqrt() const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::rsqrt(const_cast<Tensor&>(*this));
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::rsqrt", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}
inline Tensor & Tensor::rsqrt_() const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::rsqrt_(const_cast<Tensor&>(*this));
            break;
        default:
            AT_ERROR("rsqrt_ not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::rsqrt_", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor &, Tensor &>(const_cast<Tensor&>(*this));
#endif
}
#ifdef BUILD_NAMEDTENSOR
inline Tensor Tensor::select(Dimname dim, int64_t index) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::select(const_cast<Tensor&>(*this), dim, index);
#else
    static auto table = globalATenDispatch().getOpTable("aten::select.Dimname(Tensor(a) self, Dimname dim, int index) -> Tensor(a)");
    return table->getOp<Tensor (const Tensor &, Dimname, int64_t)>(at::detail::multi_dispatch_tensor_type_set(*this))(const_cast<Tensor&>(*this), dim, index);
#endif
}
#endif
inline Tensor Tensor::select(int64_t dim, int64_t index) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::select(const_cast<Tensor&>(*this), dim, index);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::select", "int"}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor, const Tensor &, int64_t, int64_t>(const_cast<Tensor&>(*this), dim, index);
#endif
}
inline Tensor Tensor::sigmoid() const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::sigmoid(const_cast<Tensor&>(*this));
            break;
        default:
            AT_ERROR("sigmoid not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::sigmoid", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}
inline Tensor & Tensor::sigmoid_() const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::sigmoid_(const_cast<Tensor&>(*this));
            break;
        default:
            AT_ERROR("sigmoid_ not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::sigmoid_", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor &, Tensor &>(const_cast<Tensor&>(*this));
#endif
}
inline Tensor Tensor::sin() const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::sin(const_cast<Tensor&>(*this));
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::sin", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}
inline Tensor & Tensor::sin_() const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::sin_(const_cast<Tensor&>(*this));
            break;
        default:
            AT_ERROR("sin_ not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::sin_", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor &, Tensor &>(const_cast<Tensor&>(*this));
#endif
}
inline Tensor Tensor::sinh() const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::sinh(const_cast<Tensor&>(*this));
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::sinh", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}
inline Tensor & Tensor::sinh_() const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::sinh_(const_cast<Tensor&>(*this));
            break;
        default:
            AT_ERROR("sinh_ not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::sinh_", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor &, Tensor &>(const_cast<Tensor&>(*this));
#endif
}
inline Tensor Tensor::detach() const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::detach(const_cast<Tensor&>(*this));
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::detach", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}
inline Tensor & Tensor::detach_() const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::detach_(const_cast<Tensor&>(*this));
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::detach_", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor &, Tensor &>(const_cast<Tensor&>(*this));
#endif
}
inline int64_t Tensor::size(int64_t dim) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::size(const_cast<Tensor&>(*this), dim);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::size", "int"}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<int64_t, const Tensor &, int64_t>(const_cast<Tensor&>(*this), dim);
#endif
}
#ifdef BUILD_NAMEDTENSOR
inline int64_t Tensor::size(Dimname dim) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::size(const_cast<Tensor&>(*this), dim);
#else
    static auto table = globalATenDispatch().getOpTable("aten::size.Dimname(Tensor self, Dimname dim) -> int");
    return table->getOp<int64_t (const Tensor &, Dimname)>(at::detail::multi_dispatch_tensor_type_set(*this))(const_cast<Tensor&>(*this), dim);
#endif
}
#endif
inline Tensor Tensor::slice(int64_t dim, int64_t start, int64_t end, int64_t step) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::slice(const_cast<Tensor&>(*this), dim, start, end, step);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::slice", "Tensor"}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor, const Tensor &, int64_t, int64_t, int64_t, int64_t>(const_cast<Tensor&>(*this), dim, start, end, step);
#endif
}
inline std::tuple<Tensor,Tensor> Tensor::slogdet() const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::slogdet(const_cast<Tensor&>(*this));
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::slogdet", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<std::tuple<Tensor,Tensor>, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}
inline Tensor Tensor::smm(const Tensor & mat2) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::smm(const_cast<Tensor&>(*this), mat2);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::smm", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this, mat2)))
        .callUnboxed<Tensor, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), mat2);
#endif
}
inline Tensor Tensor::softmax(int64_t dim, c10::optional<ScalarType> dtype) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::softmax(const_cast<Tensor&>(*this), dim, dtype);
#else
    static auto table = globalATenDispatch().getOpTable("aten::softmax(Tensor self, int dim, ScalarType? dtype=None) -> Tensor");
    return table->getOp<Tensor (const Tensor &, int64_t, c10::optional<ScalarType>)>(at::detail::multi_dispatch_tensor_type_set(*this))(const_cast<Tensor&>(*this), dim, dtype);
#endif
}
#ifdef BUILD_NAMEDTENSOR
inline Tensor Tensor::softmax(Dimname dim, c10::optional<ScalarType> dtype) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::softmax(const_cast<Tensor&>(*this), dim, dtype);
#else
    static auto table = globalATenDispatch().getOpTable("aten::softmax(Tensor self, Dimname dim, *, ScalarType? dtype=None) -> Tensor");
    return table->getOp<Tensor (const Tensor &, Dimname, c10::optional<ScalarType>)>(at::detail::multi_dispatch_tensor_type_set(*this))(const_cast<Tensor&>(*this), dim, dtype);
#endif
}
#endif
inline std::vector<Tensor> Tensor::split(int64_t split_size, int64_t dim) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::split(const_cast<Tensor&>(*this), split_size, dim);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::split", "Tensor"}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<std::vector<Tensor>, const Tensor &, int64_t, int64_t>(const_cast<Tensor&>(*this), split_size, dim);
#endif
}
inline std::vector<Tensor> Tensor::split_with_sizes(IntArrayRef split_sizes, int64_t dim) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::split_with_sizes(const_cast<Tensor&>(*this), split_sizes, dim);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::split_with_sizes", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<std::vector<Tensor>, const Tensor &, IntArrayRef, int64_t>(const_cast<Tensor&>(*this), split_sizes, dim);
#endif
}
inline Tensor Tensor::squeeze() const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::squeeze(const_cast<Tensor&>(*this));
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::squeeze", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}
inline Tensor Tensor::squeeze(int64_t dim) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::squeeze(const_cast<Tensor&>(*this), dim);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::squeeze", "dim"}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor, const Tensor &, int64_t>(const_cast<Tensor&>(*this), dim);
#endif
}
inline Tensor & Tensor::squeeze_() const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::squeeze_(const_cast<Tensor&>(*this));
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::squeeze_", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor &, Tensor &>(const_cast<Tensor&>(*this));
#endif
}
inline Tensor & Tensor::squeeze_(int64_t dim) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::squeeze_(const_cast<Tensor&>(*this), dim);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::squeeze_", "dim"}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor &, Tensor &, int64_t>(const_cast<Tensor&>(*this), dim);
#endif
}
inline Tensor Tensor::sspaddmm(const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::sspaddmm(const_cast<Tensor&>(*this), mat1, mat2, beta, alpha);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::sspaddmm", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this, mat1, mat2)))
        .callUnboxed<Tensor, const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar>(const_cast<Tensor&>(*this), mat1, mat2, beta, alpha);
#endif
}
inline Tensor Tensor::stft(int64_t n_fft, c10::optional<int64_t> hop_length, c10::optional<int64_t> win_length, const Tensor & window, bool normalized, bool onesided) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::stft(const_cast<Tensor&>(*this), n_fft, hop_length, win_length, window, normalized, onesided);
#else
    static auto table = globalATenDispatch().getOpTable("aten::stft(Tensor self, int n_fft, int? hop_length=None, int? win_length=None, Tensor? window=None, bool normalized=False, bool onesided=True) -> Tensor");
    return table->getOp<Tensor (const Tensor &, int64_t, c10::optional<int64_t>, c10::optional<int64_t>, const Tensor &, bool, bool)>(at::detail::multi_dispatch_tensor_type_set(*this, window))(const_cast<Tensor&>(*this), n_fft, hop_length, win_length, window, normalized, onesided);
#endif
}
inline int64_t Tensor::stride(int64_t dim) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::stride(const_cast<Tensor&>(*this), dim);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::stride", "int"}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<int64_t, const Tensor &, int64_t>(const_cast<Tensor&>(*this), dim);
#endif
}
#ifdef BUILD_NAMEDTENSOR
inline int64_t Tensor::stride(Dimname dim) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::stride(const_cast<Tensor&>(*this), dim);
#else
    static auto table = globalATenDispatch().getOpTable("aten::stride.Dimname(Tensor self, Dimname dim) -> int");
    return table->getOp<int64_t (const Tensor &, Dimname)>(at::detail::multi_dispatch_tensor_type_set(*this))(const_cast<Tensor&>(*this), dim);
#endif
}
#endif
inline Tensor Tensor::sum(c10::optional<ScalarType> dtype) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::sum(const_cast<Tensor&>(*this), dtype);
#else
    static auto table = globalATenDispatch().getOpTable("aten::sum(Tensor self, *, ScalarType? dtype=None) -> Tensor");
    return table->getOp<Tensor (const Tensor &, c10::optional<ScalarType>)>(at::detail::multi_dispatch_tensor_type_set(*this))(const_cast<Tensor&>(*this), dtype);
#endif
}
inline Tensor Tensor::sum(IntArrayRef dim, bool keepdim, c10::optional<ScalarType> dtype) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::sum(const_cast<Tensor&>(*this), dim, keepdim, dtype);
#else
    static auto table = globalATenDispatch().getOpTable("aten::sum.dim_IntList(Tensor self, int[1] dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor");
    return table->getOp<Tensor (const Tensor &, IntArrayRef, bool, c10::optional<ScalarType>)>(at::detail::multi_dispatch_tensor_type_set(*this))(const_cast<Tensor&>(*this), dim, keepdim, dtype);
#endif
}
#ifdef BUILD_NAMEDTENSOR
inline Tensor Tensor::sum(DimnameList dim, bool keepdim, c10::optional<ScalarType> dtype) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::sum(const_cast<Tensor&>(*this), dim, keepdim, dtype);
#else
    static auto table = globalATenDispatch().getOpTable("aten::sum.dim_DimnameList(Tensor self, Dimname[1] dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor");
    return table->getOp<Tensor (const Tensor &, DimnameList, bool, c10::optional<ScalarType>)>(at::detail::multi_dispatch_tensor_type_set(*this))(const_cast<Tensor&>(*this), dim, keepdim, dtype);
#endif
}
#endif
inline Tensor Tensor::sum_to_size(IntArrayRef size) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::sum_to_size(const_cast<Tensor&>(*this), size);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::sum_to_size", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor, const Tensor &, IntArrayRef>(const_cast<Tensor&>(*this), size);
#endif
}
inline Tensor Tensor::sqrt() const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::sqrt(const_cast<Tensor&>(*this));
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::sqrt", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}
inline Tensor & Tensor::sqrt_() const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::sqrt_(const_cast<Tensor&>(*this));
            break;
        default:
            AT_ERROR("sqrt_ not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::sqrt_", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor &, Tensor &>(const_cast<Tensor&>(*this));
#endif
}
inline Tensor Tensor::std(bool unbiased) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::std(const_cast<Tensor&>(*this), unbiased);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::std", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor, const Tensor &, bool>(const_cast<Tensor&>(*this), unbiased);
#endif
}
inline Tensor Tensor::std(IntArrayRef dim, bool unbiased, bool keepdim) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::std(const_cast<Tensor&>(*this), dim, unbiased, keepdim);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::std", "dim"}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor, const Tensor &, IntArrayRef, bool, bool>(const_cast<Tensor&>(*this), dim, unbiased, keepdim);
#endif
}
#ifdef BUILD_NAMEDTENSOR
inline Tensor Tensor::std(DimnameList dim, bool unbiased, bool keepdim) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::std(const_cast<Tensor&>(*this), dim, unbiased, keepdim);
#else
    static auto table = globalATenDispatch().getOpTable("aten::std.names_dim(Tensor self, Dimname[1] dim, bool unbiased=True, bool keepdim=False) -> Tensor");
    return table->getOp<Tensor (const Tensor &, DimnameList, bool, bool)>(at::detail::multi_dispatch_tensor_type_set(*this))(const_cast<Tensor&>(*this), dim, unbiased, keepdim);
#endif
}
#endif
inline Tensor Tensor::prod(c10::optional<ScalarType> dtype) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::prod(const_cast<Tensor&>(*this), dtype);
#else
    static auto table = globalATenDispatch().getOpTable("aten::prod(Tensor self, *, ScalarType? dtype=None) -> Tensor");
    return table->getOp<Tensor (const Tensor &, c10::optional<ScalarType>)>(at::detail::multi_dispatch_tensor_type_set(*this))(const_cast<Tensor&>(*this), dtype);
#endif
}
inline Tensor Tensor::prod(int64_t dim, bool keepdim, c10::optional<ScalarType> dtype) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::prod(const_cast<Tensor&>(*this), dim, keepdim, dtype);
#else
    static auto table = globalATenDispatch().getOpTable("aten::prod.dim_int(Tensor self, int dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor");
    return table->getOp<Tensor (const Tensor &, int64_t, bool, c10::optional<ScalarType>)>(at::detail::multi_dispatch_tensor_type_set(*this))(const_cast<Tensor&>(*this), dim, keepdim, dtype);
#endif
}
#ifdef BUILD_NAMEDTENSOR
inline Tensor Tensor::prod(Dimname dim, bool keepdim, c10::optional<ScalarType> dtype) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::prod(const_cast<Tensor&>(*this), dim, keepdim, dtype);
#else
    static auto table = globalATenDispatch().getOpTable("aten::prod.dim_Dimname(Tensor self, Dimname dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor");
    return table->getOp<Tensor (const Tensor &, Dimname, bool, c10::optional<ScalarType>)>(at::detail::multi_dispatch_tensor_type_set(*this))(const_cast<Tensor&>(*this), dim, keepdim, dtype);
#endif
}
#endif
inline Tensor Tensor::t() const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::t(const_cast<Tensor&>(*this));
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::t", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}
inline Tensor & Tensor::t_() const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::t_(const_cast<Tensor&>(*this));
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::t_", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor &, Tensor &>(const_cast<Tensor&>(*this));
#endif
}
inline Tensor Tensor::tan() const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::tan(const_cast<Tensor&>(*this));
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::tan", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}
inline Tensor & Tensor::tan_() const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::tan_(const_cast<Tensor&>(*this));
            break;
        default:
            AT_ERROR("tan_ not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::tan_", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor &, Tensor &>(const_cast<Tensor&>(*this));
#endif
}
inline Tensor Tensor::tanh() const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::tanh(const_cast<Tensor&>(*this));
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::tanh", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}
inline Tensor & Tensor::tanh_() const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::tanh_(const_cast<Tensor&>(*this));
            break;
        default:
            AT_ERROR("tanh_ not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::tanh_", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor &, Tensor &>(const_cast<Tensor&>(*this));
#endif
}
inline Tensor Tensor::transpose(int64_t dim0, int64_t dim1) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::transpose(const_cast<Tensor&>(*this), dim0, dim1);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::transpose", "int"}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor, const Tensor &, int64_t, int64_t>(const_cast<Tensor&>(*this), dim0, dim1);
#endif
}
#ifdef BUILD_NAMEDTENSOR
inline Tensor Tensor::transpose(Dimname dim0, Dimname dim1) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::transpose(const_cast<Tensor&>(*this), dim0, dim1);
#else
    static auto table = globalATenDispatch().getOpTable("aten::transpose.Dimname(Tensor(a) self, Dimname dim0, Dimname dim1) -> Tensor(a)");
    return table->getOp<Tensor (const Tensor &, Dimname, Dimname)>(at::detail::multi_dispatch_tensor_type_set(*this))(const_cast<Tensor&>(*this), dim0, dim1);
#endif
}
#endif
inline Tensor & Tensor::transpose_(int64_t dim0, int64_t dim1) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::transpose_(const_cast<Tensor&>(*this), dim0, dim1);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::transpose_", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor &, Tensor &, int64_t, int64_t>(const_cast<Tensor&>(*this), dim0, dim1);
#endif
}
inline Tensor Tensor::flip(IntArrayRef dims) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::flip(const_cast<Tensor&>(*this), dims);
            break;
        default:
            AT_ERROR("flip not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::flip", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor, const Tensor &, IntArrayRef>(const_cast<Tensor&>(*this), dims);
#endif
}
inline Tensor Tensor::roll(IntArrayRef shifts, IntArrayRef dims) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::roll(const_cast<Tensor&>(*this), shifts, dims);
            break;
        default:
            AT_ERROR("roll not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::roll", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor, const Tensor &, IntArrayRef, IntArrayRef>(const_cast<Tensor&>(*this), shifts, dims);
#endif
}
inline Tensor Tensor::rot90(int64_t k, IntArrayRef dims) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::rot90(const_cast<Tensor&>(*this), k, dims);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::rot90", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor, const Tensor &, int64_t, IntArrayRef>(const_cast<Tensor&>(*this), k, dims);
#endif
}
inline Tensor Tensor::trunc() const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::trunc(const_cast<Tensor&>(*this));
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::trunc", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}
inline Tensor & Tensor::trunc_() const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::trunc_(const_cast<Tensor&>(*this));
            break;
        default:
            AT_ERROR("trunc_ not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::trunc_", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor &, Tensor &>(const_cast<Tensor&>(*this));
#endif
}
inline Tensor Tensor::type_as(const Tensor & other) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::type_as(const_cast<Tensor&>(*this), other);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::type_as", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this, other)))
        .callUnboxed<Tensor, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), other);
#endif
}
inline Tensor Tensor::unsqueeze(int64_t dim) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::unsqueeze(const_cast<Tensor&>(*this), dim);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::unsqueeze", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor, const Tensor &, int64_t>(const_cast<Tensor&>(*this), dim);
#endif
}
inline Tensor & Tensor::unsqueeze_(int64_t dim) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::unsqueeze_(const_cast<Tensor&>(*this), dim);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::unsqueeze_", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor &, Tensor &, int64_t>(const_cast<Tensor&>(*this), dim);
#endif
}
inline Tensor Tensor::var(bool unbiased) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::var(const_cast<Tensor&>(*this), unbiased);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::var", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor, const Tensor &, bool>(const_cast<Tensor&>(*this), unbiased);
#endif
}
inline Tensor Tensor::var(IntArrayRef dim, bool unbiased, bool keepdim) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::var(const_cast<Tensor&>(*this), dim, unbiased, keepdim);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::var", "dim"}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor, const Tensor &, IntArrayRef, bool, bool>(const_cast<Tensor&>(*this), dim, unbiased, keepdim);
#endif
}
#ifdef BUILD_NAMEDTENSOR
inline Tensor Tensor::var(DimnameList dim, bool unbiased, bool keepdim) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::var(const_cast<Tensor&>(*this), dim, unbiased, keepdim);
#else
    static auto table = globalATenDispatch().getOpTable("aten::var.names_dim(Tensor self, Dimname[1] dim, bool unbiased=True, bool keepdim=False) -> Tensor");
    return table->getOp<Tensor (const Tensor &, DimnameList, bool, bool)>(at::detail::multi_dispatch_tensor_type_set(*this))(const_cast<Tensor&>(*this), dim, unbiased, keepdim);
#endif
}
#endif
inline Tensor Tensor::view_as(const Tensor & other) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::view_as(const_cast<Tensor&>(*this), other);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::view_as", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this, other)))
        .callUnboxed<Tensor, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), other);
#endif
}
inline Tensor Tensor::where(const Tensor & condition, const Tensor & other) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::where(condition, const_cast<Tensor&>(*this), other);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::where", "self"}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(condition, *this, other)))
        .callUnboxed<Tensor, const Tensor &, const Tensor &, const Tensor &>(condition, const_cast<Tensor&>(*this), other);
#endif
}
inline Tensor Tensor::norm(c10::optional<Scalar> p, ScalarType dtype) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::norm(const_cast<Tensor&>(*this), p, dtype);
#else
    static auto table = globalATenDispatch().getOpTable("aten::norm.ScalarOpt_dtype(Tensor self, Scalar? p, *, ScalarType dtype) -> Tensor");
    return table->getOp<Tensor (const Tensor &, c10::optional<Scalar>, ScalarType)>(at::detail::multi_dispatch_tensor_type_set(*this))(const_cast<Tensor&>(*this), p, dtype);
#endif
}
inline Tensor Tensor::norm(Scalar p) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::norm(const_cast<Tensor&>(*this), p);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::norm", "Scalar"}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor, const Tensor &, Scalar>(const_cast<Tensor&>(*this), p);
#endif
}
inline Tensor Tensor::norm(c10::optional<Scalar> p, IntArrayRef dim, bool keepdim, ScalarType dtype) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::norm(const_cast<Tensor&>(*this), p, dim, keepdim, dtype);
#else
    static auto table = globalATenDispatch().getOpTable("aten::norm.ScalarOpt_dim_dtype(Tensor self, Scalar? p, int[1] dim, bool keepdim, *, ScalarType dtype) -> Tensor");
    return table->getOp<Tensor (const Tensor &, c10::optional<Scalar>, IntArrayRef, bool, ScalarType)>(at::detail::multi_dispatch_tensor_type_set(*this))(const_cast<Tensor&>(*this), p, dim, keepdim, dtype);
#endif
}
inline Tensor Tensor::norm(c10::optional<Scalar> p, IntArrayRef dim, bool keepdim) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::norm(const_cast<Tensor&>(*this), p, dim, keepdim);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::norm", "ScalarOpt_dim"}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor, const Tensor &, c10::optional<Scalar>, IntArrayRef, bool>(const_cast<Tensor&>(*this), p, dim, keepdim);
#endif
}
#ifdef BUILD_NAMEDTENSOR
inline Tensor Tensor::norm(c10::optional<Scalar> p, DimnameList dim, bool keepdim, ScalarType dtype) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::norm(const_cast<Tensor&>(*this), p, dim, keepdim, dtype);
#else
    static auto table = globalATenDispatch().getOpTable("aten::norm.names_ScalarOpt_dim_dtype(Tensor self, Scalar? p, Dimname[1] dim, bool keepdim, *, ScalarType dtype) -> Tensor");
    return table->getOp<Tensor (const Tensor &, c10::optional<Scalar>, DimnameList, bool, ScalarType)>(at::detail::multi_dispatch_tensor_type_set(*this))(const_cast<Tensor&>(*this), p, dim, keepdim, dtype);
#endif
}
#endif
#ifdef BUILD_NAMEDTENSOR
inline Tensor Tensor::norm(c10::optional<Scalar> p, DimnameList dim, bool keepdim) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::norm(const_cast<Tensor&>(*this), p, dim, keepdim);
#else
    static auto table = globalATenDispatch().getOpTable("aten::norm.names_ScalarOpt_dim(Tensor self, Scalar? p, Dimname[1] dim, bool keepdim=False) -> Tensor");
    return table->getOp<Tensor (const Tensor &, c10::optional<Scalar>, DimnameList, bool)>(at::detail::multi_dispatch_tensor_type_set(*this))(const_cast<Tensor&>(*this), p, dim, keepdim);
#endif
}
#endif
inline Tensor Tensor::clone() const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::clone(const_cast<Tensor&>(*this));
            break;
        case Backend::QuantizedCPU:
            return QuantizedCPUType::clone(const_cast<Tensor&>(*this));
            break;
        case Backend::SparseCPU:
            return SparseCPUType::clone(const_cast<Tensor&>(*this));
            break;
        default:
            AT_ERROR("clone not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::clone", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}
inline Tensor & Tensor::resize_as_(const Tensor & the_template) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::resize_as_(const_cast<Tensor&>(*this), the_template);
            break;
        case Backend::SparseCPU:
            return SparseCPUType::resize_as_(const_cast<Tensor&>(*this), the_template);
            break;
        default:
            AT_ERROR("resize_as_ not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::resize_as_", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this, the_template)))
        .callUnboxed<Tensor &, Tensor &, const Tensor &>(const_cast<Tensor&>(*this), the_template);
#endif
}
inline Tensor Tensor::pow(Scalar exponent) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::pow(const_cast<Tensor&>(*this), exponent);
            break;
        case Backend::SparseCPU:
            return SparseCPUType::pow(const_cast<Tensor&>(*this), exponent);
            break;
        default:
            AT_ERROR("pow not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::pow", "Tensor_Scalar"}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor, const Tensor &, Scalar>(const_cast<Tensor&>(*this), exponent);
#endif
}
inline Tensor & Tensor::zero_() const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::zero_(const_cast<Tensor&>(*this));
            break;
        case Backend::SparseCPU:
            return SparseCPUType::zero_(const_cast<Tensor&>(*this));
            break;
        default:
            AT_ERROR("zero_ not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::zero_", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor &, Tensor &>(const_cast<Tensor&>(*this));
#endif
}
inline Tensor Tensor::sub(const Tensor & other, Scalar alpha) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::sub(const_cast<Tensor&>(*this), other, alpha);
            break;
        case Backend::SparseCPU:
            return SparseCPUType::sub(const_cast<Tensor&>(*this), other, alpha);
            break;
        default:
            AT_ERROR("sub not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::sub", "Tensor"}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this, other)))
        .callUnboxed<Tensor, const Tensor &, const Tensor &, Scalar>(const_cast<Tensor&>(*this), other, alpha);
#endif
}
inline Tensor & Tensor::sub_(const Tensor & other, Scalar alpha) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::sub_(const_cast<Tensor&>(*this), other, alpha);
            break;
        case Backend::SparseCPU:
            return SparseCPUType::sub_(const_cast<Tensor&>(*this), other, alpha);
            break;
        default:
            AT_ERROR("sub_ not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::sub_", "Tensor"}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this, other)))
        .callUnboxed<Tensor &, Tensor &, const Tensor &, Scalar>(const_cast<Tensor&>(*this), other, alpha);
#endif
}
inline Tensor Tensor::sub(Scalar other, Scalar alpha) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::sub(const_cast<Tensor&>(*this), other, alpha);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::sub", "Scalar"}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor, const Tensor &, Scalar, Scalar>(const_cast<Tensor&>(*this), other, alpha);
#endif
}
inline Tensor & Tensor::sub_(Scalar other, Scalar alpha) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::sub_(const_cast<Tensor&>(*this), other, alpha);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::sub_", "Scalar"}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor &, Tensor &, Scalar, Scalar>(const_cast<Tensor&>(*this), other, alpha);
#endif
}
inline Tensor Tensor::addmm(const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::addmm(const_cast<Tensor&>(*this), mat1, mat2, beta, alpha);
            break;
        case Backend::SparseCPU:
            return SparseCPUType::addmm(const_cast<Tensor&>(*this), mat1, mat2, beta, alpha);
            break;
        default:
            AT_ERROR("addmm not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::addmm", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this, mat1, mat2)))
        .callUnboxed<Tensor, const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar>(const_cast<Tensor&>(*this), mat1, mat2, beta, alpha);
#endif
}
inline Tensor & Tensor::addmm_(const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::addmm_(const_cast<Tensor&>(*this), mat1, mat2, beta, alpha);
            break;
        case Backend::SparseCPU:
            return SparseCPUType::addmm_(const_cast<Tensor&>(*this), mat1, mat2, beta, alpha);
            break;
        default:
            AT_ERROR("addmm_ not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::addmm_", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this, mat1, mat2)))
        .callUnboxed<Tensor &, Tensor &, const Tensor &, const Tensor &, Scalar, Scalar>(const_cast<Tensor&>(*this), mat1, mat2, beta, alpha);
#endif
}
inline Tensor & Tensor::sparse_resize_(IntArrayRef size, int64_t sparse_dim, int64_t dense_dim) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::SparseCPU:
            return SparseCPUType::sparse_resize_(const_cast<Tensor&>(*this), size, sparse_dim, dense_dim);
            break;
        default:
            AT_ERROR("sparse_resize_ not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::sparse_resize_", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor &, Tensor &, IntArrayRef, int64_t, int64_t>(const_cast<Tensor&>(*this), size, sparse_dim, dense_dim);
#endif
}
inline Tensor & Tensor::sparse_resize_and_clear_(IntArrayRef size, int64_t sparse_dim, int64_t dense_dim) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::SparseCPU:
            return SparseCPUType::sparse_resize_and_clear_(const_cast<Tensor&>(*this), size, sparse_dim, dense_dim);
            break;
        default:
            AT_ERROR("sparse_resize_and_clear_ not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::sparse_resize_and_clear_", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor &, Tensor &, IntArrayRef, int64_t, int64_t>(const_cast<Tensor&>(*this), size, sparse_dim, dense_dim);
#endif
}
inline Tensor Tensor::sparse_mask(const Tensor & mask) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::SparseCPU:
            return SparseCPUType::sparse_mask(const_cast<Tensor&>(*this), mask);
            break;
        default:
            AT_ERROR("sparse_mask not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::sparse_mask", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this, mask)))
        .callUnboxed<Tensor, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), mask);
#endif
}
inline Tensor Tensor::to_dense() const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::SparseCPU:
            return SparseCPUType::to_dense(const_cast<Tensor&>(*this));
            break;
        default:
            AT_ERROR("to_dense not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::to_dense", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}
inline int64_t Tensor::sparse_dim() const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::SparseCPU:
            return SparseCPUType::sparse_dim(const_cast<Tensor&>(*this));
            break;
        default:
            AT_ERROR("sparse_dim not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::sparse_dim", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<int64_t, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}
inline int64_t Tensor::_dimI() const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::SparseCPU:
            return SparseCPUType::_dimI(const_cast<Tensor&>(*this));
            break;
        default:
            AT_ERROR("_dimI not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::_dimI", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<int64_t, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}
inline int64_t Tensor::dense_dim() const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::SparseCPU:
            return SparseCPUType::dense_dim(const_cast<Tensor&>(*this));
            break;
        default:
            AT_ERROR("dense_dim not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::dense_dim", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<int64_t, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}
inline int64_t Tensor::_dimV() const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::SparseCPU:
            return SparseCPUType::_dimV(const_cast<Tensor&>(*this));
            break;
        default:
            AT_ERROR("_dimV not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::_dimV", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<int64_t, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}
inline int64_t Tensor::_nnz() const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::SparseCPU:
            return SparseCPUType::_nnz(const_cast<Tensor&>(*this));
            break;
        default:
            AT_ERROR("_nnz not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::_nnz", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<int64_t, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}
inline Tensor Tensor::coalesce() const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::SparseCPU:
            return SparseCPUType::coalesce(const_cast<Tensor&>(*this));
            break;
        default:
            AT_ERROR("coalesce not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::coalesce", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}
inline bool Tensor::is_coalesced() const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::SparseCPU:
            return SparseCPUType::is_coalesced(const_cast<Tensor&>(*this));
            break;
        default:
            AT_ERROR("is_coalesced not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::is_coalesced", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<bool, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}
inline Tensor Tensor::_indices() const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::SparseCPU:
            return SparseCPUType::_indices(const_cast<Tensor&>(*this));
            break;
        default:
            AT_ERROR("_indices not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::_indices", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}
inline Tensor Tensor::_values() const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::SparseCPU:
            return SparseCPUType::_values(const_cast<Tensor&>(*this));
            break;
        default:
            AT_ERROR("_values not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::_values", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}
inline Tensor & Tensor::_coalesced_(bool coalesced) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::SparseCPU:
            return SparseCPUType::_coalesced_(const_cast<Tensor&>(*this), coalesced);
            break;
        default:
            AT_ERROR("_coalesced_ not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::_coalesced_", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor &, Tensor &, bool>(const_cast<Tensor&>(*this), coalesced);
#endif
}
inline Tensor Tensor::indices() const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::SparseCPU:
            return SparseCPUType::indices(const_cast<Tensor&>(*this));
            break;
        default:
            AT_ERROR("indices not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::indices", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}
inline Tensor Tensor::values() const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::SparseCPU:
            return SparseCPUType::values(const_cast<Tensor&>(*this));
            break;
        default:
            AT_ERROR("values not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::values", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}
inline int64_t Tensor::numel() const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::numel(const_cast<Tensor&>(*this));
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::numel", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<int64_t, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}
inline std::vector<Tensor> Tensor::unbind(int64_t dim) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::unbind(const_cast<Tensor&>(*this), dim);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::unbind", "int"}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<std::vector<Tensor>, const Tensor &, int64_t>(const_cast<Tensor&>(*this), dim);
#endif
}
#ifdef BUILD_NAMEDTENSOR
inline std::vector<Tensor> Tensor::unbind(Dimname dim) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::unbind(const_cast<Tensor&>(*this), dim);
#else
    static auto table = globalATenDispatch().getOpTable("aten::unbind.Dimname(Tensor(a) self, Dimname dim) -> Tensor(a)[]");
    return table->getOp<std::vector<Tensor> (const Tensor &, Dimname)>(at::detail::multi_dispatch_tensor_type_set(*this))(const_cast<Tensor&>(*this), dim);
#endif
}
#endif
inline Tensor Tensor::to_sparse(int64_t sparse_dim) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::to_sparse(const_cast<Tensor&>(*this), sparse_dim);
            break;
        default:
            AT_ERROR("to_sparse not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::to_sparse", "sparse_dim"}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor, const Tensor &, int64_t>(const_cast<Tensor&>(*this), sparse_dim);
#endif
}
inline Tensor Tensor::to_sparse() const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::to_sparse(const_cast<Tensor&>(*this));
            break;
        default:
            AT_ERROR("to_sparse not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::to_sparse", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}
inline Tensor Tensor::to_mkldnn() const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::to_mkldnn(const_cast<Tensor&>(*this));
            break;
        default:
            AT_ERROR("to_mkldnn not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::to_mkldnn", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}
inline Tensor Tensor::dequantize() const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::QuantizedCPU:
            return QuantizedCPUType::dequantize(const_cast<Tensor&>(*this));
            break;
        default:
            AT_ERROR("dequantize not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::dequantize", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}
inline double Tensor::q_scale() const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::QuantizedCPU:
            return QuantizedCPUType::q_scale(const_cast<Tensor&>(*this));
            break;
        default:
            AT_ERROR("q_scale not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::q_scale", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<double, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}
inline int64_t Tensor::q_zero_point() const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::QuantizedCPU:
            return QuantizedCPUType::q_zero_point(const_cast<Tensor&>(*this));
            break;
        default:
            AT_ERROR("q_zero_point not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::q_zero_point", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<int64_t, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}
inline Tensor Tensor::q_per_channel_scales() const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::QuantizedCPU:
            return QuantizedCPUType::q_per_channel_scales(const_cast<Tensor&>(*this));
            break;
        default:
            AT_ERROR("q_per_channel_scales not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::q_per_channel_scales", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}
inline Tensor Tensor::q_per_channel_zero_points() const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::QuantizedCPU:
            return QuantizedCPUType::q_per_channel_zero_points(const_cast<Tensor&>(*this));
            break;
        default:
            AT_ERROR("q_per_channel_zero_points not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::q_per_channel_zero_points", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}
inline Tensor Tensor::int_repr() const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::QuantizedCPU:
            return QuantizedCPUType::int_repr(const_cast<Tensor&>(*this));
            break;
        default:
            AT_ERROR("int_repr not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::int_repr", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}
inline QScheme Tensor::qscheme() const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::QuantizedCPU:
            return QuantizedCPUType::qscheme(const_cast<Tensor&>(*this));
            break;
        default:
            AT_ERROR("qscheme not implemented for ", at::toString(type_set()));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::qscheme(Tensor self) -> QScheme");
    return table->getOp<QScheme (const Tensor &)>(at::detail::multi_dispatch_tensor_type_set(*this))(const_cast<Tensor&>(*this));
#endif
}
inline Tensor Tensor::to(const TensorOptions & options, bool non_blocking, bool copy) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::to(const_cast<Tensor&>(*this), options, non_blocking, copy);
#else
    static auto table = globalATenDispatch().getOpTable("aten::to.dtype_layout(Tensor self, *, ScalarType dtype, Layout layout, Device device, bool pin_memory=False, bool non_blocking=False, bool copy=False) -> Tensor");
    return table->getOp<Tensor (const Tensor &, const TensorOptions &, bool, bool)>(at::detail::multi_dispatch_tensor_type_set(*this, options))(const_cast<Tensor&>(*this), options, non_blocking, copy);
#endif
}
inline Tensor Tensor::to(Device device, ScalarType dtype, bool non_blocking, bool copy) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::to(const_cast<Tensor&>(*this), device, dtype, non_blocking, copy);
#else
    static auto table = globalATenDispatch().getOpTable("aten::to.device(Tensor self, Device device, ScalarType dtype, bool non_blocking=False, bool copy=False) -> Tensor");
    return table->getOp<Tensor (const Tensor &, Device, ScalarType, bool, bool)>(at::detail::multi_dispatch_tensor_type_set(*this))(const_cast<Tensor&>(*this), device, dtype, non_blocking, copy);
#endif
}
inline Tensor Tensor::to(ScalarType dtype, bool non_blocking, bool copy) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::to(const_cast<Tensor&>(*this), dtype, non_blocking, copy);
#else
    static auto table = globalATenDispatch().getOpTable("aten::to.dtype(Tensor self, ScalarType dtype, bool non_blocking=False, bool copy=False) -> Tensor");
    return table->getOp<Tensor (const Tensor &, ScalarType, bool, bool)>(at::detail::multi_dispatch_tensor_type_set(*this))(const_cast<Tensor&>(*this), dtype, non_blocking, copy);
#endif
}
inline Tensor Tensor::to(const Tensor & other, bool non_blocking, bool copy) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::to(const_cast<Tensor&>(*this), other, non_blocking, copy);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::to", "other"}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this, other)))
        .callUnboxed<Tensor, const Tensor &, const Tensor &, bool, bool>(const_cast<Tensor&>(*this), other, non_blocking, copy);
#endif
}
inline Scalar Tensor::item() const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::item(const_cast<Tensor&>(*this));
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::item", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Scalar, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}
inline Tensor & Tensor::set_(Storage source) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::set_(const_cast<Tensor&>(*this), source);
            break;
        default:
            AT_ERROR("set_ not implemented for ", at::toString(type_set()));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::set_.source_Storage(Tensor(a!) self, Storage source) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, Storage)>(at::detail::multi_dispatch_tensor_type_set(*this))(const_cast<Tensor&>(*this), source);
#endif
}
inline Tensor & Tensor::set_(Storage source, int64_t storage_offset, IntArrayRef size, IntArrayRef stride) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::set_(const_cast<Tensor&>(*this), source, storage_offset, size, stride);
            break;
        case Backend::QuantizedCPU:
            return QuantizedCPUType::set_(const_cast<Tensor&>(*this), source, storage_offset, size, stride);
            break;
        default:
            AT_ERROR("set_ not implemented for ", at::toString(type_set()));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::set_.source_Storage_storage_offset(Tensor(a!) self, Storage source, int storage_offset, int[] size, int[] stride=[]) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, Storage, int64_t, IntArrayRef, IntArrayRef)>(at::detail::multi_dispatch_tensor_type_set(*this))(const_cast<Tensor&>(*this), source, storage_offset, size, stride);
#endif
}
inline Tensor & Tensor::set_(const Tensor & source) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::set_(const_cast<Tensor&>(*this), source);
            break;
        default:
            AT_ERROR("set_ not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::set_", "source_Tensor"}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this, source)))
        .callUnboxed<Tensor &, Tensor &, const Tensor &>(const_cast<Tensor&>(*this), source);
#endif
}
inline Tensor & Tensor::set_() const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::set_(const_cast<Tensor&>(*this));
            break;
        default:
            AT_ERROR("set_ not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::set_", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor &, Tensor &>(const_cast<Tensor&>(*this));
#endif
}
inline Tensor & Tensor::set_quantizer_(ConstQuantizerPtr quantizer) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::QuantizedCPU:
            return QuantizedCPUType::set_quantizer_(const_cast<Tensor&>(*this), quantizer);
            break;
        default:
            AT_ERROR("set_quantizer_ not implemented for ", at::toString(type_set()));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::set_quantizer_(Tensor(a!) self, ConstQuantizerPtr quantizer) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, ConstQuantizerPtr)>(at::detail::multi_dispatch_tensor_type_set(*this))(const_cast<Tensor&>(*this), quantizer);
#endif
}
inline bool Tensor::is_set_to(const Tensor & tensor) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::is_set_to(const_cast<Tensor&>(*this), tensor);
            break;
        default:
            AT_ERROR("is_set_to not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::is_set_to", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this, tensor)))
        .callUnboxed<bool, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), tensor);
#endif
}
inline Tensor & Tensor::masked_fill_(const Tensor & mask, Scalar value) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::masked_fill_(const_cast<Tensor&>(*this), mask, value);
            break;
        default:
            AT_ERROR("masked_fill_ not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::masked_fill_", "Scalar"}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this, mask)))
        .callUnboxed<Tensor &, Tensor &, const Tensor &, Scalar>(const_cast<Tensor&>(*this), mask, value);
#endif
}
inline Tensor Tensor::masked_fill(const Tensor & mask, Scalar value) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::masked_fill(const_cast<Tensor&>(*this), mask, value);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::masked_fill", "Scalar"}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this, mask)))
        .callUnboxed<Tensor, const Tensor &, const Tensor &, Scalar>(const_cast<Tensor&>(*this), mask, value);
#endif
}
inline Tensor & Tensor::masked_fill_(const Tensor & mask, const Tensor & value) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::masked_fill_(const_cast<Tensor&>(*this), mask, value);
            break;
        default:
            AT_ERROR("masked_fill_ not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::masked_fill_", "Tensor"}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this, mask, value)))
        .callUnboxed<Tensor &, Tensor &, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), mask, value);
#endif
}
inline Tensor Tensor::masked_fill(const Tensor & mask, const Tensor & value) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::masked_fill(const_cast<Tensor&>(*this), mask, value);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::masked_fill", "Tensor"}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this, mask, value)))
        .callUnboxed<Tensor, const Tensor &, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), mask, value);
#endif
}
inline Tensor & Tensor::masked_scatter_(const Tensor & mask, const Tensor & source) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::masked_scatter_(const_cast<Tensor&>(*this), mask, source);
            break;
        default:
            AT_ERROR("masked_scatter_ not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::masked_scatter_", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this, mask, source)))
        .callUnboxed<Tensor &, Tensor &, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), mask, source);
#endif
}
inline Tensor Tensor::masked_scatter(const Tensor & mask, const Tensor & source) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::masked_scatter(const_cast<Tensor&>(*this), mask, source);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::masked_scatter", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this, mask, source)))
        .callUnboxed<Tensor, const Tensor &, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), mask, source);
#endif
}
inline Tensor Tensor::view(IntArrayRef size) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::view(const_cast<Tensor&>(*this), size);
            break;
        case Backend::QuantizedCPU:
            return QuantizedCPUType::view(const_cast<Tensor&>(*this), size);
            break;
        default:
            AT_ERROR("view not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::view", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor, const Tensor &, IntArrayRef>(const_cast<Tensor&>(*this), size);
#endif
}
inline Tensor & Tensor::put_(const Tensor & index, const Tensor & source, bool accumulate) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::put_(const_cast<Tensor&>(*this), index, source, accumulate);
            break;
        default:
            AT_ERROR("put_ not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::put_", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this, index, source)))
        .callUnboxed<Tensor &, Tensor &, const Tensor &, const Tensor &, bool>(const_cast<Tensor&>(*this), index, source, accumulate);
#endif
}
inline Tensor & Tensor::index_add_(int64_t dim, const Tensor & index, const Tensor & source) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::index_add_(const_cast<Tensor&>(*this), dim, index, source);
            break;
        default:
            AT_ERROR("index_add_ not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::index_add_", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this, index, source)))
        .callUnboxed<Tensor &, Tensor &, int64_t, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), dim, index, source);
#endif
}
inline Tensor Tensor::index_add(int64_t dim, const Tensor & index, const Tensor & source) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::index_add(const_cast<Tensor&>(*this), dim, index, source);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::index_add", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this, index, source)))
        .callUnboxed<Tensor, const Tensor &, int64_t, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), dim, index, source);
#endif
}
inline Tensor & Tensor::index_fill_(int64_t dim, const Tensor & index, Scalar value) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::index_fill_(const_cast<Tensor&>(*this), dim, index, value);
            break;
        default:
            AT_ERROR("index_fill_ not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::index_fill_", "Scalar"}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this, index)))
        .callUnboxed<Tensor &, Tensor &, int64_t, const Tensor &, Scalar>(const_cast<Tensor&>(*this), dim, index, value);
#endif
}
inline Tensor Tensor::index_fill(int64_t dim, const Tensor & index, Scalar value) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::index_fill(const_cast<Tensor&>(*this), dim, index, value);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::index_fill", "Scalar"}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this, index)))
        .callUnboxed<Tensor, const Tensor &, int64_t, const Tensor &, Scalar>(const_cast<Tensor&>(*this), dim, index, value);
#endif
}
inline Tensor & Tensor::index_fill_(int64_t dim, const Tensor & index, const Tensor & value) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::index_fill_(const_cast<Tensor&>(*this), dim, index, value);
            break;
        default:
            AT_ERROR("index_fill_ not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::index_fill_", "Tensor"}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this, index, value)))
        .callUnboxed<Tensor &, Tensor &, int64_t, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), dim, index, value);
#endif
}
inline Tensor Tensor::index_fill(int64_t dim, const Tensor & index, const Tensor & value) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::index_fill(const_cast<Tensor&>(*this), dim, index, value);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::index_fill", "Tensor"}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this, index, value)))
        .callUnboxed<Tensor, const Tensor &, int64_t, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), dim, index, value);
#endif
}
inline Tensor & Tensor::scatter_(int64_t dim, const Tensor & index, const Tensor & src) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::scatter_(const_cast<Tensor&>(*this), dim, index, src);
            break;
        default:
            AT_ERROR("scatter_ not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::scatter_", "src"}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this, index, src)))
        .callUnboxed<Tensor &, Tensor &, int64_t, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), dim, index, src);
#endif
}
inline Tensor Tensor::scatter(int64_t dim, const Tensor & index, const Tensor & src) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::scatter(const_cast<Tensor&>(*this), dim, index, src);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::scatter", "src"}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this, index, src)))
        .callUnboxed<Tensor, const Tensor &, int64_t, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), dim, index, src);
#endif
}
inline Tensor & Tensor::scatter_(int64_t dim, const Tensor & index, Scalar value) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::scatter_(const_cast<Tensor&>(*this), dim, index, value);
            break;
        default:
            AT_ERROR("scatter_ not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::scatter_", "value"}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this, index)))
        .callUnboxed<Tensor &, Tensor &, int64_t, const Tensor &, Scalar>(const_cast<Tensor&>(*this), dim, index, value);
#endif
}
inline Tensor Tensor::scatter(int64_t dim, const Tensor & index, Scalar value) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::scatter(const_cast<Tensor&>(*this), dim, index, value);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::scatter", "value"}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this, index)))
        .callUnboxed<Tensor, const Tensor &, int64_t, const Tensor &, Scalar>(const_cast<Tensor&>(*this), dim, index, value);
#endif
}
inline Tensor & Tensor::scatter_add_(int64_t dim, const Tensor & index, const Tensor & src) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::scatter_add_(const_cast<Tensor&>(*this), dim, index, src);
            break;
        default:
            AT_ERROR("scatter_add_ not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::scatter_add_", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this, index, src)))
        .callUnboxed<Tensor &, Tensor &, int64_t, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), dim, index, src);
#endif
}
inline Tensor Tensor::scatter_add(int64_t dim, const Tensor & index, const Tensor & src) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::scatter_add(const_cast<Tensor&>(*this), dim, index, src);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::scatter_add", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this, index, src)))
        .callUnboxed<Tensor, const Tensor &, int64_t, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), dim, index, src);
#endif
}
inline Tensor & Tensor::lt_(Scalar other) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::lt_(const_cast<Tensor&>(*this), other);
            break;
        default:
            AT_ERROR("lt_ not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::lt_", "Scalar"}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor &, Tensor &, Scalar>(const_cast<Tensor&>(*this), other);
#endif
}
inline Tensor & Tensor::lt_(const Tensor & other) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::lt_(const_cast<Tensor&>(*this), other);
            break;
        default:
            AT_ERROR("lt_ not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::lt_", "Tensor"}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this, other)))
        .callUnboxed<Tensor &, Tensor &, const Tensor &>(const_cast<Tensor&>(*this), other);
#endif
}
inline Tensor & Tensor::gt_(Scalar other) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::gt_(const_cast<Tensor&>(*this), other);
            break;
        default:
            AT_ERROR("gt_ not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::gt_", "Scalar"}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor &, Tensor &, Scalar>(const_cast<Tensor&>(*this), other);
#endif
}
inline Tensor & Tensor::gt_(const Tensor & other) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::gt_(const_cast<Tensor&>(*this), other);
            break;
        default:
            AT_ERROR("gt_ not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::gt_", "Tensor"}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this, other)))
        .callUnboxed<Tensor &, Tensor &, const Tensor &>(const_cast<Tensor&>(*this), other);
#endif
}
inline Tensor & Tensor::le_(Scalar other) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::le_(const_cast<Tensor&>(*this), other);
            break;
        default:
            AT_ERROR("le_ not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::le_", "Scalar"}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor &, Tensor &, Scalar>(const_cast<Tensor&>(*this), other);
#endif
}
inline Tensor & Tensor::le_(const Tensor & other) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::le_(const_cast<Tensor&>(*this), other);
            break;
        default:
            AT_ERROR("le_ not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::le_", "Tensor"}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this, other)))
        .callUnboxed<Tensor &, Tensor &, const Tensor &>(const_cast<Tensor&>(*this), other);
#endif
}
inline Tensor & Tensor::ge_(Scalar other) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::ge_(const_cast<Tensor&>(*this), other);
            break;
        default:
            AT_ERROR("ge_ not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::ge_", "Scalar"}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor &, Tensor &, Scalar>(const_cast<Tensor&>(*this), other);
#endif
}
inline Tensor & Tensor::ge_(const Tensor & other) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::ge_(const_cast<Tensor&>(*this), other);
            break;
        default:
            AT_ERROR("ge_ not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::ge_", "Tensor"}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this, other)))
        .callUnboxed<Tensor &, Tensor &, const Tensor &>(const_cast<Tensor&>(*this), other);
#endif
}
inline Tensor & Tensor::eq_(Scalar other) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::eq_(const_cast<Tensor&>(*this), other);
            break;
        default:
            AT_ERROR("eq_ not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::eq_", "Scalar"}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor &, Tensor &, Scalar>(const_cast<Tensor&>(*this), other);
#endif
}
inline Tensor & Tensor::eq_(const Tensor & other) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::eq_(const_cast<Tensor&>(*this), other);
            break;
        default:
            AT_ERROR("eq_ not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::eq_", "Tensor"}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this, other)))
        .callUnboxed<Tensor &, Tensor &, const Tensor &>(const_cast<Tensor&>(*this), other);
#endif
}
inline Tensor & Tensor::ne_(Scalar other) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::ne_(const_cast<Tensor&>(*this), other);
            break;
        default:
            AT_ERROR("ne_ not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::ne_", "Scalar"}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor &, Tensor &, Scalar>(const_cast<Tensor&>(*this), other);
#endif
}
inline Tensor & Tensor::ne_(const Tensor & other) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::ne_(const_cast<Tensor&>(*this), other);
            break;
        default:
            AT_ERROR("ne_ not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::ne_", "Tensor"}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this, other)))
        .callUnboxed<Tensor &, Tensor &, const Tensor &>(const_cast<Tensor&>(*this), other);
#endif
}
inline Tensor Tensor::__and__(Scalar other) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::__and__(const_cast<Tensor&>(*this), other);
            break;
        default:
            AT_ERROR("__and__ not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::__and__", "Scalar"}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor, const Tensor &, Scalar>(const_cast<Tensor&>(*this), other);
#endif
}
inline Tensor Tensor::__and__(const Tensor & other) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::__and__(const_cast<Tensor&>(*this), other);
            break;
        default:
            AT_ERROR("__and__ not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::__and__", "Tensor"}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this, other)))
        .callUnboxed<Tensor, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), other);
#endif
}
inline Tensor & Tensor::__iand__(Scalar other) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::__iand__(const_cast<Tensor&>(*this), other);
            break;
        default:
            AT_ERROR("__iand__ not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::__iand__", "Scalar"}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor &, Tensor &, Scalar>(const_cast<Tensor&>(*this), other);
#endif
}
inline Tensor & Tensor::__iand__(const Tensor & other) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::__iand__(const_cast<Tensor&>(*this), other);
            break;
        default:
            AT_ERROR("__iand__ not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::__iand__", "Tensor"}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this, other)))
        .callUnboxed<Tensor &, Tensor &, const Tensor &>(const_cast<Tensor&>(*this), other);
#endif
}
inline Tensor Tensor::__or__(Scalar other) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::__or__(const_cast<Tensor&>(*this), other);
            break;
        default:
            AT_ERROR("__or__ not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::__or__", "Scalar"}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor, const Tensor &, Scalar>(const_cast<Tensor&>(*this), other);
#endif
}
inline Tensor Tensor::__or__(const Tensor & other) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::__or__(const_cast<Tensor&>(*this), other);
            break;
        default:
            AT_ERROR("__or__ not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::__or__", "Tensor"}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this, other)))
        .callUnboxed<Tensor, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), other);
#endif
}
inline Tensor & Tensor::__ior__(Scalar other) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::__ior__(const_cast<Tensor&>(*this), other);
            break;
        default:
            AT_ERROR("__ior__ not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::__ior__", "Scalar"}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor &, Tensor &, Scalar>(const_cast<Tensor&>(*this), other);
#endif
}
inline Tensor & Tensor::__ior__(const Tensor & other) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::__ior__(const_cast<Tensor&>(*this), other);
            break;
        default:
            AT_ERROR("__ior__ not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::__ior__", "Tensor"}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this, other)))
        .callUnboxed<Tensor &, Tensor &, const Tensor &>(const_cast<Tensor&>(*this), other);
#endif
}
inline Tensor Tensor::__xor__(Scalar other) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::__xor__(const_cast<Tensor&>(*this), other);
            break;
        default:
            AT_ERROR("__xor__ not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::__xor__", "Scalar"}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor, const Tensor &, Scalar>(const_cast<Tensor&>(*this), other);
#endif
}
inline Tensor Tensor::__xor__(const Tensor & other) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::__xor__(const_cast<Tensor&>(*this), other);
            break;
        default:
            AT_ERROR("__xor__ not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::__xor__", "Tensor"}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this, other)))
        .callUnboxed<Tensor, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), other);
#endif
}
inline Tensor & Tensor::__ixor__(Scalar other) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::__ixor__(const_cast<Tensor&>(*this), other);
            break;
        default:
            AT_ERROR("__ixor__ not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::__ixor__", "Scalar"}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor &, Tensor &, Scalar>(const_cast<Tensor&>(*this), other);
#endif
}
inline Tensor & Tensor::__ixor__(const Tensor & other) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::__ixor__(const_cast<Tensor&>(*this), other);
            break;
        default:
            AT_ERROR("__ixor__ not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::__ixor__", "Tensor"}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this, other)))
        .callUnboxed<Tensor &, Tensor &, const Tensor &>(const_cast<Tensor&>(*this), other);
#endif
}
inline Tensor Tensor::__lshift__(Scalar other) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::__lshift__(const_cast<Tensor&>(*this), other);
            break;
        default:
            AT_ERROR("__lshift__ not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::__lshift__", "Scalar"}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor, const Tensor &, Scalar>(const_cast<Tensor&>(*this), other);
#endif
}
inline Tensor Tensor::__lshift__(const Tensor & other) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::__lshift__(const_cast<Tensor&>(*this), other);
            break;
        default:
            AT_ERROR("__lshift__ not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::__lshift__", "Tensor"}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this, other)))
        .callUnboxed<Tensor, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), other);
#endif
}
inline Tensor & Tensor::__ilshift__(Scalar other) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::__ilshift__(const_cast<Tensor&>(*this), other);
            break;
        default:
            AT_ERROR("__ilshift__ not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::__ilshift__", "Scalar"}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor &, Tensor &, Scalar>(const_cast<Tensor&>(*this), other);
#endif
}
inline Tensor & Tensor::__ilshift__(const Tensor & other) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::__ilshift__(const_cast<Tensor&>(*this), other);
            break;
        default:
            AT_ERROR("__ilshift__ not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::__ilshift__", "Tensor"}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this, other)))
        .callUnboxed<Tensor &, Tensor &, const Tensor &>(const_cast<Tensor&>(*this), other);
#endif
}
inline Tensor Tensor::__rshift__(Scalar other) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::__rshift__(const_cast<Tensor&>(*this), other);
            break;
        default:
            AT_ERROR("__rshift__ not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::__rshift__", "Scalar"}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor, const Tensor &, Scalar>(const_cast<Tensor&>(*this), other);
#endif
}
inline Tensor Tensor::__rshift__(const Tensor & other) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::__rshift__(const_cast<Tensor&>(*this), other);
            break;
        default:
            AT_ERROR("__rshift__ not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::__rshift__", "Tensor"}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this, other)))
        .callUnboxed<Tensor, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), other);
#endif
}
inline Tensor & Tensor::__irshift__(Scalar other) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::__irshift__(const_cast<Tensor&>(*this), other);
            break;
        default:
            AT_ERROR("__irshift__ not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::__irshift__", "Scalar"}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor &, Tensor &, Scalar>(const_cast<Tensor&>(*this), other);
#endif
}
inline Tensor & Tensor::__irshift__(const Tensor & other) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::__irshift__(const_cast<Tensor&>(*this), other);
            break;
        default:
            AT_ERROR("__irshift__ not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::__irshift__", "Tensor"}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this, other)))
        .callUnboxed<Tensor &, Tensor &, const Tensor &>(const_cast<Tensor&>(*this), other);
#endif
}
inline Tensor & Tensor::lgamma_() const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::lgamma_(const_cast<Tensor&>(*this));
            break;
        default:
            AT_ERROR("lgamma_ not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::lgamma_", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor &, Tensor &>(const_cast<Tensor&>(*this));
#endif
}
inline Tensor & Tensor::atan2_(const Tensor & other) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::atan2_(const_cast<Tensor&>(*this), other);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::atan2_", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this, other)))
        .callUnboxed<Tensor &, Tensor &, const Tensor &>(const_cast<Tensor&>(*this), other);
#endif
}
inline Tensor & Tensor::tril_(int64_t diagonal) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::tril_(const_cast<Tensor&>(*this), diagonal);
            break;
        default:
            AT_ERROR("tril_ not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::tril_", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor &, Tensor &, int64_t>(const_cast<Tensor&>(*this), diagonal);
#endif
}
inline Tensor & Tensor::triu_(int64_t diagonal) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::triu_(const_cast<Tensor&>(*this), diagonal);
            break;
        default:
            AT_ERROR("triu_ not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::triu_", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor &, Tensor &, int64_t>(const_cast<Tensor&>(*this), diagonal);
#endif
}
inline Tensor & Tensor::digamma_() const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::digamma_(const_cast<Tensor&>(*this));
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::digamma_", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor &, Tensor &>(const_cast<Tensor&>(*this));
#endif
}
inline Tensor & Tensor::polygamma_(int64_t n) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::polygamma_(const_cast<Tensor&>(*this), n);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::polygamma_", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor &, Tensor &, int64_t>(const_cast<Tensor&>(*this), n);
#endif
}
inline Tensor & Tensor::renorm_(Scalar p, int64_t dim, Scalar maxnorm) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::renorm_(const_cast<Tensor&>(*this), p, dim, maxnorm);
            break;
        default:
            AT_ERROR("renorm_ not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::renorm_", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor &, Tensor &, Scalar, int64_t, Scalar>(const_cast<Tensor&>(*this), p, dim, maxnorm);
#endif
}
inline Tensor & Tensor::pow_(Scalar exponent) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::pow_(const_cast<Tensor&>(*this), exponent);
            break;
        default:
            AT_ERROR("pow_ not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::pow_", "Scalar"}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor &, Tensor &, Scalar>(const_cast<Tensor&>(*this), exponent);
#endif
}
inline Tensor & Tensor::pow_(const Tensor & exponent) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::pow_(const_cast<Tensor&>(*this), exponent);
            break;
        default:
            AT_ERROR("pow_ not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::pow_", "Tensor"}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this, exponent)))
        .callUnboxed<Tensor &, Tensor &, const Tensor &>(const_cast<Tensor&>(*this), exponent);
#endif
}
inline Tensor & Tensor::lerp_(const Tensor & end, Scalar weight) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::lerp_(const_cast<Tensor&>(*this), end, weight);
            break;
        default:
            AT_ERROR("lerp_ not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::lerp_", "Scalar"}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this, end)))
        .callUnboxed<Tensor &, Tensor &, const Tensor &, Scalar>(const_cast<Tensor&>(*this), end, weight);
#endif
}
inline Tensor & Tensor::lerp_(const Tensor & end, const Tensor & weight) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::lerp_(const_cast<Tensor&>(*this), end, weight);
            break;
        default:
            AT_ERROR("lerp_ not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::lerp_", "Tensor"}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this, end, weight)))
        .callUnboxed<Tensor &, Tensor &, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), end, weight);
#endif
}
inline Tensor & Tensor::fmod_(Scalar other) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::fmod_(const_cast<Tensor&>(*this), other);
            break;
        default:
            AT_ERROR("fmod_ not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::fmod_", "Scalar"}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor &, Tensor &, Scalar>(const_cast<Tensor&>(*this), other);
#endif
}
inline Tensor & Tensor::fmod_(const Tensor & other) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::fmod_(const_cast<Tensor&>(*this), other);
            break;
        default:
            AT_ERROR("fmod_ not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::fmod_", "Tensor"}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this, other)))
        .callUnboxed<Tensor &, Tensor &, const Tensor &>(const_cast<Tensor&>(*this), other);
#endif
}
inline Tensor & Tensor::remainder_(Scalar other) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::remainder_(const_cast<Tensor&>(*this), other);
            break;
        default:
            AT_ERROR("remainder_ not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::remainder_", "Scalar"}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor &, Tensor &, Scalar>(const_cast<Tensor&>(*this), other);
#endif
}
inline Tensor & Tensor::remainder_(const Tensor & other) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::remainder_(const_cast<Tensor&>(*this), other);
            break;
        default:
            AT_ERROR("remainder_ not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::remainder_", "Tensor"}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this, other)))
        .callUnboxed<Tensor &, Tensor &, const Tensor &>(const_cast<Tensor&>(*this), other);
#endif
}
inline Tensor & Tensor::addbmm_(const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::addbmm_(const_cast<Tensor&>(*this), batch1, batch2, beta, alpha);
            break;
        default:
            AT_ERROR("addbmm_ not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::addbmm_", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this, batch1, batch2)))
        .callUnboxed<Tensor &, Tensor &, const Tensor &, const Tensor &, Scalar, Scalar>(const_cast<Tensor&>(*this), batch1, batch2, beta, alpha);
#endif
}
inline Tensor Tensor::addbmm(const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::addbmm(const_cast<Tensor&>(*this), batch1, batch2, beta, alpha);
            break;
        default:
            AT_ERROR("addbmm not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::addbmm", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this, batch1, batch2)))
        .callUnboxed<Tensor, const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar>(const_cast<Tensor&>(*this), batch1, batch2, beta, alpha);
#endif
}
inline Tensor & Tensor::addcdiv_(const Tensor & tensor1, const Tensor & tensor2, Scalar value) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::addcdiv_(const_cast<Tensor&>(*this), tensor1, tensor2, value);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::addcdiv_", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this, tensor1, tensor2)))
        .callUnboxed<Tensor &, Tensor &, const Tensor &, const Tensor &, Scalar>(const_cast<Tensor&>(*this), tensor1, tensor2, value);
#endif
}
inline Tensor & Tensor::random_(int64_t from, int64_t to, Generator * generator) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::random_(const_cast<Tensor&>(*this), from, to, generator);
            break;
        default:
            AT_ERROR("random_ not implemented for ", at::toString(type_set()));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::random_.from(Tensor(a!) self, int from, int to, *, Generator? generator=None) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, int64_t, int64_t, Generator *)>(at::detail::multi_dispatch_tensor_type_set(*this))(const_cast<Tensor&>(*this), from, to, generator);
#endif
}
inline Tensor & Tensor::random_(int64_t to, Generator * generator) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::random_(const_cast<Tensor&>(*this), to, generator);
            break;
        default:
            AT_ERROR("random_ not implemented for ", at::toString(type_set()));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::random_.to(Tensor(a!) self, int to, *, Generator? generator=None) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, int64_t, Generator *)>(at::detail::multi_dispatch_tensor_type_set(*this))(const_cast<Tensor&>(*this), to, generator);
#endif
}
inline Tensor & Tensor::random_(Generator * generator) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::random_(const_cast<Tensor&>(*this), generator);
            break;
        default:
            AT_ERROR("random_ not implemented for ", at::toString(type_set()));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::random_(Tensor(a!) self, *, Generator? generator=None) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, Generator *)>(at::detail::multi_dispatch_tensor_type_set(*this))(const_cast<Tensor&>(*this), generator);
#endif
}
inline Tensor & Tensor::uniform_(double from, double to, Generator * generator) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::uniform_(const_cast<Tensor&>(*this), from, to, generator);
            break;
        default:
            AT_ERROR("uniform_ not implemented for ", at::toString(type_set()));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::uniform_(Tensor(a!) self, float from=0, float to=1, *, Generator? generator=None) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, double, double, Generator *)>(at::detail::multi_dispatch_tensor_type_set(*this))(const_cast<Tensor&>(*this), from, to, generator);
#endif
}
inline Tensor & Tensor::normal_(double mean, double std, Generator * generator) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::normal_(const_cast<Tensor&>(*this), mean, std, generator);
            break;
        default:
            AT_ERROR("normal_ not implemented for ", at::toString(type_set()));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::normal_(Tensor(a!) self, float mean=0, float std=1, *, Generator? generator=None) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, double, double, Generator *)>(at::detail::multi_dispatch_tensor_type_set(*this))(const_cast<Tensor&>(*this), mean, std, generator);
#endif
}
inline Tensor & Tensor::cauchy_(double median, double sigma, Generator * generator) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::cauchy_(const_cast<Tensor&>(*this), median, sigma, generator);
            break;
        default:
            AT_ERROR("cauchy_ not implemented for ", at::toString(type_set()));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::cauchy_(Tensor(a!) self, float median=0, float sigma=1, *, Generator? generator=None) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, double, double, Generator *)>(at::detail::multi_dispatch_tensor_type_set(*this))(const_cast<Tensor&>(*this), median, sigma, generator);
#endif
}
inline Tensor & Tensor::log_normal_(double mean, double std, Generator * generator) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::log_normal_(const_cast<Tensor&>(*this), mean, std, generator);
            break;
        default:
            AT_ERROR("log_normal_ not implemented for ", at::toString(type_set()));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::log_normal_(Tensor(a!) self, float mean=1, float std=2, *, Generator? generator=None) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, double, double, Generator *)>(at::detail::multi_dispatch_tensor_type_set(*this))(const_cast<Tensor&>(*this), mean, std, generator);
#endif
}
inline Tensor & Tensor::exponential_(double lambd, Generator * generator) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::exponential_(const_cast<Tensor&>(*this), lambd, generator);
            break;
        default:
            AT_ERROR("exponential_ not implemented for ", at::toString(type_set()));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::exponential_(Tensor(a!) self, float lambd=1, *, Generator? generator=None) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, double, Generator *)>(at::detail::multi_dispatch_tensor_type_set(*this))(const_cast<Tensor&>(*this), lambd, generator);
#endif
}
inline Tensor & Tensor::geometric_(double p, Generator * generator) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::geometric_(const_cast<Tensor&>(*this), p, generator);
            break;
        default:
            AT_ERROR("geometric_ not implemented for ", at::toString(type_set()));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::geometric_(Tensor(a!) self, float p, *, Generator? generator=None) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, double, Generator *)>(at::detail::multi_dispatch_tensor_type_set(*this))(const_cast<Tensor&>(*this), p, generator);
#endif
}
inline Tensor Tensor::diag(int64_t diagonal) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::diag(const_cast<Tensor&>(*this), diagonal);
            break;
        default:
            AT_ERROR("diag not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::diag", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor, const Tensor &, int64_t>(const_cast<Tensor&>(*this), diagonal);
#endif
}
inline Tensor Tensor::cross(const Tensor & other, c10::optional<int64_t> dim) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::cross(const_cast<Tensor&>(*this), other, dim);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::cross", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this, other)))
        .callUnboxed<Tensor, const Tensor &, const Tensor &, c10::optional<int64_t>>(const_cast<Tensor&>(*this), other, dim);
#endif
}
inline Tensor Tensor::triu(int64_t diagonal) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::triu(const_cast<Tensor&>(*this), diagonal);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::triu", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor, const Tensor &, int64_t>(const_cast<Tensor&>(*this), diagonal);
#endif
}
inline Tensor Tensor::tril(int64_t diagonal) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::tril(const_cast<Tensor&>(*this), diagonal);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::tril", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor, const Tensor &, int64_t>(const_cast<Tensor&>(*this), diagonal);
#endif
}
inline Tensor Tensor::trace() const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::trace(const_cast<Tensor&>(*this));
            break;
        default:
            AT_ERROR("trace not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::trace", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}
inline Tensor Tensor::ne(Scalar other) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::ne(const_cast<Tensor&>(*this), other);
            break;
        case Backend::QuantizedCPU:
            return QuantizedCPUType::ne(const_cast<Tensor&>(*this), other);
            break;
        default:
            AT_ERROR("ne not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::ne", "Scalar"}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor, const Tensor &, Scalar>(const_cast<Tensor&>(*this), other);
#endif
}
inline Tensor Tensor::ne(const Tensor & other) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::ne(const_cast<Tensor&>(*this), other);
            break;
        case Backend::QuantizedCPU:
            return QuantizedCPUType::ne(const_cast<Tensor&>(*this), other);
            break;
        default:
            AT_ERROR("ne not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::ne", "Tensor"}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this, other)))
        .callUnboxed<Tensor, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), other);
#endif
}
inline Tensor Tensor::eq(Scalar other) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::eq(const_cast<Tensor&>(*this), other);
            break;
        case Backend::QuantizedCPU:
            return QuantizedCPUType::eq(const_cast<Tensor&>(*this), other);
            break;
        default:
            AT_ERROR("eq not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::eq", "Scalar"}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor, const Tensor &, Scalar>(const_cast<Tensor&>(*this), other);
#endif
}
inline Tensor Tensor::eq(const Tensor & other) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::eq(const_cast<Tensor&>(*this), other);
            break;
        case Backend::QuantizedCPU:
            return QuantizedCPUType::eq(const_cast<Tensor&>(*this), other);
            break;
        default:
            AT_ERROR("eq not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::eq", "Tensor"}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this, other)))
        .callUnboxed<Tensor, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), other);
#endif
}
inline Tensor Tensor::ge(Scalar other) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::ge(const_cast<Tensor&>(*this), other);
            break;
        case Backend::QuantizedCPU:
            return QuantizedCPUType::ge(const_cast<Tensor&>(*this), other);
            break;
        default:
            AT_ERROR("ge not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::ge", "Scalar"}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor, const Tensor &, Scalar>(const_cast<Tensor&>(*this), other);
#endif
}
inline Tensor Tensor::ge(const Tensor & other) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::ge(const_cast<Tensor&>(*this), other);
            break;
        case Backend::QuantizedCPU:
            return QuantizedCPUType::ge(const_cast<Tensor&>(*this), other);
            break;
        default:
            AT_ERROR("ge not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::ge", "Tensor"}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this, other)))
        .callUnboxed<Tensor, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), other);
#endif
}
inline Tensor Tensor::le(Scalar other) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::le(const_cast<Tensor&>(*this), other);
            break;
        case Backend::QuantizedCPU:
            return QuantizedCPUType::le(const_cast<Tensor&>(*this), other);
            break;
        default:
            AT_ERROR("le not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::le", "Scalar"}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor, const Tensor &, Scalar>(const_cast<Tensor&>(*this), other);
#endif
}
inline Tensor Tensor::le(const Tensor & other) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::le(const_cast<Tensor&>(*this), other);
            break;
        case Backend::QuantizedCPU:
            return QuantizedCPUType::le(const_cast<Tensor&>(*this), other);
            break;
        default:
            AT_ERROR("le not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::le", "Tensor"}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this, other)))
        .callUnboxed<Tensor, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), other);
#endif
}
inline Tensor Tensor::gt(Scalar other) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::gt(const_cast<Tensor&>(*this), other);
            break;
        case Backend::QuantizedCPU:
            return QuantizedCPUType::gt(const_cast<Tensor&>(*this), other);
            break;
        default:
            AT_ERROR("gt not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::gt", "Scalar"}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor, const Tensor &, Scalar>(const_cast<Tensor&>(*this), other);
#endif
}
inline Tensor Tensor::gt(const Tensor & other) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::gt(const_cast<Tensor&>(*this), other);
            break;
        case Backend::QuantizedCPU:
            return QuantizedCPUType::gt(const_cast<Tensor&>(*this), other);
            break;
        default:
            AT_ERROR("gt not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::gt", "Tensor"}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this, other)))
        .callUnboxed<Tensor, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), other);
#endif
}
inline Tensor Tensor::lt(Scalar other) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::lt(const_cast<Tensor&>(*this), other);
            break;
        case Backend::QuantizedCPU:
            return QuantizedCPUType::lt(const_cast<Tensor&>(*this), other);
            break;
        default:
            AT_ERROR("lt not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::lt", "Scalar"}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor, const Tensor &, Scalar>(const_cast<Tensor&>(*this), other);
#endif
}
inline Tensor Tensor::lt(const Tensor & other) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::lt(const_cast<Tensor&>(*this), other);
            break;
        case Backend::QuantizedCPU:
            return QuantizedCPUType::lt(const_cast<Tensor&>(*this), other);
            break;
        default:
            AT_ERROR("lt not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::lt", "Tensor"}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this, other)))
        .callUnboxed<Tensor, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), other);
#endif
}
inline Tensor Tensor::take(const Tensor & index) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::take(const_cast<Tensor&>(*this), index);
            break;
        default:
            AT_ERROR("take not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::take", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this, index)))
        .callUnboxed<Tensor, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), index);
#endif
}
inline Tensor Tensor::index_select(int64_t dim, const Tensor & index) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::index_select(const_cast<Tensor&>(*this), dim, index);
            break;
        case Backend::SparseCPU:
            return SparseCPUType::index_select(const_cast<Tensor&>(*this), dim, index);
            break;
        default:
            AT_ERROR("index_select not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::index_select", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this, index)))
        .callUnboxed<Tensor, const Tensor &, int64_t, const Tensor &>(const_cast<Tensor&>(*this), dim, index);
#endif
}
inline Tensor Tensor::masked_select(const Tensor & mask) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::masked_select(const_cast<Tensor&>(*this), mask);
            break;
        default:
            AT_ERROR("masked_select not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::masked_select", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this, mask)))
        .callUnboxed<Tensor, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), mask);
#endif
}
inline Tensor Tensor::nonzero() const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::nonzero(const_cast<Tensor&>(*this));
            break;
        default:
            AT_ERROR("nonzero not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::nonzero", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}
inline std::vector<Tensor> Tensor::nonzero_numpy() const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::nonzero_numpy(const_cast<Tensor&>(*this));
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::nonzero_numpy", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<std::vector<Tensor>, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}
inline Tensor Tensor::gather(int64_t dim, const Tensor & index, bool sparse_grad) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::gather(const_cast<Tensor&>(*this), dim, index, sparse_grad);
            break;
        default:
            AT_ERROR("gather not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::gather", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this, index)))
        .callUnboxed<Tensor, const Tensor &, int64_t, const Tensor &, bool>(const_cast<Tensor&>(*this), dim, index, sparse_grad);
#endif
}
inline Tensor Tensor::addcmul(const Tensor & tensor1, const Tensor & tensor2, Scalar value) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::addcmul(const_cast<Tensor&>(*this), tensor1, tensor2, value);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::addcmul", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this, tensor1, tensor2)))
        .callUnboxed<Tensor, const Tensor &, const Tensor &, const Tensor &, Scalar>(const_cast<Tensor&>(*this), tensor1, tensor2, value);
#endif
}
inline Tensor & Tensor::addcmul_(const Tensor & tensor1, const Tensor & tensor2, Scalar value) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::addcmul_(const_cast<Tensor&>(*this), tensor1, tensor2, value);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::addcmul_", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this, tensor1, tensor2)))
        .callUnboxed<Tensor &, Tensor &, const Tensor &, const Tensor &, Scalar>(const_cast<Tensor&>(*this), tensor1, tensor2, value);
#endif
}
inline Tensor Tensor::addcdiv(const Tensor & tensor1, const Tensor & tensor2, Scalar value) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::addcdiv(const_cast<Tensor&>(*this), tensor1, tensor2, value);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::addcdiv", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this, tensor1, tensor2)))
        .callUnboxed<Tensor, const Tensor &, const Tensor &, const Tensor &, Scalar>(const_cast<Tensor&>(*this), tensor1, tensor2, value);
#endif
}
inline std::tuple<Tensor,Tensor> Tensor::lstsq(const Tensor & A) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::lstsq(const_cast<Tensor&>(*this), A);
            break;
        default:
            AT_ERROR("lstsq not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::lstsq", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this, A)))
        .callUnboxed<std::tuple<Tensor,Tensor>, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), A);
#endif
}
inline std::tuple<Tensor,Tensor> Tensor::triangular_solve(const Tensor & A, bool upper, bool transpose, bool unitriangular) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::triangular_solve(const_cast<Tensor&>(*this), A, upper, transpose, unitriangular);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::triangular_solve", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this, A)))
        .callUnboxed<std::tuple<Tensor,Tensor>, const Tensor &, const Tensor &, bool, bool, bool>(const_cast<Tensor&>(*this), A, upper, transpose, unitriangular);
#endif
}
inline std::tuple<Tensor,Tensor> Tensor::symeig(bool eigenvectors, bool upper) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::symeig(const_cast<Tensor&>(*this), eigenvectors, upper);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::symeig", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<std::tuple<Tensor,Tensor>, const Tensor &, bool, bool>(const_cast<Tensor&>(*this), eigenvectors, upper);
#endif
}
inline std::tuple<Tensor,Tensor> Tensor::eig(bool eigenvectors) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::eig(const_cast<Tensor&>(*this), eigenvectors);
            break;
        default:
            AT_ERROR("eig not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::eig", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<std::tuple<Tensor,Tensor>, const Tensor &, bool>(const_cast<Tensor&>(*this), eigenvectors);
#endif
}
inline std::tuple<Tensor,Tensor,Tensor> Tensor::svd(bool some, bool compute_uv) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::svd(const_cast<Tensor&>(*this), some, compute_uv);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::svd", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<std::tuple<Tensor,Tensor,Tensor>, const Tensor &, bool, bool>(const_cast<Tensor&>(*this), some, compute_uv);
#endif
}
inline Tensor Tensor::cholesky(bool upper) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::cholesky(const_cast<Tensor&>(*this), upper);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::cholesky", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor, const Tensor &, bool>(const_cast<Tensor&>(*this), upper);
#endif
}
inline Tensor Tensor::cholesky_solve(const Tensor & input2, bool upper) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::cholesky_solve(const_cast<Tensor&>(*this), input2, upper);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::cholesky_solve", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this, input2)))
        .callUnboxed<Tensor, const Tensor &, const Tensor &, bool>(const_cast<Tensor&>(*this), input2, upper);
#endif
}
inline std::tuple<Tensor,Tensor> Tensor::solve(const Tensor & A) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::solve(const_cast<Tensor&>(*this), A);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::solve", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this, A)))
        .callUnboxed<std::tuple<Tensor,Tensor>, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), A);
#endif
}
inline Tensor Tensor::cholesky_inverse(bool upper) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::cholesky_inverse(const_cast<Tensor&>(*this), upper);
            break;
        default:
            AT_ERROR("cholesky_inverse not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::cholesky_inverse", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor, const Tensor &, bool>(const_cast<Tensor&>(*this), upper);
#endif
}
inline std::tuple<Tensor,Tensor> Tensor::qr(bool some) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::qr(const_cast<Tensor&>(*this), some);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::qr", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<std::tuple<Tensor,Tensor>, const Tensor &, bool>(const_cast<Tensor&>(*this), some);
#endif
}
inline std::tuple<Tensor,Tensor> Tensor::geqrf() const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::geqrf(const_cast<Tensor&>(*this));
            break;
        default:
            AT_ERROR("geqrf not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::geqrf", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<std::tuple<Tensor,Tensor>, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}
inline Tensor Tensor::orgqr(const Tensor & input2) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::orgqr(const_cast<Tensor&>(*this), input2);
            break;
        default:
            AT_ERROR("orgqr not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::orgqr", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this, input2)))
        .callUnboxed<Tensor, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), input2);
#endif
}
inline Tensor Tensor::ormqr(const Tensor & input2, const Tensor & input3, bool left, bool transpose) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::ormqr(const_cast<Tensor&>(*this), input2, input3, left, transpose);
            break;
        default:
            AT_ERROR("ormqr not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::ormqr", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this, input2, input3)))
        .callUnboxed<Tensor, const Tensor &, const Tensor &, const Tensor &, bool, bool>(const_cast<Tensor&>(*this), input2, input3, left, transpose);
#endif
}
inline Tensor Tensor::lu_solve(const Tensor & LU_data, const Tensor & LU_pivots) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::lu_solve(const_cast<Tensor&>(*this), LU_data, LU_pivots);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::lu_solve", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this, LU_data, LU_pivots)))
        .callUnboxed<Tensor, const Tensor &, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), LU_data, LU_pivots);
#endif
}
inline Tensor Tensor::multinomial(int64_t num_samples, bool replacement, Generator * generator) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::multinomial(const_cast<Tensor&>(*this), num_samples, replacement, generator);
            break;
        default:
            AT_ERROR("multinomial not implemented for ", at::toString(type_set()));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::multinomial(Tensor self, int num_samples, bool replacement=False, *, Generator? generator=None) -> Tensor");
    return table->getOp<Tensor (const Tensor &, int64_t, bool, Generator *)>(at::detail::multi_dispatch_tensor_type_set(*this))(const_cast<Tensor&>(*this), num_samples, replacement, generator);
#endif
}
inline Tensor Tensor::lgamma() const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::lgamma(const_cast<Tensor&>(*this));
            break;
        default:
            AT_ERROR("lgamma not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::lgamma", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}
inline Tensor Tensor::digamma() const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::digamma(const_cast<Tensor&>(*this));
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::digamma", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}
inline Tensor Tensor::polygamma(int64_t n) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::polygamma(n, const_cast<Tensor&>(*this));
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::polygamma", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor, int64_t, const Tensor &>(n, const_cast<Tensor&>(*this));
#endif
}
inline Tensor Tensor::erfinv() const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::erfinv(const_cast<Tensor&>(*this));
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::erfinv", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}
inline Tensor & Tensor::erfinv_() const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::erfinv_(const_cast<Tensor&>(*this));
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::erfinv_", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor &, Tensor &>(const_cast<Tensor&>(*this));
#endif
}
inline Tensor Tensor::sign() const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::sign(const_cast<Tensor&>(*this));
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::sign", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}
inline Tensor & Tensor::sign_() const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::sign_(const_cast<Tensor&>(*this));
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::sign_", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor &, Tensor &>(const_cast<Tensor&>(*this));
#endif
}
inline Tensor Tensor::dist(const Tensor & other, Scalar p) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::dist(const_cast<Tensor&>(*this), other, p);
            break;
        default:
            AT_ERROR("dist not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::dist", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this, other)))
        .callUnboxed<Tensor, const Tensor &, const Tensor &, Scalar>(const_cast<Tensor&>(*this), other, p);
#endif
}
inline Tensor Tensor::atan2(const Tensor & other) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::atan2(const_cast<Tensor&>(*this), other);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::atan2", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this, other)))
        .callUnboxed<Tensor, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), other);
#endif
}
inline Tensor Tensor::lerp(const Tensor & end, Scalar weight) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::lerp(const_cast<Tensor&>(*this), end, weight);
            break;
        default:
            AT_ERROR("lerp not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::lerp", "Scalar"}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this, end)))
        .callUnboxed<Tensor, const Tensor &, const Tensor &, Scalar>(const_cast<Tensor&>(*this), end, weight);
#endif
}
inline Tensor Tensor::lerp(const Tensor & end, const Tensor & weight) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::lerp(const_cast<Tensor&>(*this), end, weight);
            break;
        default:
            AT_ERROR("lerp not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::lerp", "Tensor"}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this, end, weight)))
        .callUnboxed<Tensor, const Tensor &, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), end, weight);
#endif
}
inline Tensor Tensor::histc(int64_t bins, Scalar min, Scalar max) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::histc(const_cast<Tensor&>(*this), bins, min, max);
            break;
        default:
            AT_ERROR("histc not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::histc", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor, const Tensor &, int64_t, Scalar, Scalar>(const_cast<Tensor&>(*this), bins, min, max);
#endif
}
inline Tensor Tensor::fmod(Scalar other) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::fmod(const_cast<Tensor&>(*this), other);
            break;
        default:
            AT_ERROR("fmod not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::fmod", "Scalar"}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor, const Tensor &, Scalar>(const_cast<Tensor&>(*this), other);
#endif
}
inline Tensor Tensor::fmod(const Tensor & other) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::fmod(const_cast<Tensor&>(*this), other);
            break;
        default:
            AT_ERROR("fmod not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::fmod", "Tensor"}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this, other)))
        .callUnboxed<Tensor, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), other);
#endif
}
inline Tensor Tensor::remainder(Scalar other) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::remainder(const_cast<Tensor&>(*this), other);
            break;
        default:
            AT_ERROR("remainder not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::remainder", "Scalar"}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor, const Tensor &, Scalar>(const_cast<Tensor&>(*this), other);
#endif
}
inline Tensor Tensor::remainder(const Tensor & other) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::remainder(const_cast<Tensor&>(*this), other);
            break;
        default:
            AT_ERROR("remainder not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::remainder", "Tensor"}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this, other)))
        .callUnboxed<Tensor, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), other);
#endif
}
inline Tensor Tensor::min(const Tensor & other) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::min(const_cast<Tensor&>(*this), other);
            break;
        default:
            AT_ERROR("min not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::min", "other"}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this, other)))
        .callUnboxed<Tensor, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), other);
#endif
}
inline Tensor Tensor::min() const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::min(const_cast<Tensor&>(*this));
            break;
        case Backend::QuantizedCPU:
            return QuantizedCPUType::min(const_cast<Tensor&>(*this));
            break;
        default:
            AT_ERROR("min not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::min", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}
inline Tensor Tensor::max(const Tensor & other) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::max(const_cast<Tensor&>(*this), other);
            break;
        default:
            AT_ERROR("max not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::max", "other"}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this, other)))
        .callUnboxed<Tensor, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), other);
#endif
}
inline Tensor Tensor::max() const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::max(const_cast<Tensor&>(*this));
            break;
        case Backend::QuantizedCPU:
            return QuantizedCPUType::max(const_cast<Tensor&>(*this));
            break;
        default:
            AT_ERROR("max not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::max", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}
inline Tensor Tensor::median() const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::median(const_cast<Tensor&>(*this));
            break;
        default:
            AT_ERROR("median not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::median", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}
inline std::tuple<Tensor,Tensor> Tensor::sort(int64_t dim, bool descending) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::sort(const_cast<Tensor&>(*this), dim, descending);
            break;
        case Backend::QuantizedCPU:
            return QuantizedCPUType::sort(const_cast<Tensor&>(*this), dim, descending);
            break;
        default:
            AT_ERROR("sort not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::sort", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<std::tuple<Tensor,Tensor>, const Tensor &, int64_t, bool>(const_cast<Tensor&>(*this), dim, descending);
#endif
}
inline Tensor Tensor::argsort(int64_t dim, bool descending) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::argsort(const_cast<Tensor&>(*this), dim, descending);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::argsort", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor, const Tensor &, int64_t, bool>(const_cast<Tensor&>(*this), dim, descending);
#endif
}
inline std::tuple<Tensor,Tensor> Tensor::topk(int64_t k, int64_t dim, bool largest, bool sorted) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::topk(const_cast<Tensor&>(*this), k, dim, largest, sorted);
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::topk", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<std::tuple<Tensor,Tensor>, const Tensor &, int64_t, int64_t, bool, bool>(const_cast<Tensor&>(*this), k, dim, largest, sorted);
#endif
}
inline Tensor Tensor::all() const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::all(const_cast<Tensor&>(*this));
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::all", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}
inline Tensor Tensor::any() const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::any(const_cast<Tensor&>(*this));
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::any", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}
inline Tensor Tensor::renorm(Scalar p, int64_t dim, Scalar maxnorm) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::renorm(const_cast<Tensor&>(*this), p, dim, maxnorm);
            break;
        default:
            AT_ERROR("renorm not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::renorm", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor, const Tensor &, Scalar, int64_t, Scalar>(const_cast<Tensor&>(*this), p, dim, maxnorm);
#endif
}
inline Tensor Tensor::unfold(int64_t dimension, int64_t size, int64_t step) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::unfold(const_cast<Tensor&>(*this), dimension, size, step);
            break;
        default:
            AT_ERROR("unfold not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::unfold", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor, const Tensor &, int64_t, int64_t, int64_t>(const_cast<Tensor&>(*this), dimension, size, step);
#endif
}
inline bool Tensor::equal(const Tensor & other) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::equal(const_cast<Tensor&>(*this), other);
            break;
        case Backend::QuantizedCPU:
            return QuantizedCPUType::equal(const_cast<Tensor&>(*this), other);
            break;
        default:
            AT_ERROR("equal not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::equal", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this, other)))
        .callUnboxed<bool, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), other);
#endif
}
inline Tensor Tensor::pow(const Tensor & exponent) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(impl::dispatchTypeId(type_set()))) {
        case Backend::CPU:
            return CPUType::pow(const_cast<Tensor&>(*this), exponent);
            break;
        default:
            AT_ERROR("pow not implemented for ", at::toString(type_set()));
    }
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::pow", "Tensor_Tensor"}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this, exponent)))
        .callUnboxed<Tensor, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), exponent);
#endif
}
inline Tensor Tensor::alias() const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::alias(const_cast<Tensor&>(*this));
#else
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::alias", ""}).value();
    return c10::Dispatcher::singleton().lookup(op, impl::dispatchTypeId(at::detail::multi_dispatch_tensor_type_set(*this)))
        .callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
#endif
}

inline caffe2::TypeMeta Tensor::dtype() const noexcept {
  return impl_->dtype();
}

inline Layout Tensor::layout() const noexcept {
  return impl_->layout();
}

inline Device Tensor::device() const {
  return impl_->device();
}

inline int64_t Tensor::get_device() const {
  // NB: this is not a native function to avoid dispatching overhead.
  return impl_->get_device();
}

inline int64_t get_device(Tensor self) {
  return self.get_device();
}

inline bool Tensor::is_cuda() const {
  // NB: this is not a native function to avoid dispatching overhead.
  return impl_->is_cuda();
}

#ifdef BUILD_NAMEDTENSOR
inline NamedTensorMeta* Tensor::get_named_tensor_meta() {
  return static_cast<NamedTensorMeta*>(impl_->named_tensor_meta());
}

inline const NamedTensorMeta* Tensor::get_named_tensor_meta() const {
  return static_cast<NamedTensorMeta*>(impl_->named_tensor_meta());
}

inline bool Tensor::has_names() const {
  return impl::has_names(unsafeGetTensorImpl());
}
#endif

inline bool is_cuda(Tensor self) {
  return self.is_cuda();
}

inline bool Tensor::is_hip() const {
  // NB: this is not a native function to avoid dispatching overhead.
  return impl_->is_hip();
}

inline bool is_hip(Tensor self) {
  return self.is_hip();
}

inline bool Tensor::is_sparse() const {
  // NB: this is not a native function to avoid dispatching overhead.
  return impl_->is_sparse();
}

inline bool is_sparse(Tensor self) {
  return self.is_sparse();
}

inline bool Tensor::is_mkldnn() const {
  // NB: this is not a native function to avoid dispatching overhead.
  return impl_->is_mkldnn();
}

inline bool is_mkldnn(Tensor self) {
  return self.is_mkldnn();
}

inline bool Tensor::is_quantized() const {
  // NB: this is not a native function to avoid dispatching overhead.
  return impl_->is_quantized();
}

inline bool is_quantized(Tensor self) {
  return self.is_quantized();
}

#define DEFINE_CAST(T, name)                     \
  template <>                                    \
  inline T* Tensor::data_ptr() const {           \
    TORCH_CHECK(                                 \
        scalar_type() == ScalarType::name,       \
        "expected scalar type ",                 \
        #name,                                   \
        " but found ",                           \
        c10::toString(scalar_type()));           \
    return static_cast<T*>(this->unsafeGetTensorImpl()->data());    \
  }

AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_EXCEPT_COMPLEX_HALF(DEFINE_CAST)
AT_FORALL_QINT_TYPES(DEFINE_CAST)
#undef DEFINE_CAST

#define DEFINE_ITEM(T, name)      \
  template <>                     \
  inline T Tensor::item() const { \
    return item().to##name();     \
  }

AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_EXCEPT_COMPLEX_HALF(DEFINE_ITEM)
#undef DEFINE_ITEM

} //namespace at
