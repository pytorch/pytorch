#pragma once

#include <c10/core/Scalar.h>
#include <c10/core/MemoryFormat.h>
#include <c10/core/QScheme.h>
#include <c10/macros/Macros.h>
#include <c10/util/intrusive_ptr.h>
#include <ATen/core/DeprecatedTypeProperties.h>
#include <ATen/core/ATenDispatch.h>
#include <ATen/core/TensorOptions.h>
#if !defined(CAFFE2_IS_XPLAT_BUILD)
#include <ATen/core/dispatch/Dispatcher.h>
#endif
#ifdef BUILD_NAMEDTENSOR
#include <ATen/NamedTensor.h>
#endif

namespace at {

struct Quantizer;
// This is temporary typedef to enable Quantizer in aten native function API
// we'll remove them when we are actually exposing Quantizer class
// to frontend
using ConstQuantizerPtr = const c10::intrusive_ptr<Quantizer>&;

inline Tensor Tensor::toType(const DeprecatedTypeProperties & t, bool non_blocking) const {
  if(type() == t)
    return *this;
  return to(
      at::device(t.device_type()).layout(t.layout()).dtype(t.scalarType()),
      non_blocking,
      /*copy=*/ true);
}

inline Tensor Tensor::cpu() const {
  return toType(type().cpu());
}

inline Tensor Tensor::cuda() const {
  return toType(type().cuda());
}

inline Tensor Tensor::hip() const {
  return toType(type().hip());
}

inline Tensor Tensor::toType(ScalarType t) const {
  return toType(type().toScalarType(t));
}

inline Tensor Tensor::toBackend(Backend b) const {
  return toType(type().toBackend(b));
}

inline TensorOptions Tensor::options() const {
  return TensorOptions().dtype(dtype())
                        .device(device())
                        .layout(layout())
                        .is_variable(is_variable());
}

// all static inline to allow for inlining of the non-dynamic part of dispatch
inline int64_t Tensor::bench__one_arg_at() const {
    static auto table = globalATenDispatch().getOpTable("aten::bench__one_arg_at(Tensor self) -> int");
    return table->getOp<int64_t (const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this));
}
inline int64_t Tensor::bench__one_arg_c10() const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::bench__one_arg_c10", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<int64_t, const Tensor &>(op, const_cast<Tensor&>(*this));
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<int64_t, const Tensor &>(const_cast<Tensor&>(*this));
    }
}
inline Tensor Tensor::bench__one_arg_return_at() const {
    static auto table = globalATenDispatch().getOpTable("aten::bench__one_arg_return_at(Tensor self) -> Tensor");
    return table->getOp<Tensor (const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this));
}
inline Tensor Tensor::bench__one_arg_return_c10() const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::bench__one_arg_return_c10", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &>(op, const_cast<Tensor&>(*this));
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
    }
}
inline int64_t Tensor::bench__two_args_at(const Tensor & other) const {
    static auto table = globalATenDispatch().getOpTable("aten::bench__two_args_at(Tensor self, Tensor other) -> int");
    return table->getOp<int64_t (const Tensor &, const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), other);
}
inline int64_t Tensor::bench__two_args_c10(const Tensor & other) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::bench__two_args_c10", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<int64_t, const Tensor &, const Tensor &>(op, const_cast<Tensor&>(*this), other);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<int64_t, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), other);
    }
}
inline Tensor Tensor::bench__two_args_return_at(const Tensor & other) const {
    static auto table = globalATenDispatch().getOpTable("aten::bench__two_args_return_at(Tensor self, Tensor other) -> Tensor");
    return table->getOp<Tensor (const Tensor &, const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), other);
}
inline Tensor Tensor::bench__two_args_return_c10(const Tensor & other) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::bench__two_args_return_c10", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &, const Tensor &>(op, const_cast<Tensor&>(*this), other);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), other);
    }
}
inline int64_t Tensor::bench__three_args_at(const Tensor & other, const Tensor & third) const {
    static auto table = globalATenDispatch().getOpTable("aten::bench__three_args_at(Tensor self, Tensor other, Tensor third) -> int");
    return table->getOp<int64_t (const Tensor &, const Tensor &, const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), other, third);
}
inline int64_t Tensor::bench__three_args_c10(const Tensor & other, const Tensor & third) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::bench__three_args_c10", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<int64_t, const Tensor &, const Tensor &, const Tensor &>(op, const_cast<Tensor&>(*this), other, third);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<int64_t, const Tensor &, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), other, third);
    }
}
inline Tensor Tensor::bench__three_args_return_at(const Tensor & other, const Tensor & third) const {
    static auto table = globalATenDispatch().getOpTable("aten::bench__three_args_return_at(Tensor self, Tensor other, Tensor third) -> Tensor");
    return table->getOp<Tensor (const Tensor &, const Tensor &, const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), other, third);
}
inline Tensor Tensor::bench__three_args_return_c10(const Tensor & other, const Tensor & third) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::bench__three_args_return_c10", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &, const Tensor &, const Tensor &>(op, const_cast<Tensor&>(*this), other, third);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), other, third);
    }
}
inline Tensor Tensor::bench__add_at(const Tensor & b) const {
    static auto table = globalATenDispatch().getOpTable("aten::bench__add_at(Tensor self, Tensor b) -> Tensor");
    return table->getOp<Tensor (const Tensor &, const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), b);
}
inline Tensor Tensor::bench__add_c10(const Tensor & b) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::bench__add_c10", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &, const Tensor &>(op, const_cast<Tensor&>(*this), b);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), b);
    }
}
inline int64_t Tensor::bench__one_arg_dispatch_at() const {
    static auto table = globalATenDispatch().getOpTable("aten::bench__one_arg_dispatch_at(Tensor self) -> int");
    return table->getOp<int64_t (const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this));
}
inline int64_t Tensor::bench__one_arg_dispatch_c10() const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::bench__one_arg_dispatch_c10", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<int64_t, const Tensor &>(op, const_cast<Tensor&>(*this));
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<int64_t, const Tensor &>(const_cast<Tensor&>(*this));
    }
}
inline void Tensor::backward(const Tensor & gradient, bool keep_graph, bool create_graph) const {
    static auto table = globalATenDispatch().getOpTable("aten::backward(Tensor self, Tensor? gradient=None, bool keep_graph=False, bool create_graph=False) -> void");
    return table->getOp<void (const Tensor &, const Tensor &, bool, bool)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), gradient, keep_graph, create_graph);
}
inline void Tensor::set_data(const Tensor & new_data) const {
    static auto table = globalATenDispatch().getOpTable("aten::set_data(Tensor(a!) self, Tensor new_data) -> void");
    return table->getOp<void (const Tensor &, const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), new_data);
}
#ifdef BUILD_NAMEDTENSOR
inline Tensor & Tensor::set_names_(c10::optional<DimnameList> names) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::set_names_", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor &, Tensor &, c10::optional<DimnameList>>(op, const_cast<Tensor&>(*this), names);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor &, Tensor &, c10::optional<DimnameList>>(const_cast<Tensor&>(*this), names);
    }
}
#endif
inline Tensor Tensor::abs() const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::abs", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &>(op, const_cast<Tensor&>(*this));
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
    }
}
inline Tensor & Tensor::abs_() const {
    static auto table = globalATenDispatch().getOpTable("aten::abs_(Tensor(a!) self) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this));
}
inline Tensor Tensor::acos() const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::acos", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &>(op, const_cast<Tensor&>(*this));
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
    }
}
inline Tensor & Tensor::acos_() const {
    static auto table = globalATenDispatch().getOpTable("aten::acos_(Tensor(a!) self) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this));
}
inline Tensor Tensor::add(const Tensor & other, Scalar alpha) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::add", "Tensor"}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &, const Tensor &, Scalar>(op, const_cast<Tensor&>(*this), other, alpha);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &, const Tensor &, Scalar>(const_cast<Tensor&>(*this), other, alpha);
    }
}
inline Tensor & Tensor::add_(const Tensor & other, Scalar alpha) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::add_", "Tensor"}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor &, Tensor &, const Tensor &, Scalar>(op, const_cast<Tensor&>(*this), other, alpha);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor &, Tensor &, const Tensor &, Scalar>(const_cast<Tensor&>(*this), other, alpha);
    }
}
inline Tensor Tensor::add(Scalar other, Scalar alpha) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::add", "Scalar"}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &, Scalar, Scalar>(op, const_cast<Tensor&>(*this), other, alpha);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &, Scalar, Scalar>(const_cast<Tensor&>(*this), other, alpha);
    }
}
inline Tensor & Tensor::add_(Scalar other, Scalar alpha) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::add_", "Scalar"}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor &, Tensor &, Scalar, Scalar>(op, const_cast<Tensor&>(*this), other, alpha);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor &, Tensor &, Scalar, Scalar>(const_cast<Tensor&>(*this), other, alpha);
    }
}
inline Tensor Tensor::addmv(const Tensor & mat, const Tensor & vec, Scalar beta, Scalar alpha) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::addmv", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar>(op, const_cast<Tensor&>(*this), mat, vec, beta, alpha);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar>(const_cast<Tensor&>(*this), mat, vec, beta, alpha);
    }
}
inline Tensor & Tensor::addmv_(const Tensor & mat, const Tensor & vec, Scalar beta, Scalar alpha) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::addmv_", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor &, Tensor &, const Tensor &, const Tensor &, Scalar, Scalar>(op, const_cast<Tensor&>(*this), mat, vec, beta, alpha);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor &, Tensor &, const Tensor &, const Tensor &, Scalar, Scalar>(const_cast<Tensor&>(*this), mat, vec, beta, alpha);
    }
}
inline Tensor Tensor::addr(const Tensor & vec1, const Tensor & vec2, Scalar beta, Scalar alpha) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::addr", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar>(op, const_cast<Tensor&>(*this), vec1, vec2, beta, alpha);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar>(const_cast<Tensor&>(*this), vec1, vec2, beta, alpha);
    }
}
inline Tensor & Tensor::addr_(const Tensor & vec1, const Tensor & vec2, Scalar beta, Scalar alpha) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::addr_", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor &, Tensor &, const Tensor &, const Tensor &, Scalar, Scalar>(op, const_cast<Tensor&>(*this), vec1, vec2, beta, alpha);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor &, Tensor &, const Tensor &, const Tensor &, Scalar, Scalar>(const_cast<Tensor&>(*this), vec1, vec2, beta, alpha);
    }
}
inline Tensor Tensor::all(int64_t dim, bool keepdim) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::all", "dim"}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &, int64_t, bool>(op, const_cast<Tensor&>(*this), dim, keepdim);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &, int64_t, bool>(const_cast<Tensor&>(*this), dim, keepdim);
    }
}
inline bool Tensor::allclose(const Tensor & other, double rtol, double atol, bool equal_nan) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::allclose", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<bool, const Tensor &, const Tensor &, double, double, bool>(op, const_cast<Tensor&>(*this), other, rtol, atol, equal_nan);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<bool, const Tensor &, const Tensor &, double, double, bool>(const_cast<Tensor&>(*this), other, rtol, atol, equal_nan);
    }
}
inline Tensor Tensor::any(int64_t dim, bool keepdim) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::any", "dim"}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &, int64_t, bool>(op, const_cast<Tensor&>(*this), dim, keepdim);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &, int64_t, bool>(const_cast<Tensor&>(*this), dim, keepdim);
    }
}
inline Tensor Tensor::argmax(c10::optional<int64_t> dim, bool keepdim) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::argmax", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &, c10::optional<int64_t>, bool>(op, const_cast<Tensor&>(*this), dim, keepdim);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &, c10::optional<int64_t>, bool>(const_cast<Tensor&>(*this), dim, keepdim);
    }
}
inline Tensor Tensor::argmin(c10::optional<int64_t> dim, bool keepdim) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::argmin", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &, c10::optional<int64_t>, bool>(op, const_cast<Tensor&>(*this), dim, keepdim);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &, c10::optional<int64_t>, bool>(const_cast<Tensor&>(*this), dim, keepdim);
    }
}
inline Tensor Tensor::as_strided(IntArrayRef size, IntArrayRef stride, c10::optional<int64_t> storage_offset) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::as_strided", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &, IntArrayRef, IntArrayRef, c10::optional<int64_t>>(op, const_cast<Tensor&>(*this), size, stride, storage_offset);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &, IntArrayRef, IntArrayRef, c10::optional<int64_t>>(const_cast<Tensor&>(*this), size, stride, storage_offset);
    }
}
inline Tensor & Tensor::as_strided_(IntArrayRef size, IntArrayRef stride, c10::optional<int64_t> storage_offset) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::as_strided_", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor &, Tensor &, IntArrayRef, IntArrayRef, c10::optional<int64_t>>(op, const_cast<Tensor&>(*this), size, stride, storage_offset);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor &, Tensor &, IntArrayRef, IntArrayRef, c10::optional<int64_t>>(const_cast<Tensor&>(*this), size, stride, storage_offset);
    }
}
inline Tensor Tensor::asin() const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::asin", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &>(op, const_cast<Tensor&>(*this));
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
    }
}
inline Tensor & Tensor::asin_() const {
    static auto table = globalATenDispatch().getOpTable("aten::asin_(Tensor(a!) self) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this));
}
inline Tensor Tensor::atan() const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::atan", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &>(op, const_cast<Tensor&>(*this));
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
    }
}
inline Tensor & Tensor::atan_() const {
    static auto table = globalATenDispatch().getOpTable("aten::atan_(Tensor(a!) self) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this));
}
inline Tensor Tensor::baddbmm(const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::baddbmm", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar>(op, const_cast<Tensor&>(*this), batch1, batch2, beta, alpha);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar>(const_cast<Tensor&>(*this), batch1, batch2, beta, alpha);
    }
}
inline Tensor & Tensor::baddbmm_(const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::baddbmm_", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor &, Tensor &, const Tensor &, const Tensor &, Scalar, Scalar>(op, const_cast<Tensor&>(*this), batch1, batch2, beta, alpha);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor &, Tensor &, const Tensor &, const Tensor &, Scalar, Scalar>(const_cast<Tensor&>(*this), batch1, batch2, beta, alpha);
    }
}
inline Tensor Tensor::bernoulli(Generator * generator) const {
    static auto table = globalATenDispatch().getOpTable("aten::bernoulli(Tensor self, *, Generator? generator=None) -> Tensor");
    return table->getOp<Tensor (const Tensor &, Generator *)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), generator);
}
inline Tensor & Tensor::bernoulli_(const Tensor & p, Generator * generator) const {
    static auto table = globalATenDispatch().getOpTable("aten::bernoulli_.Tensor(Tensor(a!) self, Tensor p, *, Generator? generator=None) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, const Tensor &, Generator *)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), p, generator);
}
inline Tensor & Tensor::bernoulli_(double p, Generator * generator) const {
    static auto table = globalATenDispatch().getOpTable("aten::bernoulli_.float(Tensor(a!) self, float p=0.5, *, Generator? generator=None) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, double, Generator *)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), p, generator);
}
inline Tensor Tensor::bernoulli(double p, Generator * generator) const {
    static auto table = globalATenDispatch().getOpTable("aten::bernoulli.p(Tensor self, float p, *, Generator? generator=None) -> Tensor");
    return table->getOp<Tensor (const Tensor &, double, Generator *)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), p, generator);
}
inline Tensor Tensor::bincount(const Tensor & weights, int64_t minlength) const {
    static auto table = globalATenDispatch().getOpTable("aten::bincount(Tensor self, Tensor? weights=None, int minlength=0) -> Tensor");
    return table->getOp<Tensor (const Tensor &, const Tensor &, int64_t)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), weights, minlength);
}
inline Tensor Tensor::bitwise_not() const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::bitwise_not", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &>(op, const_cast<Tensor&>(*this));
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
    }
}
inline Tensor & Tensor::bitwise_not_() const {
    static auto table = globalATenDispatch().getOpTable("aten::bitwise_not_(Tensor(a!) self) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this));
}
inline Tensor Tensor::bmm(const Tensor & mat2) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::bmm", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &, const Tensor &>(op, const_cast<Tensor&>(*this), mat2);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), mat2);
    }
}
inline Tensor Tensor::ceil() const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::ceil", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &>(op, const_cast<Tensor&>(*this));
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
    }
}
inline Tensor & Tensor::ceil_() const {
    static auto table = globalATenDispatch().getOpTable("aten::ceil_(Tensor(a!) self) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this));
}
inline std::vector<Tensor> Tensor::chunk(int64_t chunks, int64_t dim) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::chunk", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<std::vector<Tensor>, const Tensor &, int64_t, int64_t>(op, const_cast<Tensor&>(*this), chunks, dim);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<std::vector<Tensor>, const Tensor &, int64_t, int64_t>(const_cast<Tensor&>(*this), chunks, dim);
    }
}
inline Tensor Tensor::clamp(c10::optional<Scalar> min, c10::optional<Scalar> max) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::clamp", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &, c10::optional<Scalar>, c10::optional<Scalar>>(op, const_cast<Tensor&>(*this), min, max);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &, c10::optional<Scalar>, c10::optional<Scalar>>(const_cast<Tensor&>(*this), min, max);
    }
}
inline Tensor & Tensor::clamp_(c10::optional<Scalar> min, c10::optional<Scalar> max) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::clamp_", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor &, Tensor &, c10::optional<Scalar>, c10::optional<Scalar>>(op, const_cast<Tensor&>(*this), min, max);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor &, Tensor &, c10::optional<Scalar>, c10::optional<Scalar>>(const_cast<Tensor&>(*this), min, max);
    }
}
inline Tensor Tensor::clamp_max(Scalar max) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::clamp_max", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &, Scalar>(op, const_cast<Tensor&>(*this), max);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &, Scalar>(const_cast<Tensor&>(*this), max);
    }
}
inline Tensor & Tensor::clamp_max_(Scalar max) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::clamp_max_", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor &, Tensor &, Scalar>(op, const_cast<Tensor&>(*this), max);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor &, Tensor &, Scalar>(const_cast<Tensor&>(*this), max);
    }
}
inline Tensor Tensor::clamp_min(Scalar min) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::clamp_min", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &, Scalar>(op, const_cast<Tensor&>(*this), min);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &, Scalar>(const_cast<Tensor&>(*this), min);
    }
}
inline Tensor & Tensor::clamp_min_(Scalar min) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::clamp_min_", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor &, Tensor &, Scalar>(op, const_cast<Tensor&>(*this), min);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor &, Tensor &, Scalar>(const_cast<Tensor&>(*this), min);
    }
}
inline Tensor Tensor::contiguous(MemoryFormat memory_format) const {
    static auto table = globalATenDispatch().getOpTable("aten::contiguous(Tensor self, *, MemoryFormat memory_format=contiguous_format) -> Tensor");
    return table->getOp<Tensor (const Tensor &, MemoryFormat)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), memory_format);
}
inline Tensor & Tensor::copy_(const Tensor & src, bool non_blocking) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::copy_", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor &, Tensor &, const Tensor &, bool>(op, const_cast<Tensor&>(*this), src, non_blocking);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor &, Tensor &, const Tensor &, bool>(const_cast<Tensor&>(*this), src, non_blocking);
    }
}
inline Tensor Tensor::cos() const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::cos", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &>(op, const_cast<Tensor&>(*this));
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
    }
}
inline Tensor & Tensor::cos_() const {
    static auto table = globalATenDispatch().getOpTable("aten::cos_(Tensor(a!) self) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this));
}
inline Tensor Tensor::cosh() const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::cosh", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &>(op, const_cast<Tensor&>(*this));
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
    }
}
inline Tensor & Tensor::cosh_() const {
    static auto table = globalATenDispatch().getOpTable("aten::cosh_(Tensor(a!) self) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this));
}
inline Tensor Tensor::cumsum(int64_t dim, c10::optional<ScalarType> dtype) const {
    static auto table = globalATenDispatch().getOpTable("aten::cumsum(Tensor self, int dim, *, ScalarType? dtype=None) -> Tensor");
    return table->getOp<Tensor (const Tensor &, int64_t, c10::optional<ScalarType>)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), dim, dtype);
}
inline Tensor Tensor::cumprod(int64_t dim, c10::optional<ScalarType> dtype) const {
    static auto table = globalATenDispatch().getOpTable("aten::cumprod(Tensor self, int dim, *, ScalarType? dtype=None) -> Tensor");
    return table->getOp<Tensor (const Tensor &, int64_t, c10::optional<ScalarType>)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), dim, dtype);
}
inline Tensor Tensor::det() const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::det", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &>(op, const_cast<Tensor&>(*this));
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
    }
}
inline Tensor Tensor::diag_embed(int64_t offset, int64_t dim1, int64_t dim2) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::diag_embed", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &, int64_t, int64_t, int64_t>(op, const_cast<Tensor&>(*this), offset, dim1, dim2);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &, int64_t, int64_t, int64_t>(const_cast<Tensor&>(*this), offset, dim1, dim2);
    }
}
inline Tensor Tensor::diagflat(int64_t offset) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::diagflat", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &, int64_t>(op, const_cast<Tensor&>(*this), offset);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &, int64_t>(const_cast<Tensor&>(*this), offset);
    }
}
inline Tensor Tensor::diagonal(int64_t offset, int64_t dim1, int64_t dim2) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::diagonal", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &, int64_t, int64_t, int64_t>(op, const_cast<Tensor&>(*this), offset, dim1, dim2);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &, int64_t, int64_t, int64_t>(const_cast<Tensor&>(*this), offset, dim1, dim2);
    }
}
inline Tensor & Tensor::fill_diagonal_(Scalar fill_value, bool wrap) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::fill_diagonal_", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor &, Tensor &, Scalar, bool>(op, const_cast<Tensor&>(*this), fill_value, wrap);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor &, Tensor &, Scalar, bool>(const_cast<Tensor&>(*this), fill_value, wrap);
    }
}
inline Tensor Tensor::div(const Tensor & other) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::div", "Tensor"}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &, const Tensor &>(op, const_cast<Tensor&>(*this), other);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), other);
    }
}
inline Tensor & Tensor::div_(const Tensor & other) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::div_", "Tensor"}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor &, Tensor &, const Tensor &>(op, const_cast<Tensor&>(*this), other);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor &, Tensor &, const Tensor &>(const_cast<Tensor&>(*this), other);
    }
}
inline Tensor Tensor::div(Scalar other) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::div", "Scalar"}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &, Scalar>(op, const_cast<Tensor&>(*this), other);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &, Scalar>(const_cast<Tensor&>(*this), other);
    }
}
inline Tensor & Tensor::div_(Scalar other) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::div_", "Scalar"}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor &, Tensor &, Scalar>(op, const_cast<Tensor&>(*this), other);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor &, Tensor &, Scalar>(const_cast<Tensor&>(*this), other);
    }
}
inline Tensor Tensor::dot(const Tensor & tensor) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::dot", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &, const Tensor &>(op, const_cast<Tensor&>(*this), tensor);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), tensor);
    }
}
inline Tensor & Tensor::resize_(IntArrayRef size) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::resize_", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor &, Tensor &, IntArrayRef>(op, const_cast<Tensor&>(*this), size);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor &, Tensor &, IntArrayRef>(const_cast<Tensor&>(*this), size);
    }
}
inline Tensor Tensor::erf() const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::erf", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &>(op, const_cast<Tensor&>(*this));
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
    }
}
inline Tensor & Tensor::erf_() const {
    static auto table = globalATenDispatch().getOpTable("aten::erf_(Tensor(a!) self) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this));
}
inline Tensor Tensor::erfc() const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::erfc", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &>(op, const_cast<Tensor&>(*this));
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
    }
}
inline Tensor & Tensor::erfc_() const {
    static auto table = globalATenDispatch().getOpTable("aten::erfc_(Tensor(a!) self) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this));
}
inline Tensor Tensor::exp() const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::exp", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &>(op, const_cast<Tensor&>(*this));
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
    }
}
inline Tensor & Tensor::exp_() const {
    static auto table = globalATenDispatch().getOpTable("aten::exp_(Tensor(a!) self) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this));
}
inline Tensor Tensor::expm1() const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::expm1", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &>(op, const_cast<Tensor&>(*this));
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
    }
}
inline Tensor & Tensor::expm1_() const {
    static auto table = globalATenDispatch().getOpTable("aten::expm1_(Tensor(a!) self) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this));
}
inline Tensor Tensor::expand(IntArrayRef size, bool implicit) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::expand", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &, IntArrayRef, bool>(op, const_cast<Tensor&>(*this), size, implicit);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &, IntArrayRef, bool>(const_cast<Tensor&>(*this), size, implicit);
    }
}
inline Tensor Tensor::expand_as(const Tensor & other) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::expand_as", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &, const Tensor &>(op, const_cast<Tensor&>(*this), other);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), other);
    }
}
inline Tensor Tensor::flatten(int64_t start_dim, int64_t end_dim) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::flatten", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &, int64_t, int64_t>(op, const_cast<Tensor&>(*this), start_dim, end_dim);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &, int64_t, int64_t>(const_cast<Tensor&>(*this), start_dim, end_dim);
    }
}
inline Tensor & Tensor::fill_(Scalar value) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::fill_", "Scalar"}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor &, Tensor &, Scalar>(op, const_cast<Tensor&>(*this), value);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor &, Tensor &, Scalar>(const_cast<Tensor&>(*this), value);
    }
}
inline Tensor & Tensor::fill_(const Tensor & value) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::fill_", "Tensor"}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor &, Tensor &, const Tensor &>(op, const_cast<Tensor&>(*this), value);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor &, Tensor &, const Tensor &>(const_cast<Tensor&>(*this), value);
    }
}
inline Tensor Tensor::floor() const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::floor", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &>(op, const_cast<Tensor&>(*this));
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
    }
}
inline Tensor & Tensor::floor_() const {
    static auto table = globalATenDispatch().getOpTable("aten::floor_(Tensor(a!) self) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this));
}
inline Tensor Tensor::frac() const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::frac", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &>(op, const_cast<Tensor&>(*this));
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
    }
}
inline Tensor & Tensor::frac_() const {
    static auto table = globalATenDispatch().getOpTable("aten::frac_(Tensor(a!) self) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this));
}
inline Tensor Tensor::ger(const Tensor & vec2) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::ger", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &, const Tensor &>(op, const_cast<Tensor&>(*this), vec2);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), vec2);
    }
}
inline Tensor Tensor::fft(int64_t signal_ndim, bool normalized) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::fft", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &, int64_t, bool>(op, const_cast<Tensor&>(*this), signal_ndim, normalized);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &, int64_t, bool>(const_cast<Tensor&>(*this), signal_ndim, normalized);
    }
}
inline Tensor Tensor::ifft(int64_t signal_ndim, bool normalized) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::ifft", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &, int64_t, bool>(op, const_cast<Tensor&>(*this), signal_ndim, normalized);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &, int64_t, bool>(const_cast<Tensor&>(*this), signal_ndim, normalized);
    }
}
inline Tensor Tensor::rfft(int64_t signal_ndim, bool normalized, bool onesided) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::rfft", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &, int64_t, bool, bool>(op, const_cast<Tensor&>(*this), signal_ndim, normalized, onesided);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &, int64_t, bool, bool>(const_cast<Tensor&>(*this), signal_ndim, normalized, onesided);
    }
}
inline Tensor Tensor::irfft(int64_t signal_ndim, bool normalized, bool onesided, IntArrayRef signal_sizes) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::irfft", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &, int64_t, bool, bool, IntArrayRef>(op, const_cast<Tensor&>(*this), signal_ndim, normalized, onesided, signal_sizes);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &, int64_t, bool, bool, IntArrayRef>(const_cast<Tensor&>(*this), signal_ndim, normalized, onesided, signal_sizes);
    }
}
inline Tensor Tensor::index(TensorList indices) const {
    static auto table = globalATenDispatch().getOpTable("aten::index(Tensor self, Tensor?[] indices) -> Tensor");
    return table->getOp<Tensor (const Tensor &, TensorList)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), indices);
}
inline Tensor & Tensor::index_copy_(int64_t dim, const Tensor & index, const Tensor & source) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::index_copy_", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor &, Tensor &, int64_t, const Tensor &, const Tensor &>(op, const_cast<Tensor&>(*this), dim, index, source);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor &, Tensor &, int64_t, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), dim, index, source);
    }
}
inline Tensor Tensor::index_copy(int64_t dim, const Tensor & index, const Tensor & source) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::index_copy", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &, int64_t, const Tensor &, const Tensor &>(op, const_cast<Tensor&>(*this), dim, index, source);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &, int64_t, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), dim, index, source);
    }
}
inline Tensor & Tensor::index_put_(TensorList indices, const Tensor & values, bool accumulate) const {
    static auto table = globalATenDispatch().getOpTable("aten::index_put_(Tensor(a!) self, Tensor?[] indices, Tensor values, bool accumulate=False) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, TensorList, const Tensor &, bool)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), indices, values, accumulate);
}
inline Tensor Tensor::index_put(TensorList indices, const Tensor & values, bool accumulate) const {
    static auto table = globalATenDispatch().getOpTable("aten::index_put(Tensor self, Tensor?[] indices, Tensor values, bool accumulate=False) -> Tensor");
    return table->getOp<Tensor (const Tensor &, TensorList, const Tensor &, bool)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), indices, values, accumulate);
}
inline Tensor Tensor::inverse() const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::inverse", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &>(op, const_cast<Tensor&>(*this));
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
    }
}
inline Tensor Tensor::isclose(const Tensor & other, double rtol, double atol, bool equal_nan) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::isclose", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &, const Tensor &, double, double, bool>(op, const_cast<Tensor&>(*this), other, rtol, atol, equal_nan);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &, const Tensor &, double, double, bool>(const_cast<Tensor&>(*this), other, rtol, atol, equal_nan);
    }
}
inline bool Tensor::is_distributed() const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::is_distributed", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<bool, const Tensor &>(op, const_cast<Tensor&>(*this));
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<bool, const Tensor &>(const_cast<Tensor&>(*this));
    }
}
inline bool Tensor::is_floating_point() const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::is_floating_point", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<bool, const Tensor &>(op, const_cast<Tensor&>(*this));
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<bool, const Tensor &>(const_cast<Tensor&>(*this));
    }
}
inline bool Tensor::is_complex() const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::is_complex", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<bool, const Tensor &>(op, const_cast<Tensor&>(*this));
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<bool, const Tensor &>(const_cast<Tensor&>(*this));
    }
}
inline bool Tensor::is_nonzero() const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::is_nonzero", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<bool, const Tensor &>(op, const_cast<Tensor&>(*this));
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<bool, const Tensor &>(const_cast<Tensor&>(*this));
    }
}
inline bool Tensor::is_same_size(const Tensor & other) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::is_same_size", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<bool, const Tensor &, const Tensor &>(op, const_cast<Tensor&>(*this), other);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<bool, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), other);
    }
}
inline bool Tensor::is_signed() const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::is_signed", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<bool, const Tensor &>(op, const_cast<Tensor&>(*this));
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<bool, const Tensor &>(const_cast<Tensor&>(*this));
    }
}
inline std::tuple<Tensor,Tensor> Tensor::kthvalue(int64_t k, int64_t dim, bool keepdim) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::kthvalue", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<std::tuple<Tensor,Tensor>, const Tensor &, int64_t, int64_t, bool>(op, const_cast<Tensor&>(*this), k, dim, keepdim);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<std::tuple<Tensor,Tensor>, const Tensor &, int64_t, int64_t, bool>(const_cast<Tensor&>(*this), k, dim, keepdim);
    }
}
inline Tensor Tensor::log() const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::log", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &>(op, const_cast<Tensor&>(*this));
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
    }
}
inline Tensor & Tensor::log_() const {
    static auto table = globalATenDispatch().getOpTable("aten::log_(Tensor(a!) self) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this));
}
inline Tensor Tensor::log10() const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::log10", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &>(op, const_cast<Tensor&>(*this));
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
    }
}
inline Tensor & Tensor::log10_() const {
    static auto table = globalATenDispatch().getOpTable("aten::log10_(Tensor(a!) self) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this));
}
inline Tensor Tensor::log1p() const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::log1p", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &>(op, const_cast<Tensor&>(*this));
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
    }
}
inline Tensor & Tensor::log1p_() const {
    static auto table = globalATenDispatch().getOpTable("aten::log1p_(Tensor(a!) self) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this));
}
inline Tensor Tensor::log2() const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::log2", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &>(op, const_cast<Tensor&>(*this));
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
    }
}
inline Tensor & Tensor::log2_() const {
    static auto table = globalATenDispatch().getOpTable("aten::log2_(Tensor(a!) self) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this));
}
inline Tensor Tensor::logdet() const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::logdet", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &>(op, const_cast<Tensor&>(*this));
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
    }
}
inline Tensor Tensor::log_softmax(int64_t dim, c10::optional<ScalarType> dtype) const {
    static auto table = globalATenDispatch().getOpTable("aten::log_softmax(Tensor self, int dim, ScalarType? dtype=None) -> Tensor");
    return table->getOp<Tensor (const Tensor &, int64_t, c10::optional<ScalarType>)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), dim, dtype);
}
inline Tensor Tensor::logsumexp(IntArrayRef dim, bool keepdim) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::logsumexp", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &, IntArrayRef, bool>(op, const_cast<Tensor&>(*this), dim, keepdim);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &, IntArrayRef, bool>(const_cast<Tensor&>(*this), dim, keepdim);
    }
}
inline Tensor Tensor::matmul(const Tensor & other) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::matmul", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &, const Tensor &>(op, const_cast<Tensor&>(*this), other);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), other);
    }
}
inline Tensor Tensor::matrix_power(int64_t n) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::matrix_power", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &, int64_t>(op, const_cast<Tensor&>(*this), n);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &, int64_t>(const_cast<Tensor&>(*this), n);
    }
}
inline std::tuple<Tensor,Tensor> Tensor::max(int64_t dim, bool keepdim) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::max", "dim"}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<std::tuple<Tensor,Tensor>, const Tensor &, int64_t, bool>(op, const_cast<Tensor&>(*this), dim, keepdim);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<std::tuple<Tensor,Tensor>, const Tensor &, int64_t, bool>(const_cast<Tensor&>(*this), dim, keepdim);
    }
}
inline Tensor Tensor::max_values(IntArrayRef dim, bool keepdim) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::max_values", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &, IntArrayRef, bool>(op, const_cast<Tensor&>(*this), dim, keepdim);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &, IntArrayRef, bool>(const_cast<Tensor&>(*this), dim, keepdim);
    }
}
inline Tensor Tensor::mean(c10::optional<ScalarType> dtype) const {
    static auto table = globalATenDispatch().getOpTable("aten::mean(Tensor self, *, ScalarType? dtype=None) -> Tensor");
    return table->getOp<Tensor (const Tensor &, c10::optional<ScalarType>)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), dtype);
}
inline Tensor Tensor::mean(IntArrayRef dim, bool keepdim, c10::optional<ScalarType> dtype) const {
    static auto table = globalATenDispatch().getOpTable("aten::mean.dim(Tensor self, int[1] dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor");
    return table->getOp<Tensor (const Tensor &, IntArrayRef, bool, c10::optional<ScalarType>)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), dim, keepdim, dtype);
}
inline std::tuple<Tensor,Tensor> Tensor::median(int64_t dim, bool keepdim) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::median", "dim"}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<std::tuple<Tensor,Tensor>, const Tensor &, int64_t, bool>(op, const_cast<Tensor&>(*this), dim, keepdim);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<std::tuple<Tensor,Tensor>, const Tensor &, int64_t, bool>(const_cast<Tensor&>(*this), dim, keepdim);
    }
}
inline std::tuple<Tensor,Tensor> Tensor::min(int64_t dim, bool keepdim) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::min", "dim"}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<std::tuple<Tensor,Tensor>, const Tensor &, int64_t, bool>(op, const_cast<Tensor&>(*this), dim, keepdim);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<std::tuple<Tensor,Tensor>, const Tensor &, int64_t, bool>(const_cast<Tensor&>(*this), dim, keepdim);
    }
}
inline Tensor Tensor::min_values(IntArrayRef dim, bool keepdim) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::min_values", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &, IntArrayRef, bool>(op, const_cast<Tensor&>(*this), dim, keepdim);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &, IntArrayRef, bool>(const_cast<Tensor&>(*this), dim, keepdim);
    }
}
inline Tensor Tensor::mm(const Tensor & mat2) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::mm", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &, const Tensor &>(op, const_cast<Tensor&>(*this), mat2);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), mat2);
    }
}
inline std::tuple<Tensor,Tensor> Tensor::mode(int64_t dim, bool keepdim) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::mode", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<std::tuple<Tensor,Tensor>, const Tensor &, int64_t, bool>(op, const_cast<Tensor&>(*this), dim, keepdim);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<std::tuple<Tensor,Tensor>, const Tensor &, int64_t, bool>(const_cast<Tensor&>(*this), dim, keepdim);
    }
}
inline Tensor Tensor::mul(const Tensor & other) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::mul", "Tensor"}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &, const Tensor &>(op, const_cast<Tensor&>(*this), other);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), other);
    }
}
inline Tensor & Tensor::mul_(const Tensor & other) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::mul_", "Tensor"}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor &, Tensor &, const Tensor &>(op, const_cast<Tensor&>(*this), other);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor &, Tensor &, const Tensor &>(const_cast<Tensor&>(*this), other);
    }
}
inline Tensor Tensor::mul(Scalar other) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::mul", "Scalar"}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &, Scalar>(op, const_cast<Tensor&>(*this), other);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &, Scalar>(const_cast<Tensor&>(*this), other);
    }
}
inline Tensor & Tensor::mul_(Scalar other) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::mul_", "Scalar"}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor &, Tensor &, Scalar>(op, const_cast<Tensor&>(*this), other);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor &, Tensor &, Scalar>(const_cast<Tensor&>(*this), other);
    }
}
inline Tensor Tensor::mv(const Tensor & vec) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::mv", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &, const Tensor &>(op, const_cast<Tensor&>(*this), vec);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), vec);
    }
}
inline Tensor Tensor::mvlgamma(int64_t p) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::mvlgamma", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &, int64_t>(op, const_cast<Tensor&>(*this), p);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &, int64_t>(const_cast<Tensor&>(*this), p);
    }
}
inline Tensor & Tensor::mvlgamma_(int64_t p) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::mvlgamma_", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor &, Tensor &, int64_t>(op, const_cast<Tensor&>(*this), p);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor &, Tensor &, int64_t>(const_cast<Tensor&>(*this), p);
    }
}
inline Tensor Tensor::narrow_copy(int64_t dim, int64_t start, int64_t length) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::narrow_copy", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &, int64_t, int64_t, int64_t>(op, const_cast<Tensor&>(*this), dim, start, length);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &, int64_t, int64_t, int64_t>(const_cast<Tensor&>(*this), dim, start, length);
    }
}
inline Tensor Tensor::narrow(int64_t dim, int64_t start, int64_t length) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::narrow", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &, int64_t, int64_t, int64_t>(op, const_cast<Tensor&>(*this), dim, start, length);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &, int64_t, int64_t, int64_t>(const_cast<Tensor&>(*this), dim, start, length);
    }
}
inline Tensor Tensor::permute(IntArrayRef dims) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::permute", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &, IntArrayRef>(op, const_cast<Tensor&>(*this), dims);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &, IntArrayRef>(const_cast<Tensor&>(*this), dims);
    }
}
inline Tensor Tensor::numpy_T() const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::numpy_T", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &>(op, const_cast<Tensor&>(*this));
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
    }
}
inline bool Tensor::is_pinned() const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::is_pinned", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<bool, const Tensor &>(op, const_cast<Tensor&>(*this));
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<bool, const Tensor &>(const_cast<Tensor&>(*this));
    }
}
inline Tensor Tensor::pin_memory() const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::pin_memory", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &>(op, const_cast<Tensor&>(*this));
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
    }
}
inline Tensor Tensor::pinverse(double rcond) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::pinverse", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &, double>(op, const_cast<Tensor&>(*this), rcond);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &, double>(const_cast<Tensor&>(*this), rcond);
    }
}
inline Tensor Tensor::reciprocal() const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::reciprocal", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &>(op, const_cast<Tensor&>(*this));
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
    }
}
inline Tensor & Tensor::reciprocal_() const {
    static auto table = globalATenDispatch().getOpTable("aten::reciprocal_(Tensor(a!) self) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this));
}
inline Tensor Tensor::neg() const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::neg", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &>(op, const_cast<Tensor&>(*this));
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
    }
}
inline Tensor & Tensor::neg_() const {
    static auto table = globalATenDispatch().getOpTable("aten::neg_(Tensor(a!) self) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this));
}
inline Tensor Tensor::repeat(IntArrayRef repeats) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::repeat", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &, IntArrayRef>(op, const_cast<Tensor&>(*this), repeats);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &, IntArrayRef>(const_cast<Tensor&>(*this), repeats);
    }
}
inline Tensor Tensor::repeat_interleave(const Tensor & repeats, c10::optional<int64_t> dim) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::repeat_interleave", "self_Tensor"}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &, const Tensor &, c10::optional<int64_t>>(op, const_cast<Tensor&>(*this), repeats, dim);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &, const Tensor &, c10::optional<int64_t>>(const_cast<Tensor&>(*this), repeats, dim);
    }
}
inline Tensor Tensor::repeat_interleave(int64_t repeats, c10::optional<int64_t> dim) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::repeat_interleave", "self_int"}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &, int64_t, c10::optional<int64_t>>(op, const_cast<Tensor&>(*this), repeats, dim);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &, int64_t, c10::optional<int64_t>>(const_cast<Tensor&>(*this), repeats, dim);
    }
}
inline Tensor Tensor::reshape(IntArrayRef shape) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::reshape", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &, IntArrayRef>(op, const_cast<Tensor&>(*this), shape);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &, IntArrayRef>(const_cast<Tensor&>(*this), shape);
    }
}
inline Tensor Tensor::reshape_as(const Tensor & other) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::reshape_as", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &, const Tensor &>(op, const_cast<Tensor&>(*this), other);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), other);
    }
}
inline Tensor Tensor::round() const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::round", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &>(op, const_cast<Tensor&>(*this));
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
    }
}
inline Tensor & Tensor::round_() const {
    static auto table = globalATenDispatch().getOpTable("aten::round_(Tensor(a!) self) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this));
}
inline Tensor Tensor::relu() const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::relu", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &>(op, const_cast<Tensor&>(*this));
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
    }
}
inline Tensor & Tensor::relu_() const {
    static auto table = globalATenDispatch().getOpTable("aten::relu_(Tensor(a!) self) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this));
}
inline Tensor Tensor::prelu(const Tensor & weight) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::prelu", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &, const Tensor &>(op, const_cast<Tensor&>(*this), weight);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), weight);
    }
}
inline std::tuple<Tensor,Tensor> Tensor::prelu_backward(const Tensor & grad_output, const Tensor & weight) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::prelu_backward", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<std::tuple<Tensor,Tensor>, const Tensor &, const Tensor &, const Tensor &>(op, grad_output, const_cast<Tensor&>(*this), weight);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<std::tuple<Tensor,Tensor>, const Tensor &, const Tensor &, const Tensor &>(grad_output, const_cast<Tensor&>(*this), weight);
    }
}
inline Tensor Tensor::hardshrink(Scalar lambd) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::hardshrink", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &, Scalar>(op, const_cast<Tensor&>(*this), lambd);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &, Scalar>(const_cast<Tensor&>(*this), lambd);
    }
}
inline Tensor Tensor::hardshrink_backward(const Tensor & grad_out, Scalar lambd) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::hardshrink_backward", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &, const Tensor &, Scalar>(op, grad_out, const_cast<Tensor&>(*this), lambd);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &, const Tensor &, Scalar>(grad_out, const_cast<Tensor&>(*this), lambd);
    }
}
inline Tensor Tensor::rsqrt() const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::rsqrt", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &>(op, const_cast<Tensor&>(*this));
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
    }
}
inline Tensor & Tensor::rsqrt_() const {
    static auto table = globalATenDispatch().getOpTable("aten::rsqrt_(Tensor(a!) self) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this));
}
#ifdef BUILD_NAMEDTENSOR
inline Tensor Tensor::select(Dimname dim, int64_t index) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::select", "Dimname"}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &, Dimname, int64_t>(op, const_cast<Tensor&>(*this), dim, index);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &, Dimname, int64_t>(const_cast<Tensor&>(*this), dim, index);
    }
}
#endif
inline Tensor Tensor::select(int64_t dim, int64_t index) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::select", "int"}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &, int64_t, int64_t>(op, const_cast<Tensor&>(*this), dim, index);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &, int64_t, int64_t>(const_cast<Tensor&>(*this), dim, index);
    }
}
inline Tensor Tensor::sigmoid() const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::sigmoid", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &>(op, const_cast<Tensor&>(*this));
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
    }
}
inline Tensor & Tensor::sigmoid_() const {
    static auto table = globalATenDispatch().getOpTable("aten::sigmoid_(Tensor(a!) self) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this));
}
inline Tensor Tensor::sin() const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::sin", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &>(op, const_cast<Tensor&>(*this));
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
    }
}
inline Tensor & Tensor::sin_() const {
    static auto table = globalATenDispatch().getOpTable("aten::sin_(Tensor(a!) self) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this));
}
inline Tensor Tensor::sinh() const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::sinh", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &>(op, const_cast<Tensor&>(*this));
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
    }
}
inline Tensor & Tensor::sinh_() const {
    static auto table = globalATenDispatch().getOpTable("aten::sinh_(Tensor(a!) self) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this));
}
inline Tensor Tensor::detach() const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::detach", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &>(op, const_cast<Tensor&>(*this));
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
    }
}
inline Tensor & Tensor::detach_() const {
    static auto table = globalATenDispatch().getOpTable("aten::detach_(Tensor(a!) self) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this));
}
inline int64_t Tensor::size(int64_t dim) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::size", "int"}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<int64_t, const Tensor &, int64_t>(op, const_cast<Tensor&>(*this), dim);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<int64_t, const Tensor &, int64_t>(const_cast<Tensor&>(*this), dim);
    }
}
#ifdef BUILD_NAMEDTENSOR
inline int64_t Tensor::size(Dimname dim) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::size", "Dimname"}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<int64_t, const Tensor &, Dimname>(op, const_cast<Tensor&>(*this), dim);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<int64_t, const Tensor &, Dimname>(const_cast<Tensor&>(*this), dim);
    }
}
#endif
inline Tensor Tensor::slice(int64_t dim, int64_t start, int64_t end, int64_t step) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::slice", "Tensor"}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &, int64_t, int64_t, int64_t, int64_t>(op, const_cast<Tensor&>(*this), dim, start, end, step);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &, int64_t, int64_t, int64_t, int64_t>(const_cast<Tensor&>(*this), dim, start, end, step);
    }
}
inline std::tuple<Tensor,Tensor> Tensor::slogdet() const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::slogdet", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<std::tuple<Tensor,Tensor>, const Tensor &>(op, const_cast<Tensor&>(*this));
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<std::tuple<Tensor,Tensor>, const Tensor &>(const_cast<Tensor&>(*this));
    }
}
inline Tensor Tensor::smm(const Tensor & mat2) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::smm", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &, const Tensor &>(op, const_cast<Tensor&>(*this), mat2);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), mat2);
    }
}
inline Tensor Tensor::softmax(int64_t dim, c10::optional<ScalarType> dtype) const {
    static auto table = globalATenDispatch().getOpTable("aten::softmax(Tensor self, int dim, ScalarType? dtype=None) -> Tensor");
    return table->getOp<Tensor (const Tensor &, int64_t, c10::optional<ScalarType>)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), dim, dtype);
}
inline std::vector<Tensor> Tensor::split(int64_t split_size, int64_t dim) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::split", "Tensor"}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<std::vector<Tensor>, const Tensor &, int64_t, int64_t>(op, const_cast<Tensor&>(*this), split_size, dim);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<std::vector<Tensor>, const Tensor &, int64_t, int64_t>(const_cast<Tensor&>(*this), split_size, dim);
    }
}
inline std::vector<Tensor> Tensor::split_with_sizes(IntArrayRef split_sizes, int64_t dim) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::split_with_sizes", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<std::vector<Tensor>, const Tensor &, IntArrayRef, int64_t>(op, const_cast<Tensor&>(*this), split_sizes, dim);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<std::vector<Tensor>, const Tensor &, IntArrayRef, int64_t>(const_cast<Tensor&>(*this), split_sizes, dim);
    }
}
inline Tensor Tensor::squeeze() const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::squeeze", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &>(op, const_cast<Tensor&>(*this));
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
    }
}
inline Tensor Tensor::squeeze(int64_t dim) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::squeeze", "dim"}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &, int64_t>(op, const_cast<Tensor&>(*this), dim);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &, int64_t>(const_cast<Tensor&>(*this), dim);
    }
}
inline Tensor & Tensor::squeeze_() const {
    static auto table = globalATenDispatch().getOpTable("aten::squeeze_(Tensor(a!) self) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this));
}
inline Tensor & Tensor::squeeze_(int64_t dim) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::squeeze_", "dim"}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor &, Tensor &, int64_t>(op, const_cast<Tensor&>(*this), dim);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor &, Tensor &, int64_t>(const_cast<Tensor&>(*this), dim);
    }
}
inline Tensor Tensor::sspaddmm(const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::sspaddmm", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar>(op, const_cast<Tensor&>(*this), mat1, mat2, beta, alpha);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar>(const_cast<Tensor&>(*this), mat1, mat2, beta, alpha);
    }
}
inline Tensor Tensor::stft(int64_t n_fft, c10::optional<int64_t> hop_length, c10::optional<int64_t> win_length, const Tensor & window, bool normalized, bool onesided) const {
    static auto table = globalATenDispatch().getOpTable("aten::stft(Tensor self, int n_fft, int? hop_length=None, int? win_length=None, Tensor? window=None, bool normalized=False, bool onesided=True) -> Tensor");
    return table->getOp<Tensor (const Tensor &, int64_t, c10::optional<int64_t>, c10::optional<int64_t>, const Tensor &, bool, bool)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), n_fft, hop_length, win_length, window, normalized, onesided);
}
inline int64_t Tensor::stride(int64_t dim) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::stride", "int"}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<int64_t, const Tensor &, int64_t>(op, const_cast<Tensor&>(*this), dim);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<int64_t, const Tensor &, int64_t>(const_cast<Tensor&>(*this), dim);
    }
}
#ifdef BUILD_NAMEDTENSOR
inline int64_t Tensor::stride(Dimname dim) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::stride", "Dimname"}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<int64_t, const Tensor &, Dimname>(op, const_cast<Tensor&>(*this), dim);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<int64_t, const Tensor &, Dimname>(const_cast<Tensor&>(*this), dim);
    }
}
#endif
inline Tensor Tensor::sum(c10::optional<ScalarType> dtype) const {
    static auto table = globalATenDispatch().getOpTable("aten::sum(Tensor self, *, ScalarType? dtype=None) -> Tensor");
    return table->getOp<Tensor (const Tensor &, c10::optional<ScalarType>)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), dtype);
}
inline Tensor Tensor::sum(IntArrayRef dim, bool keepdim, c10::optional<ScalarType> dtype) const {
    static auto table = globalATenDispatch().getOpTable("aten::sum.dim_IntList(Tensor self, int[1] dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor");
    return table->getOp<Tensor (const Tensor &, IntArrayRef, bool, c10::optional<ScalarType>)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), dim, keepdim, dtype);
}
#ifdef BUILD_NAMEDTENSOR
inline Tensor Tensor::sum(DimnameList dim, bool keepdim, c10::optional<ScalarType> dtype) const {
    static auto table = globalATenDispatch().getOpTable("aten::sum.dim_DimnameList(Tensor self, Dimname[1] dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor");
    return table->getOp<Tensor (const Tensor &, DimnameList, bool, c10::optional<ScalarType>)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), dim, keepdim, dtype);
}
#endif
inline Tensor Tensor::sum_to_size(IntArrayRef size) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::sum_to_size", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &, IntArrayRef>(op, const_cast<Tensor&>(*this), size);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &, IntArrayRef>(const_cast<Tensor&>(*this), size);
    }
}
inline Tensor Tensor::sqrt() const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::sqrt", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &>(op, const_cast<Tensor&>(*this));
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
    }
}
inline Tensor & Tensor::sqrt_() const {
    static auto table = globalATenDispatch().getOpTable("aten::sqrt_(Tensor(a!) self) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this));
}
inline Tensor Tensor::std(bool unbiased) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::std", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &, bool>(op, const_cast<Tensor&>(*this), unbiased);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &, bool>(const_cast<Tensor&>(*this), unbiased);
    }
}
inline Tensor Tensor::std(IntArrayRef dim, bool unbiased, bool keepdim) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::std", "dim"}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &, IntArrayRef, bool, bool>(op, const_cast<Tensor&>(*this), dim, unbiased, keepdim);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &, IntArrayRef, bool, bool>(const_cast<Tensor&>(*this), dim, unbiased, keepdim);
    }
}
inline Tensor Tensor::prod(c10::optional<ScalarType> dtype) const {
    static auto table = globalATenDispatch().getOpTable("aten::prod(Tensor self, *, ScalarType? dtype=None) -> Tensor");
    return table->getOp<Tensor (const Tensor &, c10::optional<ScalarType>)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), dtype);
}
inline Tensor Tensor::prod(int64_t dim, bool keepdim, c10::optional<ScalarType> dtype) const {
    static auto table = globalATenDispatch().getOpTable("aten::prod.dim_int(Tensor self, int dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor");
    return table->getOp<Tensor (const Tensor &, int64_t, bool, c10::optional<ScalarType>)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), dim, keepdim, dtype);
}
#ifdef BUILD_NAMEDTENSOR
inline Tensor Tensor::prod(Dimname dim, bool keepdim, c10::optional<ScalarType> dtype) const {
    static auto table = globalATenDispatch().getOpTable("aten::prod.dim_Dimname(Tensor self, Dimname dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor");
    return table->getOp<Tensor (const Tensor &, Dimname, bool, c10::optional<ScalarType>)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), dim, keepdim, dtype);
}
#endif
inline Tensor Tensor::t() const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::t", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &>(op, const_cast<Tensor&>(*this));
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
    }
}
inline Tensor & Tensor::t_() const {
    static auto table = globalATenDispatch().getOpTable("aten::t_(Tensor(a!) self) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this));
}
inline Tensor Tensor::tan() const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::tan", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &>(op, const_cast<Tensor&>(*this));
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
    }
}
inline Tensor & Tensor::tan_() const {
    static auto table = globalATenDispatch().getOpTable("aten::tan_(Tensor(a!) self) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this));
}
inline Tensor Tensor::tanh() const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::tanh", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &>(op, const_cast<Tensor&>(*this));
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
    }
}
inline Tensor & Tensor::tanh_() const {
    static auto table = globalATenDispatch().getOpTable("aten::tanh_(Tensor(a!) self) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this));
}
inline Tensor Tensor::transpose(int64_t dim0, int64_t dim1) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::transpose", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &, int64_t, int64_t>(op, const_cast<Tensor&>(*this), dim0, dim1);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &, int64_t, int64_t>(const_cast<Tensor&>(*this), dim0, dim1);
    }
}
inline Tensor & Tensor::transpose_(int64_t dim0, int64_t dim1) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::transpose_", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor &, Tensor &, int64_t, int64_t>(op, const_cast<Tensor&>(*this), dim0, dim1);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor &, Tensor &, int64_t, int64_t>(const_cast<Tensor&>(*this), dim0, dim1);
    }
}
inline Tensor Tensor::flip(IntArrayRef dims) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::flip", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &, IntArrayRef>(op, const_cast<Tensor&>(*this), dims);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &, IntArrayRef>(const_cast<Tensor&>(*this), dims);
    }
}
inline Tensor Tensor::roll(IntArrayRef shifts, IntArrayRef dims) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::roll", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &, IntArrayRef, IntArrayRef>(op, const_cast<Tensor&>(*this), shifts, dims);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &, IntArrayRef, IntArrayRef>(const_cast<Tensor&>(*this), shifts, dims);
    }
}
inline Tensor Tensor::rot90(int64_t k, IntArrayRef dims) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::rot90", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &, int64_t, IntArrayRef>(op, const_cast<Tensor&>(*this), k, dims);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &, int64_t, IntArrayRef>(const_cast<Tensor&>(*this), k, dims);
    }
}
inline Tensor Tensor::trunc() const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::trunc", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &>(op, const_cast<Tensor&>(*this));
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
    }
}
inline Tensor & Tensor::trunc_() const {
    static auto table = globalATenDispatch().getOpTable("aten::trunc_(Tensor(a!) self) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this));
}
inline Tensor Tensor::type_as(const Tensor & other) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::type_as", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &, const Tensor &>(op, const_cast<Tensor&>(*this), other);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), other);
    }
}
inline Tensor Tensor::unsqueeze(int64_t dim) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::unsqueeze", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &, int64_t>(op, const_cast<Tensor&>(*this), dim);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &, int64_t>(const_cast<Tensor&>(*this), dim);
    }
}
inline Tensor & Tensor::unsqueeze_(int64_t dim) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::unsqueeze_", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor &, Tensor &, int64_t>(op, const_cast<Tensor&>(*this), dim);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor &, Tensor &, int64_t>(const_cast<Tensor&>(*this), dim);
    }
}
inline Tensor Tensor::var(bool unbiased) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::var", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &, bool>(op, const_cast<Tensor&>(*this), unbiased);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &, bool>(const_cast<Tensor&>(*this), unbiased);
    }
}
inline Tensor Tensor::var(IntArrayRef dim, bool unbiased, bool keepdim) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::var", "dim"}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &, IntArrayRef, bool, bool>(op, const_cast<Tensor&>(*this), dim, unbiased, keepdim);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &, IntArrayRef, bool, bool>(const_cast<Tensor&>(*this), dim, unbiased, keepdim);
    }
}
inline Tensor Tensor::view_as(const Tensor & other) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::view_as", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &, const Tensor &>(op, const_cast<Tensor&>(*this), other);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), other);
    }
}
inline Tensor Tensor::where(const Tensor & condition, const Tensor & other) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::where", "self"}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &, const Tensor &, const Tensor &>(op, condition, const_cast<Tensor&>(*this), other);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &, const Tensor &, const Tensor &>(condition, const_cast<Tensor&>(*this), other);
    }
}
inline Tensor Tensor::norm(c10::optional<Scalar> p, ScalarType dtype) const {
    static auto table = globalATenDispatch().getOpTable("aten::norm.ScalarOpt_dtype(Tensor self, Scalar? p, *, ScalarType dtype) -> Tensor");
    return table->getOp<Tensor (const Tensor &, c10::optional<Scalar>, ScalarType)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), p, dtype);
}
inline Tensor Tensor::norm(Scalar p) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::norm", "Scalar"}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &, Scalar>(op, const_cast<Tensor&>(*this), p);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &, Scalar>(const_cast<Tensor&>(*this), p);
    }
}
inline Tensor Tensor::norm(c10::optional<Scalar> p, IntArrayRef dim, bool keepdim, ScalarType dtype) const {
    static auto table = globalATenDispatch().getOpTable("aten::norm.ScalarOpt_dim_dtype(Tensor self, Scalar? p, int[1] dim, bool keepdim, *, ScalarType dtype) -> Tensor");
    return table->getOp<Tensor (const Tensor &, c10::optional<Scalar>, IntArrayRef, bool, ScalarType)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), p, dim, keepdim, dtype);
}
inline Tensor Tensor::norm(c10::optional<Scalar> p, IntArrayRef dim, bool keepdim) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::norm", "ScalarOpt_dim"}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &, c10::optional<Scalar>, IntArrayRef, bool>(op, const_cast<Tensor&>(*this), p, dim, keepdim);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &, c10::optional<Scalar>, IntArrayRef, bool>(const_cast<Tensor&>(*this), p, dim, keepdim);
    }
}
inline Tensor Tensor::clone() const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::clone", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &>(op, const_cast<Tensor&>(*this));
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
    }
}
inline Tensor & Tensor::resize_as_(const Tensor & the_template) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::resize_as_", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor &, Tensor &, const Tensor &>(op, const_cast<Tensor&>(*this), the_template);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor &, Tensor &, const Tensor &>(const_cast<Tensor&>(*this), the_template);
    }
}
inline Tensor Tensor::pow(Scalar exponent) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::pow", "Tensor_Scalar"}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &, Scalar>(op, const_cast<Tensor&>(*this), exponent);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &, Scalar>(const_cast<Tensor&>(*this), exponent);
    }
}
inline Tensor & Tensor::zero_() const {
    static auto table = globalATenDispatch().getOpTable("aten::zero_(Tensor(a!) self) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this));
}
inline Tensor Tensor::sub(const Tensor & other, Scalar alpha) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::sub", "Tensor"}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &, const Tensor &, Scalar>(op, const_cast<Tensor&>(*this), other, alpha);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &, const Tensor &, Scalar>(const_cast<Tensor&>(*this), other, alpha);
    }
}
inline Tensor & Tensor::sub_(const Tensor & other, Scalar alpha) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::sub_", "Tensor"}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor &, Tensor &, const Tensor &, Scalar>(op, const_cast<Tensor&>(*this), other, alpha);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor &, Tensor &, const Tensor &, Scalar>(const_cast<Tensor&>(*this), other, alpha);
    }
}
inline Tensor Tensor::sub(Scalar other, Scalar alpha) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::sub", "Scalar"}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &, Scalar, Scalar>(op, const_cast<Tensor&>(*this), other, alpha);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &, Scalar, Scalar>(const_cast<Tensor&>(*this), other, alpha);
    }
}
inline Tensor & Tensor::sub_(Scalar other, Scalar alpha) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::sub_", "Scalar"}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor &, Tensor &, Scalar, Scalar>(op, const_cast<Tensor&>(*this), other, alpha);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor &, Tensor &, Scalar, Scalar>(const_cast<Tensor&>(*this), other, alpha);
    }
}
inline Tensor Tensor::addmm(const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::addmm", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar>(op, const_cast<Tensor&>(*this), mat1, mat2, beta, alpha);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar>(const_cast<Tensor&>(*this), mat1, mat2, beta, alpha);
    }
}
inline Tensor & Tensor::addmm_(const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::addmm_", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor &, Tensor &, const Tensor &, const Tensor &, Scalar, Scalar>(op, const_cast<Tensor&>(*this), mat1, mat2, beta, alpha);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor &, Tensor &, const Tensor &, const Tensor &, Scalar, Scalar>(const_cast<Tensor&>(*this), mat1, mat2, beta, alpha);
    }
}
inline Tensor & Tensor::sparse_resize_(IntArrayRef size, int64_t sparse_dim, int64_t dense_dim) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::sparse_resize_", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor &, Tensor &, IntArrayRef, int64_t, int64_t>(op, const_cast<Tensor&>(*this), size, sparse_dim, dense_dim);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor &, Tensor &, IntArrayRef, int64_t, int64_t>(const_cast<Tensor&>(*this), size, sparse_dim, dense_dim);
    }
}
inline Tensor & Tensor::sparse_resize_and_clear_(IntArrayRef size, int64_t sparse_dim, int64_t dense_dim) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::sparse_resize_and_clear_", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor &, Tensor &, IntArrayRef, int64_t, int64_t>(op, const_cast<Tensor&>(*this), size, sparse_dim, dense_dim);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor &, Tensor &, IntArrayRef, int64_t, int64_t>(const_cast<Tensor&>(*this), size, sparse_dim, dense_dim);
    }
}
inline Tensor Tensor::sparse_mask(const Tensor & mask) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::sparse_mask", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &, const Tensor &>(op, const_cast<Tensor&>(*this), mask);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), mask);
    }
}
inline Tensor Tensor::to_dense() const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::to_dense", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &>(op, const_cast<Tensor&>(*this));
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
    }
}
inline int64_t Tensor::sparse_dim() const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::sparse_dim", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<int64_t, const Tensor &>(op, const_cast<Tensor&>(*this));
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<int64_t, const Tensor &>(const_cast<Tensor&>(*this));
    }
}
inline int64_t Tensor::_dimI() const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::_dimI", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<int64_t, const Tensor &>(op, const_cast<Tensor&>(*this));
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<int64_t, const Tensor &>(const_cast<Tensor&>(*this));
    }
}
inline int64_t Tensor::dense_dim() const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::dense_dim", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<int64_t, const Tensor &>(op, const_cast<Tensor&>(*this));
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<int64_t, const Tensor &>(const_cast<Tensor&>(*this));
    }
}
inline int64_t Tensor::_dimV() const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::_dimV", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<int64_t, const Tensor &>(op, const_cast<Tensor&>(*this));
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<int64_t, const Tensor &>(const_cast<Tensor&>(*this));
    }
}
inline int64_t Tensor::_nnz() const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::_nnz", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<int64_t, const Tensor &>(op, const_cast<Tensor&>(*this));
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<int64_t, const Tensor &>(const_cast<Tensor&>(*this));
    }
}
inline Tensor Tensor::coalesce() const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::coalesce", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &>(op, const_cast<Tensor&>(*this));
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
    }
}
inline bool Tensor::is_coalesced() const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::is_coalesced", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<bool, const Tensor &>(op, const_cast<Tensor&>(*this));
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<bool, const Tensor &>(const_cast<Tensor&>(*this));
    }
}
inline Tensor Tensor::_indices() const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::_indices", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &>(op, const_cast<Tensor&>(*this));
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
    }
}
inline Tensor Tensor::_values() const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::_values", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &>(op, const_cast<Tensor&>(*this));
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
    }
}
inline Tensor & Tensor::_coalesced_(bool coalesced) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::_coalesced_", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor &, Tensor &, bool>(op, const_cast<Tensor&>(*this), coalesced);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor &, Tensor &, bool>(const_cast<Tensor&>(*this), coalesced);
    }
}
inline Tensor Tensor::indices() const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::indices", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &>(op, const_cast<Tensor&>(*this));
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
    }
}
inline Tensor Tensor::values() const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::values", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &>(op, const_cast<Tensor&>(*this));
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
    }
}
inline int64_t Tensor::numel() const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::numel", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<int64_t, const Tensor &>(op, const_cast<Tensor&>(*this));
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<int64_t, const Tensor &>(const_cast<Tensor&>(*this));
    }
}
inline std::vector<Tensor> Tensor::unbind(int64_t dim) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::unbind", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<std::vector<Tensor>, const Tensor &, int64_t>(op, const_cast<Tensor&>(*this), dim);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<std::vector<Tensor>, const Tensor &, int64_t>(const_cast<Tensor&>(*this), dim);
    }
}
inline Tensor Tensor::to_sparse(int64_t sparse_dim) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::to_sparse", "sparse_dim"}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &, int64_t>(op, const_cast<Tensor&>(*this), sparse_dim);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &, int64_t>(const_cast<Tensor&>(*this), sparse_dim);
    }
}
inline Tensor Tensor::to_sparse() const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::to_sparse", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &>(op, const_cast<Tensor&>(*this));
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
    }
}
inline Tensor Tensor::to_mkldnn() const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::to_mkldnn", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &>(op, const_cast<Tensor&>(*this));
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
    }
}
inline Tensor Tensor::dequantize() const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::dequantize", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &>(op, const_cast<Tensor&>(*this));
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
    }
}
inline double Tensor::q_scale() const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::q_scale", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<double, const Tensor &>(op, const_cast<Tensor&>(*this));
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<double, const Tensor &>(const_cast<Tensor&>(*this));
    }
}
inline int64_t Tensor::q_zero_point() const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::q_zero_point", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<int64_t, const Tensor &>(op, const_cast<Tensor&>(*this));
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<int64_t, const Tensor &>(const_cast<Tensor&>(*this));
    }
}
inline Tensor Tensor::int_repr() const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::int_repr", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &>(op, const_cast<Tensor&>(*this));
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
    }
}
inline QScheme Tensor::qscheme() const {
    static auto table = globalATenDispatch().getOpTable("aten::qscheme(Tensor self) -> QScheme");
    return table->getOp<QScheme (const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this));
}
inline Tensor Tensor::to(const TensorOptions & options, bool non_blocking, bool copy) const {
    static auto table = globalATenDispatch().getOpTable("aten::to.dtype_layout(Tensor self, *, ScalarType dtype, Layout layout, Device device, bool pin_memory=False, bool non_blocking=False, bool copy=False) -> Tensor");
    return table->getOp<Tensor (const Tensor &, const TensorOptions &, bool, bool)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), options, non_blocking, copy);
}
inline Tensor Tensor::to(Device device, ScalarType dtype, bool non_blocking, bool copy) const {
    static auto table = globalATenDispatch().getOpTable("aten::to.device(Tensor self, Device device, ScalarType dtype, bool non_blocking=False, bool copy=False) -> Tensor");
    return table->getOp<Tensor (const Tensor &, Device, ScalarType, bool, bool)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), device, dtype, non_blocking, copy);
}
inline Tensor Tensor::to(ScalarType dtype, bool non_blocking, bool copy) const {
    static auto table = globalATenDispatch().getOpTable("aten::to.dtype(Tensor self, ScalarType dtype, bool non_blocking=False, bool copy=False) -> Tensor");
    return table->getOp<Tensor (const Tensor &, ScalarType, bool, bool)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), dtype, non_blocking, copy);
}
inline Tensor Tensor::to(const Tensor & other, bool non_blocking, bool copy) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::to", "other"}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &, const Tensor &, bool, bool>(op, const_cast<Tensor&>(*this), other, non_blocking, copy);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &, const Tensor &, bool, bool>(const_cast<Tensor&>(*this), other, non_blocking, copy);
    }
}
inline Scalar Tensor::item() const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::item", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Scalar, const Tensor &>(op, const_cast<Tensor&>(*this));
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Scalar, const Tensor &>(const_cast<Tensor&>(*this));
    }
}
inline Tensor & Tensor::set_(Storage source) const {
    static auto table = globalATenDispatch().getOpTable("aten::set_.source_Storage(Tensor(a!) self, Storage source) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, Storage)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), source);
}
inline Tensor & Tensor::set_(Storage source, int64_t storage_offset, IntArrayRef size, IntArrayRef stride) const {
    static auto table = globalATenDispatch().getOpTable("aten::set_.source_Storage_storage_offset(Tensor(a!) self, Storage source, int storage_offset, int[] size, int[] stride=[]) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, Storage, int64_t, IntArrayRef, IntArrayRef)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), source, storage_offset, size, stride);
}
inline Tensor & Tensor::set_(const Tensor & source) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::set_", "source_Tensor"}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor &, Tensor &, const Tensor &>(op, const_cast<Tensor&>(*this), source);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor &, Tensor &, const Tensor &>(const_cast<Tensor&>(*this), source);
    }
}
inline Tensor & Tensor::set_() const {
    static auto table = globalATenDispatch().getOpTable("aten::set_(Tensor(a!) self) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this));
}
inline Tensor & Tensor::set_quantizer_(ConstQuantizerPtr quantizer) const {
    static auto table = globalATenDispatch().getOpTable("aten::set_quantizer_(Tensor(a!) self, ConstQuantizerPtr quantizer) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, ConstQuantizerPtr)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), quantizer);
}
inline bool Tensor::is_set_to(const Tensor & tensor) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::is_set_to", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<bool, const Tensor &, const Tensor &>(op, const_cast<Tensor&>(*this), tensor);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<bool, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), tensor);
    }
}
inline Tensor & Tensor::masked_fill_(const Tensor & mask, Scalar value) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::masked_fill_", "Scalar"}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor &, Tensor &, const Tensor &, Scalar>(op, const_cast<Tensor&>(*this), mask, value);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor &, Tensor &, const Tensor &, Scalar>(const_cast<Tensor&>(*this), mask, value);
    }
}
inline Tensor Tensor::masked_fill(const Tensor & mask, Scalar value) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::masked_fill", "Scalar"}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &, const Tensor &, Scalar>(op, const_cast<Tensor&>(*this), mask, value);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &, const Tensor &, Scalar>(const_cast<Tensor&>(*this), mask, value);
    }
}
inline Tensor & Tensor::masked_fill_(const Tensor & mask, const Tensor & value) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::masked_fill_", "Tensor"}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor &, Tensor &, const Tensor &, const Tensor &>(op, const_cast<Tensor&>(*this), mask, value);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor &, Tensor &, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), mask, value);
    }
}
inline Tensor Tensor::masked_fill(const Tensor & mask, const Tensor & value) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::masked_fill", "Tensor"}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &, const Tensor &, const Tensor &>(op, const_cast<Tensor&>(*this), mask, value);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), mask, value);
    }
}
inline Tensor & Tensor::masked_scatter_(const Tensor & mask, const Tensor & source) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::masked_scatter_", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor &, Tensor &, const Tensor &, const Tensor &>(op, const_cast<Tensor&>(*this), mask, source);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor &, Tensor &, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), mask, source);
    }
}
inline Tensor Tensor::masked_scatter(const Tensor & mask, const Tensor & source) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::masked_scatter", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &, const Tensor &, const Tensor &>(op, const_cast<Tensor&>(*this), mask, source);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), mask, source);
    }
}
inline Tensor Tensor::view(IntArrayRef size) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::view", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &, IntArrayRef>(op, const_cast<Tensor&>(*this), size);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &, IntArrayRef>(const_cast<Tensor&>(*this), size);
    }
}
inline Tensor & Tensor::put_(const Tensor & index, const Tensor & source, bool accumulate) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::put_", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor &, Tensor &, const Tensor &, const Tensor &, bool>(op, const_cast<Tensor&>(*this), index, source, accumulate);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor &, Tensor &, const Tensor &, const Tensor &, bool>(const_cast<Tensor&>(*this), index, source, accumulate);
    }
}
inline Tensor & Tensor::index_add_(int64_t dim, const Tensor & index, const Tensor & source) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::index_add_", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor &, Tensor &, int64_t, const Tensor &, const Tensor &>(op, const_cast<Tensor&>(*this), dim, index, source);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor &, Tensor &, int64_t, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), dim, index, source);
    }
}
inline Tensor Tensor::index_add(int64_t dim, const Tensor & index, const Tensor & source) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::index_add", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &, int64_t, const Tensor &, const Tensor &>(op, const_cast<Tensor&>(*this), dim, index, source);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &, int64_t, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), dim, index, source);
    }
}
inline Tensor & Tensor::index_fill_(int64_t dim, const Tensor & index, Scalar value) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::index_fill_", "Scalar"}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor &, Tensor &, int64_t, const Tensor &, Scalar>(op, const_cast<Tensor&>(*this), dim, index, value);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor &, Tensor &, int64_t, const Tensor &, Scalar>(const_cast<Tensor&>(*this), dim, index, value);
    }
}
inline Tensor Tensor::index_fill(int64_t dim, const Tensor & index, Scalar value) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::index_fill", "Scalar"}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &, int64_t, const Tensor &, Scalar>(op, const_cast<Tensor&>(*this), dim, index, value);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &, int64_t, const Tensor &, Scalar>(const_cast<Tensor&>(*this), dim, index, value);
    }
}
inline Tensor & Tensor::index_fill_(int64_t dim, const Tensor & index, const Tensor & value) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::index_fill_", "Tensor"}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor &, Tensor &, int64_t, const Tensor &, const Tensor &>(op, const_cast<Tensor&>(*this), dim, index, value);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor &, Tensor &, int64_t, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), dim, index, value);
    }
}
inline Tensor Tensor::index_fill(int64_t dim, const Tensor & index, const Tensor & value) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::index_fill", "Tensor"}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &, int64_t, const Tensor &, const Tensor &>(op, const_cast<Tensor&>(*this), dim, index, value);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &, int64_t, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), dim, index, value);
    }
}
inline Tensor & Tensor::scatter_(int64_t dim, const Tensor & index, const Tensor & src) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::scatter_", "src"}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor &, Tensor &, int64_t, const Tensor &, const Tensor &>(op, const_cast<Tensor&>(*this), dim, index, src);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor &, Tensor &, int64_t, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), dim, index, src);
    }
}
inline Tensor Tensor::scatter(int64_t dim, const Tensor & index, const Tensor & src) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::scatter", "src"}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &, int64_t, const Tensor &, const Tensor &>(op, const_cast<Tensor&>(*this), dim, index, src);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &, int64_t, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), dim, index, src);
    }
}
inline Tensor & Tensor::scatter_(int64_t dim, const Tensor & index, Scalar value) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::scatter_", "value"}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor &, Tensor &, int64_t, const Tensor &, Scalar>(op, const_cast<Tensor&>(*this), dim, index, value);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor &, Tensor &, int64_t, const Tensor &, Scalar>(const_cast<Tensor&>(*this), dim, index, value);
    }
}
inline Tensor Tensor::scatter(int64_t dim, const Tensor & index, Scalar value) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::scatter", "value"}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &, int64_t, const Tensor &, Scalar>(op, const_cast<Tensor&>(*this), dim, index, value);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &, int64_t, const Tensor &, Scalar>(const_cast<Tensor&>(*this), dim, index, value);
    }
}
inline Tensor & Tensor::scatter_add_(int64_t dim, const Tensor & index, const Tensor & src) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::scatter_add_", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor &, Tensor &, int64_t, const Tensor &, const Tensor &>(op, const_cast<Tensor&>(*this), dim, index, src);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor &, Tensor &, int64_t, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), dim, index, src);
    }
}
inline Tensor Tensor::scatter_add(int64_t dim, const Tensor & index, const Tensor & src) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::scatter_add", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &, int64_t, const Tensor &, const Tensor &>(op, const_cast<Tensor&>(*this), dim, index, src);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &, int64_t, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), dim, index, src);
    }
}
inline Tensor & Tensor::lt_(Scalar other) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::lt_", "Scalar"}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor &, Tensor &, Scalar>(op, const_cast<Tensor&>(*this), other);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor &, Tensor &, Scalar>(const_cast<Tensor&>(*this), other);
    }
}
inline Tensor & Tensor::lt_(const Tensor & other) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::lt_", "Tensor"}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor &, Tensor &, const Tensor &>(op, const_cast<Tensor&>(*this), other);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor &, Tensor &, const Tensor &>(const_cast<Tensor&>(*this), other);
    }
}
inline Tensor & Tensor::gt_(Scalar other) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::gt_", "Scalar"}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor &, Tensor &, Scalar>(op, const_cast<Tensor&>(*this), other);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor &, Tensor &, Scalar>(const_cast<Tensor&>(*this), other);
    }
}
inline Tensor & Tensor::gt_(const Tensor & other) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::gt_", "Tensor"}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor &, Tensor &, const Tensor &>(op, const_cast<Tensor&>(*this), other);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor &, Tensor &, const Tensor &>(const_cast<Tensor&>(*this), other);
    }
}
inline Tensor & Tensor::le_(Scalar other) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::le_", "Scalar"}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor &, Tensor &, Scalar>(op, const_cast<Tensor&>(*this), other);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor &, Tensor &, Scalar>(const_cast<Tensor&>(*this), other);
    }
}
inline Tensor & Tensor::le_(const Tensor & other) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::le_", "Tensor"}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor &, Tensor &, const Tensor &>(op, const_cast<Tensor&>(*this), other);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor &, Tensor &, const Tensor &>(const_cast<Tensor&>(*this), other);
    }
}
inline Tensor & Tensor::ge_(Scalar other) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::ge_", "Scalar"}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor &, Tensor &, Scalar>(op, const_cast<Tensor&>(*this), other);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor &, Tensor &, Scalar>(const_cast<Tensor&>(*this), other);
    }
}
inline Tensor & Tensor::ge_(const Tensor & other) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::ge_", "Tensor"}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor &, Tensor &, const Tensor &>(op, const_cast<Tensor&>(*this), other);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor &, Tensor &, const Tensor &>(const_cast<Tensor&>(*this), other);
    }
}
inline Tensor & Tensor::eq_(Scalar other) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::eq_", "Scalar"}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor &, Tensor &, Scalar>(op, const_cast<Tensor&>(*this), other);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor &, Tensor &, Scalar>(const_cast<Tensor&>(*this), other);
    }
}
inline Tensor & Tensor::eq_(const Tensor & other) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::eq_", "Tensor"}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor &, Tensor &, const Tensor &>(op, const_cast<Tensor&>(*this), other);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor &, Tensor &, const Tensor &>(const_cast<Tensor&>(*this), other);
    }
}
inline Tensor & Tensor::ne_(Scalar other) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::ne_", "Scalar"}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor &, Tensor &, Scalar>(op, const_cast<Tensor&>(*this), other);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor &, Tensor &, Scalar>(const_cast<Tensor&>(*this), other);
    }
}
inline Tensor & Tensor::ne_(const Tensor & other) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::ne_", "Tensor"}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor &, Tensor &, const Tensor &>(op, const_cast<Tensor&>(*this), other);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor &, Tensor &, const Tensor &>(const_cast<Tensor&>(*this), other);
    }
}
inline Tensor Tensor::__and__(Scalar other) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::__and__", "Scalar"}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &, Scalar>(op, const_cast<Tensor&>(*this), other);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &, Scalar>(const_cast<Tensor&>(*this), other);
    }
}
inline Tensor Tensor::__and__(const Tensor & other) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::__and__", "Tensor"}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &, const Tensor &>(op, const_cast<Tensor&>(*this), other);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), other);
    }
}
inline Tensor & Tensor::__iand__(Scalar other) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::__iand__", "Scalar"}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor &, Tensor &, Scalar>(op, const_cast<Tensor&>(*this), other);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor &, Tensor &, Scalar>(const_cast<Tensor&>(*this), other);
    }
}
inline Tensor & Tensor::__iand__(const Tensor & other) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::__iand__", "Tensor"}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor &, Tensor &, const Tensor &>(op, const_cast<Tensor&>(*this), other);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor &, Tensor &, const Tensor &>(const_cast<Tensor&>(*this), other);
    }
}
inline Tensor Tensor::__or__(Scalar other) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::__or__", "Scalar"}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &, Scalar>(op, const_cast<Tensor&>(*this), other);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &, Scalar>(const_cast<Tensor&>(*this), other);
    }
}
inline Tensor Tensor::__or__(const Tensor & other) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::__or__", "Tensor"}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &, const Tensor &>(op, const_cast<Tensor&>(*this), other);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), other);
    }
}
inline Tensor & Tensor::__ior__(Scalar other) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::__ior__", "Scalar"}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor &, Tensor &, Scalar>(op, const_cast<Tensor&>(*this), other);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor &, Tensor &, Scalar>(const_cast<Tensor&>(*this), other);
    }
}
inline Tensor & Tensor::__ior__(const Tensor & other) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::__ior__", "Tensor"}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor &, Tensor &, const Tensor &>(op, const_cast<Tensor&>(*this), other);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor &, Tensor &, const Tensor &>(const_cast<Tensor&>(*this), other);
    }
}
inline Tensor Tensor::__xor__(Scalar other) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::__xor__", "Scalar"}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &, Scalar>(op, const_cast<Tensor&>(*this), other);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &, Scalar>(const_cast<Tensor&>(*this), other);
    }
}
inline Tensor Tensor::__xor__(const Tensor & other) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::__xor__", "Tensor"}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &, const Tensor &>(op, const_cast<Tensor&>(*this), other);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), other);
    }
}
inline Tensor & Tensor::__ixor__(Scalar other) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::__ixor__", "Scalar"}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor &, Tensor &, Scalar>(op, const_cast<Tensor&>(*this), other);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor &, Tensor &, Scalar>(const_cast<Tensor&>(*this), other);
    }
}
inline Tensor & Tensor::__ixor__(const Tensor & other) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::__ixor__", "Tensor"}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor &, Tensor &, const Tensor &>(op, const_cast<Tensor&>(*this), other);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor &, Tensor &, const Tensor &>(const_cast<Tensor&>(*this), other);
    }
}
inline Tensor Tensor::__lshift__(Scalar other) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::__lshift__", "Scalar"}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &, Scalar>(op, const_cast<Tensor&>(*this), other);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &, Scalar>(const_cast<Tensor&>(*this), other);
    }
}
inline Tensor Tensor::__lshift__(const Tensor & other) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::__lshift__", "Tensor"}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &, const Tensor &>(op, const_cast<Tensor&>(*this), other);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), other);
    }
}
inline Tensor & Tensor::__ilshift__(Scalar other) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::__ilshift__", "Scalar"}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor &, Tensor &, Scalar>(op, const_cast<Tensor&>(*this), other);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor &, Tensor &, Scalar>(const_cast<Tensor&>(*this), other);
    }
}
inline Tensor & Tensor::__ilshift__(const Tensor & other) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::__ilshift__", "Tensor"}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor &, Tensor &, const Tensor &>(op, const_cast<Tensor&>(*this), other);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor &, Tensor &, const Tensor &>(const_cast<Tensor&>(*this), other);
    }
}
inline Tensor Tensor::__rshift__(Scalar other) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::__rshift__", "Scalar"}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &, Scalar>(op, const_cast<Tensor&>(*this), other);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &, Scalar>(const_cast<Tensor&>(*this), other);
    }
}
inline Tensor Tensor::__rshift__(const Tensor & other) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::__rshift__", "Tensor"}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &, const Tensor &>(op, const_cast<Tensor&>(*this), other);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), other);
    }
}
inline Tensor & Tensor::__irshift__(Scalar other) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::__irshift__", "Scalar"}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor &, Tensor &, Scalar>(op, const_cast<Tensor&>(*this), other);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor &, Tensor &, Scalar>(const_cast<Tensor&>(*this), other);
    }
}
inline Tensor & Tensor::__irshift__(const Tensor & other) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::__irshift__", "Tensor"}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor &, Tensor &, const Tensor &>(op, const_cast<Tensor&>(*this), other);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor &, Tensor &, const Tensor &>(const_cast<Tensor&>(*this), other);
    }
}
inline Tensor & Tensor::lgamma_() const {
    static auto table = globalATenDispatch().getOpTable("aten::lgamma_(Tensor(a!) self) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this));
}
inline Tensor & Tensor::atan2_(const Tensor & other) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::atan2_", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor &, Tensor &, const Tensor &>(op, const_cast<Tensor&>(*this), other);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor &, Tensor &, const Tensor &>(const_cast<Tensor&>(*this), other);
    }
}
inline Tensor & Tensor::tril_(int64_t diagonal) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::tril_", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor &, Tensor &, int64_t>(op, const_cast<Tensor&>(*this), diagonal);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor &, Tensor &, int64_t>(const_cast<Tensor&>(*this), diagonal);
    }
}
inline Tensor & Tensor::triu_(int64_t diagonal) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::triu_", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor &, Tensor &, int64_t>(op, const_cast<Tensor&>(*this), diagonal);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor &, Tensor &, int64_t>(const_cast<Tensor&>(*this), diagonal);
    }
}
inline Tensor & Tensor::digamma_() const {
    static auto table = globalATenDispatch().getOpTable("aten::digamma_(Tensor(a!) self) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this));
}
inline Tensor & Tensor::polygamma_(int64_t n) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::polygamma_", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor &, Tensor &, int64_t>(op, const_cast<Tensor&>(*this), n);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor &, Tensor &, int64_t>(const_cast<Tensor&>(*this), n);
    }
}
inline Tensor & Tensor::erfinv_() const {
    static auto table = globalATenDispatch().getOpTable("aten::erfinv_(Tensor(a!) self) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this));
}
inline Tensor & Tensor::renorm_(Scalar p, int64_t dim, Scalar maxnorm) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::renorm_", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor &, Tensor &, Scalar, int64_t, Scalar>(op, const_cast<Tensor&>(*this), p, dim, maxnorm);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor &, Tensor &, Scalar, int64_t, Scalar>(const_cast<Tensor&>(*this), p, dim, maxnorm);
    }
}
inline Tensor & Tensor::pow_(Scalar exponent) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::pow_", "Scalar"}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor &, Tensor &, Scalar>(op, const_cast<Tensor&>(*this), exponent);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor &, Tensor &, Scalar>(const_cast<Tensor&>(*this), exponent);
    }
}
inline Tensor & Tensor::pow_(const Tensor & exponent) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::pow_", "Tensor"}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor &, Tensor &, const Tensor &>(op, const_cast<Tensor&>(*this), exponent);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor &, Tensor &, const Tensor &>(const_cast<Tensor&>(*this), exponent);
    }
}
inline Tensor & Tensor::lerp_(const Tensor & end, Scalar weight) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::lerp_", "Scalar"}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor &, Tensor &, const Tensor &, Scalar>(op, const_cast<Tensor&>(*this), end, weight);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor &, Tensor &, const Tensor &, Scalar>(const_cast<Tensor&>(*this), end, weight);
    }
}
inline Tensor & Tensor::lerp_(const Tensor & end, const Tensor & weight) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::lerp_", "Tensor"}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor &, Tensor &, const Tensor &, const Tensor &>(op, const_cast<Tensor&>(*this), end, weight);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor &, Tensor &, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), end, weight);
    }
}
inline Tensor & Tensor::sign_() const {
    static auto table = globalATenDispatch().getOpTable("aten::sign_(Tensor(a!) self) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this));
}
inline Tensor & Tensor::fmod_(Scalar other) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::fmod_", "Scalar"}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor &, Tensor &, Scalar>(op, const_cast<Tensor&>(*this), other);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor &, Tensor &, Scalar>(const_cast<Tensor&>(*this), other);
    }
}
inline Tensor & Tensor::fmod_(const Tensor & other) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::fmod_", "Tensor"}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor &, Tensor &, const Tensor &>(op, const_cast<Tensor&>(*this), other);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor &, Tensor &, const Tensor &>(const_cast<Tensor&>(*this), other);
    }
}
inline Tensor & Tensor::remainder_(Scalar other) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::remainder_", "Scalar"}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor &, Tensor &, Scalar>(op, const_cast<Tensor&>(*this), other);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor &, Tensor &, Scalar>(const_cast<Tensor&>(*this), other);
    }
}
inline Tensor & Tensor::remainder_(const Tensor & other) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::remainder_", "Tensor"}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor &, Tensor &, const Tensor &>(op, const_cast<Tensor&>(*this), other);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor &, Tensor &, const Tensor &>(const_cast<Tensor&>(*this), other);
    }
}
inline Tensor & Tensor::addbmm_(const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::addbmm_", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor &, Tensor &, const Tensor &, const Tensor &, Scalar, Scalar>(op, const_cast<Tensor&>(*this), batch1, batch2, beta, alpha);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor &, Tensor &, const Tensor &, const Tensor &, Scalar, Scalar>(const_cast<Tensor&>(*this), batch1, batch2, beta, alpha);
    }
}
inline Tensor Tensor::addbmm(const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::addbmm", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar>(op, const_cast<Tensor&>(*this), batch1, batch2, beta, alpha);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar>(const_cast<Tensor&>(*this), batch1, batch2, beta, alpha);
    }
}
inline Tensor & Tensor::addcdiv_(const Tensor & tensor1, const Tensor & tensor2, Scalar value) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::addcdiv_", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor &, Tensor &, const Tensor &, const Tensor &, Scalar>(op, const_cast<Tensor&>(*this), tensor1, tensor2, value);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor &, Tensor &, const Tensor &, const Tensor &, Scalar>(const_cast<Tensor&>(*this), tensor1, tensor2, value);
    }
}
inline Tensor & Tensor::random_(int64_t from, int64_t to, Generator * generator) const {
    static auto table = globalATenDispatch().getOpTable("aten::random_.from(Tensor(a!) self, int from, int to, *, Generator? generator=None) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, int64_t, int64_t, Generator *)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), from, to, generator);
}
inline Tensor & Tensor::random_(int64_t to, Generator * generator) const {
    static auto table = globalATenDispatch().getOpTable("aten::random_.to(Tensor(a!) self, int to, *, Generator? generator=None) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, int64_t, Generator *)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), to, generator);
}
inline Tensor & Tensor::random_(Generator * generator) const {
    static auto table = globalATenDispatch().getOpTable("aten::random_(Tensor(a!) self, *, Generator? generator=None) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, Generator *)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), generator);
}
inline Tensor & Tensor::uniform_(double from, double to, Generator * generator) const {
    static auto table = globalATenDispatch().getOpTable("aten::uniform_(Tensor(a!) self, float from=0, float to=1, *, Generator? generator=None) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, double, double, Generator *)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), from, to, generator);
}
inline Tensor & Tensor::normal_(double mean, double std, Generator * generator) const {
    static auto table = globalATenDispatch().getOpTable("aten::normal_(Tensor(a!) self, float mean=0, float std=1, *, Generator? generator=None) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, double, double, Generator *)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), mean, std, generator);
}
inline Tensor & Tensor::cauchy_(double median, double sigma, Generator * generator) const {
    static auto table = globalATenDispatch().getOpTable("aten::cauchy_(Tensor(a!) self, float median=0, float sigma=1, *, Generator? generator=None) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, double, double, Generator *)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), median, sigma, generator);
}
inline Tensor & Tensor::log_normal_(double mean, double std, Generator * generator) const {
    static auto table = globalATenDispatch().getOpTable("aten::log_normal_(Tensor(a!) self, float mean=1, float std=2, *, Generator? generator=None) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, double, double, Generator *)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), mean, std, generator);
}
inline Tensor & Tensor::exponential_(double lambd, Generator * generator) const {
    static auto table = globalATenDispatch().getOpTable("aten::exponential_(Tensor(a!) self, float lambd=1, *, Generator? generator=None) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, double, Generator *)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), lambd, generator);
}
inline Tensor & Tensor::geometric_(double p, Generator * generator) const {
    static auto table = globalATenDispatch().getOpTable("aten::geometric_(Tensor(a!) self, float p, *, Generator? generator=None) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, double, Generator *)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), p, generator);
}
inline Tensor Tensor::diag(int64_t diagonal) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::diag", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &, int64_t>(op, const_cast<Tensor&>(*this), diagonal);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &, int64_t>(const_cast<Tensor&>(*this), diagonal);
    }
}
inline Tensor Tensor::cross(const Tensor & other, c10::optional<int64_t> dim) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::cross", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &, const Tensor &, c10::optional<int64_t>>(op, const_cast<Tensor&>(*this), other, dim);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &, const Tensor &, c10::optional<int64_t>>(const_cast<Tensor&>(*this), other, dim);
    }
}
inline Tensor Tensor::triu(int64_t diagonal) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::triu", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &, int64_t>(op, const_cast<Tensor&>(*this), diagonal);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &, int64_t>(const_cast<Tensor&>(*this), diagonal);
    }
}
inline Tensor Tensor::tril(int64_t diagonal) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::tril", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &, int64_t>(op, const_cast<Tensor&>(*this), diagonal);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &, int64_t>(const_cast<Tensor&>(*this), diagonal);
    }
}
inline Tensor Tensor::trace() const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::trace", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &>(op, const_cast<Tensor&>(*this));
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
    }
}
inline Tensor Tensor::ne(Scalar other) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::ne", "Scalar"}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &, Scalar>(op, const_cast<Tensor&>(*this), other);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &, Scalar>(const_cast<Tensor&>(*this), other);
    }
}
inline Tensor Tensor::ne(const Tensor & other) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::ne", "Tensor"}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &, const Tensor &>(op, const_cast<Tensor&>(*this), other);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), other);
    }
}
inline Tensor Tensor::eq(Scalar other) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::eq", "Scalar"}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &, Scalar>(op, const_cast<Tensor&>(*this), other);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &, Scalar>(const_cast<Tensor&>(*this), other);
    }
}
inline Tensor Tensor::eq(const Tensor & other) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::eq", "Tensor"}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &, const Tensor &>(op, const_cast<Tensor&>(*this), other);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), other);
    }
}
inline Tensor Tensor::ge(Scalar other) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::ge", "Scalar"}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &, Scalar>(op, const_cast<Tensor&>(*this), other);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &, Scalar>(const_cast<Tensor&>(*this), other);
    }
}
inline Tensor Tensor::ge(const Tensor & other) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::ge", "Tensor"}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &, const Tensor &>(op, const_cast<Tensor&>(*this), other);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), other);
    }
}
inline Tensor Tensor::le(Scalar other) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::le", "Scalar"}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &, Scalar>(op, const_cast<Tensor&>(*this), other);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &, Scalar>(const_cast<Tensor&>(*this), other);
    }
}
inline Tensor Tensor::le(const Tensor & other) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::le", "Tensor"}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &, const Tensor &>(op, const_cast<Tensor&>(*this), other);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), other);
    }
}
inline Tensor Tensor::gt(Scalar other) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::gt", "Scalar"}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &, Scalar>(op, const_cast<Tensor&>(*this), other);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &, Scalar>(const_cast<Tensor&>(*this), other);
    }
}
inline Tensor Tensor::gt(const Tensor & other) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::gt", "Tensor"}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &, const Tensor &>(op, const_cast<Tensor&>(*this), other);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), other);
    }
}
inline Tensor Tensor::lt(Scalar other) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::lt", "Scalar"}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &, Scalar>(op, const_cast<Tensor&>(*this), other);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &, Scalar>(const_cast<Tensor&>(*this), other);
    }
}
inline Tensor Tensor::lt(const Tensor & other) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::lt", "Tensor"}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &, const Tensor &>(op, const_cast<Tensor&>(*this), other);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), other);
    }
}
inline Tensor Tensor::take(const Tensor & index) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::take", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &, const Tensor &>(op, const_cast<Tensor&>(*this), index);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), index);
    }
}
inline Tensor Tensor::index_select(int64_t dim, const Tensor & index) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::index_select", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &, int64_t, const Tensor &>(op, const_cast<Tensor&>(*this), dim, index);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &, int64_t, const Tensor &>(const_cast<Tensor&>(*this), dim, index);
    }
}
inline Tensor Tensor::masked_select(const Tensor & mask) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::masked_select", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &, const Tensor &>(op, const_cast<Tensor&>(*this), mask);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), mask);
    }
}
inline Tensor Tensor::nonzero() const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::nonzero", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &>(op, const_cast<Tensor&>(*this));
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
    }
}
inline std::vector<Tensor> Tensor::nonzero_numpy() const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::nonzero_numpy", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<std::vector<Tensor>, const Tensor &>(op, const_cast<Tensor&>(*this));
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<std::vector<Tensor>, const Tensor &>(const_cast<Tensor&>(*this));
    }
}
inline Tensor Tensor::gather(int64_t dim, const Tensor & index, bool sparse_grad) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::gather", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &, int64_t, const Tensor &, bool>(op, const_cast<Tensor&>(*this), dim, index, sparse_grad);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &, int64_t, const Tensor &, bool>(const_cast<Tensor&>(*this), dim, index, sparse_grad);
    }
}
inline Tensor Tensor::addcmul(const Tensor & tensor1, const Tensor & tensor2, Scalar value) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::addcmul", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &, const Tensor &, const Tensor &, Scalar>(op, const_cast<Tensor&>(*this), tensor1, tensor2, value);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &, const Tensor &, const Tensor &, Scalar>(const_cast<Tensor&>(*this), tensor1, tensor2, value);
    }
}
inline Tensor & Tensor::addcmul_(const Tensor & tensor1, const Tensor & tensor2, Scalar value) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::addcmul_", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor &, Tensor &, const Tensor &, const Tensor &, Scalar>(op, const_cast<Tensor&>(*this), tensor1, tensor2, value);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor &, Tensor &, const Tensor &, const Tensor &, Scalar>(const_cast<Tensor&>(*this), tensor1, tensor2, value);
    }
}
inline Tensor Tensor::addcdiv(const Tensor & tensor1, const Tensor & tensor2, Scalar value) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::addcdiv", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &, const Tensor &, const Tensor &, Scalar>(op, const_cast<Tensor&>(*this), tensor1, tensor2, value);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &, const Tensor &, const Tensor &, Scalar>(const_cast<Tensor&>(*this), tensor1, tensor2, value);
    }
}
inline std::tuple<Tensor,Tensor> Tensor::lstsq(const Tensor & A) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::lstsq", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<std::tuple<Tensor,Tensor>, const Tensor &, const Tensor &>(op, const_cast<Tensor&>(*this), A);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<std::tuple<Tensor,Tensor>, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), A);
    }
}
inline std::tuple<Tensor,Tensor> Tensor::triangular_solve(const Tensor & A, bool upper, bool transpose, bool unitriangular) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::triangular_solve", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<std::tuple<Tensor,Tensor>, const Tensor &, const Tensor &, bool, bool, bool>(op, const_cast<Tensor&>(*this), A, upper, transpose, unitriangular);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<std::tuple<Tensor,Tensor>, const Tensor &, const Tensor &, bool, bool, bool>(const_cast<Tensor&>(*this), A, upper, transpose, unitriangular);
    }
}
inline std::tuple<Tensor,Tensor> Tensor::symeig(bool eigenvectors, bool upper) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::symeig", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<std::tuple<Tensor,Tensor>, const Tensor &, bool, bool>(op, const_cast<Tensor&>(*this), eigenvectors, upper);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<std::tuple<Tensor,Tensor>, const Tensor &, bool, bool>(const_cast<Tensor&>(*this), eigenvectors, upper);
    }
}
inline std::tuple<Tensor,Tensor> Tensor::eig(bool eigenvectors) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::eig", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<std::tuple<Tensor,Tensor>, const Tensor &, bool>(op, const_cast<Tensor&>(*this), eigenvectors);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<std::tuple<Tensor,Tensor>, const Tensor &, bool>(const_cast<Tensor&>(*this), eigenvectors);
    }
}
inline std::tuple<Tensor,Tensor,Tensor> Tensor::svd(bool some, bool compute_uv) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::svd", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<std::tuple<Tensor,Tensor,Tensor>, const Tensor &, bool, bool>(op, const_cast<Tensor&>(*this), some, compute_uv);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<std::tuple<Tensor,Tensor,Tensor>, const Tensor &, bool, bool>(const_cast<Tensor&>(*this), some, compute_uv);
    }
}
inline Tensor Tensor::cholesky(bool upper) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::cholesky", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &, bool>(op, const_cast<Tensor&>(*this), upper);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &, bool>(const_cast<Tensor&>(*this), upper);
    }
}
inline Tensor Tensor::cholesky_solve(const Tensor & input2, bool upper) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::cholesky_solve", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &, const Tensor &, bool>(op, const_cast<Tensor&>(*this), input2, upper);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &, const Tensor &, bool>(const_cast<Tensor&>(*this), input2, upper);
    }
}
inline std::tuple<Tensor,Tensor> Tensor::solve(const Tensor & A) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::solve", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<std::tuple<Tensor,Tensor>, const Tensor &, const Tensor &>(op, const_cast<Tensor&>(*this), A);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<std::tuple<Tensor,Tensor>, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), A);
    }
}
inline Tensor Tensor::cholesky_inverse(bool upper) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::cholesky_inverse", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &, bool>(op, const_cast<Tensor&>(*this), upper);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &, bool>(const_cast<Tensor&>(*this), upper);
    }
}
inline std::tuple<Tensor,Tensor> Tensor::qr(bool some) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::qr", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<std::tuple<Tensor,Tensor>, const Tensor &, bool>(op, const_cast<Tensor&>(*this), some);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<std::tuple<Tensor,Tensor>, const Tensor &, bool>(const_cast<Tensor&>(*this), some);
    }
}
inline std::tuple<Tensor,Tensor> Tensor::geqrf() const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::geqrf", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<std::tuple<Tensor,Tensor>, const Tensor &>(op, const_cast<Tensor&>(*this));
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<std::tuple<Tensor,Tensor>, const Tensor &>(const_cast<Tensor&>(*this));
    }
}
inline Tensor Tensor::orgqr(const Tensor & input2) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::orgqr", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &, const Tensor &>(op, const_cast<Tensor&>(*this), input2);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), input2);
    }
}
inline Tensor Tensor::ormqr(const Tensor & input2, const Tensor & input3, bool left, bool transpose) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::ormqr", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &, const Tensor &, const Tensor &, bool, bool>(op, const_cast<Tensor&>(*this), input2, input3, left, transpose);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &, const Tensor &, const Tensor &, bool, bool>(const_cast<Tensor&>(*this), input2, input3, left, transpose);
    }
}
inline Tensor Tensor::lu_solve(const Tensor & LU_data, const Tensor & LU_pivots) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::lu_solve", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &, const Tensor &, const Tensor &>(op, const_cast<Tensor&>(*this), LU_data, LU_pivots);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), LU_data, LU_pivots);
    }
}
inline Tensor Tensor::multinomial(int64_t num_samples, bool replacement, Generator * generator) const {
    static auto table = globalATenDispatch().getOpTable("aten::multinomial(Tensor self, int num_samples, bool replacement=False, *, Generator? generator=None) -> Tensor");
    return table->getOp<Tensor (const Tensor &, int64_t, bool, Generator *)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), num_samples, replacement, generator);
}
inline Tensor Tensor::lgamma() const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::lgamma", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &>(op, const_cast<Tensor&>(*this));
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
    }
}
inline Tensor Tensor::digamma() const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::digamma", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &>(op, const_cast<Tensor&>(*this));
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
    }
}
inline Tensor Tensor::polygamma(int64_t n) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::polygamma", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, int64_t, const Tensor &>(op, n, const_cast<Tensor&>(*this));
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, int64_t, const Tensor &>(n, const_cast<Tensor&>(*this));
    }
}
inline Tensor Tensor::erfinv() const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::erfinv", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &>(op, const_cast<Tensor&>(*this));
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
    }
}
inline Tensor Tensor::dist(const Tensor & other, Scalar p) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::dist", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &, const Tensor &, Scalar>(op, const_cast<Tensor&>(*this), other, p);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &, const Tensor &, Scalar>(const_cast<Tensor&>(*this), other, p);
    }
}
inline Tensor Tensor::atan2(const Tensor & other) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::atan2", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &, const Tensor &>(op, const_cast<Tensor&>(*this), other);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), other);
    }
}
inline Tensor Tensor::lerp(const Tensor & end, Scalar weight) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::lerp", "Scalar"}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &, const Tensor &, Scalar>(op, const_cast<Tensor&>(*this), end, weight);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &, const Tensor &, Scalar>(const_cast<Tensor&>(*this), end, weight);
    }
}
inline Tensor Tensor::lerp(const Tensor & end, const Tensor & weight) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::lerp", "Tensor"}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &, const Tensor &, const Tensor &>(op, const_cast<Tensor&>(*this), end, weight);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), end, weight);
    }
}
inline Tensor Tensor::histc(int64_t bins, Scalar min, Scalar max) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::histc", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &, int64_t, Scalar, Scalar>(op, const_cast<Tensor&>(*this), bins, min, max);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &, int64_t, Scalar, Scalar>(const_cast<Tensor&>(*this), bins, min, max);
    }
}
inline Tensor Tensor::sign() const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::sign", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &>(op, const_cast<Tensor&>(*this));
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
    }
}
inline Tensor Tensor::fmod(Scalar other) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::fmod", "Scalar"}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &, Scalar>(op, const_cast<Tensor&>(*this), other);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &, Scalar>(const_cast<Tensor&>(*this), other);
    }
}
inline Tensor Tensor::fmod(const Tensor & other) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::fmod", "Tensor"}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &, const Tensor &>(op, const_cast<Tensor&>(*this), other);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), other);
    }
}
inline Tensor Tensor::remainder(Scalar other) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::remainder", "Scalar"}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &, Scalar>(op, const_cast<Tensor&>(*this), other);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &, Scalar>(const_cast<Tensor&>(*this), other);
    }
}
inline Tensor Tensor::remainder(const Tensor & other) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::remainder", "Tensor"}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &, const Tensor &>(op, const_cast<Tensor&>(*this), other);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), other);
    }
}
inline Tensor Tensor::min(const Tensor & other) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::min", "other"}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &, const Tensor &>(op, const_cast<Tensor&>(*this), other);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), other);
    }
}
inline Tensor Tensor::min() const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::min", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &>(op, const_cast<Tensor&>(*this));
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
    }
}
inline Tensor Tensor::max(const Tensor & other) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::max", "other"}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &, const Tensor &>(op, const_cast<Tensor&>(*this), other);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), other);
    }
}
inline Tensor Tensor::max() const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::max", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &>(op, const_cast<Tensor&>(*this));
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
    }
}
inline Tensor Tensor::median() const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::median", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &>(op, const_cast<Tensor&>(*this));
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
    }
}
inline std::tuple<Tensor,Tensor> Tensor::sort(int64_t dim, bool descending) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::sort", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<std::tuple<Tensor,Tensor>, const Tensor &, int64_t, bool>(op, const_cast<Tensor&>(*this), dim, descending);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<std::tuple<Tensor,Tensor>, const Tensor &, int64_t, bool>(const_cast<Tensor&>(*this), dim, descending);
    }
}
inline Tensor Tensor::argsort(int64_t dim, bool descending) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::argsort", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &, int64_t, bool>(op, const_cast<Tensor&>(*this), dim, descending);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &, int64_t, bool>(const_cast<Tensor&>(*this), dim, descending);
    }
}
inline std::tuple<Tensor,Tensor> Tensor::topk(int64_t k, int64_t dim, bool largest, bool sorted) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::topk", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<std::tuple<Tensor,Tensor>, const Tensor &, int64_t, int64_t, bool, bool>(op, const_cast<Tensor&>(*this), k, dim, largest, sorted);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<std::tuple<Tensor,Tensor>, const Tensor &, int64_t, int64_t, bool, bool>(const_cast<Tensor&>(*this), k, dim, largest, sorted);
    }
}
inline Tensor Tensor::all() const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::all", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &>(op, const_cast<Tensor&>(*this));
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
    }
}
inline Tensor Tensor::any() const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::any", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &>(op, const_cast<Tensor&>(*this));
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
    }
}
inline Tensor Tensor::renorm(Scalar p, int64_t dim, Scalar maxnorm) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::renorm", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &, Scalar, int64_t, Scalar>(op, const_cast<Tensor&>(*this), p, dim, maxnorm);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &, Scalar, int64_t, Scalar>(const_cast<Tensor&>(*this), p, dim, maxnorm);
    }
}
inline Tensor Tensor::unfold(int64_t dimension, int64_t size, int64_t step) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::unfold", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &, int64_t, int64_t, int64_t>(op, const_cast<Tensor&>(*this), dimension, size, step);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &, int64_t, int64_t, int64_t>(const_cast<Tensor&>(*this), dimension, size, step);
    }
}
inline bool Tensor::equal(const Tensor & other) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::equal", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<bool, const Tensor &, const Tensor &>(op, const_cast<Tensor&>(*this), other);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<bool, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), other);
    }
}
inline Tensor Tensor::pow(const Tensor & exponent) const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::pow", "Tensor_Tensor"}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &, const Tensor &>(op, const_cast<Tensor&>(*this), exponent);
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &, const Tensor &>(const_cast<Tensor&>(*this), exponent);
    }
}
inline Tensor Tensor::alias() const {
    static c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"aten::alias", ""}).value();
    if (is_variable()) {
        return c10::Dispatcher::singleton().callUnboxedAutogradKernel<Tensor, const Tensor &>(op, const_cast<Tensor&>(*this));
    } else {
        return c10::Dispatcher::singleton().lookup(op, type_id()).callUnboxed<Tensor, const Tensor &>(const_cast<Tensor&>(*this));
    }
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
  return impl::internal_has_names(unsafeGetTensorImpl());
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
  inline T* Tensor::data() const {               \
    TORCH_CHECK(                                 \
        scalar_type() == ScalarType::name,       \
        "expected scalar type ",                 \
        #name,                                   \
        " but found ",                           \
        c10::toString(scalar_type()));           \
    return static_cast<T*>(this->data_ptr());    \
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
