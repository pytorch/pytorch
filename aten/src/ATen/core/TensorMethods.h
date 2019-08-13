#pragma once

#include <c10/core/Scalar.h>
#include <c10/core/MemoryFormat.h>
#include <c10/core/QScheme.h>
#include <c10/macros/Macros.h>
#include <c10/util/intrusive_ptr.h>
#include <ATen/core/DeprecatedTypeProperties.h>
#include <ATen/core/ATenDispatch.h>
#include <ATen/core/TensorOptions.h>
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
inline void Tensor::backward(const Tensor & gradient, bool keep_graph, bool create_graph) const {
    static auto table = globalATenDispatch().getOpTable("aten::backward(Tensor self, Tensor? gradient=None, bool keep_graph=False, bool create_graph=False) -> void");
    return table->getOp<void (const Tensor &, const Tensor &, bool, bool)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), gradient, keep_graph, create_graph);
}
inline void Tensor::set_data(const Tensor & new_data) const {
    static auto table = globalATenDispatch().getOpTable("aten::set_data(Tensor(a!) self, Tensor new_data) -> void");
    return table->getOp<void (const Tensor &, const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), new_data);
}
#ifdef BUILD_NAMEDTENSOR
inline Tensor & Tensor::names_(c10::optional<DimnameList> names) const {
    static auto table = globalATenDispatch().getOpTable("aten::names_(Tensor(a!) self, Dimname[]? names) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, c10::optional<DimnameList>)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), names);
}
#endif
#ifdef BUILD_NAMEDTENSOR
inline Tensor Tensor::view_names(c10::optional<DimnameList> names) const {
    static auto table = globalATenDispatch().getOpTable("aten::view_names(Tensor(a) self, Dimname[]? names) -> Tensor(a)");
    return table->getOp<Tensor (const Tensor &, c10::optional<DimnameList>)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), names);
}
#endif
#ifdef BUILD_NAMEDTENSOR
inline Tensor Tensor::align_to(DimnameList names) const {
    static auto table = globalATenDispatch().getOpTable("aten::align_to(Tensor self, DimnameList names) -> Tensor");
    return table->getOp<Tensor (const Tensor &, DimnameList)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), names);
}
#endif
inline Tensor Tensor::abs() const {
    static auto table = globalATenDispatch().getOpTable("aten::abs(Tensor self) -> Tensor");
    return table->getOp<Tensor (const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this));
}
inline Tensor & Tensor::abs_() const {
    static auto table = globalATenDispatch().getOpTable("aten::abs_(Tensor(a!) self) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this));
}
inline Tensor Tensor::acos() const {
    static auto table = globalATenDispatch().getOpTable("aten::acos(Tensor self) -> Tensor");
    return table->getOp<Tensor (const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this));
}
inline Tensor & Tensor::acos_() const {
    static auto table = globalATenDispatch().getOpTable("aten::acos_(Tensor(a!) self) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this));
}
inline Tensor Tensor::add(const Tensor & other, Scalar alpha) const {
    static auto table = globalATenDispatch().getOpTable("aten::add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor");
    return table->getOp<Tensor (const Tensor &, const Tensor &, Scalar)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), other, alpha);
}
inline Tensor & Tensor::add_(const Tensor & other, Scalar alpha) const {
    static auto table = globalATenDispatch().getOpTable("aten::add_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, const Tensor &, Scalar)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), other, alpha);
}
inline Tensor Tensor::add(Scalar other, Scalar alpha) const {
    static auto table = globalATenDispatch().getOpTable("aten::add.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> Tensor");
    return table->getOp<Tensor (const Tensor &, Scalar, Scalar)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), other, alpha);
}
inline Tensor & Tensor::add_(Scalar other, Scalar alpha) const {
    static auto table = globalATenDispatch().getOpTable("aten::add_.Scalar(Tensor(a!) self, Scalar other, Scalar alpha=1) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, Scalar, Scalar)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), other, alpha);
}
inline Tensor Tensor::addmv(const Tensor & mat, const Tensor & vec, Scalar beta, Scalar alpha) const {
    static auto table = globalATenDispatch().getOpTable("aten::addmv(Tensor self, Tensor mat, Tensor vec, *, Scalar beta=1, Scalar alpha=1) -> Tensor");
    return table->getOp<Tensor (const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), mat, vec, beta, alpha);
}
inline Tensor & Tensor::addmv_(const Tensor & mat, const Tensor & vec, Scalar beta, Scalar alpha) const {
    static auto table = globalATenDispatch().getOpTable("aten::addmv_(Tensor(a!) self, Tensor mat, Tensor vec, *, Scalar beta=1, Scalar alpha=1) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, const Tensor &, const Tensor &, Scalar, Scalar)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), mat, vec, beta, alpha);
}
inline Tensor Tensor::addr(const Tensor & vec1, const Tensor & vec2, Scalar beta, Scalar alpha) const {
    static auto table = globalATenDispatch().getOpTable("aten::addr(Tensor self, Tensor vec1, Tensor vec2, *, Scalar beta=1, Scalar alpha=1) -> Tensor");
    return table->getOp<Tensor (const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), vec1, vec2, beta, alpha);
}
inline Tensor & Tensor::addr_(const Tensor & vec1, const Tensor & vec2, Scalar beta, Scalar alpha) const {
    static auto table = globalATenDispatch().getOpTable("aten::addr_(Tensor(a!) self, Tensor vec1, Tensor vec2, *, Scalar beta=1, Scalar alpha=1) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, const Tensor &, const Tensor &, Scalar, Scalar)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), vec1, vec2, beta, alpha);
}
inline Tensor Tensor::all(int64_t dim, bool keepdim) const {
    static auto table = globalATenDispatch().getOpTable("aten::all.dim(Tensor self, int dim, bool keepdim=False) -> Tensor");
    return table->getOp<Tensor (const Tensor &, int64_t, bool)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), dim, keepdim);
}
inline bool Tensor::allclose(const Tensor & other, double rtol, double atol, bool equal_nan) const {
    static auto table = globalATenDispatch().getOpTable("aten::allclose(Tensor self, Tensor other, float rtol=1e-05, float atol=1e-08, bool equal_nan=False) -> bool");
    return table->getOp<bool (const Tensor &, const Tensor &, double, double, bool)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), other, rtol, atol, equal_nan);
}
inline Tensor Tensor::any(int64_t dim, bool keepdim) const {
    static auto table = globalATenDispatch().getOpTable("aten::any.dim(Tensor self, int dim, bool keepdim=False) -> Tensor");
    return table->getOp<Tensor (const Tensor &, int64_t, bool)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), dim, keepdim);
}
inline Tensor Tensor::argmax(c10::optional<int64_t> dim, bool keepdim) const {
    static auto table = globalATenDispatch().getOpTable("aten::argmax(Tensor self, int? dim=None, bool keepdim=False) -> Tensor");
    return table->getOp<Tensor (const Tensor &, c10::optional<int64_t>, bool)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), dim, keepdim);
}
inline Tensor Tensor::argmin(c10::optional<int64_t> dim, bool keepdim) const {
    static auto table = globalATenDispatch().getOpTable("aten::argmin(Tensor self, int? dim=None, bool keepdim=False) -> Tensor");
    return table->getOp<Tensor (const Tensor &, c10::optional<int64_t>, bool)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), dim, keepdim);
}
inline Tensor Tensor::as_strided(IntArrayRef size, IntArrayRef stride, c10::optional<int64_t> storage_offset) const {
    static auto table = globalATenDispatch().getOpTable("aten::as_strided(Tensor(a) self, int[] size, int[] stride, int? storage_offset=None) -> Tensor(a)");
    return table->getOp<Tensor (const Tensor &, IntArrayRef, IntArrayRef, c10::optional<int64_t>)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), size, stride, storage_offset);
}
inline Tensor & Tensor::as_strided_(IntArrayRef size, IntArrayRef stride, c10::optional<int64_t> storage_offset) const {
    static auto table = globalATenDispatch().getOpTable("aten::as_strided_(Tensor(a!) self, int[] size, int[] stride, int? storage_offset=None) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, IntArrayRef, IntArrayRef, c10::optional<int64_t>)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), size, stride, storage_offset);
}
inline Tensor Tensor::asin() const {
    static auto table = globalATenDispatch().getOpTable("aten::asin(Tensor self) -> Tensor");
    return table->getOp<Tensor (const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this));
}
inline Tensor & Tensor::asin_() const {
    static auto table = globalATenDispatch().getOpTable("aten::asin_(Tensor(a!) self) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this));
}
inline Tensor Tensor::atan() const {
    static auto table = globalATenDispatch().getOpTable("aten::atan(Tensor self) -> Tensor");
    return table->getOp<Tensor (const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this));
}
inline Tensor & Tensor::atan_() const {
    static auto table = globalATenDispatch().getOpTable("aten::atan_(Tensor(a!) self) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this));
}
inline Tensor Tensor::baddbmm(const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) const {
    static auto table = globalATenDispatch().getOpTable("aten::baddbmm(Tensor self, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1) -> Tensor");
    return table->getOp<Tensor (const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), batch1, batch2, beta, alpha);
}
inline Tensor & Tensor::baddbmm_(const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) const {
    static auto table = globalATenDispatch().getOpTable("aten::baddbmm_(Tensor(a!) self, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, const Tensor &, const Tensor &, Scalar, Scalar)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), batch1, batch2, beta, alpha);
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
    static auto table = globalATenDispatch().getOpTable("aten::bitwise_not(Tensor self) -> Tensor");
    return table->getOp<Tensor (const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this));
}
inline Tensor & Tensor::bitwise_not_() const {
    static auto table = globalATenDispatch().getOpTable("aten::bitwise_not_(Tensor(a!) self) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this));
}
inline Tensor Tensor::bmm(const Tensor & mat2) const {
    static auto table = globalATenDispatch().getOpTable("aten::bmm(Tensor self, Tensor mat2) -> Tensor");
    return table->getOp<Tensor (const Tensor &, const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), mat2);
}
inline Tensor Tensor::ceil() const {
    static auto table = globalATenDispatch().getOpTable("aten::ceil(Tensor self) -> Tensor");
    return table->getOp<Tensor (const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this));
}
inline Tensor & Tensor::ceil_() const {
    static auto table = globalATenDispatch().getOpTable("aten::ceil_(Tensor(a!) self) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this));
}
inline std::vector<Tensor> Tensor::chunk(int64_t chunks, int64_t dim) const {
    static auto table = globalATenDispatch().getOpTable("aten::chunk(Tensor(a) self, int chunks, int dim=0) -> Tensor(a)[]");
    return table->getOp<std::vector<Tensor> (const Tensor &, int64_t, int64_t)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), chunks, dim);
}
inline Tensor Tensor::clamp(c10::optional<Scalar> min, c10::optional<Scalar> max) const {
    static auto table = globalATenDispatch().getOpTable("aten::clamp(Tensor self, Scalar? min=None, Scalar? max=None) -> Tensor");
    return table->getOp<Tensor (const Tensor &, c10::optional<Scalar>, c10::optional<Scalar>)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), min, max);
}
inline Tensor & Tensor::clamp_(c10::optional<Scalar> min, c10::optional<Scalar> max) const {
    static auto table = globalATenDispatch().getOpTable("aten::clamp_(Tensor(a!) self, Scalar? min=None, Scalar? max=None) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, c10::optional<Scalar>, c10::optional<Scalar>)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), min, max);
}
inline Tensor Tensor::clamp_max(Scalar max) const {
    static auto table = globalATenDispatch().getOpTable("aten::clamp_max(Tensor self, Scalar max) -> Tensor");
    return table->getOp<Tensor (const Tensor &, Scalar)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), max);
}
inline Tensor & Tensor::clamp_max_(Scalar max) const {
    static auto table = globalATenDispatch().getOpTable("aten::clamp_max_(Tensor(a!) self, Scalar max) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, Scalar)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), max);
}
inline Tensor Tensor::clamp_min(Scalar min) const {
    static auto table = globalATenDispatch().getOpTable("aten::clamp_min(Tensor self, Scalar min) -> Tensor");
    return table->getOp<Tensor (const Tensor &, Scalar)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), min);
}
inline Tensor & Tensor::clamp_min_(Scalar min) const {
    static auto table = globalATenDispatch().getOpTable("aten::clamp_min_(Tensor(a!) self, Scalar min) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, Scalar)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), min);
}
inline Tensor Tensor::contiguous(MemoryFormat memory_format) const {
    static auto table = globalATenDispatch().getOpTable("aten::contiguous(Tensor self, *, MemoryFormat memory_format=contiguous_format) -> Tensor");
    return table->getOp<Tensor (const Tensor &, MemoryFormat)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), memory_format);
}
inline Tensor & Tensor::copy_(const Tensor & src, bool non_blocking) const {
    static auto table = globalATenDispatch().getOpTable("aten::copy_(Tensor(a!) self, Tensor src, bool non_blocking=False) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, const Tensor &, bool)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), src, non_blocking);
}
inline Tensor Tensor::cos() const {
    static auto table = globalATenDispatch().getOpTable("aten::cos(Tensor self) -> Tensor");
    return table->getOp<Tensor (const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this));
}
inline Tensor & Tensor::cos_() const {
    static auto table = globalATenDispatch().getOpTable("aten::cos_(Tensor(a!) self) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this));
}
inline Tensor Tensor::cosh() const {
    static auto table = globalATenDispatch().getOpTable("aten::cosh(Tensor self) -> Tensor");
    return table->getOp<Tensor (const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this));
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
    static auto table = globalATenDispatch().getOpTable("aten::det(Tensor self) -> Tensor");
    return table->getOp<Tensor (const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this));
}
inline Tensor Tensor::diag_embed(int64_t offset, int64_t dim1, int64_t dim2) const {
    static auto table = globalATenDispatch().getOpTable("aten::diag_embed(Tensor self, int offset=0, int dim1=-2, int dim2=-1) -> Tensor");
    return table->getOp<Tensor (const Tensor &, int64_t, int64_t, int64_t)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), offset, dim1, dim2);
}
inline Tensor Tensor::diagflat(int64_t offset) const {
    static auto table = globalATenDispatch().getOpTable("aten::diagflat(Tensor self, int offset=0) -> Tensor");
    return table->getOp<Tensor (const Tensor &, int64_t)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), offset);
}
inline Tensor Tensor::diagonal(int64_t offset, int64_t dim1, int64_t dim2) const {
    static auto table = globalATenDispatch().getOpTable("aten::diagonal(Tensor(a) self, int offset=0, int dim1=0, int dim2=1) -> Tensor(a)");
    return table->getOp<Tensor (const Tensor &, int64_t, int64_t, int64_t)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), offset, dim1, dim2);
}
inline Tensor & Tensor::fill_diagonal_(Scalar fill_value, bool wrap) const {
    static auto table = globalATenDispatch().getOpTable("aten::fill_diagonal_(Tensor(a!) self, Scalar fill_value, bool wrap=False) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, Scalar, bool)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), fill_value, wrap);
}
inline Tensor Tensor::div(const Tensor & other) const {
    static auto table = globalATenDispatch().getOpTable("aten::div.Tensor(Tensor self, Tensor other) -> Tensor");
    return table->getOp<Tensor (const Tensor &, const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), other);
}
inline Tensor & Tensor::div_(const Tensor & other) const {
    static auto table = globalATenDispatch().getOpTable("aten::div_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), other);
}
inline Tensor Tensor::div(Scalar other) const {
    static auto table = globalATenDispatch().getOpTable("aten::div.Scalar(Tensor self, Scalar other) -> Tensor");
    return table->getOp<Tensor (const Tensor &, Scalar)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), other);
}
inline Tensor & Tensor::div_(Scalar other) const {
    static auto table = globalATenDispatch().getOpTable("aten::div_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, Scalar)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), other);
}
inline Tensor Tensor::dot(const Tensor & tensor) const {
    static auto table = globalATenDispatch().getOpTable("aten::dot(Tensor self, Tensor tensor) -> Tensor");
    return table->getOp<Tensor (const Tensor &, const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), tensor);
}
inline Tensor & Tensor::resize_(IntArrayRef size) const {
    static auto table = globalATenDispatch().getOpTable("aten::resize_(Tensor(a!) self, int[] size) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, IntArrayRef)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), size);
}
inline Tensor Tensor::erf() const {
    static auto table = globalATenDispatch().getOpTable("aten::erf(Tensor self) -> Tensor");
    return table->getOp<Tensor (const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this));
}
inline Tensor & Tensor::erf_() const {
    static auto table = globalATenDispatch().getOpTable("aten::erf_(Tensor(a!) self) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this));
}
inline Tensor Tensor::erfc() const {
    static auto table = globalATenDispatch().getOpTable("aten::erfc(Tensor self) -> Tensor");
    return table->getOp<Tensor (const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this));
}
inline Tensor & Tensor::erfc_() const {
    static auto table = globalATenDispatch().getOpTable("aten::erfc_(Tensor(a!) self) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this));
}
inline Tensor Tensor::exp() const {
    static auto table = globalATenDispatch().getOpTable("aten::exp(Tensor self) -> Tensor");
    return table->getOp<Tensor (const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this));
}
inline Tensor & Tensor::exp_() const {
    static auto table = globalATenDispatch().getOpTable("aten::exp_(Tensor(a!) self) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this));
}
inline Tensor Tensor::expm1() const {
    static auto table = globalATenDispatch().getOpTable("aten::expm1(Tensor self) -> Tensor");
    return table->getOp<Tensor (const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this));
}
inline Tensor & Tensor::expm1_() const {
    static auto table = globalATenDispatch().getOpTable("aten::expm1_(Tensor(a!) self) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this));
}
inline Tensor Tensor::expand(IntArrayRef size, bool implicit) const {
    static auto table = globalATenDispatch().getOpTable("aten::expand(Tensor(a) self, int[] size, *, bool implicit=False) -> Tensor(a)");
    return table->getOp<Tensor (const Tensor &, IntArrayRef, bool)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), size, implicit);
}
inline Tensor Tensor::expand_as(const Tensor & other) const {
    static auto table = globalATenDispatch().getOpTable("aten::expand_as(Tensor self, Tensor other) -> Tensor");
    return table->getOp<Tensor (const Tensor &, const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), other);
}
inline Tensor Tensor::flatten(int64_t start_dim, int64_t end_dim) const {
    static auto table = globalATenDispatch().getOpTable("aten::flatten(Tensor self, int start_dim=0, int end_dim=-1) -> Tensor");
    return table->getOp<Tensor (const Tensor &, int64_t, int64_t)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), start_dim, end_dim);
}
inline Tensor & Tensor::fill_(Scalar value) const {
    static auto table = globalATenDispatch().getOpTable("aten::fill_.Scalar(Tensor(a!) self, Scalar value) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, Scalar)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), value);
}
inline Tensor & Tensor::fill_(const Tensor & value) const {
    static auto table = globalATenDispatch().getOpTable("aten::fill_.Tensor(Tensor(a!) self, Tensor value) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), value);
}
inline Tensor Tensor::floor() const {
    static auto table = globalATenDispatch().getOpTable("aten::floor(Tensor self) -> Tensor");
    return table->getOp<Tensor (const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this));
}
inline Tensor & Tensor::floor_() const {
    static auto table = globalATenDispatch().getOpTable("aten::floor_(Tensor(a!) self) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this));
}
inline Tensor Tensor::frac() const {
    static auto table = globalATenDispatch().getOpTable("aten::frac(Tensor self) -> Tensor");
    return table->getOp<Tensor (const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this));
}
inline Tensor & Tensor::frac_() const {
    static auto table = globalATenDispatch().getOpTable("aten::frac_(Tensor(a!) self) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this));
}
inline Tensor Tensor::ger(const Tensor & vec2) const {
    static auto table = globalATenDispatch().getOpTable("aten::ger(Tensor self, Tensor vec2) -> Tensor");
    return table->getOp<Tensor (const Tensor &, const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), vec2);
}
inline Tensor Tensor::fft(int64_t signal_ndim, bool normalized) const {
    static auto table = globalATenDispatch().getOpTable("aten::fft(Tensor self, int signal_ndim, bool normalized=False) -> Tensor");
    return table->getOp<Tensor (const Tensor &, int64_t, bool)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), signal_ndim, normalized);
}
inline Tensor Tensor::ifft(int64_t signal_ndim, bool normalized) const {
    static auto table = globalATenDispatch().getOpTable("aten::ifft(Tensor self, int signal_ndim, bool normalized=False) -> Tensor");
    return table->getOp<Tensor (const Tensor &, int64_t, bool)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), signal_ndim, normalized);
}
inline Tensor Tensor::rfft(int64_t signal_ndim, bool normalized, bool onesided) const {
    static auto table = globalATenDispatch().getOpTable("aten::rfft(Tensor self, int signal_ndim, bool normalized=False, bool onesided=True) -> Tensor");
    return table->getOp<Tensor (const Tensor &, int64_t, bool, bool)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), signal_ndim, normalized, onesided);
}
inline Tensor Tensor::irfft(int64_t signal_ndim, bool normalized, bool onesided, IntArrayRef signal_sizes) const {
    static auto table = globalATenDispatch().getOpTable("aten::irfft(Tensor self, int signal_ndim, bool normalized=False, bool onesided=True, int[] signal_sizes=[]) -> Tensor");
    return table->getOp<Tensor (const Tensor &, int64_t, bool, bool, IntArrayRef)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), signal_ndim, normalized, onesided, signal_sizes);
}
inline Tensor Tensor::index(TensorList indices) const {
    static auto table = globalATenDispatch().getOpTable("aten::index(Tensor self, Tensor?[] indices) -> Tensor");
    return table->getOp<Tensor (const Tensor &, TensorList)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), indices);
}
inline Tensor & Tensor::index_copy_(int64_t dim, const Tensor & index, const Tensor & source) const {
    static auto table = globalATenDispatch().getOpTable("aten::index_copy_(Tensor(a!) self, int dim, Tensor index, Tensor source) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, int64_t, const Tensor &, const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), dim, index, source);
}
inline Tensor Tensor::index_copy(int64_t dim, const Tensor & index, const Tensor & source) const {
    static auto table = globalATenDispatch().getOpTable("aten::index_copy(Tensor self, int dim, Tensor index, Tensor source) -> Tensor");
    return table->getOp<Tensor (const Tensor &, int64_t, const Tensor &, const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), dim, index, source);
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
    static auto table = globalATenDispatch().getOpTable("aten::inverse(Tensor self) -> Tensor");
    return table->getOp<Tensor (const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this));
}
inline Tensor Tensor::isclose(const Tensor & other, double rtol, double atol, bool equal_nan) const {
    static auto table = globalATenDispatch().getOpTable("aten::isclose(Tensor self, Tensor other, float rtol=1e-05, float atol=1e-08, bool equal_nan=False) -> Tensor");
    return table->getOp<Tensor (const Tensor &, const Tensor &, double, double, bool)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), other, rtol, atol, equal_nan);
}
inline bool Tensor::is_distributed() const {
    static auto table = globalATenDispatch().getOpTable("aten::is_distributed(Tensor self) -> bool");
    return table->getOp<bool (const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this));
}
inline bool Tensor::is_floating_point() const {
    static auto table = globalATenDispatch().getOpTable("aten::is_floating_point(Tensor self) -> bool");
    return table->getOp<bool (const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this));
}
inline bool Tensor::is_complex() const {
    static auto table = globalATenDispatch().getOpTable("aten::is_complex(Tensor self) -> bool");
    return table->getOp<bool (const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this));
}
inline bool Tensor::is_nonzero() const {
    static auto table = globalATenDispatch().getOpTable("aten::is_nonzero(Tensor self) -> bool");
    return table->getOp<bool (const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this));
}
inline bool Tensor::is_same_size(const Tensor & other) const {
    static auto table = globalATenDispatch().getOpTable("aten::is_same_size(Tensor self, Tensor other) -> bool");
    return table->getOp<bool (const Tensor &, const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), other);
}
inline bool Tensor::is_signed() const {
    static auto table = globalATenDispatch().getOpTable("aten::is_signed(Tensor self) -> bool");
    return table->getOp<bool (const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this));
}
inline std::tuple<Tensor,Tensor> Tensor::kthvalue(int64_t k, int64_t dim, bool keepdim) const {
    static auto table = globalATenDispatch().getOpTable("aten::kthvalue(Tensor self, int k, int dim=-1, bool keepdim=False) -> (Tensor values, Tensor indices)");
    return table->getOp<std::tuple<Tensor,Tensor> (const Tensor &, int64_t, int64_t, bool)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), k, dim, keepdim);
}
inline Tensor Tensor::log() const {
    static auto table = globalATenDispatch().getOpTable("aten::log(Tensor self) -> Tensor");
    return table->getOp<Tensor (const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this));
}
inline Tensor & Tensor::log_() const {
    static auto table = globalATenDispatch().getOpTable("aten::log_(Tensor(a!) self) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this));
}
inline Tensor Tensor::log10() const {
    static auto table = globalATenDispatch().getOpTable("aten::log10(Tensor self) -> Tensor");
    return table->getOp<Tensor (const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this));
}
inline Tensor & Tensor::log10_() const {
    static auto table = globalATenDispatch().getOpTable("aten::log10_(Tensor(a!) self) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this));
}
inline Tensor Tensor::log1p() const {
    static auto table = globalATenDispatch().getOpTable("aten::log1p(Tensor self) -> Tensor");
    return table->getOp<Tensor (const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this));
}
inline Tensor & Tensor::log1p_() const {
    static auto table = globalATenDispatch().getOpTable("aten::log1p_(Tensor(a!) self) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this));
}
inline Tensor Tensor::log2() const {
    static auto table = globalATenDispatch().getOpTable("aten::log2(Tensor self) -> Tensor");
    return table->getOp<Tensor (const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this));
}
inline Tensor & Tensor::log2_() const {
    static auto table = globalATenDispatch().getOpTable("aten::log2_(Tensor(a!) self) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this));
}
inline Tensor Tensor::logdet() const {
    static auto table = globalATenDispatch().getOpTable("aten::logdet(Tensor self) -> Tensor");
    return table->getOp<Tensor (const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this));
}
inline Tensor Tensor::log_softmax(int64_t dim, c10::optional<ScalarType> dtype) const {
    static auto table = globalATenDispatch().getOpTable("aten::log_softmax(Tensor self, int dim, ScalarType? dtype=None) -> Tensor");
    return table->getOp<Tensor (const Tensor &, int64_t, c10::optional<ScalarType>)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), dim, dtype);
}
inline Tensor Tensor::logsumexp(IntArrayRef dim, bool keepdim) const {
    static auto table = globalATenDispatch().getOpTable("aten::logsumexp(Tensor self, int[1] dim, bool keepdim=False) -> Tensor");
    return table->getOp<Tensor (const Tensor &, IntArrayRef, bool)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), dim, keepdim);
}
inline Tensor Tensor::matmul(const Tensor & other) const {
    static auto table = globalATenDispatch().getOpTable("aten::matmul(Tensor self, Tensor other) -> Tensor");
    return table->getOp<Tensor (const Tensor &, const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), other);
}
inline Tensor Tensor::matrix_power(int64_t n) const {
    static auto table = globalATenDispatch().getOpTable("aten::matrix_power(Tensor self, int n) -> Tensor");
    return table->getOp<Tensor (const Tensor &, int64_t)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), n);
}
inline std::tuple<Tensor,Tensor> Tensor::max(int64_t dim, bool keepdim) const {
    static auto table = globalATenDispatch().getOpTable("aten::max.dim(Tensor self, int dim, bool keepdim=False) -> (Tensor values, Tensor indices)");
    return table->getOp<std::tuple<Tensor,Tensor> (const Tensor &, int64_t, bool)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), dim, keepdim);
}
inline Tensor Tensor::max_values(IntArrayRef dim, bool keepdim) const {
    static auto table = globalATenDispatch().getOpTable("aten::max_values(Tensor self, int[1] dim, bool keepdim=False) -> Tensor");
    return table->getOp<Tensor (const Tensor &, IntArrayRef, bool)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), dim, keepdim);
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
    static auto table = globalATenDispatch().getOpTable("aten::median.dim(Tensor self, int dim, bool keepdim=False) -> (Tensor values, Tensor indices)");
    return table->getOp<std::tuple<Tensor,Tensor> (const Tensor &, int64_t, bool)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), dim, keepdim);
}
inline std::tuple<Tensor,Tensor> Tensor::min(int64_t dim, bool keepdim) const {
    static auto table = globalATenDispatch().getOpTable("aten::min.dim(Tensor self, int dim, bool keepdim=False) -> (Tensor values, Tensor indices)");
    return table->getOp<std::tuple<Tensor,Tensor> (const Tensor &, int64_t, bool)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), dim, keepdim);
}
inline Tensor Tensor::min_values(IntArrayRef dim, bool keepdim) const {
    static auto table = globalATenDispatch().getOpTable("aten::min_values(Tensor self, int[1] dim, bool keepdim=False) -> Tensor");
    return table->getOp<Tensor (const Tensor &, IntArrayRef, bool)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), dim, keepdim);
}
inline Tensor Tensor::mm(const Tensor & mat2) const {
    static auto table = globalATenDispatch().getOpTable("aten::mm(Tensor self, Tensor mat2) -> Tensor");
    return table->getOp<Tensor (const Tensor &, const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), mat2);
}
inline std::tuple<Tensor,Tensor> Tensor::mode(int64_t dim, bool keepdim) const {
    static auto table = globalATenDispatch().getOpTable("aten::mode(Tensor self, int dim=-1, bool keepdim=False) -> (Tensor values, Tensor indices)");
    return table->getOp<std::tuple<Tensor,Tensor> (const Tensor &, int64_t, bool)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), dim, keepdim);
}
inline Tensor Tensor::mul(const Tensor & other) const {
    static auto table = globalATenDispatch().getOpTable("aten::mul.Tensor(Tensor self, Tensor other) -> Tensor");
    return table->getOp<Tensor (const Tensor &, const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), other);
}
inline Tensor & Tensor::mul_(const Tensor & other) const {
    static auto table = globalATenDispatch().getOpTable("aten::mul_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), other);
}
inline Tensor Tensor::mul(Scalar other) const {
    static auto table = globalATenDispatch().getOpTable("aten::mul.Scalar(Tensor self, Scalar other) -> Tensor");
    return table->getOp<Tensor (const Tensor &, Scalar)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), other);
}
inline Tensor & Tensor::mul_(Scalar other) const {
    static auto table = globalATenDispatch().getOpTable("aten::mul_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, Scalar)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), other);
}
inline Tensor Tensor::mv(const Tensor & vec) const {
    static auto table = globalATenDispatch().getOpTable("aten::mv(Tensor self, Tensor vec) -> Tensor");
    return table->getOp<Tensor (const Tensor &, const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), vec);
}
inline Tensor Tensor::mvlgamma(int64_t p) const {
    static auto table = globalATenDispatch().getOpTable("aten::mvlgamma(Tensor self, int p) -> Tensor");
    return table->getOp<Tensor (const Tensor &, int64_t)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), p);
}
inline Tensor & Tensor::mvlgamma_(int64_t p) const {
    static auto table = globalATenDispatch().getOpTable("aten::mvlgamma_(Tensor(a!) self, int p) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, int64_t)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), p);
}
inline Tensor Tensor::narrow_copy(int64_t dim, int64_t start, int64_t length) const {
    static auto table = globalATenDispatch().getOpTable("aten::narrow_copy(Tensor self, int dim, int start, int length) -> Tensor");
    return table->getOp<Tensor (const Tensor &, int64_t, int64_t, int64_t)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), dim, start, length);
}
inline Tensor Tensor::narrow(int64_t dim, int64_t start, int64_t length) const {
    static auto table = globalATenDispatch().getOpTable("aten::narrow(Tensor(a) self, int dim, int start, int length) -> Tensor(a)");
    return table->getOp<Tensor (const Tensor &, int64_t, int64_t, int64_t)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), dim, start, length);
}
inline Tensor Tensor::permute(IntArrayRef dims) const {
    static auto table = globalATenDispatch().getOpTable("aten::permute(Tensor(a) self, int[] dims) -> Tensor(a)");
    return table->getOp<Tensor (const Tensor &, IntArrayRef)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), dims);
}
inline Tensor Tensor::numpy_T() const {
    static auto table = globalATenDispatch().getOpTable("aten::numpy_T(Tensor(a) self) -> Tensor(a)");
    return table->getOp<Tensor (const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this));
}
inline bool Tensor::is_pinned() const {
    static auto table = globalATenDispatch().getOpTable("aten::is_pinned(Tensor self) -> bool");
    return table->getOp<bool (const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this));
}
inline Tensor Tensor::pin_memory() const {
    static auto table = globalATenDispatch().getOpTable("aten::pin_memory(Tensor self) -> Tensor");
    return table->getOp<Tensor (const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this));
}
inline Tensor Tensor::pinverse(double rcond) const {
    static auto table = globalATenDispatch().getOpTable("aten::pinverse(Tensor self, float rcond=1e-15) -> Tensor");
    return table->getOp<Tensor (const Tensor &, double)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), rcond);
}
inline Tensor Tensor::reciprocal() const {
    static auto table = globalATenDispatch().getOpTable("aten::reciprocal(Tensor self) -> Tensor");
    return table->getOp<Tensor (const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this));
}
inline Tensor & Tensor::reciprocal_() const {
    static auto table = globalATenDispatch().getOpTable("aten::reciprocal_(Tensor(a!) self) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this));
}
inline Tensor Tensor::neg() const {
    static auto table = globalATenDispatch().getOpTable("aten::neg(Tensor self) -> Tensor");
    return table->getOp<Tensor (const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this));
}
inline Tensor & Tensor::neg_() const {
    static auto table = globalATenDispatch().getOpTable("aten::neg_(Tensor(a!) self) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this));
}
inline Tensor Tensor::repeat(IntArrayRef repeats) const {
    static auto table = globalATenDispatch().getOpTable("aten::repeat(Tensor self, int[] repeats) -> Tensor");
    return table->getOp<Tensor (const Tensor &, IntArrayRef)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), repeats);
}
inline Tensor Tensor::repeat_interleave(const Tensor & repeats, c10::optional<int64_t> dim) const {
    static auto table = globalATenDispatch().getOpTable("aten::repeat_interleave.self_Tensor(Tensor self, Tensor repeats, int? dim=None) -> Tensor");
    return table->getOp<Tensor (const Tensor &, const Tensor &, c10::optional<int64_t>)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), repeats, dim);
}
inline Tensor Tensor::repeat_interleave(int64_t repeats, c10::optional<int64_t> dim) const {
    static auto table = globalATenDispatch().getOpTable("aten::repeat_interleave.self_int(Tensor self, int repeats, int? dim=None) -> Tensor");
    return table->getOp<Tensor (const Tensor &, int64_t, c10::optional<int64_t>)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), repeats, dim);
}
inline Tensor Tensor::reshape(IntArrayRef shape) const {
    static auto table = globalATenDispatch().getOpTable("aten::reshape(Tensor self, int[] shape) -> Tensor");
    return table->getOp<Tensor (const Tensor &, IntArrayRef)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), shape);
}
inline Tensor Tensor::reshape_as(const Tensor & other) const {
    static auto table = globalATenDispatch().getOpTable("aten::reshape_as(Tensor self, Tensor other) -> Tensor");
    return table->getOp<Tensor (const Tensor &, const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), other);
}
inline Tensor Tensor::round() const {
    static auto table = globalATenDispatch().getOpTable("aten::round(Tensor self) -> Tensor");
    return table->getOp<Tensor (const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this));
}
inline Tensor & Tensor::round_() const {
    static auto table = globalATenDispatch().getOpTable("aten::round_(Tensor(a!) self) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this));
}
inline Tensor Tensor::relu() const {
    static auto table = globalATenDispatch().getOpTable("aten::relu(Tensor self) -> Tensor");
    return table->getOp<Tensor (const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this));
}
inline Tensor & Tensor::relu_() const {
    static auto table = globalATenDispatch().getOpTable("aten::relu_(Tensor(a!) self) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this));
}
inline Tensor Tensor::prelu(const Tensor & weight) const {
    static auto table = globalATenDispatch().getOpTable("aten::prelu(Tensor self, Tensor weight) -> Tensor");
    return table->getOp<Tensor (const Tensor &, const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), weight);
}
inline std::tuple<Tensor,Tensor> Tensor::prelu_backward(const Tensor & grad_output, const Tensor & weight) const {
    static auto table = globalATenDispatch().getOpTable("aten::prelu_backward(Tensor grad_output, Tensor self, Tensor weight) -> (Tensor, Tensor)");
    return table->getOp<std::tuple<Tensor,Tensor> (const Tensor &, const Tensor &, const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(grad_output, const_cast<Tensor&>(*this), weight);
}
inline Tensor Tensor::hardshrink(Scalar lambd) const {
    static auto table = globalATenDispatch().getOpTable("aten::hardshrink(Tensor self, Scalar lambd=0.5) -> Tensor");
    return table->getOp<Tensor (const Tensor &, Scalar)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), lambd);
}
inline Tensor Tensor::hardshrink_backward(const Tensor & grad_out, Scalar lambd) const {
    static auto table = globalATenDispatch().getOpTable("aten::hardshrink_backward(Tensor grad_out, Tensor self, Scalar lambd) -> Tensor");
    return table->getOp<Tensor (const Tensor &, const Tensor &, Scalar)>(tensorTypeIdToBackend(type_id()), is_variable())(grad_out, const_cast<Tensor&>(*this), lambd);
}
inline Tensor Tensor::rsqrt() const {
    static auto table = globalATenDispatch().getOpTable("aten::rsqrt(Tensor self) -> Tensor");
    return table->getOp<Tensor (const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this));
}
inline Tensor & Tensor::rsqrt_() const {
    static auto table = globalATenDispatch().getOpTable("aten::rsqrt_(Tensor(a!) self) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this));
}
#ifdef BUILD_NAMEDTENSOR
inline Tensor Tensor::select(Dimname dim, int64_t index) const {
    static auto table = globalATenDispatch().getOpTable("aten::select.Dimname(Tensor(a) self, Dimname dim, int index) -> Tensor(a)");
    return table->getOp<Tensor (const Tensor &, Dimname, int64_t)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), dim, index);
}
#endif
inline Tensor Tensor::select(int64_t dim, int64_t index) const {
    static auto table = globalATenDispatch().getOpTable("aten::select.int(Tensor(a) self, int dim, int index) -> Tensor(a)");
    return table->getOp<Tensor (const Tensor &, int64_t, int64_t)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), dim, index);
}
inline Tensor Tensor::sigmoid() const {
    static auto table = globalATenDispatch().getOpTable("aten::sigmoid(Tensor self) -> Tensor");
    return table->getOp<Tensor (const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this));
}
inline Tensor & Tensor::sigmoid_() const {
    static auto table = globalATenDispatch().getOpTable("aten::sigmoid_(Tensor(a!) self) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this));
}
inline Tensor Tensor::sin() const {
    static auto table = globalATenDispatch().getOpTable("aten::sin(Tensor self) -> Tensor");
    return table->getOp<Tensor (const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this));
}
inline Tensor & Tensor::sin_() const {
    static auto table = globalATenDispatch().getOpTable("aten::sin_(Tensor(a!) self) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this));
}
inline Tensor Tensor::sinh() const {
    static auto table = globalATenDispatch().getOpTable("aten::sinh(Tensor self) -> Tensor");
    return table->getOp<Tensor (const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this));
}
inline Tensor & Tensor::sinh_() const {
    static auto table = globalATenDispatch().getOpTable("aten::sinh_(Tensor(a!) self) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this));
}
inline Tensor Tensor::detach() const {
    static auto table = globalATenDispatch().getOpTable("aten::detach(Tensor self) -> Tensor");
    return table->getOp<Tensor (const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this));
}
inline Tensor & Tensor::detach_() const {
    static auto table = globalATenDispatch().getOpTable("aten::detach_(Tensor(a!) self) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this));
}
inline int64_t Tensor::size(int64_t dim) const {
    static auto table = globalATenDispatch().getOpTable("aten::size.int(Tensor self, int dim) -> int");
    return table->getOp<int64_t (const Tensor &, int64_t)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), dim);
}
#ifdef BUILD_NAMEDTENSOR
inline int64_t Tensor::size(Dimname dim) const {
    static auto table = globalATenDispatch().getOpTable("aten::size.Dimname(Tensor self, Dimname dim) -> int");
    return table->getOp<int64_t (const Tensor &, Dimname)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), dim);
}
#endif
inline Tensor Tensor::slice(int64_t dim, int64_t start, int64_t end, int64_t step) const {
    static auto table = globalATenDispatch().getOpTable("aten::slice.Tensor(Tensor(a) self, int dim=0, int start=0, int end=9223372036854775807, int step=1) -> Tensor(a)");
    return table->getOp<Tensor (const Tensor &, int64_t, int64_t, int64_t, int64_t)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), dim, start, end, step);
}
inline std::tuple<Tensor,Tensor> Tensor::slogdet() const {
    static auto table = globalATenDispatch().getOpTable("aten::slogdet(Tensor self) -> (Tensor sign, Tensor logabsdet)");
    return table->getOp<std::tuple<Tensor,Tensor> (const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this));
}
inline Tensor Tensor::smm(const Tensor & mat2) const {
    static auto table = globalATenDispatch().getOpTable("aten::smm(Tensor self, Tensor mat2) -> Tensor");
    return table->getOp<Tensor (const Tensor &, const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), mat2);
}
inline Tensor Tensor::softmax(int64_t dim, c10::optional<ScalarType> dtype) const {
    static auto table = globalATenDispatch().getOpTable("aten::softmax(Tensor self, int dim, ScalarType? dtype=None) -> Tensor");
    return table->getOp<Tensor (const Tensor &, int64_t, c10::optional<ScalarType>)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), dim, dtype);
}
inline std::vector<Tensor> Tensor::split(int64_t split_size, int64_t dim) const {
    static auto table = globalATenDispatch().getOpTable("aten::split.Tensor(Tensor(a) self, int split_size, int dim=0) -> Tensor(a)[]");
    return table->getOp<std::vector<Tensor> (const Tensor &, int64_t, int64_t)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), split_size, dim);
}
inline std::vector<Tensor> Tensor::split_with_sizes(IntArrayRef split_sizes, int64_t dim) const {
    static auto table = globalATenDispatch().getOpTable("aten::split_with_sizes(Tensor self, int[] split_sizes, int dim=0) -> Tensor[]");
    return table->getOp<std::vector<Tensor> (const Tensor &, IntArrayRef, int64_t)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), split_sizes, dim);
}
inline Tensor Tensor::squeeze() const {
    static auto table = globalATenDispatch().getOpTable("aten::squeeze(Tensor(a) self) -> Tensor(a)");
    return table->getOp<Tensor (const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this));
}
inline Tensor Tensor::squeeze(int64_t dim) const {
    static auto table = globalATenDispatch().getOpTable("aten::squeeze.dim(Tensor(a) self, int dim) -> Tensor(a)");
    return table->getOp<Tensor (const Tensor &, int64_t)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), dim);
}
inline Tensor & Tensor::squeeze_() const {
    static auto table = globalATenDispatch().getOpTable("aten::squeeze_(Tensor(a!) self) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this));
}
inline Tensor & Tensor::squeeze_(int64_t dim) const {
    static auto table = globalATenDispatch().getOpTable("aten::squeeze_.dim(Tensor(a!) self, int dim) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, int64_t)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), dim);
}
inline Tensor Tensor::sspaddmm(const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) const {
    static auto table = globalATenDispatch().getOpTable("aten::sspaddmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1) -> Tensor");
    return table->getOp<Tensor (const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), mat1, mat2, beta, alpha);
}
inline Tensor Tensor::stft(int64_t n_fft, c10::optional<int64_t> hop_length, c10::optional<int64_t> win_length, const Tensor & window, bool normalized, bool onesided) const {
    static auto table = globalATenDispatch().getOpTable("aten::stft(Tensor self, int n_fft, int? hop_length=None, int? win_length=None, Tensor? window=None, bool normalized=False, bool onesided=True) -> Tensor");
    return table->getOp<Tensor (const Tensor &, int64_t, c10::optional<int64_t>, c10::optional<int64_t>, const Tensor &, bool, bool)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), n_fft, hop_length, win_length, window, normalized, onesided);
}
inline int64_t Tensor::stride(int64_t dim) const {
    static auto table = globalATenDispatch().getOpTable("aten::stride.int(Tensor self, int dim) -> int");
    return table->getOp<int64_t (const Tensor &, int64_t)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), dim);
}
#ifdef BUILD_NAMEDTENSOR
inline int64_t Tensor::stride(Dimname dim) const {
    static auto table = globalATenDispatch().getOpTable("aten::stride.Dimname(Tensor self, Dimname dim) -> int");
    return table->getOp<int64_t (const Tensor &, Dimname)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), dim);
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
    static auto table = globalATenDispatch().getOpTable("aten::sum_to_size(Tensor self, int[] size) -> Tensor");
    return table->getOp<Tensor (const Tensor &, IntArrayRef)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), size);
}
inline Tensor Tensor::sqrt() const {
    static auto table = globalATenDispatch().getOpTable("aten::sqrt(Tensor self) -> Tensor");
    return table->getOp<Tensor (const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this));
}
inline Tensor & Tensor::sqrt_() const {
    static auto table = globalATenDispatch().getOpTable("aten::sqrt_(Tensor(a!) self) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this));
}
inline Tensor Tensor::std(bool unbiased) const {
    static auto table = globalATenDispatch().getOpTable("aten::std(Tensor self, bool unbiased=True) -> Tensor");
    return table->getOp<Tensor (const Tensor &, bool)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), unbiased);
}
inline Tensor Tensor::std(IntArrayRef dim, bool unbiased, bool keepdim) const {
    static auto table = globalATenDispatch().getOpTable("aten::std.dim(Tensor self, int[1] dim, bool unbiased=True, bool keepdim=False) -> Tensor");
    return table->getOp<Tensor (const Tensor &, IntArrayRef, bool, bool)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), dim, unbiased, keepdim);
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
    static auto table = globalATenDispatch().getOpTable("aten::t(Tensor(a) self) -> Tensor(a)");
    return table->getOp<Tensor (const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this));
}
inline Tensor & Tensor::t_() const {
    static auto table = globalATenDispatch().getOpTable("aten::t_(Tensor(a!) self) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this));
}
inline Tensor Tensor::tan() const {
    static auto table = globalATenDispatch().getOpTable("aten::tan(Tensor self) -> Tensor");
    return table->getOp<Tensor (const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this));
}
inline Tensor & Tensor::tan_() const {
    static auto table = globalATenDispatch().getOpTable("aten::tan_(Tensor(a!) self) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this));
}
inline Tensor Tensor::tanh() const {
    static auto table = globalATenDispatch().getOpTable("aten::tanh(Tensor self) -> Tensor");
    return table->getOp<Tensor (const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this));
}
inline Tensor & Tensor::tanh_() const {
    static auto table = globalATenDispatch().getOpTable("aten::tanh_(Tensor(a!) self) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this));
}
inline Tensor Tensor::transpose(int64_t dim0, int64_t dim1) const {
    static auto table = globalATenDispatch().getOpTable("aten::transpose(Tensor(a) self, int dim0, int dim1) -> Tensor(a)");
    return table->getOp<Tensor (const Tensor &, int64_t, int64_t)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), dim0, dim1);
}
inline Tensor & Tensor::transpose_(int64_t dim0, int64_t dim1) const {
    static auto table = globalATenDispatch().getOpTable("aten::transpose_(Tensor(a!) self, int dim0, int dim1) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, int64_t, int64_t)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), dim0, dim1);
}
inline Tensor Tensor::flip(IntArrayRef dims) const {
    static auto table = globalATenDispatch().getOpTable("aten::flip(Tensor self, int[] dims) -> Tensor");
    return table->getOp<Tensor (const Tensor &, IntArrayRef)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), dims);
}
inline Tensor Tensor::roll(IntArrayRef shifts, IntArrayRef dims) const {
    static auto table = globalATenDispatch().getOpTable("aten::roll(Tensor self, int[1] shifts, int[1] dims=[]) -> Tensor");
    return table->getOp<Tensor (const Tensor &, IntArrayRef, IntArrayRef)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), shifts, dims);
}
inline Tensor Tensor::rot90(int64_t k, IntArrayRef dims) const {
    static auto table = globalATenDispatch().getOpTable("aten::rot90(Tensor self, int k=1, int[] dims=[0,1]) -> Tensor");
    return table->getOp<Tensor (const Tensor &, int64_t, IntArrayRef)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), k, dims);
}
inline Tensor Tensor::trunc() const {
    static auto table = globalATenDispatch().getOpTable("aten::trunc(Tensor self) -> Tensor");
    return table->getOp<Tensor (const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this));
}
inline Tensor & Tensor::trunc_() const {
    static auto table = globalATenDispatch().getOpTable("aten::trunc_(Tensor(a!) self) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this));
}
inline Tensor Tensor::type_as(const Tensor & other) const {
    static auto table = globalATenDispatch().getOpTable("aten::type_as(Tensor self, Tensor other) -> Tensor");
    return table->getOp<Tensor (const Tensor &, const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), other);
}
inline Tensor Tensor::unsqueeze(int64_t dim) const {
    static auto table = globalATenDispatch().getOpTable("aten::unsqueeze(Tensor(a) self, int dim) -> Tensor(a)");
    return table->getOp<Tensor (const Tensor &, int64_t)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), dim);
}
inline Tensor & Tensor::unsqueeze_(int64_t dim) const {
    static auto table = globalATenDispatch().getOpTable("aten::unsqueeze_(Tensor(a!) self, int dim) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, int64_t)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), dim);
}
inline Tensor Tensor::var(bool unbiased) const {
    static auto table = globalATenDispatch().getOpTable("aten::var(Tensor self, bool unbiased=True) -> Tensor");
    return table->getOp<Tensor (const Tensor &, bool)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), unbiased);
}
inline Tensor Tensor::var(IntArrayRef dim, bool unbiased, bool keepdim) const {
    static auto table = globalATenDispatch().getOpTable("aten::var.dim(Tensor self, int[1] dim, bool unbiased=True, bool keepdim=False) -> Tensor");
    return table->getOp<Tensor (const Tensor &, IntArrayRef, bool, bool)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), dim, unbiased, keepdim);
}
inline Tensor Tensor::view_as(const Tensor & other) const {
    static auto table = globalATenDispatch().getOpTable("aten::view_as(Tensor self, Tensor other) -> Tensor");
    return table->getOp<Tensor (const Tensor &, const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), other);
}
inline Tensor Tensor::where(const Tensor & condition, const Tensor & other) const {
    static auto table = globalATenDispatch().getOpTable("aten::where.self(Tensor condition, Tensor self, Tensor other) -> Tensor");
    return table->getOp<Tensor (const Tensor &, const Tensor &, const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(condition, const_cast<Tensor&>(*this), other);
}
inline Tensor Tensor::norm(c10::optional<Scalar> p, ScalarType dtype) const {
    static auto table = globalATenDispatch().getOpTable("aten::norm.ScalarOpt_dtype(Tensor self, Scalar? p, *, ScalarType dtype) -> Tensor");
    return table->getOp<Tensor (const Tensor &, c10::optional<Scalar>, ScalarType)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), p, dtype);
}
inline Tensor Tensor::norm(Scalar p) const {
    static auto table = globalATenDispatch().getOpTable("aten::norm.Scalar(Tensor self, Scalar p=2) -> Tensor");
    return table->getOp<Tensor (const Tensor &, Scalar)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), p);
}
inline Tensor Tensor::norm(c10::optional<Scalar> p, IntArrayRef dim, bool keepdim, ScalarType dtype) const {
    static auto table = globalATenDispatch().getOpTable("aten::norm.ScalarOpt_dim_dtype(Tensor self, Scalar? p, int[1] dim, bool keepdim, *, ScalarType dtype) -> Tensor");
    return table->getOp<Tensor (const Tensor &, c10::optional<Scalar>, IntArrayRef, bool, ScalarType)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), p, dim, keepdim, dtype);
}
inline Tensor Tensor::norm(c10::optional<Scalar> p, IntArrayRef dim, bool keepdim) const {
    static auto table = globalATenDispatch().getOpTable("aten::norm.ScalarOpt_dim(Tensor self, Scalar? p, int[1] dim, bool keepdim=False) -> Tensor");
    return table->getOp<Tensor (const Tensor &, c10::optional<Scalar>, IntArrayRef, bool)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), p, dim, keepdim);
}
inline Tensor Tensor::clone() const {
    static auto table = globalATenDispatch().getOpTable("aten::clone(Tensor self) -> Tensor");
    return table->getOp<Tensor (const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this));
}
inline Tensor & Tensor::resize_as_(const Tensor & the_template) const {
    static auto table = globalATenDispatch().getOpTable("aten::resize_as_(Tensor(a!) self, Tensor the_template) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), the_template);
}
inline Tensor Tensor::pow(Scalar exponent) const {
    static auto table = globalATenDispatch().getOpTable("aten::pow.Tensor_Scalar(Tensor self, Scalar exponent) -> Tensor");
    return table->getOp<Tensor (const Tensor &, Scalar)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), exponent);
}
inline Tensor & Tensor::zero_() const {
    static auto table = globalATenDispatch().getOpTable("aten::zero_(Tensor(a!) self) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this));
}
inline Tensor Tensor::sub(const Tensor & other, Scalar alpha) const {
    static auto table = globalATenDispatch().getOpTable("aten::sub.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor");
    return table->getOp<Tensor (const Tensor &, const Tensor &, Scalar)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), other, alpha);
}
inline Tensor & Tensor::sub_(const Tensor & other, Scalar alpha) const {
    static auto table = globalATenDispatch().getOpTable("aten::sub_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, const Tensor &, Scalar)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), other, alpha);
}
inline Tensor Tensor::sub(Scalar other, Scalar alpha) const {
    static auto table = globalATenDispatch().getOpTable("aten::sub.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> Tensor");
    return table->getOp<Tensor (const Tensor &, Scalar, Scalar)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), other, alpha);
}
inline Tensor & Tensor::sub_(Scalar other, Scalar alpha) const {
    static auto table = globalATenDispatch().getOpTable("aten::sub_.Scalar(Tensor(a!) self, Scalar other, Scalar alpha=1) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, Scalar, Scalar)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), other, alpha);
}
inline Tensor Tensor::addmm(const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) const {
    static auto table = globalATenDispatch().getOpTable("aten::addmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1) -> Tensor");
    return table->getOp<Tensor (const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), mat1, mat2, beta, alpha);
}
inline Tensor & Tensor::addmm_(const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) const {
    static auto table = globalATenDispatch().getOpTable("aten::addmm_(Tensor(a!) self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, const Tensor &, const Tensor &, Scalar, Scalar)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), mat1, mat2, beta, alpha);
}
inline Tensor & Tensor::sparse_resize_(IntArrayRef size, int64_t sparse_dim, int64_t dense_dim) const {
    static auto table = globalATenDispatch().getOpTable("aten::sparse_resize_(Tensor(a!) self, int[] size, int sparse_dim, int dense_dim) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, IntArrayRef, int64_t, int64_t)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), size, sparse_dim, dense_dim);
}
inline Tensor & Tensor::sparse_resize_and_clear_(IntArrayRef size, int64_t sparse_dim, int64_t dense_dim) const {
    static auto table = globalATenDispatch().getOpTable("aten::sparse_resize_and_clear_(Tensor(a!) self, int[] size, int sparse_dim, int dense_dim) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, IntArrayRef, int64_t, int64_t)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), size, sparse_dim, dense_dim);
}
inline Tensor Tensor::sparse_mask(const Tensor & mask) const {
    static auto table = globalATenDispatch().getOpTable("aten::sparse_mask(Tensor self, Tensor mask) -> Tensor");
    return table->getOp<Tensor (const Tensor &, const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), mask);
}
inline Tensor Tensor::to_dense() const {
    static auto table = globalATenDispatch().getOpTable("aten::to_dense(Tensor self) -> Tensor");
    return table->getOp<Tensor (const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this));
}
inline int64_t Tensor::sparse_dim() const {
    static auto table = globalATenDispatch().getOpTable("aten::sparse_dim(Tensor self) -> int");
    return table->getOp<int64_t (const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this));
}
inline int64_t Tensor::_dimI() const {
    static auto table = globalATenDispatch().getOpTable("aten::_dimI(Tensor self) -> int");
    return table->getOp<int64_t (const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this));
}
inline int64_t Tensor::dense_dim() const {
    static auto table = globalATenDispatch().getOpTable("aten::dense_dim(Tensor self) -> int");
    return table->getOp<int64_t (const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this));
}
inline int64_t Tensor::_dimV() const {
    static auto table = globalATenDispatch().getOpTable("aten::_dimV(Tensor self) -> int");
    return table->getOp<int64_t (const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this));
}
inline int64_t Tensor::_nnz() const {
    static auto table = globalATenDispatch().getOpTable("aten::_nnz(Tensor self) -> int");
    return table->getOp<int64_t (const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this));
}
inline Tensor Tensor::coalesce() const {
    static auto table = globalATenDispatch().getOpTable("aten::coalesce(Tensor self) -> Tensor");
    return table->getOp<Tensor (const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this));
}
inline bool Tensor::is_coalesced() const {
    static auto table = globalATenDispatch().getOpTable("aten::is_coalesced(Tensor self) -> bool");
    return table->getOp<bool (const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this));
}
inline Tensor Tensor::_indices() const {
    static auto table = globalATenDispatch().getOpTable("aten::_indices(Tensor(a) self) -> Tensor(a)");
    return table->getOp<Tensor (const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this));
}
inline Tensor Tensor::_values() const {
    static auto table = globalATenDispatch().getOpTable("aten::_values(Tensor(a) self) -> Tensor(a)");
    return table->getOp<Tensor (const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this));
}
inline Tensor & Tensor::_coalesced_(bool coalesced) const {
    static auto table = globalATenDispatch().getOpTable("aten::_coalesced_(Tensor(a!) self, bool coalesced) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, bool)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), coalesced);
}
inline Tensor Tensor::indices() const {
    static auto table = globalATenDispatch().getOpTable("aten::indices(Tensor(a) self) -> Tensor(a)");
    return table->getOp<Tensor (const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this));
}
inline Tensor Tensor::values() const {
    static auto table = globalATenDispatch().getOpTable("aten::values(Tensor(a) self) -> Tensor(a)");
    return table->getOp<Tensor (const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this));
}
inline int64_t Tensor::numel() const {
    static auto table = globalATenDispatch().getOpTable("aten::numel(Tensor self) -> int");
    return table->getOp<int64_t (const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this));
}
inline std::vector<Tensor> Tensor::unbind(int64_t dim) const {
    static auto table = globalATenDispatch().getOpTable("aten::unbind(Tensor(a) self, int dim=0) -> Tensor(a)[]");
    return table->getOp<std::vector<Tensor> (const Tensor &, int64_t)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), dim);
}
inline Tensor Tensor::to_sparse(int64_t sparse_dim) const {
    static auto table = globalATenDispatch().getOpTable("aten::to_sparse.sparse_dim(Tensor self, int sparse_dim) -> Tensor");
    return table->getOp<Tensor (const Tensor &, int64_t)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), sparse_dim);
}
inline Tensor Tensor::to_sparse() const {
    static auto table = globalATenDispatch().getOpTable("aten::to_sparse(Tensor self) -> Tensor");
    return table->getOp<Tensor (const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this));
}
inline Tensor Tensor::to_mkldnn() const {
    static auto table = globalATenDispatch().getOpTable("aten::to_mkldnn(Tensor self) -> Tensor");
    return table->getOp<Tensor (const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this));
}
inline Tensor Tensor::dequantize() const {
    static auto table = globalATenDispatch().getOpTable("aten::dequantize(Tensor self) -> Tensor");
    return table->getOp<Tensor (const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this));
}
inline double Tensor::q_scale() const {
    static auto table = globalATenDispatch().getOpTable("aten::q_scale(Tensor self) -> float");
    return table->getOp<double (const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this));
}
inline int64_t Tensor::q_zero_point() const {
    static auto table = globalATenDispatch().getOpTable("aten::q_zero_point(Tensor self) -> int");
    return table->getOp<int64_t (const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this));
}
inline Tensor Tensor::int_repr() const {
    static auto table = globalATenDispatch().getOpTable("aten::int_repr(Tensor self) -> Tensor");
    return table->getOp<Tensor (const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this));
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
    static auto table = globalATenDispatch().getOpTable("aten::to.other(Tensor self, Tensor other, bool non_blocking=False, bool copy=False) -> Tensor");
    return table->getOp<Tensor (const Tensor &, const Tensor &, bool, bool)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), other, non_blocking, copy);
}
inline Scalar Tensor::item() const {
    static auto table = globalATenDispatch().getOpTable("aten::item(Tensor self) -> Scalar");
    return table->getOp<Scalar (const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this));
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
    static auto table = globalATenDispatch().getOpTable("aten::set_.source_Tensor(Tensor(a!) self, Tensor source) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), source);
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
    static auto table = globalATenDispatch().getOpTable("aten::is_set_to(Tensor self, Tensor tensor) -> bool");
    return table->getOp<bool (const Tensor &, const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), tensor);
}
inline Tensor & Tensor::masked_fill_(const Tensor & mask, Scalar value) const {
    static auto table = globalATenDispatch().getOpTable("aten::masked_fill_.Scalar(Tensor(a!) self, Tensor mask, Scalar value) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, const Tensor &, Scalar)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), mask, value);
}
inline Tensor Tensor::masked_fill(const Tensor & mask, Scalar value) const {
    static auto table = globalATenDispatch().getOpTable("aten::masked_fill.Scalar(Tensor self, Tensor mask, Scalar value) -> Tensor");
    return table->getOp<Tensor (const Tensor &, const Tensor &, Scalar)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), mask, value);
}
inline Tensor & Tensor::masked_fill_(const Tensor & mask, const Tensor & value) const {
    static auto table = globalATenDispatch().getOpTable("aten::masked_fill_.Tensor(Tensor(a!) self, Tensor mask, Tensor value) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, const Tensor &, const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), mask, value);
}
inline Tensor Tensor::masked_fill(const Tensor & mask, const Tensor & value) const {
    static auto table = globalATenDispatch().getOpTable("aten::masked_fill.Tensor(Tensor self, Tensor mask, Tensor value) -> Tensor");
    return table->getOp<Tensor (const Tensor &, const Tensor &, const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), mask, value);
}
inline Tensor & Tensor::masked_scatter_(const Tensor & mask, const Tensor & source) const {
    static auto table = globalATenDispatch().getOpTable("aten::masked_scatter_(Tensor(a!) self, Tensor mask, Tensor source) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, const Tensor &, const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), mask, source);
}
inline Tensor Tensor::masked_scatter(const Tensor & mask, const Tensor & source) const {
    static auto table = globalATenDispatch().getOpTable("aten::masked_scatter(Tensor self, Tensor mask, Tensor source) -> Tensor");
    return table->getOp<Tensor (const Tensor &, const Tensor &, const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), mask, source);
}
inline Tensor Tensor::view(IntArrayRef size) const {
    static auto table = globalATenDispatch().getOpTable("aten::view(Tensor(a) self, int[] size) -> Tensor(a)");
    return table->getOp<Tensor (const Tensor &, IntArrayRef)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), size);
}
inline Tensor & Tensor::put_(const Tensor & index, const Tensor & source, bool accumulate) const {
    static auto table = globalATenDispatch().getOpTable("aten::put_(Tensor(a!) self, Tensor index, Tensor source, bool accumulate=False) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, const Tensor &, const Tensor &, bool)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), index, source, accumulate);
}
inline Tensor & Tensor::index_add_(int64_t dim, const Tensor & index, const Tensor & source) const {
    static auto table = globalATenDispatch().getOpTable("aten::index_add_(Tensor(a!) self, int dim, Tensor index, Tensor source) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, int64_t, const Tensor &, const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), dim, index, source);
}
inline Tensor Tensor::index_add(int64_t dim, const Tensor & index, const Tensor & source) const {
    static auto table = globalATenDispatch().getOpTable("aten::index_add(Tensor self, int dim, Tensor index, Tensor source) -> Tensor");
    return table->getOp<Tensor (const Tensor &, int64_t, const Tensor &, const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), dim, index, source);
}
inline Tensor & Tensor::index_fill_(int64_t dim, const Tensor & index, Scalar value) const {
    static auto table = globalATenDispatch().getOpTable("aten::index_fill_.Scalar(Tensor(a!) self, int dim, Tensor index, Scalar value) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, int64_t, const Tensor &, Scalar)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), dim, index, value);
}
inline Tensor Tensor::index_fill(int64_t dim, const Tensor & index, Scalar value) const {
    static auto table = globalATenDispatch().getOpTable("aten::index_fill.Scalar(Tensor self, int dim, Tensor index, Scalar value) -> Tensor");
    return table->getOp<Tensor (const Tensor &, int64_t, const Tensor &, Scalar)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), dim, index, value);
}
inline Tensor & Tensor::index_fill_(int64_t dim, const Tensor & index, const Tensor & value) const {
    static auto table = globalATenDispatch().getOpTable("aten::index_fill_.Tensor(Tensor(a!) self, int dim, Tensor index, Tensor value) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, int64_t, const Tensor &, const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), dim, index, value);
}
inline Tensor Tensor::index_fill(int64_t dim, const Tensor & index, const Tensor & value) const {
    static auto table = globalATenDispatch().getOpTable("aten::index_fill.Tensor(Tensor self, int dim, Tensor index, Tensor value) -> Tensor");
    return table->getOp<Tensor (const Tensor &, int64_t, const Tensor &, const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), dim, index, value);
}
inline Tensor & Tensor::scatter_(int64_t dim, const Tensor & index, const Tensor & src) const {
    static auto table = globalATenDispatch().getOpTable("aten::scatter_.src(Tensor(a!) self, int dim, Tensor index, Tensor src) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, int64_t, const Tensor &, const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), dim, index, src);
}
inline Tensor Tensor::scatter(int64_t dim, const Tensor & index, const Tensor & src) const {
    static auto table = globalATenDispatch().getOpTable("aten::scatter.src(Tensor self, int dim, Tensor index, Tensor src) -> Tensor");
    return table->getOp<Tensor (const Tensor &, int64_t, const Tensor &, const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), dim, index, src);
}
inline Tensor & Tensor::scatter_(int64_t dim, const Tensor & index, Scalar value) const {
    static auto table = globalATenDispatch().getOpTable("aten::scatter_.value(Tensor(a!) self, int dim, Tensor index, Scalar value) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, int64_t, const Tensor &, Scalar)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), dim, index, value);
}
inline Tensor Tensor::scatter(int64_t dim, const Tensor & index, Scalar value) const {
    static auto table = globalATenDispatch().getOpTable("aten::scatter.value(Tensor self, int dim, Tensor index, Scalar value) -> Tensor");
    return table->getOp<Tensor (const Tensor &, int64_t, const Tensor &, Scalar)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), dim, index, value);
}
inline Tensor & Tensor::scatter_add_(int64_t dim, const Tensor & index, const Tensor & src) const {
    static auto table = globalATenDispatch().getOpTable("aten::scatter_add_(Tensor(a!) self, int dim, Tensor index, Tensor src) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, int64_t, const Tensor &, const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), dim, index, src);
}
inline Tensor Tensor::scatter_add(int64_t dim, const Tensor & index, const Tensor & src) const {
    static auto table = globalATenDispatch().getOpTable("aten::scatter_add(Tensor self, int dim, Tensor index, Tensor src) -> Tensor");
    return table->getOp<Tensor (const Tensor &, int64_t, const Tensor &, const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), dim, index, src);
}
inline Tensor & Tensor::lt_(Scalar other) const {
    static auto table = globalATenDispatch().getOpTable("aten::lt_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, Scalar)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), other);
}
inline Tensor & Tensor::lt_(const Tensor & other) const {
    static auto table = globalATenDispatch().getOpTable("aten::lt_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), other);
}
inline Tensor & Tensor::gt_(Scalar other) const {
    static auto table = globalATenDispatch().getOpTable("aten::gt_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, Scalar)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), other);
}
inline Tensor & Tensor::gt_(const Tensor & other) const {
    static auto table = globalATenDispatch().getOpTable("aten::gt_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), other);
}
inline Tensor & Tensor::le_(Scalar other) const {
    static auto table = globalATenDispatch().getOpTable("aten::le_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, Scalar)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), other);
}
inline Tensor & Tensor::le_(const Tensor & other) const {
    static auto table = globalATenDispatch().getOpTable("aten::le_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), other);
}
inline Tensor & Tensor::ge_(Scalar other) const {
    static auto table = globalATenDispatch().getOpTable("aten::ge_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, Scalar)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), other);
}
inline Tensor & Tensor::ge_(const Tensor & other) const {
    static auto table = globalATenDispatch().getOpTable("aten::ge_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), other);
}
inline Tensor & Tensor::eq_(Scalar other) const {
    static auto table = globalATenDispatch().getOpTable("aten::eq_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, Scalar)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), other);
}
inline Tensor & Tensor::eq_(const Tensor & other) const {
    static auto table = globalATenDispatch().getOpTable("aten::eq_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), other);
}
inline Tensor & Tensor::ne_(Scalar other) const {
    static auto table = globalATenDispatch().getOpTable("aten::ne_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, Scalar)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), other);
}
inline Tensor & Tensor::ne_(const Tensor & other) const {
    static auto table = globalATenDispatch().getOpTable("aten::ne_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), other);
}
inline Tensor Tensor::__and__(Scalar other) const {
    static auto table = globalATenDispatch().getOpTable("aten::__and__.Scalar(Tensor self, Scalar other) -> Tensor");
    return table->getOp<Tensor (const Tensor &, Scalar)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), other);
}
inline Tensor Tensor::__and__(const Tensor & other) const {
    static auto table = globalATenDispatch().getOpTable("aten::__and__.Tensor(Tensor self, Tensor other) -> Tensor");
    return table->getOp<Tensor (const Tensor &, const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), other);
}
inline Tensor & Tensor::__iand__(Scalar other) const {
    static auto table = globalATenDispatch().getOpTable("aten::__iand__.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, Scalar)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), other);
}
inline Tensor & Tensor::__iand__(const Tensor & other) const {
    static auto table = globalATenDispatch().getOpTable("aten::__iand__.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), other);
}
inline Tensor Tensor::__or__(Scalar other) const {
    static auto table = globalATenDispatch().getOpTable("aten::__or__.Scalar(Tensor self, Scalar other) -> Tensor");
    return table->getOp<Tensor (const Tensor &, Scalar)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), other);
}
inline Tensor Tensor::__or__(const Tensor & other) const {
    static auto table = globalATenDispatch().getOpTable("aten::__or__.Tensor(Tensor self, Tensor other) -> Tensor");
    return table->getOp<Tensor (const Tensor &, const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), other);
}
inline Tensor & Tensor::__ior__(Scalar other) const {
    static auto table = globalATenDispatch().getOpTable("aten::__ior__.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, Scalar)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), other);
}
inline Tensor & Tensor::__ior__(const Tensor & other) const {
    static auto table = globalATenDispatch().getOpTable("aten::__ior__.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), other);
}
inline Tensor Tensor::__xor__(Scalar other) const {
    static auto table = globalATenDispatch().getOpTable("aten::__xor__.Scalar(Tensor self, Scalar other) -> Tensor");
    return table->getOp<Tensor (const Tensor &, Scalar)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), other);
}
inline Tensor Tensor::__xor__(const Tensor & other) const {
    static auto table = globalATenDispatch().getOpTable("aten::__xor__.Tensor(Tensor self, Tensor other) -> Tensor");
    return table->getOp<Tensor (const Tensor &, const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), other);
}
inline Tensor & Tensor::__ixor__(Scalar other) const {
    static auto table = globalATenDispatch().getOpTable("aten::__ixor__.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, Scalar)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), other);
}
inline Tensor & Tensor::__ixor__(const Tensor & other) const {
    static auto table = globalATenDispatch().getOpTable("aten::__ixor__.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), other);
}
inline Tensor Tensor::__lshift__(Scalar other) const {
    static auto table = globalATenDispatch().getOpTable("aten::__lshift__.Scalar(Tensor self, Scalar other) -> Tensor");
    return table->getOp<Tensor (const Tensor &, Scalar)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), other);
}
inline Tensor Tensor::__lshift__(const Tensor & other) const {
    static auto table = globalATenDispatch().getOpTable("aten::__lshift__.Tensor(Tensor self, Tensor other) -> Tensor");
    return table->getOp<Tensor (const Tensor &, const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), other);
}
inline Tensor & Tensor::__ilshift__(Scalar other) const {
    static auto table = globalATenDispatch().getOpTable("aten::__ilshift__.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, Scalar)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), other);
}
inline Tensor & Tensor::__ilshift__(const Tensor & other) const {
    static auto table = globalATenDispatch().getOpTable("aten::__ilshift__.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), other);
}
inline Tensor Tensor::__rshift__(Scalar other) const {
    static auto table = globalATenDispatch().getOpTable("aten::__rshift__.Scalar(Tensor self, Scalar other) -> Tensor");
    return table->getOp<Tensor (const Tensor &, Scalar)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), other);
}
inline Tensor Tensor::__rshift__(const Tensor & other) const {
    static auto table = globalATenDispatch().getOpTable("aten::__rshift__.Tensor(Tensor self, Tensor other) -> Tensor");
    return table->getOp<Tensor (const Tensor &, const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), other);
}
inline Tensor & Tensor::__irshift__(Scalar other) const {
    static auto table = globalATenDispatch().getOpTable("aten::__irshift__.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, Scalar)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), other);
}
inline Tensor & Tensor::__irshift__(const Tensor & other) const {
    static auto table = globalATenDispatch().getOpTable("aten::__irshift__.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), other);
}
inline Tensor & Tensor::lgamma_() const {
    static auto table = globalATenDispatch().getOpTable("aten::lgamma_(Tensor(a!) self) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this));
}
inline Tensor & Tensor::atan2_(const Tensor & other) const {
    static auto table = globalATenDispatch().getOpTable("aten::atan2_(Tensor(a!) self, Tensor other) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), other);
}
inline Tensor & Tensor::tril_(int64_t diagonal) const {
    static auto table = globalATenDispatch().getOpTable("aten::tril_(Tensor(a!) self, int diagonal=0) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, int64_t)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), diagonal);
}
inline Tensor & Tensor::triu_(int64_t diagonal) const {
    static auto table = globalATenDispatch().getOpTable("aten::triu_(Tensor(a!) self, int diagonal=0) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, int64_t)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), diagonal);
}
inline Tensor & Tensor::digamma_() const {
    static auto table = globalATenDispatch().getOpTable("aten::digamma_(Tensor(a!) self) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this));
}
inline Tensor & Tensor::polygamma_(int64_t n) const {
    static auto table = globalATenDispatch().getOpTable("aten::polygamma_(Tensor(a!) self, int n) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, int64_t)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), n);
}
inline Tensor & Tensor::erfinv_() const {
    static auto table = globalATenDispatch().getOpTable("aten::erfinv_(Tensor(a!) self) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this));
}
inline Tensor & Tensor::renorm_(Scalar p, int64_t dim, Scalar maxnorm) const {
    static auto table = globalATenDispatch().getOpTable("aten::renorm_(Tensor(a!) self, Scalar p, int dim, Scalar maxnorm) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, Scalar, int64_t, Scalar)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), p, dim, maxnorm);
}
inline Tensor & Tensor::pow_(Scalar exponent) const {
    static auto table = globalATenDispatch().getOpTable("aten::pow_.Scalar(Tensor(a!) self, Scalar exponent) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, Scalar)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), exponent);
}
inline Tensor & Tensor::pow_(const Tensor & exponent) const {
    static auto table = globalATenDispatch().getOpTable("aten::pow_.Tensor(Tensor(a!) self, Tensor exponent) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), exponent);
}
inline Tensor & Tensor::lerp_(const Tensor & end, Scalar weight) const {
    static auto table = globalATenDispatch().getOpTable("aten::lerp_.Scalar(Tensor(a!) self, Tensor end, Scalar weight) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, const Tensor &, Scalar)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), end, weight);
}
inline Tensor & Tensor::lerp_(const Tensor & end, const Tensor & weight) const {
    static auto table = globalATenDispatch().getOpTable("aten::lerp_.Tensor(Tensor(a!) self, Tensor end, Tensor weight) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, const Tensor &, const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), end, weight);
}
inline Tensor & Tensor::sign_() const {
    static auto table = globalATenDispatch().getOpTable("aten::sign_(Tensor(a!) self) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this));
}
inline Tensor & Tensor::fmod_(Scalar other) const {
    static auto table = globalATenDispatch().getOpTable("aten::fmod_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, Scalar)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), other);
}
inline Tensor & Tensor::fmod_(const Tensor & other) const {
    static auto table = globalATenDispatch().getOpTable("aten::fmod_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), other);
}
inline Tensor & Tensor::remainder_(Scalar other) const {
    static auto table = globalATenDispatch().getOpTable("aten::remainder_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, Scalar)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), other);
}
inline Tensor & Tensor::remainder_(const Tensor & other) const {
    static auto table = globalATenDispatch().getOpTable("aten::remainder_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), other);
}
inline Tensor & Tensor::addbmm_(const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) const {
    static auto table = globalATenDispatch().getOpTable("aten::addbmm_(Tensor(a!) self, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, const Tensor &, const Tensor &, Scalar, Scalar)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), batch1, batch2, beta, alpha);
}
inline Tensor Tensor::addbmm(const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) const {
    static auto table = globalATenDispatch().getOpTable("aten::addbmm(Tensor self, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1) -> Tensor");
    return table->getOp<Tensor (const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), batch1, batch2, beta, alpha);
}
inline Tensor & Tensor::addcdiv_(const Tensor & tensor1, const Tensor & tensor2, Scalar value) const {
    static auto table = globalATenDispatch().getOpTable("aten::addcdiv_(Tensor(a!) self, Tensor tensor1, Tensor tensor2, *, Scalar value=1) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, const Tensor &, const Tensor &, Scalar)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), tensor1, tensor2, value);
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
    static auto table = globalATenDispatch().getOpTable("aten::diag(Tensor self, int diagonal=0) -> Tensor");
    return table->getOp<Tensor (const Tensor &, int64_t)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), diagonal);
}
inline Tensor Tensor::cross(const Tensor & other, c10::optional<int64_t> dim) const {
    static auto table = globalATenDispatch().getOpTable("aten::cross(Tensor self, Tensor other, int? dim=None) -> Tensor");
    return table->getOp<Tensor (const Tensor &, const Tensor &, c10::optional<int64_t>)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), other, dim);
}
inline Tensor Tensor::triu(int64_t diagonal) const {
    static auto table = globalATenDispatch().getOpTable("aten::triu(Tensor self, int diagonal=0) -> Tensor");
    return table->getOp<Tensor (const Tensor &, int64_t)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), diagonal);
}
inline Tensor Tensor::tril(int64_t diagonal) const {
    static auto table = globalATenDispatch().getOpTable("aten::tril(Tensor self, int diagonal=0) -> Tensor");
    return table->getOp<Tensor (const Tensor &, int64_t)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), diagonal);
}
inline Tensor Tensor::trace() const {
    static auto table = globalATenDispatch().getOpTable("aten::trace(Tensor self) -> Tensor");
    return table->getOp<Tensor (const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this));
}
inline Tensor Tensor::ne(Scalar other) const {
    static auto table = globalATenDispatch().getOpTable("aten::ne.Scalar(Tensor self, Scalar other) -> Tensor");
    return table->getOp<Tensor (const Tensor &, Scalar)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), other);
}
inline Tensor Tensor::ne(const Tensor & other) const {
    static auto table = globalATenDispatch().getOpTable("aten::ne.Tensor(Tensor self, Tensor other) -> Tensor");
    return table->getOp<Tensor (const Tensor &, const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), other);
}
inline Tensor Tensor::eq(Scalar other) const {
    static auto table = globalATenDispatch().getOpTable("aten::eq.Scalar(Tensor self, Scalar other) -> Tensor");
    return table->getOp<Tensor (const Tensor &, Scalar)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), other);
}
inline Tensor Tensor::eq(const Tensor & other) const {
    static auto table = globalATenDispatch().getOpTable("aten::eq.Tensor(Tensor self, Tensor other) -> Tensor");
    return table->getOp<Tensor (const Tensor &, const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), other);
}
inline Tensor Tensor::ge(Scalar other) const {
    static auto table = globalATenDispatch().getOpTable("aten::ge.Scalar(Tensor self, Scalar other) -> Tensor");
    return table->getOp<Tensor (const Tensor &, Scalar)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), other);
}
inline Tensor Tensor::ge(const Tensor & other) const {
    static auto table = globalATenDispatch().getOpTable("aten::ge.Tensor(Tensor self, Tensor other) -> Tensor");
    return table->getOp<Tensor (const Tensor &, const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), other);
}
inline Tensor Tensor::le(Scalar other) const {
    static auto table = globalATenDispatch().getOpTable("aten::le.Scalar(Tensor self, Scalar other) -> Tensor");
    return table->getOp<Tensor (const Tensor &, Scalar)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), other);
}
inline Tensor Tensor::le(const Tensor & other) const {
    static auto table = globalATenDispatch().getOpTable("aten::le.Tensor(Tensor self, Tensor other) -> Tensor");
    return table->getOp<Tensor (const Tensor &, const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), other);
}
inline Tensor Tensor::gt(Scalar other) const {
    static auto table = globalATenDispatch().getOpTable("aten::gt.Scalar(Tensor self, Scalar other) -> Tensor");
    return table->getOp<Tensor (const Tensor &, Scalar)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), other);
}
inline Tensor Tensor::gt(const Tensor & other) const {
    static auto table = globalATenDispatch().getOpTable("aten::gt.Tensor(Tensor self, Tensor other) -> Tensor");
    return table->getOp<Tensor (const Tensor &, const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), other);
}
inline Tensor Tensor::lt(Scalar other) const {
    static auto table = globalATenDispatch().getOpTable("aten::lt.Scalar(Tensor self, Scalar other) -> Tensor");
    return table->getOp<Tensor (const Tensor &, Scalar)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), other);
}
inline Tensor Tensor::lt(const Tensor & other) const {
    static auto table = globalATenDispatch().getOpTable("aten::lt.Tensor(Tensor self, Tensor other) -> Tensor");
    return table->getOp<Tensor (const Tensor &, const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), other);
}
inline Tensor Tensor::take(const Tensor & index) const {
    static auto table = globalATenDispatch().getOpTable("aten::take(Tensor self, Tensor index) -> Tensor");
    return table->getOp<Tensor (const Tensor &, const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), index);
}
inline Tensor Tensor::index_select(int64_t dim, const Tensor & index) const {
    static auto table = globalATenDispatch().getOpTable("aten::index_select(Tensor self, int dim, Tensor index) -> Tensor");
    return table->getOp<Tensor (const Tensor &, int64_t, const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), dim, index);
}
inline Tensor Tensor::masked_select(const Tensor & mask) const {
    static auto table = globalATenDispatch().getOpTable("aten::masked_select(Tensor self, Tensor mask) -> Tensor");
    return table->getOp<Tensor (const Tensor &, const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), mask);
}
inline Tensor Tensor::nonzero() const {
    static auto table = globalATenDispatch().getOpTable("aten::nonzero(Tensor self) -> Tensor");
    return table->getOp<Tensor (const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this));
}
inline std::vector<Tensor> Tensor::nonzero_numpy() const {
    static auto table = globalATenDispatch().getOpTable("aten::nonzero_numpy(Tensor self) -> Tensor[]");
    return table->getOp<std::vector<Tensor> (const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this));
}
inline Tensor Tensor::gather(int64_t dim, const Tensor & index, bool sparse_grad) const {
    static auto table = globalATenDispatch().getOpTable("aten::gather(Tensor self, int dim, Tensor index, *, bool sparse_grad=False) -> Tensor");
    return table->getOp<Tensor (const Tensor &, int64_t, const Tensor &, bool)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), dim, index, sparse_grad);
}
inline Tensor Tensor::addcmul(const Tensor & tensor1, const Tensor & tensor2, Scalar value) const {
    static auto table = globalATenDispatch().getOpTable("aten::addcmul(Tensor self, Tensor tensor1, Tensor tensor2, *, Scalar value=1) -> Tensor");
    return table->getOp<Tensor (const Tensor &, const Tensor &, const Tensor &, Scalar)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), tensor1, tensor2, value);
}
inline Tensor & Tensor::addcmul_(const Tensor & tensor1, const Tensor & tensor2, Scalar value) const {
    static auto table = globalATenDispatch().getOpTable("aten::addcmul_(Tensor(a!) self, Tensor tensor1, Tensor tensor2, *, Scalar value=1) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, const Tensor &, const Tensor &, Scalar)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), tensor1, tensor2, value);
}
inline Tensor Tensor::addcdiv(const Tensor & tensor1, const Tensor & tensor2, Scalar value) const {
    static auto table = globalATenDispatch().getOpTable("aten::addcdiv(Tensor self, Tensor tensor1, Tensor tensor2, *, Scalar value=1) -> Tensor");
    return table->getOp<Tensor (const Tensor &, const Tensor &, const Tensor &, Scalar)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), tensor1, tensor2, value);
}
inline std::tuple<Tensor,Tensor> Tensor::lstsq(const Tensor & A) const {
    static auto table = globalATenDispatch().getOpTable("aten::lstsq(Tensor self, Tensor A) -> (Tensor solution, Tensor QR)");
    return table->getOp<std::tuple<Tensor,Tensor> (const Tensor &, const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), A);
}
inline std::tuple<Tensor,Tensor> Tensor::triangular_solve(const Tensor & A, bool upper, bool transpose, bool unitriangular) const {
    static auto table = globalATenDispatch().getOpTable("aten::triangular_solve(Tensor self, Tensor A, bool upper=True, bool transpose=False, bool unitriangular=False) -> (Tensor solution, Tensor cloned_coefficient)");
    return table->getOp<std::tuple<Tensor,Tensor> (const Tensor &, const Tensor &, bool, bool, bool)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), A, upper, transpose, unitriangular);
}
inline std::tuple<Tensor,Tensor> Tensor::symeig(bool eigenvectors, bool upper) const {
    static auto table = globalATenDispatch().getOpTable("aten::symeig(Tensor self, bool eigenvectors=False, bool upper=True) -> (Tensor eigenvalues, Tensor eigenvectors)");
    return table->getOp<std::tuple<Tensor,Tensor> (const Tensor &, bool, bool)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), eigenvectors, upper);
}
inline std::tuple<Tensor,Tensor> Tensor::eig(bool eigenvectors) const {
    static auto table = globalATenDispatch().getOpTable("aten::eig(Tensor self, bool eigenvectors=False) -> (Tensor eigenvalues, Tensor eigenvectors)");
    return table->getOp<std::tuple<Tensor,Tensor> (const Tensor &, bool)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), eigenvectors);
}
inline std::tuple<Tensor,Tensor,Tensor> Tensor::svd(bool some, bool compute_uv) const {
    static auto table = globalATenDispatch().getOpTable("aten::svd(Tensor self, bool some=True, bool compute_uv=True) -> (Tensor U, Tensor S, Tensor V)");
    return table->getOp<std::tuple<Tensor,Tensor,Tensor> (const Tensor &, bool, bool)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), some, compute_uv);
}
inline Tensor Tensor::cholesky(bool upper) const {
    static auto table = globalATenDispatch().getOpTable("aten::cholesky(Tensor self, bool upper=False) -> Tensor");
    return table->getOp<Tensor (const Tensor &, bool)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), upper);
}
inline Tensor Tensor::cholesky_solve(const Tensor & input2, bool upper) const {
    static auto table = globalATenDispatch().getOpTable("aten::cholesky_solve(Tensor self, Tensor input2, bool upper=False) -> Tensor");
    return table->getOp<Tensor (const Tensor &, const Tensor &, bool)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), input2, upper);
}
inline std::tuple<Tensor,Tensor> Tensor::solve(const Tensor & A) const {
    static auto table = globalATenDispatch().getOpTable("aten::solve(Tensor self, Tensor A) -> (Tensor solution, Tensor LU)");
    return table->getOp<std::tuple<Tensor,Tensor> (const Tensor &, const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), A);
}
inline Tensor Tensor::cholesky_inverse(bool upper) const {
    static auto table = globalATenDispatch().getOpTable("aten::cholesky_inverse(Tensor self, bool upper=False) -> Tensor");
    return table->getOp<Tensor (const Tensor &, bool)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), upper);
}
inline std::tuple<Tensor,Tensor> Tensor::qr(bool some) const {
    static auto table = globalATenDispatch().getOpTable("aten::qr(Tensor self, bool some=True) -> (Tensor Q, Tensor R)");
    return table->getOp<std::tuple<Tensor,Tensor> (const Tensor &, bool)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), some);
}
inline std::tuple<Tensor,Tensor> Tensor::geqrf() const {
    static auto table = globalATenDispatch().getOpTable("aten::geqrf(Tensor self) -> (Tensor a, Tensor tau)");
    return table->getOp<std::tuple<Tensor,Tensor> (const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this));
}
inline Tensor Tensor::orgqr(const Tensor & input2) const {
    static auto table = globalATenDispatch().getOpTable("aten::orgqr(Tensor self, Tensor input2) -> Tensor");
    return table->getOp<Tensor (const Tensor &, const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), input2);
}
inline Tensor Tensor::ormqr(const Tensor & input2, const Tensor & input3, bool left, bool transpose) const {
    static auto table = globalATenDispatch().getOpTable("aten::ormqr(Tensor self, Tensor input2, Tensor input3, bool left=True, bool transpose=False) -> Tensor");
    return table->getOp<Tensor (const Tensor &, const Tensor &, const Tensor &, bool, bool)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), input2, input3, left, transpose);
}
inline Tensor Tensor::lu_solve(const Tensor & LU_data, const Tensor & LU_pivots) const {
    static auto table = globalATenDispatch().getOpTable("aten::lu_solve(Tensor self, Tensor LU_data, Tensor LU_pivots) -> Tensor");
    return table->getOp<Tensor (const Tensor &, const Tensor &, const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), LU_data, LU_pivots);
}
inline Tensor Tensor::multinomial(int64_t num_samples, bool replacement, Generator * generator) const {
    static auto table = globalATenDispatch().getOpTable("aten::multinomial(Tensor self, int num_samples, bool replacement=False, *, Generator? generator=None) -> Tensor");
    return table->getOp<Tensor (const Tensor &, int64_t, bool, Generator *)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), num_samples, replacement, generator);
}
inline Tensor Tensor::lgamma() const {
    static auto table = globalATenDispatch().getOpTable("aten::lgamma(Tensor self) -> Tensor");
    return table->getOp<Tensor (const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this));
}
inline Tensor Tensor::digamma() const {
    static auto table = globalATenDispatch().getOpTable("aten::digamma(Tensor self) -> Tensor");
    return table->getOp<Tensor (const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this));
}
inline Tensor Tensor::polygamma(int64_t n) const {
    static auto table = globalATenDispatch().getOpTable("aten::polygamma(int n, Tensor self) -> Tensor");
    return table->getOp<Tensor (int64_t, const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(n, const_cast<Tensor&>(*this));
}
inline Tensor Tensor::erfinv() const {
    static auto table = globalATenDispatch().getOpTable("aten::erfinv(Tensor self) -> Tensor");
    return table->getOp<Tensor (const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this));
}
inline Tensor Tensor::dist(const Tensor & other, Scalar p) const {
    static auto table = globalATenDispatch().getOpTable("aten::dist(Tensor self, Tensor other, Scalar p=2) -> Tensor");
    return table->getOp<Tensor (const Tensor &, const Tensor &, Scalar)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), other, p);
}
inline Tensor Tensor::atan2(const Tensor & other) const {
    static auto table = globalATenDispatch().getOpTable("aten::atan2(Tensor self, Tensor other) -> Tensor");
    return table->getOp<Tensor (const Tensor &, const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), other);
}
inline Tensor Tensor::lerp(const Tensor & end, Scalar weight) const {
    static auto table = globalATenDispatch().getOpTable("aten::lerp.Scalar(Tensor self, Tensor end, Scalar weight) -> Tensor");
    return table->getOp<Tensor (const Tensor &, const Tensor &, Scalar)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), end, weight);
}
inline Tensor Tensor::lerp(const Tensor & end, const Tensor & weight) const {
    static auto table = globalATenDispatch().getOpTable("aten::lerp.Tensor(Tensor self, Tensor end, Tensor weight) -> Tensor");
    return table->getOp<Tensor (const Tensor &, const Tensor &, const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), end, weight);
}
inline Tensor Tensor::histc(int64_t bins, Scalar min, Scalar max) const {
    static auto table = globalATenDispatch().getOpTable("aten::histc(Tensor self, int bins=100, Scalar min=0, Scalar max=0) -> Tensor");
    return table->getOp<Tensor (const Tensor &, int64_t, Scalar, Scalar)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), bins, min, max);
}
inline Tensor Tensor::sign() const {
    static auto table = globalATenDispatch().getOpTable("aten::sign(Tensor self) -> Tensor");
    return table->getOp<Tensor (const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this));
}
inline Tensor Tensor::fmod(Scalar other) const {
    static auto table = globalATenDispatch().getOpTable("aten::fmod.Scalar(Tensor self, Scalar other) -> Tensor");
    return table->getOp<Tensor (const Tensor &, Scalar)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), other);
}
inline Tensor Tensor::fmod(const Tensor & other) const {
    static auto table = globalATenDispatch().getOpTable("aten::fmod.Tensor(Tensor self, Tensor other) -> Tensor");
    return table->getOp<Tensor (const Tensor &, const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), other);
}
inline Tensor Tensor::remainder(Scalar other) const {
    static auto table = globalATenDispatch().getOpTable("aten::remainder.Scalar(Tensor self, Scalar other) -> Tensor");
    return table->getOp<Tensor (const Tensor &, Scalar)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), other);
}
inline Tensor Tensor::remainder(const Tensor & other) const {
    static auto table = globalATenDispatch().getOpTable("aten::remainder.Tensor(Tensor self, Tensor other) -> Tensor");
    return table->getOp<Tensor (const Tensor &, const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), other);
}
inline Tensor Tensor::min(const Tensor & other) const {
    static auto table = globalATenDispatch().getOpTable("aten::min.other(Tensor self, Tensor other) -> Tensor");
    return table->getOp<Tensor (const Tensor &, const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), other);
}
inline Tensor Tensor::min() const {
    static auto table = globalATenDispatch().getOpTable("aten::min(Tensor self) -> Tensor");
    return table->getOp<Tensor (const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this));
}
inline Tensor Tensor::max(const Tensor & other) const {
    static auto table = globalATenDispatch().getOpTable("aten::max.other(Tensor self, Tensor other) -> Tensor");
    return table->getOp<Tensor (const Tensor &, const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), other);
}
inline Tensor Tensor::max() const {
    static auto table = globalATenDispatch().getOpTable("aten::max(Tensor self) -> Tensor");
    return table->getOp<Tensor (const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this));
}
inline Tensor Tensor::median() const {
    static auto table = globalATenDispatch().getOpTable("aten::median(Tensor self) -> Tensor");
    return table->getOp<Tensor (const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this));
}
inline std::tuple<Tensor,Tensor> Tensor::sort(int64_t dim, bool descending) const {
    static auto table = globalATenDispatch().getOpTable("aten::sort(Tensor self, int dim=-1, bool descending=False) -> (Tensor values, Tensor indices)");
    return table->getOp<std::tuple<Tensor,Tensor> (const Tensor &, int64_t, bool)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), dim, descending);
}
inline Tensor Tensor::argsort(int64_t dim, bool descending) const {
    static auto table = globalATenDispatch().getOpTable("aten::argsort(Tensor self, int dim=-1, bool descending=False) -> Tensor");
    return table->getOp<Tensor (const Tensor &, int64_t, bool)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), dim, descending);
}
inline std::tuple<Tensor,Tensor> Tensor::topk(int64_t k, int64_t dim, bool largest, bool sorted) const {
    static auto table = globalATenDispatch().getOpTable("aten::topk(Tensor self, int k, int dim=-1, bool largest=True, bool sorted=True) -> (Tensor values, Tensor indices)");
    return table->getOp<std::tuple<Tensor,Tensor> (const Tensor &, int64_t, int64_t, bool, bool)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), k, dim, largest, sorted);
}
inline Tensor Tensor::all() const {
    static auto table = globalATenDispatch().getOpTable("aten::all(Tensor self) -> Tensor");
    return table->getOp<Tensor (const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this));
}
inline Tensor Tensor::any() const {
    static auto table = globalATenDispatch().getOpTable("aten::any(Tensor self) -> Tensor");
    return table->getOp<Tensor (const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this));
}
inline Tensor Tensor::renorm(Scalar p, int64_t dim, Scalar maxnorm) const {
    static auto table = globalATenDispatch().getOpTable("aten::renorm(Tensor self, Scalar p, int dim, Scalar maxnorm) -> Tensor");
    return table->getOp<Tensor (const Tensor &, Scalar, int64_t, Scalar)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), p, dim, maxnorm);
}
inline Tensor Tensor::unfold(int64_t dimension, int64_t size, int64_t step) const {
    static auto table = globalATenDispatch().getOpTable("aten::unfold(Tensor(a) self, int dimension, int size, int step) -> Tensor(a)");
    return table->getOp<Tensor (const Tensor &, int64_t, int64_t, int64_t)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), dimension, size, step);
}
inline bool Tensor::equal(const Tensor & other) const {
    static auto table = globalATenDispatch().getOpTable("aten::equal(Tensor self, Tensor other) -> bool");
    return table->getOp<bool (const Tensor &, const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), other);
}
inline Tensor Tensor::pow(const Tensor & exponent) const {
    static auto table = globalATenDispatch().getOpTable("aten::pow.Tensor_Tensor(Tensor self, Tensor exponent) -> Tensor");
    return table->getOp<Tensor (const Tensor &, const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this), exponent);
}
inline Tensor Tensor::alias() const {
    static auto table = globalATenDispatch().getOpTable("aten::alias(Tensor(a) self) -> Tensor(a)");
    return table->getOp<Tensor (const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(const_cast<Tensor&>(*this));
}

inline bool Tensor::is_variable() const noexcept {
  return impl_->is_variable();
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
