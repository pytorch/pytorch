#pragma once

#include <c10/core/Scalar.h>
#include <c10/core/MemoryFormat.h>
#include <c10/core/QScheme.h>
#include <c10/macros/Macros.h>
#include <c10/core/TensorOptions.h>
#include <c10/util/intrusive_ptr.h>
#include <ATen/core/DeprecatedTypeProperties.h>
#include <ATen/core/ATenDispatch.h>
#ifdef NAMEDTENSOR_ENABLED
#include <ATen/NamedTensor.h>
#endif
#ifdef USE_STATIC_DISPATCH
#include <ATen/TypeDefault.h>
#include <ATen/CPUType.h>
#include <ATen/QuantizedCPUType.h>
#include <ATen/SparseCPUType.h>
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

inline void Tensor::backward(
    c10::optional<Tensor> gradient,
    bool keep_graph,
    bool create_graph) {
  dispatch_type().backward(*this, std::move(gradient), keep_graph, create_graph);
}

inline void Tensor::set_data(Tensor new_data) {
  dispatch_type().set_data(*this, new_data);
}

// all static inline to allow for inlining of the non-dynamic part of dispatch
inline Tensor Tensor::abs() const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::abs(*this);
#else
    static auto table = globalATenDispatch().getOpTable("aten::abs(Tensor self) -> Tensor");
    return table->getOp<Tensor (const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this);
#endif
}
inline Tensor & Tensor::abs_() {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::abs_(*this);
            break;
        default:
            AT_ERROR("abs_ not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::abs_(Tensor(a!) self) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this);
#endif
}
inline Tensor Tensor::acos() const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::acos(*this);
#else
    static auto table = globalATenDispatch().getOpTable("aten::acos(Tensor self) -> Tensor");
    return table->getOp<Tensor (const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this);
#endif
}
inline Tensor & Tensor::acos_() {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::acos_(*this);
            break;
        default:
            AT_ERROR("acos_ not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::acos_(Tensor(a!) self) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this);
#endif
}
inline Tensor Tensor::add(const Tensor & other, Scalar alpha) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::add(*this, other, alpha);
            break;
        case Backend::SparseCPU:
            return SparseCPUType::add(*this, other, alpha);
            break;
        default:
            AT_ERROR("add not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::add(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor");
    return table->getOp<Tensor (const Tensor &, const Tensor &, Scalar)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, other, alpha);
#endif
}
inline Tensor & Tensor::add_(const Tensor & other, Scalar alpha) {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::add_(*this, other, alpha);
            break;
        case Backend::SparseCPU:
            return SparseCPUType::add_(*this, other, alpha);
            break;
        default:
            AT_ERROR("add_ not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::add_(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, const Tensor &, Scalar)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, other, alpha);
#endif
}
inline Tensor Tensor::add(Scalar other, Scalar alpha) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::add(*this, other, alpha);
#else
    static auto table = globalATenDispatch().getOpTable("aten::add(Tensor self, Scalar other, Scalar alpha=1) -> Tensor");
    return table->getOp<Tensor (const Tensor &, Scalar, Scalar)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, other, alpha);
#endif
}
inline Tensor & Tensor::add_(Scalar other, Scalar alpha) {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::add_(*this, other, alpha);
#else
    static auto table = globalATenDispatch().getOpTable("aten::add_(Tensor(a!) self, Scalar other, Scalar alpha=1) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, Scalar, Scalar)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, other, alpha);
#endif
}
inline Tensor Tensor::addmv(const Tensor & mat, const Tensor & vec, Scalar beta, Scalar alpha) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::addmv(*this, mat, vec, beta, alpha);
            break;
        default:
            AT_ERROR("addmv not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::addmv(Tensor self, Tensor mat, Tensor vec, *, Scalar beta=1, Scalar alpha=1) -> Tensor");
    return table->getOp<Tensor (const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, mat, vec, beta, alpha);
#endif
}
inline Tensor & Tensor::addmv_(const Tensor & mat, const Tensor & vec, Scalar beta, Scalar alpha) {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::addmv_(*this, mat, vec, beta, alpha);
            break;
        default:
            AT_ERROR("addmv_ not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::addmv_(Tensor(a!) self, Tensor mat, Tensor vec, *, Scalar beta=1, Scalar alpha=1) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, const Tensor &, const Tensor &, Scalar, Scalar)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, mat, vec, beta, alpha);
#endif
}
inline Tensor Tensor::addr(const Tensor & vec1, const Tensor & vec2, Scalar beta, Scalar alpha) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::addr(*this, vec1, vec2, beta, alpha);
#else
    static auto table = globalATenDispatch().getOpTable("aten::addr(Tensor self, Tensor vec1, Tensor vec2, *, Scalar beta=1, Scalar alpha=1) -> Tensor");
    return table->getOp<Tensor (const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, vec1, vec2, beta, alpha);
#endif
}
inline Tensor & Tensor::addr_(const Tensor & vec1, const Tensor & vec2, Scalar beta, Scalar alpha) {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::addr_(*this, vec1, vec2, beta, alpha);
#else
    static auto table = globalATenDispatch().getOpTable("aten::addr_(Tensor(a!) self, Tensor vec1, Tensor vec2, *, Scalar beta=1, Scalar alpha=1) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, const Tensor &, const Tensor &, Scalar, Scalar)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, vec1, vec2, beta, alpha);
#endif
}
inline Tensor Tensor::all(int64_t dim, bool keepdim) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::all(*this, dim, keepdim);
#else
    static auto table = globalATenDispatch().getOpTable("aten::all(Tensor self, int dim, bool keepdim=False) -> Tensor");
    return table->getOp<Tensor (const Tensor &, int64_t, bool)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, dim, keepdim);
#endif
}
inline bool Tensor::allclose(const Tensor & other, double rtol, double atol, bool equal_nan) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::allclose(*this, other, rtol, atol, equal_nan);
#else
    static auto table = globalATenDispatch().getOpTable("aten::allclose(Tensor self, Tensor other, float rtol=1e-05, float atol=1e-08, bool equal_nan=False) -> bool");
    return table->getOp<bool (const Tensor &, const Tensor &, double, double, bool)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, other, rtol, atol, equal_nan);
#endif
}
inline Tensor Tensor::any(int64_t dim, bool keepdim) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::any(*this, dim, keepdim);
#else
    static auto table = globalATenDispatch().getOpTable("aten::any(Tensor self, int dim, bool keepdim=False) -> Tensor");
    return table->getOp<Tensor (const Tensor &, int64_t, bool)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, dim, keepdim);
#endif
}
inline Tensor Tensor::argmax(c10::optional<int64_t> dim, bool keepdim) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::argmax(*this, dim, keepdim);
#else
    static auto table = globalATenDispatch().getOpTable("aten::argmax(Tensor self, int? dim=None, bool keepdim=False) -> Tensor");
    return table->getOp<Tensor (const Tensor &, c10::optional<int64_t>, bool)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, dim, keepdim);
#endif
}
inline Tensor Tensor::argmin(c10::optional<int64_t> dim, bool keepdim) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::argmin(*this, dim, keepdim);
#else
    static auto table = globalATenDispatch().getOpTable("aten::argmin(Tensor self, int? dim=None, bool keepdim=False) -> Tensor");
    return table->getOp<Tensor (const Tensor &, c10::optional<int64_t>, bool)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, dim, keepdim);
#endif
}
inline Tensor Tensor::as_strided(IntArrayRef size, IntArrayRef stride, c10::optional<int64_t> storage_offset) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::as_strided(*this, size, stride, storage_offset);
            break;
        case Backend::QuantizedCPU:
            return QuantizedCPUType::as_strided(*this, size, stride, storage_offset);
            break;
        default:
            AT_ERROR("as_strided not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::as_strided(Tensor(a) self, int[] size, int[] stride, int? storage_offset=None) -> Tensor(a)");
    return table->getOp<Tensor (const Tensor &, IntArrayRef, IntArrayRef, c10::optional<int64_t>)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, size, stride, storage_offset);
#endif
}
inline Tensor & Tensor::as_strided_(IntArrayRef size, IntArrayRef stride, c10::optional<int64_t> storage_offset) {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::as_strided_(*this, size, stride, storage_offset);
#else
    static auto table = globalATenDispatch().getOpTable("aten::as_strided_(Tensor(a!) self, int[] size, int[] stride, int? storage_offset=None) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, IntArrayRef, IntArrayRef, c10::optional<int64_t>)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, size, stride, storage_offset);
#endif
}
inline Tensor Tensor::asin() const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::asin(*this);
#else
    static auto table = globalATenDispatch().getOpTable("aten::asin(Tensor self) -> Tensor");
    return table->getOp<Tensor (const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this);
#endif
}
inline Tensor & Tensor::asin_() {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::asin_(*this);
            break;
        default:
            AT_ERROR("asin_ not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::asin_(Tensor(a!) self) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this);
#endif
}
inline Tensor Tensor::atan() const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::atan(*this);
#else
    static auto table = globalATenDispatch().getOpTable("aten::atan(Tensor self) -> Tensor");
    return table->getOp<Tensor (const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this);
#endif
}
inline Tensor & Tensor::atan_() {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::atan_(*this);
            break;
        default:
            AT_ERROR("atan_ not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::atan_(Tensor(a!) self) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this);
#endif
}
inline Tensor Tensor::baddbmm(const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::baddbmm(*this, batch1, batch2, beta, alpha);
            break;
        default:
            AT_ERROR("baddbmm not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::baddbmm(Tensor self, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1) -> Tensor");
    return table->getOp<Tensor (const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, batch1, batch2, beta, alpha);
#endif
}
inline Tensor & Tensor::baddbmm_(const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::baddbmm_(*this, batch1, batch2, beta, alpha);
            break;
        default:
            AT_ERROR("baddbmm_ not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::baddbmm_(Tensor(a!) self, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, const Tensor &, const Tensor &, Scalar, Scalar)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, batch1, batch2, beta, alpha);
#endif
}
inline Tensor Tensor::bernoulli(Generator * generator) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::bernoulli(*this, generator);
#else
    static auto table = globalATenDispatch().getOpTable("aten::bernoulli(Tensor self, *, Generator? generator=None) -> Tensor");
    return table->getOp<Tensor (const Tensor &, Generator *)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, generator);
#endif
}
inline Tensor & Tensor::bernoulli_(const Tensor & p, Generator * generator) {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::bernoulli_(*this, p, generator);
            break;
        default:
            AT_ERROR("bernoulli_ not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::bernoulli_(Tensor(a!) self, Tensor p, *, Generator? generator=None) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, const Tensor &, Generator *)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, p, generator);
#endif
}
inline Tensor & Tensor::bernoulli_(double p, Generator * generator) {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::bernoulli_(*this, p, generator);
            break;
        default:
            AT_ERROR("bernoulli_ not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::bernoulli_(Tensor(a!) self, float p=0.5, *, Generator? generator=None) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, double, Generator *)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, p, generator);
#endif
}
inline Tensor Tensor::bernoulli(double p, Generator * generator) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::bernoulli(*this, p, generator);
#else
    static auto table = globalATenDispatch().getOpTable("aten::bernoulli(Tensor self, float p, *, Generator? generator=None) -> Tensor");
    return table->getOp<Tensor (const Tensor &, double, Generator *)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, p, generator);
#endif
}
inline Tensor Tensor::bincount(const Tensor & weights, int64_t minlength) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::bincount(*this, weights, minlength);
            break;
        default:
            AT_ERROR("bincount not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::bincount(Tensor self, Tensor? weights=None, int minlength=0) -> Tensor");
    return table->getOp<Tensor (const Tensor &, const Tensor &, int64_t)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, weights, minlength);
#endif
}
inline Tensor Tensor::bmm(const Tensor & mat2) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::bmm(*this, mat2);
            break;
        default:
            AT_ERROR("bmm not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::bmm(Tensor self, Tensor mat2) -> Tensor");
    return table->getOp<Tensor (const Tensor &, const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, mat2);
#endif
}
inline Tensor Tensor::ceil() const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::ceil(*this);
#else
    static auto table = globalATenDispatch().getOpTable("aten::ceil(Tensor self) -> Tensor");
    return table->getOp<Tensor (const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this);
#endif
}
inline Tensor & Tensor::ceil_() {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::ceil_(*this);
            break;
        default:
            AT_ERROR("ceil_ not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::ceil_(Tensor(a!) self) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this);
#endif
}
inline std::vector<Tensor> Tensor::chunk(int64_t chunks, int64_t dim) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::chunk(*this, chunks, dim);
#else
    static auto table = globalATenDispatch().getOpTable("aten::chunk(Tensor(a) self, int chunks, int dim=0) -> Tensor(a)[]");
    return table->getOp<std::vector<Tensor> (const Tensor &, int64_t, int64_t)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, chunks, dim);
#endif
}
inline Tensor Tensor::clamp(c10::optional<Scalar> min, c10::optional<Scalar> max) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::clamp(*this, min, max);
#else
    static auto table = globalATenDispatch().getOpTable("aten::clamp(Tensor self, Scalar? min=None, Scalar? max=None) -> Tensor");
    return table->getOp<Tensor (const Tensor &, c10::optional<Scalar>, c10::optional<Scalar>)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, min, max);
#endif
}
inline Tensor & Tensor::clamp_(c10::optional<Scalar> min, c10::optional<Scalar> max) {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::clamp_(*this, min, max);
            break;
        default:
            AT_ERROR("clamp_ not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::clamp_(Tensor(a!) self, Scalar? min=None, Scalar? max=None) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, c10::optional<Scalar>, c10::optional<Scalar>)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, min, max);
#endif
}
inline Tensor Tensor::clamp_max(Scalar max) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::clamp_max(*this, max);
#else
    static auto table = globalATenDispatch().getOpTable("aten::clamp_max(Tensor self, Scalar max) -> Tensor");
    return table->getOp<Tensor (const Tensor &, Scalar)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, max);
#endif
}
inline Tensor & Tensor::clamp_max_(Scalar max) {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::clamp_max_(*this, max);
            break;
        default:
            AT_ERROR("clamp_max_ not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::clamp_max_(Tensor(a!) self, Scalar max) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, Scalar)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, max);
#endif
}
inline Tensor Tensor::clamp_min(Scalar min) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::clamp_min(*this, min);
#else
    static auto table = globalATenDispatch().getOpTable("aten::clamp_min(Tensor self, Scalar min) -> Tensor");
    return table->getOp<Tensor (const Tensor &, Scalar)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, min);
#endif
}
inline Tensor & Tensor::clamp_min_(Scalar min) {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::clamp_min_(*this, min);
            break;
        default:
            AT_ERROR("clamp_min_ not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::clamp_min_(Tensor(a!) self, Scalar min) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, Scalar)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, min);
#endif
}
inline Tensor Tensor::contiguous(MemoryFormat memory_format) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::contiguous(*this, memory_format);
#else
    static auto table = globalATenDispatch().getOpTable("aten::contiguous(Tensor self, *, MemoryFormat memory_format=contiguous_format) -> Tensor");
    return table->getOp<Tensor (const Tensor &, MemoryFormat)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, memory_format);
#endif
}
inline Tensor & Tensor::copy_(const Tensor & src, bool non_blocking) {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::copy_(*this, src, non_blocking);
#else
    static auto table = globalATenDispatch().getOpTable("aten::copy_(Tensor(a!) self, Tensor src, bool non_blocking=False) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, const Tensor &, bool)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, src, non_blocking);
#endif
}
inline Tensor Tensor::cos() const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::cos(*this);
#else
    static auto table = globalATenDispatch().getOpTable("aten::cos(Tensor self) -> Tensor");
    return table->getOp<Tensor (const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this);
#endif
}
inline Tensor & Tensor::cos_() {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::cos_(*this);
            break;
        default:
            AT_ERROR("cos_ not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::cos_(Tensor(a!) self) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this);
#endif
}
inline Tensor Tensor::cosh() const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::cosh(*this);
#else
    static auto table = globalATenDispatch().getOpTable("aten::cosh(Tensor self) -> Tensor");
    return table->getOp<Tensor (const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this);
#endif
}
inline Tensor & Tensor::cosh_() {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::cosh_(*this);
            break;
        default:
            AT_ERROR("cosh_ not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::cosh_(Tensor(a!) self) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this);
#endif
}
inline Tensor Tensor::cumsum(int64_t dim, c10::optional<ScalarType> dtype) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::cumsum(*this, dim, dtype);
#else
    static auto table = globalATenDispatch().getOpTable("aten::cumsum(Tensor self, int dim, *, ScalarType? dtype=None) -> Tensor");
    return table->getOp<Tensor (const Tensor &, int64_t, c10::optional<ScalarType>)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, dim, dtype);
#endif
}
inline Tensor Tensor::cumprod(int64_t dim, c10::optional<ScalarType> dtype) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::cumprod(*this, dim, dtype);
#else
    static auto table = globalATenDispatch().getOpTable("aten::cumprod(Tensor self, int dim, *, ScalarType? dtype=None) -> Tensor");
    return table->getOp<Tensor (const Tensor &, int64_t, c10::optional<ScalarType>)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, dim, dtype);
#endif
}
inline Tensor Tensor::det() const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::det(*this);
#else
    static auto table = globalATenDispatch().getOpTable("aten::det(Tensor self) -> Tensor");
    return table->getOp<Tensor (const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this);
#endif
}
inline Tensor Tensor::diag_embed(int64_t offset, int64_t dim1, int64_t dim2) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::diag_embed(*this, offset, dim1, dim2);
#else
    static auto table = globalATenDispatch().getOpTable("aten::diag_embed(Tensor self, int offset=0, int dim1=-2, int dim2=-1) -> Tensor");
    return table->getOp<Tensor (const Tensor &, int64_t, int64_t, int64_t)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, offset, dim1, dim2);
#endif
}
inline Tensor Tensor::diagflat(int64_t offset) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::diagflat(*this, offset);
#else
    static auto table = globalATenDispatch().getOpTable("aten::diagflat(Tensor self, int offset=0) -> Tensor");
    return table->getOp<Tensor (const Tensor &, int64_t)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, offset);
#endif
}
inline Tensor Tensor::diagonal(int64_t offset, int64_t dim1, int64_t dim2) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::diagonal(*this, offset, dim1, dim2);
#else
    static auto table = globalATenDispatch().getOpTable("aten::diagonal(Tensor(a) self, int offset=0, int dim1=0, int dim2=1) -> Tensor(a)");
    return table->getOp<Tensor (const Tensor &, int64_t, int64_t, int64_t)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, offset, dim1, dim2);
#endif
}
inline Tensor Tensor::div(const Tensor & other) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::div(*this, other);
#else
    static auto table = globalATenDispatch().getOpTable("aten::div(Tensor self, Tensor other) -> Tensor");
    return table->getOp<Tensor (const Tensor &, const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, other);
#endif
}
inline Tensor & Tensor::div_(const Tensor & other) {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::div_(*this, other);
#else
    static auto table = globalATenDispatch().getOpTable("aten::div_(Tensor(a!) self, Tensor other) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, other);
#endif
}
inline Tensor Tensor::div(Scalar other) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::div(*this, other);
#else
    static auto table = globalATenDispatch().getOpTable("aten::div(Tensor self, Scalar other) -> Tensor");
    return table->getOp<Tensor (const Tensor &, Scalar)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, other);
#endif
}
inline Tensor & Tensor::div_(Scalar other) {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::div_(*this, other);
#else
    static auto table = globalATenDispatch().getOpTable("aten::div_(Tensor(a!) self, Scalar other) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, Scalar)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, other);
#endif
}
inline Tensor Tensor::dot(const Tensor & tensor) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::dot(*this, tensor);
            break;
        default:
            AT_ERROR("dot not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::dot(Tensor self, Tensor tensor) -> Tensor");
    return table->getOp<Tensor (const Tensor &, const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, tensor);
#endif
}
inline Tensor & Tensor::resize_(IntArrayRef size) {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::resize_(*this, size);
            break;
        default:
            AT_ERROR("resize_ not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::resize_(Tensor(a!) self, int[] size) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, IntArrayRef)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, size);
#endif
}
inline Tensor Tensor::erf() const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::erf(*this);
#else
    static auto table = globalATenDispatch().getOpTable("aten::erf(Tensor self) -> Tensor");
    return table->getOp<Tensor (const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this);
#endif
}
inline Tensor & Tensor::erf_() {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::erf_(*this);
            break;
        default:
            AT_ERROR("erf_ not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::erf_(Tensor(a!) self) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this);
#endif
}
inline Tensor Tensor::erfc() const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::erfc(*this);
#else
    static auto table = globalATenDispatch().getOpTable("aten::erfc(Tensor self) -> Tensor");
    return table->getOp<Tensor (const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this);
#endif
}
inline Tensor & Tensor::erfc_() {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::erfc_(*this);
            break;
        default:
            AT_ERROR("erfc_ not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::erfc_(Tensor(a!) self) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this);
#endif
}
inline Tensor Tensor::exp() const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::exp(*this);
#else
    static auto table = globalATenDispatch().getOpTable("aten::exp(Tensor self) -> Tensor");
    return table->getOp<Tensor (const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this);
#endif
}
inline Tensor & Tensor::exp_() {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::exp_(*this);
            break;
        default:
            AT_ERROR("exp_ not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::exp_(Tensor(a!) self) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this);
#endif
}
inline Tensor Tensor::expm1() const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::expm1(*this);
#else
    static auto table = globalATenDispatch().getOpTable("aten::expm1(Tensor self) -> Tensor");
    return table->getOp<Tensor (const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this);
#endif
}
inline Tensor & Tensor::expm1_() {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::expm1_(*this);
            break;
        default:
            AT_ERROR("expm1_ not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::expm1_(Tensor(a!) self) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this);
#endif
}
inline Tensor Tensor::expand(IntArrayRef size, bool implicit) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::expand(*this, size, implicit);
#else
    static auto table = globalATenDispatch().getOpTable("aten::expand(Tensor(a) self, int[] size, *, bool implicit=False) -> Tensor(a)");
    return table->getOp<Tensor (const Tensor &, IntArrayRef, bool)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, size, implicit);
#endif
}
inline Tensor Tensor::expand_as(const Tensor & other) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::expand_as(*this, other);
#else
    static auto table = globalATenDispatch().getOpTable("aten::expand_as(Tensor self, Tensor other) -> Tensor");
    return table->getOp<Tensor (const Tensor &, const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, other);
#endif
}
inline Tensor Tensor::flatten(int64_t start_dim, int64_t end_dim) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::flatten(*this, start_dim, end_dim);
#else
    static auto table = globalATenDispatch().getOpTable("aten::flatten(Tensor self, int start_dim=0, int end_dim=-1) -> Tensor");
    return table->getOp<Tensor (const Tensor &, int64_t, int64_t)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, start_dim, end_dim);
#endif
}
inline Tensor & Tensor::fill_(Scalar value) {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::fill_(*this, value);
#else
    static auto table = globalATenDispatch().getOpTable("aten::fill_(Tensor(a!) self, Scalar value) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, Scalar)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, value);
#endif
}
inline Tensor & Tensor::fill_(const Tensor & value) {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::fill_(*this, value);
#else
    static auto table = globalATenDispatch().getOpTable("aten::fill_(Tensor(a!) self, Tensor value) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, value);
#endif
}
inline Tensor Tensor::floor() const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::floor(*this);
#else
    static auto table = globalATenDispatch().getOpTable("aten::floor(Tensor self) -> Tensor");
    return table->getOp<Tensor (const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this);
#endif
}
inline Tensor & Tensor::floor_() {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::floor_(*this);
            break;
        default:
            AT_ERROR("floor_ not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::floor_(Tensor(a!) self) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this);
#endif
}
inline Tensor Tensor::frac() const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::frac(*this);
#else
    static auto table = globalATenDispatch().getOpTable("aten::frac(Tensor self) -> Tensor");
    return table->getOp<Tensor (const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this);
#endif
}
inline Tensor & Tensor::frac_() {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::frac_(*this);
            break;
        default:
            AT_ERROR("frac_ not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::frac_(Tensor(a!) self) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this);
#endif
}
inline Tensor Tensor::ger(const Tensor & vec2) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::ger(*this, vec2);
            break;
        default:
            AT_ERROR("ger not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::ger(Tensor self, Tensor vec2) -> Tensor");
    return table->getOp<Tensor (const Tensor &, const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, vec2);
#endif
}
inline Tensor Tensor::fft(int64_t signal_ndim, bool normalized) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::fft(*this, signal_ndim, normalized);
#else
    static auto table = globalATenDispatch().getOpTable("aten::fft(Tensor self, int signal_ndim, bool normalized=False) -> Tensor");
    return table->getOp<Tensor (const Tensor &, int64_t, bool)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, signal_ndim, normalized);
#endif
}
inline Tensor Tensor::ifft(int64_t signal_ndim, bool normalized) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::ifft(*this, signal_ndim, normalized);
#else
    static auto table = globalATenDispatch().getOpTable("aten::ifft(Tensor self, int signal_ndim, bool normalized=False) -> Tensor");
    return table->getOp<Tensor (const Tensor &, int64_t, bool)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, signal_ndim, normalized);
#endif
}
inline Tensor Tensor::rfft(int64_t signal_ndim, bool normalized, bool onesided) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::rfft(*this, signal_ndim, normalized, onesided);
#else
    static auto table = globalATenDispatch().getOpTable("aten::rfft(Tensor self, int signal_ndim, bool normalized=False, bool onesided=True) -> Tensor");
    return table->getOp<Tensor (const Tensor &, int64_t, bool, bool)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, signal_ndim, normalized, onesided);
#endif
}
inline Tensor Tensor::irfft(int64_t signal_ndim, bool normalized, bool onesided, IntArrayRef signal_sizes) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::irfft(*this, signal_ndim, normalized, onesided, signal_sizes);
#else
    static auto table = globalATenDispatch().getOpTable("aten::irfft(Tensor self, int signal_ndim, bool normalized=False, bool onesided=True, int[] signal_sizes=[]) -> Tensor");
    return table->getOp<Tensor (const Tensor &, int64_t, bool, bool, IntArrayRef)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, signal_ndim, normalized, onesided, signal_sizes);
#endif
}
inline Tensor Tensor::index(TensorList indices) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::index(*this, indices);
#else
    static auto table = globalATenDispatch().getOpTable("aten::index(Tensor self, Tensor?[] indices) -> Tensor");
    return table->getOp<Tensor (const Tensor &, TensorList)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, indices);
#endif
}
inline Tensor & Tensor::index_copy_(int64_t dim, const Tensor & index, const Tensor & source) {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::index_copy_(*this, dim, index, source);
#else
    static auto table = globalATenDispatch().getOpTable("aten::index_copy_(Tensor(a!) self, int dim, Tensor index, Tensor source) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, int64_t, const Tensor &, const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, dim, index, source);
#endif
}
inline Tensor Tensor::index_copy(int64_t dim, const Tensor & index, const Tensor & source) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::index_copy(*this, dim, index, source);
#else
    static auto table = globalATenDispatch().getOpTable("aten::index_copy(Tensor self, int dim, Tensor index, Tensor source) -> Tensor");
    return table->getOp<Tensor (const Tensor &, int64_t, const Tensor &, const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, dim, index, source);
#endif
}
inline Tensor & Tensor::index_put_(TensorList indices, const Tensor & values, bool accumulate) {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::index_put_(*this, indices, values, accumulate);
#else
    static auto table = globalATenDispatch().getOpTable("aten::index_put_(Tensor(a!) self, Tensor?[] indices, Tensor values, bool accumulate=False) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, TensorList, const Tensor &, bool)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, indices, values, accumulate);
#endif
}
inline Tensor Tensor::index_put(TensorList indices, const Tensor & values, bool accumulate) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::index_put(*this, indices, values, accumulate);
#else
    static auto table = globalATenDispatch().getOpTable("aten::index_put(Tensor self, Tensor?[] indices, Tensor values, bool accumulate=False) -> Tensor");
    return table->getOp<Tensor (const Tensor &, TensorList, const Tensor &, bool)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, indices, values, accumulate);
#endif
}
inline Tensor Tensor::inverse() const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::inverse(*this);
#else
    static auto table = globalATenDispatch().getOpTable("aten::inverse(Tensor self) -> Tensor");
    return table->getOp<Tensor (const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this);
#endif
}
inline Tensor Tensor::isclose(const Tensor & other, double rtol, double atol, bool equal_nan) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::isclose(*this, other, rtol, atol, equal_nan);
#else
    static auto table = globalATenDispatch().getOpTable("aten::isclose(Tensor self, Tensor other, float rtol=1e-05, float atol=1e-08, bool equal_nan=False) -> Tensor");
    return table->getOp<Tensor (const Tensor &, const Tensor &, double, double, bool)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, other, rtol, atol, equal_nan);
#endif
}
inline bool Tensor::is_distributed() const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::is_distributed(*this);
#else
    static auto table = globalATenDispatch().getOpTable("aten::is_distributed(Tensor self) -> bool");
    return table->getOp<bool (const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this);
#endif
}
inline bool Tensor::is_floating_point() const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::is_floating_point(*this);
#else
    static auto table = globalATenDispatch().getOpTable("aten::is_floating_point(Tensor self) -> bool");
    return table->getOp<bool (const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this);
#endif
}
inline bool Tensor::is_complex() const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::is_complex(*this);
#else
    static auto table = globalATenDispatch().getOpTable("aten::is_complex(Tensor self) -> bool");
    return table->getOp<bool (const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this);
#endif
}
inline bool Tensor::is_nonzero() const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::is_nonzero(*this);
#else
    static auto table = globalATenDispatch().getOpTable("aten::is_nonzero(Tensor self) -> bool");
    return table->getOp<bool (const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this);
#endif
}
inline bool Tensor::is_same_size(const Tensor & other) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::is_same_size(*this, other);
#else
    static auto table = globalATenDispatch().getOpTable("aten::is_same_size(Tensor self, Tensor other) -> bool");
    return table->getOp<bool (const Tensor &, const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, other);
#endif
}
inline bool Tensor::is_signed() const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::is_signed(*this);
#else
    static auto table = globalATenDispatch().getOpTable("aten::is_signed(Tensor self) -> bool");
    return table->getOp<bool (const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this);
#endif
}
inline std::tuple<Tensor,Tensor> Tensor::kthvalue(int64_t k, int64_t dim, bool keepdim) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::kthvalue(*this, k, dim, keepdim);
#else
    static auto table = globalATenDispatch().getOpTable("aten::kthvalue(Tensor self, int k, int dim=-1, bool keepdim=False) -> (Tensor values, Tensor indices)");
    return table->getOp<std::tuple<Tensor,Tensor> (const Tensor &, int64_t, int64_t, bool)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, k, dim, keepdim);
#endif
}
inline Tensor Tensor::log() const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::log(*this);
#else
    static auto table = globalATenDispatch().getOpTable("aten::log(Tensor self) -> Tensor");
    return table->getOp<Tensor (const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this);
#endif
}
inline Tensor & Tensor::log_() {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::log_(*this);
            break;
        default:
            AT_ERROR("log_ not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::log_(Tensor(a!) self) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this);
#endif
}
inline Tensor Tensor::log10() const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::log10(*this);
#else
    static auto table = globalATenDispatch().getOpTable("aten::log10(Tensor self) -> Tensor");
    return table->getOp<Tensor (const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this);
#endif
}
inline Tensor & Tensor::log10_() {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::log10_(*this);
            break;
        default:
            AT_ERROR("log10_ not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::log10_(Tensor(a!) self) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this);
#endif
}
inline Tensor Tensor::log1p() const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::log1p(*this);
#else
    static auto table = globalATenDispatch().getOpTable("aten::log1p(Tensor self) -> Tensor");
    return table->getOp<Tensor (const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this);
#endif
}
inline Tensor & Tensor::log1p_() {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::log1p_(*this);
            break;
        case Backend::SparseCPU:
            return SparseCPUType::log1p_(*this);
            break;
        default:
            AT_ERROR("log1p_ not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::log1p_(Tensor(a!) self) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this);
#endif
}
inline Tensor Tensor::log2() const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::log2(*this);
#else
    static auto table = globalATenDispatch().getOpTable("aten::log2(Tensor self) -> Tensor");
    return table->getOp<Tensor (const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this);
#endif
}
inline Tensor & Tensor::log2_() {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::log2_(*this);
            break;
        default:
            AT_ERROR("log2_ not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::log2_(Tensor(a!) self) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this);
#endif
}
inline Tensor Tensor::logdet() const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::logdet(*this);
#else
    static auto table = globalATenDispatch().getOpTable("aten::logdet(Tensor self) -> Tensor");
    return table->getOp<Tensor (const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this);
#endif
}
inline Tensor Tensor::log_softmax(int64_t dim, c10::optional<ScalarType> dtype) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::log_softmax(*this, dim, dtype);
#else
    static auto table = globalATenDispatch().getOpTable("aten::log_softmax(Tensor self, int dim, *, ScalarType? dtype=None) -> Tensor");
    return table->getOp<Tensor (const Tensor &, int64_t, c10::optional<ScalarType>)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, dim, dtype);
#endif
}
inline Tensor Tensor::logsumexp(IntArrayRef dim, bool keepdim) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::logsumexp(*this, dim, keepdim);
#else
    static auto table = globalATenDispatch().getOpTable("aten::logsumexp(Tensor self, int[1] dim, bool keepdim=False) -> Tensor");
    return table->getOp<Tensor (const Tensor &, IntArrayRef, bool)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, dim, keepdim);
#endif
}
inline Tensor Tensor::matmul(const Tensor & other) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::matmul(*this, other);
#else
    static auto table = globalATenDispatch().getOpTable("aten::matmul(Tensor self, Tensor other) -> Tensor");
    return table->getOp<Tensor (const Tensor &, const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, other);
#endif
}
inline Tensor Tensor::matrix_power(int64_t n) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::matrix_power(*this, n);
#else
    static auto table = globalATenDispatch().getOpTable("aten::matrix_power(Tensor self, int n) -> Tensor");
    return table->getOp<Tensor (const Tensor &, int64_t)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, n);
#endif
}
inline std::tuple<Tensor,Tensor> Tensor::max(int64_t dim, bool keepdim) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::max(*this, dim, keepdim);
#else
    static auto table = globalATenDispatch().getOpTable("aten::max(Tensor self, int dim, bool keepdim=False) -> (Tensor values, Tensor indices)");
    return table->getOp<std::tuple<Tensor,Tensor> (const Tensor &, int64_t, bool)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, dim, keepdim);
#endif
}
inline Tensor Tensor::max_values(IntArrayRef dim, bool keepdim) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::max_values(*this, dim, keepdim);
#else
    static auto table = globalATenDispatch().getOpTable("aten::max_values(Tensor self, int[1] dim, bool keepdim=False) -> Tensor");
    return table->getOp<Tensor (const Tensor &, IntArrayRef, bool)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, dim, keepdim);
#endif
}
inline Tensor Tensor::mean(c10::optional<ScalarType> dtype) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::mean(*this, dtype);
#else
    static auto table = globalATenDispatch().getOpTable("aten::mean(Tensor self, *, ScalarType? dtype=None) -> Tensor");
    return table->getOp<Tensor (const Tensor &, c10::optional<ScalarType>)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, dtype);
#endif
}
inline Tensor Tensor::mean(IntArrayRef dim, bool keepdim, c10::optional<ScalarType> dtype) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::mean(*this, dim, keepdim, dtype);
#else
    static auto table = globalATenDispatch().getOpTable("aten::mean(Tensor self, int[1] dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor");
    return table->getOp<Tensor (const Tensor &, IntArrayRef, bool, c10::optional<ScalarType>)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, dim, keepdim, dtype);
#endif
}
inline std::tuple<Tensor,Tensor> Tensor::median(int64_t dim, bool keepdim) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::median(*this, dim, keepdim);
#else
    static auto table = globalATenDispatch().getOpTable("aten::median(Tensor self, int dim, bool keepdim=False) -> (Tensor values, Tensor indices)");
    return table->getOp<std::tuple<Tensor,Tensor> (const Tensor &, int64_t, bool)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, dim, keepdim);
#endif
}
inline std::tuple<Tensor,Tensor> Tensor::min(int64_t dim, bool keepdim) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::min(*this, dim, keepdim);
#else
    static auto table = globalATenDispatch().getOpTable("aten::min(Tensor self, int dim, bool keepdim=False) -> (Tensor values, Tensor indices)");
    return table->getOp<std::tuple<Tensor,Tensor> (const Tensor &, int64_t, bool)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, dim, keepdim);
#endif
}
inline Tensor Tensor::min_values(IntArrayRef dim, bool keepdim) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::min_values(*this, dim, keepdim);
#else
    static auto table = globalATenDispatch().getOpTable("aten::min_values(Tensor self, int[1] dim, bool keepdim=False) -> Tensor");
    return table->getOp<Tensor (const Tensor &, IntArrayRef, bool)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, dim, keepdim);
#endif
}
inline Tensor Tensor::mm(const Tensor & mat2) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::mm(*this, mat2);
            break;
        case Backend::SparseCPU:
            return SparseCPUType::mm(*this, mat2);
            break;
        default:
            AT_ERROR("mm not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::mm(Tensor self, Tensor mat2) -> Tensor");
    return table->getOp<Tensor (const Tensor &, const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, mat2);
#endif
}
inline std::tuple<Tensor,Tensor> Tensor::mode(int64_t dim, bool keepdim) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::mode(*this, dim, keepdim);
#else
    static auto table = globalATenDispatch().getOpTable("aten::mode(Tensor self, int dim=-1, bool keepdim=False) -> (Tensor values, Tensor indices)");
    return table->getOp<std::tuple<Tensor,Tensor> (const Tensor &, int64_t, bool)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, dim, keepdim);
#endif
}
inline Tensor Tensor::mul(const Tensor & other) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::mul(*this, other);
            break;
        case Backend::SparseCPU:
            return SparseCPUType::mul(*this, other);
            break;
        default:
            AT_ERROR("mul not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::mul(Tensor self, Tensor other) -> Tensor");
    return table->getOp<Tensor (const Tensor &, const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, other);
#endif
}
inline Tensor & Tensor::mul_(const Tensor & other) {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::mul_(*this, other);
            break;
        case Backend::SparseCPU:
            return SparseCPUType::mul_(*this, other);
            break;
        default:
            AT_ERROR("mul_ not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::mul_(Tensor(a!) self, Tensor other) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, other);
#endif
}
inline Tensor Tensor::mul(Scalar other) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::mul(*this, other);
#else
    static auto table = globalATenDispatch().getOpTable("aten::mul(Tensor self, Scalar other) -> Tensor");
    return table->getOp<Tensor (const Tensor &, Scalar)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, other);
#endif
}
inline Tensor & Tensor::mul_(Scalar other) {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::mul_(*this, other);
#else
    static auto table = globalATenDispatch().getOpTable("aten::mul_(Tensor(a!) self, Scalar other) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, Scalar)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, other);
#endif
}
inline Tensor Tensor::mv(const Tensor & vec) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::mv(*this, vec);
            break;
        default:
            AT_ERROR("mv not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::mv(Tensor self, Tensor vec) -> Tensor");
    return table->getOp<Tensor (const Tensor &, const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, vec);
#endif
}
inline Tensor Tensor::mvlgamma(int64_t p) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::mvlgamma(*this, p);
#else
    static auto table = globalATenDispatch().getOpTable("aten::mvlgamma(Tensor self, int p) -> Tensor");
    return table->getOp<Tensor (const Tensor &, int64_t)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, p);
#endif
}
inline Tensor & Tensor::mvlgamma_(int64_t p) {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::mvlgamma_(*this, p);
#else
    static auto table = globalATenDispatch().getOpTable("aten::mvlgamma_(Tensor(a!) self, int p) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, int64_t)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, p);
#endif
}
inline Tensor Tensor::narrow_copy(int64_t dim, int64_t start, int64_t length) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::narrow_copy(*this, dim, start, length);
            break;
        case Backend::SparseCPU:
            return SparseCPUType::narrow_copy(*this, dim, start, length);
            break;
        default:
            AT_ERROR("narrow_copy not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::narrow_copy(Tensor self, int dim, int start, int length) -> Tensor");
    return table->getOp<Tensor (const Tensor &, int64_t, int64_t, int64_t)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, dim, start, length);
#endif
}
inline Tensor Tensor::narrow(int64_t dim, int64_t start, int64_t length) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::narrow(*this, dim, start, length);
#else
    static auto table = globalATenDispatch().getOpTable("aten::narrow(Tensor(a) self, int dim, int start, int length) -> Tensor(a)");
    return table->getOp<Tensor (const Tensor &, int64_t, int64_t, int64_t)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, dim, start, length);
#endif
}
inline Tensor Tensor::permute(IntArrayRef dims) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::permute(*this, dims);
#else
    static auto table = globalATenDispatch().getOpTable("aten::permute(Tensor(a) self, int[] dims) -> Tensor(a)");
    return table->getOp<Tensor (const Tensor &, IntArrayRef)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, dims);
#endif
}
inline Tensor Tensor::numpy_T() const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::numpy_T(*this);
#else
    static auto table = globalATenDispatch().getOpTable("aten::numpy_T(Tensor(a) self) -> Tensor(a)");
    return table->getOp<Tensor (const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this);
#endif
}
inline Tensor Tensor::pin_memory() const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::pin_memory(*this);
#else
    static auto table = globalATenDispatch().getOpTable("aten::pin_memory(Tensor self) -> Tensor");
    return table->getOp<Tensor (const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this);
#endif
}
inline Tensor Tensor::pinverse(double rcond) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::pinverse(*this, rcond);
#else
    static auto table = globalATenDispatch().getOpTable("aten::pinverse(Tensor self, float rcond=1e-15) -> Tensor");
    return table->getOp<Tensor (const Tensor &, double)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, rcond);
#endif
}
inline Tensor Tensor::reciprocal() const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::reciprocal(*this);
#else
    static auto table = globalATenDispatch().getOpTable("aten::reciprocal(Tensor self) -> Tensor");
    return table->getOp<Tensor (const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this);
#endif
}
inline Tensor & Tensor::reciprocal_() {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::reciprocal_(*this);
            break;
        default:
            AT_ERROR("reciprocal_ not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::reciprocal_(Tensor(a!) self) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this);
#endif
}
inline Tensor Tensor::neg() const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::neg(*this);
#else
    static auto table = globalATenDispatch().getOpTable("aten::neg(Tensor self) -> Tensor");
    return table->getOp<Tensor (const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this);
#endif
}
inline Tensor & Tensor::neg_() {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::neg_(*this);
            break;
        default:
            AT_ERROR("neg_ not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::neg_(Tensor(a!) self) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this);
#endif
}
inline Tensor Tensor::repeat(IntArrayRef repeats) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::repeat(*this, repeats);
#else
    static auto table = globalATenDispatch().getOpTable("aten::repeat(Tensor self, int[] repeats) -> Tensor");
    return table->getOp<Tensor (const Tensor &, IntArrayRef)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, repeats);
#endif
}
inline Tensor Tensor::repeat_interleave(const Tensor & repeats, c10::optional<int64_t> dim) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::repeat_interleave(*this, repeats, dim);
#else
    static auto table = globalATenDispatch().getOpTable("aten::repeat_interleave(Tensor self, Tensor repeats, int? dim=None) -> Tensor");
    return table->getOp<Tensor (const Tensor &, const Tensor &, c10::optional<int64_t>)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, repeats, dim);
#endif
}
inline Tensor Tensor::repeat_interleave(int64_t repeats, c10::optional<int64_t> dim) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::repeat_interleave(*this, repeats, dim);
#else
    static auto table = globalATenDispatch().getOpTable("aten::repeat_interleave(Tensor self, int repeats, int? dim=None) -> Tensor");
    return table->getOp<Tensor (const Tensor &, int64_t, c10::optional<int64_t>)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, repeats, dim);
#endif
}
inline Tensor Tensor::reshape(IntArrayRef shape) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::reshape(*this, shape);
#else
    static auto table = globalATenDispatch().getOpTable("aten::reshape(Tensor self, int[] shape) -> Tensor");
    return table->getOp<Tensor (const Tensor &, IntArrayRef)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, shape);
#endif
}
inline Tensor Tensor::reshape_as(const Tensor & other) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::reshape_as(*this, other);
#else
    static auto table = globalATenDispatch().getOpTable("aten::reshape_as(Tensor self, Tensor other) -> Tensor");
    return table->getOp<Tensor (const Tensor &, const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, other);
#endif
}
inline Tensor Tensor::round() const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::round(*this);
#else
    static auto table = globalATenDispatch().getOpTable("aten::round(Tensor self) -> Tensor");
    return table->getOp<Tensor (const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this);
#endif
}
inline Tensor & Tensor::round_() {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::round_(*this);
            break;
        default:
            AT_ERROR("round_ not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::round_(Tensor(a!) self) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this);
#endif
}
inline Tensor Tensor::relu() const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::relu(*this);
            break;
        default:
            AT_ERROR("relu not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::relu(Tensor self) -> Tensor");
    return table->getOp<Tensor (const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this);
#endif
}
inline Tensor & Tensor::relu_() {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::relu_(*this);
            break;
        default:
            AT_ERROR("relu_ not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::relu_(Tensor(a!) self) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this);
#endif
}
inline Tensor Tensor::prelu(const Tensor & weight) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::prelu(*this, weight);
            break;
        default:
            AT_ERROR("prelu not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::prelu(Tensor self, Tensor weight) -> Tensor");
    return table->getOp<Tensor (const Tensor &, const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, weight);
#endif
}
inline std::tuple<Tensor,Tensor> Tensor::prelu_backward(const Tensor & grad_output, const Tensor & weight) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::prelu_backward(grad_output, *this, weight);
            break;
        default:
            AT_ERROR("prelu_backward not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::prelu_backward(Tensor grad_output, Tensor self, Tensor weight) -> (Tensor, Tensor)");
    return table->getOp<std::tuple<Tensor,Tensor> (const Tensor &, const Tensor &, const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(grad_output, *this, weight);
#endif
}
inline Tensor Tensor::hardshrink(Scalar lambd) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::hardshrink(*this, lambd);
            break;
        default:
            AT_ERROR("hardshrink not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::hardshrink(Tensor self, Scalar lambd=0.5) -> Tensor");
    return table->getOp<Tensor (const Tensor &, Scalar)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, lambd);
#endif
}
inline Tensor Tensor::hardshrink_backward(const Tensor & grad_out, Scalar lambd) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::hardshrink_backward(grad_out, *this, lambd);
            break;
        default:
            AT_ERROR("hardshrink_backward not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::hardshrink_backward(Tensor grad_out, Tensor self, Scalar lambd) -> Tensor");
    return table->getOp<Tensor (const Tensor &, const Tensor &, Scalar)>(tensorTypeIdToBackend(type_id()), is_variable())(grad_out, *this, lambd);
#endif
}
inline Tensor Tensor::rsqrt() const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::rsqrt(*this);
#else
    static auto table = globalATenDispatch().getOpTable("aten::rsqrt(Tensor self) -> Tensor");
    return table->getOp<Tensor (const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this);
#endif
}
inline Tensor & Tensor::rsqrt_() {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::rsqrt_(*this);
            break;
        default:
            AT_ERROR("rsqrt_ not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::rsqrt_(Tensor(a!) self) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this);
#endif
}
#ifdef NAMEDTENSOR_ENABLED
inline Tensor Tensor::select(Dimname dim, int64_t index) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::select(*this, dim, index);
#else
    static auto table = globalATenDispatch().getOpTable("aten::select(Tensor(a) self, Dimname dim, int index) -> Tensor(a)");
    return table->getOp<Tensor (const Tensor &, Dimname, int64_t)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, dim, index);
#endif
}
#endif
inline Tensor Tensor::select(int64_t dim, int64_t index) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::select(*this, dim, index);
#else
    static auto table = globalATenDispatch().getOpTable("aten::select(Tensor(a) self, int dim, int index) -> Tensor(a)");
    return table->getOp<Tensor (const Tensor &, int64_t, int64_t)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, dim, index);
#endif
}
inline Tensor Tensor::sigmoid() const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::sigmoid(*this);
            break;
        default:
            AT_ERROR("sigmoid not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::sigmoid(Tensor self) -> Tensor");
    return table->getOp<Tensor (const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this);
#endif
}
inline Tensor & Tensor::sigmoid_() {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::sigmoid_(*this);
            break;
        default:
            AT_ERROR("sigmoid_ not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::sigmoid_(Tensor(a!) self) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this);
#endif
}
inline Tensor Tensor::sin() const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::sin(*this);
#else
    static auto table = globalATenDispatch().getOpTable("aten::sin(Tensor self) -> Tensor");
    return table->getOp<Tensor (const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this);
#endif
}
inline Tensor & Tensor::sin_() {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::sin_(*this);
            break;
        default:
            AT_ERROR("sin_ not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::sin_(Tensor(a!) self) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this);
#endif
}
inline Tensor Tensor::sinh() const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::sinh(*this);
#else
    static auto table = globalATenDispatch().getOpTable("aten::sinh(Tensor self) -> Tensor");
    return table->getOp<Tensor (const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this);
#endif
}
inline Tensor & Tensor::sinh_() {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::sinh_(*this);
            break;
        default:
            AT_ERROR("sinh_ not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::sinh_(Tensor(a!) self) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this);
#endif
}
inline Tensor Tensor::detach() const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::detach(*this);
#else
    static auto table = globalATenDispatch().getOpTable("aten::detach(Tensor self) -> Tensor");
    return table->getOp<Tensor (const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this);
#endif
}
inline Tensor & Tensor::detach_() {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::detach_(*this);
#else
    static auto table = globalATenDispatch().getOpTable("aten::detach_(Tensor(a!) self) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this);
#endif
}
inline int64_t Tensor::size(int64_t dim) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::size(*this, dim);
#else
    static auto table = globalATenDispatch().getOpTable("aten::size(Tensor self, int dim) -> int");
    return table->getOp<int64_t (const Tensor &, int64_t)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, dim);
#endif
}
inline Tensor Tensor::slice(int64_t dim, int64_t start, int64_t end, int64_t step) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::slice(*this, dim, start, end, step);
#else
    static auto table = globalATenDispatch().getOpTable("aten::slice(Tensor(a) self, int dim=0, int start=0, int end=9223372036854775807, int step=1) -> Tensor(a)");
    return table->getOp<Tensor (const Tensor &, int64_t, int64_t, int64_t, int64_t)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, dim, start, end, step);
#endif
}
inline std::tuple<Tensor,Tensor> Tensor::slogdet() const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::slogdet(*this);
#else
    static auto table = globalATenDispatch().getOpTable("aten::slogdet(Tensor self) -> (Tensor sign, Tensor logabsdet)");
    return table->getOp<std::tuple<Tensor,Tensor> (const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this);
#endif
}
inline Tensor Tensor::smm(const Tensor & mat2) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::smm(*this, mat2);
#else
    static auto table = globalATenDispatch().getOpTable("aten::smm(Tensor self, Tensor mat2) -> Tensor");
    return table->getOp<Tensor (const Tensor &, const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, mat2);
#endif
}
inline Tensor Tensor::softmax(int64_t dim, c10::optional<ScalarType> dtype) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::softmax(*this, dim, dtype);
#else
    static auto table = globalATenDispatch().getOpTable("aten::softmax(Tensor self, int dim, ScalarType? dtype=None) -> Tensor");
    return table->getOp<Tensor (const Tensor &, int64_t, c10::optional<ScalarType>)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, dim, dtype);
#endif
}
inline std::vector<Tensor> Tensor::split(int64_t split_size, int64_t dim) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::split(*this, split_size, dim);
#else
    static auto table = globalATenDispatch().getOpTable("aten::split(Tensor(a) self, int split_size, int dim=0) -> Tensor(a)[]");
    return table->getOp<std::vector<Tensor> (const Tensor &, int64_t, int64_t)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, split_size, dim);
#endif
}
inline std::vector<Tensor> Tensor::split_with_sizes(IntArrayRef split_sizes, int64_t dim) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::split_with_sizes(*this, split_sizes, dim);
#else
    static auto table = globalATenDispatch().getOpTable("aten::split_with_sizes(Tensor self, int[] split_sizes, int dim=0) -> Tensor[]");
    return table->getOp<std::vector<Tensor> (const Tensor &, IntArrayRef, int64_t)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, split_sizes, dim);
#endif
}
inline Tensor Tensor::squeeze() const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::squeeze(*this);
#else
    static auto table = globalATenDispatch().getOpTable("aten::squeeze(Tensor(a) self) -> Tensor(a)");
    return table->getOp<Tensor (const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this);
#endif
}
inline Tensor Tensor::squeeze(int64_t dim) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::squeeze(*this, dim);
#else
    static auto table = globalATenDispatch().getOpTable("aten::squeeze(Tensor(a) self, int dim) -> Tensor(a)");
    return table->getOp<Tensor (const Tensor &, int64_t)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, dim);
#endif
}
inline Tensor & Tensor::squeeze_() {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::squeeze_(*this);
#else
    static auto table = globalATenDispatch().getOpTable("aten::squeeze_(Tensor(a!) self) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this);
#endif
}
inline Tensor & Tensor::squeeze_(int64_t dim) {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::squeeze_(*this, dim);
#else
    static auto table = globalATenDispatch().getOpTable("aten::squeeze_(Tensor(a!) self, int dim) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, int64_t)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, dim);
#endif
}
inline Tensor Tensor::sspaddmm(const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::sspaddmm(*this, mat1, mat2, beta, alpha);
#else
    static auto table = globalATenDispatch().getOpTable("aten::sspaddmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1) -> Tensor");
    return table->getOp<Tensor (const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, mat1, mat2, beta, alpha);
#endif
}
inline Tensor Tensor::stft(int64_t n_fft, c10::optional<int64_t> hop_length, c10::optional<int64_t> win_length, const Tensor & window, bool normalized, bool onesided) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::stft(*this, n_fft, hop_length, win_length, window, normalized, onesided);
#else
    static auto table = globalATenDispatch().getOpTable("aten::stft(Tensor self, int n_fft, int? hop_length=None, int? win_length=None, Tensor? window=None, bool normalized=False, bool onesided=True) -> Tensor");
    return table->getOp<Tensor (const Tensor &, int64_t, c10::optional<int64_t>, c10::optional<int64_t>, const Tensor &, bool, bool)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, n_fft, hop_length, win_length, window, normalized, onesided);
#endif
}
inline int64_t Tensor::stride(int64_t dim) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::stride(*this, dim);
#else
    static auto table = globalATenDispatch().getOpTable("aten::stride(Tensor self, int dim) -> int");
    return table->getOp<int64_t (const Tensor &, int64_t)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, dim);
#endif
}
inline Tensor Tensor::sum(c10::optional<ScalarType> dtype) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::sum(*this, dtype);
#else
    static auto table = globalATenDispatch().getOpTable("aten::sum(Tensor self, *, ScalarType? dtype=None) -> Tensor");
    return table->getOp<Tensor (const Tensor &, c10::optional<ScalarType>)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, dtype);
#endif
}
inline Tensor Tensor::sum(IntArrayRef dim, bool keepdim, c10::optional<ScalarType> dtype) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::sum(*this, dim, keepdim, dtype);
#else
    static auto table = globalATenDispatch().getOpTable("aten::sum(Tensor self, int[1] dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor");
    return table->getOp<Tensor (const Tensor &, IntArrayRef, bool, c10::optional<ScalarType>)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, dim, keepdim, dtype);
#endif
}
inline Tensor Tensor::sum_to_size(IntArrayRef size) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::sum_to_size(*this, size);
#else
    static auto table = globalATenDispatch().getOpTable("aten::sum_to_size(Tensor self, int[] size) -> Tensor");
    return table->getOp<Tensor (const Tensor &, IntArrayRef)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, size);
#endif
}
inline Tensor Tensor::sqrt() const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::sqrt(*this);
#else
    static auto table = globalATenDispatch().getOpTable("aten::sqrt(Tensor self) -> Tensor");
    return table->getOp<Tensor (const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this);
#endif
}
inline Tensor & Tensor::sqrt_() {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::sqrt_(*this);
            break;
        default:
            AT_ERROR("sqrt_ not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::sqrt_(Tensor(a!) self) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this);
#endif
}
inline Tensor Tensor::std(bool unbiased) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::std(*this, unbiased);
#else
    static auto table = globalATenDispatch().getOpTable("aten::std(Tensor self, bool unbiased=True) -> Tensor");
    return table->getOp<Tensor (const Tensor &, bool)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, unbiased);
#endif
}
inline Tensor Tensor::std(IntArrayRef dim, bool unbiased, bool keepdim) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::std(*this, dim, unbiased, keepdim);
#else
    static auto table = globalATenDispatch().getOpTable("aten::std(Tensor self, int[1] dim, bool unbiased=True, bool keepdim=False) -> Tensor");
    return table->getOp<Tensor (const Tensor &, IntArrayRef, bool, bool)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, dim, unbiased, keepdim);
#endif
}
inline Tensor Tensor::prod(c10::optional<ScalarType> dtype) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::prod(*this, dtype);
#else
    static auto table = globalATenDispatch().getOpTable("aten::prod(Tensor self, *, ScalarType? dtype=None) -> Tensor");
    return table->getOp<Tensor (const Tensor &, c10::optional<ScalarType>)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, dtype);
#endif
}
inline Tensor Tensor::prod(int64_t dim, bool keepdim, c10::optional<ScalarType> dtype) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::prod(*this, dim, keepdim, dtype);
#else
    static auto table = globalATenDispatch().getOpTable("aten::prod(Tensor self, int dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor");
    return table->getOp<Tensor (const Tensor &, int64_t, bool, c10::optional<ScalarType>)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, dim, keepdim, dtype);
#endif
}
inline Tensor Tensor::t() const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::t(*this);
#else
    static auto table = globalATenDispatch().getOpTable("aten::t(Tensor(a) self) -> Tensor(a)");
    return table->getOp<Tensor (const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this);
#endif
}
inline Tensor & Tensor::t_() {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::t_(*this);
#else
    static auto table = globalATenDispatch().getOpTable("aten::t_(Tensor(a!) self) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this);
#endif
}
inline Tensor Tensor::tan() const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::tan(*this);
#else
    static auto table = globalATenDispatch().getOpTable("aten::tan(Tensor self) -> Tensor");
    return table->getOp<Tensor (const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this);
#endif
}
inline Tensor & Tensor::tan_() {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::tan_(*this);
            break;
        default:
            AT_ERROR("tan_ not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::tan_(Tensor(a!) self) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this);
#endif
}
inline Tensor Tensor::tanh() const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::tanh(*this);
#else
    static auto table = globalATenDispatch().getOpTable("aten::tanh(Tensor self) -> Tensor");
    return table->getOp<Tensor (const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this);
#endif
}
inline Tensor & Tensor::tanh_() {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::tanh_(*this);
            break;
        default:
            AT_ERROR("tanh_ not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::tanh_(Tensor(a!) self) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this);
#endif
}
inline Tensor Tensor::transpose(int64_t dim0, int64_t dim1) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::transpose(*this, dim0, dim1);
#else
    static auto table = globalATenDispatch().getOpTable("aten::transpose(Tensor(a) self, int dim0, int dim1) -> Tensor(a)");
    return table->getOp<Tensor (const Tensor &, int64_t, int64_t)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, dim0, dim1);
#endif
}
inline Tensor & Tensor::transpose_(int64_t dim0, int64_t dim1) {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::transpose_(*this, dim0, dim1);
#else
    static auto table = globalATenDispatch().getOpTable("aten::transpose_(Tensor(a!) self, int dim0, int dim1) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, int64_t, int64_t)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, dim0, dim1);
#endif
}
inline Tensor Tensor::flip(IntArrayRef dims) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::flip(*this, dims);
            break;
        default:
            AT_ERROR("flip not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::flip(Tensor self, int[] dims) -> Tensor");
    return table->getOp<Tensor (const Tensor &, IntArrayRef)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, dims);
#endif
}
inline Tensor Tensor::roll(IntArrayRef shifts, IntArrayRef dims) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::roll(*this, shifts, dims);
            break;
        default:
            AT_ERROR("roll not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::roll(Tensor self, int[1] shifts, int[1] dims=[]) -> Tensor");
    return table->getOp<Tensor (const Tensor &, IntArrayRef, IntArrayRef)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, shifts, dims);
#endif
}
inline Tensor Tensor::rot90(int64_t k, IntArrayRef dims) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::rot90(*this, k, dims);
#else
    static auto table = globalATenDispatch().getOpTable("aten::rot90(Tensor self, int k=1, int[] dims=[0,1]) -> Tensor");
    return table->getOp<Tensor (const Tensor &, int64_t, IntArrayRef)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, k, dims);
#endif
}
inline Tensor Tensor::trunc() const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::trunc(*this);
#else
    static auto table = globalATenDispatch().getOpTable("aten::trunc(Tensor self) -> Tensor");
    return table->getOp<Tensor (const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this);
#endif
}
inline Tensor & Tensor::trunc_() {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::trunc_(*this);
            break;
        default:
            AT_ERROR("trunc_ not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::trunc_(Tensor(a!) self) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this);
#endif
}
inline Tensor Tensor::type_as(const Tensor & other) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::type_as(*this, other);
#else
    static auto table = globalATenDispatch().getOpTable("aten::type_as(Tensor self, Tensor other) -> Tensor");
    return table->getOp<Tensor (const Tensor &, const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, other);
#endif
}
inline Tensor Tensor::unsqueeze(int64_t dim) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::unsqueeze(*this, dim);
#else
    static auto table = globalATenDispatch().getOpTable("aten::unsqueeze(Tensor(a) self, int dim) -> Tensor(a)");
    return table->getOp<Tensor (const Tensor &, int64_t)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, dim);
#endif
}
inline Tensor & Tensor::unsqueeze_(int64_t dim) {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::unsqueeze_(*this, dim);
#else
    static auto table = globalATenDispatch().getOpTable("aten::unsqueeze_(Tensor(a!) self, int dim) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, int64_t)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, dim);
#endif
}
inline Tensor Tensor::var(bool unbiased) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::var(*this, unbiased);
#else
    static auto table = globalATenDispatch().getOpTable("aten::var(Tensor self, bool unbiased=True) -> Tensor");
    return table->getOp<Tensor (const Tensor &, bool)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, unbiased);
#endif
}
inline Tensor Tensor::var(IntArrayRef dim, bool unbiased, bool keepdim) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::var(*this, dim, unbiased, keepdim);
#else
    static auto table = globalATenDispatch().getOpTable("aten::var(Tensor self, int[1] dim, bool unbiased=True, bool keepdim=False) -> Tensor");
    return table->getOp<Tensor (const Tensor &, IntArrayRef, bool, bool)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, dim, unbiased, keepdim);
#endif
}
inline Tensor Tensor::view_as(const Tensor & other) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::view_as(*this, other);
#else
    static auto table = globalATenDispatch().getOpTable("aten::view_as(Tensor self, Tensor other) -> Tensor");
    return table->getOp<Tensor (const Tensor &, const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, other);
#endif
}
inline Tensor Tensor::where(const Tensor & condition, const Tensor & other) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::where(condition, *this, other);
#else
    static auto table = globalATenDispatch().getOpTable("aten::where(Tensor condition, Tensor self, Tensor other) -> Tensor");
    return table->getOp<Tensor (const Tensor &, const Tensor &, const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(condition, *this, other);
#endif
}
inline Tensor Tensor::norm(c10::optional<Scalar> p, ScalarType dtype) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::norm(*this, p, dtype);
#else
    static auto table = globalATenDispatch().getOpTable("aten::norm(Tensor self, Scalar? p, *, ScalarType dtype) -> Tensor");
    return table->getOp<Tensor (const Tensor &, c10::optional<Scalar>, ScalarType)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, p, dtype);
#endif
}
inline Tensor Tensor::norm(Scalar p) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::norm(*this, p);
#else
    static auto table = globalATenDispatch().getOpTable("aten::norm(Tensor self, Scalar p=2) -> Tensor");
    return table->getOp<Tensor (const Tensor &, Scalar)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, p);
#endif
}
inline Tensor Tensor::norm(c10::optional<Scalar> p, IntArrayRef dim, bool keepdim, ScalarType dtype) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::norm(*this, p, dim, keepdim, dtype);
#else
    static auto table = globalATenDispatch().getOpTable("aten::norm(Tensor self, Scalar? p, int[1] dim, bool keepdim, *, ScalarType dtype) -> Tensor");
    return table->getOp<Tensor (const Tensor &, c10::optional<Scalar>, IntArrayRef, bool, ScalarType)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, p, dim, keepdim, dtype);
#endif
}
inline Tensor Tensor::norm(c10::optional<Scalar> p, IntArrayRef dim, bool keepdim) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::norm(*this, p, dim, keepdim);
#else
    static auto table = globalATenDispatch().getOpTable("aten::norm(Tensor self, Scalar? p, int[1] dim, bool keepdim=False) -> Tensor");
    return table->getOp<Tensor (const Tensor &, c10::optional<Scalar>, IntArrayRef, bool)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, p, dim, keepdim);
#endif
}
inline Tensor Tensor::clone() const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::clone(*this);
            break;
        case Backend::SparseCPU:
            return SparseCPUType::clone(*this);
            break;
        default:
            AT_ERROR("clone not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::clone(Tensor self) -> Tensor");
    return table->getOp<Tensor (const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this);
#endif
}
inline Tensor & Tensor::resize_as_(const Tensor & the_template) {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::resize_as_(*this, the_template);
            break;
        case Backend::SparseCPU:
            return SparseCPUType::resize_as_(*this, the_template);
            break;
        default:
            AT_ERROR("resize_as_ not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::resize_as_(Tensor(a!) self, Tensor the_template) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, the_template);
#endif
}
inline Tensor Tensor::pow(Scalar exponent) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::pow(*this, exponent);
            break;
        case Backend::SparseCPU:
            return SparseCPUType::pow(*this, exponent);
            break;
        default:
            AT_ERROR("pow not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::pow(Tensor self, Scalar exponent) -> Tensor");
    return table->getOp<Tensor (const Tensor &, Scalar)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, exponent);
#endif
}
inline Tensor & Tensor::zero_() {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::zero_(*this);
            break;
        case Backend::SparseCPU:
            return SparseCPUType::zero_(*this);
            break;
        default:
            AT_ERROR("zero_ not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::zero_(Tensor(a!) self) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this);
#endif
}
inline Tensor Tensor::sub(const Tensor & other, Scalar alpha) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::sub(*this, other, alpha);
#else
    static auto table = globalATenDispatch().getOpTable("aten::sub(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor");
    return table->getOp<Tensor (const Tensor &, const Tensor &, Scalar)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, other, alpha);
#endif
}
inline Tensor & Tensor::sub_(const Tensor & other, Scalar alpha) {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::sub_(*this, other, alpha);
#else
    static auto table = globalATenDispatch().getOpTable("aten::sub_(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, const Tensor &, Scalar)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, other, alpha);
#endif
}
inline Tensor Tensor::sub(Scalar other, Scalar alpha) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::sub(*this, other, alpha);
#else
    static auto table = globalATenDispatch().getOpTable("aten::sub(Tensor self, Scalar other, Scalar alpha=1) -> Tensor");
    return table->getOp<Tensor (const Tensor &, Scalar, Scalar)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, other, alpha);
#endif
}
inline Tensor & Tensor::sub_(Scalar other, Scalar alpha) {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::sub_(*this, other, alpha);
#else
    static auto table = globalATenDispatch().getOpTable("aten::sub_(Tensor(a!) self, Scalar other, Scalar alpha=1) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, Scalar, Scalar)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, other, alpha);
#endif
}
inline Tensor Tensor::addmm(const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::addmm(*this, mat1, mat2, beta, alpha);
#else
    static auto table = globalATenDispatch().getOpTable("aten::addmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1) -> Tensor");
    return table->getOp<Tensor (const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, mat1, mat2, beta, alpha);
#endif
}
inline Tensor & Tensor::addmm_(const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::addmm_(*this, mat1, mat2, beta, alpha);
#else
    static auto table = globalATenDispatch().getOpTable("aten::addmm_(Tensor(a!) self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, const Tensor &, const Tensor &, Scalar, Scalar)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, mat1, mat2, beta, alpha);
#endif
}
inline Tensor & Tensor::sparse_resize_(IntArrayRef size, int64_t sparse_dim, int64_t dense_dim) {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::SparseCPU:
            return SparseCPUType::sparse_resize_(*this, size, sparse_dim, dense_dim);
            break;
        default:
            AT_ERROR("sparse_resize_ not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::sparse_resize_(Tensor(a!) self, int[] size, int sparse_dim, int dense_dim) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, IntArrayRef, int64_t, int64_t)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, size, sparse_dim, dense_dim);
#endif
}
inline Tensor & Tensor::sparse_resize_and_clear_(IntArrayRef size, int64_t sparse_dim, int64_t dense_dim) {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::SparseCPU:
            return SparseCPUType::sparse_resize_and_clear_(*this, size, sparse_dim, dense_dim);
            break;
        default:
            AT_ERROR("sparse_resize_and_clear_ not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::sparse_resize_and_clear_(Tensor(a!) self, int[] size, int sparse_dim, int dense_dim) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, IntArrayRef, int64_t, int64_t)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, size, sparse_dim, dense_dim);
#endif
}
inline Tensor Tensor::sparse_mask(const Tensor & mask) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::sparse_mask(*this, mask);
            break;
        default:
            AT_ERROR("sparse_mask not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::sparse_mask(Tensor self, Tensor mask) -> Tensor");
    return table->getOp<Tensor (const Tensor &, const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, mask);
#endif
}
inline Tensor Tensor::to_dense() const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::SparseCPU:
            return SparseCPUType::to_dense(*this);
            break;
        default:
            AT_ERROR("to_dense not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::to_dense(Tensor self) -> Tensor");
    return table->getOp<Tensor (const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this);
#endif
}
inline int64_t Tensor::sparse_dim() const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::SparseCPU:
            return SparseCPUType::sparse_dim(*this);
            break;
        default:
            AT_ERROR("sparse_dim not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::sparse_dim(Tensor self) -> int");
    return table->getOp<int64_t (const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this);
#endif
}
inline int64_t Tensor::_dimI() const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::SparseCPU:
            return SparseCPUType::_dimI(*this);
            break;
        default:
            AT_ERROR("_dimI not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::_dimI(Tensor self) -> int");
    return table->getOp<int64_t (const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this);
#endif
}
inline int64_t Tensor::dense_dim() const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::SparseCPU:
            return SparseCPUType::dense_dim(*this);
            break;
        default:
            AT_ERROR("dense_dim not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::dense_dim(Tensor self) -> int");
    return table->getOp<int64_t (const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this);
#endif
}
inline int64_t Tensor::_dimV() const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::SparseCPU:
            return SparseCPUType::_dimV(*this);
            break;
        default:
            AT_ERROR("_dimV not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::_dimV(Tensor self) -> int");
    return table->getOp<int64_t (const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this);
#endif
}
inline int64_t Tensor::_nnz() const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::SparseCPU:
            return SparseCPUType::_nnz(*this);
            break;
        default:
            AT_ERROR("_nnz not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::_nnz(Tensor self) -> int");
    return table->getOp<int64_t (const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this);
#endif
}
inline Tensor Tensor::coalesce() const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::SparseCPU:
            return SparseCPUType::coalesce(*this);
            break;
        default:
            AT_ERROR("coalesce not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::coalesce(Tensor self) -> Tensor");
    return table->getOp<Tensor (const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this);
#endif
}
inline bool Tensor::is_coalesced() const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::SparseCPU:
            return SparseCPUType::is_coalesced(*this);
            break;
        default:
            AT_ERROR("is_coalesced not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::is_coalesced(Tensor self) -> bool");
    return table->getOp<bool (const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this);
#endif
}
inline Tensor Tensor::_indices() const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::SparseCPU:
            return SparseCPUType::_indices(*this);
            break;
        default:
            AT_ERROR("_indices not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::_indices(Tensor(a) self) -> Tensor(a)");
    return table->getOp<Tensor (const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this);
#endif
}
inline Tensor Tensor::_values() const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::SparseCPU:
            return SparseCPUType::_values(*this);
            break;
        default:
            AT_ERROR("_values not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::_values(Tensor(a) self) -> Tensor(a)");
    return table->getOp<Tensor (const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this);
#endif
}
inline Tensor & Tensor::_coalesced_(bool coalesced) {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::SparseCPU:
            return SparseCPUType::_coalesced_(*this, coalesced);
            break;
        default:
            AT_ERROR("_coalesced_ not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::_coalesced_(Tensor(a!) self, bool coalesced) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, bool)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, coalesced);
#endif
}
inline Tensor Tensor::indices() const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::SparseCPU:
            return SparseCPUType::indices(*this);
            break;
        default:
            AT_ERROR("indices not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::indices(Tensor(a) self) -> Tensor(a)");
    return table->getOp<Tensor (const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this);
#endif
}
inline Tensor Tensor::values() const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::SparseCPU:
            return SparseCPUType::values(*this);
            break;
        default:
            AT_ERROR("values not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::values(Tensor(a) self) -> Tensor(a)");
    return table->getOp<Tensor (const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this);
#endif
}
inline int64_t Tensor::numel() const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::numel(*this);
#else
    static auto table = globalATenDispatch().getOpTable("aten::numel(Tensor self) -> int");
    return table->getOp<int64_t (const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this);
#endif
}
inline std::vector<Tensor> Tensor::unbind(int64_t dim) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::unbind(*this, dim);
#else
    static auto table = globalATenDispatch().getOpTable("aten::unbind(Tensor(a) self, int dim=0) -> Tensor(a)[]");
    return table->getOp<std::vector<Tensor> (const Tensor &, int64_t)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, dim);
#endif
}
inline Tensor Tensor::to_sparse(int64_t sparse_dim) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::to_sparse(*this, sparse_dim);
            break;
        default:
            AT_ERROR("to_sparse not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::to_sparse(Tensor self, int sparse_dim) -> Tensor");
    return table->getOp<Tensor (const Tensor &, int64_t)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, sparse_dim);
#endif
}
inline Tensor Tensor::to_sparse() const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::to_sparse(*this);
            break;
        default:
            AT_ERROR("to_sparse not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::to_sparse(Tensor self) -> Tensor");
    return table->getOp<Tensor (const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this);
#endif
}
inline Tensor Tensor::to_mkldnn() const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::to_mkldnn(*this);
            break;
        default:
            AT_ERROR("to_mkldnn not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::to_mkldnn(Tensor self) -> Tensor");
    return table->getOp<Tensor (const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this);
#endif
}
inline Tensor Tensor::dequantize() const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::QuantizedCPU:
            return QuantizedCPUType::dequantize(*this);
            break;
        default:
            AT_ERROR("dequantize not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::dequantize(Tensor self) -> Tensor");
    return table->getOp<Tensor (const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this);
#endif
}
inline double Tensor::q_scale() const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::QuantizedCPU:
            return QuantizedCPUType::q_scale(*this);
            break;
        default:
            AT_ERROR("q_scale not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::q_scale(Tensor self) -> float");
    return table->getOp<double (const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this);
#endif
}
inline int64_t Tensor::q_zero_point() const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::QuantizedCPU:
            return QuantizedCPUType::q_zero_point(*this);
            break;
        default:
            AT_ERROR("q_zero_point not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::q_zero_point(Tensor self) -> int");
    return table->getOp<int64_t (const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this);
#endif
}
inline Tensor Tensor::int_repr() const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::QuantizedCPU:
            return QuantizedCPUType::int_repr(*this);
            break;
        default:
            AT_ERROR("int_repr not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::int_repr(Tensor self) -> Tensor");
    return table->getOp<Tensor (const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this);
#endif
}
inline QScheme Tensor::qscheme() const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::QuantizedCPU:
            return QuantizedCPUType::qscheme(*this);
            break;
        default:
            AT_ERROR("qscheme not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::qscheme(Tensor self) -> QScheme");
    return table->getOp<QScheme (const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this);
#endif
}
inline Tensor Tensor::to(const TensorOptions & options, bool non_blocking, bool copy) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::to(*this, options, non_blocking, copy);
#else
    static auto table = globalATenDispatch().getOpTable("aten::to(Tensor self, *, ScalarType dtype, Layout layout, Device device, bool pin_memory=False, bool non_blocking=False, bool copy=False) -> Tensor");
    return table->getOp<Tensor (const Tensor &, const TensorOptions &, bool, bool)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, options, non_blocking, copy);
#endif
}
inline Tensor Tensor::to(Device device, ScalarType dtype, bool non_blocking, bool copy) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::to(*this, device, dtype, non_blocking, copy);
#else
    static auto table = globalATenDispatch().getOpTable("aten::to(Tensor self, Device device, ScalarType dtype, bool non_blocking=False, bool copy=False) -> Tensor");
    return table->getOp<Tensor (const Tensor &, Device, ScalarType, bool, bool)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, device, dtype, non_blocking, copy);
#endif
}
inline Tensor Tensor::to(ScalarType dtype, bool non_blocking, bool copy) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::to(*this, dtype, non_blocking, copy);
#else
    static auto table = globalATenDispatch().getOpTable("aten::to(Tensor self, ScalarType dtype, bool non_blocking=False, bool copy=False) -> Tensor");
    return table->getOp<Tensor (const Tensor &, ScalarType, bool, bool)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, dtype, non_blocking, copy);
#endif
}
inline Tensor Tensor::to(const Tensor & other, bool non_blocking, bool copy) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::to(*this, other, non_blocking, copy);
#else
    static auto table = globalATenDispatch().getOpTable("aten::to(Tensor self, Tensor other, bool non_blocking=False, bool copy=False) -> Tensor");
    return table->getOp<Tensor (const Tensor &, const Tensor &, bool, bool)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, other, non_blocking, copy);
#endif
}
inline Scalar Tensor::item() const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::item(*this);
#else
    static auto table = globalATenDispatch().getOpTable("aten::item(Tensor self) -> Scalar");
    return table->getOp<Scalar (const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this);
#endif
}
inline Tensor & Tensor::set_(Storage source) {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::set_(*this, source);
            break;
        default:
            AT_ERROR("set_ not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::set_(Tensor(a!) self, Storage source) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, Storage)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, source);
#endif
}
inline Tensor & Tensor::set_(Storage source, int64_t storage_offset, IntArrayRef size, IntArrayRef stride) {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::set_(*this, source, storage_offset, size, stride);
            break;
        case Backend::QuantizedCPU:
            return QuantizedCPUType::set_(*this, source, storage_offset, size, stride);
            break;
        default:
            AT_ERROR("set_ not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::set_(Tensor(a!) self, Storage source, int storage_offset, int[] size, int[] stride=[]) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, Storage, int64_t, IntArrayRef, IntArrayRef)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, source, storage_offset, size, stride);
#endif
}
inline Tensor & Tensor::set_(const Tensor & source) {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::set_(*this, source);
            break;
        default:
            AT_ERROR("set_ not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::set_(Tensor(a!) self, Tensor source) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, source);
#endif
}
inline Tensor & Tensor::set_() {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::set_(*this);
            break;
        default:
            AT_ERROR("set_ not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::set_(Tensor(a!) self) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this);
#endif
}
inline Tensor & Tensor::set_quantizer_(ConstQuantizerPtr quantizer) {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::QuantizedCPU:
            return QuantizedCPUType::set_quantizer_(*this, quantizer);
            break;
        default:
            AT_ERROR("set_quantizer_ not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::set_quantizer_(Tensor(a!) self, ConstQuantizerPtr quantizer) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, ConstQuantizerPtr)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, quantizer);
#endif
}
inline bool Tensor::is_set_to(const Tensor & tensor) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::is_set_to(*this, tensor);
            break;
        default:
            AT_ERROR("is_set_to not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::is_set_to(Tensor self, Tensor tensor) -> bool");
    return table->getOp<bool (const Tensor &, const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, tensor);
#endif
}
inline Tensor & Tensor::masked_fill_(const Tensor & mask, Scalar value) {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::masked_fill_(*this, mask, value);
            break;
        default:
            AT_ERROR("masked_fill_ not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::masked_fill_(Tensor(a!) self, Tensor mask, Scalar value) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, const Tensor &, Scalar)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, mask, value);
#endif
}
inline Tensor Tensor::masked_fill(const Tensor & mask, Scalar value) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::masked_fill(*this, mask, value);
#else
    static auto table = globalATenDispatch().getOpTable("aten::masked_fill(Tensor self, Tensor mask, Scalar value) -> Tensor");
    return table->getOp<Tensor (const Tensor &, const Tensor &, Scalar)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, mask, value);
#endif
}
inline Tensor & Tensor::masked_fill_(const Tensor & mask, const Tensor & value) {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::masked_fill_(*this, mask, value);
            break;
        default:
            AT_ERROR("masked_fill_ not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::masked_fill_(Tensor(a!) self, Tensor mask, Tensor value) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, const Tensor &, const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, mask, value);
#endif
}
inline Tensor Tensor::masked_fill(const Tensor & mask, const Tensor & value) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::masked_fill(*this, mask, value);
#else
    static auto table = globalATenDispatch().getOpTable("aten::masked_fill(Tensor self, Tensor mask, Tensor value) -> Tensor");
    return table->getOp<Tensor (const Tensor &, const Tensor &, const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, mask, value);
#endif
}
inline Tensor & Tensor::masked_scatter_(const Tensor & mask, const Tensor & source) {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::masked_scatter_(*this, mask, source);
            break;
        default:
            AT_ERROR("masked_scatter_ not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::masked_scatter_(Tensor(a!) self, Tensor mask, Tensor source) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, const Tensor &, const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, mask, source);
#endif
}
inline Tensor Tensor::masked_scatter(const Tensor & mask, const Tensor & source) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::masked_scatter(*this, mask, source);
#else
    static auto table = globalATenDispatch().getOpTable("aten::masked_scatter(Tensor self, Tensor mask, Tensor source) -> Tensor");
    return table->getOp<Tensor (const Tensor &, const Tensor &, const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, mask, source);
#endif
}
inline Tensor Tensor::view(IntArrayRef size) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::view(*this, size);
            break;
        default:
            AT_ERROR("view not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::view(Tensor(a) self, int[] size) -> Tensor(a)");
    return table->getOp<Tensor (const Tensor &, IntArrayRef)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, size);
#endif
}
inline Tensor & Tensor::put_(const Tensor & index, const Tensor & source, bool accumulate) {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::put_(*this, index, source, accumulate);
            break;
        default:
            AT_ERROR("put_ not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::put_(Tensor(a!) self, Tensor index, Tensor source, bool accumulate=False) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, const Tensor &, const Tensor &, bool)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, index, source, accumulate);
#endif
}
inline Tensor & Tensor::index_add_(int64_t dim, const Tensor & index, const Tensor & source) {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::index_add_(*this, dim, index, source);
            break;
        default:
            AT_ERROR("index_add_ not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::index_add_(Tensor(a!) self, int dim, Tensor index, Tensor source) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, int64_t, const Tensor &, const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, dim, index, source);
#endif
}
inline Tensor Tensor::index_add(int64_t dim, const Tensor & index, const Tensor & source) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::index_add(*this, dim, index, source);
#else
    static auto table = globalATenDispatch().getOpTable("aten::index_add(Tensor self, int dim, Tensor index, Tensor source) -> Tensor");
    return table->getOp<Tensor (const Tensor &, int64_t, const Tensor &, const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, dim, index, source);
#endif
}
inline Tensor & Tensor::index_fill_(int64_t dim, const Tensor & index, Scalar value) {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::index_fill_(*this, dim, index, value);
            break;
        default:
            AT_ERROR("index_fill_ not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::index_fill_(Tensor(a!) self, int dim, Tensor index, Scalar value) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, int64_t, const Tensor &, Scalar)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, dim, index, value);
#endif
}
inline Tensor Tensor::index_fill(int64_t dim, const Tensor & index, Scalar value) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::index_fill(*this, dim, index, value);
#else
    static auto table = globalATenDispatch().getOpTable("aten::index_fill(Tensor self, int dim, Tensor index, Scalar value) -> Tensor");
    return table->getOp<Tensor (const Tensor &, int64_t, const Tensor &, Scalar)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, dim, index, value);
#endif
}
inline Tensor & Tensor::index_fill_(int64_t dim, const Tensor & index, const Tensor & value) {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::index_fill_(*this, dim, index, value);
            break;
        default:
            AT_ERROR("index_fill_ not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::index_fill_(Tensor(a!) self, int dim, Tensor index, Tensor value) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, int64_t, const Tensor &, const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, dim, index, value);
#endif
}
inline Tensor Tensor::index_fill(int64_t dim, const Tensor & index, const Tensor & value) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::index_fill(*this, dim, index, value);
#else
    static auto table = globalATenDispatch().getOpTable("aten::index_fill(Tensor self, int dim, Tensor index, Tensor value) -> Tensor");
    return table->getOp<Tensor (const Tensor &, int64_t, const Tensor &, const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, dim, index, value);
#endif
}
inline Tensor & Tensor::scatter_(int64_t dim, const Tensor & index, const Tensor & src) {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::scatter_(*this, dim, index, src);
            break;
        default:
            AT_ERROR("scatter_ not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::scatter_(Tensor(a!) self, int dim, Tensor index, Tensor src) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, int64_t, const Tensor &, const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, dim, index, src);
#endif
}
inline Tensor Tensor::scatter(int64_t dim, const Tensor & index, const Tensor & src) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::scatter(*this, dim, index, src);
#else
    static auto table = globalATenDispatch().getOpTable("aten::scatter(Tensor self, int dim, Tensor index, Tensor src) -> Tensor");
    return table->getOp<Tensor (const Tensor &, int64_t, const Tensor &, const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, dim, index, src);
#endif
}
inline Tensor & Tensor::scatter_(int64_t dim, const Tensor & index, Scalar value) {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::scatter_(*this, dim, index, value);
            break;
        default:
            AT_ERROR("scatter_ not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::scatter_(Tensor(a!) self, int dim, Tensor index, Scalar value) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, int64_t, const Tensor &, Scalar)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, dim, index, value);
#endif
}
inline Tensor Tensor::scatter(int64_t dim, const Tensor & index, Scalar value) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::scatter(*this, dim, index, value);
#else
    static auto table = globalATenDispatch().getOpTable("aten::scatter(Tensor self, int dim, Tensor index, Scalar value) -> Tensor");
    return table->getOp<Tensor (const Tensor &, int64_t, const Tensor &, Scalar)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, dim, index, value);
#endif
}
inline Tensor & Tensor::scatter_add_(int64_t dim, const Tensor & index, const Tensor & src) {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::scatter_add_(*this, dim, index, src);
            break;
        default:
            AT_ERROR("scatter_add_ not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::scatter_add_(Tensor(a!) self, int dim, Tensor index, Tensor src) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, int64_t, const Tensor &, const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, dim, index, src);
#endif
}
inline Tensor Tensor::scatter_add(int64_t dim, const Tensor & index, const Tensor & src) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::scatter_add(*this, dim, index, src);
#else
    static auto table = globalATenDispatch().getOpTable("aten::scatter_add(Tensor self, int dim, Tensor index, Tensor src) -> Tensor");
    return table->getOp<Tensor (const Tensor &, int64_t, const Tensor &, const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, dim, index, src);
#endif
}
inline Tensor & Tensor::lt_(Scalar other) {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::lt_(*this, other);
            break;
        default:
            AT_ERROR("lt_ not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::lt_(Tensor(a!) self, Scalar other) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, Scalar)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, other);
#endif
}
inline Tensor & Tensor::lt_(const Tensor & other) {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::lt_(*this, other);
            break;
        default:
            AT_ERROR("lt_ not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::lt_(Tensor(a!) self, Tensor other) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, other);
#endif
}
inline Tensor & Tensor::gt_(Scalar other) {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::gt_(*this, other);
            break;
        default:
            AT_ERROR("gt_ not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::gt_(Tensor(a!) self, Scalar other) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, Scalar)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, other);
#endif
}
inline Tensor & Tensor::gt_(const Tensor & other) {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::gt_(*this, other);
            break;
        default:
            AT_ERROR("gt_ not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::gt_(Tensor(a!) self, Tensor other) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, other);
#endif
}
inline Tensor & Tensor::le_(Scalar other) {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::le_(*this, other);
            break;
        default:
            AT_ERROR("le_ not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::le_(Tensor(a!) self, Scalar other) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, Scalar)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, other);
#endif
}
inline Tensor & Tensor::le_(const Tensor & other) {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::le_(*this, other);
            break;
        default:
            AT_ERROR("le_ not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::le_(Tensor(a!) self, Tensor other) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, other);
#endif
}
inline Tensor & Tensor::ge_(Scalar other) {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::ge_(*this, other);
            break;
        default:
            AT_ERROR("ge_ not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::ge_(Tensor(a!) self, Scalar other) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, Scalar)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, other);
#endif
}
inline Tensor & Tensor::ge_(const Tensor & other) {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::ge_(*this, other);
            break;
        default:
            AT_ERROR("ge_ not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::ge_(Tensor(a!) self, Tensor other) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, other);
#endif
}
inline Tensor & Tensor::eq_(Scalar other) {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::eq_(*this, other);
            break;
        default:
            AT_ERROR("eq_ not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::eq_(Tensor(a!) self, Scalar other) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, Scalar)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, other);
#endif
}
inline Tensor & Tensor::eq_(const Tensor & other) {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::eq_(*this, other);
            break;
        default:
            AT_ERROR("eq_ not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::eq_(Tensor(a!) self, Tensor other) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, other);
#endif
}
inline Tensor & Tensor::ne_(Scalar other) {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::ne_(*this, other);
            break;
        default:
            AT_ERROR("ne_ not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::ne_(Tensor(a!) self, Scalar other) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, Scalar)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, other);
#endif
}
inline Tensor & Tensor::ne_(const Tensor & other) {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::ne_(*this, other);
            break;
        default:
            AT_ERROR("ne_ not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::ne_(Tensor(a!) self, Tensor other) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, other);
#endif
}
inline Tensor Tensor::__and__(Scalar other) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::__and__(*this, other);
            break;
        default:
            AT_ERROR("__and__ not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::__and__(Tensor self, Scalar other) -> Tensor");
    return table->getOp<Tensor (const Tensor &, Scalar)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, other);
#endif
}
inline Tensor Tensor::__and__(const Tensor & other) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::__and__(*this, other);
            break;
        default:
            AT_ERROR("__and__ not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::__and__(Tensor self, Tensor other) -> Tensor");
    return table->getOp<Tensor (const Tensor &, const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, other);
#endif
}
inline Tensor & Tensor::__iand__(Scalar other) {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::__iand__(*this, other);
            break;
        default:
            AT_ERROR("__iand__ not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::__iand__(Tensor(a!) self, Scalar other) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, Scalar)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, other);
#endif
}
inline Tensor & Tensor::__iand__(const Tensor & other) {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::__iand__(*this, other);
            break;
        default:
            AT_ERROR("__iand__ not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::__iand__(Tensor(a!) self, Tensor other) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, other);
#endif
}
inline Tensor Tensor::__or__(Scalar other) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::__or__(*this, other);
            break;
        default:
            AT_ERROR("__or__ not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::__or__(Tensor self, Scalar other) -> Tensor");
    return table->getOp<Tensor (const Tensor &, Scalar)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, other);
#endif
}
inline Tensor Tensor::__or__(const Tensor & other) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::__or__(*this, other);
            break;
        default:
            AT_ERROR("__or__ not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::__or__(Tensor self, Tensor other) -> Tensor");
    return table->getOp<Tensor (const Tensor &, const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, other);
#endif
}
inline Tensor & Tensor::__ior__(Scalar other) {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::__ior__(*this, other);
            break;
        default:
            AT_ERROR("__ior__ not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::__ior__(Tensor(a!) self, Scalar other) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, Scalar)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, other);
#endif
}
inline Tensor & Tensor::__ior__(const Tensor & other) {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::__ior__(*this, other);
            break;
        default:
            AT_ERROR("__ior__ not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::__ior__(Tensor(a!) self, Tensor other) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, other);
#endif
}
inline Tensor Tensor::__xor__(Scalar other) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::__xor__(*this, other);
            break;
        default:
            AT_ERROR("__xor__ not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::__xor__(Tensor self, Scalar other) -> Tensor");
    return table->getOp<Tensor (const Tensor &, Scalar)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, other);
#endif
}
inline Tensor Tensor::__xor__(const Tensor & other) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::__xor__(*this, other);
            break;
        default:
            AT_ERROR("__xor__ not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::__xor__(Tensor self, Tensor other) -> Tensor");
    return table->getOp<Tensor (const Tensor &, const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, other);
#endif
}
inline Tensor & Tensor::__ixor__(Scalar other) {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::__ixor__(*this, other);
            break;
        default:
            AT_ERROR("__ixor__ not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::__ixor__(Tensor(a!) self, Scalar other) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, Scalar)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, other);
#endif
}
inline Tensor & Tensor::__ixor__(const Tensor & other) {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::__ixor__(*this, other);
            break;
        default:
            AT_ERROR("__ixor__ not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::__ixor__(Tensor(a!) self, Tensor other) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, other);
#endif
}
inline Tensor Tensor::__lshift__(Scalar other) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::__lshift__(*this, other);
            break;
        default:
            AT_ERROR("__lshift__ not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::__lshift__(Tensor self, Scalar other) -> Tensor");
    return table->getOp<Tensor (const Tensor &, Scalar)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, other);
#endif
}
inline Tensor Tensor::__lshift__(const Tensor & other) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::__lshift__(*this, other);
            break;
        default:
            AT_ERROR("__lshift__ not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::__lshift__(Tensor self, Tensor other) -> Tensor");
    return table->getOp<Tensor (const Tensor &, const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, other);
#endif
}
inline Tensor & Tensor::__ilshift__(Scalar other) {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::__ilshift__(*this, other);
            break;
        default:
            AT_ERROR("__ilshift__ not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::__ilshift__(Tensor(a!) self, Scalar other) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, Scalar)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, other);
#endif
}
inline Tensor & Tensor::__ilshift__(const Tensor & other) {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::__ilshift__(*this, other);
            break;
        default:
            AT_ERROR("__ilshift__ not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::__ilshift__(Tensor(a!) self, Tensor other) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, other);
#endif
}
inline Tensor Tensor::__rshift__(Scalar other) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::__rshift__(*this, other);
            break;
        default:
            AT_ERROR("__rshift__ not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::__rshift__(Tensor self, Scalar other) -> Tensor");
    return table->getOp<Tensor (const Tensor &, Scalar)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, other);
#endif
}
inline Tensor Tensor::__rshift__(const Tensor & other) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::__rshift__(*this, other);
            break;
        default:
            AT_ERROR("__rshift__ not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::__rshift__(Tensor self, Tensor other) -> Tensor");
    return table->getOp<Tensor (const Tensor &, const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, other);
#endif
}
inline Tensor & Tensor::__irshift__(Scalar other) {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::__irshift__(*this, other);
            break;
        default:
            AT_ERROR("__irshift__ not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::__irshift__(Tensor(a!) self, Scalar other) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, Scalar)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, other);
#endif
}
inline Tensor & Tensor::__irshift__(const Tensor & other) {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::__irshift__(*this, other);
            break;
        default:
            AT_ERROR("__irshift__ not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::__irshift__(Tensor(a!) self, Tensor other) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, other);
#endif
}
inline Tensor & Tensor::lgamma_() {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::lgamma_(*this);
            break;
        default:
            AT_ERROR("lgamma_ not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::lgamma_(Tensor(a!) self) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this);
#endif
}
inline Tensor & Tensor::atan2_(const Tensor & other) {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::atan2_(*this, other);
            break;
        default:
            AT_ERROR("atan2_ not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::atan2_(Tensor(a!) self, Tensor other) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, other);
#endif
}
inline Tensor & Tensor::tril_(int64_t diagonal) {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::tril_(*this, diagonal);
            break;
        default:
            AT_ERROR("tril_ not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::tril_(Tensor(a!) self, int diagonal=0) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, int64_t)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, diagonal);
#endif
}
inline Tensor & Tensor::triu_(int64_t diagonal) {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::triu_(*this, diagonal);
            break;
        default:
            AT_ERROR("triu_ not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::triu_(Tensor(a!) self, int diagonal=0) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, int64_t)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, diagonal);
#endif
}
inline Tensor & Tensor::digamma_() {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::digamma_(*this);
            break;
        default:
            AT_ERROR("digamma_ not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::digamma_(Tensor(a!) self) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this);
#endif
}
inline Tensor & Tensor::polygamma_(int64_t n) {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::polygamma_(*this, n);
            break;
        default:
            AT_ERROR("polygamma_ not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::polygamma_(Tensor(a!) self, int n) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, int64_t)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, n);
#endif
}
inline Tensor & Tensor::erfinv_() {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::erfinv_(*this);
            break;
        default:
            AT_ERROR("erfinv_ not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::erfinv_(Tensor(a!) self) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this);
#endif
}
inline Tensor & Tensor::renorm_(Scalar p, int64_t dim, Scalar maxnorm) {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::renorm_(*this, p, dim, maxnorm);
            break;
        default:
            AT_ERROR("renorm_ not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::renorm_(Tensor(a!) self, Scalar p, int dim, Scalar maxnorm) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, Scalar, int64_t, Scalar)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, p, dim, maxnorm);
#endif
}
inline Tensor & Tensor::pow_(Scalar exponent) {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::pow_(*this, exponent);
            break;
        default:
            AT_ERROR("pow_ not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::pow_(Tensor(a!) self, Scalar exponent) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, Scalar)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, exponent);
#endif
}
inline Tensor & Tensor::pow_(const Tensor & exponent) {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::pow_(*this, exponent);
            break;
        default:
            AT_ERROR("pow_ not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::pow_(Tensor(a!) self, Tensor exponent) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, exponent);
#endif
}
inline Tensor & Tensor::lerp_(const Tensor & end, Scalar weight) {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::lerp_(*this, end, weight);
            break;
        default:
            AT_ERROR("lerp_ not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::lerp_(Tensor(a!) self, Tensor end, Scalar weight) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, const Tensor &, Scalar)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, end, weight);
#endif
}
inline Tensor & Tensor::lerp_(const Tensor & end, const Tensor & weight) {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::lerp_(*this, end, weight);
            break;
        default:
            AT_ERROR("lerp_ not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::lerp_(Tensor(a!) self, Tensor end, Tensor weight) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, const Tensor &, const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, end, weight);
#endif
}
inline Tensor & Tensor::sign_() {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::sign_(*this);
            break;
        default:
            AT_ERROR("sign_ not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::sign_(Tensor(a!) self) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this);
#endif
}
inline Tensor & Tensor::fmod_(Scalar other) {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::fmod_(*this, other);
            break;
        default:
            AT_ERROR("fmod_ not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::fmod_(Tensor(a!) self, Scalar other) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, Scalar)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, other);
#endif
}
inline Tensor & Tensor::fmod_(const Tensor & other) {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::fmod_(*this, other);
            break;
        default:
            AT_ERROR("fmod_ not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::fmod_(Tensor(a!) self, Tensor other) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, other);
#endif
}
inline Tensor & Tensor::remainder_(Scalar other) {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::remainder_(*this, other);
            break;
        default:
            AT_ERROR("remainder_ not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::remainder_(Tensor(a!) self, Scalar other) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, Scalar)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, other);
#endif
}
inline Tensor & Tensor::remainder_(const Tensor & other) {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::remainder_(*this, other);
            break;
        default:
            AT_ERROR("remainder_ not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::remainder_(Tensor(a!) self, Tensor other) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, other);
#endif
}
inline Tensor & Tensor::addbmm_(const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::addbmm_(*this, batch1, batch2, beta, alpha);
            break;
        default:
            AT_ERROR("addbmm_ not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::addbmm_(Tensor(a!) self, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, const Tensor &, const Tensor &, Scalar, Scalar)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, batch1, batch2, beta, alpha);
#endif
}
inline Tensor Tensor::addbmm(const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::addbmm(*this, batch1, batch2, beta, alpha);
            break;
        default:
            AT_ERROR("addbmm not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::addbmm(Tensor self, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1) -> Tensor");
    return table->getOp<Tensor (const Tensor &, const Tensor &, const Tensor &, Scalar, Scalar)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, batch1, batch2, beta, alpha);
#endif
}
inline Tensor & Tensor::addcmul_(const Tensor & tensor1, const Tensor & tensor2, Scalar value) {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::addcmul_(*this, tensor1, tensor2, value);
            break;
        default:
            AT_ERROR("addcmul_ not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::addcmul_(Tensor(a!) self, Tensor tensor1, Tensor tensor2, *, Scalar value=1) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, const Tensor &, const Tensor &, Scalar)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, tensor1, tensor2, value);
#endif
}
inline Tensor & Tensor::addcdiv_(const Tensor & tensor1, const Tensor & tensor2, Scalar value) {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::addcdiv_(*this, tensor1, tensor2, value);
            break;
        default:
            AT_ERROR("addcdiv_ not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::addcdiv_(Tensor(a!) self, Tensor tensor1, Tensor tensor2, *, Scalar value=1) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, const Tensor &, const Tensor &, Scalar)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, tensor1, tensor2, value);
#endif
}
inline Tensor & Tensor::random_(int64_t from, int64_t to, Generator * generator) {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::random_(*this, from, to, generator);
            break;
        default:
            AT_ERROR("random_ not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::random_(Tensor(a!) self, int from, int to, *, Generator? generator=None) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, int64_t, int64_t, Generator *)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, from, to, generator);
#endif
}
inline Tensor & Tensor::random_(int64_t to, Generator * generator) {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::random_(*this, to, generator);
            break;
        default:
            AT_ERROR("random_ not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::random_(Tensor(a!) self, int to, *, Generator? generator=None) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, int64_t, Generator *)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, to, generator);
#endif
}
inline Tensor & Tensor::random_(Generator * generator) {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::random_(*this, generator);
            break;
        default:
            AT_ERROR("random_ not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::random_(Tensor(a!) self, *, Generator? generator=None) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, Generator *)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, generator);
#endif
}
inline Tensor & Tensor::uniform_(double from, double to, Generator * generator) {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::uniform_(*this, from, to, generator);
            break;
        default:
            AT_ERROR("uniform_ not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::uniform_(Tensor(a!) self, float from=0, float to=1, *, Generator? generator=None) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, double, double, Generator *)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, from, to, generator);
#endif
}
inline Tensor & Tensor::normal_(double mean, double std, Generator * generator) {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::normal_(*this, mean, std, generator);
            break;
        default:
            AT_ERROR("normal_ not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::normal_(Tensor(a!) self, float mean=0, float std=1, *, Generator? generator=None) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, double, double, Generator *)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, mean, std, generator);
#endif
}
inline Tensor & Tensor::cauchy_(double median, double sigma, Generator * generator) {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::cauchy_(*this, median, sigma, generator);
            break;
        default:
            AT_ERROR("cauchy_ not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::cauchy_(Tensor(a!) self, float median=0, float sigma=1, *, Generator? generator=None) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, double, double, Generator *)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, median, sigma, generator);
#endif
}
inline Tensor & Tensor::log_normal_(double mean, double std, Generator * generator) {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::log_normal_(*this, mean, std, generator);
            break;
        default:
            AT_ERROR("log_normal_ not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::log_normal_(Tensor(a!) self, float mean=1, float std=2, *, Generator? generator=None) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, double, double, Generator *)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, mean, std, generator);
#endif
}
inline Tensor & Tensor::exponential_(double lambd, Generator * generator) {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::exponential_(*this, lambd, generator);
            break;
        default:
            AT_ERROR("exponential_ not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::exponential_(Tensor(a!) self, float lambd=1, *, Generator? generator=None) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, double, Generator *)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, lambd, generator);
#endif
}
inline Tensor & Tensor::geometric_(double p, Generator * generator) {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::geometric_(*this, p, generator);
            break;
        default:
            AT_ERROR("geometric_ not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::geometric_(Tensor(a!) self, float p, *, Generator? generator=None) -> Tensor(a!)");
    return table->getOp<Tensor & (Tensor &, double, Generator *)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, p, generator);
#endif
}
inline Tensor Tensor::diag(int64_t diagonal) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::diag(*this, diagonal);
            break;
        default:
            AT_ERROR("diag not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::diag(Tensor self, int diagonal=0) -> Tensor");
    return table->getOp<Tensor (const Tensor &, int64_t)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, diagonal);
#endif
}
inline Tensor Tensor::cross(const Tensor & other, c10::optional<int64_t> dim) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::cross(*this, other, dim);
#else
    static auto table = globalATenDispatch().getOpTable("aten::cross(Tensor self, Tensor other, int? dim=None) -> Tensor");
    return table->getOp<Tensor (const Tensor &, const Tensor &, c10::optional<int64_t>)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, other, dim);
#endif
}
inline Tensor Tensor::triu(int64_t diagonal) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::triu(*this, diagonal);
#else
    static auto table = globalATenDispatch().getOpTable("aten::triu(Tensor self, int diagonal=0) -> Tensor");
    return table->getOp<Tensor (const Tensor &, int64_t)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, diagonal);
#endif
}
inline Tensor Tensor::tril(int64_t diagonal) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::tril(*this, diagonal);
#else
    static auto table = globalATenDispatch().getOpTable("aten::tril(Tensor self, int diagonal=0) -> Tensor");
    return table->getOp<Tensor (const Tensor &, int64_t)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, diagonal);
#endif
}
inline Tensor Tensor::trace() const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::trace(*this);
            break;
        default:
            AT_ERROR("trace not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::trace(Tensor self) -> Tensor");
    return table->getOp<Tensor (const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this);
#endif
}
inline Tensor Tensor::ne(Scalar other) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::ne(*this, other);
            break;
        default:
            AT_ERROR("ne not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::ne(Tensor self, Scalar other) -> Tensor");
    return table->getOp<Tensor (const Tensor &, Scalar)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, other);
#endif
}
inline Tensor Tensor::ne(const Tensor & other) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::ne(*this, other);
            break;
        default:
            AT_ERROR("ne not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::ne(Tensor self, Tensor other) -> Tensor");
    return table->getOp<Tensor (const Tensor &, const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, other);
#endif
}
inline Tensor Tensor::eq(Scalar other) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::eq(*this, other);
            break;
        default:
            AT_ERROR("eq not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::eq(Tensor self, Scalar other) -> Tensor");
    return table->getOp<Tensor (const Tensor &, Scalar)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, other);
#endif
}
inline Tensor Tensor::eq(const Tensor & other) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::eq(*this, other);
            break;
        default:
            AT_ERROR("eq not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::eq(Tensor self, Tensor other) -> Tensor");
    return table->getOp<Tensor (const Tensor &, const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, other);
#endif
}
inline Tensor Tensor::ge(Scalar other) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::ge(*this, other);
            break;
        default:
            AT_ERROR("ge not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::ge(Tensor self, Scalar other) -> Tensor");
    return table->getOp<Tensor (const Tensor &, Scalar)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, other);
#endif
}
inline Tensor Tensor::ge(const Tensor & other) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::ge(*this, other);
            break;
        default:
            AT_ERROR("ge not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::ge(Tensor self, Tensor other) -> Tensor");
    return table->getOp<Tensor (const Tensor &, const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, other);
#endif
}
inline Tensor Tensor::le(Scalar other) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::le(*this, other);
            break;
        default:
            AT_ERROR("le not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::le(Tensor self, Scalar other) -> Tensor");
    return table->getOp<Tensor (const Tensor &, Scalar)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, other);
#endif
}
inline Tensor Tensor::le(const Tensor & other) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::le(*this, other);
            break;
        default:
            AT_ERROR("le not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::le(Tensor self, Tensor other) -> Tensor");
    return table->getOp<Tensor (const Tensor &, const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, other);
#endif
}
inline Tensor Tensor::gt(Scalar other) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::gt(*this, other);
            break;
        default:
            AT_ERROR("gt not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::gt(Tensor self, Scalar other) -> Tensor");
    return table->getOp<Tensor (const Tensor &, Scalar)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, other);
#endif
}
inline Tensor Tensor::gt(const Tensor & other) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::gt(*this, other);
            break;
        default:
            AT_ERROR("gt not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::gt(Tensor self, Tensor other) -> Tensor");
    return table->getOp<Tensor (const Tensor &, const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, other);
#endif
}
inline Tensor Tensor::lt(Scalar other) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::lt(*this, other);
            break;
        default:
            AT_ERROR("lt not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::lt(Tensor self, Scalar other) -> Tensor");
    return table->getOp<Tensor (const Tensor &, Scalar)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, other);
#endif
}
inline Tensor Tensor::lt(const Tensor & other) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::lt(*this, other);
            break;
        default:
            AT_ERROR("lt not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::lt(Tensor self, Tensor other) -> Tensor");
    return table->getOp<Tensor (const Tensor &, const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, other);
#endif
}
inline Tensor Tensor::take(const Tensor & index) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::take(*this, index);
            break;
        default:
            AT_ERROR("take not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::take(Tensor self, Tensor index) -> Tensor");
    return table->getOp<Tensor (const Tensor &, const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, index);
#endif
}
inline Tensor Tensor::index_select(int64_t dim, const Tensor & index) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::index_select(*this, dim, index);
            break;
        default:
            AT_ERROR("index_select not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::index_select(Tensor self, int dim, Tensor index) -> Tensor");
    return table->getOp<Tensor (const Tensor &, int64_t, const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, dim, index);
#endif
}
inline Tensor Tensor::masked_select(const Tensor & mask) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::masked_select(*this, mask);
            break;
        default:
            AT_ERROR("masked_select not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::masked_select(Tensor self, Tensor mask) -> Tensor");
    return table->getOp<Tensor (const Tensor &, const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, mask);
#endif
}
inline Tensor Tensor::nonzero() const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::nonzero(*this);
            break;
        default:
            AT_ERROR("nonzero not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::nonzero(Tensor self) -> Tensor");
    return table->getOp<Tensor (const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this);
#endif
}
inline std::vector<Tensor> Tensor::nonzero_numpy() const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::nonzero_numpy(*this);
#else
    static auto table = globalATenDispatch().getOpTable("aten::nonzero_numpy(Tensor self) -> Tensor[]");
    return table->getOp<std::vector<Tensor> (const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this);
#endif
}
inline Tensor Tensor::gather(int64_t dim, const Tensor & index, bool sparse_grad) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::gather(*this, dim, index, sparse_grad);
            break;
        default:
            AT_ERROR("gather not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::gather(Tensor self, int dim, Tensor index, *, bool sparse_grad=False) -> Tensor");
    return table->getOp<Tensor (const Tensor &, int64_t, const Tensor &, bool)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, dim, index, sparse_grad);
#endif
}
inline Tensor Tensor::addcmul(const Tensor & tensor1, const Tensor & tensor2, Scalar value) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::addcmul(*this, tensor1, tensor2, value);
            break;
        default:
            AT_ERROR("addcmul not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::addcmul(Tensor self, Tensor tensor1, Tensor tensor2, *, Scalar value=1) -> Tensor");
    return table->getOp<Tensor (const Tensor &, const Tensor &, const Tensor &, Scalar)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, tensor1, tensor2, value);
#endif
}
inline Tensor Tensor::addcdiv(const Tensor & tensor1, const Tensor & tensor2, Scalar value) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::addcdiv(*this, tensor1, tensor2, value);
            break;
        default:
            AT_ERROR("addcdiv not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::addcdiv(Tensor self, Tensor tensor1, Tensor tensor2, *, Scalar value=1) -> Tensor");
    return table->getOp<Tensor (const Tensor &, const Tensor &, const Tensor &, Scalar)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, tensor1, tensor2, value);
#endif
}
inline std::tuple<Tensor,Tensor> Tensor::gels(const Tensor & A) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::gels(*this, A);
            break;
        default:
            AT_ERROR("gels not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::gels(Tensor self, Tensor A) -> (Tensor solution, Tensor QR)");
    return table->getOp<std::tuple<Tensor,Tensor> (const Tensor &, const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, A);
#endif
}
inline std::tuple<Tensor,Tensor> Tensor::triangular_solve(const Tensor & A, bool upper, bool transpose, bool unitriangular) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::triangular_solve(*this, A, upper, transpose, unitriangular);
#else
    static auto table = globalATenDispatch().getOpTable("aten::triangular_solve(Tensor self, Tensor A, bool upper=True, bool transpose=False, bool unitriangular=False) -> (Tensor solution, Tensor cloned_coefficient)");
    return table->getOp<std::tuple<Tensor,Tensor> (const Tensor &, const Tensor &, bool, bool, bool)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, A, upper, transpose, unitriangular);
#endif
}
inline std::tuple<Tensor,Tensor> Tensor::symeig(bool eigenvectors, bool upper) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::symeig(*this, eigenvectors, upper);
#else
    static auto table = globalATenDispatch().getOpTable("aten::symeig(Tensor self, bool eigenvectors=False, bool upper=True) -> (Tensor eigenvalues, Tensor eigenvectors)");
    return table->getOp<std::tuple<Tensor,Tensor> (const Tensor &, bool, bool)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, eigenvectors, upper);
#endif
}
inline std::tuple<Tensor,Tensor> Tensor::eig(bool eigenvectors) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::eig(*this, eigenvectors);
            break;
        default:
            AT_ERROR("eig not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::eig(Tensor self, bool eigenvectors=False) -> (Tensor eigenvalues, Tensor eigenvectors)");
    return table->getOp<std::tuple<Tensor,Tensor> (const Tensor &, bool)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, eigenvectors);
#endif
}
inline std::tuple<Tensor,Tensor,Tensor> Tensor::svd(bool some, bool compute_uv) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::svd(*this, some, compute_uv);
            break;
        default:
            AT_ERROR("svd not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::svd(Tensor self, bool some=True, bool compute_uv=True) -> (Tensor U, Tensor S, Tensor V)");
    return table->getOp<std::tuple<Tensor,Tensor,Tensor> (const Tensor &, bool, bool)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, some, compute_uv);
#endif
}
inline Tensor Tensor::cholesky(bool upper) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::cholesky(*this, upper);
#else
    static auto table = globalATenDispatch().getOpTable("aten::cholesky(Tensor self, bool upper=False) -> Tensor");
    return table->getOp<Tensor (const Tensor &, bool)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, upper);
#endif
}
inline Tensor Tensor::cholesky_solve(const Tensor & input2, bool upper) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::cholesky_solve(*this, input2, upper);
#else
    static auto table = globalATenDispatch().getOpTable("aten::cholesky_solve(Tensor self, Tensor input2, bool upper=False) -> Tensor");
    return table->getOp<Tensor (const Tensor &, const Tensor &, bool)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, input2, upper);
#endif
}
inline std::tuple<Tensor,Tensor> Tensor::solve(const Tensor & A) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::solve(*this, A);
#else
    static auto table = globalATenDispatch().getOpTable("aten::solve(Tensor self, Tensor A) -> (Tensor solution, Tensor LU)");
    return table->getOp<std::tuple<Tensor,Tensor> (const Tensor &, const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, A);
#endif
}
inline Tensor Tensor::cholesky_inverse(bool upper) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::cholesky_inverse(*this, upper);
            break;
        default:
            AT_ERROR("cholesky_inverse not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::cholesky_inverse(Tensor self, bool upper=False) -> Tensor");
    return table->getOp<Tensor (const Tensor &, bool)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, upper);
#endif
}
inline std::tuple<Tensor,Tensor> Tensor::pstrf(bool upper, Scalar tol) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::pstrf(*this, upper, tol);
            break;
        default:
            AT_ERROR("pstrf not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::pstrf(Tensor self, bool upper=True, Scalar tol=-1) -> (Tensor u, Tensor pivot)");
    return table->getOp<std::tuple<Tensor,Tensor> (const Tensor &, bool, Scalar)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, upper, tol);
#endif
}
inline std::tuple<Tensor,Tensor> Tensor::qr(bool some) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::qr(*this, some);
#else
    static auto table = globalATenDispatch().getOpTable("aten::qr(Tensor self, bool some=True) -> (Tensor Q, Tensor R)");
    return table->getOp<std::tuple<Tensor,Tensor> (const Tensor &, bool)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, some);
#endif
}
inline std::tuple<Tensor,Tensor> Tensor::geqrf() const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::geqrf(*this);
            break;
        default:
            AT_ERROR("geqrf not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::geqrf(Tensor self) -> (Tensor a, Tensor tau)");
    return table->getOp<std::tuple<Tensor,Tensor> (const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this);
#endif
}
inline Tensor Tensor::orgqr(const Tensor & input2) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::orgqr(*this, input2);
            break;
        default:
            AT_ERROR("orgqr not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::orgqr(Tensor self, Tensor input2) -> Tensor");
    return table->getOp<Tensor (const Tensor &, const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, input2);
#endif
}
inline Tensor Tensor::ormqr(const Tensor & input2, const Tensor & input3, bool left, bool transpose) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::ormqr(*this, input2, input3, left, transpose);
            break;
        default:
            AT_ERROR("ormqr not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::ormqr(Tensor self, Tensor input2, Tensor input3, bool left=True, bool transpose=False) -> Tensor");
    return table->getOp<Tensor (const Tensor &, const Tensor &, const Tensor &, bool, bool)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, input2, input3, left, transpose);
#endif
}
inline Tensor Tensor::lu_solve(const Tensor & LU_data, const Tensor & LU_pivots) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::lu_solve(*this, LU_data, LU_pivots);
            break;
        default:
            AT_ERROR("lu_solve not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::lu_solve(Tensor self, Tensor LU_data, Tensor LU_pivots) -> Tensor");
    return table->getOp<Tensor (const Tensor &, const Tensor &, const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, LU_data, LU_pivots);
#endif
}
inline Tensor Tensor::multinomial(int64_t num_samples, bool replacement, Generator * generator) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::multinomial(*this, num_samples, replacement, generator);
            break;
        default:
            AT_ERROR("multinomial not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::multinomial(Tensor self, int num_samples, bool replacement=False, *, Generator? generator=None) -> Tensor");
    return table->getOp<Tensor (const Tensor &, int64_t, bool, Generator *)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, num_samples, replacement, generator);
#endif
}
inline Tensor Tensor::lgamma() const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::lgamma(*this);
            break;
        default:
            AT_ERROR("lgamma not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::lgamma(Tensor self) -> Tensor");
    return table->getOp<Tensor (const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this);
#endif
}
inline Tensor Tensor::digamma() const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::digamma(*this);
            break;
        default:
            AT_ERROR("digamma not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::digamma(Tensor self) -> Tensor");
    return table->getOp<Tensor (const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this);
#endif
}
inline Tensor Tensor::polygamma(int64_t n) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::polygamma(n, *this);
            break;
        default:
            AT_ERROR("polygamma not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::polygamma(int n, Tensor self) -> Tensor");
    return table->getOp<Tensor (int64_t, const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(n, *this);
#endif
}
inline Tensor Tensor::erfinv() const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::erfinv(*this);
            break;
        default:
            AT_ERROR("erfinv not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::erfinv(Tensor self) -> Tensor");
    return table->getOp<Tensor (const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this);
#endif
}
inline Tensor Tensor::dist(const Tensor & other, Scalar p) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::dist(*this, other, p);
            break;
        default:
            AT_ERROR("dist not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::dist(Tensor self, Tensor other, Scalar p=2) -> Tensor");
    return table->getOp<Tensor (const Tensor &, const Tensor &, Scalar)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, other, p);
#endif
}
inline Tensor Tensor::atan2(const Tensor & other) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::atan2(*this, other);
            break;
        default:
            AT_ERROR("atan2 not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::atan2(Tensor self, Tensor other) -> Tensor");
    return table->getOp<Tensor (const Tensor &, const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, other);
#endif
}
inline Tensor Tensor::lerp(const Tensor & end, Scalar weight) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::lerp(*this, end, weight);
            break;
        default:
            AT_ERROR("lerp not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::lerp(Tensor self, Tensor end, Scalar weight) -> Tensor");
    return table->getOp<Tensor (const Tensor &, const Tensor &, Scalar)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, end, weight);
#endif
}
inline Tensor Tensor::lerp(const Tensor & end, const Tensor & weight) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::lerp(*this, end, weight);
            break;
        default:
            AT_ERROR("lerp not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::lerp(Tensor self, Tensor end, Tensor weight) -> Tensor");
    return table->getOp<Tensor (const Tensor &, const Tensor &, const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, end, weight);
#endif
}
inline Tensor Tensor::histc(int64_t bins, Scalar min, Scalar max) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::histc(*this, bins, min, max);
            break;
        default:
            AT_ERROR("histc not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::histc(Tensor self, int bins=100, Scalar min=0, Scalar max=0) -> Tensor");
    return table->getOp<Tensor (const Tensor &, int64_t, Scalar, Scalar)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, bins, min, max);
#endif
}
inline Tensor Tensor::sign() const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::sign(*this);
            break;
        default:
            AT_ERROR("sign not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::sign(Tensor self) -> Tensor");
    return table->getOp<Tensor (const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this);
#endif
}
inline Tensor Tensor::fmod(Scalar other) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::fmod(*this, other);
            break;
        default:
            AT_ERROR("fmod not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::fmod(Tensor self, Scalar other) -> Tensor");
    return table->getOp<Tensor (const Tensor &, Scalar)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, other);
#endif
}
inline Tensor Tensor::fmod(const Tensor & other) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::fmod(*this, other);
            break;
        default:
            AT_ERROR("fmod not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::fmod(Tensor self, Tensor other) -> Tensor");
    return table->getOp<Tensor (const Tensor &, const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, other);
#endif
}
inline Tensor Tensor::remainder(Scalar other) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::remainder(*this, other);
            break;
        default:
            AT_ERROR("remainder not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::remainder(Tensor self, Scalar other) -> Tensor");
    return table->getOp<Tensor (const Tensor &, Scalar)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, other);
#endif
}
inline Tensor Tensor::remainder(const Tensor & other) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::remainder(*this, other);
            break;
        default:
            AT_ERROR("remainder not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::remainder(Tensor self, Tensor other) -> Tensor");
    return table->getOp<Tensor (const Tensor &, const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, other);
#endif
}
inline Tensor Tensor::min(const Tensor & other) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::min(*this, other);
            break;
        default:
            AT_ERROR("min not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::min(Tensor self, Tensor other) -> Tensor");
    return table->getOp<Tensor (const Tensor &, const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, other);
#endif
}
inline Tensor Tensor::min() const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::min(*this);
            break;
        case Backend::QuantizedCPU:
            return QuantizedCPUType::min(*this);
            break;
        default:
            AT_ERROR("min not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::min(Tensor self) -> Tensor");
    return table->getOp<Tensor (const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this);
#endif
}
inline Tensor Tensor::max(const Tensor & other) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::max(*this, other);
            break;
        default:
            AT_ERROR("max not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::max(Tensor self, Tensor other) -> Tensor");
    return table->getOp<Tensor (const Tensor &, const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, other);
#endif
}
inline Tensor Tensor::max() const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::max(*this);
            break;
        case Backend::QuantizedCPU:
            return QuantizedCPUType::max(*this);
            break;
        default:
            AT_ERROR("max not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::max(Tensor self) -> Tensor");
    return table->getOp<Tensor (const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this);
#endif
}
inline Tensor Tensor::median() const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::median(*this);
            break;
        default:
            AT_ERROR("median not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::median(Tensor self) -> Tensor");
    return table->getOp<Tensor (const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this);
#endif
}
inline std::tuple<Tensor,Tensor> Tensor::sort(int64_t dim, bool descending) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::sort(*this, dim, descending);
            break;
        case Backend::QuantizedCPU:
            return QuantizedCPUType::sort(*this, dim, descending);
            break;
        default:
            AT_ERROR("sort not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::sort(Tensor self, int dim=-1, bool descending=False) -> (Tensor values, Tensor indices)");
    return table->getOp<std::tuple<Tensor,Tensor> (const Tensor &, int64_t, bool)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, dim, descending);
#endif
}
inline Tensor Tensor::argsort(int64_t dim, bool descending) const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::argsort(*this, dim, descending);
#else
    static auto table = globalATenDispatch().getOpTable("aten::argsort(Tensor self, int dim=-1, bool descending=False) -> Tensor");
    return table->getOp<Tensor (const Tensor &, int64_t, bool)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, dim, descending);
#endif
}
inline std::tuple<Tensor,Tensor> Tensor::topk(int64_t k, int64_t dim, bool largest, bool sorted) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::topk(*this, k, dim, largest, sorted);
            break;
        default:
            AT_ERROR("topk not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::topk(Tensor self, int k, int dim=-1, bool largest=True, bool sorted=True) -> (Tensor values, Tensor indices)");
    return table->getOp<std::tuple<Tensor,Tensor> (const Tensor &, int64_t, int64_t, bool, bool)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, k, dim, largest, sorted);
#endif
}
inline Tensor Tensor::all() const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::all(*this);
#else
    static auto table = globalATenDispatch().getOpTable("aten::all(Tensor self) -> Tensor");
    return table->getOp<Tensor (const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this);
#endif
}
inline Tensor Tensor::any() const {
#ifdef USE_STATIC_DISPATCH
    return TypeDefault::any(*this);
#else
    static auto table = globalATenDispatch().getOpTable("aten::any(Tensor self) -> Tensor");
    return table->getOp<Tensor (const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this);
#endif
}
inline Tensor Tensor::renorm(Scalar p, int64_t dim, Scalar maxnorm) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::renorm(*this, p, dim, maxnorm);
            break;
        default:
            AT_ERROR("renorm not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::renorm(Tensor self, Scalar p, int dim, Scalar maxnorm) -> Tensor");
    return table->getOp<Tensor (const Tensor &, Scalar, int64_t, Scalar)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, p, dim, maxnorm);
#endif
}
inline Tensor Tensor::unfold(int64_t dimension, int64_t size, int64_t step) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::unfold(*this, dimension, size, step);
            break;
        default:
            AT_ERROR("unfold not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::unfold(Tensor(a) self, int dimension, int size, int step) -> Tensor(a)");
    return table->getOp<Tensor (const Tensor &, int64_t, int64_t, int64_t)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, dimension, size, step);
#endif
}
inline bool Tensor::equal(const Tensor & other) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::equal(*this, other);
            break;
        default:
            AT_ERROR("equal not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::equal(Tensor self, Tensor other) -> bool");
    return table->getOp<bool (const Tensor &, const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, other);
#endif
}
inline Tensor Tensor::pow(const Tensor & exponent) const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::pow(*this, exponent);
            break;
        default:
            AT_ERROR("pow not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::pow(Tensor self, Tensor exponent) -> Tensor");
    return table->getOp<Tensor (const Tensor &, const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this, exponent);
#endif
}
inline Tensor Tensor::alias() const {
#ifdef USE_STATIC_DISPATCH
    switch(tensorTypeIdToBackend(type_id())) {
        case Backend::CPU:
            return CPUType::alias(*this);
            break;
        default:
            AT_ERROR("alias not implemented for ", at::toString(tensorTypeIdToBackend(type_id())));
    }
#else
    static auto table = globalATenDispatch().getOpTable("aten::alias(Tensor(a) self) -> Tensor(a)");
    return table->getOp<Tensor (const Tensor &)>(tensorTypeIdToBackend(type_id()), is_variable())(*this);
#endif
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

#ifdef NAMEDTENSOR_ENABLED
inline NamedTensorMeta* Tensor::get_named_tensor_meta() {
  return static_cast<NamedTensorMeta*>(impl_->named_tensor_meta());
}

inline const NamedTensorMeta* Tensor::get_named_tensor_meta() const {
  return static_cast<NamedTensorMeta*>(impl_->named_tensor_meta());
}

inline bool Tensor::is_named() const {
  return impl::internal_is_named(unsafeGetTensorImpl());
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

#define DEFINE_CAST(T, name, _)                  \
  template <>                                    \
  inline T* Tensor::data() const {               \
    TORCH_CHECK(                                    \
        scalar_type() == ScalarType::name,       \
        "expected scalar type ",                 \
        #name,                                   \
        " but found ",                           \
        c10::toString(scalar_type()));           \
    return static_cast<T*>(this->data_ptr());    \
  }

AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_EXCEPT_COMPLEX_HALF(DEFINE_CAST)
#undef DEFINE_CAST

#define DEFINE_ITEM(T, name, _)   \
  template <>                     \
  inline T Tensor::item() const { \
    return item().to##name();     \
  }

AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_EXCEPT_COMPLEX_HALF_AND_QINT(DEFINE_ITEM)
#undef DEFINE_ITEM

} //namespace at
