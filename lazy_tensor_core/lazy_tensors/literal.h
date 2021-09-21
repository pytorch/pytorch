#pragma once

#include <ATen/TensorIndexing.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/TensorFactories.h>

#include <string>

#include "lazy_tensors/shape.h"
#include "lazy_tensors/shape_util.h"
#include "lazy_tensors/span.h"
#include "lazy_tensors/primitive_util.h"

namespace lazy_tensors {

class Literal {
 public:
  Literal() : shape_(ShapeUtil::MakeTupleShape({})) {}

  explicit Literal(const Shape& shape);

  const Shape& shape() const;

  template <typename NativeT>
  lazy_tensors::Span<const NativeT> data(
      const ShapeIndex& shape_index = {}) const {
    LTC_CHECK(shape_index.empty()) << "Sub-literals not supported yet";
    return MakeConstSpan(static_cast<const NativeT*>(value_.data_ptr()),
                         value_.numel());
  }

  template <typename NativeT>
  lazy_tensors::Span<NativeT> data(const ShapeIndex& shape_index = {}) {
    LTC_CHECK(shape_index.empty()) << "Sub-literals not supported yet";
    return lazy_tensors::Span<NativeT>(static_cast<NativeT*>(value_.data_ptr()),
                                       value_.numel());
  }

  const void* untyped_data(const ShapeIndex& shape_index = {}) const {
    return value_.data_ptr();
  }

  void* untyped_data(const ShapeIndex& shape_index = {}) {
    LTC_LOG(FATAL) << "Not implemented yet.";
  }
  int64 size_bytes(const ShapeIndex& shape_index = {}) const {
    return value_.numel() * value_.element_size();
  }

  std::string ToStringWithoutShape() const {
    LTC_LOG(FATAL) << "Not implemented yet.";
  }

  size_t Hash() const;

  Literal Clone() const { LTC_LOG(FATAL) << "Not implemented yet."; }

  template <typename NativeT>
  void Set(lazy_tensors::Span<const int64> multi_index, NativeT value) {
    if (multi_index.empty()) {
      value_.fill_(value);
      return;
    }
    auto options = at::TensorOptions().device(at::kCPU).dtype(at::kLong);
    const auto index_tensor = at::tensor(
        std::vector<int64_t>(multi_index.begin(), multi_index.end()), options);
    value_.index_put_({at::indexing::TensorIndex(index_tensor)}, value);
  }

  template <typename NativeT>
  void PopulateR1(lazy_tensors::Span<const NativeT> values);

  const at::Tensor& value() const { return value_; }

 private:
  at::Tensor value_;
  Shape shape_;
};

template <>
inline void Literal::Set<lazy_tensors::uint32>(
    lazy_tensors::Span<const int64> multi_index, lazy_tensors::uint32 value) {
  Set<int64_t>(multi_index, static_cast<int64_t>(value));
}

template <>
inline void Literal::Set<lazy_tensors::uint64>(
    lazy_tensors::Span<const int64> multi_index, lazy_tensors::uint64 value) {
  LTC_LOG(FATAL) << "Not implemented yet.";
}

template <>
inline void Literal::Set<lazy_tensors::bfloat16>(
    lazy_tensors::Span<const int64> multi_index, lazy_tensors::bfloat16 value) {
  LTC_LOG(FATAL) << "Not implemented yet.";
}

template <>
inline void Literal::Set<lazy_tensors::half>(
    lazy_tensors::Span<const int64> multi_index, lazy_tensors::half value) {
  LTC_LOG(FATAL) << "Not implemented yet.";
}

template <>
inline void Literal::Set<lazy_tensors::complex64>(
    lazy_tensors::Span<const int64> multi_index,
    lazy_tensors::complex64 value) {
  LTC_LOG(FATAL) << "Not implemented yet.";
}

template <>
inline void Literal::Set<lazy_tensors::complex128>(
    lazy_tensors::Span<const int64> multi_index,
    lazy_tensors::complex128 value) {
  LTC_LOG(FATAL) << "Not implemented yet.";
}

template <typename NativeT>
inline void Literal::PopulateR1(lazy_tensors::Span<const NativeT> values) {
  LTC_CHECK(shape().IsArray());
  LTC_CHECK_EQ(shape().rank(), 1);
  LTC_CHECK_EQ(ShapeUtil::ElementsIn(shape()), values.size());
  LTC_CHECK_EQ(shape().element_type(),
               primitive_util::NativeToPrimitiveType<NativeT>());
  auto data_span = data<NativeT>();
  std::copy(values.begin(), values.end(), data_span.begin());
}

class LiteralSlice {
 public:
  LiteralSlice(const Literal& literal) : literal_(&literal) {}

  const Literal* literal() const { return literal_; }

 private:
  const Literal* literal_;
};

}  // namespace lazy_tensors
