#pragma once

// yf225 TODO: should this actually be in ATen/core, or should it be in ATen/ ?

#include <c10/util/Exception.h>
#include <c10/util/Optional.h>
#include <ATen/ExpandUtils.h>
#include <ATen/NativeFunctions.h>
#include <ATen/DeviceGuard.h>
#include <ATen/core/TensorBody.h>

namespace at {

class CAFFE2_API IndexedTensor : public Tensor {
  // yf225 TODO: hmm do we need to handle other operators as well? (like `+=`?) Try it out in Python and see what the expected behavior is!
  /* yf225 TODO
  1. how would it work in Python if `b = a[advanced indexing with tensor], b = some scalar value`? and `b = a[advanced indexing with tensor], b.zero_()`? compared it to C++ API behavior!
  2. also how would a[adv indexing][adv indexing] = some scalar value work in Python and in C++? try it out in both languages
  */
 public:
  IndexedTensor(Tensor tensor, Tensor original_tensor, std::vector<TensorIndex> indices)
    : Tensor(std::move(tensor)), original_tensor_(std::move(original_tensor)), indices_(std::move(indices)) {}

  IndexedTensor(const IndexedTensor&) = default;
  IndexedTensor(IndexedTensor&&) = default;

  Tensor & operator=(IndexedTensor const & rhs) &&;
  Tensor & operator=(IndexedTensor && rhs) &&;
  Tensor & operator=(Tensor const & rhs) &&;
  Tensor & operator=(Tensor && rhs) &&;
  Tensor & operator=(Scalar v) &&;

  void clear_history() {
    if (indices_.capacity() > 0) {
      indices_ = std::vector<TensorIndex>();
      TORCH_INTERNAL_ASSERT(indices_.capacity() == 0);
    }
    if (original_tensor_.defined()) {
      original_tensor_ = {};
    }
  }

  const Tensor& original_tensor() const {
    return original_tensor_;
  }

  const std::vector<TensorIndex>& indices() const {
    return indices_;
  }

 private:
  Tensor original_tensor_;
  std::vector<TensorIndex> indices_;
};

namespace indexing {

enum class SliceIndexType { None, Integer };
enum class TensorIndexType { None, Ellipsis, Integer, Boolean, Slice, Tensor };

struct NoneIndexType { NoneIndexType() {} };
struct EllipsisIndexType { EllipsisIndexType() {} };

CAFFE2_API extern const NoneIndexType None;
CAFFE2_API extern const EllipsisIndexType Ellipsis;

struct CAFFE2_API SliceIndex {
  SliceIndex(at::indexing::NoneIndexType none) : type_(SliceIndexType::None) {}
  SliceIndex(int64_t integer) : integer_(integer), type_(SliceIndexType::Integer) {}

  bool is_none() const {
    return type_ == SliceIndexType::None;
  }

  bool is_integer() const {
    return type_ == SliceIndexType::Integer;
  }

  int64_t integer() const {
    return integer_;
  }

 private:
  int64_t integer_;
  SliceIndexType type_;
};

struct CAFFE2_API Slice {
 public:
  Slice() {}
  Slice(int64_t start, int64_t stop, int64_t step) : start_(start), stop_(stop), step_(step) {}

  const int64_t& start() const {
    return start_;
  }

  const int64_t& stop() const {
    return stop_;
  }

  const int64_t& step() const {
    return step_;
  }

 private:
  int64_t start_;
  int64_t stop_;
  int64_t step_;
};

inline std::ostream& operator<<(std::ostream& stream, const Slice& slice) {
  stream << "{" << slice.start() << ", " << slice.stop() << ", " << slice.step() << "}";
  return stream;
}

inline Slice unpackSlice(
    SliceIndex start_index = at::indexing::None,
    SliceIndex stop_index = at::indexing::None,
    SliceIndex step_index = at::indexing::None) {
  int64_t start, stop, step;
  const int64_t INT64_T_MAX = std::numeric_limits<int64_t>::max();
  const int64_t INT64_T_MIN = std::numeric_limits<int64_t>::min();
  if (step_index.is_none()) {
    step = 1;
  } else {
    step = step_index.integer();
    if (step == 0) {
      TORCH_CHECK(false, "slice step cannot be zero");
    }
    /* Here *step might be -INT64_T_MAX-1; in this case we replace it
     * with -INT64_T_MAX.  This doesn't affect the semantics, and it
     * guards against later undefined behaviour resulting from code that
     * does "step = -step" as part of a slice reversal.
     */
    if (step < -INT64_T_MAX)
      step = -INT64_T_MAX;
  }
  if (start_index.is_none()) {
    start = step < 0 ? INT64_T_MAX : 0;
  } else {
    start = start_index.integer();
  }
  if (stop_index.is_none()) {
    stop = step < 0 ? INT64_T_MIN : INT64_T_MAX;
  } else {
    stop = stop_index.integer();
  }
  return Slice(start, stop, step);
}

struct CAFFE2_API TensorIndex {
  TensorIndex(at::indexing::NoneIndexType) : type_(TensorIndexType::None) {}
  TensorIndex(at::indexing::EllipsisIndexType) : type_(TensorIndexType::Ellipsis) {}
  TensorIndex(const char *str) : TensorIndex(at::indexing::Ellipsis) {
    TORCH_CHECK(strcmp(str, "...") == 0, "yf225 TODO: err msg");  // yf225 TODO: add test for this
  }

  // yf225 TODO: can we macro generate this somehow?
  TensorIndex(uint8_t integer) : integer_(integer), type_(TensorIndexType::Integer) {}
  TensorIndex(int8_t integer) : integer_(integer), type_(TensorIndexType::Integer) {}
  TensorIndex(int16_t integer) : integer_(integer), type_(TensorIndexType::Integer) {}
  TensorIndex(int integer) : integer_(integer), type_(TensorIndexType::Integer) {}
  TensorIndex(int64_t integer) : integer_(integer), type_(TensorIndexType::Integer) {}

  TensorIndex(bool boolean) : boolean_(boolean), type_(TensorIndexType::Boolean) {}
  TensorIndex(std::initializer_list<SliceIndex> init_list) : type_(TensorIndexType::Slice) {
    if (init_list.size() == 0) {
      slice_ = unpackSlice();
    } else if (init_list.size() == 2) {
      slice_ = unpackSlice(*init_list.begin(), *(init_list.begin() + 1));
    } else if (init_list.size() == 3) {
      slice_ = unpackSlice(*init_list.begin(), *(init_list.begin() + 1), *(init_list.begin() + 2));
    } else {
      TORCH_CHECK(false, "yf225 TODO err msg: Wrong # of elements in braced-init-list, expect 0 / 2 / 3 elements"); // yf225 TODO: add test for this
    }
  }
  TensorIndex(Tensor tensor) : tensor_(tensor), type_(TensorIndexType::Tensor) {}

  bool is_none() const {
    return type_ == TensorIndexType::None;
  }

  bool is_ellipsis() const {
    return type_ == TensorIndexType::Ellipsis;
  }

  bool is_integer() const {
    return type_ == TensorIndexType::Integer;
  }

  int64_t integer() const {
    return integer_;
  }

  bool is_boolean() const {
    return type_ == TensorIndexType::Boolean;
  }

  bool boolean() const {
    return boolean_;
  }

  bool is_slice() const {
    return type_ == TensorIndexType::Slice;
  }

  const Slice& slice() const {
    return slice_;
  }

  bool is_tensor() const {
    return type_ == TensorIndexType::Tensor;
  }

  const Tensor& tensor() const {
    return tensor_;
  }

  int64_t integer_;
  bool boolean_;
  Slice slice_;
  Tensor tensor_;
  TensorIndexType type_;
};

inline std::ostream& operator<<(std::ostream& stream, const TensorIndex& tensor_index) {
  if (tensor_index.is_none()) {
    stream << "None";
  } else if (tensor_index.is_ellipsis()) {
    stream << "...";
  } else if (tensor_index.is_integer()) {
    stream << tensor_index.integer();
  } else if (tensor_index.is_boolean()) {
    stream << tensor_index.boolean();
  } else if (tensor_index.is_slice()) {
    stream << tensor_index.slice();
  } else if (tensor_index.is_tensor()) {
    stream << tensor_index.tensor();
  }
  return stream;
}

inline std::ostream& operator<<(std::ostream& stream, const std::vector<TensorIndex>& tensor_indices) {
  stream << "(";
  for (size_t i = 0; i < tensor_indices.size(); i++) {
    stream << tensor_indices[i];
    if (i < tensor_indices.size() - 1) stream << ", ";
  }
  stream << ")";
  return stream;
}

inline int64_t count_specified_dimensions(const std::vector<TensorIndex>& indices) {
  // Count the number of indexed dimensions (everything but ellipsis and None)
  int64_t count = 0;
  size_t size = indices.size();
  for (size_t i = 0; i < size; i++) {
    auto& obj = indices[i];
    if (obj.is_tensor()) {
      auto& tensor = obj.tensor();
      if (tensor.scalar_type() == kByte || tensor.scalar_type() == kBool) {
        count += tensor.dim();
      } else {
        count++;
      }
    } else if (!obj.is_none() && !obj.is_ellipsis() && !obj.is_boolean()) {
      count++;
    }
  }
  return count;
}

// To match numpy semantics:
// As a special case for backwards compatibility,
// strip away unit dimensions from the left of 'src'
inline IntArrayRef slicePrefix1sSize(IntArrayRef sizes) {
  size_t first_non1_src = sizes.size();
  for (size_t i = 0; i < sizes.size(); ++i) {
    if (sizes[i] != 1) {
      first_non1_src = i;
      break;
    }
  }

  return sizes.slice(first_non1_src);
}

inline Tensor boolToIndexingTensor(const Tensor& self, bool value) {
  // booleans add a dimension of size 1. true indexes this dimension as if 0:, false as empty.
  if (value) {
    return at::native::zeros({1}, {}, self.options().dtype(kLong));
  } else {
    return at::native::empty({0}, {}, self.options().dtype(kLong));
  }
}

inline void copy_to(Tensor dst, const Tensor& src) {
  Tensor b_src;
  IntArrayRef sliced_src_sizes = slicePrefix1sSize(src.sizes());
  std::tie(b_src) = expand_inplace(dst, src.view(sliced_src_sizes), "setitem");
  dst.copy_(b_src);
}

inline Tensor applySelect(const Tensor& self, int64_t dim, int64_t index, int64_t real_dim=0) {
  if (index == 0 && dim == 0 && self.dim() == 0) {
    TORCH_CHECK(false,
        "invalid index of a 0-dim tensor. ",
        "Use tensor.item() to convert a 0-dim tensor to a Python number");
  }
  int64_t size = self.size(dim);
  if (index < -size || index >= size) {
    TORCH_CHECK(false,
      "index ", index, " is out of bounds for dimension ", real_dim, " with size ", size);
  }
  // if the index is negative, do not normalize it because that would fix the index
  // on the current tensor size in the tracer.
  // aten::select also works on negative indices
  return self.select(dim, index);
}

inline Tensor applySlice(const Tensor& self, int64_t dim, const Slice& slice, bool ensure_view=false) {
  const auto& start = slice.start();
  const auto& stop = slice.stop();
  const auto& step = slice.step();
  const auto& length = self.size(dim);
  if (step == 0) {
    TORCH_CHECK(false, "step cannot be zero");
  }
  if (step < 0) {
    // TODO: implement negative step
    TORCH_CHECK(false, "negative step not yet supported");
  }
  if (!ensure_view && start == 0 && stop == length && step == 1) {
    return self;
  }
  return self.slice(dim, start, stop, step);
}

inline Tensor applySlicing(const Tensor& self, const std::vector<TensorIndex>& indices, std::vector<Tensor>& outIndices) {
  int64_t size = indices.size();
  int64_t dim = 0;
  int64_t specified_dims = count_specified_dimensions(indices);

  auto handle_tensor = [&](const Tensor& tensor) {
    // TODO: check scalarType
    outIndices.resize(dim + 1);
    outIndices[dim] = tensor;
    dim++;
  };

  if (specified_dims > self.dim()) {
    TORCH_CHECK(false, "too many indices for tensor of dimension ", (int)self.dim());
  }

  Tensor result = self;
  for (int64_t i = 0; i < size; i++) {
    auto& obj = indices[i];
    if (obj.is_integer()) {
      result = applySelect(result, dim, obj.integer(), i);
    } else if (obj.is_slice()) {
      result = applySlice(result, dim, obj.slice());
      dim++;
    } else if (obj.is_ellipsis()) {
      dim += self.dim() - specified_dims;
    } else if (obj.is_none()) {
      result = result.unsqueeze(dim);
      dim++;
    } else if (obj.is_boolean()) {
      result = result.unsqueeze(dim);
      handle_tensor(boolToIndexingTensor(result, obj.boolean()));
    } else if (obj.is_tensor()) {
      auto& tensor = obj.tensor();
      auto scalar_type = tensor.scalar_type();
      if (tensor.dim() == 0 && at::isIntegralType(scalar_type, /*includeBool=*/true)) {
        if (scalar_type != at::kByte && scalar_type != at::kBool) {
          result = applySelect(result, dim, tensor.item<int64_t>(), i);
        } else {
          result = result.unsqueeze(dim);
          if(scalar_type == at::kBool) {
            handle_tensor(boolToIndexingTensor(result, tensor.item<bool>() != 0));
          } else {
            handle_tensor(boolToIndexingTensor(result, tensor.item<uint8_t>() != 0));
          }
        }
      } else {
        handle_tensor(tensor);
      }
// yf225 TODO: we don't support passing std::vector as index at the moment
//  } else if (PySequence_Check(obj)) {
//    // TODO: Naughty naughty get out of jail free
//    // (Fixing this means I have to fix the call chain though :/)
//    handle_tensor(sequenceToVariable(legacyExtractTypeId(self), obj));
//  } else {  // yf225 TODO: at this point, we know that `obj` is not a valid index
//    auto index = THPObjectPtr(PyNumber_Index(obj));
//    if (!index) {
//      PyErr_Clear();
//      invalid_index(obj);
//    }
//    result = applySelect(result, dim, THPUtils_unpackLong(index), i);
    } else {
      TORCH_CHECK(false, "Invalid TensorIndex type");
    }
  }
  return result;
}

inline std::vector<Tensor> typeConvertIndices(const Tensor& self, const std::vector<Tensor>& indices) {
  std::vector<Tensor> converted_inds(indices.size());
  for (size_t i = 0; i < indices.size(); ++i) {
    const auto &ind = indices[i];
    if (ind.defined()) {
      converted_inds[i] = ind.to(ind.options().device(self.device()));
    } else {
      converted_inds[i] = indices[i];
    }
  }
  return converted_inds;
}

inline Tensor dispatch_index(const Tensor& self, const std::vector<Tensor>& indices) {
  std::vector<Tensor> converted_indices = typeConvertIndices(self, indices);
  OptionalDeviceGuard device_guard(device_of(self));
  return self.index(converted_indices);
}

inline Tensor dispatch_index_put_(Tensor& self, const std::vector<Tensor>& indices, const Tensor& value) {
  std::vector<Tensor> converted_indices = typeConvertIndices(self, indices);
  OptionalDeviceGuard device_guard(device_of(self));
  return self.index_put_(converted_indices, value);
}

inline Tensor get_item(const Tensor& self, const std::vector<TensorIndex>& indices) {
  OptionalDeviceGuard device_guard(device_of(self));

  // handle simple types: integers, slices, ellipsis
  if (indices.size() == 1) {
    const TensorIndex& index = indices[0];
    if (index.is_none()) {
      return self.unsqueeze(0);
    } else if (index.is_ellipsis()) {
      return self.alias();
    } else if (index.is_integer()) {
      return applySelect(self, 0, index.integer());
    } else if (index.is_slice()) {
      return applySlice(self, 0, index.slice(), true);
    }
  }

  std::vector<Tensor> tensorIndices;
  Tensor sliced = applySlicing(self, indices, tensorIndices);
  if (tensorIndices.empty()) {
    if (sliced.is_same(self)) {
      // ensure we return a shallow copy for things like x[...]
      sliced = sliced.alias();
    }
    return sliced;
  }

  // indexing by tensors ("advanced" indexing)
  return dispatch_index(sliced, tensorIndices);
}

inline void set_item(Tensor& self, const std::vector<TensorIndex>& indices, const Tensor& value) {
  OptionalDeviceGuard device_guard(device_of(self));

  // handle simple types: integers, slices, ellipsis, bool
  if (indices.size() == 1) {
    const TensorIndex& index = indices[0];
    if (index.is_boolean() && !index.boolean()) {
      // do nothing for false (technically we should check the size, but we don't have
      // real 0-sized shapes.
      return;
    } else if (index.is_ellipsis()) {
      copy_to(self, value);
      return;
    } else if (index.is_none() || (index.is_boolean() && index.boolean())) {
      copy_to(self.unsqueeze(0), value);
      return;
    } else if (index.is_integer()) {
      copy_to(applySelect(self, 0, index.integer()), value);
      return;
    } else if (index.is_slice()) {
      copy_to(applySlice(self, 0, index.slice()), value);
      return;
    }
  }

  std::vector<Tensor> tensorIndices;
  Tensor sliced = applySlicing(self, indices, tensorIndices);
  if (tensorIndices.empty()) {
    copy_to(sliced, value);
    return;
  }

  IntArrayRef slicedValueSizes = slicePrefix1sSize(value.sizes());
  Tensor valuesSliced;
  if (!value.sizes().equals(slicedValueSizes)) {
    valuesSliced = value.view(slicedValueSizes);
  } else {
    valuesSliced = value;
  }
  dispatch_index_put_(sliced, tensorIndices, valuesSliced);
  return;
}

inline Tensor valueToTensor(c10::TensorOptions options, Scalar v) {
  return at::native::scalar_tensor(v, options);
}

inline void set_item(Tensor& self, const std::vector<TensorIndex>& indices, Scalar v) {
  OptionalDeviceGuard device_guard(device_of(self));
  Tensor value;

  // TODO: This qint special case looks very suspicious...
  if (isQIntType(self.scalar_type())) {
    value = valueToTensor(device(kCPU).dtype(kFloat), v);
  } else {
    value = valueToTensor(self.options(), v);
  }

  return set_item(self, indices, value);
}

} // namespace indexing
} // namespace at

namespace at {

// yf225 TODO: in the copy / move assignment operator of IndexedTensor,
// we can force the indices field and the original_tensor field to be cleared after assignment, to save memory.
// See https://stackoverflow.com/questions/35514909/how-to-clear-vector-in-c-from-memory
// Essentially, after `auto b = a(...)`, we want `b` to not carry a populated `indices` field and `original_tensor` field
/* something like:
if (indices_.capacity() > 0) {
  indices_ = std::vector<TensorIndex>();
  TORCH_INTERNAL_ASSERT(indices_.capacity() == 0);
}
if (original_tensor_.defined()) {
  original_tensor_ = {};
}
*/
inline Tensor & IndexedTensor::operator=(IndexedTensor const & rhs) && {
  at::indexing::set_item(original_tensor_, indices_, rhs);
  this->clear_history();
  return *this;
}
inline Tensor & IndexedTensor::operator=(IndexedTensor && rhs) && {
  at::indexing::set_item(original_tensor_, indices_, rhs);
  this->clear_history();
  return *this;
}
inline Tensor & IndexedTensor::operator=(Tensor const & rhs) && {
  at::indexing::set_item(original_tensor_, indices_, rhs);
  this->clear_history();
  return *this;
}
inline Tensor & IndexedTensor::operator=(Tensor && rhs) && {
  at::indexing::set_item(original_tensor_, indices_, rhs);
  this->clear_history();
  return *this;
}
inline Tensor & IndexedTensor::operator=(Scalar v) && {
  at::indexing::set_item(original_tensor_, indices_, v);
  this->clear_history();
  return *this;
}

// yf225 TODO: move these to TensorOperators.h

// This means we can get the whole indices list into one `std::vector`, and can have logic very similar to
// `applySlicing` to handle everything in one function! :D
inline IndexedTensor Tensor::operator()(const TensorIndex& index_dim1) const {
  std::vector<TensorIndex> indices = {index_dim1};
  return {at::indexing::get_item(*this, indices), *this, indices};
}
inline IndexedTensor Tensor::operator()(const TensorIndex& index_dim1, const TensorIndex& index_dim2) const {
  std::vector<TensorIndex> indices = {index_dim1, index_dim2};
  return {at::indexing::get_item(*this, indices), *this, indices};
}
inline IndexedTensor Tensor::operator()(const TensorIndex& index_dim1, const TensorIndex& index_dim2, const TensorIndex& index_dim3) const {
  std::vector<TensorIndex> indices = {index_dim1, index_dim2, index_dim3};
  return {at::indexing::get_item(*this, indices), *this, indices};
}
inline IndexedTensor Tensor::operator()(const TensorIndex& index_dim1, const TensorIndex& index_dim2, const TensorIndex& index_dim3, const TensorIndex& index_dim4) const {
  std::vector<TensorIndex> indices = {index_dim1, index_dim2, index_dim3, index_dim4};
  return {at::indexing::get_item(*this, indices), *this, indices};
}
inline IndexedTensor Tensor::operator()(const TensorIndex& index_dim1, const TensorIndex& index_dim2, const TensorIndex& index_dim3, const TensorIndex& index_dim4, const TensorIndex& index_dim5) const {
  std::vector<TensorIndex> indices = {index_dim1, index_dim2, index_dim3, index_dim4, index_dim5};
  return {at::indexing::get_item(*this, indices), *this, indices};
}
inline IndexedTensor Tensor::operator()(const TensorIndex& index_dim1, const TensorIndex& index_dim2, const TensorIndex& index_dim3, const TensorIndex& index_dim4, const TensorIndex& index_dim5, const TensorIndex& index_dim6) const {
  std::vector<TensorIndex> indices = {index_dim1, index_dim2, index_dim3, index_dim4, index_dim5, index_dim6};
  return {at::indexing::get_item(*this, indices), *this, indices};
}
inline IndexedTensor Tensor::operator()(const TensorIndex& index_dim1, const TensorIndex& index_dim2, const TensorIndex& index_dim3, const TensorIndex& index_dim4, const TensorIndex& index_dim5, const TensorIndex& index_dim6, const TensorIndex& index_dim7) const {
  std::vector<TensorIndex> indices = {index_dim1, index_dim2, index_dim3, index_dim4, index_dim5, index_dim6, index_dim7};
  return {at::indexing::get_item(*this, indices), *this, indices};
}
inline IndexedTensor Tensor::operator()(const TensorIndex& index_dim1, const TensorIndex& index_dim2, const TensorIndex& index_dim3, const TensorIndex& index_dim4, const TensorIndex& index_dim5, const TensorIndex& index_dim6, const TensorIndex& index_dim7, const TensorIndex& index_dim8) const {
  std::vector<TensorIndex> indices = {index_dim1, index_dim2, index_dim3, index_dim4, index_dim5, index_dim6, index_dim7, index_dim8};
  return {at::indexing::get_item(*this, indices), *this, indices};
}
inline IndexedTensor Tensor::operator()(const TensorIndex& index_dim1, const TensorIndex& index_dim2, const TensorIndex& index_dim3, const TensorIndex& index_dim4, const TensorIndex& index_dim5, const TensorIndex& index_dim6, const TensorIndex& index_dim7, const TensorIndex& index_dim8, const TensorIndex& index_dim9) const {
  std::vector<TensorIndex> indices = {index_dim1, index_dim2, index_dim3, index_dim4, index_dim5, index_dim6, index_dim7, index_dim8, index_dim9};
  return {at::indexing::get_item(*this, indices), *this, indices};
}
inline IndexedTensor Tensor::operator()(const TensorIndex& index_dim1, const TensorIndex& index_dim2, const TensorIndex& index_dim3, const TensorIndex& index_dim4, const TensorIndex& index_dim5, const TensorIndex& index_dim6, const TensorIndex& index_dim7, const TensorIndex& index_dim8, const TensorIndex& index_dim9, const TensorIndex& index_dim10) const {
  std::vector<TensorIndex> indices = {index_dim1, index_dim2, index_dim3, index_dim4, index_dim5, index_dim6, index_dim7, index_dim8, index_dim9, index_dim10};
  return {at::indexing::get_item(*this, indices), *this, indices};
}

} // namespace at
