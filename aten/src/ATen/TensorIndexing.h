#pragma once

// yf225 TODO: should this actually be in ATen/core, or should it be in ATen/ ?

#include <c10/util/Exception.h>
#include <c10/util/Optional.h>
#include <ATen/NativeFunctions.h>
#include <ATen/DeviceGuard.h>
#include <ATen/core/TensorBody.h>

namespace at {
namespace indexing {

enum class SliceIndexType { None, Integer };
enum class TensorIndexType { None, Ellipsis, Integer, Boolean, Slice, Tensor };

using NoneIndexType = c10::nullopt_t;
TORCH_API extern const NoneIndexType None;

struct TORCH_API SliceIndex {
  SliceIndex(at::indexing::NoneIndexType none) : type_(SliceIndexType::None) {}
  SliceIndex(int64_t integer) : integer_(integer), type_(SliceIndexType::Integer) {}

  const SliceIndexType& type() const {
    return type_;
  }

  int64_t integer() const {
    return integer_;
  }

 private:
  int64_t integer_;
  SliceIndexType type_;
};

struct TORCH_API Slice {
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

inline Slice unpackSlice(
    SliceIndex start_index = at::indexing::None,
    SliceIndex stop_index = at::indexing::None,
    SliceIndex step_index = at::indexing::None) {
  int64_t start, stop, step;
  const int64_t INT64_T_MAX = std::numeric_limits<int64_t>::max();
  const int64_t INT64_T_MIN = std::numeric_limits<int64_t>::min();
  if (step_index.type() == SliceIndexType::None) {
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
  if (start_index.type() == SliceIndexType::None) {
    start = step < 0 ? INT64_T_MAX : 0;
  } else {
    start = start_index.integer();
  }
  if (stop_index.type() == SliceIndexType::None) {
    stop = step < 0 ? INT64_T_MIN : INT64_T_MAX;
  } else {
    stop = stop_index.integer();
  }
  return Slice(start, stop, step);
}

struct TORCH_API TensorIndex {
  TensorIndex(at::indexing::NoneIndexType) : type_(TensorIndexType::None) {}
  TensorIndex(const char *str) : type_(TensorIndexType::Ellipsis) {
    TORCH_CHECK(strcmp(str, "...") == 0, "yf225 TODO: err msg");
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
      TORCH_CHECK(false, "yf225 TODO err msg: Wrong # of elements in braced-init-list, expect 0 / 2 / 3 elements");
    }
  }
  TensorIndex(Tensor tensor) : tensor_(tensor), type_(TensorIndexType::Tensor) {}

  const TensorIndexType& type() const {
    return type_;
  }

  int64_t integer() const {
    return integer_;
  }

  bool boolean() const {
    return boolean_;
  }

  const Slice& slice() const {
    return slice_;
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

inline int64_t count_specified_dimensions(std::vector<TensorIndex> indices) {
  // Count the number of indexed dimensions (everything but ellipsis and None)
  int64_t count = 0;
  size_t size = indices.size();
  for (size_t i = 0; i < size; i++) {
    auto& obj = indices[i];
    if (obj.type() == TensorIndexType::Tensor) {
      auto& tensor = obj.tensor();
      if (tensor.scalar_type() == kByte || tensor.scalar_type() == kBool) {
        count += tensor.dim();
      } else {
        count++;
      }
    } else if (obj.type() != TensorIndexType::None && obj.type() != TensorIndexType::Ellipsis && obj.type() != TensorIndexType::Boolean) {
      count++;
    }
  }
  return count;
}

inline Tensor boolToIndexingTensor(const Tensor& self, bool value) {
  // booleans add a dimension of size 1. true indexes this dimension as if 0:, false as empty.
  if (value) {
    return at::native::zeros({1}, {}, self.options().dtype(kLong));
  } else {
    return at::native::empty({0}, {}, self.options().dtype(kLong));
  }
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
      "index %lld is out of bounds for dimension %lld with size %lld",
      index, real_dim, size);
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

inline Tensor applySlicing(const Tensor& self, std::vector<TensorIndex> indices, std::vector<Tensor>& outIndices) {
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
    TORCH_CHECK(false, "too many indices for tensor of dimension %d", (int)self.dim());
  }

  Tensor result = self;
  for (int64_t i = 0; i < size; i++) {
    auto& obj = indices[i];
    if (obj.type() == TensorIndexType::Integer) {
      result = applySelect(result, dim, obj.integer(), i);
    } else if (obj.type() == TensorIndexType::Slice) {
      result = applySlice(result, dim, obj.slice());
      dim++;
    } else if (obj.type() == TensorIndexType::Ellipsis) {
      dim += self.dim() - specified_dims;
    } else if (obj.type() == TensorIndexType::None) {
      result = result.unsqueeze(dim);
      dim++;
    } else if (obj.type() == TensorIndexType::Boolean) {
      result = result.unsqueeze(dim);
      handle_tensor(boolToIndexingTensor(result, obj.boolean()));
    } else if (obj.type() == TensorIndexType::Tensor) {
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

inline Tensor getitem(const Tensor& self, std::vector<TensorIndex> indices) {
  OptionalDeviceGuard device_guard(device_of(self));

  // handle simple types: integers, slices, ellipsis
  if (indices.size() == 1) {
    const TensorIndex& index = indices[0];
    if (index.type() == TensorIndexType::None) {
      return self.unsqueeze(0);
    } else if (index.type() == TensorIndexType::Ellipsis) {
      return at::native::alias(self);
    } else if (index.type() == TensorIndexType::Integer) {
      return applySelect(self, 0, index.integer());
    } else if (index.type() == TensorIndexType::Slice) {
      return applySlice(self, 0, index.slice(), true);
    }
  }

  std::vector<Tensor> tensorIndices;
  Tensor sliced = applySlicing(self, indices, tensorIndices);
  if (tensorIndices.empty()) {
    if (sliced.is_same(self)) {
      // ensure we return a shallow copy for things like x[...]
      sliced = at::native::alias(sliced);
    }
    return sliced;
  }

  // indexing by tensors ("advanced" indexing)
  return dispatch_index(sliced, tensorIndices);
}

} // namespace indexing
} // namespace at

namespace at {

// yf225 TODO: move these to TensorOperators.h

// This means we can get the whole indices list into one `std::vector`, and can have logic very similar to
// `applySlicing` to handle everything in one function! :D
inline Tensor Tensor::operator()(const TensorIndex& index_dim1) const {
  return at::indexing::getitem(*this, {index_dim1});
}
inline Tensor Tensor::operator()(const TensorIndex& index_dim1, const TensorIndex& index_dim2) const{
  return at::indexing::getitem(*this, {index_dim1, index_dim2});
}
inline Tensor Tensor::operator()(const TensorIndex& index_dim1, const TensorIndex& index_dim2, const TensorIndex& index_dim3) const{
  return at::indexing::getitem(*this, {index_dim1, index_dim2, index_dim3});
}
inline Tensor Tensor::operator()(const TensorIndex& index_dim1, const TensorIndex& index_dim2, const TensorIndex& index_dim3, const TensorIndex& index_dim4) const{
  return at::indexing::getitem(*this, {index_dim1, index_dim2, index_dim3, index_dim4});
}
inline Tensor Tensor::operator()(const TensorIndex& index_dim1, const TensorIndex& index_dim2, const TensorIndex& index_dim3, const TensorIndex& index_dim4, const TensorIndex& index_dim5) const{
  return at::indexing::getitem(*this, {index_dim1, index_dim2, index_dim3, index_dim4, index_dim5});
}
inline Tensor Tensor::operator()(const TensorIndex& index_dim1, const TensorIndex& index_dim2, const TensorIndex& index_dim3, const TensorIndex& index_dim4, const TensorIndex& index_dim5, const TensorIndex& index_dim6) const{
  return at::indexing::getitem(*this, {index_dim1, index_dim2, index_dim3, index_dim4, index_dim5, index_dim6});
}
inline Tensor Tensor::operator()(const TensorIndex& index_dim1, const TensorIndex& index_dim2, const TensorIndex& index_dim3, const TensorIndex& index_dim4, const TensorIndex& index_dim5, const TensorIndex& index_dim6, const TensorIndex& index_dim7) const{
  return at::indexing::getitem(*this, {index_dim1, index_dim2, index_dim3, index_dim4, index_dim5, index_dim6, index_dim7});
}
inline Tensor Tensor::operator()(const TensorIndex& index_dim1, const TensorIndex& index_dim2, const TensorIndex& index_dim3, const TensorIndex& index_dim4, const TensorIndex& index_dim5, const TensorIndex& index_dim6, const TensorIndex& index_dim7, const TensorIndex& index_dim8) const{
  return at::indexing::getitem(*this, {index_dim1, index_dim2, index_dim3, index_dim4, index_dim5, index_dim6, index_dim7, index_dim8});
}
inline Tensor Tensor::operator()(const TensorIndex& index_dim1, const TensorIndex& index_dim2, const TensorIndex& index_dim3, const TensorIndex& index_dim4, const TensorIndex& index_dim5, const TensorIndex& index_dim6, const TensorIndex& index_dim7, const TensorIndex& index_dim8, const TensorIndex& index_dim9) const{
  return at::indexing::getitem(*this, {index_dim1, index_dim2, index_dim3, index_dim4, index_dim5, index_dim6, index_dim7, index_dim8, index_dim9});
}
inline Tensor Tensor::operator()(const TensorIndex& index_dim1, const TensorIndex& index_dim2, const TensorIndex& index_dim3, const TensorIndex& index_dim4, const TensorIndex& index_dim5, const TensorIndex& index_dim6, const TensorIndex& index_dim7, const TensorIndex& index_dim8, const TensorIndex& index_dim9, const TensorIndex& index_dim10) const{
  return at::indexing::getitem(*this, {index_dim1, index_dim2, index_dim3, index_dim4, index_dim5, index_dim6, index_dim7, index_dim8, index_dim9, index_dim10});
}

} // namespace at
