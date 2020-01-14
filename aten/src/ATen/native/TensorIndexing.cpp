#include <ATen/ExpandUtils.h>
#include <ATen/NativeFunctions.h>
#include <ATen/DeviceGuard.h>

#include <ATen/native/TensorIndexing.h>

#include <c10/util/Exception.h>
#include <torch/csrc/jit/tracer.h>

namespace at {
namespace indexing {

const EllipsisIndexType Ellipsis = EllipsisIndexType();

Slice::Slice() {}
Slice::Slice(
    int64_t start,
    int64_t stop,
    int64_t step,
    Tensor start_tensor,
    Tensor stop_tensor,
    Tensor step_tensor)
  : start_(start),
    stop_(stop),
    step_(step),
    start_tensor_(start_tensor),
    stop_tensor_(stop_tensor),
    step_tensor_(step_tensor) {}

int64_t Slice::start() const {
  return start_;
}

int64_t Slice::stop() const {
  return stop_;
}

int64_t Slice::step() const {
  return step_;
}

const Tensor& Slice::start_tensor() const {
  return start_tensor_;
}

const Tensor& Slice::stop_tensor() const {
  return stop_tensor_;
}

const Tensor& Slice::step_tensor() const {
  return step_tensor_;
}

std::ostream& operator<<(std::ostream& stream, const Slice& slice) {
  stream << slice.start() << ":" << slice.stop() << ":" << slice.step();
  return stream;
}

// This mirrors `__PySlice_Unpack` in torch/csrc/utils/python_compat.h
Slice unpackSlice(
    c10::optional<int64_t> start_index,
    c10::optional<int64_t> stop_index,
    c10::optional<int64_t> step_index,
    Tensor start_index_tensor,
    Tensor stop_index_tensor,
    Tensor step_index_tensor) {
  int64_t start, stop, step;
  if (!step_index.has_value()) {
    step = 1;
  } else {
    step = step_index.value();
    if (step == 0) {
      TORCH_CHECK(false, "slice step cannot be zero");
    }
    // Here step might be -INDEX_MAX-1; in this case we replace it
    // with -INDEX_MAX.  This doesn't affect the semantics, and it
    // guards against later undefined behaviour resulting from code that
    // does "step = -step" as part of a slice reversal.
    if (step < -INDEX_MAX)
      step = -INDEX_MAX;
  }
  if (!start_index.has_value()) {
    start = step < 0 ? INDEX_MAX : 0;
  } else {
    start = start_index.value();
  }
  if (!stop_index.has_value()) {
    stop = step < 0 ? INDEX_MIN : INDEX_MAX;
  } else {
    stop = stop_index.value();
  }
  return Slice(start, stop, step, start_index_tensor, stop_index_tensor, step_index_tensor);
}

TensorIndex::TensorIndex(c10::nullopt_t) : type_(TensorIndexType::None) {}
TensorIndex::TensorIndex(at::indexing::EllipsisIndexType) : type_(TensorIndexType::Ellipsis) {}
TensorIndex::TensorIndex(const char *str) : TensorIndex(at::indexing::Ellipsis) {
  TORCH_CHECK(
    strcmp(str, "...") == 0,
    "Expected \"...\" to represent an ellipsis index, but got \"", str, "\"");
}
TensorIndex::TensorIndex(int64_t integer, Tensor tensor) : integer_(integer), tensor_(tensor), type_(TensorIndexType::Integer) {}
TensorIndex::TensorIndex(int integer) : TensorIndex((int64_t)integer) {}
TensorIndex::TensorIndex(
    std::initializer_list<c10::optional<int64_t>> slice,
    std::initializer_list<Tensor> slice_tensors)
    : type_(TensorIndexType::Slice) {
  if (slice.size() == 0) {
    slice_ = unpackSlice(None, None, None, {}, {}, {});
  } else if (slice.size() == 2) {
    slice_ = unpackSlice(
      *slice.begin(),
      *(slice.begin() + 1),
      None,
      *slice_tensors.begin(),
      *(slice_tensors.begin() + 1),
      {});
  } else if (slice.size() == 3) {
    slice_ = unpackSlice(
      *slice.begin(),
      *(slice.begin() + 1),
      *(slice.begin() + 2),
      *slice_tensors.begin(),
      *(slice_tensors.begin() + 1),
      *(slice_tensors.begin() + 2));
  } else {
    TORCH_CHECK(
      false,
      "Expected 0 / 2 / 3 elements in the braced-init-list to represent a slice index, but got ",
      slice.size(),
      " element(s)");
  }
}
TensorIndex::TensorIndex(Tensor tensor) : tensor_(tensor), type_(TensorIndexType::Tensor) {}

bool TensorIndex::is_none() const {
  return type_ == TensorIndexType::None;
}

bool TensorIndex::is_ellipsis() const {
  return type_ == TensorIndexType::Ellipsis;
}

bool TensorIndex::is_integer() const {
  return type_ == TensorIndexType::Integer;
}

int64_t TensorIndex::integer() const {
  return integer_;
}

bool TensorIndex::is_boolean() const {
  return type_ == TensorIndexType::Boolean;
}

bool TensorIndex::boolean() const {
  return boolean_;
}

bool TensorIndex::is_slice() const {
  return type_ == TensorIndexType::Slice;
}

const Slice& TensorIndex::slice() const {
  return slice_;
}

bool TensorIndex::is_tensor() const {
  return type_ == TensorIndexType::Tensor;
}

const Tensor& TensorIndex::tensor() const {
  return tensor_;
}

std::ostream& operator<<(std::ostream& stream, const TensorIndex& tensor_index) {
  if (tensor_index.is_none()) {
    stream << "None";
  } else if (tensor_index.is_ellipsis()) {
    stream << "...";
  } else if (tensor_index.is_integer()) {
    stream << tensor_index.integer();
  } else if (tensor_index.is_boolean()) {
    stream << std::boolalpha << tensor_index.boolean();
  } else if (tensor_index.is_slice()) {
    stream << tensor_index.slice();
  } else if (tensor_index.is_tensor()) {
    stream << tensor_index.tensor();
  }
  return stream;
}

std::ostream& operator<<(std::ostream& stream, const std::vector<TensorIndex>& tensor_indices) {
  stream << "(";
  for (size_t i = 0; i < tensor_indices.size(); i++) {
    stream << tensor_indices[i];
    if (i < tensor_indices.size() - 1) stream << ", ";
  }
  stream << ")";
  return stream;
}

// This mirrors `count_specified_dimensions` in torch/csrc/autograd/python_variable_indexing.cpp
int64_t count_specified_dimensions(ArrayRef<TensorIndex> indices) {
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

// This mirrors `applySlice` in torch/csrc/autograd/python_variable_indexing.cpp
Tensor applySlice(const Tensor& self, int64_t dim, const Slice& slice, bool ensure_view=false) {
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

  if (jit::tracer::isTracing() && slice.start_tensor().defined()) {
    auto& var = slice.start_tensor();
    jit::tracer::ArgumentStash::stashValue(std::string("start"), 1, var, jit::IntType::get());
  }
  if (jit::tracer::isTracing() && slice.stop_tensor().defined()) {
    auto& var = slice.stop_tensor();
    jit::tracer::ArgumentStash::stashValue(std::string("end"), 1, var, jit::IntType::get());
  }
  if (jit::tracer::isTracing() && slice.step_tensor().defined()) {
    auto& var = slice.step_tensor();
    jit::tracer::ArgumentStash::stashValue(std::string("step"), 1, var, jit::IntType::get());
  }

  // Skip this optimization if we are tracing, as the trace may be polymorphic
  // over the shape of the `self` tensor, and we still want to record
  // the slice.
  if (!ensure_view && start == 0 && stop == length && step == 1 && !jit::tracer::isTracing()) {
    return self;
  }
  return self.slice(dim, start, stop, step);
}

// This mirrors `applySelect` in torch/csrc/autograd/python_variable_indexing.cpp
Tensor applySelect(const Tensor& self, int64_t dim, const TensorIndex& index, int64_t real_dim=0) {
  TORCH_INTERNAL_CHECK(index.is_integer());

  if (jit::tracer::isTracing() && index.tensor().defined()) {
    auto& var = index.tensor();
    jit::tracer::ArgumentStash::stashValue(std::string("index"), 1, var, jit::IntType::get());
  }

  int64_t unpacked_index = index.integer();
  if (unpacked_index == 0 && dim == 0 && self.dim() == 0) {
    TORCH_CHECK(false,
        "invalid index of a 0-dim tensor. ",
        "Use tensor.item() to convert a 0-dim tensor to a number");
  }
  int64_t size = self.size(dim);
  if (unpacked_index < -size || unpacked_index >= size) {
    TORCH_CHECK(false,
      "index ", unpacked_index, " is out of bounds for dimension ", real_dim, " with size ", size);
  }
  // if the index is negative, do not normalize it because that would fix the index
  // on the current tensor size in the tracer.
  // aten::select also works on negative indices
  return self.select(dim, unpacked_index);
}

// This mirrors `valueToTensor` in torch/csrc/autograd/python_variable_indexing.cpp
Tensor valueToTensor(c10::TensorOptions options, Scalar v) {
  return at::native::scalar_tensor(v, options);
}

// This mirrors `boolToIndexingTensor` in torch/csrc/autograd/python_variable_indexing.cpp
Tensor boolToIndexingTensor(const Tensor& self, bool value) {
  // booleans add a dimension of size 1. true indexes this dimension as if 0:, false as empty.
  if (value) {
    return at::native::zeros({1}, {}, self.options().dtype(kLong));
  } else {
    return at::native::empty({0}, {}, self.options().dtype(kLong));
  }
}

// This mirrors `applySlicing` in torch/csrc/autograd/python_variable_indexing.cpp
Tensor applySlicing(const Tensor& self, ArrayRef<TensorIndex> indices, std::vector<Tensor>& outIndices) {
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
    } else {
      TORCH_CHECK(false, "Invalid TensorIndex type");
    }
  }
  return result;
}

// This mirrors `typeConvertIndices` in torch/csrc/autograd/python_variable_indexing.cpp
std::vector<Tensor> typeConvertIndices(const Tensor& self, const std::vector<Tensor>& indices) {
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

// This mirrors `dispatch_index` in torch/csrc/autograd/python_variable_indexing.cpp
Tensor dispatch_index(const Tensor& self, const std::vector<Tensor>& indices) {
  std::vector<Tensor> converted_indices = typeConvertIndices(self, indices);
  OptionalDeviceGuard device_guard(device_of(self));
  return self.index(converted_indices);
}

// This mirrors `dispatch_index_put_` in torch/csrc/autograd/python_variable_indexing.cpp
Tensor dispatch_index_put_(Tensor& self, const std::vector<Tensor>& indices, const Tensor& value) {
  std::vector<Tensor> converted_indices = typeConvertIndices(self, indices);
  OptionalDeviceGuard device_guard(device_of(self));
  return self.index_put_(converted_indices, value);
}

// This mirrors `THPVariable_getitem` in torch/csrc/autograd/python_variable_indexing.cpp
Tensor get_item(const Tensor& self, ArrayRef<TensorIndex> indices) {
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

// This mirrors `slicePrefix1sSize` in torch/csrc/autograd/python_variable_indexing.cpp
//
// To match numpy semantics:
// As a special case for backwards compatibility,
// strip away unit dimensions from the left of 'src'
IntArrayRef slicePrefix1sSize(IntArrayRef sizes) {
  size_t first_non1_src = sizes.size();
  for (size_t i = 0; i < sizes.size(); ++i) {
    if (sizes[i] != 1) {
      first_non1_src = i;
      break;
    }
  }

  return sizes.slice(first_non1_src);
}

// This mirrors `copy_to` in torch/csrc/autograd/python_variable_indexing.cpp
void copy_to(Tensor dst, const Tensor& src) {
  Tensor b_src;
  IntArrayRef sliced_src_sizes = slicePrefix1sSize(src.sizes());
  std::tie(b_src) = expand_inplace(dst, src.view(sliced_src_sizes), "setitem");
  dst.copy_(b_src);
}

// This mirrors `THPVariable_setitem` in torch/csrc/autograd/python_variable_indexing.cpp
// for "the assigned value is a Tensor" case
void set_item(Tensor& self, ArrayRef<TensorIndex> indices, const Tensor& value) {
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

// This mirrors `set_item` in torch/csrc/autograd/python_variable_indexing.cpp
// for "the assigned value is a Scalar" case
void set_item(Tensor& self, ArrayRef<TensorIndex> indices, Scalar v) {
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

Tensor Tensor::index(ArrayRef<TensorIndex> indices) const {
  return at::indexing::get_item(*this, indices);
}
Tensor Tensor::index(std::initializer_list<TensorIndex> indices) const {
  return index(ArrayRef<TensorIndex>(indices));
}

Tensor & Tensor::index_put_(ArrayRef<TensorIndex> indices, Tensor const & rhs) {
  at::indexing::set_item(*this, indices, rhs);
  return *this;
}
Tensor & Tensor::index_put_(ArrayRef<TensorIndex> indices, Tensor && rhs) {
  at::indexing::set_item(*this, indices, rhs);
  return *this;
}
Tensor & Tensor::index_put_(ArrayRef<TensorIndex> indices, Scalar v) {
  at::indexing::set_item(*this, indices, v);
  return *this;
}
Tensor & Tensor::index_put_(std::initializer_list<TensorIndex> indices, Tensor const & rhs) {
  return index_put_(ArrayRef<TensorIndex>(indices), rhs);
}
Tensor & Tensor::index_put_(std::initializer_list<TensorIndex> indices, Tensor && rhs) {
  return index_put_(ArrayRef<TensorIndex>(indices), rhs);
}
Tensor & Tensor::index_put_(std::initializer_list<TensorIndex> indices, Scalar v) {
  return index_put_(ArrayRef<TensorIndex>(indices), v);
}

} // namespace at
