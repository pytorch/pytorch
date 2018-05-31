#include "TensorIterator.h"

#include <ATen/ExpandUtils.h>
#include <ATen/Parallel.h>
#include <time.h>

namespace at {

void TensorIterator::reorder_dimensions() {
  // Sort the dimensions based on the sum-of-strides in ascending order. NOTE:
  // that this inverts the order of C-contiguous tensors. strides[0] is the
  // fastest moving dimension instead of strides[ndim - 1].

  auto sum_of_strides = SmallVector<double, 6>(ndim(), 0.0);
  for (int dim = 0; dim < ndim(); dim++) {
    double sum = 0.0;
    for (const auto& op : operands_) {
      if (op.stride_.size() == 0) continue;
      sum += op.stride_[dim];
    }

    // Weight each dimension by its index. Given two dimensions with equal
    // some of strides, this preserves the given relative ordering.
    sum += (ndim() - dim - 1) / (double)ndim();

    sum_of_strides[dim] = sum;
  }

  // initialize perm with 0, 1, 2, ...
  perm_.resize(ndim());
  std::iota(std::begin(perm_), std::end(perm_), 0);

  std::sort(std::begin(perm_), std::end(perm_), [&](size_t i1, size_t i2) {
    return sum_of_strides[i1] < sum_of_strides[i2];
  });

  auto reorder = [](IntList data, IntList perm_) {
    auto res = DimVector(data.size(), 0);
    for (size_t i = 0; i < perm_.size(); i++) {
      res[i] = data[perm_[i]];
    }
    return res;
  };

  // Update shape and strides
  shape_ = reorder(shape_, perm_);
  for (auto& op : operands_) {
    if (op.stride_.size() > 0) {
      op.stride_ = reorder(op.stride_, perm_);
    }
  }
}

template <typename F>
static std::tuple<ScalarType, Backend>
compute_result_type(at::ArrayRef<OperandInfo> operands, const F& predicate) {
  auto result_type = ScalarType::Undefined;
  auto backend = Backend::Undefined;
  for (auto& op : operands) {
    if (!op.tensor_->defined()) continue;
    if (!predicate(*op.tensor_)) continue;
    auto dtype = op.tensor_->type().scalarType();;
    result_type = (result_type == ScalarType::Undefined
        ? dtype
        : promoteTypes(result_type, dtype));
    backend = (backend == Backend::Undefined
        ? op.tensor_->type().backend()
        : backend);
  }
  return std::make_tuple(result_type, backend);
}

void TensorIterator::compute_common_type() {
  // The result dtype is computed with the precedence:
  // 1) Tensors of rank one or higher
  // 2) Tensors of any rank that aren't wrapped numbers
  // 3) Any tensor
  //
  // For example:
  //  torch.randn(3, 3) + torch.tensor([1.0], dtype=torch.double) -> torch.double
  //  torch.randn(3, 3) + torch.tensor(1.0, dtype=torch.double) -> torch.float
  //  torch.tensor(3, dtype=torch.uint8) + 5 -> torch.uint8
  //
  auto result_type = ScalarType::Undefined;
  auto backend = Backend::Undefined;
  std::tie(result_type, backend) = compute_result_type(operands_, [](const Tensor& t) {
    return t.dim() > 0;
  });
  if (result_type == ScalarType::Undefined) {
    std::tie(result_type, backend) = compute_result_type(operands_, [](const Tensor& t) {
      return !t.get()->is_wrapped_number();
    });
  }
  if (result_type == ScalarType::Undefined) {
    std::tie(result_type, backend) = compute_result_type(operands_, [](const Tensor& t) {
      return true;
    });
  }

  AT_ASSERT(result_type != ScalarType::Undefined);
  AT_ASSERT(backend != Backend::Undefined);

  auto& type = at::globalContext().getType(backend, result_type);

  for (auto& op : operands_) {
    if (!op.type_) {
      op.type_ = &type;
      if (op.tensor_->defined() && type != op.tensor_->type()) {
        if (op.tensor_->dim() == 0) {
          if (type.backend() != at::kCUDA) {
            *op.tensor_ = op.tensor_->toType(type);
          }
        } else {
          op.needs_cast_ = true;
        }
      }
    }
  }
}

DimVector TensorIterator::compatible_stride(int element_size) const {
  auto stride = DimVector();
  stride.push_back(element_size);
  for (int i = 0; i < ndim() - 1; i++) {
    stride.push_back(shape_[i] * stride[i]);
  }
  return stride;
}

DimVector TensorIterator::invert_perm(IntList input) const {
  // Invert the permutation caused by reorder_dimensions. This is not valid
  // after coalesce_dimensions is called.
  auto res = DimVector(input.size(), 0);
  for (int dim = 0; dim < ndim(); dim++) {
    res[perm_[dim]] = input[dim];
  }
  return res;
}

void TensorIterator::allocate_outputs() {
  for (int i = 0; i < num_outputs_; i++) {
    auto& op = operands_[i];
    if (!op.tensor_->defined()) {
      int element_size = op.type_->elementSizeInBytes();
      op.stride_ = compatible_stride(element_size);

      auto tensor_shape = invert_perm(shape_);
      auto tensor_stride = invert_perm(op.stride_);
      for (int dim = 0; dim < ndim(); dim++) {
        tensor_stride[dim] /= element_size;
      }
      *op.tensor_ = op.type_->tensor(tensor_shape, tensor_stride);
    }
  }
}

void TensorIterator::coalesce_dimensions() {
  if (ndim() == 0) {
    return;
  }

  // We can coalesce two adjacent dimensions if either dim has size 1 or if:
  // shape[n] * stride[n] == shape[n + 1].
  auto can_coalesce = [&](int dim0, int dim1) {
    auto shape0 = shape_[dim0];
    auto shape1 = shape_[dim1];
    if (shape0 == 1 || shape1 == 1) {
      return true;
    }
    for (int i = 0; i < ntensors(); i++) {
      auto& stride = operands_[i].stride_;
      if (shape0 * stride[dim0] != stride[dim1]) {
        return false;
      }
    }
    return true;
  };

  auto copy_strides = [&](int dim0, int dim1) {
    for (int i = 0; i < ntensors(); i++) {
      auto& stride = operands_[i].stride_;
      stride[dim0] = stride[dim1];
    }
  };

  int prev_dim = 0;
  for (int dim = 1; dim < ndim(); dim++) {
    if (can_coalesce(prev_dim, dim)) {
      if (shape_[prev_dim] == 1) {
        copy_strides(prev_dim, dim);
      }
      shape_[prev_dim] *= shape_[dim];
    } else {
      prev_dim++;
      copy_strides(prev_dim, dim);
      shape_[prev_dim] = shape_[dim];
    }
  }

  shape_.resize(prev_dim + 1);
  for (int i = 0; i < ntensors(); i++) {
    operands_[i].stride_.resize(ndim());
  }
}

int64_t TensorIterator::numel() const {
  int64_t numel = 1;
  for (int64_t size : shape_) {
    numel *= size;
  }
  return numel;
}

DimVector TensorIterator::get_inner_strides() const {
  auto dims = ndim();
  auto inner_strides = DimVector();
  for (auto& op : operands_) {
    inner_strides.push_back(dims == 0 ? 0 : op.stride_[0]);
  }
  return inner_strides;
}

SmallVector<char*, 4> TensorIterator::get_data_ptrs(ArrayRef<char*> base, IntList counter) const {
  auto ptrs = SmallVector<char*, 4>(base);
  for (int i = 0; i < ntensors(); i++) {
    auto& stride = operands_[i].stride_;
    for (int dim = 0; dim < ndim(); dim++) {
      ptrs[i] += counter[dim] * stride[dim];
    }
  }
  return ptrs;
}

SmallVector<char*, 4> TensorIterator::get_base_ptrs() const {
  auto ptrs = SmallVector<char*, 4>();
  for (int i = 0; i < ntensors(); i++) {
    ptrs.push_back((char*)data_ptr(i));
  }
  return ptrs;
}

DimVector TensorIterator::make_counter(int64_t linear_offset) const {
  auto counter = DimVector();
  int64_t x = linear_offset;
  for (auto size : shape_) {
    counter.push_back(x % size);
    x /= size;
  }
  AT_ASSERT(x == 0);
  return counter;
}

void TensorIterator::increment_counter(DimVector& counter, int64_t n) const {
  int64_t overflow = n;
  for (int i = 0; i < ndim(); i++) {
    auto size = shape_[i];
    auto value = counter[i];
    value += overflow;
    overflow = value / size;
    counter[i] = value % size;
  }
}

void TensorIterator::for_each(loop_t loop) {
  auto inner_strides = get_inner_strides();
  auto base_ptrs = get_base_ptrs();

  at::parallel_for(0, numel(), internal::GRAIN_SIZE, [&](int64_t begin, int64_t end) {
    serial_for_each(loop, base_ptrs, inner_strides, begin, end - begin);
  });
}

void TensorIterator::serial_for_each(loop_t loop, ArrayRef<char*> base_ptrs, IntList inner_strides, int64_t start, int64_t n) {
  if (ndim() <= 1) {
    auto ptrs = get_data_ptrs(base_ptrs, { start });
    loop(ntensors(), ptrs.data(), inner_strides.data(), n);
  } else {
    auto counter = make_counter(start);
    while (n > 0) {
      auto ptrs = get_data_ptrs(base_ptrs, counter);
      int64_t loop_size = std::min(n, shape_[0] - counter[0]);
      loop(ntensors(), ptrs.data(), inner_strides.data(), loop_size);
      n -= loop_size;
      if (n == 0) break;
      increment_counter(counter, loop_size);
    }
  }
}

bool TensorIterator::is_trivial_1d() const {
  // TODO: check for casting once it's supported
  return ndim() == 1;
}

bool TensorIterator::is_scalar(int arg) const {
  const auto& stride = operands_[arg].stride_;
  for (int i = 0; i < ndim(); i++) {
    if (stride[i] != 0 && shape_[i] != 1) {
      return false;
    }
  }
  return true;
}

bool TensorIterator::is_cpu_scalar(int arg) const {
  return is_scalar(arg) && operands_[arg].tensor_->type().backend() == at::kCPU;
}

void* TensorIterator::data_ptr(int arg) const {
  return operands_[arg].data_;
}

void TensorIterator::remove_operand(int arg) {
  operands_.erase(operands_.begin() + arg);
}

void TensorIterator::narrow(int dim, int64_t start, int64_t size) {
  AT_ASSERT(dim < ndim() && size >= 1);
  shape_[dim] = size;
  for (auto& op : operands_) {
    op.data_ = ((char*)op.data_) + op.stride_[dim] * start;
  }
  if (size == 1) {
    coalesce_dimensions();
  }
}

std::unique_ptr<TensorIterator> TensorIterator::binary_op(const Tensor& a, const Tensor& b, Tensor& out) {
  auto builder = TensorIterator::Builder();
  builder.add_output(out);
  builder.add_input(a);
  builder.add_input(b);
  return builder.build();
}

TensorIterator TensorIterator::reduce_op(const Tensor& a, IntList dims) {
  return TensorIterator();
}

void TensorIterator::compute_shape() {
  for (auto& op : operands_) {
    if (!op.tensor_->defined()) continue;
    auto shape = op.tensor_->sizes();
    if (shape_.empty()) {
      shape_ = shape;
    } else if (!shape.equals(shape_)) {
      shape_ = DimVector(infer_size(shape_, shape));
    }
  }

  // Outputs cannot be broadcasted. Check that the shape of the outputs matches
  // the inferred shape.
  for (int i = 0; i < num_outputs_; i++) {
    auto& tensor = *operands_[i].tensor_;
    if (tensor.defined() && !tensor.sizes().equals(shape_)) {
      AT_ERROR("output with shape ", tensor.sizes(), " doesn't match the broadcast shape ",
               shape_);
    }
  }
}

static DimVector compute_stride(const Tensor& tensor, IntList shape) {
  int ndim = shape.size();
  auto original_shape = tensor.sizes();
  auto original_stride = tensor.strides();
  auto element_size_in_bytes = tensor.type().elementSizeInBytes();

  auto stride = DimVector(ndim, 0);
  auto offset = ndim - original_shape.size();
  for (size_t i = 0; i < original_shape.size(); i++) {
    if (original_shape[i] == 1) {
      stride[offset + i] = 0;
    } else {
      stride[offset + i] = original_stride[i] * element_size_in_bytes;
    }
  }
  return stride;
}

void TensorIterator::compute_strides() {
  for (auto& op : operands_) {
    if (op.tensor_->defined()) {
      op.stride_ = compute_stride(*op.tensor_, shape_);
    }
  }
}

void TensorIterator::check_type_conversions() {
  for (auto& op : operands_) {
    if (op.needs_cast_) {
      AT_ERROR("TensorIterator expected type ", type().toString(), " but got ", op.tensor_->type().toString(),
            op.tensor_->sizes());
    }
  }
}

bool TensorIterator::can_use_32bit_indexing() const {
  int64_t max_value = std::numeric_limits<int32_t>::max();
  if (numel() > max_value) {
    return false;
  }
  for (auto& op : operands_) {
    int64_t max_offset = 1;
    for (int dim = 0; dim < ndim(); dim++) {
      max_offset += (shape_[dim] - 1) * op.stride_[dim];
    }
    if (max_offset > max_value) {
      return false;
    }
  }
  return true;
}

std::unique_ptr<TensorIterator> TensorIterator::split() {
  AT_ASSERT(ndim() >= 1 && shape().back() >= 2);
  std::unique_ptr<TensorIterator> copy(new TensorIterator(*this));

  int last_dim = ndim() - 1;
  auto copy_size = shape().back() / 2;
  auto this_size = shape().back() - copy_size;
  this->narrow(last_dim, 0, this_size);
  copy->narrow(last_dim, this_size, copy_size);

  return copy;
}

SplitUntil32Bit TensorIterator::with_32bit_indexing() const {
  return SplitUntil32Bit(*this);
}

std::unique_ptr<TensorIterator> TensorIterator::Builder::build() {
  // compute the broadcasted shape
  iter_->compute_shape();
  // compute each tensor's stride after broadcasting
  iter_->compute_strides();
  // re-order dimensions to improve coalescing
  iter_->reorder_dimensions();
  // compute the result dtype and backend
  iter_->compute_common_type();
  // allocate the output tensor if it's not provided
  iter_->allocate_outputs();
  // coalesce adjacent dimensions when possible
  iter_->coalesce_dimensions();

  for (auto& op : iter_->operands_) {
    AT_ASSERT(op.tensor_->defined());
    op.data_ = op.tensor_->data_ptr();
  }

  iter_->check_type_conversions();

  return std::move(iter_);
}

/// SplitUntil32Bit. Recursively splits an iterator into sub-iterators that
/// can use 32-bit indexing.

SplitUntil32Bit::iterator::iterator(const TensorIterator& iter) {
  vec.emplace_back(new TensorIterator(iter));
  vec.emplace_back(nullptr); // ++ first pops the last element
  ++(*this);
}

SplitUntil32Bit::iterator& SplitUntil32Bit::iterator::operator++() {
  vec.pop_back();
  while (!vec.empty() && !vec.back()->can_use_32bit_indexing()) {
    vec.emplace_back(vec.back()->split());
  }
  return *this;
}

TensorIterator& SplitUntil32Bit::iterator::operator*() const {
  return *vec.back();
}

SplitUntil32Bit::iterator SplitUntil32Bit::begin() const {
  return SplitUntil32Bit::iterator(iter);
}

SplitUntil32Bit::iterator SplitUntil32Bit::end() const {
  return SplitUntil32Bit::iterator();
}

}  // namespace at
