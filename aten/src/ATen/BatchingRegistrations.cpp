#include <torch/library.h>
#include <ATen/VmapTransforms.h>
#include <ATen/BatchedFallback.h>
#include <ATen/native/ResizeCommon.h>
#include <ATen/ATen.h>

namespace at {

// NOTE: [What is a batching rule?]
//
// A *batching rule* implements the logic of how to call an operator on inputs
// that have zero or more additional batch dimensions. When one does a vmap, the
// dimension(s) being vmap'ed over get recorded as batch dimensions.
//
// For example, vmap(torch.add)(x, y)
// 1. wraps `x` into batched_x = BatchedTensor(x, bdims=[(lvl=1, dim=0)];
// 2. wraps `y` into batched_y = BatchedTensor(y, bdims=[(lvl=1, dim=0)];
// 3. and then runs `torch.add(batched_x, batched_y)`.

// NOTE: [When should I add a batching rule?]
// When you are adding a new operator, you'll need to add a batching rule so
// that vmap can work efficiently with said operator. If you do not, we'll attempt
// to generate a slow fallback for the batching rule.

// NOTE: [How to write batching rules?]
// The signature of a batching rule should look like exactly like the C++ signature
// of its operator.
//
// First, see NOTE: [Logical vs physical args] in VmapTransforms.h for terminology.
//
// At a high level, what a batching rule does is the following:
// 1. Converts (logical) BatchedTensors to views on physical tensors.
// 2. Converts logical arguments (e.g. dimension indexes, shapes) to physical
//    arguments that correspond to the physical tensors.
// 3. Calls at:: operations on the physical tensors and arguments to produce
//    some physical results.
// 4. Converts physical results back to BatchedTensors.
//
// Steps 1, 2, and 4 differ for operators with different batching behaviors. When
// writing a new batching rule, please select a VmapTransform that matches the
// batching behavior of your operation. The VmapTransform provides helper functions
// to do steps (1), (2), and (4).
// (see NOTE: [What is an VmapTransform?] in VmapTransforms.h)

// Note: [Future plans]
// The API for writing a batching rule isn't stable. In the future, we'd like
// to think about the problem of translating these batching rules to TorchScript.
// Ideally batching rules in eager mode vs TorchScript would look pretty similar,
// if not use the same mechanism. In order to accomplish that we might have to
// do some refactoring.

// PyTorch allows operations to specify dim 0 and dim -1 on a scalar tensor.
static bool is_allowed_dim_on_scalar_tensor(int64_t dim) {
  return dim == 0 || dim == -1;
}

Tensor sum_batching_rule(const Tensor& self, IntArrayRef dims, bool keepdim, optional<ScalarType> dtype) {
  // PyTorch has a special case where sum(scalar_tensor, dim=0) does not fail
  // and instead returns a new scalar tensor (this also happens for dim=-1)
  // If the following happens:
  // >>> x = torch.randn(B0)  # the per-examples are all scalars
  // >>> vmap(partial(torch.sum, dim=0), x)
  // then we replicate the behavior of sum(scalar_tensor, dim=0).
  if (/*logical*/self.dim() == 0 && dims.size() == 1 && is_allowed_dim_on_scalar_tensor(dims[0])) {
    return self.clone();
  }
  auto self_physical = MultiBatchVmapTransform::logicalToPhysical(self);
  auto dims_physical = self_physical.getPhysicalDims(dims);
  auto result = at::sum(self_physical.tensor(), dims_physical, keepdim, dtype);
  return self_physical.getPhysicalToLogicalMap().apply(result);
}

bool isPhysicalScalarTensor(const Tensor& logical_tensor) {
  if (logical_tensor.dim() > 0) {
    return false;
  }
  auto* batched = maybeGetBatchedImpl(logical_tensor);
  if (batched) {
    return false;
  }
  return true;
}

template <typename F, F Func, typename... ExtraArgs>
Tensor binary_pointwise_batching_rule(
    const Tensor& self, const Tensor& other, ExtraArgs... args) {
  if (self.dim() > 0 && other.dim() > 0) {
    auto physical_args = BroadcastingVmapTransform::logicalToPhysical({self, other});
    auto result = Func(physical_args[0].tensor(), physical_args[1].tensor(), args...);
    return physical_args[0].getPhysicalToLogicalMap().apply(result);
  }
  if (isPhysicalScalarTensor(self)) {
    auto other_physical = MultiBatchVmapTransform::logicalToPhysical(other);
    auto result = Func(self, other_physical.tensor(), args...);
    return other_physical.getPhysicalToLogicalMap().apply(result);
  }
  if (isPhysicalScalarTensor(other)) {
    auto self_physical = MultiBatchVmapTransform::logicalToPhysical(self);
    auto result = Func(self_physical.tensor(), other, args...);
    return self_physical.getPhysicalToLogicalMap().apply(result);
  }

  // At this point, we know at least one of the operands is a logical Scalar tensor.
  // Here we must emulate TensorIterator's special behavior on Scalars.
  //
  // As a motivating example, consider the following:
  //   x = torch.randn(3, 10)
  //   y = torch.randn(3, dtype=torch.double)
  //   vmap(torch.mul)(torch.randn(3, 10), torch.randn(3, dtype=torch.double))
  //
  // At a per-example level, we are adding FloatTensor[10] and DoubleTensor[];
  // Type Promotion dictates that the result should be FloatTensor[10].
  // This means we cannot directly pass the physical tensors (x and y) to
  // TensorIterator (if we did, it would promote them to DoubleTensor).
  //
  // FIXME(rzou): I didn't want to go down the slippery slope of emulating
  // everything TensorIterator does (it would be better to refactor out the
  // TensorIterator logic). The one thing that this code doesn't handle
  // is cross-device logical scalar tensors.
  //   cpu_tensor = torch.randn(3)
  //   cuda_tensor = torch.randn(3, 10, device='cuda')
  //   vmap(torch.mul)(cpu_tensor, cuda_tensor)
  //
  // At a per-example level, we are adding CPUTensor[] and CUDATensor[10].
  // TensorIterator allows for this cross-device operation because one of the
  // tensors is a Scalar CPU tensor. However, the following code will throw an
  // error in that case. I don't expect to see many use cases for this, so
  // this is probably fine as-is.
  auto logical_self = self;
  auto logical_other = other;
  auto result_type = at::native::result_type(logical_self, logical_other);
  if (logical_self.scalar_type() != result_type) {
    logical_self = logical_self.to(result_type);
  }
  if (logical_other.scalar_type() != result_type) {
    logical_other = logical_other.to(result_type);
  }
  auto physical_args = BroadcastingVmapTransform::logicalToPhysical(
      {logical_self, logical_other});
  auto result = Func(physical_args[0].tensor(), physical_args[1].tensor(), args...);
  return physical_args[0].getPhysicalToLogicalMap().apply(result);
}

Tensor expand_batching_rule(const Tensor& self, IntArrayRef size, bool implicit) {
  auto self_physical = MultiBatchVmapTransform::logicalToPhysical(self);
  auto size_physical = self_physical.getPhysicalShape(size);
  auto self_physical_dim = self_physical.tensor().dim();

  TORCH_CHECK(self_physical_dim <= static_cast<int64_t>(size_physical.size()),
       "expand: the number of sizes provided (", /*logical*/size.size(), ") ",
       "must be greater or equal to the number of dimensions in the tensor (",
       /*logical dim*/self.dim(), ")");

  if (self_physical_dim == static_cast<int64_t>(size_physical.size())) {
    auto result = self_physical.tensor().expand(size_physical, implicit);
    return self_physical.getPhysicalToLogicalMap().apply(result);
  }

  TORCH_INTERNAL_ASSERT(self_physical_dim < static_cast<int64_t>(size_physical.size()));
  // Here, we know we are expanding a (logical) tensor to a larger number
  // of dimensions. We have to be careful because we can't call expand directly
  // due to the presence of batch dimensions.
  //
  // As an example, let B0 be a batch dimension and consider expand(Tensor[B0, 3], [2, 3]).
  // The result should be a tensor of size [B0, 2, 3].
  // A physical view of size [B0, 3] can't directly be expanded to size [B0, 2, 3]
  // so the strategy here is to view it first as a tensor of size [B0, 1, 3] and
  // then expand.
  auto self_physical_size = self_physical.tensor().sizes();
  auto extra_dims = size_physical.size() - self_physical_dim;
  VmapDimVector view_shape(size_physical.size(), 1);
  std::copy(self_physical_size.begin(),
            self_physical_size.begin() + self_physical.numBatchDims(),
            view_shape.begin());
  std::copy(self_physical_size.begin() + self_physical.numBatchDims(),
            self_physical_size.end(),
            view_shape.begin() + self_physical.numBatchDims() + extra_dims);
  auto result = self_physical.tensor().view(view_shape).expand(size_physical, implicit);
  return self_physical.getPhysicalToLogicalMap().apply(result);
}

std::vector<Tensor> chunk_batching_rule(const Tensor& self, int64_t chunks, int64_t dim) {
  auto self_physical = MultiBatchVmapTransform::logicalToPhysical(self);
  auto dim_physical = self_physical.getPhysicalDim(dim);
  auto result = at::chunk(self_physical.tensor(), chunks, dim_physical);
  self_physical.getPhysicalToLogicalMap().applyInplace(result);
  return result;
}

Tensor clamp_batching_rule(const Tensor& self, const optional<Scalar>& min, const optional<Scalar>& max) {
  auto self_physical = MultiBatchVmapTransform::logicalToPhysical(self);
  auto result = at::clamp(self_physical.tensor(), min, max);
  return self_physical.getPhysicalToLogicalMap().apply(result);
}

Tensor clamp_min_batching_rule(const Tensor& self, const Scalar& min) {
  auto self_physical = MultiBatchVmapTransform::logicalToPhysical(self);
  auto result = at::clamp_min(self_physical.tensor(), min);
  return self_physical.getPhysicalToLogicalMap().apply(result);
}

Tensor clamp_max_batching_rule(const Tensor& self, const Scalar& max) {
  auto self_physical = MultiBatchVmapTransform::logicalToPhysical(self);
  auto result = at::clamp_max(self_physical.tensor(), max);
  return self_physical.getPhysicalToLogicalMap().apply(result);
}

std::vector<Tensor> tensor_split_sections_batching_rule(const Tensor& self, int64_t sections, int64_t dim) {
  auto self_physical = MultiBatchVmapTransform::logicalToPhysical(self);
  auto dim_physical = self_physical.getPhysicalDim(dim);
  auto result = at::tensor_split(self_physical.tensor(), sections, dim_physical);
  self_physical.getPhysicalToLogicalMap().applyInplace(result);
  return result;
}

std::vector<Tensor> tensor_split_indices_batching_rule(const Tensor& self, IntArrayRef indices, int64_t dim) {
  auto self_physical = MultiBatchVmapTransform::logicalToPhysical(self);
  auto dim_physical = self_physical.getPhysicalDim(dim);
  auto result = at::tensor_split(self_physical.tensor(), indices, dim_physical);
  self_physical.getPhysicalToLogicalMap().applyInplace(result);
  return result;
}

Tensor unsqueeze_batching_rule(const Tensor& self, int64_t dim) {
  auto self_physical = MultiBatchVmapTransform::logicalToPhysical(self);
  // NB: unsqueeze has some special handling of its `dim` argument so we can't call
  // self_physical.getPhysicalDim directly. In particular, native::unsqueeze
  // wraps the dim to (the logical dimension) + 1, so we need to do that here too.
  // https://github.com/pytorch/pytorch/blob/b623bdeabb0aa8da44285d303246e7f8ac06c2a9/aten/src/ATen/native/TensorShape.cpp#L1413
  auto dim_physical =
      self_physical.numBatchDims() + maybe_wrap_dim(dim, /*logical_dim*/self.dim() + 1);
  auto result = self_physical.tensor().unsqueeze(dim_physical);
  return self_physical.getPhysicalToLogicalMap().apply(result);
}

Tensor& fill_inplace_scalar_batching_rule(Tensor& self, const Scalar& value) {
  auto self_physical = MultiBatchVmapTransform::logicalToPhysical(self);
  self_physical.tensor().fill_(value);
  return self;
}

Tensor& fill_inplace_tensor_batching_rule(Tensor& self, const Tensor& value) {
  auto value_batched = isBatchedTensor(value);

  if (value_batched) {
    auto physical_args =
      BroadcastingVmapTransform::logicalToPhysical({self, value});
    physical_args[0].tensor().copy_(physical_args[1].tensor());
  } else {
    auto self_physical = MultiBatchVmapTransform::logicalToPhysical(self);
    self_physical.tensor().fill_(value);
  }
  return self;
}

Tensor& zero_inplace_batching_rule(Tensor &self) {
  auto self_physical = MultiBatchVmapTransform::logicalToPhysical(self);
  self_physical.tensor().zero_();
  return self;
}

Tensor squeeze_batching_rule(const Tensor& self) {
  auto self_physical = MultiBatchVmapTransform::logicalToPhysical(self);
  auto physical_sizes = self_physical.tensor().sizes();

  // Don't squeeze the batch dims!
  VmapDimVector squeezed_sizes;
  int64_t num_batch_dims = self_physical.numBatchDims();
  squeezed_sizes.insert(
      squeezed_sizes.end(),
      physical_sizes.begin(),
      physical_sizes.begin() + num_batch_dims);
  for (auto it = physical_sizes.begin() + num_batch_dims; it != physical_sizes.end(); ++it) {
    if (*it != 1) {
      squeezed_sizes.push_back(*it);
    }
  }

  auto result = self_physical.tensor().view(squeezed_sizes);
  return self_physical.getPhysicalToLogicalMap().apply(result);
}

Tensor squeeze_dim_batching_rule(const Tensor& self, int64_t dim) {
  auto self_physical = MultiBatchVmapTransform::logicalToPhysical(self);
  auto dim_physical = self_physical.getPhysicalDim(dim);
  auto result = self_physical.tensor().squeeze(dim_physical);
  return self_physical.getPhysicalToLogicalMap().apply(result);
}

Tensor trace_batching_rule(const Tensor& self) {
  auto self_physical = MultiBatchVmapTransform::logicalToPhysical(self);
  // Batched Diagonal View
  auto self_diag = at::diagonal(self_physical.tensor(), /*offset*/0, /*dim1*/-2, /*dim2*/-1);
  auto result =  at::sum(self_diag, -1);
  return self_physical.getPhysicalToLogicalMap().apply(result);
}

Tensor trace_backward_batching_rule(const Tensor& grad, IntArrayRef input_sizes) {
  auto grad_physical = MultiBatchVmapTransform::logicalToPhysical(grad);
  auto grad_input = at::zeros(grad_physical.getPhysicalShape(input_sizes), grad.options());
  // Batched Diagonal View
  auto grad_input_diag = at::diagonal(grad_input, /*offset*/0, /*dim1*/-2, /*dim2*/-1);
  // Append a dimension of size one to the grad output
  auto grad_physical_tensor = grad_physical.tensor().unsqueeze(-1);
  grad_input_diag.copy_(grad_physical_tensor);
  return grad_physical.getPhysicalToLogicalMap().apply(grad_input);
}

Tensor transpose_int_batching_rule(const Tensor& self, int64_t dim0, int64_t dim1) {
  // PyTorch has a special case where scalar_tensor.transpose(dim0, dim1) works
  // for dim0, dim1 in {0, -1} and returns the scalar tensor. If the following happens:
  // >>> x = torch.randn(B0)  # the per-examples are all scalars
  // >>> vmap(lambda x: x.transpose(0, -1), x)
  // then we replicate this behavior.
  if (/*logical*/self.dim() == 0 && is_allowed_dim_on_scalar_tensor(dim0) &&
      is_allowed_dim_on_scalar_tensor(dim1)) {
    return self;
  }
  auto self_physical = MultiBatchVmapTransform::logicalToPhysical(self);
  auto dim0_physical = self_physical.getPhysicalDim(dim0);
  auto dim1_physical = self_physical.getPhysicalDim(dim1);
  auto result = self_physical.tensor().transpose(dim0_physical, dim1_physical);
  return self_physical.getPhysicalToLogicalMap().apply(result);
}

Tensor permute_batching_rule(const Tensor& self, IntArrayRef dims) {
  auto self_physical = MultiBatchVmapTransform::logicalToPhysical(self);
  auto dims_physical = self_physical.getPhysicalDims(dims);

  VmapDimVector all_dims_physical;
  all_dims_physical.reserve(self_physical.tensor().dim());
  for (int64_t bdim = 0; bdim < self_physical.numBatchDims(); bdim++) {
    all_dims_physical.push_back(bdim);
  }
  all_dims_physical.insert(
      all_dims_physical.end(),
      dims_physical.begin(),
      dims_physical.end());
  auto result = self_physical.tensor().permute(all_dims_physical);
  return self_physical.getPhysicalToLogicalMap().apply(result);
}

Tensor select_batching_rule(const Tensor& self, int64_t dim, int64_t index) {
  auto self_physical = MultiBatchVmapTransform::logicalToPhysical(self);
  auto dim_physical = self_physical.getPhysicalDim(dim);
  auto result = self_physical.tensor().select(dim_physical, index);
  return self_physical.getPhysicalToLogicalMap().apply(result);
}

static int64_t getGradInputPhysicalDim(int64_t dim, IntArrayRef input_sizes, int64_t num_batch_dims) {
  return maybe_wrap_dim(dim, input_sizes.size()) + num_batch_dims;
}

Tensor select_backward_batching_rule(const Tensor& grad, IntArrayRef input_sizes, int64_t dim, int64_t index) {
  auto grad_physical = MultiBatchVmapTransform::logicalToPhysical(grad);
  auto grad_input = at::zeros(grad_physical.getPhysicalShape(input_sizes), grad.options());
  auto physical_dim = getGradInputPhysicalDim(dim, input_sizes, grad_physical.numBatchDims());
  grad_input.select(physical_dim, index).copy_(grad_physical.tensor());
  return grad_physical.getPhysicalToLogicalMap().apply(grad_input);
}

Tensor slice_batching_rule(
    const Tensor& self,
    int64_t dim,
    c10::optional<int64_t> start,
    c10::optional<int64_t> end,
    int64_t step) {
  auto self_physical = MultiBatchVmapTransform::logicalToPhysical(self);
  auto dim_physical = self_physical.getPhysicalDim(dim);
  auto result = self_physical.tensor().slice(dim_physical, start, end, step);
  return self_physical.getPhysicalToLogicalMap().apply(result);
}

Tensor slice_backward_batching_rule(const Tensor& grad, IntArrayRef input_sizes, int64_t dim, int64_t start, int64_t end, int64_t step) {
  auto grad_physical = MultiBatchVmapTransform::logicalToPhysical(grad);
  auto grad_input = at::zeros(grad_physical.getPhysicalShape(input_sizes), grad.options());
  auto physical_dim = getGradInputPhysicalDim(dim, input_sizes, grad_physical.numBatchDims());
  grad_input.slice(physical_dim, start, end, step).copy_(grad_physical.tensor());
  return grad_physical.getPhysicalToLogicalMap().apply(grad_input);
}

Tensor diagonal_batching_rule(const Tensor& self, int64_t offset, int64_t dim1, int64_t dim2) {
  auto self_physical = MultiBatchVmapTransform::logicalToPhysical(self);
  auto dim1_physical = self_physical.getPhysicalDim(dim1);
  auto dim2_physical = self_physical.getPhysicalDim(dim2);
  auto result = at::diagonal(self_physical.tensor(), offset, dim1_physical, dim2_physical);
  return self_physical.getPhysicalToLogicalMap().apply(result);
}

Tensor diagonal_backward_batching_rule(const Tensor& grad, IntArrayRef input_sizes, int64_t offset, int64_t dim1, int64_t dim2) {
  auto grad_physical = MultiBatchVmapTransform::logicalToPhysical(grad);
  auto grad_input = at::zeros(grad_physical.getPhysicalShape(input_sizes), grad.options());
  auto dim1_physical = getGradInputPhysicalDim(dim1, input_sizes, grad_physical.numBatchDims());
  auto dim2_physical = getGradInputPhysicalDim(dim2, input_sizes, grad_physical.numBatchDims());
  grad_input.diagonal(offset, dim1_physical, dim2_physical).copy_(grad_physical.tensor());
  return grad_physical.getPhysicalToLogicalMap().apply(grad_input);
}

Tensor movedim_batching_rule(const Tensor& self, IntArrayRef source, IntArrayRef destination) {
  auto self_physical = MultiBatchVmapTransform::logicalToPhysical(self);
  auto source_physical = self_physical.getPhysicalDims(source);
  auto destination_physical = self_physical.getPhysicalDims(destination);
  auto result = at::movedim(self_physical.tensor(), source_physical, destination_physical);
  return self_physical.getPhysicalToLogicalMap().apply(result);
}

Tensor reshape_batching_rule(const Tensor& self, IntArrayRef shape) {
  auto self_physical = MultiBatchVmapTransform::logicalToPhysical(self);
  auto shape_physical = self_physical.getPhysicalShape(shape);
  auto result = self_physical.tensor().reshape(shape_physical);
  return self_physical.getPhysicalToLogicalMap().apply(result);
}

std::vector<Tensor> split_batching_rule(const Tensor& self, int64_t split_size, int64_t dim) {
  auto self_physical = MultiBatchVmapTransform::logicalToPhysical(self);
  auto dim_physical = self_physical.getPhysicalDim(dim);
  auto result = at::split(self_physical.tensor(), split_size, dim_physical);
  self_physical.getPhysicalToLogicalMap().applyInplace(result);
  return result;
}

std::vector<Tensor> split_with_sizes_batching_rule(const Tensor& self, IntArrayRef split_sizes, int64_t dim) {
  auto self_physical = MultiBatchVmapTransform::logicalToPhysical(self);
  auto dim_physical = self_physical.getPhysicalDim(dim);
  auto result = at::split_with_sizes(self_physical.tensor(), split_sizes, dim_physical);
  self_physical.getPhysicalToLogicalMap().applyInplace(result);
  return result;
}

std::vector<Tensor> unbind_batching_rule(const Tensor& self, int64_t dim) {
  auto self_physical = MultiBatchVmapTransform::logicalToPhysical(self);
  auto dim_physical = self_physical.getPhysicalDim(dim);
  auto result = at::unbind(self_physical.tensor(), dim_physical);
  self_physical.getPhysicalToLogicalMap().applyInplace(result);
  return result;
}

Tensor unfold_batching_rule(const Tensor& self, int64_t dim, int64_t size, int64_t step) {
  auto self_physical = MultiBatchVmapTransform::logicalToPhysical(self);
  auto dim_physical = self_physical.getPhysicalDim(dim);
  auto result = self_physical.tensor().unfold(dim_physical, size, step);
  return self_physical.getPhysicalToLogicalMap().apply(result);
}

Tensor contiguous_batching_rule(const Tensor& self, MemoryFormat memory_format) {
  TORCH_CHECK(memory_format == MemoryFormat::Contiguous,
      "NYI: Tensor.contiguous(...) inside of vmap for memory_format other ",
      "than torch.contiguous_format");
  auto physical_view = MultiBatchVmapTransform::logicalToPhysical(self);
  auto result = physical_view.tensor().contiguous(memory_format);
  return physical_view.getPhysicalToLogicalMap().apply(result);
}

Tensor view_batching_rule(const Tensor& self, IntArrayRef size) {
  auto self_physical = MultiBatchVmapTransform::logicalToPhysical(self);
  auto size_physical = self_physical.getPhysicalShape(size);
  auto result = self_physical.tensor().view(size_physical);
  return self_physical.getPhysicalToLogicalMap().apply(result);
}

Tensor view_as_complex_batching_rule(const Tensor& self) {
  // guard against the user passing in a batch of scalar tensors with batch
  // size equal to 2.
  TORCH_CHECK(self.sizes().size() != 0, "Input tensor must have one or more dimensions");
  auto self_physical = MultiBatchVmapTransform::logicalToPhysical(self);
  auto result = at::view_as_complex(self_physical.tensor());
  return self_physical.getPhysicalToLogicalMap().apply(result);
}

// Checks that the smallest batch stride is greater than the largest example
// stride. This is something we can support but we choose not to because it's
// potentially error prone.
static void checkBatchDimsAtFrontInLayout(IntArrayRef physical_strides, int64_t num_batch_dims) {
  auto smallest_batch_stride = std::min_element(
      physical_strides.begin(), physical_strides.begin() + num_batch_dims);
  auto largest_example_stride = std::max_element(
      physical_strides.begin() + num_batch_dims, physical_strides.end());
  if (largest_example_stride == physical_strides.end()) {
    // No example dimensions
    return;
  }
  TORCH_CHECK(*smallest_batch_stride >= *largest_example_stride,
    "vmap: Calling Tensor.as_strided is not supported unless the batch dims being ",
    "vmapped over are at the front of the tensor (in memory layout). When they are ",
    "not at the front of the tensor this operation can be error prone so we "
    "actively discourage it; please file us a bug report and/or try to ",
    "express the as_strided operation in terms of PyTorch view operations");
}

// given (sizes, strides, storage_offset) returns the maximum location that
// can be indexed (or nullopt if such a location doesn't exist, e.g., tensors
// with zero-size dims).
static optional<int64_t> maximum_indexable_location(
    IntArrayRef sizes, IntArrayRef strides, int64_t storage_offset) {
  auto result = native::storage_size_for(sizes, strides);
  if (result == 0) {
    return nullopt;
  }
  return result + storage_offset;
}

// Let x be the "first slice" of physical_tensor.
// This checks that the range of possible memory locations accessible by
// x.as_strided(sizes, strides, maybe_storage_offset)
// are within the bounds of possible memory locations accessible by x.
static void checkBasicAsStridedValidForSlice(
    const Tensor& physical_tensor,
    int64_t num_batch_dims,
    IntArrayRef sizes,
    IntArrayRef strides,
    optional<int64_t> maybe_storage_offset) {
  auto slice_sizes = physical_tensor.sizes().slice(num_batch_dims);
  auto slice_strides = physical_tensor.strides().slice(num_batch_dims);
  auto base_offset = physical_tensor.storage_offset();

  auto storage_offset = maybe_storage_offset.value_or(base_offset);

  auto max_as_strided_loc = maximum_indexable_location(sizes, strides, storage_offset);
  auto max_slice_loc = maximum_indexable_location(slice_sizes, slice_strides, base_offset);

  if (!max_as_strided_loc.has_value()) {
    return;
  }
  if (!max_slice_loc.has_value()) {
    TORCH_CHECK(false,
        "result = tensor.as_strided(", sizes, ",",  strides, ",", storage_offset, ")",
        "can access memory outside of `tensor`. `tensor` has no storage but the ",
        "passed-in (size, stride, storage_offset) imply a result with some storage. ",
        "This is not supported inside of vmap, please try to rewrite the ",
        "`as_strided` call as a sequence of PyTorch view operations");
  }

  TORCH_CHECK(
      *max_as_strided_loc <= *max_slice_loc && base_offset <= storage_offset,
      "result = tensor.as_strided(", sizes, ",",  strides, ",", storage_offset, ")",
      "can access memory outside of `tensor`. `result` can access some",
      "memory in range [", storage_offset, ", ", *max_as_strided_loc, "], but ",
      "`tensor` can only access some memory in range [", base_offset, ", ",
      *max_slice_loc, "]. This is not supported inside of vmap, please try to",
      "rewrite the `as_strided` call as a sequence of PyTorch view operations");
}

// What are the semantics of as_strided inside of vmap?
// y = vmap(lambda x: x.as_strided(sizes, strides, offset))(xs)
// This returns a view on `x`, `y`, such that each y[i] has:
// - sizes: `sizes`
// - strides: `strides`
// - storage_offset: offset + i * x.stride(batch_dim)
//
// In other words, it is as if we had treated each x[i] as having storage
// offset equal to xs.offset() and called as_strided(sizes, sizes, offset).
// (that is equivalent to x[i].as_strided(
//    sizes, sizes, offset + x[i].storage_offset() - xs.offset()) for all i)
//
// Note that this *may* be different from actually running as_strided
// in a for-loop. This is due to how as_strided takes in `offset` to be
// an *absolute* offset. As an example, consider:
// >>> x = torch.tensor([0., 1., 2., 3., 4.]).as_strided([4], [1], 1)
// >>> z = [x[i].as_strided([1], [1], 1) for i in range(4)]
// Each z[i] is actually the same view on x (z[i] == torch.tensor([1.]))!
// However, we consider the above for-loop comprehension to be a user error:
// a user should have written the following if they wanted to use as_strided
// in a per-sample way:
// >>> z = [x[i].as_strided([1], [1], 1 + x[i].storage_offset() - 1) for i in range(4)]
Tensor as_strided_batching_rule(
    const Tensor& tensor,
    IntArrayRef sizes,
    IntArrayRef strides,
    optional<int64_t> storage_offset) {
  auto physical_view = at::MultiBatchVmapTransform::logicalToPhysical(tensor);
  auto num_batch_dims = physical_view.numBatchDims();
  auto physical_sizes = physical_view.getPhysicalShape(sizes);
  const auto& physical_tensor = physical_view.tensor();

  // We can't rely on the physical as_strided call to do this for us because
  // we do some sanity checks on the size/strides before calling into as_strided.
  TORCH_CHECK(sizes.size() == strides.size(),
      "Tensor.as_strided(size, stride, ...): size and stride must have the ",
      "same length! Got size ", sizes, " and stride ", strides);

  // Sanity checks:
  // 1. All batch dims are at the front in memory layout (not necessary for
  // correctness, but we are worried the user might be doing crazy things)
  // 2. as_strided(sizes, strides, storage_offset + tensor[i].offset() - tensor.offset())
  // is valid for a slice of the input tensor.
  // See Note: [When will the as_strided batching rule fail?] for details.
  checkBatchDimsAtFrontInLayout(physical_tensor.strides(), num_batch_dims);
  checkBasicAsStridedValidForSlice(
      physical_tensor, num_batch_dims, sizes, strides, storage_offset);

  // physical_strides = physical tensor's batch strides + (logical) strides
  auto batch_strides = physical_tensor.strides().slice(0, num_batch_dims);
  at::VmapDimVector physical_strides;
  physical_strides.reserve(num_batch_dims + strides.size());
  physical_strides.insert(
      physical_strides.end(), batch_strides.begin(), batch_strides.end());
  physical_strides.insert(
      physical_strides.end(), strides.begin(), strides.end());

  // If zi = xs[i].as_strided(sizes, strides, offset + xs[i].offset() - xs.offset())
  // is valid for all i, then it turns out that
  // xs.as_strided(physical_sizes, physical_strides, offset) always succeeds
  // and creates a tensor y such that each y[i] references the same memory
  // locations as zi. See NOTE: [When will the as_strided batching rule fail?]
  auto result = physical_view.tensor().as_strided(
      physical_sizes, physical_strides, storage_offset);
  return physical_view.getPhysicalToLogicalMap().apply(result);
}

// NOTE: [When will the as_strided batching rule fail?]
// If zi = xs[i].as_strided(sizes, strides, offset + xs[i].offset() - xs.offset())
// is valid for all i, then it turns out that
// xs.as_strided(physical_sizes, physical_strides, offset) always succeeds and
// creates a tensor y such that each y[i] refers to the same memory as zi.
//
// Let's say we have xs[i].as_strided(sizes, strides, offset + xs[i].offset() - xs.offset()).
// Furthermore, let's say that as a part of being "valid" this as_strided call
// does not return a result that can index memory not indexable by xs[i].
//
// WLOG, assume that there's only one batch dim and it is at the front of the
// `xs` tensor. Let B be the batch size and S be the stride of the batch dim.
// - If the batch dim isn't at the front of the tensor, then we can just move it
// to the front with movedim/permute. This is always valid because it just swaps
// some strides around.
// - This proof also works for tensors with multiple batch dims. We just have to
// do a little accounting:
//   - instead of [B], we'd have [B0, B1, ..., Bk].
//   - instead of [S], we'd have [S0, S1, ..., Sk].
//   - instead of i, we'd have a list of indices [I0, I1, ..., Ik]
//   - instead of S * I, we'd have \sum_{i=0}^k S_i * I_i
//
// [Equation 1]
// xs[i].as_strided(sizes, strides, offset + xs[i].offset() - xs.offset()) has:
// - sizes: sizes
// - strides: strides
// - offset: offset + S * i
//
// x.as_strided itself checks that:
// - (sizes, strides, offset) are in bounds for `x`'s storage.
// - strides are positive
// - offset is positive
//
// Claim 1: if xs[i].as_strided(sizes, strides, offset + xs[i].offset() - xs.offset())
// is valid, then
// ([B] + sizes, [S] + strides, offset + xs.offset()) are in bounds for `xs`'s storage.
//
// If we have the claim, then xs.as_strided([B] + sizes, [S] + strides, offset)
// won't error out. So all we need to check is that the memory locations are
// what we expected. See [Hand-wavy proof of Claim 1] for proof (it's not very important)
//
// xs.as_strided(physical_sizes, physical_strides, offset) is equivalent to
// xs.as_strided([B] + sizes, [S] + strides, offset)
//
// xs.as_strided([B] + sizes, [S] + strides, offset) has:
// - sizes: [B] + sizes
// - strides: [S] + strides
// - offset: offset
//
// xs.as_strided([B] + sizes, [S] + strides, offset)[i] has:
// - sizes: sizes
// - strides: strides
// - offset: offset + S * i
// These memory locations are exactly the same as what we got for [Equation 1],
// so the xs.as_strided([B] + sizes, [S] + strides, offset) is valid.
//
// [Hand-wavy proof of Claim 1]
// Part of our definition of being valid is that xs[i].as_strided(...)
// must return a tensor that only uses memory indexable by xs[i].
// This means that (sizes, strides, offset + xs[i].offset() - xs.offset()) satisfies:
//    offset + xs[i].offset() - xs.offset() + 1 + \sum_j (sizes[j] - 1) * strides[j]
//    <= xs[i].offset() + 1 + \sum_j (xs[i].size(j) - 1) * xs[i].stride(j)
// (the largest-index memory location of xs[i].as_strided(...) must be \leq
// the largest-index memory location of xs[i])
//
// Fiddling that inequality gives us:
//    offset - xs.offset() + 1 + \sum_j (sizes[j] - 1) * strides[j]
//    <= 1 + \sum_j (xs[i].size(j) - 1) * xs[i].stride(j)
//
//    offset - xs.offset() + 1 + (B-1)*S + \sum_j (sizes[j] - 1) * strides[j]
//    <= 1 + (B-1)*S + \sum_j (xs[i].size(j) - 1) * xs[i].stride(j)
//
//    offset - xs.offset() + 1 + (B-1)*S + \sum_j (sizes[j] - 1) * strides[j]
//    <= 1 + \sum_j (xs.size(j) - 1) * xs.stride(j)
//
//    offset + 1 + (B-1)*S + \sum_j (sizes[j] - 1) * strides[j]
//    <= xs.offset() + 1 + \sum_j (xs.size(j) - 1) * xs.stride(j)
// (the largest-index memory location of xs.as_strided(size, stride, offset)
// is \leq than the largest-index memory location of xs)
// Under the assumptions we've made, the lower bound (lowest indexed memory)
// is trivially within the storage.
//
// Therefore ([B] + sizes, [S] + strides, offset) are in bounds for
// `xs`'s storage.

template <typename F, F Func, typename... ExtraArgs>
Tensor unwrap_and_call(const Tensor& input, ExtraArgs... args) {
  auto* input_batched = unsafeGetBatchedImpl(input);
  auto output_physical = Func(input_batched->value(), args...);
  auto old_bdims = input_batched->bdims();
  return makeBatched(output_physical, BatchDims(old_bdims.begin(), old_bdims.end()));
}

template <typename F, F Func, typename... ExtraArgs>
Tensor unwrap_and_call_method(const Tensor& input, ExtraArgs... extra_args) {
  auto* input_batched = unsafeGetBatchedImpl(input);
  auto output_physical = (input_batched->value().*Func)(extra_args...);
  auto old_bdims = input_batched->bdims();
  return makeBatched(output_physical, BatchDims(old_bdims.begin(), old_bdims.end()));
}

Tensor pow_scalar_Tensor_batching_rule(const Scalar& other, const Tensor& self) {
  auto* self_batched = unsafeGetBatchedImpl(self);
  auto output_physical = at::pow(other, self_batched->value());
  auto old_bdims = self_batched->bdims();
  return makeBatched(output_physical, BatchDims(old_bdims.begin(), old_bdims.end()));
}

Tensor clone_batching_rule(const Tensor& self, optional<MemoryFormat> memory_format) {
  // Memory format support is a little tricky because vmap is allowed to move
  // around batch dimensions and some memory formats are rank-dependent.
  // Another weird case is:
  // - a tensor with MemoryFormat::ChannelsLast MUST have 4 dimensions. Do we
  //   allow the user to clone a Tensor with 3 logical dimensions and 1 batch
  //   dim into a ChannelsLast Tensor? What about a Tensor with 3 logical dims
  //   and N>1 batch dims?
  TORCH_CHECK(!memory_format.has_value() || memory_format == MemoryFormat::Preserve
      || memory_format == MemoryFormat::Contiguous,
      "NYI: Tensor.clone(memory_format) inside vmap is only supported with ",
      "memory_format torch.preserve_format or torch.contiguous_format (got ",
      *memory_format, ")");

  if (memory_format == MemoryFormat::Contiguous) {
    // There is an ambiguity here when the batch dims are not at the front of
    // the tensor.
    // >>> x = torch.randn(3, B0, 5)
    // >>> y = vmap(lambda x: x.clone(torch.contiguous_format), in_dims=1, out_dims=0)(x)
    // >>> y[0].is_contiguous()
    // ???
    // Should we make the whole tensor contiguous, or should we
    // make the non-batch dims contiguous? We've chosen the latter because
    // philosophically vmap hides the batch dims and operates on a per-sample level.
    auto physical_view = MultiBatchVmapTransform::logicalToPhysical(self);
    auto output_physical = at::clone(physical_view.tensor(), memory_format);
    return physical_view.getPhysicalToLogicalMap().apply(output_physical);
  }

  TORCH_INTERNAL_ASSERT(!memory_format.has_value() || memory_format == MemoryFormat::Preserve);
  auto* self_batched = unsafeGetBatchedImpl(self);
  auto output_physical = at::clone(self_batched->value(), memory_format);
  auto old_bdims = self_batched->bdims();
  return makeBatched(output_physical, BatchDims(old_bdims.begin(), old_bdims.end()));
}

// Note [Batching rules for matmul-like operators]
// at::matmul doesn't "de-expand" arguments to get better performance (maybe
// it should). In the batching rules for matmul-like operators (dot, mv, mm),
// we should be careful not to expand any unnecessary dimensions. e.g., if
// only one of the two arguments is a BatchedTensor, then we should try
// not to expand batch dimensions onto the other arg.
Tensor mv_batching_rule(const Tensor& self, const Tensor& other) {
  auto self_batched = isBatchedTensor(self);
  auto other_batched = isBatchedTensor(other);

  // A shape checking API would be nice...
  TORCH_CHECK(self.dim() == 2 && other.dim() == 1,
      "mv(self, other): Shape mismatch: expected matrix "
      "(got `self` of size ", self.sizes(), ") ",
      "and vector (got `other` of size ", other.sizes(), ")");

  // See Note [Batching rules for matmul-like operators] for why we have cases
  if (self_batched && !other_batched) {
    auto self_physical = MultiBatchVmapTransform::logicalToPhysical(self);
    auto result = at::matmul(self_physical.tensor(), other);
    return self_physical.getPhysicalToLogicalMap().apply(result);
  }
  if (!self_batched && other_batched) {
    // self_physical: [L, K], other_physical: [..., K]
    // We view the tensors as [L, K], [..., K, 1], perform matmul to get
    // a tensor of size [..., L, 1], and unsqueeze the last dim.
    auto other_physical = MultiBatchVmapTransform::logicalToPhysical(other);
    auto result = at::matmul(self, other_physical.tensor().unsqueeze(-1));
    return other_physical.getPhysicalToLogicalMap().apply(result.squeeze(-1));
  }
  if (self_batched && other_batched) {
    // self_physical: [..., L, K], other_physical: [..., K]
    // We view the tensors as [..., L, K], [..., K, 1], perform matmul to get
    // a tensor of size [..., L, 1], and unsqueeze the last dim.
    auto physical_args = MultiBatchVmapTransform::logicalToPhysical({self, other});
    auto result = at::matmul(
        physical_args[0].tensor(),
        physical_args[1].tensor().unsqueeze(-1));
    return physical_args[0].getPhysicalToLogicalMap().apply(result.squeeze(-1));
  }
  TORCH_INTERNAL_ASSERT(false, "either self or other must be a BatchedTensor");
}

Tensor dot_batching_rule(const Tensor& self, const Tensor& other) {
  auto self_batched = isBatchedTensor(self);
  auto other_batched = isBatchedTensor(other);

  TORCH_CHECK(/*logical*/self.dim() == 1 && /*logical*/other.dim() == 1,
      "dot(self, other): Shape mismatch: vector "
      "(got `self` of size ", self.sizes(), ") ",
      "and vector (got `other` of size ", other.sizes(), ")");

  // See Note [Batching rules for matmul-like operators] for why we have cases
  if (self_batched && !other_batched) {
    // self_physical: [..., K], other_physical: [K]
    // View the tensors as [..., 1, K] and [K], perform matmul, and unsqueeze.
    auto self_physical = MultiBatchVmapTransform::logicalToPhysical(self);
    auto result = at::matmul(self_physical.tensor().unsqueeze(-2), other);
    return self_physical.getPhysicalToLogicalMap().apply(result.squeeze(-1));
  }
  if (!self_batched && other_batched) {
    // self_physical: [K], other_physical: [..., K]
    // View the tensors as [K] and [..., K, 1], perform matmul, and unsqueeze.
    auto other_physical = MultiBatchVmapTransform::logicalToPhysical(other);
    auto result = at::matmul(self, other_physical.tensor().unsqueeze(-1));
    return other_physical.getPhysicalToLogicalMap().apply(result.squeeze(-1));
  }
  if (self_batched && other_batched) {
    // self_physical: [..., K], other_physical: [..., K]
    // View the tensors as [..., 1, K] and [..., K, 1], perform matmul, and unsqueeze.
    auto physical_args = MultiBatchVmapTransform::logicalToPhysical({self, other});
    auto result = at::matmul(
        physical_args[0].tensor().unsqueeze(-2),
        physical_args[1].tensor().unsqueeze(-1));
    return physical_args[0].getPhysicalToLogicalMap().apply(result.squeeze(-1).squeeze(-1));
  }
  TORCH_INTERNAL_ASSERT(false, "either self or other must be a BatchedTensor");
}

Tensor bmm_batching_rule(const Tensor& self, const Tensor& other) {
  TORCH_CHECK(/*logical*/self.dim() == 3 && /*logical*/other.dim() == 3,
      "bmm(self, other): Shape mismatch: expected 3D `self` "
      "(got `self` of size ", self.sizes(), ") ",
      "and 3D `other` (got `other` of size ", other.sizes(), ")");

  auto physical_args = BroadcastingVmapTransform::logicalToPhysical({self, other});
  auto result = at::matmul(physical_args[0].tensor(), physical_args[1].tensor());
  return physical_args[0].getPhysicalToLogicalMap().apply(result);
}

Tensor mm_batching_rule(const Tensor& self, const Tensor& other) {
  auto self_batched = isBatchedTensor(self);
  auto other_batched = isBatchedTensor(other);

  TORCH_CHECK(/*logical*/self.dim() == 2 && /*logical*/other.dim() == 2,
      "mm(self, other): Shape mismatch: expected matrix "
      "(got `self` of size ", self.sizes(), ") ",
      "and matrix (got `other` of size ", other.sizes(), ")");

  // See Note [Batching rules for matmul-like operators] for why we have cases
  if (self_batched && !other_batched) {
    auto self_physical = MultiBatchVmapTransform::logicalToPhysical(self);
    auto result = at::matmul(self_physical.tensor(), other);
    return self_physical.getPhysicalToLogicalMap().apply(result);
  }
  if (!self_batched && other_batched) {
    auto other_physical = MultiBatchVmapTransform::logicalToPhysical(other);
    auto result = at::matmul(self, other_physical.tensor());
    return other_physical.getPhysicalToLogicalMap().apply(result);
  }
  if (self_batched && other_batched) {
    auto physical_args = MultiBatchVmapTransform::logicalToPhysical({self, other});
    auto result = at::matmul(physical_args[0].tensor(), physical_args[1].tensor());
    return physical_args[0].getPhysicalToLogicalMap().apply(result.squeeze(-1).squeeze(-1));
  }
  TORCH_INTERNAL_ASSERT(false, "either self or other must be a BatchedTensor");
}

Tensor cat_batching_rule(TensorList tensors, int64_t dim) {
  auto physical_views = MultiBatchVmapTransform::logicalToPhysical(tensors);
  auto physical_tensors = fmap(
      physical_views, [](const VmapPhysicalView& view) -> Tensor { return view.tensor(); });
  TORCH_INTERNAL_ASSERT(
      tensors.size() > 0, "The dispatcher should not have dispatched here otherwise.");
  auto result = at::cat(physical_tensors, physical_views[0].getPhysicalDim(dim));
  return physical_views[0].getPhysicalToLogicalMap().apply(result);
}

Tensor stack_batching_rule(TensorList tensors, int64_t dim) {
  auto physical_views = MultiBatchVmapTransform::logicalToPhysical(tensors);
  auto physical_tensors = fmap(
      physical_views, [](const VmapPhysicalView& view) -> Tensor { return view.tensor(); });
  TORCH_INTERNAL_ASSERT(
      tensors.size() > 0, "The dispatcher should not have dispatched here otherwise.");
  // NB: stack wraps the dimensionality to (logical dim + 1), so we have to
  // manually handle that here.
  auto dim_physical =
      physical_views[0].numBatchDims() + maybe_wrap_dim(dim, /*logical*/tensors[0].dim() + 1);
  auto result = at::stack(physical_tensors, dim_physical);
  return physical_views[0].getPhysicalToLogicalMap().apply(result);
}

// I am quite sad that we need to register operators with exploded TensorOptions,
// even though the native:: implementations can use TensorOptions&.
// This also makes it hard to metaprogram: i.e., we can't use
// unwrap_and_call<..., at::to> because at::to takes TensorOptions& (!!)
Tensor to_dtype_layout_batching_rule(
    const Tensor& self,
    optional<ScalarType> dtype,
    optional<Layout> layout,
    optional<Device> device,
    optional<bool> pin_memory,
    bool non_blocking, bool copy,
    optional<MemoryFormat> memory_format) {
  auto options = TensorOptions()
    .dtype(dtype)
    .layout(layout)
    .device(device)
    .pinned_memory(pin_memory);
  auto* input_batched = unsafeGetBatchedImpl(self);
  auto output_physical = input_batched->value().to(options, non_blocking, copy, memory_format);
  auto old_bdims = input_batched->bdims();
  return makeBatched(output_physical, BatchDims(old_bdims.begin(), old_bdims.end()));
}

Tensor new_zeros_batching_rule(
    const Tensor& self,
    IntArrayRef size,
    optional<ScalarType> dtype,
    optional<Layout> layout,
    optional<Device> device,
    optional<bool> pin_memory) {
  auto physical_view = MultiBatchVmapTransform::logicalToPhysical(self);
  auto physical_size = physical_view.getPhysicalShape(size);
  auto options = TensorOptions()
    .dtype(dtype)
    .layout(layout)
    .device(device)
    .pinned_memory(pin_memory);
  auto result = physical_view.tensor().new_zeros(physical_size, options);
  return physical_view.getPhysicalToLogicalMap().apply(result);
}

Tensor new_empty_batching_rule(
    const Tensor& self,
    IntArrayRef size,
    c10::optional<ScalarType> dtype,
    c10::optional<Layout> layout,
    c10::optional<Device> device,
    c10::optional<bool> pin_memory) {
  auto physical_view = MultiBatchVmapTransform::logicalToPhysical(self);
  auto physical_size = physical_view.getPhysicalShape(size);
  auto result = physical_view.tensor().new_empty(physical_size, TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory));
  return physical_view.getPhysicalToLogicalMap().apply(result);
}

Tensor new_empty_strided_batching_rule(
    const Tensor& self,
    IntArrayRef size,
    IntArrayRef stride,
    optional<ScalarType> dtype,
    optional<Layout> layout,
    optional<Device> device,
    optional<bool> pin_memory) {
  auto physical_view = MultiBatchVmapTransform::logicalToPhysical(self);
  auto physical_size = physical_view.getPhysicalShape(size);

  // Let [B0, B1, B2] be the shape of the batch dims. We're going to create
  // the batch dimensions at the front of the tensor (in memory layout),
  // irrespective of whether or not they are actually at the front (in memory layout)
  // in the original `self` tensor. This is because when a user calls
  // `new_empty_strided` in general, the `strides` they provide are for a new
  // tensor and have no relation to the strides of the original tensor.
  //
  // So, the physical shape of the result should be ([B0, B1, B2] + size),
  // but what about the physical strides?
  //
  // We're actually free to pick whatever stride we want:
  // e.g., for size=[5, 3], stride=[0, 1], we could decide to
  // use
  // - physical size: [B0, B1, B2, 5, 3]
  // - physical stride: [9999*B1*B2, 9999*B2, 9999, 0, 1]
  //
  // Let's select some reasonable strides such that:
  // - The batch dims are "contiguous" with respect to each other
  // - if empty_strided(size, stride) would have created a contiguous Tensor,
  // then this new physical Tensor (with batch dims) is also contiguous
  //
  // Let S be the size of the storage if one were to construct a tensor
  // with `size` and `stride` via empty_strided(size, stride).
  // Then the physical sizes/strides should be:
  // - physical size: [B0, B1, B2, 5, 3]
  // - physical stride: [B1 * B2 * S, B2 * S, S, 0, 1]
  auto batch_shape = IntArrayRef(
      physical_view.tensor().sizes().begin(), physical_view.numBatchDims());

  // physical_strides = [B1 * B2 * S, B2 * S, S]
  auto physical_strides = at::detail::defaultStrides(batch_shape);
  TORCH_CHECK(size.size() == stride.size(),
        "new_empty_strided(sizes, strides): dimensionality of sizes (",
        size.size(), ") must match dimensionality of strides (",
        stride.size(), ")");
  auto storage_size = native::storage_size_for(size, stride);
  for (auto& physical_stride : physical_strides) {
    physical_stride *= storage_size;
  }

  // physical_strides = [B1 * B2 * S, B2 * S, S] + strides
  physical_strides.insert(physical_strides.end(), stride.begin(), stride.end());

  auto result = physical_view.tensor().new_empty_strided(
      physical_size, physical_strides, dtype, layout, device, pin_memory);
  return physical_view.getPhysicalToLogicalMap().apply(result);
}

template <typename F, F Func>
Tensor comparison_pointwise_batching_rule(const Tensor& self, const Tensor& other) {
  auto physical_args = BroadcastingVmapTransform::logicalToPhysical({self, other});
  auto result = Func(physical_args[0].tensor(), physical_args[1].tensor());
  return physical_args[0].getPhysicalToLogicalMap().apply(result);
}

TORCH_LIBRARY_IMPL(_, Batched, m) {
  m.fallback(torch::CppFunction::makeFromBoxedFunction<&batchedTensorForLoopFallback>());
}

TORCH_LIBRARY_IMPL(aten, Batched, m) {
  // NB: Ideally we would like some operators, like size.int, to "fallthrough"
  // to the underlying implementation. However, because a BatchedTensor is a
  // Tensor wrapper, it only has one dispatch key (Batched) on it. The resolution
  // here is to just directly call the underlying implementation.
  m.impl("size.int", static_cast<int64_t (*)(const Tensor&, int64_t)>(native::size));
  m.impl("_add_batch_dim", native::_add_batch_dim);
  m.impl("_remove_batch_dim", native::_remove_batch_dim);

  m.impl("sum.dim_IntList", sum_batching_rule);
  m.impl("is_complex", native::is_complex);

  // inplace operations
  m.impl("fill_.Scalar", fill_inplace_scalar_batching_rule);
  m.impl("fill_.Tensor", fill_inplace_tensor_batching_rule);
  m.impl("zero_", zero_inplace_batching_rule);

  // view operations
  m.impl("as_strided", as_strided_batching_rule);
  m.impl("chunk", chunk_batching_rule);
  m.impl("tensor_split.sections", tensor_split_sections_batching_rule);
  m.impl("tensor_split.indices", tensor_split_indices_batching_rule);
  m.impl("diagonal", diagonal_batching_rule);
  m.impl("expand", expand_batching_rule);
  m.impl("expand_as", native::expand_as); // composite wrt autograd
  m.impl("movedim.intlist", movedim_batching_rule);
  m.impl("movedim.int", static_cast<Tensor(*)(const Tensor&,int64_t,int64_t)>(native::movedim)); // composite wrt autograd
  // NB: static_cast because there's another variant of narrow. However, we don't
  // want to support the other variant yet bc it isn't documented...
  m.impl("narrow", static_cast<Tensor(*)(const Tensor&,int64_t,int64_t,int64_t)>(native::narrow)); // composite wrt autograd
  m.impl("numpy_T", native::numpy_T); // composite wrt autograd
  m.impl("permute", permute_batching_rule);
  m.impl("reshape", reshape_batching_rule);
  m.impl("reshape_as", native::reshape_as); // composite wrt autograd
  m.impl("select.int", select_batching_rule);
  m.impl("slice.Tensor", slice_batching_rule);
  m.impl("split.Tensor", split_batching_rule);
  m.impl("split_with_sizes", split_with_sizes_batching_rule);
  m.impl("squeeze", squeeze_batching_rule);
  m.impl("squeeze.dim", squeeze_dim_batching_rule);
  m.impl("t", native::t); // composite wrt autograd
  m.impl("trace", trace_batching_rule);
  m.impl("transpose.int", transpose_int_batching_rule);
  m.impl("unbind.int", unbind_batching_rule);
  m.impl("unfold", unfold_batching_rule);
  m.impl("unsqueeze", unsqueeze_batching_rule);
  m.impl("view", view_batching_rule);
  m.impl("view_as", native::view_as); // composite wrt autograd

  // clamp operations
  m.impl("clamp", clamp_batching_rule);
  m.impl("clamp_min", clamp_min_batching_rule);
  m.impl("clamp_max", clamp_max_batching_rule);

  // unary pointwise, out-of-place, no additional arguments.
#define UNARY_POINTWISE(op) m.impl(#op, \
    unwrap_and_call<Tensor (*)(const Tensor&), at::op>);
  UNARY_POINTWISE(abs);
  UNARY_POINTWISE(acos);
  UNARY_POINTWISE(asin);
  UNARY_POINTWISE(atan);
  UNARY_POINTWISE(ceil);
  UNARY_POINTWISE(cos);
  UNARY_POINTWISE(cosh);
  UNARY_POINTWISE(conj_physical);
  UNARY_POINTWISE(digamma);
  UNARY_POINTWISE(exp);
  UNARY_POINTWISE(expm1);
  UNARY_POINTWISE(floor);
  UNARY_POINTWISE(frac);
  UNARY_POINTWISE(lgamma);
  UNARY_POINTWISE(log);
  UNARY_POINTWISE(log10);
  UNARY_POINTWISE(log1p);
  UNARY_POINTWISE(log2);
  UNARY_POINTWISE(neg);
  UNARY_POINTWISE(reciprocal);
  UNARY_POINTWISE(relu);
  UNARY_POINTWISE(round);
  UNARY_POINTWISE(rsqrt);
  UNARY_POINTWISE(sigmoid);
  UNARY_POINTWISE(sign);
  UNARY_POINTWISE(sin);
  UNARY_POINTWISE(sinh);
  UNARY_POINTWISE(sqrt);
  UNARY_POINTWISE(tan);
  UNARY_POINTWISE(tanh);
  UNARY_POINTWISE(trunc);
#undef UNARY_POINTWISE
#define TO_BATCHING_RULE(name, ...) \
  { \
    using to_type = Tensor(Tensor::*)(__VA_ARGS__) const; \
    m.impl(name, unwrap_and_call_method< \
        to_type, &Tensor::to, __VA_ARGS__>);\
  }
  TO_BATCHING_RULE("to.device", Device, ScalarType, bool, bool, optional<MemoryFormat>)
  TO_BATCHING_RULE("to.dtype", ScalarType, bool, bool, optional<MemoryFormat>)
  TO_BATCHING_RULE("to.other", const Tensor&, bool, bool, optional<MemoryFormat>)
  m.impl("to.dtype_layout", to_dtype_layout_batching_rule);
#undef TO_BATCHING_RULE
  m.impl("clone", clone_batching_rule);

  using TensorTensorScalarType = Tensor (*)(const Tensor&, const Tensor&, const Scalar&);
  using TensorTensorType = Tensor (*)(const Tensor&, const Tensor&);
  using TensorScalarType = Tensor (*)(const Tensor&, const Scalar&);

#define BINARY_POINTWISE(op) \
  m.impl(#op".Tensor", binary_pointwise_batching_rule<TensorTensorType, at::op>); \
  m.impl(#op".Scalar", unwrap_and_call<TensorScalarType, at::op, const Scalar&>);
#define BINARY_POINTWISE_VA(op, ...) \
  { \
    using Binop = Tensor (*)(const Tensor&, const Tensor&, __VA_ARGS__); \
    using Unop = Tensor (*)(const Tensor&, const Scalar&, __VA_ARGS__); \
    m.impl(#op".Tensor", binary_pointwise_batching_rule<Binop, at::op, __VA_ARGS__>); \
    m.impl(#op".Scalar", unwrap_and_call<Unop, at::op, const Scalar&, __VA_ARGS__>); \
  }

  BINARY_POINTWISE_VA(add, const Scalar&);
  BINARY_POINTWISE_VA(sub, const Scalar&);
  BINARY_POINTWISE_VA(rsub, const Scalar&);
  BINARY_POINTWISE(mul);
  BINARY_POINTWISE(div);
  {
    using Binop = Tensor (*)(const Tensor&, const Tensor&, c10::optional<c10::string_view>);
    using Unop = Tensor (*)(const Tensor&, const Scalar&, c10::optional<c10::string_view>);
    m.impl("div.Tensor_mode", binary_pointwise_batching_rule<Binop, at::div, c10::optional<c10::string_view>>);
    m.impl("div.Scalar_mode", unwrap_and_call<Unop, at::div, const Scalar&, c10::optional<c10::string_view>>);
  }

  // at::pow has three out-of-place overloads
  m.impl("pow.Tensor_Tensor", binary_pointwise_batching_rule<TensorTensorType, at::pow>);
  m.impl("pow.Tensor_Scalar", unwrap_and_call<TensorScalarType, at::pow, const Scalar&>);
  m.impl("pow.Scalar", pow_scalar_Tensor_batching_rule);

  m.impl("sigmoid_backward", binary_pointwise_batching_rule<TensorTensorType, at::sigmoid_backward>);
  m.impl(
      "threshold_backward",
      binary_pointwise_batching_rule<
          TensorTensorScalarType,
          at::threshold_backward,
          const Scalar&>);

  // for at::result_type, call the native::result_type implementation.
  // We don't have to do anything special because native::result_type operates
  // on the logical shape of the tensors.
  m.impl("result_type.Tensor", static_cast<ScalarType (*)(const Tensor&, const Tensor&)>(native::result_type));
  m.impl("result_type.Scalar", static_cast<ScalarType (*)(const Tensor&, const Scalar&)>(native::result_type));
  m.impl("result_type.Scalar_Tensor", static_cast<ScalarType (*)(const Scalar&, const Tensor&)>(native::result_type));
  m.impl("result_type.Scalar_Scalar", static_cast<ScalarType (*)(const Scalar&, const Scalar&)>(native::result_type));

#undef BINARY_POINTWISE_VA
#undef BINARY_POINTWISE


#define TRIVIAL_OP(op) m.impl(#op, \
    unwrap_and_call<Tensor (*)(const Tensor&), at::op>);
  // complex number view operators
  TRIVIAL_OP(imag)
  TRIVIAL_OP(real);
  TRIVIAL_OP(view_as_real);
  TRIVIAL_OP(conj);
  TRIVIAL_OP(_conj);
  TRIVIAL_OP(resolve_conj);
  TRIVIAL_OP(resolve_neg);
  m.impl("view_as_complex", view_as_complex_batching_rule);
#undef TRIVIAL

  // matmul-like operators
  m.impl("mv", mv_batching_rule);
  m.impl("dot", dot_batching_rule);
  m.impl("bmm", bmm_batching_rule);
  m.impl("mm", mm_batching_rule);

  // cat/stack
  m.impl("cat", cat_batching_rule);
  m.impl("stack", stack_batching_rule);

  // backward operators
  m.impl("select_backward", select_backward_batching_rule);
  m.impl("slice_backward", slice_backward_batching_rule);
  m.impl("trace_backward", trace_backward_batching_rule);
  m.impl("diagonal_backward", diagonal_backward_batching_rule);

  // Tensor.new_* operators
  m.impl("new_empty", new_empty_batching_rule);
  m.impl("new_empty_strided", new_empty_strided_batching_rule);
  m.impl("new_zeros", new_zeros_batching_rule);

  m.impl("contiguous", contiguous_batching_rule);

  // Comparison ops
#define COMPARISON_POINTWISE(op) \
  m.impl(#op".Tensor", comparison_pointwise_batching_rule<TensorTensorType, at::op>); \
  m.impl(#op".Scalar", unwrap_and_call<TensorScalarType, at::op, const Scalar&>);

  COMPARISON_POINTWISE(eq);
  COMPARISON_POINTWISE(gt);
  COMPARISON_POINTWISE(ge);
  COMPARISON_POINTWISE(le);
  COMPARISON_POINTWISE(lt);
  COMPARISON_POINTWISE(ne);

#undef COMPARISON_POINTWISE
}

} // namespace at
