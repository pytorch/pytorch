#include <torch/library.h>
#include <ATen/VmapTransforms.h>
#include <ATen/BatchedFallback.h>
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
// to generate a slow fallback for the batching rule (this is not yet implemented).

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

Tensor sum_batching_rule(const Tensor& self, IntArrayRef dims, bool keepdim, optional<ScalarType> dtype) {
  auto self_physical = MultiBatchVmapTransform::logicalToPhysical(self);
  auto dims_physical = self_physical.getPhysicalDims(dims);
  auto result = at::sum(self_physical.tensor(), dims_physical, keepdim, dtype);
  return self_physical.newLogicalFromPhysical(result);
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
    return physical_args[0].newLogicalFromPhysical(result);
  }
  if (isPhysicalScalarTensor(self)) {
    auto other_physical = MultiBatchVmapTransform::logicalToPhysical(other);
    auto result = Func(self, other_physical.tensor(), args...);
    return other_physical.newLogicalFromPhysical(result);
  }
  if (isPhysicalScalarTensor(other)) {
    auto self_physical = MultiBatchVmapTransform::logicalToPhysical(self);
    auto result = Func(self_physical.tensor(), other, args...);
    return self_physical.newLogicalFromPhysical(result);
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
  return physical_args[0].newLogicalFromPhysical(result);
}

Tensor expand_batching_rule(const Tensor& self, IntArrayRef size, bool implicit) {
  auto self_physical = MultiBatchVmapTransform::logicalToPhysical(self);
  auto size_physical = self_physical.getPhysicalShape(size);
  auto self_physical_dim = self_physical.tensor().dim();

  TORCH_CHECK(self_physical_dim <= size_physical.size(),
       "expand: the number of sizes provided (", /*logical*/size.size(), ") ",
       "must be greater or equal to the number of dimensions in the tensor (",
       /*logical dim*/self.dim(), ")");

  if (self_physical_dim == size_physical.size()) {
    auto result = self_physical.tensor().expand(size_physical, implicit);
    return self_physical.newLogicalFromPhysical(result);
  }

  TORCH_INTERNAL_ASSERT(self_physical_dim < size_physical.size());
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
  return self_physical.newLogicalFromPhysical(result);
}

std::vector<Tensor> chunk_batching_rule(const Tensor& self, int64_t chunks, int64_t dim) {
  auto self_physical = MultiBatchVmapTransform::logicalToPhysical(self);
  auto dim_physical = self_physical.getPhysicalDim(dim);
  auto result = at::chunk(self_physical.tensor(), chunks, dim_physical);
  self_physical.makeLogicalFromPhysicalListInplace(result);
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
  return self_physical.newLogicalFromPhysical(result);
}

Tensor squeeze_dim_batching_rule(const Tensor& self, int64_t dim) {
  auto self_physical = MultiBatchVmapTransform::logicalToPhysical(self);
  auto dim_physical = self_physical.getPhysicalDim(dim);
  auto result = self_physical.tensor().squeeze(dim_physical);
  return self_physical.newLogicalFromPhysical(result);
}

Tensor transpose_int_batching_rule(const Tensor& self, int64_t dim0, int64_t dim1) {
  auto self_physical = MultiBatchVmapTransform::logicalToPhysical(self);
  auto dim0_physical = self_physical.getPhysicalDim(dim0);
  auto dim1_physical = self_physical.getPhysicalDim(dim1);
  auto result = self_physical.tensor().transpose(dim0_physical, dim1_physical);
  return self_physical.newLogicalFromPhysical(result);
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
  return self_physical.newLogicalFromPhysical(result);
}

Tensor select_batching_rule(const Tensor& self, int64_t dim, int64_t index) {
  auto self_physical = MultiBatchVmapTransform::logicalToPhysical(self);
  auto dim_physical = self_physical.getPhysicalDim(dim);
  auto result = self_physical.tensor().select(dim_physical, index);
  return self_physical.newLogicalFromPhysical(result);
}

Tensor slice_batching_rule(const Tensor& self, int64_t dim, int64_t start, int64_t end, int64_t step) {
  auto self_physical = MultiBatchVmapTransform::logicalToPhysical(self);
  auto dim_physical = self_physical.getPhysicalDim(dim);
  auto result = self_physical.tensor().slice(dim_physical, start, end, step);
  return self_physical.newLogicalFromPhysical(result);
}

Tensor diagonal_batching_rule(const Tensor& self, int64_t offset, int64_t dim1, int64_t dim2) {
  auto self_physical = MultiBatchVmapTransform::logicalToPhysical(self);
  auto dim1_physical = self_physical.getPhysicalDim(dim1);
  auto dim2_physical = self_physical.getPhysicalDim(dim2);
  auto result = at::diagonal(self_physical.tensor(), offset, dim1_physical, dim2_physical);
  return self_physical.newLogicalFromPhysical(result);
}

Tensor movedim_batching_rule(const Tensor& self, IntArrayRef source, IntArrayRef destination) {
  auto self_physical = MultiBatchVmapTransform::logicalToPhysical(self);
  auto source_physical = self_physical.getPhysicalDims(source);
  auto destination_physical = self_physical.getPhysicalDims(destination);
  auto result = at::movedim(self_physical.tensor(), source_physical, destination_physical);
  return self_physical.newLogicalFromPhysical(result);
}

Tensor reshape_batching_rule(const Tensor& self, IntArrayRef shape) {
  auto self_physical = MultiBatchVmapTransform::logicalToPhysical(self);
  auto shape_physical = self_physical.getPhysicalShape(shape);
  auto result = self_physical.tensor().reshape(shape_physical);
  return self_physical.newLogicalFromPhysical(result);
}

std::vector<Tensor> split_batching_rule(const Tensor& self, int64_t split_size, int64_t dim) {
  auto self_physical = MultiBatchVmapTransform::logicalToPhysical(self);
  auto dim_physical = self_physical.getPhysicalDim(dim);
  auto result = at::split(self_physical.tensor(), split_size, dim_physical);
  self_physical.makeLogicalFromPhysicalListInplace(result);
  return result;
}

std::vector<Tensor> split_with_sizes_batching_rule(const Tensor& self, IntArrayRef split_sizes, int64_t dim) {
  auto self_physical = MultiBatchVmapTransform::logicalToPhysical(self);
  auto dim_physical = self_physical.getPhysicalDim(dim);
  auto result = at::split_with_sizes(self_physical.tensor(), split_sizes, dim_physical);
  self_physical.makeLogicalFromPhysicalListInplace(result);
  return result;
}

std::vector<Tensor> unbind_batching_rule(const Tensor& self, int64_t dim) {
  auto self_physical = MultiBatchVmapTransform::logicalToPhysical(self);
  auto dim_physical = self_physical.getPhysicalDim(dim);
  auto result = at::unbind(self_physical.tensor(), dim_physical);
  self_physical.makeLogicalFromPhysicalListInplace(result);
  return result;
}

Tensor unfold_batching_rule(const Tensor& self, int64_t dim, int64_t size, int64_t step) {
  auto self_physical = MultiBatchVmapTransform::logicalToPhysical(self);
  auto dim_physical = self_physical.getPhysicalDim(dim);
  auto result = self_physical.tensor().unfold(dim_physical, size, step);
  return self_physical.newLogicalFromPhysical(result);
}

Tensor view_batching_rule(const Tensor& self, IntArrayRef size) {
  auto self_physical = MultiBatchVmapTransform::logicalToPhysical(self);
  auto size_physical = self_physical.getPhysicalShape(size);
  auto result = self_physical.tensor().view(size_physical);
  return self_physical.newLogicalFromPhysical(result);
}

template <typename F, F Func, typename... ExtraArgs>
Tensor unary_pointwise_batching_rule(const Tensor& input, ExtraArgs... args) {
  auto* input_batched = unsafeGetBatchedImpl(input);
  auto output_physical = Func(input_batched->value(), args...);
  auto old_bdims = input_batched->bdims();
  return makeBatched(output_physical, BatchDims(old_bdims.begin(), old_bdims.end()));
}

template <typename F, F Func, typename... ExtraArgs>
Tensor unary_pointwise_method_batching_rule(const Tensor& input, ExtraArgs... extra_args) {
  auto* input_batched = unsafeGetBatchedImpl(input);
  auto output_physical = (input_batched->value().*Func)(extra_args...);
  auto old_bdims = input_batched->bdims();
  return makeBatched(output_physical, BatchDims(old_bdims.begin(), old_bdims.end()));
}

Tensor pow_scalar_Tensor_batching_rule(Scalar other, const Tensor& self) {
  auto* self_batched = unsafeGetBatchedImpl(self);
  auto output_physical = at::pow(other, self_batched->value());
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
    return self_physical.newLogicalFromPhysical(result);
  }
  if (!self_batched && other_batched) {
    // self_physical: [L, K], other_physical: [..., K]
    // We view the tensors as [L, K], [..., K, 1], perform matmul to get
    // a tensor of size [..., L, 1], and unsqueeze the last dim.
    auto other_physical = MultiBatchVmapTransform::logicalToPhysical(other);
    auto result = at::matmul(self, other_physical.tensor().unsqueeze(-1));
    return other_physical.newLogicalFromPhysical(result.squeeze(-1));
  }
  if (self_batched && other_batched) {
    // self_physical: [..., L, K], other_physical: [..., K]
    // We view the tensors as [..., L, K], [..., K, 1], perform matmul to get
    // a tensor of size [..., L, 1], and unsqueeze the last dim.
    auto physical_args = MultiBatchVmapTransform::logicalToPhysical({self, other});
    auto result = at::matmul(
        physical_args[0].tensor(),
        physical_args[1].tensor().unsqueeze(-1));
    return physical_args[0].newLogicalFromPhysical(result.squeeze(-1));
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
    return self_physical.newLogicalFromPhysical(result.squeeze(-1));
  }
  if (!self_batched && other_batched) {
    // self_physical: [K], other_physical: [..., K]
    // View the tensors as [K] and [..., K, 1], perform matmul, and unsqueeze.
    auto other_physical = MultiBatchVmapTransform::logicalToPhysical(other);
    auto result = at::matmul(self, other_physical.tensor().unsqueeze(-1));
    return other_physical.newLogicalFromPhysical(result.squeeze(-1));
  }
  if (self_batched && other_batched) {
    // self_physical: [..., K], other_physical: [..., K]
    // View the tensors as [..., 1, K] and [..., K, 1], perform matmul, and unsqueeze.
    auto physical_args = MultiBatchVmapTransform::logicalToPhysical({self, other});
    auto result = at::matmul(
        physical_args[0].tensor().unsqueeze(-2),
        physical_args[1].tensor().unsqueeze(-1));
    return physical_args[0].newLogicalFromPhysical(result.squeeze(-1).squeeze(-1));
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
  return physical_args[0].newLogicalFromPhysical(result);
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
    return self_physical.newLogicalFromPhysical(result);
  }
  if (!self_batched && other_batched) {
    auto other_physical = MultiBatchVmapTransform::logicalToPhysical(other);
    auto result = at::matmul(self, other_physical.tensor());
    return other_physical.newLogicalFromPhysical(result);
  }
  if (self_batched && other_batched) {
    auto physical_args = MultiBatchVmapTransform::logicalToPhysical({self, other});
    auto result = at::matmul(physical_args[0].tensor(), physical_args[1].tensor());
    return physical_args[0].newLogicalFromPhysical(result.squeeze(-1).squeeze(-1));
  }
  TORCH_INTERNAL_ASSERT(false, "either self or other must be a BatchedTensor");
}

// I am quite sad that we need to register operators with exploded TensorOptions,
// even though the native:: implementations can use TensorOptions&.
// This also makes it hard to metaprogram: i.e., we can't use
// unary_pointwise_batching_rule<..., at::to> because at::to takes TensorOptions& (!!)
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

  m.impl_UNBOXED("sum.dim_IntList", sum_batching_rule);

  // view operations
  m.impl("chunk", chunk_batching_rule);
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
  m.impl("squeeze.dim", squeeze_dim_batching_rule);
  m.impl("t", native::t); // composite wrt autograd
  m.impl("transpose.int", transpose_int_batching_rule);
  m.impl("unbind.int", unbind_batching_rule);
  m.impl("unfold", unfold_batching_rule);
  m.impl("unsqueeze", unsqueeze_batching_rule);
  m.impl("view", view_batching_rule);
  m.impl("view_as", native::view_as); // composite wrt autograd

  // unary pointwise, out-of-place, no additional arguments.
#define UNARY_POINTWISE(op) m.impl(#op, \
    unary_pointwise_batching_rule<Tensor (*)(const Tensor&), at::op>);
  UNARY_POINTWISE(abs);
  UNARY_POINTWISE(acos);
  UNARY_POINTWISE(asin);
  UNARY_POINTWISE(atan);
  UNARY_POINTWISE(ceil);
  UNARY_POINTWISE(cos);
  UNARY_POINTWISE(cosh);
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
    m.impl(name, unary_pointwise_method_batching_rule< \
        to_type, &Tensor::to, __VA_ARGS__>);\
  }
  TO_BATCHING_RULE("to.device", Device, ScalarType, bool, bool, optional<MemoryFormat>)
  TO_BATCHING_RULE("to.dtype", ScalarType, bool, bool, optional<MemoryFormat>)
  TO_BATCHING_RULE("to.other", const Tensor&, bool, bool, optional<MemoryFormat>)
  m.impl("to.dtype_layout", to_dtype_layout_batching_rule);
#undef TO_BATCHING_RULE

  using TensorTensorType = Tensor (*)(const Tensor&, const Tensor&);
  using TensorScalarType = Tensor (*)(const Tensor&, Scalar);

#define BINARY_POINTWISE(op) \
  m.impl(#op".Tensor", binary_pointwise_batching_rule<TensorTensorType, at::op>); \
  m.impl(#op".Scalar", unary_pointwise_batching_rule<TensorScalarType, at::op, Scalar>);
#define BINARY_POINTWISE_VA(op, ...) \
  { \
    using Binop = Tensor (*)(const Tensor&, const Tensor&, __VA_ARGS__); \
    using Unop = Tensor (*)(const Tensor&, Scalar, __VA_ARGS__); \
    m.impl(#op".Tensor", binary_pointwise_batching_rule<Binop, at::op, __VA_ARGS__>); \
    m.impl(#op".Scalar", unary_pointwise_batching_rule<Unop, at::op, Scalar, __VA_ARGS__>); \
  }

  BINARY_POINTWISE_VA(add, Scalar);
  BINARY_POINTWISE_VA(sub, Scalar);
  BINARY_POINTWISE_VA(rsub, Scalar);
  BINARY_POINTWISE(mul);
  BINARY_POINTWISE(div);

  // at::pow has three out-of-place overloads
  m.impl("pow.Tensor_Tensor", binary_pointwise_batching_rule<TensorTensorType, at::pow>);
  m.impl("pow.Tensor_Scalar", unary_pointwise_batching_rule<TensorScalarType, at::pow, Scalar>);
  m.impl("pow.Scalar", pow_scalar_Tensor_batching_rule);

  m.impl("sigmoid_backward", binary_pointwise_batching_rule<TensorTensorType, at::sigmoid_backward>);

  // for at::result_type, call the native::result_type implementation.
  // We don't have to do anything special because native::result_type operates
  // on the logical shape of the tensors.
  m.impl("result_type.Tensor", static_cast<ScalarType (*)(const Tensor&, const Tensor&)>(native::result_type));
  m.impl("result_type.Scalar", static_cast<ScalarType (*)(const Tensor&, Scalar)>(native::result_type));
  m.impl("result_type.Scalar_Tensor", static_cast<ScalarType (*)(Scalar, const Tensor&)>(native::result_type));
  m.impl("result_type.Scalar_Scalar", static_cast<ScalarType (*)(Scalar, Scalar)>(native::result_type));

#undef BINARY_POINTWISE_VA
#undef BINARY_POINTWISE

  // matmul-like operators
  m.impl("mv", mv_batching_rule);
  m.impl("dot", dot_batching_rule);
  m.impl("bmm", bmm_batching_rule);
  m.impl("mm", mm_batching_rule);
}

} // namespace at
