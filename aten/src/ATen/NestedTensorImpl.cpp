#include <ATen/ATen.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/core/op_registration/op_registration.h>
#include <ATen/NestedTensorImpl.h>
#include <c10/core/DispatchKey.h>

namespace at {
namespace native {

inline std::vector<int64_t> construct_opt_sizes(const at::Tensor& sizes) {
  if (sizes.dim() == 0) {
    return std::vector<int64_t>();
  }
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(sizes.dim() == 2);
  std::vector<int64_t> result(1, sizes.sizes()[0]);
  if (sizes.dim() > 0) {
    size_t nested_dim = result.size();
    int64_t* sizes_ptr = sizes.data_ptr<int64_t>();
    result.resize(nested_dim + sizes.sizes()[1]);
    int64_t sizes_size_0 = sizes.sizes()[0];
    int64_t sizes_size_1 = sizes.sizes()[1];
    for (const auto i : c10::irange(sizes_size_1)) {
      result[nested_dim + i] = sizes_ptr[i];
    }
    for (const auto j : c10::irange(sizes_size_1)) {
      for (const auto i : c10::irange(sizes_size_0)) {
        if (result[nested_dim + j] &&
            (result[nested_dim + j] != sizes_ptr[i * sizes.size(1) + j])) {
          result[nested_dim + j] = -1;
        }
      }
    }
  }
  return result;
}

// assume contiguous, we can construct stride from size
inline at::Tensor construct_nested_stride_tensor(const at::Tensor& sizes) {
  // empty `sizes` means empty nested tensor, so return empty strides
  if (sizes.dim() == 0) {
    return sizes;
  }
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(sizes.dim() == 2);
  int64_t orig_dim = sizes.size(1);
  // `sizes`.sizes() = ntensors x 0 means empty but shaped `sizes`
  // in this case strides is also empty but shaped
  if (orig_dim == 0) {
    return sizes;
  }
  at::Tensor strides = sizes.new_empty(sizes.sizes());
  const int64_t* sizes_ptr = sizes.data_ptr<int64_t>();
  int64_t* strides_ptr = strides.data_ptr<int64_t>();
  for (int64_t i = 0; i < sizes.size(0); i++) {
    strides_ptr[orig_dim - 1] = 1;
    int64_t product = sizes_ptr[orig_dim - 1];
    for (int64_t j = orig_dim - 2; j >= 0; j--) {
      strides_ptr[j] = product;
      product *= sizes_ptr[j];
    }
    sizes_ptr += orig_dim;
    strides_ptr += orig_dim;
  }
  return strides;
}

NestedTensorImpl::NestedTensorImpl(
    at::Tensor buffer,
    at::Tensor nested_size_tensor)
    : TensorImpl(
          (c10::DispatchKeySet(DispatchKey::NestedTensor) |
           c10::DispatchKeySet(buffer.is_cuda() ? BackendComponent::CUDABit : BackendComponent::CPUBit)),
          buffer.dtype(),
          buffer.device()),
      buffer_(std::move(buffer)),
      nested_size_tensor_(std::move(nested_size_tensor)),
      nested_stride_tensor_(construct_nested_stride_tensor(nested_size_tensor_)),
      opt_sizes_(construct_opt_sizes(nested_size_tensor_))
{
  TORCH_WARN_ONCE(
      "The PyTorch API of nested tensors is in prototype stage and will change "
      "in the near future.");
  TORCH_INTERNAL_ASSERT(buffer_.is_cuda() || buffer_.is_cpu(), "NestedTensorImpl buffer must be either CUDA or CPU but got ", buffer_);
  TORCH_INTERNAL_ASSERT(nested_size_tensor_.is_contiguous());
  int64_t size_dim = nested_size_tensor_.dim();
  TORCH_INTERNAL_ASSERT(size_dim == 0 || size_dim == 2);
  refresh_dim();
  set_sizes_strides_policy(c10::TensorImpl::SizesStridesPolicy::CustomSizes);
}

void NestedTensorImpl::refresh_dim() {
  const auto my_dim = nested_size_tensor_.dim() ? nested_size_tensor_.sizes()[1] + 1 : 1;
  sizes_and_strides_.resize(my_dim);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(dim() == my_dim);
}

int64_t NestedTensorImpl::dim_custom() const {
  return dim_default();
}
int64_t NestedTensorImpl::numel_custom() const {
  TORCH_CHECK(false, "numel is disabled.");
}
bool NestedTensorImpl::is_contiguous_custom(MemoryFormat) const {
  TORCH_CHECK(false, "is_contiguous is disabled.");
}
IntArrayRef NestedTensorImpl::sizes_custom() const {
  TORCH_CHECK(false, "Internal error: NestedTensorImpl doesn't support sizes. Please file an issue on https://github.com/pytorch/nestedtensor");
}

IntArrayRef NestedTensorImpl::strides_custom() const {
  TORCH_CHECK(false, "Internal error: NestedTensorImpl doesn't support strides. Please file an issue on https://github.com/pytorch/nestedtensor");
}

const char* NestedTensorImpl::tensorimpl_type_name() const {
  return "NestedTensorImpl";
}

// implicit batch dimension index offsets, original tensors shapes
std::tuple<std::vector<int64_t>, std::vector<IntArrayRef>>
NestedTensorImpl::get_offsets_and_shapes() const {
  // unbinding empty nested tensor should have returned before calling this function
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(!opt_sizes_.empty());
  const int64_t& ntensors = opt_sizes_[0];
  std::vector<int64_t> offsets(ntensors + 1);
  std::vector<IntArrayRef> shapes(ntensors);
  int64_t orig_dim = nested_size_tensor_.size(1);
  // nesting scalars means empty `nested_size_tensor_` and `nested_stride_tensor_`
  // `nested_size_tensor_`.sizes() = `nested_stride_tensor_`.sizes() = ntensors x 0
  if (orig_dim == 0) {
    std::iota(offsets.begin(), offsets.end(), 0);
    return std::make_tuple(offsets, shapes);
  }
  const int64_t* sizemat_ptr = nested_size_tensor_.data_ptr<int64_t>();
  const int64_t* stridemat_ptr = nested_stride_tensor_.data_ptr<int64_t>();
  offsets[0] = 0;
  for (int64_t i = 0; i < ntensors - 1; i++) {
    offsets[i + 1] = offsets[i] + *sizemat_ptr * *stridemat_ptr;
    shapes[i] = IntArrayRef(sizemat_ptr, sizemat_ptr + orig_dim);
    sizemat_ptr += orig_dim;
    stridemat_ptr += orig_dim;
  }
  offsets.back() = buffer_.numel();
  shapes.back() = IntArrayRef(sizemat_ptr, sizemat_ptr + orig_dim);
  return std::make_tuple(offsets, shapes);
}

} // namespace native
} // namespace at
