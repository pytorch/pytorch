#include <ATen/NestedTensorImpl.h>
#include <ATen/native/nested/NestedTensorUtils.h>
#include <ATen/native/TensorShape.h>
#include <c10/util/Optional.h>

namespace at {
namespace native {

/**
 * Thin wrapper around get_nested_size_tensor that is registered as a native function
 *
 * @return The nested tensors' size tensor.
 */
at::Tensor _nested_tensor_size(const at::Tensor& self) {
  return get_nested_size_tensor(self);
}

at::Tensor _nested_tensor_strides(const at::Tensor& self){
  return  get_nested_tensor_impl(self) -> get_nested_stride_tensor();
}
std::vector<int64_t> _nested_tensor_offsets(const at::Tensor& self){
  return get_nested_tensor_impl(self) -> get_offsets();
}

// Helper functions for getting information about a nested tensor's shape.
std::vector<int64_t> NestedTensor_get_max_size_from_size_tensor(
    const Tensor& sizes) {
  if (sizes.dim() == 0) {
    return {};
  }
  const auto sizes_ptr = sizes.data_ptr<int64_t>();
  const auto sizes_size_0 = sizes.sizes()[0];
  const auto sizes_size_1 = sizes.sizes()[1];
  TORCH_INTERNAL_ASSERT(sizes_size_1 > 0);
  std::vector<int64_t> results(sizes_size_1, 0);
  for (const auto ii : c10::irange(sizes_size_0)) {
    for (const auto jj : c10::irange(sizes_size_1)) {
      auto val = sizes_ptr[ii * sizes_size_1 + jj];
      if (results[jj] < val) {
        results[jj] = val;
      }
    }
  }
  return results;
}

std::vector<int64_t> NestedTensor_get_max_size(const NestedTensorImpl& nt) {
  return NestedTensor_get_max_size_from_size_tensor(
      nt.get_nested_size_tensor());
}

int64_t get_consistent_last_dim_of_nested_tensor(const NestedTensorImpl& nt) {
  c10::optional<int64_t> last_dim = nt.opt_size(-1);
  TORCH_CHECK(
      last_dim != c10::nullopt,
      "Expected all tensors in nested tensor to have the same trailing dimension, instead last dimension equals: ",
      nt.get_nested_size_tensor().select(1, -1));
  return *last_dim;
}

std::vector<Tensor> chunk_nested_tensor(const Tensor& self, int64_t chunks, int64_t dim) {
  int64_t ndim = self.dim();
  if (ndim == 0) {
    TORCH_CHECK_INDEX(false, "chunk() cannot be applied to a 0-dim tensor.");
  }
  dim = maybe_wrap_dim(dim, ndim);
  TORCH_CHECK(self.dim() - 1 == dim,
           "Chunk for nested tensors is currently only supported for the last dimension.");
  TORCH_CHECK(chunks > 0,"chunk expects `chunks` to be greater than 0, got: ", chunks);
  TORCH_CHECK(self.is_contiguous(), "chunk expects `self` to be contiguous.");
  auto self_impl = get_nested_tensor_impl(self);
  auto last_dim = get_consistent_last_dim_of_nested_tensor(*self_impl);
  TORCH_CHECK(last_dim % chunks == 0,
           "Chunk for nested tensors is only supported for nested tensors with trailing dimension divisible by chunks, got: ",
           last_dim, " % ", chunks, " != 0");
  std::vector<Tensor> buffers = get_buffer(self).view({-1, last_dim}).chunk(chunks, -1);
  int64_t new_last_size = buffers[0].size(-1);
  for(auto& t : buffers) {
    t = t.contiguous().view({-1});
  }
  std::vector<Tensor> results;
  for (auto& buffer : buffers) {
    auto nt_chunk_size = self_impl->get_nested_size_tensor().clone().index_put_({at::indexing::Slice(), -1}, new_last_size);
    results.push_back(wrap_buffer(buffer, nt_chunk_size));
  }
  return results;
}

std::vector<IntArrayRef> NestedTensor_get_sizes(
    const NestedTensorImpl* self_ptr) {
  int64_t ntensors = self_ptr->size(0);
  std::vector<IntArrayRef> sizes(ntensors);
  if (ntensors == 0) {
    return sizes;
  }
  const Tensor& sizemat = self_ptr->get_nested_size_tensor();
  int64_t orig_dim = sizemat.size(1);
  // nesting scalars has empty sizes
  if (orig_dim == 0) {
    return sizes;
  }
  const int64_t* sizemat_ptr = sizemat.data_ptr<int64_t>();

  for (const auto i : c10::irange(ntensors)) {
    sizes[i] = IntArrayRef(sizemat_ptr, sizemat_ptr + orig_dim);
    sizemat_ptr += orig_dim;
  }
  return sizes;
}
Tensor slice_nested(
    const Tensor& self,
    int64_t dim,
    c10::optional<int64_t> start,
    c10::optional<int64_t> end,
    int64_t step) {
  int64_t ndim = self.dim();
  if (ndim == 0) {
    TORCH_CHECK_INDEX(false, "slice() cannot be applied to a 0-dim tensor.");
  }
  dim = maybe_wrap_dim(dim, ndim);

  // will error if the dimension is jagged
  auto dim_size = self.size(dim);

  TORCH_CHECK(step > 0, "slice step must be positive");

  auto* nt_impl = get_nested_tensor_impl(self);
  const int64_t ntensors = nt_impl->size(0);
  const auto& sizes = nt_impl->get_nested_size_tensor();
  const auto& strides = nt_impl->get_nested_stride_tensor();
  const std::vector<int64_t>& offsets = nt_impl->get_offsets();

  // special case for empty NT
  if (ntensors == 0) {
    return create_nested_view_tensor(self, sizes.clone(), strides.clone(), std::vector<int64_t>(offsets));
  }

  int64_t start_val=0, end_val=0;
  std::tie(start_val, end_val) = get_slice_range(start, end, dim_size);
  auto len = end_val - start_val;

  // special case for slicing the tensor components themselves when dim=0
  auto new_sizes = (dim > 0) ? sizes.clone() : sizes.slice(0, start_val, end_val).clone();
  auto new_strides = (dim > 0) ? strides.clone() : strides.slice(0, start_val, end_val).clone();
  auto new_offsets = (dim > 0) ? std::vector<int64_t>(offsets) :
      std::vector<int64_t>(offsets.begin() + start_val, offsets.begin() + end_val);
  if (dim > 0) {
    // account for the implicit batch dim
    --dim;
    for (int64_t i : c10::irange(ntensors)) {
      int64_t *size_ptr = new_sizes[i].data_ptr<int64_t>();
      int64_t *stride_ptr = new_strides[i].data_ptr<int64_t>();

      new_offsets[i] = offsets[i] + start_val * stride_ptr[dim];
      size_ptr[dim] = (len + step - 1) / step; // round-up
      stride_ptr[dim] *= step;
    }
  }

  return create_nested_view_tensor(self, new_sizes, new_strides, std::vector<int64_t>(new_offsets));
}

} // namespace native
} // namespace at
