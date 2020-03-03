#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/InferSize.h>
#include <ATen/NativeFunctions.h>
#include <ATen/WrapDimUtils.h>

#if !AT_MKLDNN_ENABLED()

namespace at {
namespace native {

Tensor mkldnn_view(const Tensor& self, IntArrayRef size) {
  AT_ERROR("mkldnn_reshape: ATen not compiled with MKLDNN support");
}

Tensor mkldnn_reshape(const Tensor& self, IntArrayRef size) {
  AT_ERROR("mkldnn_reshape: ATen not compiled with MKLDNN support");
}

Tensor mkldnn_clone(const Tensor& self, c10::optional<c10::MemoryFormat> optional_memory_format) {
  AT_ERROR("mkldnn_clone: ATen not compiled with MKLDNN support");
}

Tensor mkldnn_transpose(const Tensor& self, int64_t dim0, int64_t dim1) {
  AT_ERROR("mkldnn_transpose: ATen not compiled with MKLDNN support");
}

Tensor& mkldnn_transpose_(Tensor& self, int64_t dim0, int64_t dim1) {
  AT_ERROR("mkldnn_transpose_: ATen not compiled with MKLDNN support");
}

Tensor& mkldnn_cat_out(Tensor& result, const TensorList tensors, int64_t dim) {
  AT_ERROR("mkldnn_cat_out: ATen not compiled with MKLDNN support");
}

Tensor mkldnn_cat(const TensorList tensors, int64_t dim) {
  AT_ERROR("mkldnn_cat: ATen not compiled with MKLDNN support");
}

std::vector<Tensor> mkldnn_split_with_sizes(const Tensor& self, IntArrayRef split_sizes, int64_t dim) {
  AT_ERROR("mkldnn_split_with_sizes: ATen not compiled with MKLDNN support");
}

std::vector<Tensor> mkldnn_split(const Tensor& self, int64_t split_size, int64_t dim) {
  AT_ERROR("mkldnn_split: ATen not compiled with MKLDNN support");
}

} // namespace native
} // namespace at

#else // AT_MKLDNN_EBABLED

#include <ATen/native/mkldnn/MKLDNNCommon.h>

namespace {
inline void check_cat_no_zero_dim(at::TensorList tensors) {
  for (size_t i = 0; i < tensors.size(); ++i) {
    auto& t = tensors[i];
    TORCH_CHECK(t.dim() > 0,
      "zero-dimensional tensor (at position ", i, ") cannot be concatenated");
  }
}
}

namespace at {
namespace native {

Tensor mkldnn_view(const Tensor& self, IntArrayRef size) {
  AT_ERROR(
      "Currently Mkldnn tensor does not support view. Change to use reshape instead");
}

Tensor mkldnn_reshape(const Tensor& self, IntArrayRef size) {
  auto inferred_size = at::infer_size(size, self.numel());
  if (self.sizes() == inferred_size) {
    return self;
  }
  const ideep::tensor& x = itensor_from_mkldnn(self);
  ideep::tensor y{x};
  y.reshape(inferred_size);
  return new_with_itensor_mkldnn(std::move(y), self.options());
}

Tensor mkldnn_clone(const Tensor& self, c10::optional<c10::MemoryFormat> optional_memory_format) {
  TORCH_CHECK(
      !optional_memory_format.has_value(),
      "unsupported memory format option ",
      optional_memory_format.value());
  ideep::tensor& src = itensor_from_mkldnn(self);
  ideep::tensor dst;
  ideep::direct_copy::compute(src, dst);
  return new_with_itensor_mkldnn(std::move(dst), self.options());
}

Tensor mkldnn_transpose(const Tensor & self, int64_t dim0, int64_t dim1) {
  const ideep::tensor& x = itensor_from_mkldnn(self);
  ideep::tensor y;
  std::vector<int> axes(x.ndims());
  std::iota(axes.begin(), axes.end(), 0);
  std::swap(axes[dim0], axes[dim1]);
  y.transpose_from(x, axes);
  return new_with_itensor_mkldnn(std::move(y), self.options());
}

Tensor& mkldnn_transpose_(Tensor& self, int64_t dim0, int64_t dim1) {
  AT_ERROR("mkldnn_transpose_: in-place mkldnn operations are not supported yet");
}

Tensor& mkldnn_cat_out(Tensor& result, TensorList tensors, int64_t dim) {
  check_cat_no_zero_dim(tensors);
  dim = legacy_cat_wrap_dim(dim, tensors);
  std::vector<ideep::tensor> x;
  for (auto i =0; i< tensors.size(); i++) {
    TORCH_CHECK(!(tensors[i].dim() == 1 && tensors[i].sizes()[0] == 0),
      "Currently Mkldnn cat operators do not support empty tensor.");
    x.push_back(itensor_from_mkldnn(tensors[i]));
  }
  ideep::tensor& y = itensor_from_mkldnn(result);
  ideep::concat::compute(x, dim, y);
  return result;
}

Tensor mkldnn_cat(TensorList tensors, int64_t dim) {
  check_cat_no_zero_dim(tensors);
  dim = legacy_cat_wrap_dim(dim, tensors);
  std::vector<ideep::tensor> x;
  for (auto i = 0; i < tensors.size(); i++) {
    TORCH_CHECK(!(tensors[i].dim() == 1 && tensors[i].sizes()[0] == 0),
      "Currently Mkldnn cat operators do not support empty tensor.");
    x.push_back(itensor_from_mkldnn(tensors[i]));
  }
  ideep::tensor y;
  ideep::concat::compute(x, dim, y);
  return new_with_itensor_mkldnn(std::move(y), tensors[0].options());
}

std::vector<Tensor> mkldnn_split_with_sizes(const Tensor& self, IntArrayRef split_sizes, int64_t dim) {
  ideep::tensor& x = itensor_from_mkldnn(self);
  int64_t num_splits = split_sizes.size();
  std::vector<Tensor> splits(num_splits);
  std::vector<int32_t> sizes;
  for (auto i = 0; i < num_splits; i++) {
    auto length = split_sizes[i];
    TORCH_CHECK(length >= 0,
             "split_with_sizes expects split_sizes have only non-negative ",
             "entries, but got split_sizes=", split_sizes);
    sizes.push_back((int32_t)length);
  }
  auto y = ideep::spliter::compute(x, sizes, dim, false);
  for (auto j = 0; j < num_splits; j++) {
    splits[j] = new_with_itensor_mkldnn(std::move(y[j]), self.options());
  }
  return splits;
}
std::vector<Tensor> mkldnn_split(const Tensor& self, int64_t split_size, int64_t dim) {
  int64_t dim_size = self.size(dim);
  int64_t num_splits = 1;
  if (split_size != 0) {
    // ensuring num_splits is at least 1 makes consistent the case where split_size > dim_size
    // (returns a single split).  We might want to error here, but keep it for BC.
    num_splits = std::max<int64_t>((dim_size + split_size - 1) / split_size, 1);
  }
  std::vector<int64_t> split_sizes(num_splits, split_size);
  int64_t last_split_size = split_size - (split_size * num_splits - dim_size);
  split_sizes[num_splits-1] = last_split_size;
  return native::mkldnn_split_with_sizes(self, split_sizes, dim);
}

} // namespace native
} // namespace at

#endif // AT_MKLDNN_EBABLED
