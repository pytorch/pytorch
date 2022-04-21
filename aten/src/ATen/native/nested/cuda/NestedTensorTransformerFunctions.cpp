#include <type_traits>

#include <ATen/ATen.h>
#include <ATen/NestedTensorImpl.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_nested_from_padded.h>
#endif

#include <ATen/native/nested/NestedTensorTransformerFunctions.h>

namespace at {
namespace native {
namespace {
int64_t padded_tensor_numel(const Tensor& sizes) {
  const auto sizes_num_rows = sizes.sizes()[0];
  const auto sizes_row_length = sizes.sizes()[1];
  const auto* sizes_data = sizes.data_ptr<int64_t>();
  int64_t numel = 0;
  for (const auto row_num : c10::irange(sizes_num_rows)) {
    const auto* row_ptr = sizes_data + row_num * sizes_row_length;
    int64_t prod = 1;
    for (const auto idx : c10::irange(sizes_row_length)) {
      prod *= row_ptr[idx];
    }
    numel += prod;
  }
  return numel;
}
} // namespace
Tensor nested_from_padded_cuda(
    const Tensor& padded,
    const Tensor& sizes,
    bool do_transform_0213) {
  if (padded.dim() > 1 && padded.dim() < 5) {
    TORCH_CHECK(
        (padded.dim() == 4 && do_transform_0213) ||
            (padded.dim() == 3 && !do_transform_0213),
        "padded tensor size error");
    Tensor target_offsets =
        NestedTensor_batch_offsets_from_size_tensor(sizes, 0);
    Tensor padded_sizes_tensor = at::tensor(padded.sizes());
    Tensor output = at::empty({padded_tensor_numel(sizes)}, padded.options());
    Tensor target_size_sizes = sizes.reshape(-1);

    Tensor metadata =
        at::cat({target_size_sizes, padded_sizes_tensor, target_offsets});
    metadata = metadata.to(at::Device(kCUDA), kInt, true, true);

    auto output_size_ptr = metadata.data_ptr<int>();
    auto input_size_ptr = output_size_ptr + target_size_sizes.numel();
    auto offsets_ptr = input_size_ptr + padded_sizes_tensor.numel();

    if (padded.dtype() == kFloat) {
      if (do_transform_0213) {
        remove_padding_transform0213_kernelLauncher(
            padded.data_ptr<float>(),
            output.data_ptr<float>(),
            offsets_ptr,
            input_size_ptr,
            output_size_ptr,
            padded.dim() - 2,
            padded.sizes()[0]);
      } else {
        remove_padding_kernelLauncher(
            padded.data_ptr<float>(),
            output.data_ptr<float>(),
            offsets_ptr,
            input_size_ptr,
            output_size_ptr,
            padded.dim() - 1,
            padded.sizes()[0]);
      }
    } else if (padded.dtype() == kHalf) {
      if (do_transform_0213) {
        remove_padding_transform0213_kernelLauncher(
            padded.data_ptr<c10::Half>(),
            output.data_ptr<c10::Half>(),
            offsets_ptr,
            input_size_ptr,
            output_size_ptr,
            padded.dim() - 2,
            padded.sizes()[0]);
      } else {
        remove_padding_kernelLauncher(
            padded.data_ptr<c10::Half>(),
            output.data_ptr<c10::Half>(),
            offsets_ptr,
            input_size_ptr,
            output_size_ptr,
            padded.dim() - 1,
            padded.sizes()[0]);
      }
    } else {
      AT_ERROR("Only support fp32/fp16 for padded input");
    }
    return at::detail::make_tensor<NestedTensorImpl>(std::move(output), sizes);
  } else {
    return at::native::nested_from_padded_generic(padded, sizes);
  }
}

Tensor _collapse_two_dims_3(Tensor input, int64_t dim1, int64_t dim2) {
  TORCH_CHECK(dim1 > 0, "dim1: Cannot collapse dim 0.");
  TORCH_CHECK(dim2 > 0, "dim2: Cannot collapse dim 0.");
  TORCH_CHECK(dim2 - 1 == dim1, "dim2 must be one more than dim1.")
  TORCH_CHECK(dim1 == 1, "dim1 must be 1.")
  TORCH_CHECK(get_dim(input) == 3, "Expected input to be 3 dim.");

  auto* nt_input = get_nested_tensor_impl(input);
  TORCH_CHECK(nested_tensor_impl_is_contiguous(nt_input));
  Tensor nt_sizes = nt_input->get_nested_size_tensor();
  const auto& input_buffer = nt_input->get_buffer();

  Tensor sizes_dim1 = at::native::narrow(nt_sizes, 1, 0, 1);
  Tensor sizes_dim2 = at::native::narrow(nt_sizes, 1, 1, 1);

  Tensor new_nt_sizes;
  if (dim1 == 1) {
    Tensor collapsed_sizes = sizes_dim1 * sizes_dim2;
    new_nt_sizes = collapsed_sizes.contiguous();
  }
  Tensor result = at::detail::make_tensor<NestedTensorImpl>(nt_input->get_buffer(), new_nt_sizes);
  TORCH_CHECK(get_dim(result) == 2, "Expected result to be 2 dimensional.");
  return result;
}

Tensor batch_offsets_from_efficient_size(Tensor ef_sizes) {
  int64_t* nt_sizes_ptr = ef_sizes.data_ptr<int64_t>();
  Tensor offsets = torch::empty({1 + ef_sizes.size(0)}, torch::kInt64);
  int64_t* offsets_ptr = offsets.data_ptr<int64_t>();
  offsets_ptr[0] = 0;
  int64_t ef_sizes_size_1 = ef_sizes.size(1);
  for (int64_t i = 0; i < ef_sizes.size(0); i++) {
    int64_t prod = 1;
    for (int64_t j = 0; j < ef_sizes_size_1; j++) {
      prod = prod * nt_sizes_ptr[i * ef_sizes_size_1 + j];
    }
    offsets_ptr[i + 1] = offsets_ptr[i] + prod;
  }
  return offsets;
}

Tensor NestedTensor_to_padded_tensor_cuda(const Tensor& t, double padding) {
  if ((t.dim() >= 2 && t.dim() <= 4)) {
    auto nt_opt_size = get_opt_sizes(nt);
    auto orig_nt_dim = nt.dim();
    auto* nt_input = get_nested_tensor_impl(t);
    TORCH_CHECK(nested_tensor_impl_is_contiguous(nt_input));
    const auto& input_buffer = nt_input->get_buffer();

    if (get_dim(nt) == 3 && nt_iput->get_opt_size(2)) {
      nt = _collapse_two_dims_3(nt, 1, 2);
    }

    nt_input = get_nested_tensor_impl(nt);
    Tensor nt_sizes = nt_input->get_nested_size_tensor();
    Tensor offsets = batch_offsets_from_efficient_size(nt_sizes);
    auto new_size = NestedTensor_get_max_size(nt);
    at::cuda::CUDAStream defaultStream = at::cuda::getDefaultCUDAStream();
    Tensor output = at::empty(IntArrayRef(new_size), nt_buffer.options());

    int64_t input_dim = nt_sizes.size(1);
    int64_t batch_size = nt_sizes.size(0);
    at::Tensor metadata = at::cat({offsets, nt_sizes.reshape(-1)});
    metadata = metadata.to(at::Device(kCUDA), torch::kInt32, true, true);

    std::vector<int64_t> split_sizes;
    split_sizes.push_back(offsets.numel());
    split_sizes.push_back(nt_sizes.numel());

    std::vector<Tensor> split =
        at::split_with_sizes(metadata, IntArrayRef(split_sizes), 0);

    offsets = split[0];
    nt_sizes = split[1];

    if (nt_buffer.dtype() == torch::kFloat16) {
      nested_tensor::cuda::add_padding_kernelLauncher(
          nt_buffer.data_ptr<c10::Half>(),
          output.data_ptr<c10::Half>(),
          (c10::Half)(padding),
          offsets.data_ptr<int>(),
          nt_sizes.data_ptr<int>(),
          input_dim,
          new_size,
          batch_size,
          defaultStream);
      if (orig_nt_dim == 3 && nt_opt_size[2]) {
        output = output.reshape({output.size(0), -1, *nt_opt_size[2]});
      }
      return output;
    }
    if (nt_buffer.dtype() == torch::kFloat) {
      nested_tensor::cuda::add_padding_kernelLauncher(
          nt_buffer.data_ptr<float>(),
          output.data_ptr<float>(),
          (float)(padding),
          offsets.data_ptr<int>(),
          nt_sizes.data_ptr<int>(),
          input_dim,
          new_size,
          batch_size,
          defaultStream);
      if (orig_nt_dim == 3 && nt_opt_size[2]) {
        output = output.reshape({output.size(0), -1, *nt_opt_size[2]});
      }
      return output;
    }
  }
  // TODO port CUDA path in pytorch/nestedtensor to_padded_tensor!
  // TODO: skipped optimization for case of all 1x1 tensors
  auto& nt = *get_nested_tensor_impl(t);
  auto max_size = NestedTensor_get_max_size(nt);
  auto sizes = nt.get_nested_size_tensor();

  if (sizes.numel() == 0 || sizes.dim() == 0) {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(nt.get_buffer().numel() == 0);
    return nt.get_buffer();
  }

  // TODO: doesn't handle empty/scalar entries because we don't need
  // it for transformers; see to_padded_tensor in
  // pytorch/nestedtensor's masking.cpp.

  const auto sizes_num_rows = sizes.sizes()[0];
  const auto sizes_num_columns = sizes.sizes()[1];
  const auto sizes_data_start = sizes.data_ptr<int64_t>();
  const auto sizes_data_end = sizes_data_start + sizes.numel();
  std::vector<int64_t> split_sizes;
  split_sizes.reserve(sizes_num_rows);
  for (auto sizes_data = sizes_data_start; sizes_data != sizes_data_end;
       sizes_data += sizes_num_columns) {
    split_sizes.push_back(
        num_bytes(IntArrayRef(sizes_data, sizes_num_columns)));
  }
  std::vector<int64_t> nonzero_split_sizes;
  for (const auto split_size : split_sizes) {
    if (split_size > 0) {
      nonzero_split_sizes.push_back(split_size);
    }
  }
  const auto buffer = nt.get_buffer();
  std::vector<Tensor> buffers_;
  if (!nonzero_split_sizes.empty()) {
    buffers_ = at::split_with_sizes(buffer, nonzero_split_sizes, 0);
  }

  std::vector<Tensor> buffers;
  buffers.reserve(split_sizes.size());
  int64_t next_buffer = 0;
  auto sizes_ptr = sizes_data_start;
  for (const auto split_size : split_sizes) {
    Tensor to_pad;
    IntArrayRef tensor_sizes(sizes_ptr, sizes_num_columns);
    if (split_size > 0) {
      to_pad = buffers_[next_buffer++].reshape(tensor_sizes);
    } else {
      to_pad = at::empty(tensor_sizes, buffer.options());
    }
    buffers.push_back(pad_tensor_to_shape(to_pad, max_size, padding));
    sizes_ptr += sizes_num_columns;
  }
  return at::stack(buffers);
}
} // namespace native
} // namespace at
