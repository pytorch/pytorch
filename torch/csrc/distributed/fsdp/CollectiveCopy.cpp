#include <ATen/core/Tensor.h>
#include <c10/util/accumulate.h>
#include <c10/util/irange.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/distributed/fsdp/CollectiveCopy.hpp>
#include <torch/custom_class.h>
#include <torch/library.h>

#include <algorithm>
#include <utility>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/_chunk_cat.h>
#include <ATen/ops/cat.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/split_with_sizes_copy.h>
#endif

namespace c10d::fsdp {

bool check_all_gather_copy_out_inputs(
    at::TensorList out,
    const at::Tensor& input,
    at::IntArrayRef split_sizes,
    at::IntArrayRef outer_sizes,
    int64_t num_chunks) {
  TORCH_CHECK(num_chunks > 0, "expected positive num_chunks");
  TORCH_CHECK(input.layout() == at::kStrided, "expected a strided input");
  TORCH_CHECK(
      split_sizes.size() == out.size() && outer_sizes.size() == out.size(),
      "expected one split size and outer size per output");
  TORCH_CHECK(
      input.numel() % num_chunks == 0,
      "input size must be divisible by num_chunks");

  int64_t remaining = input.numel() / num_chunks;
  bool needs_resize = false;
  const bool byte_input = input.scalar_type() == at::kByte;
  for (const auto i : c10::irange(out.size())) {
    TORCH_CHECK(
        split_sizes[i] >= 0 && split_sizes[i] <= remaining,
        "split sizes must be non-negative and sum to the input chunk size");
    remaining -= split_sizes[i];
    TORCH_CHECK(outer_sizes[i] > 0, "expected a positive outer size");
    TORCH_CHECK(
        out[i].layout() == at::kStrided && out[i].is_contiguous(),
        "expected contiguous strided outputs");
    TORCH_CHECK(
        out[i].device() == input.device(),
        "input and outputs must be on the same device");
    TORCH_CHECK(
        byte_input || out[i].dtype() == input.dtype(),
        "output dtype must match the input unless the input has dtype uint8");
    TORCH_CHECK(
        out[i].numel() % num_chunks == 0,
        "output size must be divisible by num_chunks");
    const int64_t output_chunk_size =
        out[i].numel() / num_chunks * (byte_input ? out[i].element_size() : 1);
    TORCH_CHECK(
        output_chunk_size % outer_sizes[i] == 0,
        "output size per chunk must be divisible by its outer size");
    needs_resize |= output_chunk_size != split_sizes[i];
  }
  TORCH_CHECK(remaining == 0, "split sizes must sum to the input chunk size");
  return needs_resize;
}

// Called from libtorch_cuda as well as libtorch_cpu.
// NOLINTNEXTLINE(misc-use-internal-linkage)
std::vector<at::Tensor> split_all_gather_output_with_resize(
    at::TensorList out,
    const at::Tensor& input,
    at::IntArrayRef split_sizes,
    at::IntArrayRef outer_sizes,
    int64_t num_chunks) {
  // Hooks may change payload sizes while reusing cached outputs. Resize only
  // the rank-major copy views, then reassemble using the cached output layout.
  std::vector<at::Tensor> outputs;
  outputs.reserve(out.size());
  for (const auto i : c10::irange(out.size())) {
    auto buffer = outer_sizes[i] == 1 ? out[i] : at::empty_like(out[i]);
    auto output = buffer.view({num_chunks, -1});
    if (input.scalar_type() == at::kByte) {
      output = output.view(at::kByte);
    }
    outputs.push_back(std::move(output));
  }
  if (input.numel() > 0) {
    at::split_with_sizes_copy_out(
        outputs, input.view({num_chunks, -1}), split_sizes, 1);
  }
  return outputs;
}

namespace {

void all_gather_copy_out_with_resize(
    at::TensorList out,
    const at::Tensor& input,
    at::IntArrayRef split_sizes,
    at::IntArrayRef outer_sizes,
    int64_t num_chunks) {
  auto outputs = split_all_gather_output_with_resize(
      out, input, split_sizes, outer_sizes, num_chunks);
  for (const auto i : c10::irange(out.size())) {
    if (outer_sizes[i] != 1 && split_sizes[i] > 0) {
      auto output = out[i];
      if (input.scalar_type() == at::kByte) {
        output = output.view({-1}).view(at::kByte);
      }
      const auto& buffer = outputs[i];
      const auto outer_size = outer_sizes[i];
      // The copy view may have resized; reassemble the cached output layout.
      const auto rank_stride = output.numel() / num_chunks;
      const auto inner_size = rank_stride / outer_size;
      std::vector<at::Tensor> chunks;
      chunks.reserve(num_chunks);
      for (const auto rank : c10::irange(num_chunks)) {
        chunks.push_back(buffer.as_strided(
            {outer_size, inner_size},
            {inner_size, 1},
            buffer.storage_offset() + rank * rank_stride));
      }
      auto target = output.view({outer_sizes[i], -1});
      at::cat_out(target, chunks, 1);
    }
    if (!out[i].is_inference()) {
      torch::autograd::impl::bump_version(out[i]);
    }
  }
}

} // namespace

void all_gather_copy_out(
    at::TensorList out,
    const at::Tensor& input,
    at::IntArrayRef split_sizes,
    at::IntArrayRef outer_sizes,
    int64_t num_chunks) {
  const bool needs_resize = check_all_gather_copy_out_inputs(
      out, input, split_sizes, outer_sizes, num_chunks);
  if (needs_resize) {
    all_gather_copy_out_with_resize(
        out, input, split_sizes, outer_sizes, num_chunks);
    return;
  }
  std::vector<int64_t> sizes;
  std::vector<at::Tensor> outputs;
  for (const auto i : c10::irange(out.size())) {
    const auto size = split_sizes[i] / outer_sizes[i];
    if (outer_sizes[i] == 1) {
      auto output = out[i].view({num_chunks, -1});
      if (input.scalar_type() == at::kByte) {
        output = output.view(at::kByte);
      }
      sizes.push_back(size);
      outputs.push_back(std::move(output));
      continue;
    }
    auto output = out[i].view({-1});
    if (input.scalar_type() == at::kByte) {
      output = output.view(at::kByte);
    }
    auto outer_slices =
        output.view({outer_sizes[i], num_chunks, size}).unbind(0);
    sizes.insert(sizes.end(), outer_sizes[i], size);
    outputs.insert(outputs.end(), outer_slices.begin(), outer_slices.end());
  }
  if (input.numel() > 0) {
    at::split_with_sizes_copy_out(
        outputs,
        input.view({num_chunks, input.numel() / num_chunks}),
        sizes,
        1);
  }
  for (const auto& output : out) {
    if (!output.is_inference()) {
      torch::autograd::impl::bump_version(output);
    }
  }
}

void check_reduce_scatter_copy_in_inputs(
    const at::Tensor& out,
    at::TensorList tensors,
    at::IntArrayRef num_leading_dims,
    int64_t num_chunks) {
  TORCH_CHECK(num_chunks > 0, "expected positive num_chunks");
  TORCH_CHECK(!tensors.empty(), "expected a non-empty input tensor list");
  TORCH_CHECK(
      tensors.size() == num_leading_dims.size(),
      "expected one leading dimension count per input");
  TORCH_CHECK(out.layout() == at::kStrided, "expected a strided output");

  bool has_input = false;
  for (const auto i : c10::irange(tensors.size())) {
    const auto& tensor = tensors[i];
    const auto dim = num_leading_dims[i];
    TORCH_CHECK(
        tensor.layout() == at::kStrided && (dim == 0 || tensor.is_contiguous()),
        "nonzero-dimension copies require contiguous strided inputs");
    TORCH_CHECK(
        dim >= 0 && dim < tensor.dim(),
        "leading dimension count must be non-negative and less than input ndim");
    TORCH_CHECK(
        dim == 0 || tensor.size(dim) % num_chunks == 0,
        "nonzero-dimension copies require an evenly divisible shard dimension");
    TORCH_CHECK(
        tensor.dtype() == tensors[0].dtype(),
        "inputs must have the same dtype");
    TORCH_CHECK(
        tensor.device() == out.device(),
        "inputs and output must be on the same device");
    if (tensor.numel() == 0) {
      TORCH_CHECK(
          dim > 0 && c10::multiply_integers(tensor.sizes().slice(0, dim)) == 0,
          "expected non-empty inputs");
    } else {
      has_input = true;
    }
  }
  TORCH_CHECK(has_input, "expected a non-empty input tensor list");
}

at::Tensor& reduce_scatter_copy_in(
    at::Tensor& out,
    at::TensorList tensors,
    at::IntArrayRef num_leading_dims,
    int64_t num_chunks) {
  check_reduce_scatter_copy_in_inputs(
      out, tensors, num_leading_dims, num_chunks);
  if (std::all_of(
          num_leading_dims.begin(), num_leading_dims.end(), [](int64_t dim) {
            return dim == 0;
          })) {
    return at::_chunk_cat_out(out, tensors, 0, num_chunks);
  }
  std::vector<at::Tensor> inputs;
  for (const auto i : c10::irange(tensors.size())) {
    const auto& tensor = tensors[i];
    const auto dim = num_leading_dims[i];
    if (dim == 0) {
      inputs.push_back(tensor);
      continue;
    }
    auto outer_slices = tensor.flatten(0, dim - 1).unbind(0);
    inputs.insert(inputs.end(), outer_slices.begin(), outer_slices.end());
  }
  return at::_chunk_cat_out(out, inputs, 0, num_chunks);
}

} // namespace c10d::fsdp

TORCH_LIBRARY_FRAGMENT(fsdp, m) {
  m.def(
      "_all_gather_copy_out_(Tensor(a!)[] self, Tensor src, int[] split_sizes, int[] outer_sizes, int num_chunks) -> ()");
  m.def(
      "_reduce_scatter_copy_in_(Tensor(a!) self, Tensor[] tensors, int[] num_leading_dims, int num_chunks) -> Tensor(a!)");
}

TORCH_LIBRARY_IMPL(fsdp, CompositeExplicitAutograd, m) {
  m.impl("_all_gather_copy_out_", TORCH_FN(c10d::fsdp::all_gather_copy_out));
  m.impl(
      "_reduce_scatter_copy_in_", TORCH_FN(c10d::fsdp::reduce_scatter_copy_in));
}

TORCH_LIBRARY_IMPL(fsdp, Functionalize, m) {
  m.impl("_all_gather_copy_out_", TORCH_FN(c10d::fsdp::all_gather_copy_out));
  m.impl(
      "_reduce_scatter_copy_in_", TORCH_FN(c10d::fsdp::reduce_scatter_copy_in));
}
