#include <ATen/core/Tensor.h>
#include <ATen/ops/_chunk_cat.h>
#include <ATen/ops/split_with_sizes_copy.h>
#include <c10/util/accumulate.h>
#include <c10/util/irange.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/distributed/fsdp/CollectiveCopy.hpp>
#include <torch/custom_class.h>
#include <torch/library.h>

#include <utility>

namespace c10d::fsdp {

void check_split_with_sizes_copy_inputs(
    at::TensorList out,
    const at::Tensor& input,
    at::IntArrayRef split_sizes,
    at::IntArrayRef num_prefixes,
    int64_t num_chunks) {
  TORCH_CHECK(num_chunks > 0, "expected positive num_chunks");
  TORCH_CHECK(
      input.layout() == at::kStrided && input.is_contiguous(),
      "expected a contiguous strided input");
  TORCH_CHECK(
      split_sizes.size() == out.size() && num_prefixes.size() == out.size(),
      "expected one split size and prefix count per output");
  TORCH_CHECK(
      input.numel() % num_chunks == 0,
      "input size must be divisible by num_chunks");

  int64_t remaining = input.numel() / num_chunks;
  for (const auto i : c10::irange(out.size())) {
    TORCH_CHECK(
        split_sizes[i] >= 0 && split_sizes[i] <= remaining,
        "split sizes must be non-negative and sum to the input chunk size");
    remaining -= split_sizes[i];
    TORCH_CHECK(
        num_prefixes[i] > 0 && split_sizes[i] % num_prefixes[i] == 0,
        "split size must be divisible by its positive prefix count");
    TORCH_CHECK(
        out[i].layout() == at::kStrided && out[i].is_contiguous(),
        "expected contiguous strided outputs");
    TORCH_CHECK(
        out[i].device() == input.device(),
        "input and outputs must be on the same device");
    TORCH_CHECK(
        input.scalar_type() == at::kByte || out[i].dtype() == input.dtype(),
        "output dtype must match the input unless the input has dtype uint8");
    TORCH_CHECK(
        out[i].numel() % num_chunks == 0 &&
            out[i].numel() / num_chunks * out[i].element_size() ==
                split_sizes[i] * input.element_size(),
        "output size must match the split size times num_chunks");
  }
  TORCH_CHECK(remaining == 0, "split sizes must sum to the input chunk size");
}

void split_with_sizes_copy_with_prefixes(
    at::TensorList out,
    const at::Tensor& input,
    at::IntArrayRef split_sizes,
    at::IntArrayRef num_prefixes,
    int64_t num_chunks) {
  check_split_with_sizes_copy_inputs(
      out, input, split_sizes, num_prefixes, num_chunks);
  std::vector<int64_t> sizes;
  std::vector<at::Tensor> outputs;
  for (const auto i : c10::irange(out.size())) {
    const auto size = split_sizes[i] / num_prefixes[i];
    if (num_prefixes[i] == 1) {
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
    auto prefixes = output.view({num_prefixes[i], num_chunks, size}).unbind(0);
    sizes.insert(sizes.end(), num_prefixes[i], size);
    outputs.insert(outputs.end(), prefixes.begin(), prefixes.end());
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

void check_chunk_cat_inputs(
    const at::Tensor& out,
    at::TensorList tensors,
    at::IntArrayRef num_leading_dims,
    int64_t num_chunks) {
  TORCH_CHECK(num_chunks > 0, "expected positive num_chunks");
  TORCH_CHECK(!tensors.empty(), "expected a non-empty input tensor list");
  TORCH_CHECK(
      tensors.size() == num_leading_dims.size(),
      "expected one leading dimension count per input");
  TORCH_CHECK(
      out.layout() == at::kStrided && out.is_contiguous(),
      "expected a contiguous strided output");

  bool has_input = false;
  for (const auto i : c10::irange(tensors.size())) {
    const auto& tensor = tensors[i];
    const auto dim = num_leading_dims[i];
    TORCH_CHECK(
        tensor.layout() == at::kStrided && (dim == 0 || tensor.is_contiguous()),
        "prefix copies require contiguous strided inputs");
    TORCH_CHECK(
        dim >= 0 && dim < tensor.dim(),
        "leading dimension count must be non-negative and less than input ndim");
    TORCH_CHECK(
        dim == 0 || tensor.size(dim) % num_chunks == 0,
        "prefix copies require an evenly divisible shard dimension");
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

at::Tensor& chunk_cat_with_prefixes(
    at::Tensor& out,
    at::TensorList tensors,
    at::IntArrayRef num_leading_dims,
    int64_t num_chunks) {
  check_chunk_cat_inputs(out, tensors, num_leading_dims, num_chunks);
  std::vector<at::Tensor> inputs;
  for (const auto i : c10::irange(tensors.size())) {
    const auto& tensor = tensors[i];
    const auto dim = num_leading_dims[i];
    if (dim == 0) {
      inputs.push_back(tensor);
      continue;
    }
    auto prefixes = tensor.flatten(0, dim - 1).unbind(0);
    inputs.insert(inputs.end(), prefixes.begin(), prefixes.end());
  }
  return at::_chunk_cat_out(out, inputs, 0, num_chunks);
}

} // namespace c10d::fsdp

TORCH_LIBRARY_FRAGMENT(fsdp, m) {
  m.def(
      "_split_with_sizes_copy_with_prefixes_(Tensor(a!)[] self, Tensor input, int[] split_sizes, int[] num_prefixes, int num_chunks) -> ()");
  m.def(
      "_chunk_cat_with_prefixes_(Tensor(a!) self, Tensor[] tensors, int[] num_leading_dims, int num_chunks) -> Tensor(a!)");
}

TORCH_LIBRARY_IMPL(fsdp, CompositeExplicitAutograd, m) {
  m.impl(
      "_split_with_sizes_copy_with_prefixes_",
      TORCH_FN(c10d::fsdp::split_with_sizes_copy_with_prefixes));
  m.impl(
      "_chunk_cat_with_prefixes_",
      TORCH_FN(c10d::fsdp::chunk_cat_with_prefixes));
}

TORCH_LIBRARY_IMPL(fsdp, Functionalize, m) {
  m.impl(
      "_split_with_sizes_copy_with_prefixes_",
      TORCH_FN(c10d::fsdp::split_with_sizes_copy_with_prefixes));
  m.impl(
      "_chunk_cat_with_prefixes_",
      TORCH_FN(c10d::fsdp::chunk_cat_with_prefixes));
}
