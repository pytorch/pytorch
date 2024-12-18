#include <torch/csrc/xpu/comm.h>

#include <torch/csrc/utils/tensor_flatten.h>

#include <ATen/ATen.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/xpu/XPUContext.h>
#include <c10/core/DeviceGuard.h>
#include <c10/core/StreamGuard.h>
#include <c10/util/irange.h>
#include <torch/csrc/autograd/variable.h>
#include <optional>

#include <cstddef>
#include <vector>

namespace torch::xpu {
using namespace at;
using namespace torch::autograd;

// Some operations can be performed more efficiently if we're handling tensors
// of a single type only. Adding this logic directly in the loop makes it a bit
// ugly, so here's a helper for it.
struct unique_type_checker {
  void show(size_t type_id) {
    if (!unique) {
      return;
    }
    if (!type_id_) {
      type_id_ = type_id;
    }

    unique = type_id_.value() == type_id;
  }

  std::optional<size_t> type_id_;
  bool unique = true;
};

// ***************** Broadcast *******************
//
// Broadcast a source tensor (CPU or XPU) to a list of XPU devices, or XPU
// tensors on one or more devices.

// no checks
static std::vector<Tensor>& _broadcast_out_impl(
    const Tensor& tensor,
    std::vector<Tensor>& out_tensors) {
  for (auto& out_tensor : out_tensors) {
    out_tensor.copy_(tensor, /*non_blocking=*/true);
  }
  return out_tensors;
}

std::vector<Tensor>& broadcast_out(
    const Tensor& tensor,
    std::vector<Tensor>& out_tensors) {
  for (const auto i : c10::irange(out_tensors.size())) {
    TORCH_CHECK(
        out_tensors[i].is_xpu(),
        "Expected all output tensors to be XPU tensors, but output tensor at index ",
        i,
        " has device '",
        out_tensors[i].device(),
        "'");
    TORCH_CHECK(
        out_tensors[i].sizes() == tensor.sizes(),
        "Expected all output tensors to have same shape as the source tensor ",
        tensor.sizes(),
        ", but output tensor at index ",
        i,
        " has shape ",
        out_tensors[i].sizes());
  }
  return _broadcast_out_impl(tensor, out_tensors);
}

std::vector<Tensor> broadcast(const Tensor& tensor, IntArrayRef devices) {
  std::vector<Tensor> diff_device_dst_tensors;
  diff_device_dst_tensors.reserve(devices.size());
  for (auto device : devices) {
    TORCH_CHECK(
        device >= 0, "Expected non-negative device index, but got ", device);
    if (device != tensor.get_device()) {
      diff_device_dst_tensors.emplace_back(at::empty(
          tensor.sizes(),
          tensor.options().device(at::Device(
              DeviceType::XPU,
              static_cast<DeviceIndex>(device))))); // preserve memory format
    }
  }
  _broadcast_out_impl(tensor, diff_device_dst_tensors);
  std::vector<Tensor> dst_tensors;
  dst_tensors.reserve(devices.size());
  auto it = diff_device_dst_tensors.begin();
  for (auto device : devices) {
    // NOLINTNEXTLINE(bugprone-branch-clone)
    if (device != tensor.get_device()) {
      dst_tensors.emplace_back(*it++);
    } else {
      dst_tensors.emplace_back(tensor);
    }
  }
  TORCH_INTERNAL_ASSERT(it == diff_device_dst_tensors.end());
  return dst_tensors;
}

// NOTE [ Version Counter in comm.*_coalesced ]
//
// broadcast_coalesced
// ~~~~~~~~~~~~~~~~~~~
//
// In broadcast_coalesced, multiple variables may be coalesced into a single
// large one, broadcast to other devices, and the get split according to the
// original shapes.
//
// When splitting, the view operations will make all Variables broadcast
// together to share a single version counter, because they are all views of the
// large Variable. However, that large Variable is immediately discarded and all
// these Variables do not share storage at all.
//
// For example, when two buffers are broadcast together in `DataParallel` and
// one of them is modified in-place during `forward` but the other is needed in
// backward, autograd engine will complain.
//
// We thus re-wrap these Variables after broadcasting (i.e., effectively doing
// what is equivalent to .data in Python), and give them individual version
// counters.
//
// NB: Just calling detach() on the variables is not sufficient
//
// NB: For `device[0]` in broadcast_coalesced, the input Variables are always
//     returned as-is, so **do not** re-wrap them.
//
// reduce_add_coalesced
// ~~~~~~~~~~~~~~~~~~~~
//
// Similarly for reduce_add_coalesced, when the output are newly created
// Variables.
tensor_list2d broadcast_coalesced(
    TensorList tensors,
    IntArrayRef devices,
    size_t buffer_size) {
  TORCH_CHECK(
      std::all_of(
          tensors.begin(),
          tensors.end(),
          [&](const at::Tensor& t) { return t.get_device() == devices[0]; }),
      "All tensors must be on devices[0]: ",
      devices[0]);

  tensor_list2d outputs(devices.size());
  outputs[0] = tensors.vec();
  for (auto& o : outputs)
    o.reserve(tensors.size());

  unique_type_checker type_checker;
  c10::Device device(at::getAccelerator(true).value(), devices[0]);
  c10::DeviceGuard device_guard(device);
  for (auto& chunk : torch::utils::take_tensors(tensors, buffer_size)) {
    auto type_id = chunk.type_id();
    type_checker.show(type_id);
    std::vector<at::Tensor> results;
    if (chunk.options().is_sparse()) {
      auto flat_tuple = torch::utils::flatten_sparse_tensors(chunk.tensors);
      auto broadcast_indices = broadcast(flat_tuple.first, devices);
      auto broadcast_values = broadcast(flat_tuple.second, devices);
      results.reserve(devices.size());
      for (size_t i = 1, num_devices = devices.size(); i < num_devices; ++i) {
        device_guard.set_index(static_cast<DeviceIndex>(devices[i]));
        auto& device_outputs = outputs[i];
        auto& inds = broadcast_indices[i];
        auto& vals = broadcast_values[i];
        for (const auto& var : torch::utils::unflatten_sparse_tensors(
                 inds, vals, chunk.tensors)) {
          // See NOTE [ Version Counter in comm.*_coalesced ]
          device_outputs.emplace_back(make_variable(var.tensor_data(), false));
        }
      }
    } else {
      auto results = broadcast(
          torch::utils::flatten_dense_tensors(chunk.tensors), devices);
      for (size_t i = 1, num_devices = devices.size(); i < num_devices; ++i) {
        device_guard.set_index(static_cast<DeviceIndex>(devices[i]));
        auto& device_outputs = outputs[i];
        for (auto& var :
             torch::utils::unflatten_dense_tensors(results[i], chunk.tensors)) {
          // See NOTE [ Version Counter in comm.*_coalesced ]
          device_outputs.emplace_back(make_variable(var.tensor_data(), false));
        }
      }
    }
  }

  // If we only saw a single tensor type, then we can skip expensive reordering
  if (!type_checker.unique) {
    for (auto& o : outputs)
      torch::utils::reorder_tensors_like(o, tensors);
  }
  return outputs;
}

// ***************** Scatter *******************
//
// Scatter a source tensor (CPU or XPU) to a list of XPU tensors on one or
// more devices.

std::vector<at::Tensor>& scatter_out(
    const at::Tensor& tensor,
    std::vector<at::Tensor>& out_tensors,
    int64_t dim,
    const std::optional<std::vector<std::optional<at::xpu::XPUStream>>>&
        streams) {
  TORCH_CHECK(
      !out_tensors.empty(),
      "Expected at least one output tensor to scatter to");
  dim = at::maybe_wrap_dim(dim, tensor);
  int64_t total_size = 0;
  std::vector<int64_t> chunk_sizes;
  chunk_sizes.reserve(out_tensors.size());
  for (const auto i : c10::irange(out_tensors.size())) {
    TORCH_CHECK(
        out_tensors[i].is_xpu(),
        "Expected all output tensors to be XPU tensors, but output tensor at index ",
        i,
        " has device '",
        out_tensors[i].device(),
        "'");
    auto out_sizes = out_tensors[i].sizes().vec();
    bool same_ndim = out_sizes.size() == static_cast<size_t>(tensor.dim());
    if (same_ndim) {
      total_size += out_sizes[dim];
      chunk_sizes.emplace_back(out_sizes[dim]);
      out_sizes[dim] = tensor.size(dim);
    }
    TORCH_CHECK(
        same_ndim && out_sizes == tensor.sizes(),
        "Output tensor at index ",
        i,
        " has incorrect shape: ",
        out_tensors[i].sizes(),
        ". Expected same "
        "shape except for scatter dim ",
        dim,
        " as the source tensor: ",
        at::IntArrayRef(tensor.sizes()));
  }
  TORCH_CHECK(
      total_size == tensor.size(dim),
      "Total size for output tensors along scatter dim ",
      dim,
      " does not match "
      "the source tensor size at dim ",
      dim,
      ". Expected ",
      tensor.size(dim),
      ", but got total size ",
      total_size);

  auto chunks =
      tensor.split_with_sizes(/*split_sizes=*/chunk_sizes, /*dim=*/dim);
  c10::OptionalStreamGuard stream_guard;
  for (const auto i : c10::irange(chunks.size())) {
    if (i < (streams ? streams->size() : 0U) && (*streams)[i]) {
      const auto device_index = out_tensors[i].get_device();
      TORCH_CHECK(
          (*streams)[i]->device_index() == device_index,
          "Expected the device associated with the stream at index ",
          i,
          " (was ",
          (*streams)[i]->device_index(),
          ") ",
          "to match the device supplied at that index ",
          "(expected ",
          static_cast<int16_t>(device_index),
          ")");
      stream_guard.reset_stream(*(*streams)[i]);
    }
    // NB: We don't detect the case where `out_tensor` is already the correct
    //     view of `tensor` since that would be nontrivial and involve checking
    //     ptr, offset, and strides. So `scatter_out(src, src.chunk(...))` does
    //     more copying than `scatter(src)`.
    out_tensors[i].copy_(chunks[i], /*non_blocking=*/true);
  }
  return out_tensors;
}

std::vector<at::Tensor> scatter(
    const at::Tensor& tensor,
    at::IntArrayRef devices,
    const std::optional<std::vector<int64_t>>& chunk_sizes,
    int64_t dim,
    const std::optional<std::vector<std::optional<at::xpu::XPUStream>>>&
        streams) {
  TORCH_CHECK(!devices.empty(), "Expected at least one device to scatter to");
  if (chunk_sizes.has_value()) {
    TORCH_CHECK(
        chunk_sizes->size() == devices.size(),
        "Expected devices and chunk_sizes to be of same length, but got "
        "len(devices) = ",
        devices.size(),
        " and len(chunk_sizes) = ",
        chunk_sizes->size());
  }
  dim = at::maybe_wrap_dim(dim, tensor);
  std::vector<at::Tensor> chunks = chunk_sizes
      ? tensor.split_with_sizes(/*split_sizes=*/*chunk_sizes, /*dim=*/dim)
      : tensor.chunk(
            /*chunks=*/static_cast<int64_t>(devices.size()), /*dim=*/dim);
  c10::OptionalStreamGuard stream_guard;
  for (const auto i : c10::irange(chunks.size())) {
    const auto device_index = static_cast<int16_t>(devices[i]);
    if (device_index != tensor.get_device()) {
      if (i < (streams ? streams->size() : 0U) && (*streams)[i]) {
        TORCH_CHECK(
            (*streams)[i]->device_index() == device_index,
            "Expected the device associated with the stream at index ",
            i,
            " (was ",
            (*streams)[i]->device_index(),
            ") ",
            "to match the device supplied at that index ",
            "(expected ",
            device_index,
            ")");
        stream_guard.reset_stream(*(*streams)[i]);
      }
      TORCH_CHECK(
          device_index >= 0,
          "Expected non-negative device index, but got ",
          device_index);
      chunks[i] = chunks[i].to(
          {DeviceType::XPU, device_index},
          /*non_blocking=*/true,
          /*copy=*/false,
          /*memory_format=*/at::MemoryFormat::Preserve);
    }
  }
  return chunks;
}

// ***************** Gather *******************
//
// Gather a list of XPU tensors on one or more devices to a target tensor or
// device, either CPU or XPU.

// no checks
static at::Tensor& _gather_out_impl(
    at::TensorList tensors,
    at::Tensor& out_tensor,
    int64_t dim) {
  std::vector<int64_t> chunk_sizes;
  chunk_sizes.reserve(tensors.size());
  for (auto& tensor : tensors) {
    chunk_sizes.emplace_back(tensor.size(dim));
  }
  auto chunks =
      out_tensor.split_with_sizes(/*split_sizes=*/chunk_sizes, /*dim=*/dim);
  for (const auto i : c10::irange(tensors.size())) {
    chunks[i].copy_(tensors[i], /*non_blocking=*/out_tensor.is_xpu());
  }
  return out_tensor;
}

at::Tensor& gather_out(
    at::TensorList tensors,
    at::Tensor& out_tensor,
    int64_t dim) {
  TORCH_CHECK(!tensors.empty(), "Expected at least one tensor to gather from");
  int64_t total_size = 0;
  auto& first = tensors.front();
  const auto first_size = first.sizes();
  dim = at::maybe_wrap_dim(dim, first);
  std::vector<int64_t> expected_size(first_size.begin(), first_size.end());
  for (const auto i : c10::irange(tensors.size())) {
    const auto& tensor = tensors[i];
    TORCH_CHECK(
        tensor.is_xpu(),
        "Expected all input tensors to be XPU tensors, but "
        "tensor at index ",
        i,
        " has device '",
        tensor.device(),
        "'");
    TORCH_CHECK(
        tensor.ndimension() == static_cast<int64_t>(expected_size.size()),
        "Expected all input tensors to have the same number of dimensions, but ",
        "tensor at index ",
        i,
        "has ",
        tensor.ndimension(),
        " dimensions, (expected ",
        expected_size.size(),
        ")");
    expected_size[dim] = tensor.size(dim);
    for (const auto dimension : c10::irange(expected_size.size())) {
      TORCH_CHECK(
          expected_size[dimension] == tensor.size(dimension),
          "Input tensor at index ",
          i,
          " has invalid shape ",
          tensor.sizes(),
          ", but expected ",
          at::IntArrayRef(expected_size));
    }
    total_size += tensor.size(dim);
  }
  expected_size[dim] = total_size;
  TORCH_CHECK(
      out_tensor.sizes() == expected_size,
      "Expected out tensor to have shape ",
      at::IntArrayRef(expected_size),
      ", but got ",
      out_tensor.sizes())

  return _gather_out_impl(tensors, out_tensor, dim);
}

at::Tensor gather(
    at::TensorList tensors,
    int64_t dim,
    std::optional<int32_t> destination_index) {
  TORCH_CHECK(!tensors.empty(), "Expected at least one tensor to gather from");
  int64_t total_size = 0;
  auto& first = tensors.front();
  const auto first_size = first.sizes();
  dim = at::maybe_wrap_dim(dim, first);
  std::vector<int64_t> expected_size(first_size.begin(), first_size.end());
  auto memory_format = first.suggest_memory_format();
  for (const auto i : c10::irange(tensors.size())) {
    const auto& tensor = tensors[i];
    TORCH_CHECK(
        tensor.is_xpu(),
        "Expected all input tensors to be XPU tensors, but "
        "tensor at index ",
        i,
        " has device ",
        tensor.device());
    TORCH_CHECK(
        tensor.ndimension() == static_cast<int64_t>(expected_size.size()),
        "Expected all input tensors to have the same number of dimensions, but ",
        "tensor at index ",
        i,
        "has ",
        tensor.ndimension(),
        " dimensions, (expected ",
        expected_size.size(),
        ")");
    expected_size[dim] = tensor.size(dim);
    for (const auto dimension : c10::irange(expected_size.size())) {
      TORCH_CHECK(
          expected_size[dimension] == tensor.size(dimension),
          "Input tensor at index ",
          i,
          " has invalid shape ",
          tensor.sizes(),
          ", but expected ",
          at::IntArrayRef(expected_size));
    }
    total_size += tensor.size(dim);
    if (memory_format != MemoryFormat::Contiguous &&
        tensor.suggest_memory_format() != memory_format) {
      memory_format = MemoryFormat::Contiguous;
    }
  }
  expected_size[dim] = total_size;
  at::Device device(DeviceType::CPU);
  if (!destination_index || *destination_index != -1) {
    device = at::Device(
        DeviceType::XPU,
        destination_index ? static_cast<DeviceIndex>(*destination_index)
                          : DeviceIndex(-1));
  }

  at::Tensor result =
      at::empty(expected_size, first.options().device(device), memory_format);
  return _gather_out_impl(tensors, result, dim);
}

} // namespace torch::xpu
