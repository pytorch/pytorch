#include <torch/csrc/cuda/comm.h>

#include <torch/csrc/cuda/device_set.h>
#include <torch/csrc/utils/tensor_flatten.h>

#ifdef USE_NCCL
#include <torch/csrc/cuda/nccl.h>
#endif

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/Optional.h>
#include <torch/csrc/autograd/variable.h>

#include <cstddef>
#include <vector>

namespace torch { namespace cuda {
using namespace at;
using namespace torch::autograd;

// Some operations can be performed more efficiently if we're handling tensors
// of a single type only. Adding this logic directly in the loop makes it a bit
// ugly, so here's a helper for it.
struct unique_type_checker {
  void show(const at::DeprecatedTypeProperties& t) {
    if (!unique) return;
    if (!type) type = &t;
    unique = (type == &t);
  }

  const at::DeprecatedTypeProperties *type = nullptr;
  bool unique = true;
};

std::vector<Tensor> broadcast(const Tensor& tensor, IntArrayRef devices) {
  if (tensor.is_cuda() && tensor.get_device() != devices[0])
    throw std::runtime_error("device of broadcasted tensor must appear as the "
                             "first on devices list");
  std::vector<Tensor> tensors;
  tensors.reserve(devices.size());
#ifdef USE_NCCL
  if (nccl::is_available({tensor})) {
    tensors.push_back(tensor);
    for (auto device : devices.slice(1)) {
      tensors.push_back(
          at::empty(tensor.sizes(),
          tensor.options().device(at::Device(kCUDA, device))));
    }
    nccl::broadcast(tensors);
  } else {
#else
  {
#endif
    if (tensor.is_cuda()) {
      tensors.push_back(tensor);
    }
    IntArrayRef loop_devices = tensor.is_cuda() ? devices.slice(1) : devices;
    for (auto device : loop_devices) {
      tensors.push_back(tensor.to(
          at::Device(kCUDA, device),
          tensor.scalar_type(),
          /*non_blocking=*/true,
          /*copy=*/true));
    }
  }
  return tensors;
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
// these Varaibles do not share storage at all.
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
tensor_list2d broadcast_coalesced(TensorList tensors, IntArrayRef devices, size_t buffer_size) {
  if (!std::all_of(tensors.begin(), tensors.end(),
                   [&](const at::Tensor& t) { return t.get_device() == devices[0]; })) {
    throw std::runtime_error("all tensors must be on devices[0]");
  }
#ifdef USE_NCCL
  buffer_size = std::min(torch::cuda::nccl::get_max_count(), buffer_size);
#endif

  tensor_list2d outputs(devices.size());
  outputs[0] = tensors.vec();
  for (auto & o : outputs)
    o.reserve(tensors.size());

  unique_type_checker type_checker;
  at::cuda::CUDAGuard device_guard(devices[0]);
  for (auto & chunk : utils::take_tensors(tensors, buffer_size)) {
    auto & type = chunk.type();
    type_checker.show(type);
    std::vector<at::Tensor> results;
    if (chunk.type().is_sparse()) {
      auto flat_tuple = utils::flatten_sparse_tensors(chunk.tensors);
      std::vector<at::Tensor> broadcast_indices = broadcast(flat_tuple.first, devices);
      std::vector<at::Tensor> broadcast_values = broadcast(flat_tuple.second, devices);
      results.reserve(devices.size());
      for (size_t i = 1, num_devices = devices.size(); i < num_devices; ++i) {
        device_guard.set_index(devices[i]);
        auto & device_outputs = outputs[i];
        auto & inds = broadcast_indices[i];
        auto & vals = broadcast_values[i];
        for (auto & t : utils::unflatten_sparse_tensors(inds, vals, chunk.tensors)) {
          // See NOTE [ Version Counter in comm.*_coalesced ]
          Variable var = t;
          device_outputs.push_back(make_variable(var.tensor_data(), false));
        }
      }
    } else {
      std::vector<Tensor> results = broadcast(utils::flatten_dense_tensors(chunk.tensors),
                                              devices);
      for (size_t i = 1, num_devices = devices.size(); i < num_devices; ++i) {
        device_guard.set_index(devices[i]);
        auto & device_outputs = outputs[i];
        for (auto & t : utils::unflatten_dense_tensors(results[i], chunk.tensors)) {
          // See NOTE [ Version Counter in comm.*_coalesced ]
          Variable var = t;
          device_outputs.push_back(make_variable(var.tensor_data(), false));
        }
      }
    }
  }

  // If we only saw a single tensor type, then we can skip expensive reordering
  if (!type_checker.unique) {
    for (auto & o : outputs)
      utils::reorder_tensors_like(o, tensors);
  }
  return outputs;
}

std::vector<at::Tensor> scatter(
    const at::Tensor& tensor,
    at::IntArrayRef devices,
    const c10::optional<std::vector<int64_t>>& chunk_sizes,
    int64_t dim,
    const c10::optional<std::vector<c10::optional<at::cuda::CUDAStream>>>& streams) {
  std::vector<at::Tensor> chunks;
  if (chunk_sizes) {
    const int64_t chunk_size_sum =
        std::accumulate(chunk_sizes->begin(), chunk_sizes->end(), int64_t{0});
    TORCH_CHECK(
      chunk_size_sum == tensor.size(dim),
      "given chunk sizes don't sum up to the tensor's size ",
      "(sum(chunk_sizes) == ", chunk_size_sum,
      ", but expected ", tensor.size(dim), ")");
    chunks.reserve(chunk_sizes->size());
    int64_t chunk_start = 0;
    for (size_t chunk = 0; chunk < chunk_sizes->size(); ++chunk) {
      const int64_t chunk_size = (*chunk_sizes)[chunk];
      TORCH_CHECK(chunk_size > 0, "Chunk size must be positive");
      chunks.push_back(tensor.narrow(dim, chunk_start, chunk_size));
      chunk_start += chunk_size;
    }
    AT_ASSERT(chunks.size() == chunk_sizes->size());
  } else {
    chunks = tensor.chunk(/*chunks=*/devices.size(), /*dim=*/dim);
  }
  at::cuda::OptionalCUDAStreamGuard cuda_guard;
  for (size_t chunk = 0; chunk < chunks.size(); ++chunk) {
    const auto device_index = static_cast<int16_t>(devices[chunk]);
    if (streams && (*streams)[chunk]) {
      TORCH_CHECK(
          (*streams)[chunk]->device_index() == device_index,
          "Expected the device associated with the stream at index ",
          chunk, " (was ", (*streams)[chunk]->device_index(), ") ",
          "to match the device supplied at that index ",
          "(expected ", device_index, ")");
      cuda_guard.reset_stream(*(*streams)[chunk]);
    }
    chunks[chunk] = chunks[chunk].contiguous().to(
        {at::DeviceType::CUDA, device_index}, /*non_blocking=*/true);
  }
  return chunks;
}

at::Tensor gather(
    at::TensorList tensors,
    int64_t dim,
    c10::optional<int32_t> destination_index) {
  TORCH_CHECK(!tensors.empty(), "Expected at least one tensor to gather from");
  at::Tensor result;
  int64_t total_size = 0;
  auto& first = tensors.front();
  const auto first_size = first.sizes();
  std::vector<int64_t> expected_size(first_size.begin(), first_size.end());
  for (const auto& tensor : tensors) {
    TORCH_CHECK(
        tensor.is_cuda(), "Gather expects all inputs to have CUDA type");
    AT_ASSERT(tensor.ndimension() == static_cast<int64_t>(expected_size.size()));
    expected_size[dim] = tensor.size(dim);
    for (size_t dimension = 0; dimension < expected_size.size(); ++dimension) {
      TORCH_CHECK(
          expected_size[dimension] == tensor.size(dimension),
          "Gather got an input of invalid size: got ",
          tensor.sizes(), ", but expected ", at::IntArrayRef(expected_size));
    }
    total_size += tensor.size(dim);
  }
  expected_size[dim] = total_size;
  at::Device device(at::DeviceType::CPU);
  if (!destination_index || *destination_index != -1) {
    device = at::Device(at::DeviceType::CUDA, destination_index ? *destination_index : -1);
  }
  result = at::empty(expected_size, first.options().device(device));

  int64_t chunk_start = 0;
  for (const auto& tensor : tensors) {
    result.narrow(dim, chunk_start, tensor.size(dim))
        .copy_(tensor, /*non_blocking=*/true);
    chunk_start += tensor.size(dim);
  }
  return result;
}
}} // namespace torch::cuda
