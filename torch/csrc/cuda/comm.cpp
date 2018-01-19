#include "comm.h"

#include "torch/csrc/utils/tensor_flatten.h"
#include "torch/csrc/utils/auto_gpu.h"
#include "torch/csrc/cuda/device_set.h"
#ifdef WITH_NCCL
#include "torch/csrc/cuda/nccl.h"
#endif

#include <ATen/ATen.h>

namespace torch { namespace cuda {

using namespace at;

// Some operations can be performed more efficiently if we're handling tensors
// of a single type only. Adding this logic directly in the loop makes it a bit
// ugly, so here's a helper for it.
struct unique_type_checker {
  void show(const at::Type& t) {
    if (!unique) return;
    if (!type) type = &t;
    unique = (type == &t);
  }

  const at::Type *type = nullptr;
  bool unique = true;
};

std::vector<Tensor> broadcast(const Tensor& tensor, IntList devices) {
  auto & type = tensor.type();
  if (type.is_cuda() && tensor.get_device() != devices[0])
    throw std::runtime_error("device of broadcasted tensor must appear as the "
                             "first on devices list");
  std::vector<Tensor> tensors;
  tensors.reserve(devices.size());
#ifdef WITH_NCCL
  if (nccl::is_available({tensor})) {
    tensors.push_back(tensor);
    for (auto device : devices.slice(1)) {
      AutoGPU _gpu_guard(device);
      tensors.push_back(type.tensor(tensor.sizes()));
    }
    nccl::broadcast(tensors);
  } else {
#else
  {
#endif
    auto & gpu_type = type.toBackend(type.is_sparse() ? at::kSparseCUDA : at::kCUDA);
    for (auto device : devices) {
      AutoGPU _gpu_guard(device);
      tensors.push_back(gpu_type.copy(tensor, true));
    }
  }
  return tensors;
}

tensor_list2d broadcast_coalesced(TensorList tensors, IntList devices, std::size_t buffer_size) {
  if (!std::all_of(tensors.begin(), tensors.end(),
                   [&](const at::Tensor& t) { return t.get_device() == devices[0]; })) {
    throw std::runtime_error("all tensors must be on devices[0]");
  }

  tensor_list2d outputs(devices.size());
  outputs[0] = tensors;
  for (auto & o : outputs)
    o.reserve(tensors.size());

  unique_type_checker type_checker;
  for (auto & chunk : utils::take_tensors(tensors, buffer_size)) {
    auto & type = chunk.type();
    type_checker.show(type);
    std::vector<at::Tensor> results;
    if (chunk.type().is_sparse()) {
      auto flat_tuple = utils::flatten_sparse_tensors(chunk.tensors);
      std::vector<at::Tensor> broadcast_indices = broadcast(flat_tuple.first, devices);
      std::vector<at::Tensor> broadcast_values = broadcast(flat_tuple.second, devices);
      results.reserve(devices.size());
      for (std::size_t i = 1, num_devices = devices.size(); i < num_devices; ++i) {
        auto & device_outputs = outputs[i];
        auto & inds = broadcast_indices[i];
        auto & vals = broadcast_values[i];
        for (auto & t : utils::unflatten_sparse_tensors(inds, vals, chunk.tensors))
          device_outputs.push_back(std::move(t));
      }
    } else {
      std::vector<Tensor> results = broadcast(utils::flatten_dense_tensors(chunk.tensors),
                                              devices);
      for (std::size_t i = 1, num_devices = devices.size(); i < num_devices; ++i) {
        auto & device_outputs = outputs[i];
        for (auto & t : utils::unflatten_dense_tensors(results[i], chunk.tensors))
          device_outputs.push_back(std::move(t));
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

}}
