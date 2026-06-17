#include <ATen/cuda/CUDAGraph.h>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>

#include <algorithm>
#include <cstdint>
#include <vector>

namespace at::cuda {

#if !defined(USE_ROCM) && defined(CUDA_VERSION) && CUDA_VERSION >= 12040

namespace {

__global__ void set_conditional_handle_kernel(
    cudaGraphConditionalHandle handle,
    const bool* value) {
  cudaGraphSetConditional(handle, *value);
}

template <bool VoltaOrLater>
struct KernelUpdateSOA {
  static constexpr size_t kKernelParamLimitBytes = VoltaOrLater ? 32764 : 4096;
  static constexpr size_t kPerUpdate =
      sizeof(void*) + sizeof(size_t) + sizeof(cudaGraphDeviceNode_t);
  static constexpr size_t kMaxNumUpdates =
      (kKernelParamLimitBytes - sizeof(size_t)) / kPerUpdate;

  size_t num_updates;
  cudaGraphDeviceNode_t device_nodes[kMaxNumUpdates];
  void* new_pointers[kMaxNumUpdates];
  size_t param_offsets[kMaxNumUpdates];
};

template <size_t BlockSize, bool VoltaOrLater>
__global__ void __launch_bounds__(BlockSize) graph_kernel_node_updater_kernel(
    __grid_constant__ const KernelUpdateSOA<VoltaOrLater> updates) {
  __shared__ const void* pointer_indirection[BlockSize];
  __shared__ alignas(cudaGraphKernelNodeUpdate) unsigned char update_storage
      [BlockSize * sizeof(cudaGraphKernelNodeUpdate)];
  auto* shared_updates =
      reinterpret_cast<cudaGraphKernelNodeUpdate*>(update_storage);

  size_t i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= updates.num_updates) {
    return;
  }

  pointer_indirection[threadIdx.x] = updates.new_pointers[i];
  auto& update = shared_updates[threadIdx.x];
  update.node = updates.device_nodes[i];
  update.field = cudaGraphKernelNodeFieldParam;
  update.updateData.param.pValue = &pointer_indirection[threadIdx.x];
  update.updateData.param.offset = updates.param_offsets[i];
  update.updateData.param.size = sizeof(void*);

  cudaError_t error = cudaGraphKernelNodeUpdatesApply(&update, 1);
  CUDA_KERNEL_ASSERT_MSG(
      error == cudaSuccess,
      "cudaGraphKernelNodeUpdatesApply did not succeed");
}

template <bool VoltaOrLater>
void launch_device_update_kernel(
    const int64_t* device_nodes,
    const int64_t* param_offsets,
    const int64_t* alloc_indices,
    const int64_t* alloc_offsets,
    const std::vector<void*>& actual_data_ptrs,
    size_t offset,
    size_t num_updates) {
  KernelUpdateSOA<VoltaOrLater> update_soa{};
  update_soa.num_updates = num_updates;

  for (size_t i = 0; i < num_updates; ++i) {
    size_t update_idx = offset + i;
    auto alloc_idx = static_cast<size_t>(alloc_indices[update_idx]);
    update_soa.device_nodes[i] = reinterpret_cast<cudaGraphDeviceNode_t>(
        static_cast<uintptr_t>(device_nodes[update_idx]));
    update_soa.new_pointers[i] = static_cast<char*>(actual_data_ptrs[alloc_idx]) +
        alloc_offsets[update_idx];
    update_soa.param_offsets[i] = param_offsets[update_idx];
  }

  constexpr size_t kBlockSize =
      std::min<size_t>(64, KernelUpdateSOA<VoltaOrLater>::kMaxNumUpdates);
  graph_kernel_node_updater_kernel<kBlockSize, VoltaOrLater>
      <<<(num_updates + kBlockSize - 1) / kBlockSize,
         kBlockSize,
         0,
         getCurrentCUDAStream()>>>(update_soa);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <bool VoltaOrLater>
void launch_device_update_kernels(
    const int64_t* device_nodes,
    const int64_t* param_offsets,
    const int64_t* alloc_indices,
    const int64_t* alloc_offsets,
    const std::vector<void*>& actual_data_ptrs,
    size_t num_updates) {
  constexpr size_t kMaxNumUpdates = KernelUpdateSOA<VoltaOrLater>::kMaxNumUpdates;
  for (size_t offset = 0; offset < num_updates; offset += kMaxNumUpdates) {
    size_t chunk_size = std::min(kMaxNumUpdates, num_updates - offset);
    launch_device_update_kernel<VoltaOrLater>(
        device_nodes,
        param_offsets,
        alloc_indices,
        alloc_offsets,
        actual_data_ptrs,
        offset,
        chunk_size);
  }
}

void check_update_metadata_tensor(const at::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.device().is_cpu(), name, " must be a CPU tensor");
  TORCH_CHECK(tensor.scalar_type() == at::kLong, name, " must have dtype int64");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(tensor.dim() == 1, name, " must be 1-dimensional");
}

} // namespace

void CUDAGraph::set_conditional_handle(
    cudaGraphConditionalHandle handle,
    const Tensor& scalar_cuda_pred_tensor) {
  set_conditional_handle_kernel<<<1, 1, 0, getCurrentCUDAStream()>>>(
      handle, scalar_cuda_pred_tensor.const_data_ptr<bool>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void apply_device_kernel_node_updates(
    const at::Tensor& device_nodes,
    const at::Tensor& param_offsets,
    const at::Tensor& alloc_indices,
    const at::Tensor& alloc_offsets,
    const std::vector<at::Tensor>& dynamic_tensors) {
  check_update_metadata_tensor(device_nodes, "device_nodes");
  check_update_metadata_tensor(param_offsets, "param_offsets");
  check_update_metadata_tensor(alloc_indices, "alloc_indices");
  check_update_metadata_tensor(alloc_offsets, "alloc_offsets");

  size_t num_updates = static_cast<size_t>(device_nodes.numel());
  TORCH_CHECK(
      param_offsets.numel() == device_nodes.numel() &&
          alloc_indices.numel() == device_nodes.numel() &&
          alloc_offsets.numel() == device_nodes.numel(),
      "device kernel node update metadata tensors must have the same length");
  if (num_updates == 0) {
    return;
  }

  std::vector<void*> actual_data_ptrs;
  actual_data_ptrs.reserve(dynamic_tensors.size());
  for (const at::Tensor& tensor : dynamic_tensors) {
    TORCH_CHECK(tensor.is_cuda(), "dynamic_tensors must be CUDA tensors");
    actual_data_ptrs.push_back(tensor.data_ptr());
  }

  const auto* alloc_indices_data = alloc_indices.const_data_ptr<int64_t>();
  const auto* alloc_offsets_data = alloc_offsets.const_data_ptr<int64_t>();
  for (size_t i = 0; i < num_updates; ++i) {
    TORCH_CHECK(alloc_indices_data[i] >= 0, "alloc_indices must be nonnegative");
    TORCH_CHECK(alloc_offsets_data[i] >= 0, "alloc_offsets must be nonnegative");
    TORCH_CHECK(
        static_cast<size_t>(alloc_indices_data[i]) < actual_data_ptrs.size(),
        "alloc_indices entry is out of bounds for dynamic_tensors");
  }

  if (at::cuda::getCurrentDeviceProperties()->major >= 7) {
    launch_device_update_kernels<true>(
        device_nodes.const_data_ptr<int64_t>(),
        param_offsets.const_data_ptr<int64_t>(),
        alloc_indices_data,
        alloc_offsets_data,
        actual_data_ptrs,
        num_updates);
  } else {
    launch_device_update_kernels<false>(
        device_nodes.const_data_ptr<int64_t>(),
        param_offsets.const_data_ptr<int64_t>(),
        alloc_indices_data,
        alloc_offsets_data,
        actual_data_ptrs,
        num_updates);
  }
}

#else

void CUDAGraph::set_conditional_handle(
    cudaGraphConditionalHandle,
    const Tensor&) {
  AT_ERROR("not allowed");
}

void apply_device_kernel_node_updates(
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const std::vector<at::Tensor>&) {
  TORCH_CHECK(
      false,
      "Device-side CUDA graph kernel node updates require CUDA 12.4 or newer");
}

#endif

} // namespace at::cuda
