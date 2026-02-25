// Torch extension wrapper for vLLM's custom allreduce kernels.
// Exposes IPC buffer management and allreduce as Python-callable functions.
// JIT compiled via torch.utils.cpp_extension.load().

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstring>

#include "custom_all_reduce.cuh"

#define CUDACHECK_TORCH(cmd)                                              \
  do {                                                                    \
    cudaError_t e = cmd;                                                  \
    TORCH_CHECK(e == cudaSuccess, "CUDA error: ", cudaGetErrorString(e)); \
  } while (0)

// Global state per process (one CustomAllreduce instance at a time)
static std::unique_ptr<vllm::CustomAllreduce> g_custom_ar;
static void* g_rank_data = nullptr;
static constexpr size_t RANK_DATA_SZ = 16 * 1024 * 1024;
// Pointer to this rank's registered data region (inside the IPC buffer)
static void* g_registered_data_ptr = nullptr;
static int64_t g_registered_data_elems = 0;
static vllm::RankData* g_primary_rank_data_dev = nullptr;

// Second buffer slot for symm_mem nosync benchmarks (buf A)
static vllm::RankData* g_symm_rank_data_dev = nullptr;
static void* g_symm_data_ptr = nullptr;
static int64_t g_symm_data_elems = 0;

// Third buffer slot for double-buffered nosync benchmarks (buf B)
static vllm::RankData* g_symm_rank_data_dev_b = nullptr;
static void* g_symm_data_ptr_b = nullptr;
static int64_t g_symm_data_elems_b = 0;

// Allocate a device buffer and return its IPC handle as a byte tensor.
// Returns (device_ptr_as_int, ipc_handle_tensor)
std::vector<torch::Tensor> allocate_and_get_handle(int64_t size_bytes) {
  void* ptr;
  CUDACHECK_TORCH(cudaMalloc(&ptr, size_bytes));
  CUDACHECK_TORCH(cudaMemset(ptr, 0, size_bytes));

  cudaIpcMemHandle_t handle;
  CUDACHECK_TORCH(cudaIpcGetMemHandle(&handle, ptr));

  // Pack pointer as int64 tensor
  auto ptr_tensor = torch::tensor({reinterpret_cast<int64_t>(ptr)}, torch::kInt64);
  // Pack IPC handle as byte tensor
  auto handle_tensor = torch::empty({static_cast<int64_t>(sizeof(handle))}, torch::kUInt8);
  std::memcpy(handle_tensor.data_ptr(), &handle, sizeof(handle));

  return {ptr_tensor, handle_tensor};
}

// Open an IPC handle (byte tensor) and return the device pointer as int64.
torch::Tensor open_ipc_handle(torch::Tensor handle_tensor) {
  TORCH_CHECK(
      handle_tensor.numel() == sizeof(cudaIpcMemHandle_t),
      "Handle tensor size mismatch");
  cudaIpcMemHandle_t handle;
  std::memcpy(&handle, handle_tensor.data_ptr(), sizeof(handle));

  void* ptr;
  CUDACHECK_TORCH(
      cudaIpcOpenMemHandle(&ptr, handle, cudaIpcMemLazyEnablePeerAccess));
  return torch::tensor({reinterpret_cast<int64_t>(ptr)}, torch::kInt64);
}

// Initialize CustomAllreduce from an array of signal pointers (as int64 tensor).
void init_custom_ar(torch::Tensor signal_ptrs_tensor, int64_t rank,
                    int64_t world_size) {
  TORCH_CHECK(signal_ptrs_tensor.numel() == world_size);
  auto* signal_ptrs_data = signal_ptrs_tensor.data_ptr<int64_t>();

  vllm::Signal* signals[8];
  for (int i = 0; i < world_size; i++) {
    signals[i] = reinterpret_cast<vllm::Signal*>(signal_ptrs_data[i]);
  }

  CUDACHECK_TORCH(cudaMalloc(&g_rank_data, RANK_DATA_SZ));
  g_custom_ar = std::make_unique<vllm::CustomAllreduce>(
      signals, g_rank_data, RANK_DATA_SZ, rank, world_size);
}

// Register a buffer: data_ptrs_tensor is int64 tensor of length world_size
// containing the data pointers from each rank.
// local_rank: which index in data_ptrs_tensor is this rank's pointer
// max_elems: number of elements the data region can hold
void register_buffer(torch::Tensor data_ptrs_tensor, int64_t local_rank,
                     int64_t max_elems) {
  TORCH_CHECK(g_custom_ar != nullptr, "CustomAllreduce not initialized");
  auto* ptrs_data = data_ptrs_tensor.data_ptr<int64_t>();
  void* ptrs[8];
  for (int i = 0; i < g_custom_ar->world_size_; i++) {
    ptrs[i] = reinterpret_cast<void*>(ptrs_data[i]);
  }
  g_custom_ar->register_buffer(ptrs);
  g_registered_data_ptr = ptrs[local_rank];
  g_registered_data_elems = max_elems;
  g_primary_rank_data_dev = g_custom_ar->buffers_[g_registered_data_ptr];
}

// Register symm_mem buffer pointers for nosync benchmarks.
// Uses the custom AR barrier infrastructure with symm_mem data buffers.
void register_symm_buffer(torch::Tensor data_ptrs_tensor, int64_t local_rank,
                          int64_t max_elems) {
  TORCH_CHECK(g_custom_ar != nullptr, "CustomAllreduce not initialized");
  auto* ptrs_data = data_ptrs_tensor.data_ptr<int64_t>();
  vllm::RankData data;
  for (int i = 0; i < g_custom_ar->world_size_; i++) {
    data.ptrs[i] = reinterpret_cast<void*>(ptrs_data[i]);
  }
  CUDACHECK_TORCH(cudaMalloc(&g_symm_rank_data_dev, sizeof(vllm::RankData)));
  CUDACHECK_TORCH(cudaMemcpy(g_symm_rank_data_dev, &data,
                              sizeof(vllm::RankData),
                              cudaMemcpyHostToDevice));
  g_symm_data_ptr = const_cast<void*>(data.ptrs[local_rank]);
  g_symm_data_elems = max_elems;
}

// Run allreduce. inp and out are torch tensors (bf16 or fp16).
// They must point into a registered buffer region.
void allreduce(torch::Tensor inp, torch::Tensor out) {
  TORCH_CHECK(g_custom_ar != nullptr, "CustomAllreduce not initialized");
  TORCH_CHECK(inp.is_cuda() && out.is_cuda());
  TORCH_CHECK(inp.numel() == out.numel());

  auto stream = c10::cuda::getCurrentCUDAStream().stream();
  int size = inp.numel();

  if (inp.scalar_type() == at::ScalarType::BFloat16) {
    g_custom_ar->allreduce<nv_bfloat16>(
        stream, reinterpret_cast<nv_bfloat16*>(inp.data_ptr()),
        reinterpret_cast<nv_bfloat16*>(out.data_ptr()), size);
  } else if (inp.scalar_type() == at::ScalarType::Half) {
    g_custom_ar->allreduce<half>(
        stream, reinterpret_cast<half*>(inp.data_ptr()),
        reinterpret_cast<half*>(out.data_ptr()), size);
  } else {
    TORCH_CHECK(false, "Unsupported dtype for custom allreduce");
  }
}

// High-level allreduce: copies inp into the registered region, runs allreduce,
// copies result to out. inp and out are regular torch tensors.
void allreduce_with_copy(torch::Tensor inp, torch::Tensor out) {
  TORCH_CHECK(g_custom_ar != nullptr, "CustomAllreduce not initialized");
  TORCH_CHECK(g_registered_data_ptr != nullptr, "No buffer registered");
  TORCH_CHECK(inp.is_cuda() && out.is_cuda());
  int64_t n = inp.numel();
  TORCH_CHECK(n == out.numel());
  TORCH_CHECK(n <= g_registered_data_elems, "Tensor too large for registered buffer");

  auto stream = c10::cuda::getCurrentCUDAStream().stream();
  size_t bytes = n * inp.element_size();

  // Copy input into the registered region
  CUDACHECK_TORCH(cudaMemcpyAsync(g_registered_data_ptr, inp.data_ptr(),
                                   bytes, cudaMemcpyDeviceToDevice, stream));

  // Run allreduce (output goes to out's storage, but we need a registered
  // output pointer too — use same registered region for in-place)
  if (inp.scalar_type() == at::ScalarType::BFloat16) {
    auto* ptr = reinterpret_cast<nv_bfloat16*>(g_registered_data_ptr);
    g_custom_ar->allreduce<nv_bfloat16>(stream, ptr, ptr, n);
  } else if (inp.scalar_type() == at::ScalarType::Half) {
    auto* ptr = reinterpret_cast<half*>(g_registered_data_ptr);
    g_custom_ar->allreduce<half>(stream, ptr, ptr, n);
  } else {
    TORCH_CHECK(false, "Unsupported dtype for custom allreduce");
  }

  // Copy result out
  CUDACHECK_TORCH(cudaMemcpyAsync(out.data_ptr(), g_registered_data_ptr,
                                   bytes, cudaMemcpyDeviceToDevice, stream));
}

// Allreduce directly on the registered buffer (no copies).
// Assumes data is already in the registered region.
void allreduce_inplace(int64_t numel, const std::string& dtype) {
  TORCH_CHECK(g_custom_ar != nullptr, "CustomAllreduce not initialized");
  TORCH_CHECK(g_registered_data_ptr != nullptr, "No buffer registered");
  TORCH_CHECK(numel <= g_registered_data_elems);

  auto stream = c10::cuda::getCurrentCUDAStream().stream();
  if (dtype == "bf16") {
    auto* ptr = reinterpret_cast<nv_bfloat16*>(g_registered_data_ptr);
    g_custom_ar->allreduce<nv_bfloat16>(stream, ptr, ptr, numel);
  } else if (dtype == "fp16") {
    auto* ptr = reinterpret_cast<half*>(g_registered_data_ptr);
    g_custom_ar->allreduce<half>(stream, ptr, ptr, numel);
  } else {
    TORCH_CHECK(false, "Unsupported dtype: " + dtype);
  }
}

// Dispatch helpers for nosync and 2-stage kernels on arbitrary buffer pointers.
template <typename T>
void dispatch_1stage_nosync(cudaStream_t stream, vllm::RankData* ptrs,
                            T* result, int numel) {
  auto d = vllm::packed_t<T>::P::size;
  int size = numel / d;
  int threads = 512;
  int blocks = std::min(36, (size + threads - 1) / threads);
  auto& sg = g_custom_ar->sg_;
  auto* self_sg = g_custom_ar->self_sg_;
  int rk = g_custom_ar->rank_;

#define NOSYNC_CASE(ngpus)                                                  \
  case ngpus:                                                               \
    vllm::cross_device_reduce_1stage_nosync<T, ngpus>                       \
        <<<blocks, threads, 0, stream>>>(ptrs, sg, self_sg, result, rk,     \
                                         size);                             \
    break;
  switch (g_custom_ar->world_size_) {
    NOSYNC_CASE(2)
    NOSYNC_CASE(4)
    NOSYNC_CASE(8)
  }
#undef NOSYNC_CASE
}

template <typename T>
void dispatch_2stage(cudaStream_t stream, vllm::RankData* ptrs, T* result,
                     int numel) {
  auto d = vllm::packed_t<T>::P::size;
  int size = numel / d;
  int threads = 512;
  int blocks = std::min(36, (size + threads - 1) / threads);
  auto& sg = g_custom_ar->sg_;
  auto* self_sg = g_custom_ar->self_sg_;
  int rk = g_custom_ar->rank_;

#define STAGE2_CASE(ngpus)                                                  \
  case ngpus:                                                               \
    vllm::cross_device_reduce_2stage<T, ngpus>                              \
        <<<blocks, threads, 0, stream>>>(ptrs, sg, self_sg, result, rk,     \
                                         size);                             \
    break;
  switch (g_custom_ar->world_size_) {
    STAGE2_CASE(2)
    STAGE2_CASE(4)
    STAGE2_CASE(8)
  }
#undef STAGE2_CASE
}

// Nosync allreduce on the primary (custom AR) registered buffer.
void allreduce_inplace_nosync(int64_t numel, const std::string& dtype) {
  TORCH_CHECK(g_custom_ar != nullptr && g_primary_rank_data_dev != nullptr);
  TORCH_CHECK(numel <= g_registered_data_elems);
  auto stream = c10::cuda::getCurrentCUDAStream().stream();
  if (dtype == "bf16") {
    dispatch_1stage_nosync(stream, g_primary_rank_data_dev,
                           reinterpret_cast<nv_bfloat16*>(g_registered_data_ptr),
                           numel);
  } else {
    TORCH_CHECK(false, "Unsupported dtype: " + dtype);
  }
}

// Nosync 1-stage allreduce on symm_mem buffer (one_shot_nosync).
void allreduce_symm_nosync(int64_t numel, const std::string& dtype) {
  TORCH_CHECK(g_custom_ar != nullptr && g_symm_rank_data_dev != nullptr);
  TORCH_CHECK(numel <= g_symm_data_elems);
  auto stream = c10::cuda::getCurrentCUDAStream().stream();
  if (dtype == "bf16") {
    dispatch_1stage_nosync(stream, g_symm_rank_data_dev,
                           reinterpret_cast<nv_bfloat16*>(g_symm_data_ptr),
                           numel);
  } else {
    TORCH_CHECK(false, "Unsupported dtype: " + dtype);
  }
}

// 2-stage allreduce on symm_mem buffer (two_shot_nosync).
// The 2-stage kernel already has no final end barrier.
void allreduce_symm_2stage(int64_t numel, const std::string& dtype) {
  TORCH_CHECK(g_custom_ar != nullptr && g_symm_rank_data_dev != nullptr);
  TORCH_CHECK(numel <= g_symm_data_elems);
  auto stream = c10::cuda::getCurrentCUDAStream().stream();
  if (dtype == "bf16") {
    dispatch_2stage(stream, g_symm_rank_data_dev,
                    reinterpret_cast<nv_bfloat16*>(g_symm_data_ptr), numel);
  } else {
    TORCH_CHECK(false, "Unsupported dtype: " + dtype);
  }
}

// Register second symm_mem buffer for double-buffered nosync benchmarks.
void register_symm_buffer_b(torch::Tensor data_ptrs_tensor, int64_t local_rank,
                             int64_t max_elems) {
  TORCH_CHECK(g_custom_ar != nullptr, "CustomAllreduce not initialized");
  auto* ptrs_data = data_ptrs_tensor.data_ptr<int64_t>();
  vllm::RankData data;
  for (int i = 0; i < g_custom_ar->world_size_; i++) {
    data.ptrs[i] = reinterpret_cast<void*>(ptrs_data[i]);
  }
  CUDACHECK_TORCH(cudaMalloc(&g_symm_rank_data_dev_b, sizeof(vllm::RankData)));
  CUDACHECK_TORCH(cudaMemcpy(g_symm_rank_data_dev_b, &data,
                              sizeof(vllm::RankData),
                              cudaMemcpyHostToDevice));
  g_symm_data_ptr_b = const_cast<void*>(data.ptrs[local_rank]);
  g_symm_data_elems_b = max_elems;
}

// Dispatch synced 1-stage (WITH end barrier) on arbitrary buffer.
template <typename T>
void dispatch_1stage_sync(cudaStream_t stream, vllm::RankData* ptrs,
                          T* result, int numel) {
  auto d = vllm::packed_t<T>::P::size;
  int size = numel / d;
  int threads = 512;
  int blocks = std::min(36, (size + threads - 1) / threads);
  auto& sg = g_custom_ar->sg_;
  auto* self_sg = g_custom_ar->self_sg_;
  int rk = g_custom_ar->rank_;

#define SYNC_CASE(ngpus)                                                    \
  case ngpus:                                                               \
    vllm::cross_device_reduce_1stage<T, ngpus>                              \
        <<<blocks, threads, 0, stream>>>(ptrs, sg, self_sg, result, rk,     \
                                         size);                             \
    break;
  switch (g_custom_ar->world_size_) {
    SYNC_CASE(2)
    SYNC_CASE(4)
    SYNC_CASE(8)
  }
#undef SYNC_CASE
}

// Nosync allreduce on symm_mem buffer B.
void allreduce_symm_nosync_b(int64_t numel, const std::string& dtype) {
  TORCH_CHECK(g_custom_ar != nullptr && g_symm_rank_data_dev_b != nullptr);
  TORCH_CHECK(numel <= g_symm_data_elems_b);
  auto stream = c10::cuda::getCurrentCUDAStream().stream();
  if (dtype == "bf16") {
    dispatch_1stage_nosync(stream, g_symm_rank_data_dev_b,
                           reinterpret_cast<nv_bfloat16*>(g_symm_data_ptr_b),
                           numel);
  } else {
    TORCH_CHECK(false, "Unsupported dtype: " + dtype);
  }
}

// Synced 1-stage allreduce on symm_mem buffer A (WITH end barrier).
void allreduce_symm_sync_a(int64_t numel, const std::string& dtype) {
  TORCH_CHECK(g_custom_ar != nullptr && g_symm_rank_data_dev != nullptr);
  TORCH_CHECK(numel <= g_symm_data_elems);
  auto stream = c10::cuda::getCurrentCUDAStream().stream();
  if (dtype == "bf16") {
    dispatch_1stage_sync(stream, g_symm_rank_data_dev,
                         reinterpret_cast<nv_bfloat16*>(g_symm_data_ptr),
                         numel);
  } else {
    TORCH_CHECK(false, "Unsupported dtype: " + dtype);
  }
}

// Synced 1-stage allreduce on symm_mem buffer B (WITH end barrier).
void allreduce_symm_sync_b(int64_t numel, const std::string& dtype) {
  TORCH_CHECK(g_custom_ar != nullptr && g_symm_rank_data_dev_b != nullptr);
  TORCH_CHECK(numel <= g_symm_data_elems_b);
  auto stream = c10::cuda::getCurrentCUDAStream().stream();
  if (dtype == "bf16") {
    dispatch_1stage_sync(stream, g_symm_rank_data_dev_b,
                         reinterpret_cast<nv_bfloat16*>(g_symm_data_ptr_b),
                         numel);
  } else {
    TORCH_CHECK(false, "Unsupported dtype: " + dtype);
  }
}

void dispose() {
  g_custom_ar.reset();
  if (g_rank_data) {
    cudaFree(g_rank_data);
    g_rank_data = nullptr;
  }
  if (g_symm_rank_data_dev) {
    cudaFree(g_symm_rank_data_dev);
    g_symm_rank_data_dev = nullptr;
  }
  if (g_symm_rank_data_dev_b) {
    cudaFree(g_symm_rank_data_dev_b);
    g_symm_rank_data_dev_b = nullptr;
  }
  g_primary_rank_data_dev = nullptr;
  g_symm_data_ptr = nullptr;
  g_symm_data_ptr_b = nullptr;
}

// Return a bf16 tensor wrapping the registered IPC data region.
// This allows matmuls to write directly into the IPC buffer.
torch::Tensor get_registered_buffer_tensor(int64_t numel) {
  TORCH_CHECK(g_registered_data_ptr != nullptr, "No buffer registered");
  TORCH_CHECK(numel <= g_registered_data_elems, "Requested size exceeds registered buffer");
  auto options = torch::TensorOptions()
      .dtype(torch::kBFloat16)
      .device(torch::kCUDA);
  return torch::from_blob(g_registered_data_ptr, {numel}, options);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("allocate_and_get_handle", &allocate_and_get_handle);
  m.def("open_ipc_handle", &open_ipc_handle);
  m.def("init_custom_ar", &init_custom_ar);
  m.def("register_buffer", &register_buffer);
  m.def("register_symm_buffer", &register_symm_buffer);
  m.def("allreduce", &allreduce);
  m.def("allreduce_with_copy", &allreduce_with_copy);
  m.def("allreduce_inplace", &allreduce_inplace);
  m.def("allreduce_inplace_nosync", &allreduce_inplace_nosync);
  m.def("allreduce_symm_nosync", &allreduce_symm_nosync);
  m.def("allreduce_symm_2stage", &allreduce_symm_2stage);
  m.def("register_symm_buffer_b", &register_symm_buffer_b);
  m.def("allreduce_symm_nosync_b", &allreduce_symm_nosync_b);
  m.def("allreduce_symm_sync_a", &allreduce_symm_sync_a);
  m.def("allreduce_symm_sync_b", &allreduce_symm_sync_b);
  m.def("get_registered_buffer_tensor", &get_registered_buffer_tensor);
  m.def("dispose", &dispose);
}
