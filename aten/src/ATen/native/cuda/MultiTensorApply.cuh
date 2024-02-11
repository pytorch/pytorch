#pragma once
#include <ATen/ceil_div.h>
#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/cuda/MemoryAccess.cuh>
#include <vector>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/empty.h>
#endif

namespace at::native {

namespace {

static constexpr int64_t kILP = 4;
static constexpr int64_t kChunkSize = 65536;
static constexpr int64_t kBlockSize = 512;

// TODO(crcrpar): Add `n>5` for `low prec params & their higher prec copy`
// TensorListMetadata has to be < 4KB - the limit for kernel launch argument
static constexpr int depth_to_max_tensors[5] = {110, 64, 48, 36, 30};
static constexpr int depth_to_max_blocks[5] = {320, 320, 320, 320, 320};
static constexpr int depth_to_max_tensors_scalarlist[5] = {96, 64, 48, 36, 30};
static constexpr int depth_to_max_tensors_scalarlist_of_complex_double[2] = {
    72,
    60};

template <typename T>
__device__ __forceinline__ bool is_aligned(T* p) {
  return ((uint64_t)p) % (kILP * sizeof(T)) == 0;
}

template <typename T>
__device__ __forceinline__ void load_store(
    T* dst,
    T* src,
    int64_t dst_offset,
    int64_t src_offset) {
  using LT = at::native::memory::aligned_vector<T, kILP>;
  ((LT*)dst)[dst_offset] = ((LT*)src)[src_offset];
}

template <int n>
struct TensorListMetadataStatic {
  const void* addresses[n][depth_to_max_tensors[n - 1]];
  int64_t numel_for_tensor[depth_to_max_tensors[n - 1]];
  unsigned char block_to_tensor[depth_to_max_blocks[n - 1]];
  int block_to_chunk[depth_to_max_blocks[n - 1]];
  int start_tensor_this_launch;
};

template <int n>
struct TensorListMetadata {
  const void** addresses[n];
  int64_t* numel_for_tensor;
  size_t* block_to_tensor;
  size_t* block_to_chunk;
  int start_tensor_this_launch;
};

template <typename scalar_vals_t, int n>
struct TensorListScalarListMetadata {
  const void* addresses[n][depth_to_max_tensors_scalarlist[n - 1]];
  int64_t numel_for_tensor[depth_to_max_tensors_scalarlist[n - 1]];
  scalar_vals_t scalar_vals[depth_to_max_tensors_scalarlist[n - 1]];
  unsigned char block_to_tensor[depth_to_max_blocks[n - 1]];
  int block_to_chunk[depth_to_max_blocks[n - 1]];
};

// note(mkozuki): `n` of 1&2 violate the limit of cuda kernel argument size of
// 4kb with `c10::complex<double>`
template <>
struct TensorListScalarListMetadata<c10::complex<double>, 1> {
  const void* addresses[1]
                       [depth_to_max_tensors_scalarlist_of_complex_double[0]];
  int64_t
      numel_for_tensor[depth_to_max_tensors_scalarlist_of_complex_double[0]];
  c10::complex<double>
      scalar_vals[depth_to_max_tensors_scalarlist_of_complex_double[0]];
  unsigned char block_to_tensor[depth_to_max_blocks[1 - 1]];
  int block_to_chunk[depth_to_max_blocks[1 - 1]];
};

template <>
struct TensorListScalarListMetadata<c10::complex<double>, 2> {
  const void* addresses[2]
                       [depth_to_max_tensors_scalarlist_of_complex_double[1]];
  int64_t
      numel_for_tensor[depth_to_max_tensors_scalarlist_of_complex_double[1]];
  c10::complex<double>
      scalar_vals[depth_to_max_tensors_scalarlist_of_complex_double[1]];
  unsigned char block_to_tensor[depth_to_max_blocks[2 - 1]];
  int block_to_chunk[depth_to_max_blocks[2 - 1]];
};

// NOTE(crcrpar): This is a conservative resolution to handle `state_steps`
// whose each element is `at::Tensor` of 1 element representing the number of
// `step`s called so far.
template <int n>
struct FusedOptimizerTensorListMetadata {
  const void* addresses[n][depth_to_max_tensors[n - 1]];
  int64_t numel_for_tensor[depth_to_max_tensors[n - 1]];
  const void* state_steps_addresses[depth_to_max_tensors_scalarlist[n - 1]];
  unsigned char block_to_tensor[depth_to_max_blocks[n - 1]];
  int block_to_chunk[depth_to_max_blocks[n - 1]];
  int start_tensor_this_launch;
};

bool can_use_static_tensor_list_meta(
    std::vector<std::vector<at::Tensor>>& tensor_lists,
    int depth) {
  const int64_t n_tensors = tensor_lists[0].size();
  if (n_tensors > depth_to_max_tensors[depth - 1]) {
    return false;
  }
  int64_t num_blocks = 0;
  for (const auto t : c10::irange(n_tensors)) {
    const auto numel = tensor_lists[0][t].numel();
    const auto chunks = at::ceil_div(numel, kChunkSize);
    num_blocks += chunks;
    if (num_blocks > depth_to_max_blocks[depth - 1]) {
      return false;
    }
  }
  return true;
}

// Helper for transfering multiple std::vector<T> onto device with a single
// page-locked cudaMemcpyAsync.
struct VecPacker {
  std::vector<const void*> ptrs;
  std::vector<size_t> sizes;
  std::vector<size_t> offsets;
  int64_t packed_numel = 0;
  at::Tensor packed;

  template <typename T>
  // Add a vector to be copied to device
  // NOTE: VecPacker doesn't make copies of the added vectors. They have to be
  // kept alive by the caller until .pack() is called.
  void add(const std::vector<T>& vec) {
    // 16 would cover alignment for the largest known T (c10::complex)
    static const size_t alignment = 16;
    static_assert(alignment % sizeof(T) == 0);
    ptrs.push_back(vec.data());
    const auto vec_bytes = sizeof(T) * vec.size();
    const auto vec_bytes_aligned = at::round_up(vec_bytes, alignment);
    sizes.push_back(vec_bytes);
    offsets.push_back(packed_numel);
    packed_numel += vec_bytes_aligned;
  }

  // Copy all previously added vectors onto device and return their device
  // pointers in the order they are added. We leverage the stream awareness of
  // CUDACachingAllocator to manage the lifetime of the device arguments - the
  // device memory is guaranteed to be alive as long as VecPacker is destroyed
  // after the kernel that consumes it.
  std::vector<void*> pack(const at::Device& device) {
    packed = at::empty(
        {packed_numel},
        at::TensorOptions().dtype(at::kByte).pinned_memory(true));
    for (const auto i : c10::irange(ptrs.size())) {
      memcpy(packed.data_ptr<uint8_t>() + offsets[i], ptrs[i], sizes[i]);
    }
    packed = packed.to(device, /*non_blocking=*/true);

    std::vector<void*> dev_ptrs;
    dev_ptrs.reserve(ptrs.size());
    for (const auto offset : offsets) {
      dev_ptrs.push_back(packed.data_ptr<uint8_t>() + offset);
    }
    return dev_ptrs;
  }
};

template <typename T, typename U, typename... ArgTypes>
C10_LAUNCH_BOUNDS_1(kBlockSize)
__global__ void multi_tensor_apply_kernel(
    T tensorListMeta,
    U callable,
    ArgTypes... args) {
  // Hand the chunk information to the user-supplied functor to process however
  // it likes.
  callable(kChunkSize, tensorListMeta, args...);
}

} // namespace

// multi_tensor_apply enables horizontal fusion across lists of tensors.
// For example, whereas you once had a for-loop of a + b = c, where a, b,
// and c are individual tensors in lists as, bs, and cs, you can now with
// fewer kernel launches compute as + bs = cs.
//
// You can also imagine bs to be a scalar list vs a tensor list.
//
// The function below takes in tensor lists, scalars, and a callable and
// chunks up the computation to launch as few kernels as possible by iterating
// through every "chunk" in every tensor (thus the nested for loops). In the
// simplest case, everything gets bundled into just one kernel launch, but
// due to blocksize constraints, we may need to launch multiple kernels.
// Each kernel launch is defined by one tensorListMeta construct, which we
// use to track and reset the necessary metadata for each launch.
template <int depth, typename scalar_T, typename T, typename... ArgTypes>
void multi_tensor_apply(
    std::vector<std::vector<at::Tensor>>& tensor_lists,
    at::ArrayRef<Scalar> scalars,
    T callable,
    ArgTypes... args) {
  TORCH_CHECK(
      tensor_lists.size() == depth,
      "Number of tensor lists has to match the depth.");
  const size_t n_tensors = tensor_lists[0].size();
  using scalar_vals_t = typename T::opmath_t;
  TensorListScalarListMetadata<scalar_vals_t, depth> tensorListMeta;

  int loc_block_info = 0;
  int loc_tensor_info = 0;
  for (size_t t = 0; t < n_tensors; t++) {
    // short-circuit to avoid adding empty tensors to tensorListMeta
    if (tensor_lists[0][t].numel() == 0) {
      continue;
    }
    tensorListMeta.scalar_vals[loc_tensor_info] = scalars[t].to<scalar_T>();
    tensorListMeta.numel_for_tensor[loc_tensor_info] =
        tensor_lists[0][t].numel();
    for (int d = 0; d < depth; d++) {
      tensorListMeta.addresses[d][loc_tensor_info] =
          tensor_lists[d][t].const_data_ptr();
    }
    loc_tensor_info++;

    // now we enter [chunking territory].
    // we will launch a kernel when EITHER the blocks get filled up OR
    // the tensors get filled up. There will always be at least one block
    // per tensor since the zero-sized ones will not enter the loop, so
    // the nested forloop within represents iterating through the chunks
    // of a single tensor.
    const auto numel = tensor_lists[0][t].numel();
    const auto chunks = numel / kChunkSize + (numel % kChunkSize != 0);
    for (auto chunk = 0; chunk < chunks; chunk++) {
      tensorListMeta.block_to_tensor[loc_block_info] = loc_tensor_info - 1;
      tensorListMeta.block_to_chunk[loc_block_info] = chunk;
      loc_block_info++;

      // a tensor is not considered full unless all its chunks have been
      // processed
      const bool tensors_full =
          (loc_tensor_info == depth_to_max_tensors_scalarlist[depth - 1] &&
           chunk == chunks - 1);
      const bool blocks_full =
          (loc_block_info == depth_to_max_blocks[depth - 1]);

      if (tensors_full || blocks_full) {
        multi_tensor_apply_kernel<<<
            loc_block_info,
            kBlockSize,
            0,
            at::cuda::getCurrentCUDAStream()>>>(
            tensorListMeta, callable, args...);
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        // Reset.
        loc_block_info = 0;
        // all chunks have already been handled in the kernel
        if (chunk == chunks - 1) {
          loc_tensor_info = 0;
        } else { // blocks were full and tensor chunks remain
          tensorListMeta.numel_for_tensor[0] =
              tensorListMeta.numel_for_tensor[loc_tensor_info - 1];
          tensorListMeta.scalar_vals[0] =
              tensorListMeta.scalar_vals[loc_tensor_info - 1];
          for (int d = 0; d < depth; d++) {
            tensorListMeta.addresses[d][0] =
                tensorListMeta.addresses[d][loc_tensor_info - 1];
          }
          loc_tensor_info = 1;
        }
      }
    }
  }

  // note: [finishing what we started]
  // if there's remaining work to be done but the tensors/blocks aren't full
  // yet we are at the end, submit the kernel to do the work!
  if (loc_block_info != 0) {
    multi_tensor_apply_kernel<<<
        loc_block_info,
        kBlockSize,
        0,
        at::cuda::getCurrentCUDAStream()>>>(tensorListMeta, callable, args...);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}

template <int depth, typename T, typename... ArgTypes>
void multi_tensor_apply_static(
    std::vector<std::vector<at::Tensor>>& tensor_lists,
    T callable,
    ArgTypes... args) {
  TORCH_CHECK(
      tensor_lists.size() == depth,
      "Number of tensor lists has to match the depth.");
  const size_t n_tensors = tensor_lists[0].size();
  TensorListMetadataStatic<depth> tensorListMeta;
  tensorListMeta.start_tensor_this_launch = 0;

  int loc_block_info = 0;
  int loc_tensor_info = 0;
  for (size_t t = 0; t < n_tensors; t++) {
    // short-circuit to avoid adding empty tensors to tensorListMeta
    if (tensor_lists[0][t].numel() == 0) {
      continue;
    }
    tensorListMeta.numel_for_tensor[loc_tensor_info] =
        tensor_lists[0][t].numel();
    for (int d = 0; d < depth; d++) {
      tensorListMeta.addresses[d][loc_tensor_info] =
          tensor_lists[d][t].const_data_ptr();
    }
    loc_tensor_info++;

    const auto numel = tensor_lists[0][t].numel();
    const auto chunks = numel / kChunkSize + (numel % kChunkSize != 0);
    for (auto chunk = 0; chunk < chunks; chunk++) {
      tensorListMeta.block_to_tensor[loc_block_info] = loc_tensor_info - 1;
      tensorListMeta.block_to_chunk[loc_block_info] = chunk;
      loc_block_info++;
    }
  }
  TORCH_CHECK(loc_tensor_info < depth_to_max_tensors[depth - 1]);
  TORCH_CHECK(loc_block_info < depth_to_max_blocks[depth - 1]);

  if (loc_block_info != 0) {
    multi_tensor_apply_kernel<<<
        loc_block_info,
        kBlockSize,
        0,
        at::cuda::getCurrentCUDAStream()>>>(tensorListMeta, callable, args...);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}

template <int depth, typename T, typename... ArgTypes>
void multi_tensor_apply(
    std::vector<std::vector<at::Tensor>>& tensor_lists,
    T callable,
    ArgTypes... args) {
  // Note: [static arg vs. dynamic arg]
  // Due to the dynamic nature of the workload, the kernel arguments aren't
  // guaranteed to fit in the static 4kb kernel argument memory. Previously
  // with the apex implementation, we overcame this limitation by dividing a
  // multi_tensor_apply workload into multiple kernel launches. However, this
  // led to low sustained occupancy, affecting the performance of memory bound
  // ops.
  //
  // Based on the observation that the kernel argument memory limitation
  // doesn't correlate well with available SM resources, we have adopted a
  // different approach. When the kernel arguments fit into the static kernel
  // argument memory, we use this memory to transfer the arguments. Conversely,
  // when the kernel arguments don't fit into the static kernel argument
  // memory, instead of sacrificing sustained occupancy, we use a page-locked
  // cudaMemcpyAsync to transfer the arguments, then perform the entire
  // workload in a single kernel.
  if (can_use_static_tensor_list_meta(tensor_lists, depth)) {
    multi_tensor_apply_static<depth, T, ArgTypes...>(
        tensor_lists, callable, args...);
    return;
  }

  TORCH_CHECK(
      tensor_lists.size() == depth,
      "Number of tensor lists has to match the depth.");
  const size_t n_tensors = tensor_lists[0].size();

  std::vector<const void*> addresses[depth];
  std::vector<int64_t> numel_for_tensor;
  std::vector<size_t> block_to_tensor;
  std::vector<size_t> block_to_chunk;

  for (int d = 0; d < depth; ++d) {
    addresses[d].reserve(n_tensors);
  }
  numel_for_tensor.reserve(n_tensors);
  block_to_tensor.reserve(n_tensors); // reserve for lowerbound
  block_to_chunk.reserve(n_tensors); // reserve for lowerbound

  for (size_t t = 0; t < n_tensors; t++) {
    const auto numel = tensor_lists[0][t].numel();
    // short-circuit to avoid adding empty tensors to tensorListMeta
    if (numel == 0) {
      continue;
    }
    numel_for_tensor.push_back(numel);
    for (int d = 0; d < depth; d++) {
      addresses[d].push_back(tensor_lists[d][t].const_data_ptr());
    }
    const auto chunks = at::ceil_div(numel, kChunkSize);
    block_to_tensor.insert(block_to_tensor.end(), chunks, t);
    block_to_chunk.resize(block_to_chunk.size() + chunks);
    std::iota(block_to_chunk.end() - chunks, block_to_chunk.end(), 0);
  }

  VecPacker packer;
  for (auto d = 0; d < depth; ++d) {
    packer.add(addresses[d]);
  }
  packer.add(numel_for_tensor);
  packer.add(block_to_tensor);
  packer.add(block_to_chunk);

  auto device = tensor_lists[0][0].device();
  auto dev_ptrs = packer.pack(device);

  TensorListMetadata<depth> tl;
  for (auto d = 0; d < depth; ++d) {
    tl.addresses[d] = static_cast<const void**>(dev_ptrs[d]);
  }
  tl.numel_for_tensor = static_cast<int64_t*>(dev_ptrs[depth]);
  tl.block_to_tensor = static_cast<size_t*>(dev_ptrs[depth + 1]);
  tl.block_to_chunk = static_cast<size_t*>(dev_ptrs[depth + 2]);
  tl.start_tensor_this_launch = 0;

  if (block_to_tensor.size() > 0) {
    multi_tensor_apply_kernel<<<
        block_to_tensor.size(),
        kBlockSize,
        0,
        at::cuda::getCurrentCUDAStream()>>>(tl, callable, args...);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}

template <int depth, typename T, typename... ArgTypes>
void multi_tensor_apply_for_fused_optimizer(
    std::vector<std::vector<at::Tensor>>& tensor_lists,
    at::TensorList state_steps,
    T callable,
    ArgTypes... args) {
  TORCH_CHECK(
      tensor_lists.size() == depth,
      "Number of tensor lists has to match the depth");
  const auto num_tensors = tensor_lists[0].size();
  FusedOptimizerTensorListMetadata<depth> tensorListMeta;

  int loc_block_info = 0;
  int loc_tensor_info = 0;
  for (const auto& tensor_index : c10::irange(num_tensors)) {
    // short-circuit to avoid adding empty tensors to tensorListMeta
    if (tensor_lists[0][tensor_index].numel() == 0) {
      continue;
    }
    tensorListMeta.state_steps_addresses[loc_tensor_info] =
        state_steps[tensor_index].const_data_ptr();
    tensorListMeta.numel_for_tensor[loc_tensor_info] =
        tensor_lists[0][tensor_index].numel();
    for (const auto& d : c10::irange(depth)) {
      tensorListMeta.addresses[d][loc_tensor_info] =
          tensor_lists[d][tensor_index].const_data_ptr();
    }
    loc_tensor_info++;

    // see above note: [chunking territory]
    const auto numel = tensor_lists[0][tensor_index].numel();
    const auto chunks = numel / kChunkSize + (numel % kChunkSize != 0);
    TORCH_CHECK(chunks > -1);
    for (const auto& chunk : c10::irange(chunks)) {
      tensorListMeta.block_to_tensor[loc_block_info] = loc_tensor_info - 1;
      tensorListMeta.block_to_chunk[loc_block_info] = chunk;
      loc_block_info++;

      const auto tensor_full =
          (loc_tensor_info == depth_to_max_tensors[depth - 1] &&
           chunk == chunks - 1);
      const auto blocks_full = loc_block_info == depth_to_max_blocks[depth - 1];

      if (tensor_full || blocks_full) {
        multi_tensor_apply_kernel<<<
            loc_block_info,
            kBlockSize,
            0,
            at::cuda::getCurrentCUDAStream()>>>(
            tensorListMeta, callable, args...);
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        // Reset.
        loc_block_info = 0;
        if (chunk == chunks - 1) {
          loc_tensor_info = 0;
        } else {
          tensorListMeta.numel_for_tensor[0] =
              tensorListMeta.numel_for_tensor[loc_tensor_info - 1];
          tensorListMeta.state_steps_addresses[0] =
              tensorListMeta.state_steps_addresses[loc_tensor_info - 1];
          for (const auto& d : c10::irange(depth)) {
            tensorListMeta.addresses[d][0] =
                tensorListMeta.addresses[d][loc_tensor_info - 1];
          }
          loc_tensor_info = 1;
        }
      }
    }
  }

  // see above note: [finishing what we've started]
  if (loc_block_info != 0) {
    multi_tensor_apply_kernel<<<
        loc_block_info,
        kBlockSize,
        0,
        at::cuda::getCurrentCUDAStream()>>>(tensorListMeta, callable, args...);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}

} // namespace at::native
