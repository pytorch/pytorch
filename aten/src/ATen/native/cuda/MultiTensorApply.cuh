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
#include <ATen/ops/from_blob.h>
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

static constexpr size_t max_kernel_arg_size = 4096;

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

// NOTE [small buffer optimization]
//
// The multi_tensor_apply kernel and the functors take a variant of
// TensorListMetadata as an argument. TensorListMetadata contains multiple
// arrays with dynamic sizes that may not fit within the 4KB kernel argument
// space. Therefore, a dynamically allocated kernel argument is required to
// support arbitrary problem sizes. This can be achieved by preparing the
// arrays in host memory and copying them to the device with a single
// cudaMemcpyAsync. When the problem size is large, the latency of the HtoD
// copy is negligible.
//
// However, when the problem size is small, the latency of the cudaMemcpyAsync
// becomes more pronounced. In such cases, we attempt to fit the arrays into
// the 4KB kernel argument space. This technique is akin to the small buffer
// optimization.
//
// DevArrayPack is a struct used for packing multiple vectors into a contiguous
// buffer. When the combined size of the vectors is small enough, they are
// packed into the struct itself, which is passed to the kernel through the
// static kernel argument space. Otherwise, the arguments are packed into a
// dynamically allocated buffer.
//
// NOTE: Previously, we divided a problem into multiple kernel launches so that
// the arguments for every launch would fit in the static kernel argument
// space. This approach avoided the need for dynamically allocated kernel
// arguments, but it led to multiple kernel launches and lower sustained
// occupancy.
struct DevArrayPack {
  static constexpr size_t max_arrays = 8;
  // The buffer size is selected to ensure that the combined argument size of
  // multi_tensor_apply_kernel does not exceed max_kernel_arg_size for all
  // template specializations. This is enforced at compile time with a static
  // assertion.
  static constexpr size_t small_buffer_size = 3360;

  char small_buffer[small_buffer_size];
  // i-th array => (buffer_ptr ? buffer_ptr : small_buffer) + offsets[i]
  size_t offsets[max_arrays];
  // When small_buffer is used, buffer_ptr is nullptr
  char* buffer_ptr = nullptr;

  template <typename T>
  C10_HOST_DEVICE __forceinline__ void get_array(T*& out, int idx) {
// The small buffer optimization is disabled for hipcc
#if defined(USE_ROCM)
    constexpr bool use_small_buffer = false;
#else
    bool use_small_buffer = (buffer_ptr == nullptr);
#endif
    if (use_small_buffer) {
      out = reinterpret_cast<T*>(small_buffer + offsets[idx]);
    } else {
      out = reinterpret_cast<T*>(buffer_ptr + offsets[idx]);
    }
  }
};

template <typename T>
void process_vector(
    std::vector<const void*>& ptrs,
    std::vector<size_t>& sizes,
    std::vector<size_t>& offsets,
    size_t& total_bytes,
    const std::vector<T>& vec) {
  ptrs.push_back(vec.data());
  // Align the offset by sizeof(T)
  total_bytes = at::round_up(total_bytes, sizeof(T));
  offsets.push_back(total_bytes);
  sizes.push_back(sizeof(T) * vec.size());
  total_bytes += sizeof(T) * vec.size();
}

template <typename T, typename... Ts>
void process_vectors(
    std::vector<const void*>& ptrs,
    std::vector<size_t>& sizes,
    std::vector<size_t>& offsets,
    size_t& total_bytes,
    const std::vector<T>& first,
    const std::vector<Ts>&... vecs) {
  process_vector(ptrs, sizes, offsets, total_bytes, first);
  if constexpr (sizeof...(Ts) > 0) {
    process_vectors(ptrs, sizes, offsets, total_bytes, vecs...);
  }
}

// Pack multiple vectors into a DevArrayPack
//
// When the total size of the input vectors is large, the function packs the
// vectors into a device buffer allocated with CUDACachingAllocator. Thanks to
// the allocator's stream awareness, the buffer can be safely accessed by the
// kernel as long as the tensor owning the buffer is alive at the time of the
// kernel launch. This tensor is returned to the caller, who is responsible for
// keeping it alive until the kernel is launched.
template <int n, typename... Vectors>
std::tuple<DevArrayPack, c10::optional<at::Tensor>> pack_vectors(
    const at::Device& device,
    std::vector<const void*> (&addresses)[n],
    const Vectors&... vecs) {
  std::vector<const void*> ptrs;
  std::vector<size_t> sizes;
  std::vector<size_t> offsets;
  size_t total_bytes = 0;

  for (auto d = 0; d < n; ++d) {
    process_vector(ptrs, sizes, offsets, total_bytes, addresses[d]);
  }
  process_vectors(ptrs, sizes, offsets, total_bytes, vecs...);

  TORCH_CHECK(ptrs.size() <= DevArrayPack::max_arrays);
  DevArrayPack pack{};

// hipcc generates very inefficient code for the small buffer optimization.
// Disabling the small buffer optimization for hipcc for now.
#if !defined(USE_ROCM)
  // Use the small buffer in DevArrayPack to pack the vectors
  if (total_bytes < DevArrayPack::small_buffer_size) {
    pack.buffer_ptr = nullptr;
    for (const auto i : c10::irange(ptrs.size())) {
      pack.offsets[i] = offsets[i];
      memcpy(pack.small_buffer + offsets[i], ptrs[i], sizes[i]);
    }
    return std::make_tuple(pack, c10::optional<at::Tensor>(c10::nullopt));
  }
#endif

  const bool is_capturing = at::cuda::currentStreamCaptureStatusMayInitCtx() !=
      at::cuda::CaptureStatus::None;

  at::Tensor buf_tensor = at::empty(
      {static_cast<int64_t>(total_bytes)},
      // Only use pinned memory when not capturing
      at::TensorOptions().dtype(at::kByte).pinned_memory(!is_capturing));

  // Populate the buf tensor
  for (const auto i : c10::irange(ptrs.size())) {
    pack.offsets[i] = offsets[i];
    memcpy(buf_tensor.data_ptr<uint8_t>() + offsets[i], ptrs[i], sizes[i]);
  }

  if (!is_capturing) {
    buf_tensor = buf_tensor.to(device, /*non_blocking=*/true);
    pack.buffer_ptr = static_cast<char*>(buf_tensor.data_ptr());
    return std::make_tuple(pack, buf_tensor);
  } else {
    // With CUDA Graph, we need the data to have the same lifetime as the
    // capturing graph because the data is an extension of the launch argument.
    // This is achieved by dynamically allocating a buffer, managing the
    // lifetime of the buffer with a CUDA User Object, and transferring the
    // ownership of the CUDA User Object to the capturing graph.
    void* graph_owned_buf = nullptr;
    {
#if !defined(USE_ROCM) || ROCM_VERSION >= 50300
      c10::cuda::CUDAStreamCaptureModeGuard g{cudaStreamCaptureModeRelaxed};
#endif
      // See cudaMallocMaybeCapturing
      C10_CUDA_CHECK(cudaMalloc(&graph_owned_buf, total_bytes));
      // Copy the data from the host staging buffer to the graph-owned device
      // buffer. The copy won't be captured in the graph.
      at::cuda::memcpy_and_sync(
          graph_owned_buf,
          buf_tensor.data_ptr(),
          total_bytes,
          cudaMemcpyHostToDevice,
          0);
    }

    // Manage the ownership of buf with a cuda user object
    cudaUserObject_t user_object;
    C10_CUDA_CHECK(cudaUserObjectCreate(
        &user_object,
        graph_owned_buf,
        [](void* buf) { cudaFree(buf); },
        1, // refcount
        cudaUserObjectNoDestructorSync));

    // Query the currently capturing graph
    cudaStreamCaptureStatus capture_status;
    cudaGraph_t graph;
    C10_CUDA_CHECK(cudaStreamGetCaptureInfo_v2(
        at::cuda::getCurrentCUDAStream(),
        &capture_status,
        nullptr, // id_out
        &graph));

    // Transfer the ownership to the graph
    // NOTE: according to the documentation of cudaStreamGetCaptureInfo, all
    // operations other than destroy and node removal are permitted on the
    // graph while the capture sequence is in progress.
    C10_CUDA_CHECK(cudaGraphRetainUserObject(
        graph, user_object, 1, cudaGraphUserObjectMove));

    pack.buffer_ptr = static_cast<char*>(graph_owned_buf);
    return std::make_tuple(pack, c10::optional<at::Tensor>(c10::nullopt));
  }
}

template <int n>
struct TensorListMetadata {
  const void** addresses[n];
  int64_t* numel_for_tensor;
  size_t* block_to_tensor;
  size_t* block_to_chunk;
  int start_tensor_this_launch;

  // Create a DevArrayPack for TensorListMetadata
  static std::tuple<DevArrayPack, c10::optional<at::Tensor>> make_dev_array_pack(
      std::vector<const void*> (&addresses)[n],
      std::vector<int64_t>& numel_for_tensor,
      std::vector<size_t>& block_to_tensor,
      std::vector<size_t>& block_to_chunk,
      const at::Device& device) {
    return pack_vectors(
        device, addresses, numel_for_tensor, block_to_tensor, block_to_chunk);
  }

  // Convert a DevArrayPack to TensorListMetadata
  C10_HOST_DEVICE __forceinline__ static TensorListMetadata<n>
  from_dev_array_pack(DevArrayPack& pack) {
    TensorListMetadata<n> tl;
#pragma unroll n
    for (auto d = 0; d < n; ++d) {
      pack.get_array(tl.addresses[d], d);
    }
    pack.get_array(tl.numel_for_tensor, n);
    pack.get_array(tl.block_to_tensor, n + 1);
    pack.get_array(tl.block_to_chunk, n + 2);
    tl.start_tensor_this_launch = 0;
    return tl;
  }
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

template <typename T, typename U, typename... ArgTypes>
__device__ void multi_tensor_apply_dev(
    T tensorListMeta,
    U callable,
    ArgTypes... args) {
  // Hand the chunk information to the user-supplied functor to process however
  // it likes.
  callable(kChunkSize, tensorListMeta, args...);
}

template <typename T, typename U, typename... ArgTypes>
C10_LAUNCH_BOUNDS_1(kBlockSize)
__global__ void multi_tensor_apply_kernel(
    T tensorListMeta,
    U callable,
    ArgTypes... args) {
  multi_tensor_apply_dev(tensorListMeta, callable, args...);
}

template <typename T, typename U, typename... ArgTypes>
C10_LAUNCH_BOUNDS_1(kBlockSize)
__global__ void multi_tensor_apply_kernel(
    DevArrayPack pack,
    U callable,
    ArgTypes... args) {
  static_assert(
      sizeof(DevArrayPack) + sizeof(U) + (0 + ... + sizeof(ArgTypes)) <
      max_kernel_arg_size);
  auto tensorListMeta = T::from_dev_array_pack(pack);
  multi_tensor_apply_dev(tensorListMeta, callable, args...);
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
void multi_tensor_apply(
    std::vector<std::vector<at::Tensor>>& tensor_lists,
    T callable,
    ArgTypes... args) {
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
    block_to_tensor.insert(
        block_to_tensor.end(), chunks, addresses[0].size() - 1);
    block_to_chunk.resize(block_to_chunk.size() + chunks);
    std::iota(block_to_chunk.end() - chunks, block_to_chunk.end(), 0);
  }

  auto device = tensor_lists[0][0].device();
  auto [pack, buf_tensor] = TensorListMetadata<depth>::make_dev_array_pack(
      addresses, numel_for_tensor, block_to_tensor, block_to_chunk, device);

  if (block_to_tensor.size() > 0) {
    multi_tensor_apply_kernel<TensorListMetadata<depth>>
        <<<block_to_tensor.size(),
           kBlockSize,
           0,
           at::cuda::getCurrentCUDAStream()>>>(pack, callable, args...);
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
