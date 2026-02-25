#pragma once

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#if defined(USE_ROCM)
typedef __hip_bfloat16 nv_bfloat16;
#endif

#include <iostream>
#include <array>
#include <limits>
#include <map>
#include <unordered_map>
#include <vector>
#include <cstdlib>
#include <cstring>

namespace vllm {
#define CUDACHECK(cmd)                                              \
  do {                                                              \
    cudaError_t e = cmd;                                            \
    if (e != cudaSuccess) {                                         \
      printf("Failed: Cuda error %s:%d '%s'\n", __FILE__, __LINE__, \
             cudaGetErrorString(e));                                \
      exit(EXIT_FAILURE);                                           \
    }                                                               \
  } while (0)

// Maximal number of blocks in allreduce kernel.
constexpr int kMaxBlocks = 36;

// Default number of blocks in allreduce kernel.
#ifndef USE_ROCM
const int defaultBlockLimit = 36;
CUpointer_attribute rangeStartAddrAttr = CU_POINTER_ATTRIBUTE_RANGE_START_ADDR;
#else
const int defaultBlockLimit = 16;
hipPointer_attribute rangeStartAddrAttr =
    HIP_POINTER_ATTRIBUTE_RANGE_START_ADDR;
#endif

// Counter may overflow, but it's fine since unsigned int overflow is
// well-defined behavior.
using FlagType = uint32_t;

// Two sets of peer counters are needed for two syncs: starting and ending an
// operation. The reason is that it's possible for peer GPU block to arrive at
// the second sync point while the current GPU block haven't passed the first
// sync point. Thus, peer GPU may write counter+1 while current GPU is busy
// waiting for counter. We use alternating counter array to avoid this
// possibility.
struct Signal {
  alignas(128) FlagType start[kMaxBlocks][8];
  alignas(128) FlagType end[kMaxBlocks][8];
  alignas(128) FlagType _flag[kMaxBlocks];  // incremental flags for each rank
};

struct __align__(16) RankData {
  const void* ptrs[8];
};

struct __align__(16) RankSignals {
  Signal* signals[8];
};

// like std::array, but aligned
template <typename T, int sz>
struct __align__(alignof(T) * sz) array_t {
  T data[sz];
  using type = T;
  static constexpr int size = sz;
};

// use packed type to maximize memory efficiency
// goal: generate ld.128 and st.128 instructions
template <typename T>
struct packed_t {
  // the (P)acked type for load/store
  using P = array_t<T, 16 / sizeof(T)>;
  // the (A)ccumulator type for reduction
  using A = array_t<float, 16 / sizeof(T)>;
};

#define DINLINE __device__ __forceinline__

// scalar cast functions
DINLINE float upcast_s(half val) { return __half2float(val); }

template <typename T>
DINLINE T downcast_s(float val);
template <>
DINLINE half downcast_s(float val) {
  return __float2half(val);
}

// scalar add functions
// for some reason when compiling with Pytorch, the + operator for half and
// bfloat is disabled so we call the intrinsics directly
DINLINE half& assign_add(half& a, half b) {
  a = __hadd(a, b);
  return a;
}
DINLINE float& assign_add(float& a, float b) { return a += b; }

#if (__CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__))
DINLINE float upcast_s(nv_bfloat16 val) { return __bfloat162float(val); }
template <>
DINLINE nv_bfloat16 downcast_s(float val) {
  return __float2bfloat16(val);
}
DINLINE nv_bfloat16& assign_add(nv_bfloat16& a, nv_bfloat16 b) {
  a = __hadd(a, b);
  return a;
}
#endif

template <typename T, int N>
DINLINE array_t<T, N>& packed_assign_add(array_t<T, N>& a, array_t<T, N> b) {
#pragma unroll
  for (int i = 0; i < N; i++) {
    assign_add(a.data[i], b.data[i]);
  }
  return a;
}

template <typename T, int N>
DINLINE array_t<float, N> upcast(array_t<T, N> val) {
  if constexpr (std::is_same<T, float>::value) {
    return val;
  } else {
    array_t<float, N> out;
#pragma unroll
    for (int i = 0; i < N; i++) {
      out.data[i] = upcast_s(val.data[i]);
    }
    return out;
  }
}

template <typename O>
DINLINE O downcast(array_t<float, O::size> val) {
  if constexpr (std::is_same<typename O::type, float>::value) {
    return val;
  } else {
    O out;
#pragma unroll
    for (int i = 0; i < O::size; i++) {
      out.data[i] = downcast_s<typename O::type>(val.data[i]);
    }
    return out;
  }
}

#if !defined(USE_ROCM)

static DINLINE void st_flag_release(FlagType* flag_addr, FlagType flag) {
  #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
  asm volatile("st.release.sys.global.u32 [%1], %0;" ::"r"(flag),
               "l"(flag_addr));
  #else
  asm volatile("membar.sys; st.volatile.global.u32 [%1], %0;" ::"r"(flag),
               "l"(flag_addr));
  #endif
}

static DINLINE FlagType ld_flag_acquire(FlagType* flag_addr) {
  FlagType flag;
  #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
  asm volatile("ld.acquire.sys.global.u32 %0, [%1];"
               : "=r"(flag)
               : "l"(flag_addr));
  #else
  asm volatile("ld.volatile.global.u32 %0, [%1]; membar.gl;"
               : "=r"(flag)
               : "l"(flag_addr));
  #endif
  return flag;
}

static DINLINE void st_flag_volatile(FlagType* flag_addr, FlagType flag) {
  asm volatile("st.volatile.global.u32 [%1], %0;" ::"r"(flag), "l"(flag_addr));
}

static DINLINE FlagType ld_flag_volatile(FlagType* flag_addr) {
  FlagType flag;
  asm volatile("ld.volatile.global.u32 %0, [%1];"
               : "=r"(flag)
               : "l"(flag_addr));
  return flag;
}

// This function is meant to be used as the first synchronization in the all
// reduce kernel. Thus, it doesn't need to make any visibility guarantees for
// prior memory accesses. Note: volatile writes will not be reordered against
// other volatile writes.
template <int ngpus>
DINLINE void barrier_at_start(const RankSignals& sg, Signal* self_sg,
                              int rank) {
  uint32_t flag = self_sg->_flag[blockIdx.x] + 1;
  if (threadIdx.x < ngpus) {
    auto peer_counter_ptr = &sg.signals[threadIdx.x]->start[blockIdx.x][rank];
    auto self_counter_ptr = &self_sg->start[blockIdx.x][threadIdx.x];
    // Write the expected counter value to peer and wait for correct value
    // from peer.
    st_flag_volatile(peer_counter_ptr, flag);
    while (ld_flag_volatile(self_counter_ptr) != flag);
  }
  __syncthreads();
  // use one thread to update flag
  if (threadIdx.x == 0) self_sg->_flag[blockIdx.x] = flag;
}

// This function is meant to be used as the second or the final
// synchronization barrier in the all reduce kernel. If it's the final
// synchronization barrier, we don't need to make any visibility guarantees
// for prior memory accesses.
template <int ngpus, bool final_sync = false>
DINLINE void barrier_at_end(const RankSignals& sg, Signal* self_sg, int rank) {
  __syncthreads();
  uint32_t flag = self_sg->_flag[blockIdx.x] + 1;
  if (threadIdx.x < ngpus) {
    auto peer_counter_ptr = &sg.signals[threadIdx.x]->end[blockIdx.x][rank];
    auto self_counter_ptr = &self_sg->end[blockIdx.x][threadIdx.x];
    // Write the expected counter value to peer and wait for correct value from
    // peer.
    if constexpr (!final_sync) {
      st_flag_release(peer_counter_ptr, flag);
      while (ld_flag_acquire(self_counter_ptr) != flag);
    } else {
      st_flag_volatile(peer_counter_ptr, flag);
      while (ld_flag_volatile(self_counter_ptr) != flag);
    }
  }
  if constexpr (!final_sync) __syncthreads();

  // use one thread to update flag
  if (threadIdx.x == 0) self_sg->_flag[blockIdx.x] = flag;
}

#else

template <int ngpus>
DINLINE void barrier_at_start(const RankSignals& sg, Signal* self_sg,
                              int rank) {
  uint32_t flag = self_sg->_flag[blockIdx.x] + 1;
  if (threadIdx.x < ngpus) {
    // simultaneously write to the corresponding flag of all ranks.
    // Latency = 1 p2p write
    __scoped_atomic_store_n(&sg.signals[threadIdx.x]->start[blockIdx.x][rank],
                            flag, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM);
    // wait until we got true from all ranks
    while (__scoped_atomic_load_n(&self_sg->start[blockIdx.x][threadIdx.x],
                                  __ATOMIC_RELAXED,
                                  __MEMORY_SCOPE_DEVICE) < flag);
  }
  __syncthreads();
  // use one thread to update flag
  if (threadIdx.x == 0) self_sg->_flag[blockIdx.x] = flag;
}

template <int ngpus, bool final_sync = false>
DINLINE void barrier_at_end(const RankSignals& sg, Signal* self_sg, int rank) {
  __syncthreads();
  uint32_t flag = self_sg->_flag[blockIdx.x] + 1;
  if (threadIdx.x < ngpus) {
    // simultaneously write to the corresponding flag of all ranks.
    // Latency = 1 p2p write
    __scoped_atomic_store_n(&sg.signals[threadIdx.x]->end[blockIdx.x][rank],
                            flag,
                            final_sync ? __ATOMIC_RELAXED : __ATOMIC_RELEASE,
                            __MEMORY_SCOPE_SYSTEM);
    // wait until we got true from all ranks
    while (
        __scoped_atomic_load_n(&self_sg->end[blockIdx.x][threadIdx.x],
                               final_sync ? __ATOMIC_RELAXED : __ATOMIC_ACQUIRE,
                               __MEMORY_SCOPE_DEVICE) < flag);
  }
  if constexpr (!final_sync) __syncthreads();
  // use one thread to update flag
  if (threadIdx.x == 0) self_sg->_flag[blockIdx.x] = flag;
}

#endif

template <typename P, int ngpus, typename A>
DINLINE P packed_reduce(const P* ptrs[], int idx) {
  A tmp = upcast(ptrs[0][idx]);
#pragma unroll
  for (int i = 1; i < ngpus; i++) {
    packed_assign_add(tmp, upcast(ptrs[i][idx]));
  }
  return downcast<P>(tmp);
}

template <typename T, int ngpus>
__global__ void __launch_bounds__(512, 1)
    cross_device_reduce_1stage(RankData* _dp, RankSignals sg, Signal* self_sg,
                               T* __restrict__ result, int rank, int size) {
  using P = typename packed_t<T>::P;
  using A = typename packed_t<T>::A;
  // note: we don't reorder the address so the accumulation order is the same
  // for all ranks, ensuring bitwise identical results
  auto dp = *_dp;
  barrier_at_start<ngpus>(sg, self_sg, rank);
  // do the actual reduction
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size;
       idx += gridDim.x * blockDim.x) {
    ((P*)result)[idx] = packed_reduce<P, ngpus, A>((const P**)&dp.ptrs[0], idx);
  }
  barrier_at_end<ngpus, true>(sg, self_sg, rank);
}

// 1-stage without end barrier — for benchmarking compiler sync elision.
// The next allreduce's start barrier provides the needed synchronization.
template <typename T, int ngpus>
__global__ void __launch_bounds__(512, 1)
    cross_device_reduce_1stage_nosync(RankData* _dp, RankSignals sg,
                                      Signal* self_sg,
                                      T* __restrict__ result, int rank,
                                      int size) {
  using P = typename packed_t<T>::P;
  using A = typename packed_t<T>::A;
  auto dp = *_dp;
  barrier_at_start<ngpus>(sg, self_sg, rank);
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size;
       idx += gridDim.x * blockDim.x) {
    ((P*)result)[idx] =
        packed_reduce<P, ngpus, A>((const P**)&dp.ptrs[0], idx);
  }
}

template <typename P>
DINLINE P* get_tmp_buf(Signal* sg) {
  return (P*)(((Signal*)sg) + 1);
}

template <typename T, int ngpus>
__global__ void __launch_bounds__(512, 1)
    cross_device_reduce_2stage(RankData* _dp, RankSignals sg, Signal* self_sg,
                               T* __restrict__ result, int rank, int size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  using P = typename packed_t<T>::P;
  using A = typename packed_t<T>::A;
  int part = size / ngpus;
  int start = rank * part;
  int end = rank == ngpus - 1 ? size : start + part;
  int largest_part = part + size % ngpus;
  const P* ptrs[ngpus];
  P* tmps[ngpus];
#pragma unroll
  for (int i = 0; i < ngpus; i++) {
    int target = (rank + i) % ngpus;
    ptrs[i] = (const P*)_dp->ptrs[target];
    tmps[i] = get_tmp_buf<P>(sg.signals[target]);
  }
  auto tmp_out = tmps[0];
  barrier_at_start<ngpus>(sg, self_sg, rank);

  // stage 1: reduce scatter
  for (int idx = start + tid; idx < end; idx += stride) {
    tmp_out[idx - start] = packed_reduce<P, ngpus, A>(ptrs, idx);
  }
  barrier_at_end<ngpus>(sg, self_sg, rank);

  // stage 2: allgather. Note: it's important to match the tid between
  // the two stages, because visibility across devices is only guaranteed
  // between threads that have the same tid. If thread i computes the sum of
  // start + i in the first stage, then thread i also gathers start + i from
  // all ranks.

  for (int idx = tid; idx < largest_part; idx += stride) {
#pragma unroll
    for (int i = 0; i < ngpus; i++) {
      int gather_from_rank = ((rank + i) % ngpus);
      if (gather_from_rank == ngpus - 1 || idx < part) {
        int dst_idx = gather_from_rank * part + idx;
        ((P*)result)[dst_idx] = tmps[i][idx];
      }
    }
  }
}

using IPC_KEY = std::array<uint8_t, sizeof(cudaIpcMemHandle_t)>;
static_assert(sizeof(IPC_KEY) == sizeof(cudaIpcMemHandle_t));
static_assert(alignof(IPC_KEY) == alignof(cudaIpcMemHandle_t));

class CustomAllreduce {
 public:
  int rank_;
  int world_size_;
  // Full NVLink or xGMI connection between GPUs.
  bool fully_connected_;

  RankSignals sg_;
  // Stores a map from a pointer to its peer pointers from all ranks.
  std::unordered_map<void*, RankData*> buffers_;
  Signal* self_sg_;

  // Stores rank data from all ranks. This is mainly for cuda graph purposes.
  // For cuda graph to work, all kernel arguments must be fixed during graph
  // capture time. However, the peer pointers are not known during graph
  // capture time. Therefore, during capture, we increment the rank data
  // pointer and use that as the argument to the kernel. The kernel arguments
  // are stored in graph_unreg_buffers_. The actual peer pointers will be
  // filled in at the memory pointed to by the pointers in
  // graph_unreg_buffers_ when the IPC handles are exchanged between ranks.
  //
  // The overall process looks like this:
  // 1. Graph capture.
  // 2. Each rank obtains the IPC handles for each addresses used during cuda
  // graph capture using get_graph_buffer_ipc_meta.
  // 3. (In Python) all gather the IPC handles.
  // 4. Obtain the peer pointers by opening the IPC handles, and store them in
  // the rank data array at corresponding positions.
  RankData *d_rank_data_base_, *d_rank_data_end_;
  std::vector<void*> graph_unreg_buffers_;
  // a map from IPC handles to opened IPC pointers
  std::map<IPC_KEY, char*> ipc_handles_;

  /**
   * Signals are an array of ipc-enabled buffers from all ranks.
   * For each of the buffer, the layout is as follows:
   * | -- sizeof(Signal) -- | ------ a few MB ----- |
   * The first section is for allreduce synchronization, and the second
   * section is for storing the intermediate results required by some
   * allreduce algos.
   *
   * Note: this class does not own any device memory. Any required buffers
   * are passed in from the constructor.
   */
  CustomAllreduce(Signal** signals, void* rank_data, size_t rank_data_sz,
                  int rank, int world_size, bool fully_connected = true)
      : rank_(rank),
        world_size_(world_size),
        fully_connected_(fully_connected),
        self_sg_(signals[rank]),
        d_rank_data_base_(reinterpret_cast<RankData*>(rank_data)),
        d_rank_data_end_(d_rank_data_base_ + rank_data_sz / sizeof(RankData)) {
    for (int i = 0; i < world_size_; i++) {
      sg_.signals[i] = signals[i];
    }
  }

  char* open_ipc_handle(const void* ipc_handle) {
    auto [it, new_handle] =
        ipc_handles_.insert({*((IPC_KEY*)ipc_handle), nullptr});
    if (new_handle) {
      char* ipc_ptr;
      CUDACHECK(cudaIpcOpenMemHandle((void**)&ipc_ptr,
                                     *((const cudaIpcMemHandle_t*)ipc_handle),
                                     cudaIpcMemLazyEnablePeerAccess));
      it->second = ipc_ptr;
    }
    return it->second;
  }

  std::pair<std::string, std::vector<int64_t>> get_graph_buffer_ipc_meta() {
    auto num_buffers = graph_unreg_buffers_.size();
    auto handle_sz = sizeof(cudaIpcMemHandle_t);
    std::string handles(handle_sz * num_buffers, static_cast<char>(0));
    std::vector<int64_t> offsets(num_buffers);
    for (int i = 0; i < num_buffers; i++) {
      auto ptr = graph_unreg_buffers_[i];
      void* base_ptr;
      // note: must share the base address of each allocation, or we get wrong
      // address
      if (cuPointerGetAttribute(&base_ptr, rangeStartAddrAttr,
                                (CUdeviceptr)ptr) != CUDA_SUCCESS)
        throw std::runtime_error("failed to get pointer attr");
      CUDACHECK(cudaIpcGetMemHandle(
          (cudaIpcMemHandle_t*)&handles[i * handle_sz], base_ptr));
      offsets[i] = ((char*)ptr) - ((char*)base_ptr);
    }
    return std::make_pair(handles, offsets);
  }

  void check_rank_data_capacity(size_t num = 1) {
    if (d_rank_data_base_ + num > d_rank_data_end_)
      throw std::runtime_error(
          "Rank data buffer is overflowed by " +
          std::to_string(d_rank_data_base_ + num - d_rank_data_end_));
  }

  /**
   * Register already-shared IPC pointers.
   */
  void register_buffer(void** ptrs) {
    check_rank_data_capacity();
    RankData data;
    for (int i = 0; i < world_size_; i++) {
      data.ptrs[i] = ptrs[i];
    }
    auto d_data = d_rank_data_base_++;
    CUDACHECK(
        cudaMemcpy(d_data, &data, sizeof(RankData), cudaMemcpyHostToDevice));
    buffers_[ptrs[rank_]] = d_data;
  }

  // Note: when registering graph buffers, we intentionally choose to not
  // deduplicate the addresses. That means if the allocator reuses some
  // addresses, they will be registered again. This is to account for the
  // remote possibility of different allocation patterns between ranks. For
  // example, rank 1 may get the same input address for the second allreduce,
  // but rank 2 got a different address. IPC handles have internal reference
  // counting mechanism so overhead should be small.
  void register_graph_buffers(
      const std::vector<std::string>& handles,
      const std::vector<std::vector<int64_t>>& offsets) {
    auto num_buffers = graph_unreg_buffers_.size();
    check_rank_data_capacity(num_buffers);
    std::vector<RankData> rank_data(num_buffers);
    for (int i = 0; i < num_buffers; i++) {
      auto self_ptr = graph_unreg_buffers_[i];
      auto& rd = rank_data[i];
      for (int j = 0; j < world_size_; j++) {
        if (j != rank_) {
          char* handle =
              open_ipc_handle(&handles[j][i * sizeof(cudaIpcMemHandle_t)]);
          handle += offsets[j][i];
          rd.ptrs[j] = handle;
        } else {
          rd.ptrs[j] = self_ptr;
        }
      }
    }
    CUDACHECK(cudaMemcpy(d_rank_data_base_, rank_data.data(),
                         sizeof(RankData) * num_buffers,
                         cudaMemcpyHostToDevice));
    d_rank_data_base_ += num_buffers;
    graph_unreg_buffers_.clear();
  }

  /**
   * Performs allreduce, assuming input has already been registered.
   *
   * Block and grid default configs are results after careful grid search.
   * Using 36 blocks give the best or close to the best runtime on the devices
   * I tried: A100, A10, A30, T4, V100. You'll notice that NCCL kernels also
   * only take a small amount of SMs. Not quite sure the underlying reason,
   * but my guess is that too many SMs will cause contention on NVLink bus.
   */
  template <typename T>
  void allreduce(cudaStream_t stream, T* input, T* output, int size,
                 int threads = 512, int block_limit = defaultBlockLimit) {
    auto d = packed_t<T>::P::size;
    if (size % d != 0)
      throw std::runtime_error(
          "custom allreduce currently requires input length to be multiple "
          "of " +
          std::to_string(d));
    if (block_limit > kMaxBlocks)
      throw std::runtime_error("max supported block limit is " +
                               std::to_string(kMaxBlocks) + ". Got " +
                               std::to_string(block_limit));

    RankData* ptrs;
    cudaStreamCaptureStatus status;
    CUDACHECK(cudaStreamIsCapturing(stream, &status));
    if (status == cudaStreamCaptureStatusActive) {
      ptrs = d_rank_data_base_ + graph_unreg_buffers_.size();
      graph_unreg_buffers_.push_back(input);
    } else {
      auto it = buffers_.find(input);
      if (it == buffers_.end())
        throw std::runtime_error(
            "buffer address " +
            std::to_string(reinterpret_cast<uint64_t>(input)) +
            " is not registered!");
      ptrs = it->second;
    }

    size /= d;
    auto bytes = size * sizeof(typename packed_t<T>::P);
    int blocks = std::min(block_limit, (size + threads - 1) / threads);

    // Check environment variable once
    const char* env_algo = std::getenv("VLLM_CUSTOM_ALLREDUCE_ALGO");
    bool force_1stage = false;
    bool force_2stage = false;
    if (env_algo != nullptr) {
      if (std::strcmp(env_algo, "1stage") == 0 ||
          std::strcmp(env_algo, "oneshot") == 0) {
        force_1stage = true;
      } else if (std::strcmp(env_algo, "2stage") == 0 ||
                 std::strcmp(env_algo, "twoshot") == 0) {
        force_2stage = true;
      } else {
        throw std::runtime_error(
            "Invalid VLLM_CUSTOM_ALLREDUCE_ALGO: " + std::string(env_algo) +
            ". Valid values: 1stage, oneshot, 2stage, twoshot");
      }
    }

#define KL(ngpus, name)                                                       \
  name<T, ngpus><<<blocks, threads, 0, stream>>>(ptrs, sg_, self_sg_, output, \
                                                 rank_, size);
#define REDUCE_CASE(ngpus)                              \
  case ngpus: {                                         \
    if (force_1stage) {                                 \
      KL(ngpus, cross_device_reduce_1stage);            \
    } else if (force_2stage) {                          \
      KL(ngpus, cross_device_reduce_2stage);            \
    } else {                                            \
      if (world_size_ == 2) {                           \
        KL(ngpus, cross_device_reduce_1stage);          \
      } else if (fully_connected_) {                    \
        if ((world_size_ <= 4 && bytes < 512 * 1024) || \
            (world_size_ <= 8 && bytes < 256 * 1024)) { \
          KL(ngpus, cross_device_reduce_1stage);        \
        } else {                                        \
          KL(ngpus, cross_device_reduce_2stage);        \
        }                                               \
      }                                                 \
    }                                                   \
    break;                                              \
  }

    switch (world_size_) {
      REDUCE_CASE(2)
      REDUCE_CASE(4)
      REDUCE_CASE(6)
      REDUCE_CASE(8)
      default:
        throw std::runtime_error(
            "custom allreduce only supports num gpus in (2,4,6,8). Actual "
            "num "
            "gpus = " +
            std::to_string(world_size_));
    }
#undef REDUCE_CASE
#undef KL
  }

  ~CustomAllreduce() {
    for (auto [_, ptr] : ipc_handles_) {
      CUDACHECK(cudaIpcCloseMemHandle(ptr));
    }
  }
};

/**
 * To inspect PTX/SASS, copy paste this header file to compiler explorer and
 add a template instantiation:
 * template void vllm::CustomAllreduce::allreduce<half>(cudaStream_t, half *,
 half *, int, int, int);
*/
}  // namespace vllm