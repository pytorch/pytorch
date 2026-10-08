#pragma once
#include <cstdio>
#include <cstdlib>

// Side-aware schedule for contiguous elementwise kernels on GPUs with two
// memory sides (locality domains).
//
// Tensors from c10::cuda::LocalityAllocator satisfy side(addr) = (addr >> 21) & 1
// and start 4 MiB aligned, so element i of same-dtype operands lies on one
// side. The work unit is a 64 KiB chunk of the PRIMARY operand (the widest
// element type, ties to the output), i.e. one element range in every operand.
// Persistent CTAs (2 x 512 threads per SM) claim chunks from the queue of their
// SM's side and steal from the other queue when theirs is empty. Output
// elements are independent and computed by the same functor, so results are
// bitwise identical to the default vectorized kernel.
//
// try_launch_side_aware() is called from launch_vectorized_kernel (contiguous,
// no dynamic casting, 32-bit indexing) and returns false, launching nothing,
// unless the schedule is enabled, every operand lies in the device's arena,
// the primary operand is 64 KiB aligned and at least 64 MiB, the device has an
// SM side map, and this functor's calibration (SideAwareState.h) found the
// side kernel faster than the default one.

#include <ATen/cuda/CUDAContext.h>
#include <ATen/detail/FunctionTraits.h>
#include <ATen/native/cuda/SideAwareState.h>
#include <c10/cuda/CUDALocalityAllocator.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <tuple>
#include <utility>

namespace at::native::side_aware {

constexpr size_t kChunkBytes = size_t(64) << 10; // work unit, bytes of the primary
constexpr size_t kPageBytes = c10::cuda::LocalityAllocator::kPageBytes;
constexpr uint64_t kChunksPerPage = kPageBytes / kChunkBytes;
constexpr unsigned kThreads = 512, kCtasPerSm = 2;
constexpr int kMaxSms = 256; // size of the SM side map
constexpr unsigned kStepsPerChunk = kChunkBytes / 16 / kThreads; // 16 B steps per thread
constexpr size_t kMinPrimaryBytes = size_t(64) << 20; // below this the claim cost does not pay

// Chunk g = addr >> 16 lies on side (g / kChunksPerPage) & 1 (absolute index).
struct Chunking {
  uint64_t first; // absolute chunk of the primary's first element
  uint64_t per_side[2], before[2]; // side-s chunks inside the tensor / below `first`
};

__host__ __device__ inline uint64_t side_chunks_below(uint64_t abs_chunk, int side) {
  const int64_t rem = int64_t(abs_chunk % (2 * kChunksPerPage)) - int64_t(kChunksPerPage) * side;
  const int64_t partial = rem < 0 ? 0 : rem > int64_t(kChunksPerPage) ? int64_t(kChunksPerPage) : rem;
  return kChunksPerPage * (abs_chunk / (2 * kChunksPerPage)) + uint64_t(partial);
}

__device__ __forceinline__ unsigned smid() {
  unsigned id;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(id));
  return id;
}

// Thread 0 claims one chunk per atomicAdd and prefetches the next claim while
// the CTA processes the current one. A queue seen empty is not touched again.
// The last CTA to finish zeroes the queues, so consecutive launches on one
// stream need no memset.
template <class Body>
__device__ __forceinline__ void for_each_side_chunk(
    const Context& ctx,
    const Chunking& chunking,
    Body body) {
  constexpr unsigned long long kDone = ~0ull;
  __shared__ unsigned long long slot[2]; // current and next claim (chunk index within the tensor)
  unsigned long long* queues = ctx.queues;
  const int home = ctx.sm_side[smid() % kMaxSms] & 1;
  unsigned drained = 0; // bit s: queue s seen empty (thread 0 only); a bit mask, not a local-memory array
  auto claim = [&]() -> unsigned long long {
    for (int steal = 0; steal < 2; ++steal) {
      const int side = home ^ steal;
      if (drained & (1u << side)) {
        continue;
      }
      const unsigned long long rank = atomicAdd(&queues[side], 1ull);
      if (rank < (side ? chunking.per_side[1] : chunking.per_side[0])) {
        const uint64_t global_rank = (side ? chunking.before[1] : chunking.before[0]) + rank;
        return 2 * kChunksPerPage * (global_rank / kChunksPerPage) +
            kChunksPerPage * side + global_rank % kChunksPerPage - chunking.first;
      }
      drained |= 1u << side;
    }
    return kDone;
  };
  if (threadIdx.x == 0) {
    slot[0] = claim();
  }
  __syncthreads();
  for (unsigned it = 0;; ++it) {
    const unsigned long long chunk = slot[it & 1];
    if (chunk == kDone) {
      break;
    }
    if (threadIdx.x == 0) {
      slot[(it + 1) & 1] = claim();
    }
    body(chunk);
    __syncthreads(); // publishes the next claim; this slot is rewritten only after it
  }
  if (threadIdx.x == 0) {
    __threadfence(); // this CTA's claims are ordered before its completion count
    if (atomicAdd(&queues[2], 1ull) == gridDim.x - 1) {
      atomicExch(&queues[0], 0ull);
      atomicExch(&queues[1], 0ull);
      atomicExch(&queues[2], 0ull);
    }
  }
}

template <int Bytes> struct RawT;
template <> struct RawT<1> { using type = unsigned char; };
template <> struct RawT<2> { using type = unsigned short; };
template <> struct RawT<4> { using type = unsigned; };
template <> struct RawT<8> { using type = uint2; };
template <> struct RawT<16> { using type = uint4; };
template <class T, int V> using Raw = typename RawT<int(sizeof(T)) * V>::type;
template <class T, int V> struct Vec { T v[V]; };
template <class T, int V, int G> struct Group { Vec<T, V> step[G]; };

// Loads of a whole group precede its stores, so in-place ops (out == in) stay
// exact and 2 * G loads stay in flight per thread.
template <class T, int V, int G>
__device__ __forceinline__ Group<T, V, G> load_group(const T* base, size_t elem, size_t stride) {
  Group<T, V, G> group;
#pragma unroll
  for (int k = 0; k < G; ++k) {
    Raw<T, V> raw = __ldcs(reinterpret_cast<const Raw<T, V>*>(base + elem + k * stride));
    memcpy(&group.step[k], &raw, sizeof(raw));
  }
  return group;
}

template <int V, int G, class F, class Out, class... In>
__device__ __forceinline__ void compute_store(
    const F& f, Out* out, size_t elem, size_t stride, const Group<In, V, G>&... inputs) {
#pragma unroll
  for (int k = 0; k < G; ++k) {
    Vec<Out, V> result;
#pragma unroll
    for (int j = 0; j < V; ++j) {
      result.v[j] = f(inputs.step[k].v[j]...);
    }
    Raw<Out, V> raw;
    memcpy(&raw, &result, sizeof(raw));
    __stcs(reinterpret_cast<Raw<Out, V>*>(out + elem + k * stride), raw);
  }
}

template <class... T>
constexpr size_t max_size() {
  size_t widest = 0;
  ((widest = sizeof(T) > widest ? sizeof(T) : widest), ...);
  return widest;
}

template <class F, class Out, class... In>
__global__ void __launch_bounds__(kThreads, kCtasPerSm) side_aware_kernel(
    Context ctx, Chunking chunking, size_t n, F f, Out* out, const In*... in) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 1070
  constexpr size_t kWidest = max_size<Out, In...>();
  constexpr int kVec = int(16 / kWidest); // 16 B of the widest operand
  // At most 64 input values per thread per group: 64 registers (2 x 512
  // threads per SM) hold them once a 16-bit type is widened to its opmath
  // float, without spilling. fp32 binary: 8 steps x 4 x 2; bf16 binary:
  // 4 x 8 x 2; bf16 unary: 8 x 8.
  constexpr int kValues = kVec * (sizeof...(In) > 0 ? int(sizeof...(In)) : 1);
  constexpr int kGroup = 8 * kValues <= 64 ? 8 : 4 * kValues <= 64 ? 4 : 2 * kValues <= 64 ? 2 : 1;
  static_assert(kStepsPerChunk % kGroup == 0);
  constexpr size_t kStride = size_t(kThreads) * kVec;
  constexpr size_t kChunkElems = kChunkBytes / kWidest;
  for_each_side_chunk(ctx, chunking, [&](uint64_t chunk) {
    const size_t begin = size_t(chunk) * kChunkElems;
    const size_t end = begin + kChunkElems < n ? begin + kChunkElems : n;
    if (end - begin == kChunkElems) {
#pragma unroll
      for (unsigned step = 0; step < kStepsPerChunk; step += kGroup) {
        const size_t elem = begin + (size_t(step) * kThreads + threadIdx.x) * kVec;
        compute_store<kVec, kGroup>(
            f, out, elem, kStride, load_group<In, kVec, kGroup>(in, elem, kStride)...);
      }
    } else { // the tensor's last, partial chunk
      for (size_t e = begin + threadIdx.x; e < end; e += kThreads) {
        out[e] = f(in[e]...);
      }
    }
  });
#else
  CUDA_KERNEL_ASSERT(false && "side_aware_kernel is compiled for sm_107 only");
#endif
}

template <class F, class Out, class... In, size_t... I>
bool try_launch_impl(
    int64_t N,
    const F& f,
    const std::array<char*, sizeof...(In) + 1>& data,
    DefaultLaunchTimer& timer,
    std::index_sequence<I...>) {
  constexpr size_t kWidest = max_size<Out, In...>();
  constexpr size_t sizes[] = {sizeof(Out), sizeof(In)...};
  const size_t n = size_t(N);
  static const bool debug = std::getenv("PYTORCH_SIDE_AWARE_DEBUG") != nullptr;
  auto reject = [&](const char* why, size_t i) {
    if (debug) {
      std::fprintf(stderr, "[side_aware] reject: %s (operand %zu, n %zu, widest %zu, ptr %p, sizes", why, i, n,
                   kWidest, (void*)data[i < sizeof...(In) + 1 ? i : 0]);
      for (size_t s : sizes) std::fprintf(stderr, " %zu", s);
      std::fprintf(stderr, ")\n");
    }
    return false;
  };
  if (!enabled()) return reject("disabled", 0);
  if (n * kWidest < kMinPrimaryBytes) return reject("too small", 0);
  const auto stream = at::cuda::getCurrentCUDAStream();
  const c10::DeviceIndex device = stream.device_index();
  uint64_t primary = 0;
  for (size_t i = 0; i < sizeof...(In) + 1; ++i) {
    // Every operand must be in the arena and vector aligned (16 B of the widest type).
    if (!c10::cuda::LocalityAllocator::contains(device, data[i], n * sizes[i])) return reject("not in arena", i);
    if (reinterpret_cast<uintptr_t>(data[i]) % (16 / kWidest * sizes[i]) != 0) return reject("misaligned", i);
    if (primary == 0 && sizes[i] == kWidest) {
      primary = reinterpret_cast<uintptr_t>(data[i]);
    }
  }
  Context ctx;
  if (primary % kChunkBytes != 0) return reject("primary not 64 KiB aligned", 0);
  if (!get_context(stream, &ctx)) return reject("no context (capture or invalid SM map)", 0);
  static Calibration calibration; // one per functor/type instantiation
  size_t bytes = 0;
  for (size_t s : sizes) bytes += n * s;
  cudaEvent_t stop = nullptr;
  if (!calibration.use_side(stream.stream(), bytes, &stop, __PRETTY_FUNCTION__)) {
    timer.stop = stop; // not a temporary DefaultLaunchTimer: its destructor would record now
    timer.stream = stream.stream();
    return reject("calibrated (or calibrating) to the default kernel", 0);
  }
  const uint64_t chunk_elems = kChunkBytes / kWidest;
  const uint64_t count = (n + chunk_elems - 1) / chunk_elems;
  Chunking chunking{primary / kChunkBytes, {}, {}};
  for (int side = 0; side < 2; ++side) {
    chunking.before[side] = side_chunks_below(chunking.first, side);
    chunking.per_side[side] =
        side_chunks_below(chunking.first + count, side) - chunking.before[side];
  }
  const unsigned grid = unsigned(std::min<uint64_t>(uint64_t(ctx.sm_count) * kCtasPerSm, count));
  side_aware_kernel<F, Out, In...><<<grid, kThreads, 0, stream>>>(
      ctx, chunking, n, f, reinterpret_cast<Out*>(data[0]),
      reinterpret_cast<const In*>(data[I + 1])...);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  if (stop != nullptr) {
    C10_CUDA_CHECK(cudaEventRecord(stop, stream.stream()));
  }
  count_launch();
  return true;
}

template <class F, class Out, class ArgsTuple> struct Launcher;
template <class F, class Out, class... In>
struct Launcher<F, Out, std::tuple<In...>> {
  static bool run(
      int64_t N,
      const F& f,
      const std::array<char*, sizeof...(In) + 1>& data,
      DefaultLaunchTimer& timer) {
    return try_launch_impl<F, Out, std::decay_t<In>...>(
        N, f, data, timer, std::index_sequence_for<In...>{});
  }
};

// Launches the side-aware kernel and returns true if eligible (see top of
// file). On false, `timer` may hold a calibration event for the default launch.
template <class F, class array_t>
bool try_launch_side_aware(int64_t N, const F& f, const array_t& data, DefaultLaunchTimer& timer) {
  using traits = function_traits<F>;
  return Launcher<F, typename traits::result_type, typename traits::ArgsTuple>::run(N, f, data, timer);
}

} // namespace at::native::side_aware
