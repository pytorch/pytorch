// Sort kernels for MPS. The host (Sort.mm) picks one of three paths based on
// the segment size and type:
//   Path 1 - single-block: one threadgroup per row, used when the segment
//            fits in threadgroup memory.
//   Path 2 - multi-block:  segment split into ELEMS_PER_TG-sized blocks,
//            each block sorted independently, then log2(n_blocks) passes
//            of pairwise merges. Used when sort_size exceeds what one
//            threadgroup can hold, or when single-block would dispatch
//            too few TGs to keep the GPU busy.
//   Path 3 - radix sort:   classic LSD radix over RBITS-bit digits.
//            radix_count -> radix_scan -> radix_scatter, repeated per
//            digit (with optional fused count+scan or fused scan+scatter
//            when n_blocks is small). Selected for radix-friendly types
//            (elem_size <= 4) when the dispatch count beats merge.
#include <c10/metal/utils.h>
#include <metal_stdlib>
using namespace metal;

#include <ATen/native/mps/kernels/SortMerge.h>
#include <ATen/native/mps/kernels/SortRadix.h>
#include <ATen/native/mps/kernels/SortTopK.h>

template <typename T>
kernel void median_gather(
    const device T* sorted_vals [[buffer(0)]],
    const device long* sorted_idxs [[buffer(1)]],
    device T* out_vals [[buffer(2)]],
    device long* out_idxs [[buffer(3)]],
    constant uint& sort_size [[buffer(4)]],
    constant bool& ignore_nan [[buffer(5)]],
    uint row [[thread_position_in_grid]]) {
  const ulong base = ulong(row) * sort_size;
  const device T* vals = sorted_vals + base;
  // binary search the first NaN; lo ends as the non-NaN count
  uint lo = 0;
  uint hi = sort_size;
  while (lo < hi) {
    const uint mid = lo + (hi - lo) / 2;
    const T v = vals[mid];
    if (v != v) {
      hi = mid;
    } else {
      lo = mid + 1;
    }
  }
  uint k;
  if (!ignore_nan) {
    k = lo < sort_size ? lo : (sort_size - 1) / 2;
  } else {
    k = lo > 0 ? (lo - 1) / 2 : 0;
  }
  out_vals[row] = vals[k];
  out_idxs[row] = sorted_idxs[base + k];
}

#define INSTANTIATE_MEDIAN_GATHER(T)                                        \
  template [[host_name("median_gather_" #T)]] kernel void median_gather<T>( \
      const device T*,                                                      \
      const device long*,                                                   \
      device T*,                                                            \
      device long*,                                                         \
      constant uint&,                                                       \
      constant bool&,                                                       \
      uint);

INSTANTIATE_MEDIAN_GATHER(float);
INSTANTIATE_MEDIAN_GATHER(half);
INSTANTIATE_MEDIAN_GATHER(bfloat);

// Global median via MSB-first radix selection over 8-bit digits.
struct MedianSelectState {
  ulong prefix; // key digits fixed so far, right-aligned
  uint k; // remaining rank within the candidate set
  uint nan_count;
  uint done;
};

// Keys come from SortRadix.h: radix_bits<T>::type is the key container
// (ulong for long), to_radix_key/from_radix_key map values to/from key order.
template <typename T>
kernel void median_select_hist(
    const device T* input [[buffer(0)]],
    const device MedianSelectState* state [[buffer(1)]],
    device atomic_uint* hist [[buffer(2)]], // 257 bins; [256] = NaN count
    constant uint& numel [[buffer(3)]],
    constant uint& shift [[buffer(4)]],
    constant bool& first_pass [[buffer(5)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tptg [[threads_per_threadgroup]],
    uint tpg [[threads_per_grid]]) {
  using KeyT = typename radix_bits<T>::type;
  if (state->done) {
    return;
  }
  threadgroup atomic_uint lhist[256];
  for (uint i = lid; i < 256; i += tptg) {
    atomic_store_explicit(&lhist[i], 0u, memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const KeyT prefix = KeyT(state->prefix);
  uint local_nan = 0;
  for (uint i = gid; i < numel; i += tpg) {
    const T v = input[i];
    if (first_pass && v != v) {
      local_nan++;
    }
    const KeyT key = KeyT(to_radix_key(v, /*desc=*/false));
    if (first_pass || (key >> (shift + 8)) == prefix) {
      const uint digit = uint((key >> shift) & 0xFF);
      atomic_fetch_add_explicit(&lhist[digit], 1u, memory_order_relaxed);
    }
  }
  if (first_pass) {
    const uint simd_nan = simd_sum(local_nan);
    if (simd_is_first() && simd_nan > 0) {
      atomic_fetch_add_explicit(&hist[256], simd_nan, memory_order_relaxed);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint i = lid; i < 256; i += tptg) {
    const uint c = atomic_load_explicit(&lhist[i], memory_order_relaxed);
    if (c > 0) {
      atomic_fetch_add_explicit(&hist[i], c, memory_order_relaxed);
    }
  }
}

template <typename T>
kernel void median_select_pick(
    device T* out [[buffer(0)]],
    device MedianSelectState* state [[buffer(1)]],
    device uint* hist [[buffer(2)]],
    constant uint& numel [[buffer(3)]],
    constant bool& first_pass [[buffer(4)]],
    constant bool& last_pass [[buffer(5)]],
    constant bool& ignore_nan [[buffer(6)]]) {
  using KeyT = typename radix_bits<T>::type;
  if (state->done) {
    return;
  }
  if (first_pass) {
    const uint nan_count = hist[256];
    state->nan_count = nan_count;
    if (nan_count > 0 && (!ignore_nan || nan_count == numel)) {
      out[0] = static_cast<T>(NAN);
      state->done = 1;
      return;
    }
    state->k = ignore_nan ? (numel - nan_count - 1) / 2 : (numel - 1) / 2;
  }
  uint k = state->k;
  uint digit = 0;
  for (uint i = 0; i < 256; i++) {
    const uint c = hist[i];
    if (c > k) {
      digit = i;
      break;
    }
    k -= c;
    hist[i] = 0;
  }
  for (uint i = digit; i < 257; i++) {
    hist[i] = 0;
  }
  state->prefix = (state->prefix << 8) | ulong(digit);
  state->k = k;
  if (last_pass) {
    out[0] = from_radix_key<T>(KeyT(state->prefix), /*desc=*/false);
    state->done = 1;
  }
}

#define INSTANTIATE_MEDIAN_SELECT(T)                           \
  template [[host_name("median_select_hist_" #T)]] kernel void \
  median_select_hist<T>(                                       \
      const device T*,                                         \
      const device MedianSelectState*,                         \
      device atomic_uint*,                                     \
      constant uint&,                                          \
      constant uint&,                                          \
      constant bool&,                                          \
      uint,                                                    \
      uint,                                                    \
      uint,                                                    \
      uint);                                                   \
  template [[host_name("median_select_pick_" #T)]] kernel void \
  median_select_pick<T>(                                       \
      device T*,                                               \
      device MedianSelectState*,                               \
      device uint*,                                            \
      constant uint&,                                          \
      constant bool&,                                          \
      constant bool&,                                          \
      constant bool&);

INSTANTIATE_MEDIAN_SELECT(float);
INSTANTIATE_MEDIAN_SELECT(half);
INSTANTIATE_MEDIAN_SELECT(bfloat);
INSTANTIATE_MEDIAN_SELECT(int);
INSTANTIATE_MEDIAN_SELECT(long);
INSTANTIATE_MEDIAN_SELECT(short);
INSTANTIATE_MEDIAN_SELECT(char);
INSTANTIATE_MEDIAN_SELECT(uchar);
