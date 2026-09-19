#include <metal_stdlib>
using namespace metal;

// Number of entries of the ascending array `arr[0, n)` that are less than `key`.
template <typename T>
static inline ulong lower_bound(device const T* arr, ulong n, long key) {
  ulong lo = 0;
  ulong hi = n;
  while (lo < hi) {
    ulong mid = lo + ((hi - lo) >> 1);
    if (static_cast<long>(arr[mid]) < key) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  return lo;
}

// `data_in` holds the COO row indices in ascending order, so entry `j` of the
// compressed array is just how many of them are smaller than `j`. The CPU and
// CUDA kernels instead walk the gaps between consecutive rows; a search per
// output entry costs more work in total but needs no serial scan.
template <typename input_t, typename output_t>
kernel void convert_indices_from_coo_to_csr(
    device const input_t* data_in [[buffer(0)]],
    device output_t* data_out [[buffer(1)]],
    constant long& numel [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
  data_out[gid] = static_cast<output_t>(
      lower_bound(data_in, static_cast<ulong>(numel), static_cast<long>(gid)));
}

template <typename input_t, typename output_t>
kernel void convert_indices_from_csr_to_coo(
    device const input_t* crow_indices [[buffer(0)]],
    device output_t* data_out [[buffer(1)]],
    constant long& nrows [[buffer(2)]],
    constant long& nnz [[buffer(3)]],
    constant long& nthreads [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
  if (static_cast<long>(gid) >= nthreads) {
    return;
  }
  const ulong rows = static_cast<ulong>(nrows);
  const ulong b = static_cast<ulong>(gid) / rows;
  const ulong i = static_cast<ulong>(gid) % rows;
  const ulong row = b * (rows + 1) + i;
  const ulong begin = static_cast<ulong>(crow_indices[row]);
  const ulong end = static_cast<ulong>(crow_indices[row + 1]);
  for (ulong k = begin; k < end; ++k) {
    data_out[b * static_cast<ulong>(nnz) + k] = static_cast<output_t>(i);
  }
}

#define INSTANTIATE_COO_TO_CSR(IDTYPE, ODTYPE)                               \
  template [[host_name("convert_indices_from_coo_to_csr_" #IDTYPE "_" #ODTYPE)]] \
  kernel void convert_indices_from_coo_to_csr<IDTYPE, ODTYPE>(               \
      device const IDTYPE* data_in [[buffer(0)]],                            \
      device ODTYPE* data_out [[buffer(1)]],                                 \
      constant long& numel [[buffer(2)]],                                    \
      uint gid [[thread_position_in_grid]]);

#define INSTANTIATE_CSR_TO_COO(IDTYPE, ODTYPE)                               \
  template [[host_name("convert_indices_from_csr_to_coo_" #IDTYPE "_" #ODTYPE)]] \
  kernel void convert_indices_from_csr_to_coo<IDTYPE, ODTYPE>(               \
      device const IDTYPE* crow_indices [[buffer(0)]],                       \
      device ODTYPE* data_out [[buffer(1)]],                                 \
      constant long& nrows [[buffer(2)]],                                    \
      constant long& nnz [[buffer(3)]],                                      \
      constant long& nthreads [[buffer(4)]],                                 \
      uint gid [[thread_position_in_grid]]);

INSTANTIATE_COO_TO_CSR(int, int);
INSTANTIATE_COO_TO_CSR(int, long);
INSTANTIATE_COO_TO_CSR(long, int);
INSTANTIATE_COO_TO_CSR(long, long);

INSTANTIATE_CSR_TO_COO(int, int);
INSTANTIATE_CSR_TO_COO(int, long);
INSTANTIATE_CSR_TO_COO(long, int);
INSTANTIATE_CSR_TO_COO(long, long);
