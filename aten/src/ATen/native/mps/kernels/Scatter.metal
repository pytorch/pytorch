#include <c10/metal/atomic.h>
#include <c10/metal/error.h>
#include <c10/metal/indexing.h>
#include <c10/metal/utils.h>
#include <metal_stdlib>

using namespace metal;
using namespace c10::metal;

// scatter_set kernel: implements PyTorch's scatter with reduce='set'.
//
// For each coordinate `c` in `index`'s shape (one thread per element of
// `index`), compute:
//   idx = index[c]
//   bounds-check idx in [0, dim_size); on failure, report async error.
//   output[c with c[dim] replaced by idx] = src[c]
//
// The caller is responsible for pre-copying `self` into `output` when
// they refer to different storages (out= variants); this kernel only
// writes the scattered positions.
template <typename T, typename index_t>
kernel void scatter_set(
    device T* output [[buffer(0)]],
    constant T* src [[buffer(1)]],
    constant index_t* index [[buffer(2)]],
    constant long* index_sizes [[buffer(3)]],
    constant long* output_strides [[buffer(4)]],
    constant long* src_strides [[buffer(5)]],
    constant long* index_strides [[buffer(6)]],
    constant uint3& ndim_dim [[buffer(7)]],
    constant long& dim_size [[buffer(8)]],
    device ErrorMessages* error_buf [[buffer(9)]],
    uint thread_index [[thread_position_in_grid]]) {
  const uint ndim = ndim_dim.x;
  const uint dim = ndim_dim.y;

  ::metal::array<long, max_ndim> pos;
  pos_from_thread_index<long>(long(thread_index), &pos[0], index_sizes, ndim);

  const long index_offs = offset_from_coord<long>(&pos[0], index_strides, ndim);
  long idx = long(index[index_offs]);
  if (idx < 0 || idx >= dim_size) {
    TORCH_REPORT_ERROR(
        error_buf,
        "scatter: index ",
        idx,
        " is out of bounds for dimension ",
        long(dim),
        " with size ",
        dim_size);
    return;
  }

  const long src_offs = offset_from_coord<long>(&pos[0], src_strides, ndim);
  pos[dim] = idx;
  const long out_offs = offset_from_coord<long>(&pos[0], output_strides, ndim);
  output[out_offs] = src[src_offs];
}

// Fast path for the common case where output, src, and index are all
// contiguous, src and index share the same shape, and that shape matches
// the output's shape outside `dim`. Each thread:
//   inner = tid % inner_size       (coords below dim)
//   outer = tid / (inner_size * index_dim_size)   (coords above dim)
//   output[outer * output_dim_size * inner_size + idx * inner_size + inner]
//       = src[tid]
// inner_size = prod(output.sizes()[dim+1:]).
template <typename T, typename index_t>
kernel void scatter_set_dense(
    device T* output [[buffer(0)]],
    constant T* src [[buffer(1)]],
    constant index_t* index [[buffer(2)]],
    constant long& inner_size [[buffer(3)]],
    constant long& index_dim_size [[buffer(4)]],
    constant long& output_dim_size [[buffer(5)]],
    device ErrorMessages* error_buf [[buffer(6)]],
    uint thread_index [[thread_position_in_grid]]) {
  long idx = long(index[thread_index]);
  if (idx < 0 || idx >= output_dim_size) {
    TORCH_REPORT_ERROR(
        error_buf,
        "scatter: index ",
        idx,
        " is out of bounds for dimension with size ",
        output_dim_size);
    return;
  }
  const long inner = long(thread_index) % inner_size;
  const long outer = long(thread_index) / (inner_size * index_dim_size);
  const long out_offset =
      outer * (inner_size * output_dim_size) + idx * inner_size + inner;
  output[out_offset] = src[thread_index];
}

#define REGISTER_SCATTER_SET_OP(DTYPE, IDXTYPE)                                \
  template [[host_name("scatter_set_" #DTYPE "_" #IDXTYPE)]] kernel void       \
  scatter_set<DTYPE, IDXTYPE>(                                                 \
      device DTYPE * output [[buffer(0)]],                                     \
      constant DTYPE * src [[buffer(1)]],                                      \
      constant IDXTYPE * index [[buffer(2)]],                                  \
      constant long* index_sizes [[buffer(3)]],                                \
      constant long* output_strides [[buffer(4)]],                             \
      constant long* src_strides [[buffer(5)]],                                \
      constant long* index_strides [[buffer(6)]],                              \
      constant uint3& ndim_dim [[buffer(7)]],                                  \
      constant long& dim_size [[buffer(8)]],                                   \
      device ErrorMessages* error_buf [[buffer(9)]],                           \
      uint thread_index [[thread_position_in_grid]]);                          \
  template [[host_name("scatter_set_dense_" #DTYPE "_" #IDXTYPE)]] kernel void \
  scatter_set_dense<DTYPE, IDXTYPE>(                                           \
      device DTYPE * output [[buffer(0)]],                                     \
      constant DTYPE * src [[buffer(1)]],                                      \
      constant IDXTYPE * index [[buffer(2)]],                                  \
      constant long& inner_size [[buffer(3)]],                                 \
      constant long& index_dim_size [[buffer(4)]],                             \
      constant long& output_dim_size [[buffer(5)]],                            \
      device ErrorMessages* error_buf [[buffer(6)]],                           \
      uint thread_index [[thread_position_in_grid]])

#define REGISTER_SCATTER_SET_DTYPE(DTYPE) \
  REGISTER_SCATTER_SET_OP(DTYPE, long);   \
  REGISTER_SCATTER_SET_OP(DTYPE, int)

REGISTER_SCATTER_SET_DTYPE(float);
REGISTER_SCATTER_SET_DTYPE(half);
REGISTER_SCATTER_SET_DTYPE(bfloat);
REGISTER_SCATTER_SET_DTYPE(long);
REGISTER_SCATTER_SET_DTYPE(int);
REGISTER_SCATTER_SET_DTYPE(short);
REGISTER_SCATTER_SET_DTYPE(char);
REGISTER_SCATTER_SET_DTYPE(uchar);
REGISTER_SCATTER_SET_DTYPE(bool);
REGISTER_SCATTER_SET_DTYPE(float2);
REGISTER_SCATTER_SET_DTYPE(half2);
