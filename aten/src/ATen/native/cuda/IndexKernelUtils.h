
#include <cstdint>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cuda/MemoryAccess.cuh>

namespace at::native {

struct ScatterAddOp;
struct ScatterCopyOp;
struct ScatterMinOp;
struct ScatterMaxOp;

// Below 256-byte slices, the per-entry overhead of vectorized/TMA scatter
// outweighs the benefit of processing a row at a time.
inline constexpr size_t kFastScatterMinSliceBytes = 256;

template<int alignment>
inline bool fast_gather_kernel_eligible(const TensorIteratorBase& iter, char * const out_ptr, char * const in_ptr, const size_t index_stride_bytes, const size_t element_size, const bool allow_strided_index = false) {
  using at::native::memory::get_alignment;
  const auto index_element_size = iter.element_size(2);
  //TensorIterator strides and sizes are ordered fastest moving to slowest moving,
  //in contrast to regular sizes
  // we need contiguous source and dst slices and aligned pointers and strides and slice size to do vectorized loads
  // also we need idx to be expanded in the last dimension so we can copy entire slices
  // and we need the src tensor to keep 0 stride from restriding
  // (it could have been deleted by dimension collapse, in this case iterator would still be 2d
  // but we cannot use fast path)

  if (iter.ndim() != 2) {
    return false;
  }
  const auto slice_size_bytes = static_cast<size_t>(iter.shape()[0]) * element_size;
  // With allow_strided_index the index may also step by a whole number of
  // elements along the gathered dim (launch_vectorized_gather_strided_index_kernel).
  const auto index_dim_stride = iter.strides(2)[1];
  const bool index_stride_ok = allow_strided_index
      ? index_dim_stride > 0 && index_dim_stride % static_cast<int64_t>(index_element_size) == 0
      : index_dim_stride == static_cast<int64_t>(index_element_size);
  return iter.strides(2)[0]==0 && index_stride_ok &&
         static_cast<size_t>(iter.strides(0)[0])==element_size &&
         static_cast<size_t>(iter.strides(1)[0])==element_size && iter.strides(1)[1] == 0 &&
         get_alignment(out_ptr) == alignment && get_alignment(in_ptr) == alignment &&
         get_alignment(slice_size_bytes) == alignment &&
         get_alignment(static_cast<size_t>(index_stride_bytes)) == alignment &&
         get_alignment(static_cast<size_t>(iter.strides(0)[1])) == alignment;
}

// For scatter on dim=0: self is restrided (stride[dim]=0), src and index keep original strides.
// In TensorIterator ordering (fast to slow):
//   dim 0 (slice): self stride = elem_size, src stride = elem_size, index stride = 0
//   dim 1 (index): self stride = 0 (restrided), src stride = D*elem_size, index stride = idx_elem_size
template<int alignment>
inline bool fast_scatter_kernel_eligible(const TensorIterator& iter, char * const self_ptr, char * const src_ptr, const size_t self_stride_bytes, const size_t element_size) {
  using at::native::memory::get_alignment;
  const auto index_element_size = iter.element_size(2);
  const auto slice_size_bytes = static_cast<size_t>(iter.shape()[0]) * element_size;
  return iter.ndim() == 2 &&
         iter.strides(2)[0] == 0 && iter.strides(2)[1] == index_element_size &&
         static_cast<size_t>(iter.strides(0)[0]) == element_size &&
         static_cast<size_t>(iter.strides(1)[0]) == element_size &&
         iter.strides(0)[1] == 0 &&
         get_alignment(self_ptr) == alignment && get_alignment(src_ptr) == alignment &&
         slice_size_bytes >= kFastScatterMinSliceBytes &&
         get_alignment(slice_size_bytes) == alignment &&
         get_alignment(self_stride_bytes) == alignment &&
         get_alignment(static_cast<size_t>(iter.strides(1)[1])) == alignment;
}

template <int64_t Alignment, typename index_t>
void vectorized_gather_kernel_launch(char * out, char * inp, index_t * idx, int num_ind,
                                     int64_t slice_size_in_bytes, int64_t ind_dim_size, int64_t inp_stride_bytes, int64_t out_stride_bytes,
                                     bool allow_neg_indices=false);

#ifdef USE_ROCM
// True on gfx1250; cached per device.
bool is_gfx1250_device(c10::DeviceIndex device_index);

// gfx1250: vectorized gather for an index strided along the gathered dim
// (entry i at idx[i * idx_stride]), with the row-sized launch of
// vectorized_gather_kernel_launch. The launch requires
// can_use_vectorized_gather_strided_index(device, slice_size_in_bytes, num_ind).
template <int64_t Alignment>
bool can_use_vectorized_gather_strided_index(c10::DeviceIndex device, int64_t slice_size_in_bytes, int64_t num_ind);

template <int64_t Alignment, typename index_t>
void launch_vectorized_gather_strided_index_kernel(char * out, char * inp, index_t * idx, int64_t idx_stride, int num_ind,
                                                   int64_t slice_size_in_bytes, int64_t ind_dim_size, int64_t inp_stride_bytes, int64_t out_stride_bytes);

// gfx1250: element gather for a TensorIterator over (out, in, index) with the
// gathered dim of `in` restrided to 0: out[i] = in[i] displaced by
// index[i] * index_stride_bytes, 0 <= index < index_size (negative indices wrap
// when allow_neg_indices). The launch requires can_use_batched_element_gather:
// a non-empty iterator with 32-bit indexing, element size 1, 2, 4, 8 or 16
// bytes and an int32 or int64 index.
bool can_use_batched_element_gather(const TensorIteratorBase& iter);

void launch_batched_element_gather_kernel(const TensorIteratorBase& iter, int64_t index_size, int64_t index_stride_bytes, bool allow_neg_indices);

// gfx1250: the same gather when the index is constant along the inner dim
// (TensorIterator dim 0) and out and in are contiguous along it, copied a row
// chunk at a time. The launch requires can_use_batched_row_gather: an iterator
// of that form with at least 2 dims, non-empty, with 32-bit indexing, an int32
// or int64 index and rows aligned to at least 2 bytes.
bool can_use_batched_row_gather(const TensorIteratorBase& iter, int64_t index_stride_bytes);

void launch_batched_row_gather_kernel(const TensorIteratorBase& iter, int64_t index_size, int64_t index_stride_bytes, bool allow_neg_indices);
#endif

template <typename reduce_op, int64_t Alignment, typename scalar_t, typename index_t>
void vectorized_scatter_kernel_launch(
    scalar_t* self_data, const scalar_t* src_data, index_t* idx, int num_ind,
    int64_t slice_size_in_bytes, int64_t self_dim_size,
    int64_t self_stride_bytes, int64_t src_stride_bytes);

template <typename reduce_op, typename scalar_t, typename index_t>
void tma_scatter_kernel_launch(
    scalar_t* self_data, const scalar_t* src_data, index_t* idx, int num_ind,
    int D, int64_t self_dim_size,
    int64_t self_stride_bytes, int64_t src_stride_bytes);

}
