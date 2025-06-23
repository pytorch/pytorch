#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/SegmentReduce.h>
#include <ATen/native/cuda/SegmentReduceKernels.h>

namespace at::native {

REGISTER_DISPATCH(_segment_reduce_lengths_stub, &_segment_reduce_lengths_cuda_kernel)
REGISTER_DISPATCH(_segment_reduce_offsets_stub, &_segment_reduce_offsets_cuda_kernel)
REGISTER_DISPATCH(
    _segment_reduce_lengths_backward_stub,
    &_segment_reduce_lengths_backward_cuda_kernel);
REGISTER_DISPATCH(
  _segment_reduce_offsets_backward_stub,
  &_segment_reduce_offsets_backward_cuda_kernel);

} // namespace at::native
