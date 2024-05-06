#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/ceil_div.h>
#include <ATen/mps/MPSProfiler.h>
#include <ATen/native/BucketizationUtils.h>
#include <ATen/native/Resize.h>
#include <ATen/native/mps/OperationUtils.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/bucketize_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/searchsorted_native.h>
#endif

namespace at::native {
namespace mps {

static MetalShaderLibrary lib(R"BUCKETIZE_METAL(

#include <metal_stdlib>
using namespace metal;

// The bucketization kernels are mostly copied-n-pasted from bucketization.cu.

template<typename input_t>
int64_t lower_bound(constant input_t *data_ss, int64_t start, int64_t end, const input_t val, constant int64_t *data_sort) {
  // sorter gives relative ordering for ND tensors, so we need to save and add the non-updated start as an offset
  // i.e. the second row of a 3x3 tensors starts at element 3 but sorter's second row only contains 0, 1, or 2
  const int64_t orig_start = start;
  while (start < end) {
    const int64_t mid = start + ((end - start) >> 1);
    const input_t mid_val = data_ss[orig_start + data_sort[mid]];
    if (!(mid_val >= val)) {
      start = mid + 1;
    }
    else {
      end = mid;
    }
  }
  return start;
}

template<typename input_t>
int64_t lower_bound(constant input_t *data_ss, int64_t start, int64_t end, const input_t val) {
  while (start < end) {
    const int64_t mid = start + ((end - start) >> 1);
    const input_t mid_val = data_ss[mid];
    if (!(mid_val >= val)) {
      start = mid + 1;
    }
    else {
      end = mid;
    }
  }
  return start;
}

template<typename input_t>
int64_t upper_bound(constant input_t *data_ss, int64_t start, int64_t end, const input_t val, constant int64_t *data_sort) {
  // sorter gives relative ordering for ND tensors, so we need to save and add the non-updated start as an offset
  // i.e. the second row of a 3x3 tensors starts at element 3 but sorter's second row only contains 0, 1, or 2
  const int64_t orig_start = start;
  while (start < end) {
    const int64_t mid = start + ((end - start) >> 1);
    const input_t mid_val = data_ss[orig_start + data_sort[mid]];
    if (!(mid_val > val)) {
      start = mid + 1;
    }
    else {
      end = mid;
    }
  }
  return start;
}

template<typename input_t>
int64_t upper_bound(constant input_t *data_ss, int64_t start, int64_t end, const input_t val) {
  while (start < end) {
    const int64_t mid = start + ((end - start) >> 1);
    const input_t mid_val = data_ss[mid];
    if (!(mid_val > val)) {
      start = mid + 1;
    }
    else {
      end = mid;
    }
  }
  return start;
}

template<typename input_t, typename output_t>
kernel void searchsorted_sorter(
    constant  input_t       * data_in     [[buffer(0)]],
    constant  input_t       * data_bd     [[buffer(1)]],
    device    output_t      * data_out    [[buffer(2)]],
    constant  int64_t       & idim_in     [[buffer(3)]],
    constant  int64_t       & idim_bd     [[buffer(4)]],
    constant  int64_t       & numel_in    [[buffer(5)]],
    constant  int64_t       & right       [[buffer(6)]],
    constant  int64_t       & is_1d_boundaries [[buffer(7)]],
    constant  int64_t       * data_sort   [[buffer(8)]],
    uint2     tgid                        [[threadgroup_position_in_grid]],
    uint2     tid2                        [[thread_position_in_threadgroup]],
    uint2     tptg     [[threads_per_threadgroup]]) {

  for (int64_t tid = tgid.x * tptg.x + tid2.x; tid < numel_in; tid += tptg.x) {
    // If boundaries tensor is 1d, we always search the entire boundary tensor
    int64_t start_bd = is_1d_boundaries ? 0 : tid / idim_in * idim_bd;
    int64_t end_bd = start_bd + idim_bd;

    int64_t pos = !right ?
      lower_bound<input_t>(data_bd, start_bd, end_bd, data_in[tid], data_sort) - start_bd :
      upper_bound<input_t>(data_bd, start_bd, end_bd, data_in[tid], data_sort) - start_bd;

    // type conversion might happen here
    data_out[tid] = pos;
  }
}

template<typename input_t, typename output_t>
kernel void searchsorted(
    constant  input_t       * data_in     [[buffer(0)]],
    constant  input_t       * data_bd     [[buffer(1)]],
    device    output_t      * data_out    [[buffer(2)]],
    constant  int64_t       & idim_in     [[buffer(3)]],
    constant  int64_t       & idim_bd     [[buffer(4)]],
    constant  int64_t       & numel_in    [[buffer(5)]],
    constant  int64_t       & right       [[buffer(6)]],
    constant  int64_t       & is_1d_boundaries [[buffer(7)]],
    uint2     tgid                        [[threadgroup_position_in_grid]],
    uint2     tid2                        [[thread_position_in_threadgroup]],
    uint2     tptg     [[threads_per_threadgroup]]) {

  for (int64_t tid = tgid.x * tptg.x + tid2.x; tid < numel_in; tid += tptg.x) {
    // If boundaries tensor is 1d, we always search the entire boundary tensor
    int64_t start_bd = is_1d_boundaries ? 0 : tid / idim_in * idim_bd;
    int64_t end_bd = start_bd + idim_bd;

    int64_t pos = !right ?
      lower_bound<input_t>(data_bd, start_bd, end_bd, data_in[tid]) - start_bd :
      upper_bound<input_t>(data_bd, start_bd, end_bd, data_in[tid]) - start_bd;

    // type conversion might happen here
    data_out[tid] = pos;
  }
}

#define REGISTER_SEARCHSORTED_OP(INPUT_T, OUTPUT_T)               \
template                                                          \
[[host_name("searchsorted_" #INPUT_T"_"#OUTPUT_T"_sorter")]]      \
kernel void searchsorted_sorter<INPUT_T, OUTPUT_T>(                      \
    constant  INPUT_T       * data_in     [[buffer(0)]],          \
    constant  INPUT_T       * data_bd     [[buffer(1)]],          \
    device    OUTPUT_T      * data_out    [[buffer(2)]],          \
    constant  int64_t       & idim_in     [[buffer(3)]],          \
    constant  int64_t       & idim_bd     [[buffer(4)]],          \
    constant  int64_t       & numel_in    [[buffer(5)]],          \
    constant  int64_t       & right       [[buffer(6)]],          \
    constant  int64_t       & is_1d_boundaries [[buffer(7)]],     \
    constant  int64_t       * data_sort   [[buffer(8)]],          \
    uint2     tgid          [[threadgroup_position_in_grid]],     \
    uint2     tid2          [[thread_position_in_threadgroup]],   \
    uint2     tptg          [[threads_per_threadgroup]]);         \
template                                                          \
[[host_name("searchsorted_" #INPUT_T"_"#OUTPUT_T)]]               \
kernel void searchsorted<INPUT_T, OUTPUT_T>(                      \
    constant  INPUT_T       * data_in     [[buffer(0)]],          \
    constant  INPUT_T       * data_bd     [[buffer(1)]],          \
    device    OUTPUT_T      * data_out    [[buffer(2)]],          \
    constant  int64_t       & idim_in     [[buffer(3)]],          \
    constant  int64_t       & idim_bd     [[buffer(4)]],          \
    constant  int64_t       & numel_in    [[buffer(5)]],          \
    constant  int64_t       & right       [[buffer(6)]],          \
    constant  int64_t       & is_1d_boundaries [[buffer(7)]],     \
    uint2     tgid          [[threadgroup_position_in_grid]],     \
    uint2     tid2          [[thread_position_in_threadgroup]],   \
    uint2     tptg          [[threads_per_threadgroup]]);         \


REGISTER_SEARCHSORTED_OP(float, int);
REGISTER_SEARCHSORTED_OP(float, long);
REGISTER_SEARCHSORTED_OP(half, int);
REGISTER_SEARCHSORTED_OP(half, long);
REGISTER_SEARCHSORTED_OP(char, int);
REGISTER_SEARCHSORTED_OP(char, long);
REGISTER_SEARCHSORTED_OP(uchar, int);
REGISTER_SEARCHSORTED_OP(uchar, long);
REGISTER_SEARCHSORTED_OP(short, int);
REGISTER_SEARCHSORTED_OP(short, long);
REGISTER_SEARCHSORTED_OP(int, int);
REGISTER_SEARCHSORTED_OP(int, long);
REGISTER_SEARCHSORTED_OP(long, int);
REGISTER_SEARCHSORTED_OP(long, long);

)BUCKETIZE_METAL");

static void searchsorted_mps_contiguous(Tensor& result,
                                        const Tensor& input,
                                        const Tensor& boundaries,
                                        const bool right,
                                        const Tensor& sorter) {
  TORCH_INTERNAL_ASSERT(input.is_contiguous());
  TORCH_INTERNAL_ASSERT(boundaries.is_contiguous());
  TORCH_INTERNAL_ASSERT(!sorter.defined() || sorter.is_contiguous());

  int64_t numel_in = input.numel();
  bool is_scalar_input = input.dim() == 0 && numel_in == 1;
  // inner most dim size of input and boundaries
  int64_t idim_in = is_scalar_input ? 1 : input.sizes().back();
  int64_t idim_bd = boundaries.sizes().back();
  int64_t right_i64 = right;
  int64_t is_1d_boundaries = boundaries.dim() == 1;

  MPSStream* mpsStream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(mpsStream->queue(), ^() {
    @autoreleasepool {
      id<MTLComputeCommandEncoder> computeEncoder = mpsStream->commandEncoder();

      const std::string kernel = "searchsorted_" + scalarToMetalTypeString(input) + "_" +
          scalarToMetalTypeString(result) + (sorter.defined() ? "_sorter" : "");
      id<MTLComputePipelineState> bucketizationPSO = lib.getPipelineStateForFunc(kernel);

      // this function call is a no-op if MPS Profiler is not enabled
      getMPSProfiler().beginProfileKernel(bucketizationPSO, kernel, {input, boundaries, sorter});

      [computeEncoder setComputePipelineState:bucketizationPSO];
      mtl_setBuffer(computeEncoder, input, 0);
      mtl_setBuffer(computeEncoder, boundaries, 1);
      mtl_setBuffer(computeEncoder, result, 2);
      [computeEncoder setBytes:&idim_in length:sizeof(int64_t) atIndex:3];
      [computeEncoder setBytes:&idim_bd length:sizeof(int64_t) atIndex:4];
      [computeEncoder setBytes:&numel_in length:sizeof(int64_t) atIndex:5];
      [computeEncoder setBytes:&right_i64 length:sizeof(int64_t) atIndex:6];
      [computeEncoder setBytes:&is_1d_boundaries length:sizeof(int64_t) atIndex:7];
      if (sorter.defined())
        mtl_setBuffer(computeEncoder, sorter, 8);

      // A threadGroup is equivalent to a cuda's block.
      int64_t maxThreadgroups = 1024;
      int64_t maxThreads = bucketizationPSO.maxTotalThreadsPerThreadgroup;
      NSUInteger tgSize = std::min(maxThreads, numel_in);
      MTLSize threadgroupsPerGrid = MTLSizeMake(std::min(maxThreadgroups, ceil_div<int64_t>(numel_in, tgSize)), 1, 1);
      MTLSize threadGroupSize = MTLSizeMake(tgSize, 1, 1);
      [computeEncoder dispatchThreadgroups:threadgroupsPerGrid threadsPerThreadgroup:threadGroupSize];

      getMPSProfiler().endProfileKernel(bucketizationPSO);
    }
  });
}
} // namespace mps

Tensor& searchsorted_out_mps(const Tensor& sorted_sequence,
                             const Tensor& self,
                             bool out_int32,
                             bool right,
                             const c10::optional<c10::string_view> side_opt,
                             const c10::optional<Tensor>& sorter_opt,
                             Tensor& result) {
  // See [Note: hacky wrapper removal for optional tensor]
  auto sorter_maybe_owned = at::borrow_from_optional_tensor(sorter_opt);
  const Tensor& sorter = *sorter_maybe_owned;
  searchsorted_pre_check(sorted_sequence, self, result, out_int32, right, side_opt, sorter);
  resize_output(result, self.sizes());

  // we have two inputs to set right, pre_check checks that they aren't set to opposites
  right |= (side_opt && *side_opt == "right");
  if (self.numel() == 0) {
    return result;
  }

  // for non-contiguous result tensors, we write the output to a contiguous copy so we can later copy back, maintaining
  // the original result tensor
  Tensor out = result.contiguous();

  if (sorted_sequence.is_contiguous() && self.is_contiguous() && sorted_sequence.dtype() == self.dtype() &&
      sorter.is_contiguous()) {
    mps::searchsorted_mps_contiguous(out, self, sorted_sequence, right, sorter);
  } else {
    Tensor trimmed_input;
    Tensor trimmed_boundaries;
    Tensor trimmed_sorter;
    searchsorted_maybe_trim_input_tensors(
        trimmed_input, trimmed_boundaries, trimmed_sorter, self, sorted_sequence, sorter);
    const Tensor& final_input = trimmed_input.defined() ? trimmed_input : self;
    const Tensor& final_boundaries = trimmed_boundaries.defined() ? trimmed_boundaries : sorted_sequence;
    const Tensor& final_sorter = trimmed_sorter.defined() ? trimmed_sorter : sorter;
    mps::searchsorted_mps_contiguous(out, final_input, final_boundaries, right, final_sorter);
  }

  // if result is non-contiguous, we wrote the answer to a copied version, so we copy back to the original result tensor
  if (!result.is_contiguous()) {
    result.copy_(out);
  }
  return result;
}

Tensor& searchsorted_out_mps(const Tensor& sorted_sequence,
                             const Scalar& self,
                             bool out_int32,
                             bool right,
                             const c10::optional<c10::string_view> side_opt,
                             const c10::optional<Tensor>& sorter_opt,
                             Tensor& result) {
  const Tensor& scalar_tensor = mps::wrapped_scalar_tensor_mps(self, sorted_sequence.device());
  return searchsorted_out_mps(sorted_sequence, scalar_tensor, out_int32, right, side_opt, sorter_opt, result);
}

Tensor searchsorted_mps(const Tensor& sorted_sequence,
                        const Tensor& self,
                        bool out_int32,
                        bool right,
                        const c10::optional<c10::string_view> side_opt,
                        const c10::optional<Tensor>& sorter) {
  ScalarType scalar_type = out_int32 ? ScalarType::Int : ScalarType::Long;
  c10::TensorOptions options = TensorOptions().device(self.options().device()).dtype(scalar_type);
  Tensor result = at::empty({0}, options, MemoryFormat::Contiguous);
  at::native::searchsorted_out_mps(sorted_sequence, self, out_int32, right, side_opt, sorter, result);
  return result;
}

Tensor searchsorted_mps(const Tensor& sorted_sequence,
                        const Scalar& self,
                        bool out_int32,
                        bool right,
                        const c10::optional<c10::string_view> side_opt,
                        const c10::optional<Tensor>& sorter) {
  const Tensor& scalar_tensor = mps::wrapped_scalar_tensor_mps(self, sorted_sequence.device());
  return searchsorted_mps(sorted_sequence, scalar_tensor, out_int32, right, side_opt, sorter);
}

Tensor& bucketize_out_mps(const Tensor& self, const Tensor& boundaries, bool out_int32, bool right, Tensor& result) {
  TORCH_CHECK(boundaries.dim() == 1, "boundaries tensor must be 1 dimension, but got dim(", boundaries.dim(), ")");
  at::native::searchsorted_out_mps(boundaries, self, out_int32, right, c10::nullopt, c10::nullopt, result);
  return result;
}

Tensor bucketize_mps(const Tensor& self, const Tensor& boundaries, bool out_int32, bool right) {
  ScalarType scalar_type = out_int32 ? ScalarType::Int : ScalarType::Long;
  c10::TensorOptions options = TensorOptions().device(self.options().device()).dtype(scalar_type);
  Tensor result = at::empty({0}, options, MemoryFormat::Contiguous);
  at::native::bucketize_out_mps(self, boundaries, out_int32, right, result);
  return result;
}

Tensor bucketize_mps(const Scalar& self, const Tensor& boundaries, bool out_int32, bool right) {
  return bucketize_mps(mps::wrapped_scalar_tensor_mps(self, boundaries.device()), boundaries, out_int32, right);
}

} // namespace at::native
