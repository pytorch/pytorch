#include <ATen/native/mps/kernels/ReduceOps.h>
#include <c10/metal/atomic.h>
#include <c10/metal/utils.h>
#include <metal_array>
#include <metal_stdlib>

using namespace metal;
using namespace c10::metal;

struct norm_abs_functor {
  template <typename T, enable_if_t<!is_complex_v<T>, bool> = true>
  inline T operator()(const T x) {
    return static_cast<T>(::precise::abs(x));
  }

  template <typename T, enable_if_t<is_complex_v<T>, bool> = true>
  inline float operator()(const T x) {
    const auto abs_2 = ::precise::abs(float2(x));
    return c10::metal::hypot(abs_2.x, abs_2.y);
  }
};

// `reduction_idx` is the index of a particular batch of input elements that all
// get reduced to one output element. `reduction_element_idx` is the index of
// just one input element within its batch.
static uint32_t get_input_offset(
    uint32_t reduction_element_idx,
    uint32_t reduction_idx,
    constant NormParams<>& params) {
  uint32_t input_offset = 0;

  for (int32_t dim = params.ndim - 1; dim >= 0; dim--) {
    auto input_dim_size = params.input_sizes[dim];
    auto output_dim_size = params.output_sizes[dim];

    // If the the input and output have the same size for this dim, then this
    // dim is not being reduced, so we index by `reduction_idx`
    if (input_dim_size == output_dim_size) {
      auto index_in_dim = reduction_idx % input_dim_size;
      reduction_idx /= input_dim_size;
      input_offset += index_in_dim * params.input_strides[dim];

      // Otherwise, this dim is being reduced, so we index by
      // `reduction_element_idx`
    } else {
      auto index_in_dim = reduction_element_idx % input_dim_size;
      reduction_element_idx /= input_dim_size;
      input_offset += index_in_dim * params.input_strides[dim];
    }
  }
  return input_offset;
}

// In this kernel, each threadgroup is responsible for calculating one element
// of the output.
// TI - dtype of the input tensor.
// TO - dtype of the output tensor.
template <typename TI, typename TO>
kernel void norm(
    constant TI* input [[buffer(0)]],
    device TO* output [[buffer(1)]],
    constant NormParams<>& params [[buffer(2)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tptg [[threads_per_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simdgroup_id [[simdgroup_index_in_threadgroup]],
    uint simdgroup_size [[threads_per_simdgroup]]) {
  using TA = opmath_t<TO>;
  TA output_val = 0;
  const auto p = static_cast<TA>(params.p);

  if (p == INFINITY) {
    output_val = -INFINITY;
  } else if (p == -INFINITY) {
    output_val = INFINITY;
  }

  // First, all the input elements assigned to the threadgroup are divided
  // between all the threads in the threadgroup, and each thread reduces those
  // elements down to one partial `output_val`.
  for (uint32_t reduction_element_idx = tid;
       reduction_element_idx < params.reduction_size;
       reduction_element_idx += tptg) {
    auto input_elem =
        input[get_input_offset(reduction_element_idx, tgid, params)];
    auto input_abs = static_cast<TA>(norm_abs_functor()(input_elem));

    if (p == INFINITY) {
      output_val = max(input_abs, output_val);

    } else if (p == -INFINITY) {
      output_val = min(input_abs, output_val);

    } else if (p == 0) {
      output_val += (input_abs == 0) ? 0 : 1;

    } else {
      output_val += static_cast<TA>(::precise::pow(input_abs, p));
    }
  }

  // Next, all the threads in a threadgroup reduce their `output_val`s together
  // with a series of SIMD group reductions.
  auto threads_remaining = tptg;
  threadgroup TA shared_outputs[MAX_THREADGROUP_SIZE];

  while (threads_remaining > 1) {
    if (p == INFINITY) {
      output_val = simd_max(output_val);
    } else if (p == -INFINITY) {
      output_val = simd_min(output_val);
    } else {
      output_val = simd_sum(output_val);
    }

    threads_remaining =
        (threads_remaining + simdgroup_size - 1) / simdgroup_size;

    if (threads_remaining > 1) {
      // One thread from each SIMD group writes to a shared buffer
      if (simd_lane_id == 0) {
        shared_outputs[simdgroup_id] = output_val;
      }

      threadgroup_barrier(mem_flags::mem_threadgroup);

      // The remaining threads each read one of the partial outputs from the
      // shared buffer
      if (tid < threads_remaining) {
        output_val = shared_outputs[tid];
      } else {
        return;
      }
    }
  }

  // Finally, one thread in the threadgroup writes the final output
  if (tid == 0) {
    uint32_t output_offset = 0;
    uint32_t reduction_idx = tgid;

    for (int32_t dim = params.ndim - 1; dim >= 0; dim--) {
      auto output_dim_size = params.output_sizes[dim];

      if (output_dim_size > 1) {
        auto index_in_dim = reduction_idx % output_dim_size;
        reduction_idx /= output_dim_size;
        output_offset += index_in_dim * params.output_strides[dim];
      }
    }

    if (p != 0 && p != 1 && p != INFINITY && p != -INFINITY) {
      output_val = static_cast<TA>(::precise::pow(output_val, 1 / p));
    }
    output[output_offset] = static_cast<TO>(output_val);
  }
}

#define REGISTER_NORM(TI, TO)                               \
  template [[host_name("norm_" #TI "_" #TO)]]               \
  kernel void norm<TI, TO>(                                 \
      constant TI * input [[buffer(0)]],                    \
      device TO * output [[buffer(1)]],                     \
      constant NormParams<> & params [[buffer(2)]],         \
      uint tid [[thread_position_in_threadgroup]],          \
      uint tptg [[threads_per_threadgroup]],                \
      uint tgid [[threadgroup_position_in_grid]],           \
      uint simd_lane_id [[thread_index_in_simdgroup]],      \
      uint simdgroup_id [[simdgroup_index_in_threadgroup]], \
      uint simdgroup_size [[threads_per_simdgroup]]);

REGISTER_NORM(float, float);
REGISTER_NORM(half, half);
REGISTER_NORM(bfloat, bfloat);
REGISTER_NORM(float2, float);
REGISTER_NORM(half2, half);
