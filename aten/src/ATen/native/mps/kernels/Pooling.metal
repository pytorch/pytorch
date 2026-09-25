#include <ATen/native/mps/kernels/Pooling.h>
#include <c10/metal/atomic.h>
#include <c10/metal/error.h>
#include <c10/metal/utils.h>
#include <metal_array>
#include <metal_stdlib>

using namespace metal;
using namespace c10::metal;

template <typename T>
struct IterBounds {
  T start;
  T end;
};

template <typename IdxT>
struct PoolDimGeom {
  constant IdxT* input_sizes;
  constant IdxT* output_sizes;
  constant IdxT* kernel_size;
  constant IdxT* stride;
  constant IdxT* padding;
  constant IdxT* dilation;
  bool adaptive;
};

template <int32_t dim, typename IdxT>
IterBounds<IdxT> get_input_iter_bounds(
    constant IdxT* input_sizes,
    thread IdxT (&pooling_dim_indices)[3],
    constant IdxT* kernel_size,
    constant IdxT* stride,
    constant IdxT* padding,
    constant IdxT* dilation) {
  auto d = dilation[dim];
  auto start = stride[dim] * pooling_dim_indices[dim] - padding[dim];
  auto end = min(start + kernel_size[dim] * d, input_sizes[dim]);
  auto start_correction = d * ((-start - 1 + d) / d);
  start += start < 0 ? start_correction : 0;
  return IterBounds<IdxT>{start, end};
}

template <int32_t dim, typename IdxT>
IterBounds<IdxT> get_adaptive_input_iter_bounds(
    constant IdxT* input_sizes,
    constant IdxT* output_sizes,
    thread IdxT (&pooling_dim_indices)[3]) {
  auto in_size = static_cast<int64_t>(input_sizes[dim]);
  auto out_size = static_cast<int64_t>(output_sizes[dim]);
  auto out_idx = static_cast<int64_t>(pooling_dim_indices[dim]);
  auto start = static_cast<IdxT>((out_idx * in_size) / out_size);
  auto end =
      static_cast<IdxT>(((out_idx + 1) * in_size + out_size - 1) / out_size);
  return IterBounds<IdxT>{start, end};
}

template <int32_t dim, typename IdxT>
IterBounds<IdxT> get_pool_input_iter_bounds(
    thread PoolDimGeom<IdxT>& geom,
    thread IdxT (&pooling_dim_indices)[3]) {
  if (geom.adaptive) {
    return get_adaptive_input_iter_bounds<dim, IdxT>(
        geom.input_sizes, geom.output_sizes, pooling_dim_indices);
  }
  return get_input_iter_bounds<dim, IdxT>(
      geom.input_sizes,
      pooling_dim_indices,
      geom.kernel_size,
      geom.stride,
      geom.padding,
      geom.dilation);
}

// Iterates through all the input elements that this kernel needs to
// apply max to. Specialized for 3 pooling dimensions.
// TODO: Support any number of pooling dims
template <typename T, typename IdxT>
void max_pool_3d_input_iter(
    constant T* input,
    device T* output,
    device int64_t* indices,
    constant IdxT* input_strides,
    thread IdxT (&pooling_dim_indices)[3],
    PoolDimGeom<IdxT> geom,
    bool return_indices) {
  auto bounds0 = get_pool_input_iter_bounds<0, IdxT>(geom, pooling_dim_indices);
  auto bounds1 = get_pool_input_iter_bounds<1, IdxT>(geom, pooling_dim_indices);
  auto bounds2 = get_pool_input_iter_bounds<2, IdxT>(geom, pooling_dim_indices);

  auto d0 = geom.dilation[0];
  auto d1 = geom.dilation[1];
  auto d2 = geom.dilation[2];

  T max_value = input
      [input_strides[0] * bounds0.start + input_strides[1] * bounds1.start +
       input_strides[2] * bounds2.start];
  auto size12 = geom.input_sizes[1] * geom.input_sizes[2];
  auto max_index = bounds0.start * size12 +
      bounds1.start * geom.input_sizes[2] + bounds2.start;

  for (auto i0 = bounds0.start; i0 < bounds0.end; i0 += d0) {
    auto offset0 = input_strides[0] * i0;

    for (auto i1 = bounds1.start; i1 < bounds1.end; i1 += d1) {
      auto offset1 = input_strides[1] * i1;

      for (auto i2 = bounds2.start; i2 < bounds2.end; i2 += d2) {
        auto offset2 = input_strides[2] * i2;
        auto input_value = input[offset0 + offset1 + offset2];
        bool is_greater = input_value > max_value;

        max_value = is_greater ? input_value : max_value;

        if (return_indices) {
          auto input_index = i0 * size12 + i1 * geom.input_sizes[2] + i2;
          max_index = is_greater ? input_index : max_index;
        }
      }
    }
  }
  *output = max_value;
  if (return_indices) {
    *indices = max_index;
  }
}

template <typename T, typename IdxT, bool return_indices>
void max_pool_2d_input_iter(
    constant T* input,
    device T* output,
    device int64_t* indices,
    constant IdxT* input_strides,
    thread IdxT (&pooling_dim_indices)[3],
    PoolDimGeom<IdxT> geom) {
  auto bounds0 = get_pool_input_iter_bounds<0, IdxT>(geom, pooling_dim_indices);
  auto bounds1 = get_pool_input_iter_bounds<1, IdxT>(geom, pooling_dim_indices);

  auto d0 = geom.dilation[0];
  auto d1 = geom.dilation[1];

  T max_value = input
      [input_strides[0] * bounds0.start + input_strides[1] * bounds1.start];
  auto max_index = bounds0.start * geom.input_sizes[1] + bounds1.start;

  for (auto i0 = bounds0.start; i0 < bounds0.end; i0 += d0) {
    auto offset0 = input_strides[0] * i0;

    for (auto i1 = bounds1.start; i1 < bounds1.end; i1 += d1) {
      auto offset1 = input_strides[1] * i1;

      auto input_value = input[offset0 + offset1];
      bool is_greater = input_value > max_value;

      max_value = is_greater ? input_value : max_value;

      if (return_indices) {
        auto input_index = i0 * geom.input_sizes[1] + i1;
        max_index = is_greater ? input_index : max_index;
      }
    }
  }
  *output = max_value;
  if (return_indices) {
    *indices = max_index;
  }
}

template <typename IdxT>
struct PoolOffsets {
  IdxT output;
  IdxT indices;
  IdxT input_leading;

  PoolOffsets() : output(0), indices(0), input_leading(0) {}
};

// Finds the offset of the output element that a forward pass thread will
// calculate, `output[N, C, d, h, w]`. Also, find the offset of the input for
// the leading dim indices, `input[N, C]`. Optionally, keep track of the output
// pooling dimension indices, `[d, h , w]`.
// NOTE: This is templated per number of dimensions so that the compiler can
// unroll the loop, giving better performance.
template <int32_t dims, typename IdxT>
PoolOffsets<IdxT> find_pool_offsets_dim_specific(
    constant IdxT* output_sizes,
    constant IdxT* output_strides,
    constant IdxT* indices_strides,
    constant IdxT* input_strides,
    IdxT pooling_dim_indices[3],
    int32_t leading_dims,
    bool return_indices,
    IdxT gid) {
  auto output_idx = gid;
  PoolOffsets<IdxT> offsets;

  for (auto dim = dims - 1; dim >= 0; dim--) {
    auto dim_idx = output_idx % (output_sizes[dim]);
    offsets.output += output_strides[dim] * dim_idx;
    if (return_indices) {
      offsets.indices += indices_strides[dim] * dim_idx;
    }

    if (dim < leading_dims) {
      offsets.input_leading += input_strides[dim] * dim_idx;
    } else {
      // Keep track of pooling dimension indices of the output element, so we
      // can use them in the input iteration later on.
      if (pooling_dim_indices != nullptr) {
        pooling_dim_indices[dim - leading_dims] = dim_idx;
      }
    }
    output_idx = output_idx / output_sizes[dim];
  }

  return offsets;
}

template <typename IdxT>
PoolOffsets<IdxT> find_pool_offsets(
    constant IdxT* output_sizes,
    constant IdxT* output_strides,
    constant IdxT* indices_strides,
    constant IdxT* input_strides,
    IdxT pooling_dim_indices[3],
    int32_t dims,
    int32_t leading_dims,
    bool return_indices,
    IdxT gid) {
  switch (dims) {
    case 5:
      return find_pool_offsets_dim_specific<5, IdxT>(
          output_sizes,
          output_strides,
          indices_strides,
          input_strides,
          pooling_dim_indices,
          leading_dims,
          return_indices,
          gid);
    case 4:
      return find_pool_offsets_dim_specific<4, IdxT>(
          output_sizes,
          output_strides,
          indices_strides,
          input_strides,
          pooling_dim_indices,
          leading_dims,
          return_indices,
          gid);
    case 3:
      return find_pool_offsets_dim_specific<3, IdxT>(
          output_sizes,
          output_strides,
          indices_strides,
          input_strides,
          pooling_dim_indices,
          leading_dims,
          return_indices,
          gid);
  }
  return PoolOffsets<IdxT>();
}

// Kernel computes one element of the output per kernel call.
template <typename T, typename IdxT>
kernel void max_pool(
    constant T* input [[buffer(0)]],
    device T* output [[buffer(1)]],
    device int64_t* indices [[buffer(2)]],
    constant PoolingParams<5, IdxT>& params [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  const auto gid = static_cast<IdxT>(tid) + params.tid_offset;
  bool return_indices = params.return_indices;
  auto pooling_dims = params.pooling_dims;
  auto dims = params.dims;
  auto input_sizes = params.input_sizes.data();
  auto input_strides = params.input_strides.data();
  auto output_sizes = params.output_sizes.data();
  auto output_strides = params.output_strides.data();
  auto indices_strides = params.indices_strides.data();
  auto kernel_size = params.kernel_size.data();
  auto stride = params.stride.data();
  auto padding = params.padding.data();
  auto dilation = params.dilation.data();

  auto leading_dims = dims - pooling_dims;
  PoolDimGeom<IdxT> geom{
      input_sizes + leading_dims,
      output_sizes + leading_dims,
      kernel_size,
      stride,
      padding,
      dilation,
      params.adaptive};

  // This buffer keeps track of the pooling dimension indices of this thread's
  // element of the output. We need to fill it with the proper values below.
  IdxT pooling_dim_indices[3];

  PoolOffsets<IdxT> offsets = find_pool_offsets<IdxT>(
      output_sizes,
      output_strides,
      return_indices ? indices_strides : nullptr,
      input_strides,
      pooling_dim_indices,
      dims,
      leading_dims,
      return_indices,
      gid);

  output += offsets.output;
  indices += offsets.indices;
  input += offsets.input_leading;

  switch (pooling_dims) {
    case 2:
      if (return_indices) {
        return max_pool_2d_input_iter<T, IdxT, /*return_indices=*/true>(
            input,
            output,
            indices,
            input_strides + leading_dims,
            pooling_dim_indices,
            geom);
      } else {
        return max_pool_2d_input_iter<T, IdxT, /*return_indices=*/false>(
            input,
            output,
            indices,
            input_strides + leading_dims,
            pooling_dim_indices,
            geom);
      }
    case 3:
      return max_pool_3d_input_iter<T, IdxT>(
          input,
          output,
          indices,
          input_strides + leading_dims,
          pooling_dim_indices,
          geom,
          return_indices);
  }
}

// Finds the element in the grad input which corresponds to the index into the
// pool, and then adds the grad output element to it.
template <typename T, typename IdxT>
void max_pool_backward_impl(
    device AtomicType_t<T>* grad_input,
    T grad_output_element,
    IdxT input_index,
    constant IdxT* grad_input_sizes,
    constant IdxT* grad_input_strides,
    IdxT grad_input_leading_offset,
    int32_t pooling_dims) {
  IdxT size_prod = 1;
  IdxT pool_offset = 0;

  for (auto dim = pooling_dims - 1; dim >= 0; dim--) {
    auto next_size_prod = grad_input_sizes[dim] * size_prod;
    pool_offset +=
        grad_input_strides[dim] * ((input_index % next_size_prod) / size_prod);
    size_prod *= grad_input_sizes[dim];
  }

  AtomicType<T>::atomic_add(
      grad_input, grad_input_leading_offset + pool_offset, grad_output_element);
}

// Kernel computes one element of the grad input per kernel call.
template <typename T, typename IdxT>
kernel void max_pool_backward(
    device AtomicType_t<T>* grad_input [[buffer(0)]],
    constant T* grad_output [[buffer(1)]],
    constant int64_t* indices [[buffer(2)]],
    constant PoolingBackwardParams<5, IdxT>& params [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  const auto gid = static_cast<IdxT>(tid) + params.tid_offset;
  auto pooling_dims = params.pooling_dims;
  auto dims = params.dims;
  auto grad_input_sizes = params.grad_input_sizes.data();
  auto grad_input_strides = params.grad_input_strides.data();
  auto grad_output_sizes = params.grad_output_sizes.data();
  auto grad_output_strides = params.grad_output_strides.data();
  auto indices_strides = params.indices_strides.data();

  auto leading_dims = dims - pooling_dims;

  PoolOffsets<IdxT> offsets = find_pool_offsets<IdxT>(
      grad_output_sizes,
      grad_output_strides,
      indices_strides,
      grad_input_strides,
      nullptr,
      dims,
      leading_dims,
      /*return_indices=*/true,
      gid);

  max_pool_backward_impl<T, IdxT>(
      grad_input,
      grad_output[offsets.output],
      indices[offsets.indices],
      grad_input_sizes + leading_dims,
      grad_input_strides + leading_dims,
      offsets.input_leading,
      pooling_dims);
}

template <typename T, typename IdxT>
void max_unpool_impl(
    device T* output,
    T input_element,
    int64_t input_index,
    constant IdxT* output_sizes,
    constant IdxT* output_strides,
    int32_t pooling_dims,
    device c10::metal::ErrorMessages* error_buffer) {
  // indices come from a user-supplied int64 tensor, so range-check at full
  // width before narrowing; otherwise the 32-bit specialization truncates an
  // out-of-range index into an in-range one and writes there.
  int64_t size_prod = 1;
  for (auto dim = pooling_dims - 1; dim >= 0; dim--) {
    size_prod *= output_sizes[dim];
  }

  if (input_index < 0 || input_index >= size_prod) {
    TORCH_REPORT_ERROR(
        error_buffer,
        "Found an invalid max index: ",
        input_index,
        " (size_prod is ",
        size_prod,
        ")");
    return;
  }

  // The check above ran at full width; past it the index is below size_prod,
  // which offsetsFitIn<int32_t> already caps below 2^31 on the 32-bit path, so
  // the decomposition runs at the kernel's own index width.
  const auto index = static_cast<IdxT>(input_index);
  IdxT pool_offset = 0;
  IdxT running = 1;

  for (auto dim = pooling_dims - 1; dim >= 0; dim--) {
    const auto next = output_sizes[dim] * running;
    pool_offset += output_strides[dim] * ((index % next) / running);
    running *= output_sizes[dim];
  }

  output[pool_offset] = input_element;
}

// Kernel computes one element of the grad input per kernel call.
template <typename T, typename IdxT>
kernel void max_unpool(
    device T* output [[buffer(0)]],
    constant T* input [[buffer(1)]],
    constant int64_t* indices [[buffer(2)]],
    constant MaxUnpoolingParams<5, IdxT>& params [[buffer(3)]],
    device c10::metal::ErrorMessages* error_buffer [[buffer(4)]],
    uint tid [[thread_position_in_grid]]) {
  const auto gid = static_cast<IdxT>(tid) + params.tid_offset;
  auto pooling_dims = params.pooling_dims;
  auto dims = params.dims;
  auto input_sizes = params.input_sizes.data();
  auto input_strides = params.input_strides.data();
  auto output_sizes = params.output_sizes.data();
  auto output_strides = params.output_strides.data();
  auto indices_strides = params.indices_strides.data();

  auto leading_dims = dims - pooling_dims;

  // NOTE: Since we're doing unpooling, the variable names "input" and "output"
  // are reversed compared to the pooling operations. So in `find_pool_offsets`,
  // we need to map "input" -> "output" and "output" -> "input".
  PoolOffsets<IdxT> offsets = find_pool_offsets<IdxT>(
      /*output_sizes=*/input_sizes,
      /*output_strides=*/input_strides,
      indices_strides,
      /*input_strides=*/output_strides,
      /*pooling_dim_indices=*/nullptr,
      dims,
      leading_dims,
      /*return_indices=*/true,
      gid);

  max_unpool_impl<T, IdxT>(
      output + offsets.input_leading,
      input[offsets.output],
      indices[offsets.indices],
      output_sizes + leading_dims,
      output_strides + leading_dims,
      pooling_dims,
      error_buffer);
}

template <typename T>
struct AvgPoolIterBounds {
  T start;
  T end;
  T count;
};

template <int32_t dim, typename IdxT>
AvgPoolIterBounds<IdxT> get_avg_pool_input_iter_bounds(
    constant IdxT* input_sizes,
    thread IdxT (&pooling_dim_indices)[3],
    constant IdxT* kernel_size,
    constant IdxT* stride,
    constant IdxT* padding,
    bool count_include_pad) {
  auto start = stride[dim] * pooling_dim_indices[dim] - padding[dim];
  auto end = start + kernel_size[dim];
  auto end_corrected = min(start + kernel_size[dim], input_sizes[dim]);
  auto start_corrected = (start < 0) ? 0 : start;
  auto count = count_include_pad
      ? (min(end, input_sizes[dim] + padding[dim]) - start)
      : (end_corrected - start_corrected);
  return {start_corrected, end_corrected, count};
}

// Iterates through all the input elements that this kernel needs to
// apply max to. Specialized for 3 pooling dimensions.
template <typename T, typename IdxT>
void avg_pool_3d_input_iter(
    constant T* input,
    device T* output,
    constant IdxT* input_sizes,
    constant IdxT* input_strides,
    thread IdxT (&pooling_dim_indices)[3],
    constant IdxT* kernel_size,
    constant IdxT* stride,
    constant IdxT* padding,
    bool count_include_pad,
    bool has_divisor_override,
    int32_t divisor_override) {
  auto bounds0 = get_avg_pool_input_iter_bounds<0, IdxT>(
      input_sizes,
      pooling_dim_indices,
      kernel_size,
      stride,
      padding,
      count_include_pad);
  auto bounds1 = get_avg_pool_input_iter_bounds<1, IdxT>(
      input_sizes,
      pooling_dim_indices,
      kernel_size,
      stride,
      padding,
      count_include_pad);
  auto bounds2 = get_avg_pool_input_iter_bounds<2, IdxT>(
      input_sizes,
      pooling_dim_indices,
      kernel_size,
      stride,
      padding,
      count_include_pad);

  opmath_t<T> value_sum = 0;
  opmath_t<T> divisor = has_divisor_override
      ? divisor_override
      : (bounds0.count) * (bounds1.count) * (bounds2.count);

  for (auto i0 = bounds0.start; i0 < bounds0.end; i0++) {
    auto offset0 = input_strides[0] * i0;

    for (auto i1 = bounds1.start; i1 < bounds1.end; i1++) {
      auto offset1 = input_strides[1] * i1;

      for (auto i2 = bounds2.start; i2 < bounds2.end; i2++) {
        auto offset2 = input_strides[2] * i2;
        auto input_value = input[offset0 + offset1 + offset2];
        value_sum += static_cast<opmath_t<T>>(input_value);
      }
    }
  }
  *output = static_cast<T>(value_sum / divisor);
}

// Iterates through all the input elements that this kernel needs to
// apply max to. Specialized for 2 pooling dimensions.
template <typename T, typename IdxT>
void avg_pool_2d_input_iter(
    constant T* input,
    device T* output,
    constant IdxT* input_sizes,
    constant IdxT* input_strides,
    thread IdxT (&pooling_dim_indices)[3],
    constant IdxT* kernel_size,
    constant IdxT* stride,
    constant IdxT* padding,
    bool count_include_pad,
    bool has_divisor_override,
    int32_t divisor_override) {
  auto bounds0 = get_avg_pool_input_iter_bounds<0, IdxT>(
      input_sizes,
      pooling_dim_indices,
      kernel_size,
      stride,
      padding,
      count_include_pad);
  auto bounds1 = get_avg_pool_input_iter_bounds<1, IdxT>(
      input_sizes,
      pooling_dim_indices,
      kernel_size,
      stride,
      padding,
      count_include_pad);

  opmath_t<T> value_sum = 0;
  opmath_t<T> divisor = has_divisor_override
      ? divisor_override
      : (bounds0.count) * (bounds1.count);

  for (auto i0 = bounds0.start; i0 < bounds0.end; i0++) {
    auto offset0 = input_strides[0] * i0;

    for (auto i1 = bounds1.start; i1 < bounds1.end; i1++) {
      auto offset1 = input_strides[1] * i1;
      auto input_value = input[offset0 + offset1];
      value_sum += static_cast<opmath_t<T>>(input_value);
    }
  }
  *output = static_cast<T>(value_sum / divisor);
}

template <typename T, typename IdxT>
void avg_pool_backward_3d_input_iter(
    device AtomicType_t<T>* grad_input,
    constant T* grad_output,
    constant IdxT* grad_input_sizes,
    constant IdxT* grad_input_strides,
    IdxT grad_input_leading_offset,
    thread IdxT (&pooling_dim_indices)[3],
    constant IdxT* kernel_size,
    constant IdxT* stride,
    constant IdxT* padding,
    bool count_include_pad,
    bool has_divisor_override,
    int32_t divisor_override) {
  auto bounds0 = get_avg_pool_input_iter_bounds<0, IdxT>(
      grad_input_sizes,
      pooling_dim_indices,
      kernel_size,
      stride,
      padding,
      count_include_pad);
  auto bounds1 = get_avg_pool_input_iter_bounds<1, IdxT>(
      grad_input_sizes,
      pooling_dim_indices,
      kernel_size,
      stride,
      padding,
      count_include_pad);
  auto bounds2 = get_avg_pool_input_iter_bounds<2, IdxT>(
      grad_input_sizes,
      pooling_dim_indices,
      kernel_size,
      stride,
      padding,
      count_include_pad);

  auto divisor = has_divisor_override
      ? divisor_override
      : (bounds0.count) * (bounds1.count) * (bounds2.count);
  auto grad_val = *grad_output / static_cast<T>(divisor);

  for (auto i0 = bounds0.start; i0 < bounds0.end; i0++) {
    auto offset0 = grad_input_strides[0] * i0;

    for (auto i1 = bounds1.start; i1 < bounds1.end; i1++) {
      auto offset1 = grad_input_strides[1] * i1;

      for (auto i2 = bounds2.start; i2 < bounds2.end; i2++) {
        auto offset2 = grad_input_strides[2] * i2;
        auto pool_offset = offset0 + offset1 + offset2;

        AtomicType<T>::atomic_add(
            grad_input, grad_input_leading_offset + pool_offset, grad_val);
      }
    }
  }
}

// Kernel computes one element of the output per kernel call.
template <typename T, typename IdxT>
kernel void avg_pool(
    constant T* input [[buffer(0)]],
    device T* output [[buffer(1)]],
    constant AvgPoolingParams<5, IdxT>& params [[buffer(2)]],
    uint tid [[thread_position_in_grid]]) {
  const auto gid = static_cast<IdxT>(tid) + params.tid_offset;
  auto pooling_dims = params.pooling_dims;
  auto dims = params.dims;
  auto input_sizes = params.input_sizes.data();
  auto input_strides = params.input_strides.data();
  auto output_sizes = params.output_sizes.data();
  auto output_strides = params.output_strides.data();
  auto kernel_size = params.kernel_size.data();
  auto stride = params.stride.data();
  auto padding = params.padding.data();
  auto leading_dims = dims - pooling_dims;

  // This buffer keeps track of the pooling dimension indices of this thread's
  // element of the output. We need to fill it with the proper values below.
  IdxT pooling_dim_indices[3];

  PoolOffsets<IdxT> offsets = find_pool_offsets<IdxT>(
      output_sizes,
      output_strides,
      /*indices_strides=*/nullptr,
      input_strides,
      pooling_dim_indices,
      dims,
      leading_dims,
      /*return_indices=*/false,
      gid);

  output += offsets.output;
  input += offsets.input_leading;
  input_sizes += leading_dims;
  input_strides += leading_dims;

  if (pooling_dims == 3) {
    avg_pool_3d_input_iter<T, IdxT>(
        input,
        output,
        input_sizes,
        input_strides,
        pooling_dim_indices,
        kernel_size,
        stride,
        padding,
        params.count_include_pad,
        params.has_divisor_override,
        params.divisor_override);
  } else if (pooling_dims == 2) {
    avg_pool_2d_input_iter<T, IdxT>(
        input,
        output,
        input_sizes,
        input_strides,
        pooling_dim_indices,
        kernel_size,
        stride,
        padding,
        params.count_include_pad,
        params.has_divisor_override,
        params.divisor_override);
  }
}

template <typename T, typename IdxT>
kernel void avg_pool_backward(
    device AtomicType_t<T>* grad_input [[buffer(0)]],
    constant T* grad_output [[buffer(1)]],
    constant AvgPoolingParams<5, IdxT>& params [[buffer(2)]],
    uint tid [[thread_position_in_grid]]) {
  const auto gid = static_cast<IdxT>(tid) + params.tid_offset;
  auto pooling_dims = params.pooling_dims;
  auto dims = params.dims;
  auto grad_input_sizes = params.input_sizes.data();
  auto grad_input_strides = params.input_strides.data();
  auto grad_output_sizes = params.output_sizes.data();
  auto grad_output_strides = params.output_strides.data();
  auto kernel_size = params.kernel_size.data();
  auto stride = params.stride.data();
  auto padding = params.padding.data();
  auto leading_dims = dims - pooling_dims;

  // This buffer keeps track of the pooling dimension indices of this thread's
  // element of the output. We need to fill it with the proper values below.
  IdxT pooling_dim_indices[3];

  PoolOffsets<IdxT> offsets = find_pool_offsets<IdxT>(
      grad_output_sizes,
      grad_output_strides,
      /*indices_strides=*/nullptr,
      grad_input_strides,
      pooling_dim_indices,
      dims,
      leading_dims,
      /*return_indices=*/false,
      gid);

  grad_output += offsets.output;
  grad_input_sizes += leading_dims;
  grad_input_strides += leading_dims;

  avg_pool_backward_3d_input_iter<T, IdxT>(
      grad_input,
      grad_output,
      grad_input_sizes,
      grad_input_strides,
      offsets.input_leading,
      pooling_dim_indices,
      kernel_size,
      stride,
      padding,
      params.count_include_pad,
      params.has_divisor_override,
      params.divisor_override);
}

#define REGISTER_POOL_OP_IDX(DTYPE, IDX_T, SUFFIX)                    \
  template[[host_name("max_pool_" #DTYPE SUFFIX)]] kernel void        \
  max_pool<DTYPE, IDX_T>(                                             \
      constant DTYPE * input [[buffer(0)]],                           \
      device DTYPE * output [[buffer(1)]],                            \
      device int64_t* indices [[buffer(2)]],                          \
      constant PoolingParams<5, IDX_T>& params [[buffer(3)]],         \
      uint tid [[thread_position_in_grid]]);                          \
                                                                      \
  template[[host_name("max_unpool_" #DTYPE SUFFIX)]] kernel void      \
  max_unpool<DTYPE, IDX_T>(                                           \
      device DTYPE * output [[buffer(0)]],                            \
      constant DTYPE * input [[buffer(1)]],                           \
      constant int64_t* indices [[buffer(2)]],                        \
      constant MaxUnpoolingParams<5, IDX_T>& params [[buffer(3)]],    \
      device ::c10::metal::ErrorMessages* error_buffer [[buffer(4)]], \
      uint tid [[thread_position_in_grid]]);                          \
                                                                      \
  template[[host_name("avg_pool_" #DTYPE SUFFIX)]] kernel void        \
  avg_pool<DTYPE, IDX_T>(                                             \
      constant DTYPE * input [[buffer(0)]],                           \
      device DTYPE * output [[buffer(1)]],                            \
      constant AvgPoolingParams<5, IDX_T> & params [[buffer(2)]],     \
      uint tid [[thread_position_in_grid]])

#define REGISTER_POOL_OP(DTYPE)                 \
  REGISTER_POOL_OP_IDX(DTYPE, int32_t, "_u32"); \
  REGISTER_POOL_OP_IDX(DTYPE, int64_t, "_u64")

#define REGISTER_POOL_BACKWARD_OP_IDX(DTYPE, IDX_T, SUFFIX)             \
  template[[host_name("max_pool_backward_" #DTYPE SUFFIX)]] kernel void \
  max_pool_backward<DTYPE, IDX_T>(                                      \
      device AtomicType_t<DTYPE> * grad_input [[buffer(0)]],            \
      constant DTYPE * grad_output_ [[buffer(1)]],                      \
      constant int64_t* grad_indices_ [[buffer(2)]],                    \
      constant PoolingBackwardParams<5, IDX_T>& params [[buffer(3)]],   \
      uint tid [[thread_position_in_grid]]);                            \
                                                                        \
  template[[host_name("avg_pool_backward_" #DTYPE SUFFIX)]] kernel void \
  avg_pool_backward<DTYPE, IDX_T>(                                      \
      device AtomicType_t<DTYPE> * grad_input [[buffer(0)]],            \
      constant DTYPE * grad_output [[buffer(1)]],                       \
      constant AvgPoolingParams<5, IDX_T> & params [[buffer(2)]],       \
      uint tid [[thread_position_in_grid]])

#define REGISTER_POOL_BACKWARD_OP(DTYPE)                 \
  REGISTER_POOL_BACKWARD_OP_IDX(DTYPE, int32_t, "_u32"); \
  REGISTER_POOL_BACKWARD_OP_IDX(DTYPE, int64_t, "_u64")

REGISTER_POOL_OP(float);
REGISTER_POOL_OP(half);
REGISTER_POOL_OP(bfloat);
REGISTER_POOL_OP(int);
REGISTER_POOL_OP(long);
REGISTER_POOL_OP(short);
REGISTER_POOL_OP(char);
REGISTER_POOL_OP(uchar);
REGISTER_POOL_OP(bool);

REGISTER_POOL_BACKWARD_OP(float);
REGISTER_POOL_BACKWARD_OP(half);
REGISTER_POOL_BACKWARD_OP(bfloat);
