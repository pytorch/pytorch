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

struct PoolDimGeom {
  constant int32_t* input_sizes;
  constant int32_t* output_sizes;
  constant int32_t* kernel_size;
  constant int32_t* stride;
  constant int32_t* padding;
  constant int32_t* dilation;
  bool adaptive;
};

template <int32_t dim>
IterBounds<int32_t> get_input_iter_bounds(
    constant int32_t* input_sizes,
    thread int32_t (&pooling_dim_indices)[3],
    constant int32_t* kernel_size,
    constant int32_t* stride,
    constant int32_t* padding,
    constant int32_t* dilation) {
  auto d = dilation[dim];
  auto start = stride[dim] * pooling_dim_indices[dim] - padding[dim];
  auto end = min(start + kernel_size[dim] * d, input_sizes[dim]);
  auto start_correction = d * ((-start - 1 + d) / d);
  start += start < 0 ? start_correction : 0;
  return IterBounds<int32_t>{start, end};
}

template <int32_t dim>
IterBounds<int32_t> get_adaptive_input_iter_bounds(
    constant int32_t* input_sizes,
    constant int32_t* output_sizes,
    thread int32_t (&pooling_dim_indices)[3]) {
  auto in_size = static_cast<int64_t>(input_sizes[dim]);
  auto out_size = static_cast<int64_t>(output_sizes[dim]);
  auto out_idx = static_cast<int64_t>(pooling_dim_indices[dim]);
  auto start = static_cast<int32_t>((out_idx * in_size) / out_size);
  auto end =
      static_cast<int32_t>(((out_idx + 1) * in_size + out_size - 1) / out_size);
  return IterBounds<int32_t>{start, end};
}

template <int32_t dim>
IterBounds<int32_t> get_pool_input_iter_bounds(
    thread PoolDimGeom& geom,
    thread int32_t (&pooling_dim_indices)[3]) {
  if (geom.adaptive) {
    return get_adaptive_input_iter_bounds<dim>(
        geom.input_sizes, geom.output_sizes, pooling_dim_indices);
  }
  return get_input_iter_bounds<dim>(
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
template <typename T>
void max_pool_3d_input_iter(
    device const T* input,
    device T* output,
    device int64_t* indices,
    constant int32_t* input_strides,
    thread int32_t (&pooling_dim_indices)[3],
    PoolDimGeom geom,
    bool return_indices) {
  auto bounds0 = get_pool_input_iter_bounds<0>(geom, pooling_dim_indices);
  auto bounds1 = get_pool_input_iter_bounds<1>(geom, pooling_dim_indices);
  auto bounds2 = get_pool_input_iter_bounds<2>(geom, pooling_dim_indices);

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
        // Matches CPU: NaN wins, (val != val) is false for integral types
        bool is_greater = input_value > max_value || input_value != input_value;

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

template <typename T, bool return_indices>
void max_pool_2d_input_iter(
    device const T* input,
    device T* output,
    device int64_t* indices,
    constant int32_t* input_strides,
    thread int32_t (&pooling_dim_indices)[3],
    PoolDimGeom geom) {
  auto bounds0 = get_pool_input_iter_bounds<0>(geom, pooling_dim_indices);
  auto bounds1 = get_pool_input_iter_bounds<1>(geom, pooling_dim_indices);

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
      // Matches CPU: NaN wins, (val != val) is false for integral types
      bool is_greater = input_value > max_value || input_value != input_value;

      max_value = is_greater ? input_value : max_value;

      if IF_CONSTEXPR (return_indices) {
        auto input_index = i0 * geom.input_sizes[1] + i1;
        max_index = is_greater ? input_index : max_index;
      }
    }
  }
  *output = max_value;
  if IF_CONSTEXPR (return_indices) {
    *indices = max_index;
  }
}

struct PoolOffsets {
  int32_t output;
  int32_t indices;
  int32_t input_leading;

  PoolOffsets() : output(0), indices(0), input_leading(0) {}
};

// Finds the offset of the output element that a forward pass thread will
// calculate, `output[N, C, d, h, w]`. Also, find the offset of the input for
// the leading dim indices, `input[N, C]`. Optionally, keep track of the output
// pooling dimension indices, `[d, h , w]`.
// NOTE: This is templated per number of dimensions so that the compiler can
// unroll the loop, giving better performance.
template <int32_t dims>
PoolOffsets find_pool_offsets_dim_specific(
    constant int32_t* output_sizes,
    constant int32_t* output_strides,
    constant int32_t* indices_strides,
    constant int32_t* input_strides,
    int32_t pooling_dim_indices[3],
    int32_t leading_dims,
    bool return_indices,
    uint tid) {
  auto output_idx = static_cast<int32_t>(tid);
  PoolOffsets offsets;

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

PoolOffsets find_pool_offsets(
    constant int32_t* output_sizes,
    constant int32_t* output_strides,
    constant int32_t* indices_strides,
    constant int32_t* input_strides,
    int32_t pooling_dim_indices[3],
    int32_t dims,
    int32_t leading_dims,
    bool return_indices,
    uint tid) {
  switch (dims) {
    case 5:
      return find_pool_offsets_dim_specific<5>(
          output_sizes,
          output_strides,
          indices_strides,
          input_strides,
          pooling_dim_indices,
          leading_dims,
          return_indices,
          tid);
    case 4:
      return find_pool_offsets_dim_specific<4>(
          output_sizes,
          output_strides,
          indices_strides,
          input_strides,
          pooling_dim_indices,
          leading_dims,
          return_indices,
          tid);
    case 3:
      return find_pool_offsets_dim_specific<3>(
          output_sizes,
          output_strides,
          indices_strides,
          input_strides,
          pooling_dim_indices,
          leading_dims,
          return_indices,
          tid);
  }
  return PoolOffsets();
}

// Kernel computes one element of the output per kernel call.
template <typename T>
kernel void max_pool(
    device const T* input [[buffer(0)]],
    device T* output [[buffer(1)]],
    device int64_t* indices [[buffer(2)]],
    constant PoolingParams<5>& params [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
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
  PoolDimGeom geom{
      input_sizes + leading_dims,
      output_sizes + leading_dims,
      kernel_size,
      stride,
      padding,
      dilation,
      params.adaptive};

  // This buffer keeps track of the pooling dimension indices of this thread's
  // element of the output. We need to fill it with the proper values below.
  int32_t pooling_dim_indices[3];

  PoolOffsets offsets = find_pool_offsets(
      output_sizes,
      output_strides,
      return_indices ? indices_strides : nullptr,
      input_strides,
      pooling_dim_indices,
      dims,
      leading_dims,
      return_indices,
      tid);

  output += offsets.output;
  indices += offsets.indices;
  input += offsets.input_leading;

  switch (pooling_dims) {
    case 2:
      if (return_indices) {
        return max_pool_2d_input_iter<T, /*return_indices=*/true>(
            input,
            output,
            indices,
            input_strides + leading_dims,
            pooling_dim_indices,
            geom);
      } else {
        return max_pool_2d_input_iter<T, /*return_indices=*/false>(
            input,
            output,
            indices,
            input_strides + leading_dims,
            pooling_dim_indices,
            geom);
      }
    case 3:
      return max_pool_3d_input_iter<T>(
          input,
          output,
          indices,
          input_strides + leading_dims,
          pooling_dim_indices,
          geom,
          return_indices);
  }
}

// Specialization of max_pool for 2D pooling of dense NCHW or NHWC tensors that
// ignores strides. The grid maps directly to output coordinates, which avoids
// most of the div/mod by runtime sizes that find_pool_offsets needs.
template <typename T, bool return_indices, bool channels_last>
kernel void max_pool2d_dense(
    device const T* input [[buffer(0)]],
    device T* output [[buffer(1)]],
    device int64_t* indices [[buffer(2)]],
    constant PoolingParams<5>& p [[buffer(3)]],
    uint3 pos [[thread_position_in_grid]]) {
  const int32_t C = p.input_sizes[p.dims - 3];
  const int32_t IH = p.input_sizes[p.dims - 2], IW = p.input_sizes[p.dims - 1];
  const int32_t OH = p.output_sizes[p.dims - 2],
                OW = p.output_sizes[p.dims - 1];
  int32_t oh, ow, stride_h, stride_w, in_base, out_offset;
  if IF_CONSTEXPR (channels_last) {
    // pos = (c, ow, n * OH + oh)
    oh = pos.z % OH;
    ow = pos.y;
    stride_w = C;
    stride_h = IW * C;
    in_base = (pos.z / OH) * IH * stride_h + pos.x;
    out_offset = (pos.z * OW + ow) * C + pos.x;
  } else {
    // pos = (ow, oh, n * C + c)
    oh = pos.y;
    ow = pos.x;
    stride_w = 1;
    stride_h = IW;
    in_base = pos.z * IH * IW;
    out_offset = (pos.z * OH + oh) * OW + ow;
  }
  const auto dh = p.dilation[0], dw = p.dilation[1];
  auto h_start = oh * p.stride[0] - p.padding[0];
  auto w_start = ow * p.stride[1] - p.padding[1];
  const auto h_end = min(h_start + (p.kernel_size[0] - 1) * dh + 1, IH);
  const auto w_end = min(w_start + (p.kernel_size[1] - 1) * dw + 1, IW);
  if (h_start < 0) {
    h_start += dh * ((dh - 1 - h_start) / dh);
  }
  if (w_start < 0) {
    w_start += dw * ((dw - 1 - w_start) / dw);
  }

  input += in_base;
  T max_value = input[h_start * stride_h + w_start * stride_w];
  int32_t max_index = h_start * IW + w_start;
  for (auto ih = h_start; ih < h_end; ih += dh) {
    for (auto iw = w_start; iw < w_end; iw += dw) {
      const auto val = input[ih * stride_h + iw * stride_w];
      // Matches CPU: NaN wins, (val != val) is false for integral types
      const bool take = val > max_value || val != val;
      max_value = take ? val : max_value;
      if IF_CONSTEXPR (return_indices) {
        max_index = take ? ih * IW + iw : max_index;
      }
    }
  }
  output[out_offset] = max_value;
  if IF_CONSTEXPR (return_indices) {
    indices[out_offset] = max_index;
  }
}

// Stride-1 2D max pooling of a dense NCHW tensor without dilation along W.
// Each thread computes a tile of kCols x kRows outputs, so each input row is
// loaded once per tile and reduced along W before being combined into all
// output rows whose window covers it. Columns are clamped to the input, which
// does not change the result, as without dilation along W the clamped column is
// always within the window. KW is the window width if known at compile time
// (which lets the row be kept in registers), or 0.
template <typename T, int KW>
kernel void max_pool2d_s1_nchw(
    device const T* input [[buffer(0)]],
    device T* output [[buffer(1)]],
    constant PoolingParams<5>& p [[buffer(3)]],
    uint3 pos [[thread_position_in_grid]]) {
  constexpr int kCols = max_pool2d_s1_tile_cols;
  constexpr int kRows = max_pool2d_s1_tile_rows;
  const int32_t IH = p.input_sizes[p.dims - 2], IW = p.input_sizes[p.dims - 1];
  const int32_t OH = p.output_sizes[p.dims - 2],
                OW = p.output_sizes[p.dims - 1];
  const auto kh = p.kernel_size[0], kw = KW > 0 ? KW : p.kernel_size[1];
  const auto dh = p.dilation[0];
  const int32_t ow0 = pos.x * kCols;
  const int32_t oh0 = pos.y * kRows;
  const int32_t c0 = ow0 - p.padding[1];
  input += pos.z * IH * IW;

  ::metal::array<::metal::array<T, kCols>, kRows> acc;
  ::metal::array<int32_t, kRows> first_row;
  for (int u = 0; u < kRows; ++u) {
    for (int v = 0; v < kCols; ++v) {
      acc[u][v] = T(0);
    }
    auto h = oh0 + u - p.padding[0];
    if (h < 0) {
      h += dh * ((dh - 1 - h) / dh);
    }
    first_row[u] = h;
  }

  const auto ih_begin = max(oh0 - p.padding[0], 0);
  const auto ih_end = min(oh0 + kRows - p.padding[0] + (kh - 1) * dh, IH);
  for (auto ih = ih_begin; ih < ih_end; ++ih) {
    const auto row = input + ih * IW;
    ::metal::array<T, kCols> row_max;
    if IF_CONSTEXPR (KW > 0) {
      ::metal::array<T, kCols + KW - 1> vals;
      for (int i = 0; i < kCols + KW - 1; ++i) {
        vals[i] = row[clamp(c0 + i, 0, IW - 1)];
      }
      for (int v = 0; v < kCols; ++v) {
        row_max[v] = vals[v];
        for (int j = 1; j < KW; ++j) {
          // Matches CPU: NaN wins, (val != val) is false for integral types
          const bool take =
              vals[v + j] > row_max[v] || vals[v + j] != vals[v + j];
          row_max[v] = take ? vals[v + j] : row_max[v];
        }
      }
    } else {
      for (int v = 0; v < kCols; ++v) {
        row_max[v] = row[clamp(c0 + v, 0, IW - 1)];
      }
      for (int32_t j = 1; j < kw; ++j) {
        for (int v = 0; v < kCols; ++v) {
          const T val = row[clamp(c0 + v + j, 0, IW - 1)];
          const bool take = val > row_max[v] || val != val;
          row_max[v] = take ? val : row_max[v];
        }
      }
    }
    for (int u = 0; u < kRows; ++u) {
      const auto rel = ih - (oh0 + u - p.padding[0]);
      if (rel < 0 || rel > (kh - 1) * dh || (dh != 1 && rel % dh != 0)) {
        continue;
      }
      const bool first = ih == first_row[u];
      for (int v = 0; v < kCols; ++v) {
        const bool take =
            first || row_max[v] > acc[u][v] || row_max[v] != row_max[v];
        acc[u][v] = take ? row_max[v] : acc[u][v];
      }
    }
  }

  const auto out_base = (pos.z * OH + oh0) * OW + ow0;
  for (int u = 0; u < kRows && oh0 + u < OH; ++u) {
    for (int v = 0; v < kCols && ow0 + v < OW; ++v) {
      output[out_base + u * OW + v] = acc[u][v];
    }
  }
}

// Finds the element in the grad input which corresponds to the index into the
// pool, and then adds the grad output element to it.
template <typename T>
void max_pool_backward_impl(
    device AtomicType_t<T>* grad_input,
    T grad_output_element,
    int32_t input_index,
    constant int32_t* grad_input_sizes,
    constant int32_t* grad_input_strides,
    int32_t grad_input_leading_offset,
    int32_t pooling_dims) {
  int32_t size_prod = 1;
  int32_t pool_offset = 0;

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
template <typename T>
kernel void max_pool_backward(
    device AtomicType_t<T>* grad_input [[buffer(0)]],
    device const T* grad_output [[buffer(1)]],
    device const int64_t* indices [[buffer(2)]],
    constant PoolingBackwardParams<5>& params [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  auto pooling_dims = params.pooling_dims;
  auto dims = params.dims;
  auto grad_input_sizes = params.grad_input_sizes.data();
  auto grad_input_strides = params.grad_input_strides.data();
  auto grad_output_sizes = params.grad_output_sizes.data();
  auto grad_output_strides = params.grad_output_strides.data();
  auto indices_strides = params.indices_strides.data();

  auto leading_dims = dims - pooling_dims;

  PoolOffsets offsets = find_pool_offsets(
      grad_output_sizes,
      grad_output_strides,
      indices_strides,
      grad_input_strides,
      nullptr,
      dims,
      leading_dims,
      /*return_indices=*/true,
      tid);

  max_pool_backward_impl<T>(
      grad_input,
      grad_output[offsets.output],
      indices[offsets.indices],
      grad_input_sizes + leading_dims,
      grad_input_strides + leading_dims,
      offsets.input_leading,
      pooling_dims);
}

template <typename T>
void max_unpool_impl(
    device T* output,
    T input_element,
    int32_t input_index,
    constant int32_t* output_sizes,
    constant int32_t* output_strides,
    int32_t pooling_dims,
    device c10::metal::ErrorMessages* error_buffer) {
  int32_t size_prod = 1;
  int32_t pool_offset = 0;

  for (auto dim = pooling_dims - 1; dim >= 0; dim--) {
    auto next_size_prod = output_sizes[dim] * size_prod;
    pool_offset +=
        output_strides[dim] * ((input_index % next_size_prod) / size_prod);
    size_prod *= output_sizes[dim];
  }

  // Check that the index is within the valid output range
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

  output[pool_offset] = input_element;
}

// Kernel computes one element of the grad input per kernel call.
template <typename T>
kernel void max_unpool(
    device T* output [[buffer(0)]],
    device const T* input [[buffer(1)]],
    device const int64_t* indices [[buffer(2)]],
    constant MaxUnpoolingParams<5>& params [[buffer(3)]],
    device c10::metal::ErrorMessages* error_buffer [[buffer(4)]],
    uint tid [[thread_position_in_grid]]) {
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
  PoolOffsets offsets = find_pool_offsets(
      /*output_sizes=*/input_sizes,
      /*output_strides=*/input_strides,
      indices_strides,
      /*input_strides=*/output_strides,
      /*pooling_dim_indices=*/nullptr,
      dims,
      leading_dims,
      /*return_indices=*/true,
      tid);

  max_unpool_impl<T>(
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

template <int32_t dim>
AvgPoolIterBounds<int32_t> get_avg_pool_input_iter_bounds(
    constant int32_t* input_sizes,
    thread int32_t (&pooling_dim_indices)[3],
    constant int32_t* kernel_size,
    constant int32_t* stride,
    constant int32_t* padding,
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
template <typename T>
void avg_pool_3d_input_iter(
    device const T* input,
    device T* output,
    constant int32_t* input_sizes,
    constant int32_t* input_strides,
    thread int32_t (&pooling_dim_indices)[3],
    constant int32_t* kernel_size,
    constant int32_t* stride,
    constant int32_t* padding,
    bool count_include_pad,
    bool has_divisor_override,
    int32_t divisor_override) {
  auto bounds0 = get_avg_pool_input_iter_bounds<0>(
      input_sizes,
      pooling_dim_indices,
      kernel_size,
      stride,
      padding,
      count_include_pad);
  auto bounds1 = get_avg_pool_input_iter_bounds<1>(
      input_sizes,
      pooling_dim_indices,
      kernel_size,
      stride,
      padding,
      count_include_pad);
  auto bounds2 = get_avg_pool_input_iter_bounds<2>(
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
template <typename T>
void avg_pool_2d_input_iter(
    device const T* input,
    device T* output,
    constant int32_t* input_sizes,
    constant int32_t* input_strides,
    thread int32_t (&pooling_dim_indices)[3],
    constant int32_t* kernel_size,
    constant int32_t* stride,
    constant int32_t* padding,
    bool count_include_pad,
    bool has_divisor_override,
    int32_t divisor_override) {
  auto bounds0 = get_avg_pool_input_iter_bounds<0>(
      input_sizes,
      pooling_dim_indices,
      kernel_size,
      stride,
      padding,
      count_include_pad);
  auto bounds1 = get_avg_pool_input_iter_bounds<1>(
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

template <typename T>
void avg_pool_backward_3d_input_iter(
    device AtomicType_t<T>* grad_input,
    device const T* grad_output,
    constant int32_t* grad_input_sizes,
    constant int32_t* grad_input_strides,
    int32_t grad_input_leading_offset,
    thread int32_t (&pooling_dim_indices)[3],
    constant int32_t* kernel_size,
    constant int32_t* stride,
    constant int32_t* padding,
    bool count_include_pad,
    bool has_divisor_override,
    int32_t divisor_override) {
  auto bounds0 = get_avg_pool_input_iter_bounds<0>(
      grad_input_sizes,
      pooling_dim_indices,
      kernel_size,
      stride,
      padding,
      count_include_pad);
  auto bounds1 = get_avg_pool_input_iter_bounds<1>(
      grad_input_sizes,
      pooling_dim_indices,
      kernel_size,
      stride,
      padding,
      count_include_pad);
  auto bounds2 = get_avg_pool_input_iter_bounds<2>(
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
template <typename T>
kernel void avg_pool(
    device const T* input [[buffer(0)]],
    device T* output [[buffer(1)]],
    constant AvgPoolingParams<5>& params [[buffer(2)]],
    uint tid [[thread_position_in_grid]]) {
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
  int32_t pooling_dim_indices[3];

  PoolOffsets offsets = find_pool_offsets(
      output_sizes,
      output_strides,
      /*indices_strides=*/nullptr,
      input_strides,
      pooling_dim_indices,
      dims,
      leading_dims,
      /*return_indices=*/false,
      tid);

  output += offsets.output;
  input += offsets.input_leading;
  input_sizes += leading_dims;
  input_strides += leading_dims;

  if (pooling_dims == 3) {
    avg_pool_3d_input_iter<T>(
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
    avg_pool_2d_input_iter<T>(
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

template <typename T>
kernel void avg_pool_backward(
    device AtomicType_t<T>* grad_input [[buffer(0)]],
    device const T* grad_output [[buffer(1)]],
    constant AvgPoolingParams<5>& params [[buffer(2)]],
    uint tid [[thread_position_in_grid]]) {
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
  int32_t pooling_dim_indices[3];

  PoolOffsets offsets = find_pool_offsets(
      grad_output_sizes,
      grad_output_strides,
      /*indices_strides=*/nullptr,
      grad_input_strides,
      pooling_dim_indices,
      dims,
      leading_dims,
      /*return_indices=*/false,
      tid);

  grad_output += offsets.output;
  grad_input_sizes += leading_dims;
  grad_input_strides += leading_dims;

  avg_pool_backward_3d_input_iter<T>(
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

#define REGISTER_MAX_POOL2D_DENSE(DTYPE, IDX, CL, SUFFIX)       \
  template [[host_name("max_pool2d_dense_" SUFFIX "_" #DTYPE)]] \
  kernel void max_pool2d_dense<DTYPE, IDX, CL>(                 \
      device const DTYPE* input [[buffer(0)]],                  \
      device DTYPE* output [[buffer(1)]],                       \
      device int64_t* indices [[buffer(2)]],                    \
      constant PoolingParams<5>& p [[buffer(3)]],               \
      uint3 pos [[thread_position_in_grid]]);

#define REGISTER_MAX_POOL2D_S1(DTYPE, KW)                       \
  template [[host_name("max_pool2d_s1_nchw_k" #KW "_" #DTYPE)]] \
  kernel void max_pool2d_s1_nchw<DTYPE, KW>(                    \
      device const DTYPE* input [[buffer(0)]],                  \
      device DTYPE* output [[buffer(1)]],                       \
      constant PoolingParams<5>& p [[buffer(3)]],               \
      uint3 pos [[thread_position_in_grid]]);

#define REGISTER_POOL_OP(DTYPE)                                               \
  template [[host_name("max_pool_" #DTYPE)]] kernel void max_pool<DTYPE>(     \
      device const DTYPE* input [[buffer(0)]],                                \
      device DTYPE* output [[buffer(1)]],                                     \
      device int64_t* indices [[buffer(2)]],                                  \
      constant PoolingParams<5>& params [[buffer(3)]],                        \
      uint tid [[thread_position_in_grid]]);                                  \
                                                                              \
  template [[host_name("max_unpool_" #DTYPE)]] kernel void max_unpool<DTYPE>( \
      device DTYPE * output [[buffer(0)]],                                    \
      device const DTYPE* input [[buffer(1)]],                                \
      device const int64_t* indices [[buffer(2)]],                            \
      constant MaxUnpoolingParams<5>& params [[buffer(3)]],                   \
      device ::c10::metal::ErrorMessages* error_buffer [[buffer(4)]],         \
      uint tid [[thread_position_in_grid]]);                                  \
                                                                              \
  template [[host_name("avg_pool_" #DTYPE)]] kernel void avg_pool<DTYPE>(     \
      device const DTYPE* input [[buffer(0)]],                                \
      device DTYPE* output [[buffer(1)]],                                     \
      constant AvgPoolingParams<5>& params [[buffer(2)]],                     \
      uint tid [[thread_position_in_grid]]);                                  \
  REGISTER_MAX_POOL2D_DENSE(DTYPE, false, false, "nchw")                      \
  REGISTER_MAX_POOL2D_DENSE(DTYPE, true, false, "nchw_idx")                   \
  REGISTER_MAX_POOL2D_DENSE(DTYPE, false, true, "nhwc")                       \
  REGISTER_MAX_POOL2D_DENSE(DTYPE, true, true, "nhwc_idx")                    \
  REGISTER_MAX_POOL2D_S1(DTYPE, 0)                                            \
  REGISTER_MAX_POOL2D_S1(DTYPE, 2)                                            \
  REGISTER_MAX_POOL2D_S1(DTYPE, 3)                                            \
  REGISTER_MAX_POOL2D_S1(DTYPE, 5)

#define REGISTER_POOL_BACKWARD_OP(DTYPE)                       \
  template [[host_name("max_pool_backward_" #DTYPE)]]          \
  kernel void max_pool_backward<DTYPE>(                        \
      device AtomicType_t<DTYPE> * grad_input [[buffer(0)]],   \
      device const DTYPE* grad_output_ [[buffer(1)]],          \
      device const int64_t* grad_indices_ [[buffer(2)]],       \
      constant PoolingBackwardParams<5>& params [[buffer(3)]], \
      uint tid [[thread_position_in_grid]]);                   \
                                                               \
  template [[host_name("avg_pool_backward_" #DTYPE)]]          \
  kernel void avg_pool_backward<DTYPE>(                        \
      device AtomicType_t<DTYPE> * grad_input [[buffer(0)]],   \
      device const DTYPE* grad_output [[buffer(1)]],           \
      constant AvgPoolingParams<5>& params [[buffer(2)]],      \
      uint tid [[thread_position_in_grid]]);

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
