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

// idx_t must be wide enough to hold (out_idx + 1) * in_size + out_size
template <typename idx_t>
IterBounds<int32_t> get_adaptive_bounds(
    idx_t out_idx,
    idx_t in_size,
    idx_t out_size) {
  auto start = static_cast<int32_t>((out_idx * in_size) / out_size);
  auto end =
      static_cast<int32_t>(((out_idx + 1) * in_size + out_size - 1) / out_size);
  return IterBounds<int32_t>{start, end};
}

template <int32_t dim, typename idx_t>
IterBounds<int32_t> get_adaptive_input_iter_bounds(
    constant int32_t* input_sizes,
    constant int32_t* output_sizes,
    thread int32_t (&pooling_dim_indices)[3]) {
  return get_adaptive_bounds<idx_t>(
      pooling_dim_indices[dim], input_sizes[dim], output_sizes[dim]);
}

template <int32_t dim, typename idx_t>
IterBounds<int32_t> get_pool_input_iter_bounds(
    thread PoolDimGeom& geom,
    thread int32_t (&pooling_dim_indices)[3]) {
  if (geom.adaptive) {
    return get_adaptive_input_iter_bounds<dim, idx_t>(
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

// Replaces max_value with val if val is greater or NaN, which matches CPU
// behavior, and returns whether it was replaced
template <typename T>
inline bool update_max(thread T& max_value, T val) {
  bool take = val > max_value;
  if IF_CONSTEXPR (::metal::is_floating_point_v<T>) {
    take = take || val != val;
  }
  max_value = take ? val : max_value;
  return take;
}

// Iterates through all the input elements that this kernel needs to
// apply max to. Specialized for 3 pooling dimensions.
// TODO: Support any number of pooling dims
template <typename T, typename idx_t>
void max_pool_3d_input_iter(
    device const T* input,
    device T* output,
    device int64_t* indices,
    constant int64_t* input_strides,
    thread int32_t (&pooling_dim_indices)[3],
    PoolDimGeom geom,
    bool return_indices) {
  auto bounds0 =
      get_pool_input_iter_bounds<0, idx_t>(geom, pooling_dim_indices);
  auto bounds1 =
      get_pool_input_iter_bounds<1, idx_t>(geom, pooling_dim_indices);
  auto bounds2 =
      get_pool_input_iter_bounds<2, idx_t>(geom, pooling_dim_indices);

  auto d0 = geom.dilation[0];
  auto d1 = geom.dilation[1];
  auto d2 = geom.dilation[2];

  const idx_t stride0 = input_strides[0], stride1 = input_strides[1],
              stride2 = input_strides[2];
  T max_value = input
      [stride0 * bounds0.start + stride1 * bounds1.start +
       stride2 * bounds2.start];
  const idx_t size12 = idx_t(geom.input_sizes[1]) * geom.input_sizes[2];
  idx_t max_index = bounds0.start * size12 +
      idx_t(bounds1.start) * geom.input_sizes[2] + bounds2.start;

  for (auto i0 = bounds0.start; i0 < bounds0.end; i0 += d0) {
    auto offset0 = stride0 * i0;

    for (auto i1 = bounds1.start; i1 < bounds1.end; i1 += d1) {
      auto offset1 = stride1 * i1;

      for (auto i2 = bounds2.start; i2 < bounds2.end; i2 += d2) {
        auto offset2 = stride2 * i2;
        const bool is_greater =
            update_max(max_value, input[offset0 + offset1 + offset2]);
        if (return_indices) {
          auto input_index = i0 * size12 + idx_t(i1) * geom.input_sizes[2] + i2;
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

template <typename T, typename idx_t, bool return_indices>
void max_pool_2d_input_iter(
    device const T* input,
    device T* output,
    device int64_t* indices,
    constant int64_t* input_strides,
    thread int32_t (&pooling_dim_indices)[3],
    PoolDimGeom geom) {
  auto bounds0 =
      get_pool_input_iter_bounds<0, idx_t>(geom, pooling_dim_indices);
  auto bounds1 =
      get_pool_input_iter_bounds<1, idx_t>(geom, pooling_dim_indices);

  auto d0 = geom.dilation[0];
  auto d1 = geom.dilation[1];

  const idx_t stride0 = input_strides[0], stride1 = input_strides[1];
  T max_value = input[stride0 * bounds0.start + stride1 * bounds1.start];
  idx_t max_index = idx_t(bounds0.start) * geom.input_sizes[1] + bounds1.start;

  for (auto i0 = bounds0.start; i0 < bounds0.end; i0 += d0) {
    auto offset0 = stride0 * i0;

    for (auto i1 = bounds1.start; i1 < bounds1.end; i1 += d1) {
      auto offset1 = stride1 * i1;

      const bool is_greater = update_max(max_value, input[offset0 + offset1]);
      if IF_CONSTEXPR (return_indices) {
        auto input_index = idx_t(i0) * geom.input_sizes[1] + i1;
        max_index = is_greater ? input_index : max_index;
      }
    }
  }
  *output = max_value;
  if IF_CONSTEXPR (return_indices) {
    *indices = max_index;
  }
}

template <typename idx_t>
struct PoolOffsets {
  idx_t output;
  idx_t indices;
  idx_t input_leading;

  PoolOffsets() : output(0), indices(0), input_leading(0) {}
};

// Finds the offset of the output element that a forward pass thread will
// calculate, `output[N, C, d, h, w]`. Also, find the offset of the input for
// the leading dim indices, `input[N, C]`. Optionally, keep track of the output
// pooling dimension indices, `[d, h , w]`.
// NOTE: This is templated per number of dimensions so that the compiler can
// unroll the loop, giving better performance.
template <int32_t dims, typename idx_t, typename stride_t>
PoolOffsets<idx_t> find_pool_offsets_dim_specific(
    constant int32_t* output_sizes,
    constant stride_t* output_strides,
    constant stride_t* indices_strides,
    constant stride_t* input_strides,
    int32_t pooling_dim_indices[3],
    int32_t leading_dims,
    bool return_indices,
    uint tid) {
  // Signed division is faster, but there can be more than 2^31 threads when
  // 64-bit indexing is needed
  using pos_t = ::metal::conditional_t<sizeof(idx_t) == 8, uint, int32_t>;
  auto output_idx = static_cast<pos_t>(tid);
  PoolOffsets<idx_t> offsets;

  for (auto dim = dims - 1; dim >= 0; dim--) {
    auto dim_idx = output_idx % static_cast<pos_t>(output_sizes[dim]);
    offsets.output += idx_t(output_strides[dim]) * dim_idx;
    if (return_indices) {
      offsets.indices += idx_t(indices_strides[dim]) * dim_idx;
    }

    if (dim < leading_dims) {
      offsets.input_leading += idx_t(input_strides[dim]) * dim_idx;
    } else {
      // Keep track of pooling dimension indices of the output element, so we
      // can use them in the input iteration later on.
      if (pooling_dim_indices != nullptr) {
        pooling_dim_indices[dim - leading_dims] = dim_idx;
      }
    }
    output_idx = output_idx / static_cast<pos_t>(output_sizes[dim]);
  }

  return offsets;
}

template <typename idx_t, typename stride_t>
PoolOffsets<idx_t> find_pool_offsets(
    constant int32_t* output_sizes,
    constant stride_t* output_strides,
    constant stride_t* indices_strides,
    constant stride_t* input_strides,
    int32_t pooling_dim_indices[3],
    int32_t dims,
    int32_t leading_dims,
    bool return_indices,
    uint tid) {
  switch (dims) {
    case 5:
      return find_pool_offsets_dim_specific<5, idx_t, stride_t>(
          output_sizes,
          output_strides,
          indices_strides,
          input_strides,
          pooling_dim_indices,
          leading_dims,
          return_indices,
          tid);
    case 4:
      return find_pool_offsets_dim_specific<4, idx_t, stride_t>(
          output_sizes,
          output_strides,
          indices_strides,
          input_strides,
          pooling_dim_indices,
          leading_dims,
          return_indices,
          tid);
    case 3:
      return find_pool_offsets_dim_specific<3, idx_t, stride_t>(
          output_sizes,
          output_strides,
          indices_strides,
          input_strides,
          pooling_dim_indices,
          leading_dims,
          return_indices,
          tid);
  }
  return PoolOffsets<idx_t>();
}

// Kernel computes one element of the output per kernel call. idx_t is the
// type of the offsets and adaptive window bounds, int32_t unless 64-bit
// indexing is needed.
template <typename T, typename idx_t>
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

  auto offsets = find_pool_offsets<idx_t>(
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
        return max_pool_2d_input_iter<T, idx_t, /*return_indices=*/true>(
            input,
            output,
            indices,
            input_strides + leading_dims,
            pooling_dim_indices,
            geom);
      } else {
        return max_pool_2d_input_iter<T, idx_t, /*return_indices=*/false>(
            input,
            output,
            indices,
            input_strides + leading_dims,
            pooling_dim_indices,
            geom);
      }
    case 3:
      return max_pool_3d_input_iter<T, idx_t>(
          input,
          output,
          indices,
          input_strides + leading_dims,
          pooling_dim_indices,
          geom,
          return_indices);
  }
}

// Returns the offsets of the input image (NHWC) or plane (NCHW) and of the
// output element for max_pool2d_dense, given the input and output [C, H, W]
// sizes. Only those offsets need 64-bit math.
template <bool channels_last>
inline long2 max_pool2d_dense_offsets(
    uint3 pos,
    constant int32_t* in_sizes,
    constant int32_t* out_sizes) {
  const auto C = in_sizes[0];
  if IF_CONSTEXPR (channels_last) {
    return long2(
        int64_t(pos.z / out_sizes[1]) * (in_sizes[1] * in_sizes[2] * C) + pos.x,
        int64_t(pos.z) * (out_sizes[2] * C) + (pos.y * C + pos.x));
  }
  return long2(
      int64_t(pos.z) * (in_sizes[1] * in_sizes[2]),
      int64_t(pos.z) * (out_sizes[1] * out_sizes[2]) +
          (pos.y * out_sizes[2] + pos.x));
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
  const int32_t OH = p.output_sizes[p.dims - 2];
  // pos is (c, ow, n * OH + oh) for channels_last, else (ow, oh, n * C + c)
  const int32_t oh = channels_last ? pos.z % OH : pos.y;
  const int32_t ow = channels_last ? pos.y : pos.x;
  const int32_t stride_w = channels_last ? C : 1;
  const int32_t stride_h = IW * stride_w;
  const auto offsets = max_pool2d_dense_offsets<channels_last>(
      pos,
      p.input_sizes.data() + p.dims - 3,
      p.output_sizes.data() + p.dims - 3);
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

  input += offsets.x;
  T max_value = input[h_start * stride_h + w_start * stride_w];
  int32_t max_index = h_start * IW + w_start;
  for (auto ih = h_start; ih < h_end; ih += dh) {
    for (auto iw = w_start; iw < w_end; iw += dw) {
      const bool take =
          update_max(max_value, input[ih * stride_h + iw * stride_w]);
      if IF_CONSTEXPR (return_indices) {
        max_index = take ? ih * IW + iw : max_index;
      }
    }
  }
  output[offsets.y] = max_value;
  if IF_CONSTEXPR (return_indices) {
    indices[offsets.y] = max_index;
  }
}

// Computes out[v] = max(row[c0 + v], ..., row[c0 + v + kw - 1]) for v < G,
// assuming kw >= G. All G windows share [c0 + G - 1, c0 + kw), and each one
// adds a suffix of the G - 1 columns before it and a prefix of the G - 1
// columns after it, so only kw + G - 1 elements are loaded. Columns are clamped
// to [0, IW), see max_pool2d_s1_nchw.
template <int G, typename T>
inline void max_pool_windows(
    device const T* row,
    int32_t c0,
    int32_t kw,
    int32_t IW,
    thread T* out) {
  T shared = row[clamp(c0 + G - 1, 0, IW - 1)];
  for (int32_t j = G; j < kw; ++j) {
    update_max(shared, row[clamp(c0 + j, 0, IW - 1)]);
  }
  out[G - 1] = shared;
  for (int v = G - 2; v >= 0; --v) {
    out[v] = out[v + 1];
    update_max(out[v], row[clamp(c0 + v, 0, IW - 1)]);
  }
  T suffix = row[clamp(c0 + kw, 0, IW - 1)];
  for (int v = 1; v < G; ++v) {
    update_max(out[v], suffix);
    if (v + 1 < G) {
      update_max(suffix, row[clamp(c0 + kw + v, 0, IW - 1)]);
    }
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
  input += int64_t(pos.z) * (IH * IW);

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
          update_max(row_max[v], vals[v + j]);
        }
      }
    } else if (kw >= kCols) {
      max_pool_windows<kCols>(row, c0, kw, IW, row_max.data());
    } else if (kw >= kCols / 2) {
      max_pool_windows<kCols / 2>(row, c0, kw, IW, row_max.data());
      max_pool_windows<kCols / 2>(
          row, c0 + kCols / 2, kw, IW, row_max.data() + kCols / 2);
    } else {
      for (int v = 0; v < kCols; ++v) {
        row_max[v] = row[clamp(c0 + v, 0, IW - 1)];
      }
      for (int32_t j = 1; j < kw; ++j) {
        for (int v = 0; v < kCols; ++v) {
          update_max(row_max[v], row[clamp(c0 + v + j, 0, IW - 1)]);
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
        acc[u][v] = first ? row_max[v] : acc[u][v];
        update_max(acc[u][v], row_max[v]);
      }
    }
  }

  const auto out_base = int64_t(pos.z) * (OH * OW) + (oh0 * OW + ow0);
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
void max_pool_backward_kernel_impl(
    device AtomicType_t<T>* grad_input,
    device const T* grad_output,
    device const int64_t* indices,
    constant PoolingBackwardParams<5>& params,
    uint tid) {
  auto pooling_dims = params.pooling_dims;
  auto dims = params.dims;
  auto grad_input_sizes = params.grad_input_sizes.data();
  auto grad_input_strides = params.grad_input_strides.data();
  auto grad_output_sizes = params.grad_output_sizes.data();
  auto grad_output_strides = params.grad_output_strides.data();
  auto indices_strides = params.indices_strides.data();

  auto leading_dims = dims - pooling_dims;

  auto offsets = find_pool_offsets<int32_t>(
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

template <typename T, bool serial>
kernel void max_pool_backward(
    device AtomicType_t<T>* grad_input [[buffer(0)]],
    device const T* grad_output [[buffer(1)]],
    device const int64_t* indices [[buffer(2)]],
    constant PoolingBackwardParams<5>& params [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  if IF_CONSTEXPR (!serial) {
    max_pool_backward_kernel_impl<T>(
        grad_input, grad_output, indices, params, tid);
    return;
  }
  uint pool_size = 1;
  for (auto dim = params.dims - params.pooling_dims; dim < params.dims; dim++) {
    pool_size *= params.grad_output_sizes[dim];
  }
  for (uint pool_idx = 0; pool_idx < pool_size; pool_idx++) {
    max_pool_backward_kernel_impl<T>(
        grad_input, grad_output, indices, params, tid * pool_size + pool_idx);
  }
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
  auto offsets = find_pool_offsets<int32_t>(
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

template <int32_t dim, typename idx_t = int32_t>
AvgPoolIterBounds<int32_t> get_avg_pool_input_iter_bounds(
    constant int32_t* input_sizes,
    thread int32_t (&pooling_dim_indices)[3],
    constant int32_t* kernel_size,
    constant int32_t* stride,
    constant int32_t* padding,
    bool count_include_pad,
    bool adaptive = false,
    constant int32_t* output_sizes = nullptr) {
  if (adaptive) {
    auto bounds = get_adaptive_input_iter_bounds<dim, idx_t>(
        input_sizes, output_sizes, pooling_dim_indices);
    return {bounds.start, bounds.end, bounds.end - bounds.start};
  }
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
template <typename T, typename idx_t>
void avg_pool_3d_input_iter(
    device const T* input,
    device T* output,
    constant int32_t* input_sizes,
    constant int32_t* input_strides,
    constant int32_t* output_sizes,
    thread int32_t (&pooling_dim_indices)[3],
    constant int32_t* kernel_size,
    constant int32_t* stride,
    constant int32_t* padding,
    bool count_include_pad,
    bool adaptive,
    bool has_divisor_override,
    int32_t divisor_override) {
  auto bounds0 = get_avg_pool_input_iter_bounds<0, idx_t>(
      input_sizes,
      pooling_dim_indices,
      kernel_size,
      stride,
      padding,
      count_include_pad,
      adaptive,
      output_sizes);
  auto bounds1 = get_avg_pool_input_iter_bounds<1, idx_t>(
      input_sizes,
      pooling_dim_indices,
      kernel_size,
      stride,
      padding,
      count_include_pad,
      adaptive,
      output_sizes);
  auto bounds2 = get_avg_pool_input_iter_bounds<2, idx_t>(
      input_sizes,
      pooling_dim_indices,
      kernel_size,
      stride,
      padding,
      count_include_pad,
      adaptive,
      output_sizes);

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
template <typename T, typename idx_t>
void avg_pool_2d_input_iter(
    device const T* input,
    device T* output,
    constant int32_t* input_sizes,
    constant int32_t* input_strides,
    constant int32_t* output_sizes,
    thread int32_t (&pooling_dim_indices)[3],
    constant int32_t* kernel_size,
    constant int32_t* stride,
    constant int32_t* padding,
    bool count_include_pad,
    bool adaptive,
    bool has_divisor_override,
    int32_t divisor_override) {
  auto bounds0 = get_avg_pool_input_iter_bounds<0, idx_t>(
      input_sizes,
      pooling_dim_indices,
      kernel_size,
      stride,
      padding,
      count_include_pad,
      adaptive,
      output_sizes);
  auto bounds1 = get_avg_pool_input_iter_bounds<1, idx_t>(
      input_sizes,
      pooling_dim_indices,
      kernel_size,
      stride,
      padding,
      count_include_pad,
      adaptive,
      output_sizes);

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
template <typename T, typename idx_t>
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

  auto offsets = find_pool_offsets<int32_t, int32_t>(
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
    avg_pool_3d_input_iter<T, idx_t>(
        input,
        output,
        input_sizes,
        input_strides,
        output_sizes + leading_dims,
        pooling_dim_indices,
        kernel_size,
        stride,
        padding,
        params.count_include_pad,
        params.adaptive,
        params.has_divisor_override,
        params.divisor_override);
  } else if (pooling_dims == 2) {
    avg_pool_2d_input_iter<T, idx_t>(
        input,
        output,
        input_sizes,
        input_strides,
        output_sizes + leading_dims,
        pooling_dim_indices,
        kernel_size,
        stride,
        padding,
        params.count_include_pad,
        params.adaptive,
        params.has_divisor_override,
        params.divisor_override);
  }
}

template <typename T>
void avg_pool_backward_kernel_impl(
    device AtomicType_t<T>* grad_input,
    device const T* grad_output,
    constant AvgPoolingParams<5>& params,
    uint tid) {
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

  auto offsets = find_pool_offsets<int32_t, int32_t>(
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

// One simdgroup per output element, with lanes splitting the innermost window
// dim. For large adaptive windows, where one thread per output leaves the GPU
// mostly idle while each thread sums a whole window serially.
template <typename T, typename idx_t>
kernel void adaptive_avg_pool_simd(
    device const T* input [[buffer(0)]],
    device T* output [[buffer(1)]],
    constant AvgPoolingParams<5>& params [[buffer(2)]],
    uint tid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_width [[threads_per_simdgroup]]) {
  auto pooling_dims = params.pooling_dims;
  auto leading_dims = params.dims - pooling_dims;
  int32_t pooling_dim_indices[3];
  auto offsets = find_pool_offsets<int32_t, int32_t>(
      params.output_sizes.data(),
      params.output_strides.data(),
      /*indices_strides=*/nullptr,
      params.input_strides.data(),
      pooling_dim_indices,
      params.dims,
      leading_dims,
      /*return_indices=*/false,
      tid / simd_width);
  input += offsets.input_leading;

  // Pooling dims are right-aligned, so that bounds[2] is the innermost one
  IterBounds<int32_t> bounds[3] = {{0, 1}, {0, 1}, {0, 1}};
  int32_t strides[3] = {0, 0, 0};
  for (int32_t dim = 0; dim < pooling_dims; dim++) {
    auto k = 3 - pooling_dims + dim;
    bounds[k] = get_adaptive_bounds<idx_t>(
        pooling_dim_indices[dim],
        params.input_sizes[leading_dims + dim],
        params.output_sizes[leading_dims + dim]);
    strides[k] = params.input_strides[leading_dims + dim];
  }

  opmath_t<T> sum = 0;
  for (auto i0 = bounds[0].start; i0 < bounds[0].end; i0++) {
    for (auto i1 = bounds[1].start; i1 < bounds[1].end; i1++) {
      auto row = input + i0 * strides[0] + i1 * strides[1];
      for (auto i2 = bounds[2].start + static_cast<int32_t>(lane);
           i2 < bounds[2].end;
           i2 += simd_width) {
        sum += static_cast<opmath_t<T>>(row[i2 * strides[2]]);
      }
    }
  }
  sum = simd_sum(sum);
  if (lane == 0) {
    auto count = (bounds[0].end - bounds[0].start) *
        (bounds[1].end - bounds[1].start) * (bounds[2].end - bounds[2].start);
    output[offsets.output] = static_cast<T>(sum / count);
  }
}

// Divides every grad_output element by the size of its adaptive pooling window
// once, writing them in opmath_t into the contiguous scaled_grad, so that the
// gather below is a plain sum. Also fills out_ranges, which holds for every
// input index of every pooling dim the range of outputs whose windows cover it,
// as these are the same for all the planes.
template <typename T, typename idx_t>
kernel void adaptive_avg_pool_backward_scale(
    device const T* grad_output [[buffer(0)]],
    device opmath_t<T>* scaled_grad [[buffer(1)]],
    constant AvgPoolingParams<5>& params [[buffer(2)]],
    device int2* out_ranges [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  auto leading_dims = params.dims - params.pooling_dims;
  auto range_idx = static_cast<int32_t>(tid);
  for (auto dim = leading_dims; dim < params.dims; dim++) {
    auto in_size = params.input_sizes[dim];
    if (range_idx < in_size) {
      auto b = get_adaptive_bounds<idx_t>(
          range_idx, params.output_sizes[dim], in_size);
      out_ranges[tid] = int2(b.start, b.end);
      break;
    }
    range_idx -= in_size;
  }

  uint numel = 1;
  for (auto dim = 0; dim < params.dims; dim++) {
    numel *= params.output_sizes[dim];
  }
  if (tid >= numel) {
    return;
  }
  int32_t offset = 0, count = 1;
  auto idx = tid;
  for (auto dim = params.dims - 1; dim >= 0; dim--) {
    auto size = params.output_sizes[dim];
    auto i = static_cast<int32_t>(idx % size);
    idx /= size;
    offset += i * params.output_strides[dim];
    if (dim >= leading_dims) {
      auto w = get_adaptive_bounds<idx_t>(i, params.input_sizes[dim], size);
      count *= w.end - w.start;
    }
  }
  scaled_grad[tid] = static_cast<opmath_t<T>>(grad_output[offset]) / count;
}

// Computes one element of grad_input per thread by summing the scaled
// gradients of all outputs whose adaptive pooling window contains it. Output
// windows covering an input index follow the same formula as input windows of
// an output index, with input and output sizes swapped. Supports 2 or 3
// pooling dims, params.output_strides are the strides of scaled_grad.
template <typename T, typename idx_t>
kernel void adaptive_avg_pool_backward(
    device T* grad_input [[buffer(0)]],
    device const opmath_t<T>* scaled_grad [[buffer(1)]],
    constant AvgPoolingParams<5>& params [[buffer(2)]],
    device const int2* out_ranges [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  auto dims = params.dims;
  auto pooling_dims = params.pooling_dims;
  auto leading_dims = dims - pooling_dims;
  int32_t pooling_dim_indices[3];
  auto offsets = find_pool_offsets<int32_t, int32_t>(
      params.input_sizes.data(),
      params.input_strides.data(),
      /*indices_strides=*/nullptr,
      params.output_strides.data(),
      pooling_dim_indices,
      dims,
      leading_dims,
      /*return_indices=*/false,
      tid);
  scaled_grad += offsets.input_leading;

  IterBounds<int32_t> bounds[3] = {{0, 1}, {0, 1}, {0, 1}};
  int32_t strides[3] = {0, 0, 0};
  for (int32_t dim = 0; dim < pooling_dims; dim++) {
    auto range = out_ranges[pooling_dim_indices[dim]];
    bounds[dim] = {range.x, range.y};
    strides[dim] = params.output_strides[leading_dims + dim];
    out_ranges += params.input_sizes[leading_dims + dim];
  }

  opmath_t<T> grad_sum = 0;
  for (auto o0 = bounds[0].start; o0 < bounds[0].end; o0++) {
    for (auto o1 = bounds[1].start; o1 < bounds[1].end; o1++) {
      auto row = scaled_grad + o0 * strides[0] + o1 * strides[1];
      for (auto o2 = bounds[2].start; o2 < bounds[2].end; o2++) {
        grad_sum += row[o2 * strides[2]];
      }
    }
  }
  grad_input[offsets.output] = static_cast<T>(grad_sum);
}

template <typename T, bool serial>
kernel void avg_pool_backward(
    device AtomicType_t<T>* grad_input [[buffer(0)]],
    device const T* grad_output [[buffer(1)]],
    constant AvgPoolingParams<5>& params [[buffer(2)]],
    uint tid [[thread_position_in_grid]]) {
  if IF_CONSTEXPR (!serial) {
    avg_pool_backward_kernel_impl<T>(grad_input, grad_output, params, tid);
    return;
  }
  uint pool_size = 1;
  for (auto dim = params.dims - params.pooling_dims; dim < params.dims; dim++) {
    pool_size *= params.output_sizes[dim];
  }
  for (uint pool_idx = 0; pool_idx < pool_size; pool_idx++) {
    avg_pool_backward_kernel_impl<T>(
        grad_input, grad_output, params, tid * pool_size + pool_idx);
  }
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

#define REGISTER_MAX_POOL(DTYPE, IDX_T, SUFFIX)                    \
  template [[host_name("max_pool" SUFFIX "_" #DTYPE)]] kernel void \
  max_pool<DTYPE, IDX_T>(                                          \
      device const DTYPE* input [[buffer(0)]],                     \
      device DTYPE* output [[buffer(1)]],                          \
      device int64_t* indices [[buffer(2)]],                       \
      constant PoolingParams<5>& params [[buffer(3)]],             \
      uint tid [[thread_position_in_grid]]);

#define REGISTER_AVG_POOL(DTYPE, IDX_T, SUFFIX)                \
  template[[host_name("avg_pool_" #DTYPE SUFFIX)]] kernel void \
  avg_pool<DTYPE, IDX_T>(                                      \
      device const DTYPE* input [[buffer(0)]],                 \
      device DTYPE* output [[buffer(1)]],                      \
      constant AvgPoolingParams<5>& params [[buffer(2)]],      \
      uint tid [[thread_position_in_grid]]);

#define REGISTER_POOL_OP(DTYPE)                                               \
  REGISTER_MAX_POOL(DTYPE, int32_t, "")                                       \
  REGISTER_MAX_POOL(DTYPE, int64_t, "_i64")                                   \
  REGISTER_AVG_POOL(DTYPE, int32_t, "_i32")                                   \
                                                                              \
  template [[host_name("max_unpool_" #DTYPE)]] kernel void max_unpool<DTYPE>( \
      device DTYPE * output [[buffer(0)]],                                    \
      device const DTYPE* input [[buffer(1)]],                                \
      device const int64_t* indices [[buffer(2)]],                            \
      constant MaxUnpoolingParams<5>& params [[buffer(3)]],                   \
      device ::c10::metal::ErrorMessages* error_buffer [[buffer(4)]],         \
      uint tid [[thread_position_in_grid]]);                                  \
                                                                              \
  REGISTER_MAX_POOL2D_DENSE(DTYPE, false, false, "nchw")                      \
  REGISTER_MAX_POOL2D_DENSE(DTYPE, true, false, "nchw_idx")                   \
  REGISTER_MAX_POOL2D_DENSE(DTYPE, false, true, "nhwc")                       \
  REGISTER_MAX_POOL2D_DENSE(DTYPE, true, true, "nhwc_idx")                    \
  REGISTER_MAX_POOL2D_S1(DTYPE, 0)                                            \
  REGISTER_MAX_POOL2D_S1(DTYPE, 2)                                            \
  REGISTER_MAX_POOL2D_S1(DTYPE, 3)                                            \
  REGISTER_MAX_POOL2D_S1(DTYPE, 5)

#define REGISTER_POOL_BACKWARD_OP_SERIAL(DTYPE, SERIAL)                    \
  template[[host_name("max_pool_backward_" #DTYPE C10_METAL_SERIAL_SUFFIX( \
      SERIAL))]] kernel void                                               \
  max_pool_backward<DTYPE, SERIAL>(                                        \
      device AtomicType_t<DTYPE> * grad_input [[buffer(0)]],               \
      device const DTYPE* grad_output_ [[buffer(1)]],                      \
      device const int64_t* grad_indices_ [[buffer(2)]],                   \
      constant PoolingBackwardParams<5>& params [[buffer(3)]],             \
      uint tid [[thread_position_in_grid]]);                               \
                                                                           \
  template[[host_name("avg_pool_backward_" #DTYPE C10_METAL_SERIAL_SUFFIX( \
      SERIAL))]] kernel void                                               \
  avg_pool_backward<DTYPE, SERIAL>(                                        \
      device AtomicType_t<DTYPE> * grad_input [[buffer(0)]],               \
      device const DTYPE* grad_output [[buffer(1)]],                       \
      constant AvgPoolingParams<5>& params [[buffer(2)]],                  \
      uint tid [[thread_position_in_grid]]);

#define REGISTER_ADAPTIVE_AVG_POOL(DTYPE, IDX_T, SUFFIX)                     \
  template [[host_name("adaptive_avg_pool_simd_" #DTYPE #SUFFIX)]]           \
  kernel void adaptive_avg_pool_simd<DTYPE, IDX_T>(                          \
      device const DTYPE* input [[buffer(0)]],                               \
      device DTYPE* output [[buffer(1)]],                                    \
      constant AvgPoolingParams<5>& params [[buffer(2)]],                    \
      uint tid [[thread_position_in_grid]],                                  \
      uint lane [[thread_index_in_simdgroup]],                               \
      uint simd_width [[threads_per_simdgroup]]);                            \
  template [[host_name("adaptive_avg_pool_backward_scale_" #DTYPE #SUFFIX)]] \
  kernel void adaptive_avg_pool_backward_scale<DTYPE, IDX_T>(                \
      device const DTYPE* grad_output [[buffer(0)]],                         \
      device opmath_t<DTYPE>* scaled_grad [[buffer(1)]],                     \
      constant AvgPoolingParams<5>& params [[buffer(2)]],                    \
      device int2* out_ranges [[buffer(3)]],                                 \
      uint tid [[thread_position_in_grid]]);                                 \
  template [[host_name("adaptive_avg_pool_backward_" #DTYPE #SUFFIX)]]       \
  kernel void adaptive_avg_pool_backward<DTYPE, IDX_T>(                      \
      device DTYPE * grad_input [[buffer(0)]],                               \
      device const opmath_t<DTYPE>* scaled_grad [[buffer(1)]],               \
      constant AvgPoolingParams<5>& params [[buffer(2)]],                    \
      device const int2* out_ranges [[buffer(3)]],                           \
      uint tid [[thread_position_in_grid]])

// Adaptive avg pooling supports floating types only, and is the only user of
// the 64-bit avg_pool variant
#define REGISTER_ADAPTIVE_AVG_POOL_OP(DTYPE)        \
  REGISTER_AVG_POOL(DTYPE, int64_t, "_i64")         \
  REGISTER_ADAPTIVE_AVG_POOL(DTYPE, int32_t, _i32); \
  REGISTER_ADAPTIVE_AVG_POOL(DTYPE, int64_t, _i64)

REGISTER_POOL_OP(float);
REGISTER_POOL_OP(half);
REGISTER_POOL_OP(bfloat);
REGISTER_POOL_OP(int);
REGISTER_POOL_OP(long);
REGISTER_POOL_OP(short);
REGISTER_POOL_OP(char);
REGISTER_POOL_OP(uchar);
REGISTER_POOL_OP(bool);

REGISTER_POOL_BACKWARD_OP_SERIAL(float, false);
REGISTER_POOL_BACKWARD_OP_SERIAL(half, false);
REGISTER_POOL_BACKWARD_OP_SERIAL(bfloat, false);
REGISTER_POOL_BACKWARD_OP_SERIAL(float, true);

REGISTER_ADAPTIVE_AVG_POOL_OP(float);
REGISTER_ADAPTIVE_AVG_POOL_OP(half);
REGISTER_ADAPTIVE_AVG_POOL_OP(bfloat);
