#include <ATen/native/mps/kernels/GridSampler.h>
#include <c10/metal/utils.h>
#include <metal_array>
#include <metal_stdlib>

using namespace metal;
using namespace c10::metal;

struct GridSamplerOffsets {
  int32_t output;
  int32_t input;
  int32_t grid;

  GridSamplerOffsets() : output(0), input(0), grid(0) {}
};

// Find offsets into the tensors that this thread will operate on,
// based on the thread ID.
static GridSamplerOffsets find_grid_sampler_offsets(
    constant int32_t* output_sizes,
    constant int32_t* output_strides,
    constant int32_t* input_strides,
    constant int32_t* grid_strides,
    int32_t sampler_dims,
    uint tid) {
  auto dims = sampler_dims + 2;
  auto output_idx = static_cast<int32_t>(tid);
  GridSamplerOffsets offsets;

  for (auto dim = dims - 1; dim >= 0; dim--) {
    auto dim_idx = output_idx % output_sizes[dim];
    output_idx = output_idx / output_sizes[dim];

    // Select the output element that this thread will calculate.
    // output shape:
    //   2 sampler dims: (N, C, Hout, Wout)
    //   3 sampler dims: (N, C, Dout, Hout, Wout)
    offsets.output += output_strides[dim] * dim_idx;

    // Select the batch and channel for the input.
    // input shape:
    //   2 sampler dims: (N, C, Hin, Win)
    //   3 sampler dims: (N, C, Din, Hin, Win)
    if (dim < 2) {
      offsets.input += input_strides[dim] * dim_idx;
    }

    // Select the grid coordinates for the output element.
    // grid shape:
    //   2 sampler dims: (N, Hout, Wout, 2)
    //   3 sampler dims: (N, Dout, Hout, Wout, 3)
    if (dim == 0) {
      offsets.grid += grid_strides[dim] * dim_idx;
    } else if (dim >= 2) {
      offsets.grid += grid_strides[dim - 1] * dim_idx;
    }
  }

  return offsets;
}

// Mod function which gives positive output when `a` is negative
static int32_t mod(int32_t a, int32_t b) {
  auto r = a % b;
  return r + (r < 0 ? b : 0);
}

// Sentinel index value to indicate zero padding
constant int32_t IDX_ZERO = -1;

// Apply padding to an index into the input
static int32_t pad_input_index(
    int32_t idx,
    int32_t input_size,
    GridSamplerPadding padding_mode,
    bool align_corners) {
  int32_t idx_padded = idx;

  if (padding_mode == GridSamplerPadding::Zeros) {
    idx_padded = (idx < 0) ? IDX_ZERO : idx_padded;
    idx_padded = (idx >= input_size) ? IDX_ZERO : idx_padded;

  } else if (padding_mode == GridSamplerPadding::Border) {
    idx_padded = (idx < 0) ? 0 : idx_padded;
    idx_padded = (idx >= input_size) ? input_size - 1 : idx_padded;

  } else if (padding_mode == GridSamplerPadding::Reflection) {
    auto scale_length = align_corners ? (input_size - 1) : input_size;
    auto idx_mod = mod(idx, scale_length);
    auto idx_mod_reverse = (input_size - 1) - idx_mod;
    bool is_reverse = (abs(idx - idx_mod) / scale_length) % 2 == 1;
    idx_padded = is_reverse ? idx_mod_reverse : idx_mod;
  }
  return idx_padded;
}

template <int32_t dims, typename T>
T get_tensor_val(
    constant T* input,
    constant int32_t* input_strides,
    int32_t indices[dims]) {
  bool found_idx_zero = false;
  int32_t offset = 0;

  for (auto dim = 0; dim < dims; dim++) {
    auto idx = indices[dim];
    found_idx_zero = found_idx_zero || (idx == IDX_ZERO);
    offset += (found_idx_zero ? 0 : idx) * input_strides[dim];
  }

  return found_idx_zero ? 0 : input[offset];
}

// This function performs 3D linear interpolation for one value. One way to
// think of how this works is to imagine a unit cube where each corner of the
// cube has one scalar value associated with it. Inside the cube, the values
// change linearly, so the gradient is constant. The values associated with each
// corner are given by the `input`, indexed at all eight different combinations
// of the `left_indices` and `right_indices`. Given a 3D coordinate anywhere
// within the cube, specified by the `scales` argument, we must calculate the
// value associated with that position.
template <typename T>
T interpolate_linear_3d(
    constant T* input,
    constant int32_t* input_strides,
    int32_t left_indices[3],
    int32_t right_indices[3],
    opmath_t<T> scales[3]) {
  int32_t a_idx[3] = {left_indices[0], left_indices[1], left_indices[2]};
  int32_t b_idx[3] = {left_indices[0], left_indices[1], right_indices[2]};
  int32_t c_idx[3] = {left_indices[0], right_indices[1], left_indices[2]};
  int32_t d_idx[3] = {left_indices[0], right_indices[1], right_indices[2]};
  int32_t e_idx[3] = {right_indices[0], left_indices[1], left_indices[2]};
  int32_t f_idx[3] = {right_indices[0], left_indices[1], right_indices[2]};
  int32_t g_idx[3] = {right_indices[0], right_indices[1], left_indices[2]};
  int32_t h_idx[3] = {right_indices[0], right_indices[1], right_indices[2]};
  auto a =
      static_cast<opmath_t<T>>(get_tensor_val<3>(input, input_strides, a_idx));
  auto b =
      static_cast<opmath_t<T>>(get_tensor_val<3>(input, input_strides, b_idx));
  auto c =
      static_cast<opmath_t<T>>(get_tensor_val<3>(input, input_strides, c_idx));
  auto d =
      static_cast<opmath_t<T>>(get_tensor_val<3>(input, input_strides, d_idx));
  auto e =
      static_cast<opmath_t<T>>(get_tensor_val<3>(input, input_strides, e_idx));
  auto f =
      static_cast<opmath_t<T>>(get_tensor_val<3>(input, input_strides, f_idx));
  auto g =
      static_cast<opmath_t<T>>(get_tensor_val<3>(input, input_strides, g_idx));
  auto h =
      static_cast<opmath_t<T>>(get_tensor_val<3>(input, input_strides, h_idx));

  auto scale0_right = scales[0];
  auto scale1_right = scales[1];
  auto scale2_right = scales[2];
  auto scale0_left = 1 - scale0_right;
  auto scale1_left = 1 - scale1_right;
  auto scale2_left = 1 - scale2_right;

  return static_cast<T>(
      scale0_left * scale1_left * scale2_left * a +
      scale0_left * scale1_left * scale2_right * b +
      scale0_left * scale1_right * scale2_left * c +
      scale0_left * scale1_right * scale2_right * d +
      scale0_right * scale1_left * scale2_left * e +
      scale0_right * scale1_left * scale2_right * f +
      scale0_right * scale1_right * scale2_left * g +
      scale0_right * scale1_right * scale2_right * h);
}

// Calculates a single output element.
// `input` shape:
//    2 sampler dims: (Hin, Win)
//    3 sampler dims: (Din, Hin, Win)
// `coords` values:
//    2 sampler dims: (Wcoord, Hcoord)
//    3 sampler dims: (Wcoord, Hcoord, Dcoord)
template <typename T>
void grid_sampler_single_element(
    device T* output,
    constant T* input,
    constant T* coords,
    int32_t dims,
    constant int32_t* input_sizes,
    constant int32_t* input_strides,
    int32_t coord_stride,
    GridSamplerInterpolation interpolation_mode,
    GridSamplerPadding padding_mode,
    bool align_corners) {
  int32_t left_indices[3];
  int32_t right_indices[3];
  opmath_t<T> scales[3];

  // For each dimension, find the pair of indices in the corresponding dimension
  // of `input` which surround the grid coordinate in that dimension. We'll do
  // this by mapping different coordinate spaces onto each other. There are
  // basically three different coordinate spaces to keep in mind:
  //
  //  * aligned grid space
  //    - `-1` refers to the leftmost input value.
  //    - `1` refers to the rightmost input value.
  //
  //  * unaligned grid space
  //    - `-1` refers to the midpoint between the leftmost input value and
  //      a padding value to the left of that.
  //    - `1` refers to the midpoint between the rightmost input value and
  //      a padding value to the right of that.
  //
  //  * input index space
  //    - `n` refers to the n-th value of the input.
  //    - `0` refers to the leftmost input value.
  //    - `N-1` refers to the rightmost input value.
  //
  // If `align_corners == False`, then the coordinates are is in unaligned grid
  // space, and we will map it onto aligned grid space. If `align_corners ==
  // True`, then coordinates are already in aligned grid space.
  //
  // Then we will map unaligned grid space onto input index space, making it
  // relatively simple to find the two input indices that surround the
  // coordinate.
  for (auto coord_dim = 0; coord_dim < dims; coord_dim++) {
    auto input_dim = dims - coord_dim - 1;
    auto input_size = input_sizes[input_dim];
    auto coord = static_cast<opmath_t<T>>(coords[coord_dim * coord_stride]);

    if (!align_corners) {
      // Map unaligned grid space to aligned grid space
      auto corner_alignment_factor = static_cast<opmath_t<T>>(input_size) /
          static_cast<opmath_t<T>>(input_size - 1);
      coord = coord * corner_alignment_factor;
    }

    // Map aligned grid space to input index space
    coord = (coord + 1) * (static_cast<opmath_t<T>>(input_size - 1) / 2);

    // Get the input indices surrounding the coordinate, apply padding to them,
    // and obtain the scaling factor between the two for interpolation.
    auto left_idx = static_cast<int32_t>(floor(coord));
    auto right_idx = static_cast<int32_t>(ceil(coord));
    left_indices[input_dim] =
        pad_input_index(left_idx, input_size, padding_mode, align_corners);
    right_indices[input_dim] =
        pad_input_index(right_idx, input_size, padding_mode, align_corners);

    auto scale = coord - left_idx;

    if (interpolation_mode == GridSamplerInterpolation::Nearest) {
      // TODO: For some reason, rounding the scale to 0 or 1 and then using
      // linear interpolation seems to work perfectly with zero padding mode,
      // but we get flaky failures with border and reflection padding modes.
      // Need to investigate and fix it.
      scale = (scale <= 0.5) ? 0 : 1;
    }
    scales[input_dim] = scale;
  }

  // Now that we have the bounding indices and scale factor for each dimension
  // of the input, we can interpolate.
  if (dims == 3) {
    *output = interpolate_linear_3d(
        input, input_strides, left_indices, right_indices, scales);
  }
}

template <typename T>
kernel void grid_sampler(
    device T* output [[buffer(0)]],
    constant T* input [[buffer(1)]],
    constant T* grid [[buffer(2)]],
    constant GridSamplerParams<5>& params [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  auto output_sizes = params.output_sizes.data();
  auto output_strides = params.output_strides.data();
  auto input_sizes = params.input_sizes.data();
  auto input_strides = params.input_strides.data();
  auto grid_strides = params.grid_strides.data();
  auto sampler_dims = params.sampler_dims;

  auto offsets = find_grid_sampler_offsets(
      output_sizes,
      output_strides,
      input_strides,
      grid_strides,
      sampler_dims,
      tid);

  output += offsets.output;
  input += offsets.input;
  auto coords = grid + offsets.grid;

  input_sizes += 2;
  input_strides += 2;
  auto coord_stride = grid_strides[sampler_dims + 1];

  auto interpolation_mode = params.interpolation_mode;
  auto padding_mode = params.padding_mode;
  auto align_corners = params.align_corners;

  grid_sampler_single_element(
      output,
      input,
      coords,
      sampler_dims,
      input_sizes,
      input_strides,
      coord_stride,
      interpolation_mode,
      padding_mode,
      align_corners);
}

#define REGISTER_GRID_SAMPLER_OP(DTYPE)                     \
  template [[host_name("grid_sampler_" #DTYPE)]]            \
  kernel void grid_sampler<DTYPE>(                          \
      device DTYPE * output [[buffer(0)]],                  \
      constant DTYPE * input [[buffer(1)]],                 \
      constant DTYPE * grid [[buffer(2)]],                  \
      constant GridSamplerParams<5> & params [[buffer(3)]], \
      uint tid [[thread_position_in_grid]]);

REGISTER_GRID_SAMPLER_OP(float);
REGISTER_GRID_SAMPLER_OP(half);
REGISTER_GRID_SAMPLER_OP(bfloat);
