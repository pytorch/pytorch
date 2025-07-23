// Metal kernel for 3D grid sampling on MPS
// Implements nearest neighbor interpolation

#include <metal_stdlib>
using namespace metal;

// Helper function to unnormalize coordinates from [-1, 1] to [0, size-1]
template<typename T>
T grid_sampler_unnormalize(T coord, long size, bool align_corners) {
  if (align_corners) {
    // maps from [-1, 1] to [0, size-1]
    return ((coord + T(1.0)) / T(2.0)) * T(size - 1);
  } else {
    // maps from [-1, 1] to [-0.5, size-0.5]
    return ((coord + T(1.0)) * T(size) - T(1.0)) / T(2.0);
  }
}

// Helper function to check if coordinates are within bounds
bool within_bounds_3d(long z, long y, long x, long D, long H, long W) {
  return z >= 0 && z < D && y >= 0 && y < H && x >= 0 && x < W;
}

// Helper function to clip coordinates to image borders
template<typename T>
T clip_coordinates(T coord, long size) {
  return min(static_cast<T>(size - 1), max(coord, static_cast<T>(0)));
}

// Helper function to reflect coordinates by image borders
template<typename T>
T reflect_coordinates(T coord, long twice_low, long twice_high) {
  if (twice_low == twice_high) {
    return static_cast<T>(0);
  }
  T min_val = static_cast<T>(twice_low) / T(2.0);
  T span = static_cast<T>(twice_high - twice_low) / T(2.0);
  coord = abs(coord - min_val);

  // Equivalent to std::fmod(coord, span)
  T extra = coord - span * floor(coord / span);
  long flips = static_cast<long>(floor(coord / span));

  if (flips % 2 == 0) {
    return extra + min_val;
  } else {
    return span - extra + min_val;
  }
}

// Compute coordinates with padding mode applied
template<typename T>
T compute_coordinates(T coord, long size, int padding_mode, bool align_corners) {
  if (padding_mode == 1) { // Border padding
    coord = clip_coordinates(coord, size);
  } else if (padding_mode == 2) { // Reflection padding
    if (align_corners) {
      coord = reflect_coordinates(coord, 0, 2 * (size - 1));
    } else {
      coord = reflect_coordinates(coord, -1, 2 * size - 1);
    }
    // Also clip after reflection
    coord = clip_coordinates(coord, size);
  }
  // For zeros padding (mode 0), we don't modify coordinates here
  return coord;
}

template <typename T>
kernel void grid_sampler_3d(
    constant T* inputData [[buffer(0)]],
    device T* outputData [[buffer(1)]],
    constant T* gridData [[buffer(2)]],
    constant int& interpolation_mode [[buffer(3)]],
    constant int& padding_mode [[buffer(4)]],
    constant bool& align_corners [[buffer(5)]],
    constant ulong* input_sizes [[buffer(6)]],     // [N, C, D, H, W] - all 5 dimensions
    constant ulong* output_sizes [[buffer(7)]],    // [N, C, D_out, H_out, W_out] - all 5 dimensions
    constant ulong* input_strides [[buffer(8)]],   // all 5 input strides [N, C, D, H, W]
    constant ulong* output_strides [[buffer(9)]],  // all 5 output strides [N, C, D, H, W]
    constant ulong* grid_strides [[buffer(10)]],   // all 5 grid strides [N, D, H, W, coord]
    uint3 thread_index [[thread_position_in_grid]]) {
  // thread_index is (out_W, out_H*out_D, N*C)
  const auto linear_idx = thread_index.z;
  const auto N = linear_idx / input_sizes[1]; // batch index (divide by C)
  const auto C = linear_idx % input_sizes[1]; // channel index

  const auto out_d_h = thread_index.y;
  const auto out_d = out_d_h / output_sizes[3]; // output depth index (divide by H_out)
  const auto out_h = out_d_h % output_sizes[3]; // output height index
  const auto out_w = thread_index.x;            // output width index

  // Check bounds
  if (N >= input_sizes[0] || C >= input_sizes[1] ||
      out_d >= output_sizes[2] || out_h >= output_sizes[3] || out_w >= output_sizes[4]) {
    return;
  }

  // Get input tensor dimensions: input is [N, C, D, H, W]
  const auto in_D = input_sizes[2];
  const auto in_H = input_sizes[3];
  const auto in_W = input_sizes[4];

  // Calculate grid offset for this output position
  // Grid shape: [N, D_out, H_out, W_out, 3]
  const auto grid_offset = N * grid_strides[0] +
                          out_d * grid_strides[1] +
                          out_h * grid_strides[2] +
                          out_w * grid_strides[3];

  // Read grid coordinates (x, y, z) - normalized to [-1, 1]
  // grid_strides[4] is the stride for the coordinate dimension
  const T grid_x = gridData[grid_offset];                      // x coordinate
  const T grid_y = gridData[grid_offset + grid_strides[4]];    // y coordinate
  const T grid_z = gridData[grid_offset + 2 * grid_strides[4]]; // z coordinate

  // Calculate output offset
  // Output shape: [N, C, D_out, H_out, W_out]
  const auto output_offset = N * output_strides[0] +
                            C * output_strides[1] +
                            out_d * output_strides[2] +
                            out_h * output_strides[3] +
                            out_w * output_strides[4];

  if (interpolation_mode == 0) { // bilinear (trilinear in 3D)
    // Unnormalize coordinates to input tensor space
    T ix = grid_sampler_unnormalize(grid_x, in_W, align_corners);
    T iy = grid_sampler_unnormalize(grid_y, in_H, align_corners);
    T iz = grid_sampler_unnormalize(grid_z, in_D, align_corners);

    // Apply padding mode transformations
    ix = compute_coordinates(ix, in_W, padding_mode, align_corners);
    iy = compute_coordinates(iy, in_H, padding_mode, align_corners);
    iz = compute_coordinates(iz, in_D, padding_mode, align_corners);

    // Get corner pixel values from (ix, iy, iz)
    // For 3D, we use top-bottom + north-east-south-west naming
    long ix_tnw = static_cast<long>(floor(ix));
    long iy_tnw = static_cast<long>(floor(iy));
    long iz_tnw = static_cast<long>(floor(iz));

    // Calculate all 8 corner coordinates
    long ix_tne = ix_tnw + 1; long iy_tne = iy_tnw;     long iz_tne = iz_tnw;
    long ix_tsw = ix_tnw;     long iy_tsw = iy_tnw + 1; long iz_tsw = iz_tnw;
    long ix_tse = ix_tnw + 1; long iy_tse = iy_tnw + 1; long iz_tse = iz_tnw;
    long ix_bnw = ix_tnw;     long iy_bnw = iy_tnw;     long iz_bnw = iz_tnw + 1;
    long ix_bne = ix_tnw + 1; long iy_bne = iy_tnw;     long iz_bne = iz_tnw + 1;
    long ix_bsw = ix_tnw;     long iy_bsw = iy_tnw + 1; long iz_bsw = iz_tnw + 1;
    long ix_bse = ix_tnw + 1; long iy_bse = iy_tnw + 1; long iz_bse = iz_tnw + 1;

    // Calculate trilinear interpolation weights (volumes to each neighbor)
    T tnw = (T(ix_bse) - ix) * (T(iy_bse) - iy) * (T(iz_bse) - iz);
    T tne = (ix - T(ix_bsw)) * (T(iy_bsw) - iy) * (T(iz_bsw) - iz);
    T tsw = (T(ix_bne) - ix) * (iy - T(iy_bne)) * (T(iz_bne) - iz);
    T tse = (ix - T(ix_bnw)) * (iy - T(iy_bnw)) * (T(iz_bnw) - iz);
    T bnw = (T(ix_tse) - ix) * (T(iy_tse) - iy) * (iz - T(iz_tse));
    T bne = (ix - T(ix_tsw)) * (T(iy_tsw) - iy) * (iz - T(iz_tsw));
    T bsw = (T(ix_tne) - ix) * (iy - T(iy_tne)) * (iz - T(iz_tne));
    T bse = (ix - T(ix_tnw)) * (iy - T(iy_tnw)) * (iz - T(iz_tnw));

    // Sample from the 8 corners and accumulate weighted result
    T result = T(0);

    // Accumulate weighted contributions from all 8 corners
    // tnw corner
    if (padding_mode == 0) { // zeros padding - check bounds
      if (within_bounds_3d(iz_tnw, iy_tnw, ix_tnw, in_D, in_H, in_W)) {
        const auto input_offset = N * input_strides[0] + C * input_strides[1] +
                                 iz_tnw * input_strides[2] + iy_tnw * input_strides[3] + ix_tnw * input_strides[4];
        result += inputData[input_offset] * tnw;
      }
    } else { // border/reflection padding - coordinates should be in bounds
      const auto input_offset = N * input_strides[0] + C * input_strides[1] +
                               iz_tnw * input_strides[2] + iy_tnw * input_strides[3] + ix_tnw * input_strides[4];
      result += inputData[input_offset] * tnw;
    }

    // tne corner
    if (padding_mode == 0) {
      if (within_bounds_3d(iz_tne, iy_tne, ix_tne, in_D, in_H, in_W)) {
        const auto input_offset = N * input_strides[0] + C * input_strides[1] +
                                 iz_tne * input_strides[2] + iy_tne * input_strides[3] + ix_tne * input_strides[4];
        result += inputData[input_offset] * tne;
      }
    } else {
      const auto input_offset = N * input_strides[0] + C * input_strides[1] +
                               iz_tne * input_strides[2] + iy_tne * input_strides[3] + ix_tne * input_strides[4];
      result += inputData[input_offset] * tne;
    }

    // tsw corner
    if (padding_mode == 0) {
      if (within_bounds_3d(iz_tsw, iy_tsw, ix_tsw, in_D, in_H, in_W)) {
        const auto input_offset = N * input_strides[0] + C * input_strides[1] +
                                 iz_tsw * input_strides[2] + iy_tsw * input_strides[3] + ix_tsw * input_strides[4];
        result += inputData[input_offset] * tsw;
      }
    } else {
      const auto input_offset = N * input_strides[0] + C * input_strides[1] +
                               iz_tsw * input_strides[2] + iy_tsw * input_strides[3] + ix_tsw * input_strides[4];
      result += inputData[input_offset] * tsw;
    }

    // tse corner
    if (padding_mode == 0) {
      if (within_bounds_3d(iz_tse, iy_tse, ix_tse, in_D, in_H, in_W)) {
        const auto input_offset = N * input_strides[0] + C * input_strides[1] +
                                 iz_tse * input_strides[2] + iy_tse * input_strides[3] + ix_tse * input_strides[4];
        result += inputData[input_offset] * tse;
      }
    } else {
      const auto input_offset = N * input_strides[0] + C * input_strides[1] +
                               iz_tse * input_strides[2] + iy_tse * input_strides[3] + ix_tse * input_strides[4];
      result += inputData[input_offset] * tse;
    }

    // bnw corner
    if (padding_mode == 0) {
      if (within_bounds_3d(iz_bnw, iy_bnw, ix_bnw, in_D, in_H, in_W)) {
        const auto input_offset = N * input_strides[0] + C * input_strides[1] +
                                 iz_bnw * input_strides[2] + iy_bnw * input_strides[3] + ix_bnw * input_strides[4];
        result += inputData[input_offset] * bnw;
      }
    } else {
      const auto input_offset = N * input_strides[0] + C * input_strides[1] +
                               iz_bnw * input_strides[2] + iy_bnw * input_strides[3] + ix_bnw * input_strides[4];
      result += inputData[input_offset] * bnw;
    }

    // bne corner
    if (padding_mode == 0) {
      if (within_bounds_3d(iz_bne, iy_bne, ix_bne, in_D, in_H, in_W)) {
        const auto input_offset = N * input_strides[0] + C * input_strides[1] +
                                 iz_bne * input_strides[2] + iy_bne * input_strides[3] + ix_bne * input_strides[4];
        result += inputData[input_offset] * bne;
      }
    } else {
      const auto input_offset = N * input_strides[0] + C * input_strides[1] +
                               iz_bne * input_strides[2] + iy_bne * input_strides[3] + ix_bne * input_strides[4];
      result += inputData[input_offset] * bne;
    }

    // bsw corner
    if (padding_mode == 0) {
      if (within_bounds_3d(iz_bsw, iy_bsw, ix_bsw, in_D, in_H, in_W)) {
        const auto input_offset = N * input_strides[0] + C * input_strides[1] +
                                 iz_bsw * input_strides[2] + iy_bsw * input_strides[3] + ix_bsw * input_strides[4];
        result += inputData[input_offset] * bsw;
      }
    } else {
      const auto input_offset = N * input_strides[0] + C * input_strides[1] +
                               iz_bsw * input_strides[2] + iy_bsw * input_strides[3] + ix_bsw * input_strides[4];
      result += inputData[input_offset] * bsw;
    }

    // bse corner
    if (padding_mode == 0) {
      if (within_bounds_3d(iz_bse, iy_bse, ix_bse, in_D, in_H, in_W)) {
        const auto input_offset = N * input_strides[0] + C * input_strides[1] +
                                 iz_bse * input_strides[2] + iy_bse * input_strides[3] + ix_bse * input_strides[4];
        result += inputData[input_offset] * bse;
      }
    } else {
      const auto input_offset = N * input_strides[0] + C * input_strides[1] +
                               iz_bse * input_strides[2] + iy_bse * input_strides[3] + ix_bse * input_strides[4];
      result += inputData[input_offset] * bse;
    }

    outputData[output_offset] = result;

  } else if (interpolation_mode == 1) { // nearest neighbor
    // Unnormalize coordinates to input tensor space
    T ix = grid_sampler_unnormalize(grid_x, in_W, align_corners);
    T iy = grid_sampler_unnormalize(grid_y, in_H, align_corners);
    T iz = grid_sampler_unnormalize(grid_z, in_D, align_corners);

    // Apply padding mode transformations
    ix = compute_coordinates(ix, in_W, padding_mode, align_corners);
    iy = compute_coordinates(iy, in_H, padding_mode, align_corners);
    iz = compute_coordinates(iz, in_D, padding_mode, align_corners);

    // Round to nearest integer coordinates
    const long ix_nearest = static_cast<long>(rint(ix));
    const long iy_nearest = static_cast<long>(rint(iy));
    const long iz_nearest = static_cast<long>(rint(iz));

    // For border and reflection padding, coordinates should always be in bounds after transformation
    // For zeros padding, we still need to check bounds
    if (padding_mode == 0) { // zeros padding
      if (within_bounds_3d(iz_nearest, iy_nearest, ix_nearest, in_D, in_H, in_W)) {
        // Calculate input offset and sample the value
        // Input shape: [N, C, D, H, W]
        const auto input_offset = N * input_strides[0] +
                                 C * input_strides[1] +
                                 iz_nearest * input_strides[2] +
                                 iy_nearest * input_strides[3] +
                                 ix_nearest * input_strides[4];

        outputData[output_offset] = inputData[input_offset];
      } else {
        // Out of bounds - set to zero for zeros padding
        outputData[output_offset] = T(0);
      }
    } else { // border or reflection padding - coordinates should be in bounds
      // Calculate input offset and sample the value
      // Input shape: [N, C, D, H, W]
      const auto input_offset = N * input_strides[0] +
                               C * input_strides[1] +
                               iz_nearest * input_strides[2] +
                               iy_nearest * input_strides[3] +
                               ix_nearest * input_strides[4];

      outputData[output_offset] = inputData[input_offset];
    }
  } else {
    // For unsupported interpolation modes (e.g., bicubic), set to zero
    outputData[output_offset] = T(0);
  }
}

#define INSTANTIATE_GRID_SAMPLER_3D(DTYPE)                                         \
  template [[host_name("grid_sampler_3d_" #DTYPE)]] kernel void grid_sampler_3d<DTYPE>( \
      constant DTYPE * inputData [[buffer(0)]],                                     \
      device DTYPE * outputData [[buffer(1)]],                                      \
      constant DTYPE * gridData [[buffer(2)]],                                      \
      constant int & interpolation_mode [[buffer(3)]],                              \
      constant int & padding_mode [[buffer(4)]],                                    \
      constant bool & align_corners [[buffer(5)]],                                  \
      constant ulong * input_sizes [[buffer(6)]],                                   \
      constant ulong * output_sizes [[buffer(7)]],                                  \
      constant ulong * input_strides [[buffer(8)]],                                \
      constant ulong * output_strides [[buffer(9)]],                               \
      constant ulong * grid_strides [[buffer(10)]],                                \
      uint3 thread_index [[thread_position_in_grid]])

INSTANTIATE_GRID_SAMPLER_3D(float);
INSTANTIATE_GRID_SAMPLER_3D(half);
#if __METAL_VERSION__ >= 310
// Note: bfloat support may need special handling for type conversions
// INSTANTIATE_GRID_SAMPLER_3D(bfloat);
#endif