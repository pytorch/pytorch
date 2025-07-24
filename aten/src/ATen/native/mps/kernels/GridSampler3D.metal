// Metal kernel for 3D grid sampling on MPS

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
  return coord < 0 ? 0 : coord > size - 1 ? size - 1 : coord;
}

// Helper function to reflect coordinates by image borders
template<typename T>
T reflect_coordinates(T coord, long twice_low, long twice_high) {
  if (twice_low == twice_high) {
    return static_cast<T>(0);
  }
  T min_val = static_cast<T>(twice_low) / T(2.0);
  T span = static_cast<T>(twice_high - twice_low) / T(2.0);
  coord = coord - min_val;
  coord = coord < 0 ? -coord : coord;

  // Equivalent to std::fmod(coord, span)
  T extra = coord - span * static_cast<T>(floor(coord / span));
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
  const auto out_w = thread_index.x;
  const auto out_d_h_combined = thread_index.y;
  const auto n_c_combined = thread_index.z;

  // Extract individual indices
  const auto N = n_c_combined / input_sizes[1];
  const auto C = n_c_combined % input_sizes[1];
  const auto out_d = out_d_h_combined / output_sizes[3];
  const auto out_h = out_d_h_combined % output_sizes[3];

  // Early bounds check
  if (N >= input_sizes[0] || C >= input_sizes[1] ||
      out_d >= output_sizes[2] || out_h >= output_sizes[3] || out_w >= output_sizes[4]) {
    return;
  }

  // Cache frequently used values
  const auto in_D = input_sizes[2];
  const auto in_H = input_sizes[3];
  const auto in_W = input_sizes[4];

  // Pre-calculate grid offset
  const auto grid_offset = N * grid_strides[0] +
                           out_d * grid_strides[1] +
                           out_h * grid_strides[2] +
                           out_w * grid_strides[3];

  // Read all grid coordinates at once for better memory access
  const T grid_x = gridData[grid_offset];
  const T grid_y = gridData[grid_offset + grid_strides[4]];
  const T grid_z = gridData[grid_offset + 2 * grid_strides[4]];

  // Pre-calculate output offset
  const auto output_offset = N * output_strides[0] +
                             C * output_strides[1] +
                             out_d * output_strides[2] +
                             out_h * output_strides[3] +
                             out_w * output_strides[4];

  if (interpolation_mode == 0) { // trilinear interpolation
    // Unnormalize coordinates to input tensor space
    T ix = grid_sampler_unnormalize(grid_x, in_W, align_corners);
    T iy = grid_sampler_unnormalize(grid_y, in_H, align_corners);
    T iz = grid_sampler_unnormalize(grid_z, in_D, align_corners);

    // Apply padding mode transformations
    ix = compute_coordinates(ix, in_W, padding_mode, align_corners);
    iy = compute_coordinates(iy, in_H, padding_mode, align_corners);
    iz = compute_coordinates(iz, in_D, padding_mode, align_corners);

    // Get floor coordinates for all 8 corners
    const long ix_tnw = static_cast<long>(floor(ix));
    const long iy_tnw = static_cast<long>(floor(iy));
    const long iz_tnw = static_cast<long>(floor(iz));

    // Pre-calculate all 8 corner coordinates and weights
    const long corners_x[8] = {ix_tnw, ix_tnw + 1, ix_tnw, ix_tnw + 1, ix_tnw, ix_tnw + 1, ix_tnw, ix_tnw + 1};
    const long corners_y[8] = {iy_tnw, iy_tnw, iy_tnw + 1, iy_tnw + 1, iy_tnw, iy_tnw, iy_tnw + 1, iy_tnw + 1};
    const long corners_z[8] = {iz_tnw, iz_tnw, iz_tnw, iz_tnw, iz_tnw + 1, iz_tnw + 1, iz_tnw + 1, iz_tnw + 1};

    // Pre-calculate interpolation weights
    const T x_diff = ix - T(ix_tnw);
    const T y_diff = iy - T(iy_tnw);
    const T z_diff = iz - T(iz_tnw);

    const T x_weights[2] = {T(1.0) - x_diff, x_diff};
    const T y_weights[2] = {T(1.0) - y_diff, y_diff};
    const T z_weights[2] = {T(1.0) - z_diff, z_diff};

    // Trilinear weights for 8 corners
    const T weights[8] = {
      x_weights[0] * y_weights[0] * z_weights[0], // tnw
      x_weights[1] * y_weights[0] * z_weights[0], // tne
      x_weights[0] * y_weights[1] * z_weights[0], // tsw
      x_weights[1] * y_weights[1] * z_weights[0], // tse
      x_weights[0] * y_weights[0] * z_weights[1], // bnw
      x_weights[1] * y_weights[0] * z_weights[1], // bne
      x_weights[0] * y_weights[1] * z_weights[1], // bsw
      x_weights[1] * y_weights[1] * z_weights[1]  // bse
    };

    T result = T(0);

    if (padding_mode == 0) { // zeros padding - need bounds checking
      for (int corner = 0; corner < 8; corner++) {
        if (within_bounds_3d(corners_z[corner], corners_y[corner], corners_x[corner], in_D, in_H, in_W)) {
          const auto input_offset = N * input_strides[0] + C * input_strides[1] +
                                    corners_z[corner] * input_strides[2] +
                                    corners_y[corner] * input_strides[3] +
                                    corners_x[corner] * input_strides[4];
          result += inputData[input_offset] * weights[corner];
        }
      }
    } else { // border/reflection padding - coordinates should be in bounds
      for (int corner = 0; corner < 8; corner++) {
        const auto input_offset = N * input_strides[0] + C * input_strides[1] +
                                  corners_z[corner] * input_strides[2] +
                                  corners_y[corner] * input_strides[3] +
                                  corners_x[corner] * input_strides[4];
        result += inputData[input_offset] * weights[corner];
      }
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

    if (padding_mode == 0) { // zeros padding
      if (within_bounds_3d(iz_nearest, iy_nearest, ix_nearest, in_D, in_H, in_W)) {
        const auto input_offset = N * input_strides[0] +
                                  C * input_strides[1] +
                                  iz_nearest * input_strides[2] +
                                  iy_nearest * input_strides[3] +
                                  ix_nearest * input_strides[4];
        outputData[output_offset] = inputData[input_offset];
      } else {
        outputData[output_offset] = T(0);
      }
    } else { // border or reflection padding - coordinates should be in bounds
      const auto input_offset = N * input_strides[0] + C * input_strides[1] +
                                iz_nearest * input_strides[2] +
                                iy_nearest * input_strides[3] +
                                ix_nearest * input_strides[4];
      outputData[output_offset] = inputData[input_offset];
    }
  } else {
    // For unsupported interpolation modes, set to zero
    outputData[output_offset] = T(0);
  }
}

// Vectorized kernel that processes multiple elements per thread
template <typename T>
kernel void grid_sampler_3d_vectorized(
    constant T* inputData [[buffer(0)]],
    device T* outputData [[buffer(1)]],
    constant T* gridData [[buffer(2)]],
    constant int& interpolation_mode [[buffer(3)]],
    constant int& padding_mode [[buffer(4)]],
    constant bool& align_corners [[buffer(5)]],
    constant ulong* input_sizes [[buffer(6)]],
    constant ulong* output_sizes [[buffer(7)]],
    constant ulong* input_strides [[buffer(8)]],
    constant ulong* output_strides [[buffer(9)]],
    constant ulong* grid_strides [[buffer(10)]],
    uint3 thread_index [[thread_position_in_grid]]) {

  constexpr int ELEMS_PER_THREAD = 4; // Process 4 elements per thread for better bandwidth

  const auto base_w = thread_index.x * ELEMS_PER_THREAD;
  const auto out_d_h_combined = thread_index.y;
  const auto n_c_combined = thread_index.z;

  // Extract individual indices
  const auto N = n_c_combined / input_sizes[1];
  const auto C = n_c_combined % input_sizes[1];
  const auto out_d = out_d_h_combined / output_sizes[3];
  const auto out_h = out_d_h_combined % output_sizes[3];

  // Early bounds check for batch and channel
  if (N >= input_sizes[0] || C >= input_sizes[1] ||
      out_d >= output_sizes[2] || out_h >= output_sizes[3]) {
    return;
  }

  // Cache frequently used values
  const auto in_D = input_sizes[2];
  const auto in_H = input_sizes[3];
  const auto in_W = input_sizes[4];
  const auto out_W = output_sizes[4];

  // Process up to ELEMS_PER_THREAD elements
  for (int elem = 0; elem < ELEMS_PER_THREAD; elem++) {
    const auto out_w = base_w + elem;

    // Check if this element is within bounds
    if (out_w >= out_W) break;

    // Pre-calculate grid offset for this element
    const auto grid_offset = N * grid_strides[0] +
                             out_d * grid_strides[1] +
                             out_h * grid_strides[2] +
                             out_w * grid_strides[3];

    // Read grid coordinates
    const T grid_x = gridData[grid_offset];
    const T grid_y = gridData[grid_offset + grid_strides[4]];
    const T grid_z = gridData[grid_offset + 2 * grid_strides[4]];

    // Pre-calculate output offset
    const auto output_offset = N * output_strides[0] +
                               C * output_strides[1] +
                               out_d * output_strides[2] +
                               out_h * output_strides[3] +
                               out_w * output_strides[4];

    if (interpolation_mode == 0) { // trilinear interpolation
      // Unnormalize coordinates
      T ix = grid_sampler_unnormalize(grid_x, in_W, align_corners);
      T iy = grid_sampler_unnormalize(grid_y, in_H, align_corners);
      T iz = grid_sampler_unnormalize(grid_z, in_D, align_corners);

      // Apply padding mode
      ix = compute_coordinates(ix, in_W, padding_mode, align_corners);
      iy = compute_coordinates(iy, in_H, padding_mode, align_corners);
      iz = compute_coordinates(iz, in_D, padding_mode, align_corners);

      // Get floor coordinates
      const long ix_tnw = static_cast<long>(floor(ix));
      const long iy_tnw = static_cast<long>(floor(iy));
      const long iz_tnw = static_cast<long>(floor(iz));

      // Pre-calculate interpolation weights
      const T x_diff = ix - T(ix_tnw);
      const T y_diff = iy - T(iy_tnw);
      const T z_diff = iz - T(iz_tnw);

      const T x_weights[2] = {T(1.0) - x_diff, x_diff};
      const T y_weights[2] = {T(1.0) - y_diff, y_diff};
      const T z_weights[2] = {T(1.0) - z_diff, z_diff};

      T result = T(0);

      // Sample 8 corners
      for (int z_idx = 0; z_idx < 2; z_idx++) {
        for (int y_idx = 0; y_idx < 2; y_idx++) {
          for (int x_idx = 0; x_idx < 2; x_idx++) {
            const long sample_x = ix_tnw + x_idx;
            const long sample_y = iy_tnw + y_idx;
            const long sample_z = iz_tnw + z_idx;

            const T weight = x_weights[x_idx] * y_weights[y_idx] * z_weights[z_idx];

            if (padding_mode == 0) { // zeros padding
              if (within_bounds_3d(sample_z, sample_y, sample_x, in_D, in_H, in_W)) {
                const auto input_offset = N * input_strides[0] + C * input_strides[1] +
                                         sample_z * input_strides[2] +
                                         sample_y * input_strides[3] +
                                         sample_x * input_strides[4];
                result += inputData[input_offset] * weight;
              }
            } else { // border/reflection padding
              const auto input_offset = N * input_strides[0] + C * input_strides[1] +
                                       sample_z * input_strides[2] +
                                       sample_y * input_strides[3] +
                                       sample_x * input_strides[4];
              result += inputData[input_offset] * weight;
            }
          }
        }
      }

      outputData[output_offset] = result;

    } else if (interpolation_mode == 1) { // nearest neighbor
      // Unnormalize and apply padding
      T ix = grid_sampler_unnormalize(grid_x, in_W, align_corners);
      T iy = grid_sampler_unnormalize(grid_y, in_H, align_corners);
      T iz = grid_sampler_unnormalize(grid_z, in_D, align_corners);

      ix = compute_coordinates(ix, in_W, padding_mode, align_corners);
      iy = compute_coordinates(iy, in_H, padding_mode, align_corners);
      iz = compute_coordinates(iz, in_D, padding_mode, align_corners);

      // Round to nearest
      const long ix_nearest = static_cast<long>(rint(ix));
      const long iy_nearest = static_cast<long>(rint(iy));
      const long iz_nearest = static_cast<long>(rint(iz));

      if (padding_mode == 0) { // zeros padding
        if (within_bounds_3d(iz_nearest, iy_nearest, ix_nearest, in_D, in_H, in_W)) {
          const auto input_offset = N * input_strides[0] + C * input_strides[1] +
                                    iz_nearest * input_strides[2] +
                                    iy_nearest * input_strides[3] +
                                    ix_nearest * input_strides[4];
          outputData[output_offset] = inputData[input_offset];
        } else {
          outputData[output_offset] = T(0);
        }
      } else { // border/reflection padding
        const auto input_offset = N * input_strides[0] + C * input_strides[1] +
                                  iz_nearest * input_strides[2] +
                                  iy_nearest * input_strides[3] +
                                  ix_nearest * input_strides[4];
        outputData[output_offset] = inputData[input_offset];
      }
    } else {
      outputData[output_offset] = T(0);
    }
  }
}

#define INSTANTIATE_GRID_SAMPLER_3D(DTYPE)                                          \
  template [[host_name("grid_sampler_3d_" #DTYPE)]] kernel void grid_sampler_3d<DTYPE>( \
      constant DTYPE * inputData [[buffer(0)]],                                     \
      device DTYPE * outputData [[buffer(1)]],                                      \
      constant DTYPE * gridData [[buffer(2)]],                                      \
      constant int & interpolation_mode [[buffer(3)]],                              \
      constant int & padding_mode [[buffer(4)]],                                    \
      constant bool & align_corners [[buffer(5)]],                                  \
      constant ulong * input_sizes [[buffer(6)]],                                   \
      constant ulong * output_sizes [[buffer(7)]],                                  \
      constant ulong * input_strides [[buffer(8)]],                                 \
      constant ulong * output_strides [[buffer(9)]],                                \
      constant ulong * grid_strides [[buffer(10)]],                                 \
      uint3 thread_index [[thread_position_in_grid]])

#define INSTANTIATE_GRID_SAMPLER_3D_VECTORIZED(DTYPE)                               \
  template [[host_name("grid_sampler_3d_vectorized_" #DTYPE)]] kernel void grid_sampler_3d_vectorized<DTYPE>( \
      constant DTYPE * inputData [[buffer(0)]],                                     \
      device DTYPE * outputData [[buffer(1)]],                                      \
      constant DTYPE * gridData [[buffer(2)]],                                      \
      constant int & interpolation_mode [[buffer(3)]],                              \
      constant int & padding_mode [[buffer(4)]],                                    \
      constant bool & align_corners [[buffer(5)]],                                  \
      constant ulong * input_sizes [[buffer(6)]],                                   \
      constant ulong * output_sizes [[buffer(7)]],                                  \
      constant ulong * input_strides [[buffer(8)]],                                 \
      constant ulong * output_strides [[buffer(9)]],                                \
      constant ulong * grid_strides [[buffer(10)]],                                 \
      uint3 thread_index [[thread_position_in_grid]])

INSTANTIATE_GRID_SAMPLER_3D(float);
INSTANTIATE_GRID_SAMPLER_3D(half);
INSTANTIATE_GRID_SAMPLER_3D_VECTORIZED(float);
INSTANTIATE_GRID_SAMPLER_3D_VECTORIZED(half);

#if __METAL_VERSION__ >= 310
// Note: bfloat support needs special handling for type conversions
INSTANTIATE_GRID_SAMPLER_3D(bfloat);
INSTANTIATE_GRID_SAMPLER_3D_VECTORIZED(bfloat);
#endif