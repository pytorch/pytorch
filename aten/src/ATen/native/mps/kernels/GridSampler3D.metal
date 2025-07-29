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

// Helper function to safely add gradient to input tensor with bounds checking
inline void safe_add_3d(device atomic<float>* grad_input,
                        long z, long y, long x,
                        long inp_D, long inp_H, long inp_W,
                        ulong base_offset,
                        constant ulong* grad_input_strides,
                        float weight) {
  if (within_bounds_3d(z, y, x, inp_D, inp_H, inp_W)) {
    const auto grad_input_offset = base_offset +
                                 z * grad_input_strides[2] +
                                 y * grad_input_strides[3] +
                                 x * grad_input_strides[4];
    atomic_fetch_add_explicit(&grad_input[grad_input_offset], weight, memory_order_relaxed);
  }
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

  T extra = coord - span * static_cast<T>(floor(coord / span));
  long flips = static_cast<long>(floor(coord / span));

  if (flips % 2 == 0) {
    return extra + min_val;
  } else {
    return span - extra + min_val;
  }
}

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

// Helper function to unnormalize coordinates from [-1, 1] to [0, size-1] or [-0.5, size-0.5]
// and compute the gradient multiplier
template<typename T>
T grid_sampler_unnormalize_set_grad(T coord, ulong size, bool align_corners, thread T* grad_in) {
  if (align_corners) {
    // unnormalize coord from [-1, 1] to [0, size - 1]
    *grad_in = T(size - 1) / T(2.0);
    return ((coord + T(1.0)) / T(2.0)) * T(size - 1);
  } else {
    // unnormalize coord from [-1, 1] to [-0.5, size - 0.5]
    *grad_in = T(size) / T(2.0);
    return ((coord + T(1.0)) * T(size) - T(1.0)) / T(2.0);
  }
}

// Compute source index and gradient multiplier with padding mode applied
template<typename T>
T grid_sampler_compute_source_index_set_grad(T coord, ulong size, int padding_mode, bool align_corners, thread T* grad_in) {
  coord = grid_sampler_unnormalize_set_grad(coord, size, align_corners, grad_in);

  if (padding_mode == 1) { // Border padding
    // Apply clip_coordinates_set_grad logic
    T grad_clip = T(1.0);
    if (coord < T(0.0)) {
      coord = T(0.0);
      grad_clip = T(0.0);
    } else if (coord > T(size - 1)) {
      coord = T(size - 1);
      grad_clip = T(0.0);
    }
    *grad_in = (*grad_in) * grad_clip;
  } else if (padding_mode == 2) { // Reflection padding
    // Apply reflect_coordinates_set_grad logic
    T grad_refl = T(1.0);
    T twice_low, twice_high;
    if (align_corners) {
      twice_low = T(0.0);
      twice_high = T(2 * (size - 1));
    } else {
      twice_low = T(-1.0);
      twice_high = T(2 * size - 1);
    }

    if (twice_low != twice_high) {
      T min_val = twice_low / T(2.0);
      T span = (twice_high - twice_low) / T(2.0);
      coord = coord - min_val;

      if (coord < T(0.0)) {
        coord = -coord;
        grad_refl = -grad_refl;
      }

      // Compute coord mod span
      T extra = coord - span * floor(coord / span);
      long flips = static_cast<long>(floor(coord / span));

      if (flips % 2 == 0) {
        coord = extra + min_val;
      } else {
        coord = span - extra + min_val;
        grad_refl = -grad_refl;
      }
    } else {
      coord = T(0.0);
      grad_refl = T(0.0);
    }

    // Apply clipping
    T grad_clip = T(1.0);
    if (coord < T(0.0)) {
      coord = T(0.0);
      grad_clip = T(0.0);
    } else if (coord > T(size - 1)) {
      coord = T(size - 1);
      grad_clip = T(0.0);
    }

    *grad_in = (*grad_in) * grad_refl * grad_clip;
  }

  return coord;
}

// Backward pass kernel for computing gradients w.r.t. input
kernel void grid_sampler_3d_backward_input(
    constant float* grad_output [[buffer(0)]],
    constant float* grid [[buffer(1)]],
    device atomic<float>* grad_input [[buffer(2)]],
    constant int& interpolation_mode [[buffer(3)]],
    constant int& padding_mode [[buffer(4)]],
    constant bool& align_corners [[buffer(5)]],
    constant ulong* input_sizes [[buffer(6)]],     // [N, C, D, H, W]
    constant ulong* output_sizes [[buffer(7)]],    // [N, C, D_out, H_out, W_out]
    constant ulong* grad_input_strides [[buffer(8)]],
    constant ulong* grid_strides [[buffer(9)]],
    constant ulong* grad_output_strides [[buffer(10)]],
    uint3 thread_index [[thread_position_in_grid]]) {

  // Thread indices map to output grid coordinates
  const auto out_w = thread_index.x;
  const auto out_d_h_combined = thread_index.y;
  const auto n = thread_index.z;

  // Extract individual indices
  const auto out_d = out_d_h_combined / output_sizes[3];
  const auto out_h = out_d_h_combined % output_sizes[3];

  // Bounds check
  if (n >= input_sizes[0] || out_d >= output_sizes[2] ||
      out_h >= output_sizes[3] || out_w >= output_sizes[4]) {
    return;
  }

  // Cache frequently used values
  const auto C = input_sizes[1];
  const auto inp_D = input_sizes[2];
  const auto inp_H = input_sizes[3];
  const auto inp_W = input_sizes[4];

  // Pre-calculate grid offset for this output location
  const auto grid_offset = n * grid_strides[0] +
                           out_d * grid_strides[1] +
                           out_h * grid_strides[2] +
                           out_w * grid_strides[3];

  // Read grid coordinates
  const float grid_x = grid[grid_offset];
  const float grid_y = grid[grid_offset + grid_strides[4]];
  const float grid_z = grid[grid_offset + 2 * grid_strides[4]];

  // Transform grid coordinates to input space and compute gradient multipliers
  float gix_mult, giy_mult, giz_mult;
  float ix = grid_sampler_compute_source_index_set_grad(grid_x, inp_W, padding_mode, align_corners, &gix_mult);
  float iy = grid_sampler_compute_source_index_set_grad(grid_y, inp_H, padding_mode, align_corners, &giy_mult);
  float iz = grid_sampler_compute_source_index_set_grad(grid_z, inp_D, padding_mode, align_corners, &giz_mult);

  if (interpolation_mode == 0) { // trilinear interpolation
    // Get floor coordinates for all 8 corners
    const int ix_tnw = static_cast<int>(floor(ix));
    const int iy_tnw = static_cast<int>(floor(iy));
    const int iz_tnw = static_cast<int>(floor(iz));

         // Calculate all 8 corner coordinates using CUDA naming convention
     // tnw = top north west, tne = top north east, etc.
     const int ix_tne = ix_tnw + 1;
     const int iy_tne = iy_tnw;
     const int iz_tne = iz_tnw;

     const int ix_tsw = ix_tnw;
     const int iy_tsw = iy_tnw + 1;
     const int iz_tsw = iz_tnw;

     const int ix_tse = ix_tnw + 1;
     const int iy_tse = iy_tnw + 1;
     const int iz_tse = iz_tnw;

     const int ix_bnw = ix_tnw;
     const int iy_bnw = iy_tnw;
     const int iz_bnw = iz_tnw + 1;

     const int ix_bne = ix_tnw + 1;
     const int iy_bne = iy_tnw;
     const int iz_bne = iz_tnw + 1;

     const int ix_bsw = ix_tnw;
     const int iy_bsw = iy_tnw + 1;
     const int iz_bsw = iz_tnw + 1;

     const int ix_bse = ix_tnw + 1;
     const int iy_bse = iy_tnw + 1;
     const int iz_bse = iz_tnw + 1;

     // Calculate weights using CUDA formula (surfaces to each neighbor)
     const float tnw = (ix_bse - ix) * (iy_bse - iy) * (iz_bse - iz);
     const float tne = (ix - ix_bsw) * (iy_bsw - iy) * (iz_bsw - iz);
     const float tsw = (ix_bne - ix) * (iy - iy_bne) * (iz_bne - iz);
     const float tse = (ix - ix_bnw) * (iy - iy_bnw) * (iz_bnw - iz);
     const float bnw = (ix_tse - ix) * (iy_tse - iy) * (iz - iz_tse);
     const float bne = (ix - ix_tsw) * (iy_tsw - iy) * (iz - iz_tsw);
     const float bsw = (ix_tne - ix) * (iy - iy_tne) * (iz - iz_tne);
     const float bse = (ix - ix_tnw) * (iy - iy_tnw) * (iz - iz_tnw);

     // Process all channels at this grid location
     for (ulong c = 0; c < C; c++) {
       // Get gradient output value for this channel
       const auto grad_out_offset = n * grad_output_strides[0] +
                                    c * grad_output_strides[1] +
                                    out_d * grad_output_strides[2] +
                                    out_h * grad_output_strides[3] +
                                    out_w * grad_output_strides[4];
       const float gOut = grad_output[grad_out_offset];

      // Accumulate gradients to all 8 corner pixels using safe bounds checking
      const auto base_grad_input_offset = n * grad_input_strides[0] + c * grad_input_strides[1];

      // Safely add gradients to all 8 corners using atomic add
      safe_add_3d(grad_input, iz_tnw, iy_tnw, ix_tnw, inp_D, inp_H, inp_W, base_grad_input_offset, grad_input_strides, tnw * gOut);
      safe_add_3d(grad_input, iz_tne, iy_tne, ix_tne, inp_D, inp_H, inp_W, base_grad_input_offset, grad_input_strides, tne * gOut);
      safe_add_3d(grad_input, iz_tsw, iy_tsw, ix_tsw, inp_D, inp_H, inp_W, base_grad_input_offset, grad_input_strides, tsw * gOut);
      safe_add_3d(grad_input, iz_tse, iy_tse, ix_tse, inp_D, inp_H, inp_W, base_grad_input_offset, grad_input_strides, tse * gOut);
      safe_add_3d(grad_input, iz_bnw, iy_bnw, ix_bnw, inp_D, inp_H, inp_W, base_grad_input_offset, grad_input_strides, bnw * gOut);
      safe_add_3d(grad_input, iz_bne, iy_bne, ix_bne, inp_D, inp_H, inp_W, base_grad_input_offset, grad_input_strides, bne * gOut);
      safe_add_3d(grad_input, iz_bsw, iy_bsw, ix_bsw, inp_D, inp_H, inp_W, base_grad_input_offset, grad_input_strides, bsw * gOut);
      safe_add_3d(grad_input, iz_bse, iy_bse, ix_bse, inp_D, inp_H, inp_W, base_grad_input_offset, grad_input_strides, bse * gOut);
     }

  } else if (interpolation_mode == 1) { // nearest neighbor
    // Round to nearest integer coordinates
    const int ix_nearest = static_cast<int>(rint(ix));
    const int iy_nearest = static_cast<int>(rint(iy));
    const int iz_nearest = static_cast<int>(rint(iz));

    // Process all channels at this grid location
    for (ulong c = 0; c < C; c++) {
      // Get gradient output value for this channel
      const auto grad_out_offset = n * grad_output_strides[0] +
                                   c * grad_output_strides[1] +
                                   out_d * grad_output_strides[2] +
                                   out_h * grad_output_strides[3] +
                                   out_w * grad_output_strides[4];
      const float gOut = grad_output[grad_out_offset];

      const auto base_grad_input_offset = n * grad_input_strides[0] + c * grad_input_strides[1];
      safe_add_3d(grad_input, iz_nearest, iy_nearest, ix_nearest, inp_D, inp_H, inp_W, base_grad_input_offset, grad_input_strides, gOut);
    }
  }
}

// Backward pass kernel for computing gradients w.r.t. grid
kernel void grid_sampler_3d_backward_grid(
    constant float* grad_output [[buffer(0)]],
    constant float* input [[buffer(1)]],
    constant float* grid [[buffer(2)]],
    device float* grad_grid [[buffer(3)]],
    constant int& interpolation_mode [[buffer(4)]],
    constant int& padding_mode [[buffer(5)]],
    constant bool& align_corners [[buffer(6)]],
    constant ulong* input_sizes [[buffer(7)]],
    constant ulong* output_sizes [[buffer(8)]],
    constant ulong* input_strides [[buffer(9)]],
    constant ulong* grad_grid_strides [[buffer(10)]],
    constant ulong* grid_strides [[buffer(11)]],
    constant ulong* grad_output_strides [[buffer(12)]],
    uint3 thread_index [[thread_position_in_grid]]) {

  const auto out_w = thread_index.x;
  const auto out_d_h_combined = thread_index.y;
  const auto N = thread_index.z;

  // Extract individual indices
  const auto out_d = out_d_h_combined / output_sizes[3];
  const auto out_h = out_d_h_combined % output_sizes[3];

  // Bounds check
  if (N >= input_sizes[0] || out_d >= output_sizes[2] ||
      out_h >= output_sizes[3] || out_w >= output_sizes[4]) {
    return;
  }

  // Cache frequently used values
  const auto C = input_sizes[1];
  const auto inp_D = input_sizes[2];
  const auto inp_H = input_sizes[3];
  const auto inp_W = input_sizes[4];

  // Pre-calculate grid offset
  const auto grid_offset = N * grid_strides[0] +
                           out_d * grid_strides[1] +
                           out_h * grid_strides[2] +
                           out_w * grid_strides[3];

  // Read grid coordinates
  const float grid_x = grid[grid_offset];
  const float grid_y = grid[grid_offset + grid_strides[4]];
  const float grid_z = grid[grid_offset + 2 * grid_strides[4]];

  // Transform grid coordinates and get gradient multipliers
  float gix_mult, giy_mult, giz_mult;
  float ix = grid_sampler_compute_source_index_set_grad(grid_x, inp_W, padding_mode, align_corners, &gix_mult);
  float iy = grid_sampler_compute_source_index_set_grad(grid_y, inp_H, padding_mode, align_corners, &giy_mult);
  float iz = grid_sampler_compute_source_index_set_grad(grid_z, inp_D, padding_mode, align_corners, &giz_mult);

  // Calculate grad_grid offset for this thread
  const auto grad_grid_base_offset = N * grad_grid_strides[0] +
                                     out_d * grad_grid_strides[1] +
                                     out_h * grad_grid_strides[2] +
                                     out_w * grad_grid_strides[3];

  if (interpolation_mode == 0) { // trilinear interpolation
    // Calculate corner coordinates
    const long ix_tnw = static_cast<long>(floor(ix));
    const long iy_tnw = static_cast<long>(floor(iy));
    const long iz_tnw = static_cast<long>(floor(iz));

    const long ix_tne = ix_tnw + 1;
    const long iy_tne = iy_tnw;
    const long iz_tne = iz_tnw;

    const long ix_tsw = ix_tnw;
    const long iy_tsw = iy_tnw + 1;
    const long iz_tsw = iz_tnw;

    const long ix_tse = ix_tnw + 1;
    const long iy_tse = iy_tnw + 1;
    const long iz_tse = iz_tnw;

    const long ix_bnw = ix_tnw;
    const long iy_bnw = iy_tnw;
    const long iz_bnw = iz_tnw + 1;

    const long ix_bne = ix_tnw + 1;
    const long iy_bne = iy_tnw;
    const long iz_bne = iz_tnw + 1;

    const long ix_bsw = ix_tnw;
    const long iy_bsw = iy_tnw + 1;
    const long iz_bsw = iz_tnw + 1;

    const long ix_bse = ix_tnw + 1;
    const long iy_bse = iy_tnw + 1;
    const long iz_bse = iz_tnw + 1;

    float gix = 0.0, giy = 0.0, giz = 0.0;

    // Process all channels and accumulate gradients following CUDA implementation
    for (ulong c = 0; c < C; c++) {
      const auto grad_out_offset = N * grad_output_strides[0] +
                                   c * grad_output_strides[1] +
                                   out_d * grad_output_strides[2] +
                                   out_h * grad_output_strides[3] +
                                   out_w * grad_output_strides[4];
      const float gOut = grad_output[grad_out_offset];

      const auto input_base_offset = N * input_strides[0] + c * input_strides[1];

      // Compute gradients for each corner following CUDA implementation exactly
      if (within_bounds_3d(iz_tnw, iy_tnw, ix_tnw, inp_D, inp_H, inp_W)) {
        const float tnw_val = input[input_base_offset + iz_tnw * input_strides[2] + iy_tnw * input_strides[3] + ix_tnw * input_strides[4]];
        gix -= tnw_val * (iy_bse - iy) * (iz_bse - iz) * gOut;
        giy -= tnw_val * (ix_bse - ix) * (iz_bse - iz) * gOut;
        giz -= tnw_val * (ix_bse - ix) * (iy_bse - iy) * gOut;
      }
      if (within_bounds_3d(iz_tne, iy_tne, ix_tne, inp_D, inp_H, inp_W)) {
        const float tne_val = input[input_base_offset + iz_tne * input_strides[2] + iy_tne * input_strides[3] + ix_tne * input_strides[4]];
        gix += tne_val * (iy_bsw - iy) * (iz_bsw - iz) * gOut;
        giy -= tne_val * (ix - ix_bsw) * (iz_bsw - iz) * gOut;
        giz -= tne_val * (ix - ix_bsw) * (iy_bsw - iy) * gOut;
      }
      if (within_bounds_3d(iz_tsw, iy_tsw, ix_tsw, inp_D, inp_H, inp_W)) {
        const float tsw_val = input[input_base_offset + iz_tsw * input_strides[2] + iy_tsw * input_strides[3] + ix_tsw * input_strides[4]];
        gix -= tsw_val * (iy - iy_bne) * (iz_bne - iz) * gOut;
        giy += tsw_val * (ix_bne - ix) * (iz_bne - iz) * gOut;
        giz -= tsw_val * (ix_bne - ix) * (iy - iy_bne) * gOut;
      }
      if (within_bounds_3d(iz_tse, iy_tse, ix_tse, inp_D, inp_H, inp_W)) {
        const float tse_val = input[input_base_offset + iz_tse * input_strides[2] + iy_tse * input_strides[3] + ix_tse * input_strides[4]];
        gix += tse_val * (iy - iy_bnw) * (iz_bnw - iz) * gOut;
        giy += tse_val * (ix - ix_bnw) * (iz_bnw - iz) * gOut;
        giz -= tse_val * (ix - ix_bnw) * (iy - iy_bnw) * gOut;
      }
      if (within_bounds_3d(iz_bnw, iy_bnw, ix_bnw, inp_D, inp_H, inp_W)) {
        const float bnw_val = input[input_base_offset + iz_bnw * input_strides[2] + iy_bnw * input_strides[3] + ix_bnw * input_strides[4]];
        gix -= bnw_val * (iy_tse - iy) * (iz - iz_tse) * gOut;
        giy -= bnw_val * (ix_tse - ix) * (iz - iz_tse) * gOut;
        giz += bnw_val * (ix_tse - ix) * (iy_tse - iy) * gOut;
      }
      if (within_bounds_3d(iz_bne, iy_bne, ix_bne, inp_D, inp_H, inp_W)) {
        const float bne_val = input[input_base_offset + iz_bne * input_strides[2] + iy_bne * input_strides[3] + ix_bne * input_strides[4]];
        gix += bne_val * (iy_tsw - iy) * (iz - iz_tsw) * gOut;
        giy -= bne_val * (ix - ix_tsw) * (iz - iz_tsw) * gOut;
        giz += bne_val * (ix - ix_tsw) * (iy_tsw - iy) * gOut;
      }
      if (within_bounds_3d(iz_bsw, iy_bsw, ix_bsw, inp_D, inp_H, inp_W)) {
        const float bsw_val = input[input_base_offset + iz_bsw * input_strides[2] + iy_bsw * input_strides[3] + ix_bsw * input_strides[4]];
        gix -= bsw_val * (iy - iy_tne) * (iz - iz_tne) * gOut;
        giy += bsw_val * (ix_tne - ix) * (iz - iz_tne) * gOut;
        giz += bsw_val * (ix_tne - ix) * (iy - iy_tne) * gOut;
      }
      if (within_bounds_3d(iz_bse, iy_bse, ix_bse, inp_D, inp_H, inp_W)) {
        const float bse_val = input[input_base_offset + iz_bse * input_strides[2] + iy_bse * input_strides[3] + ix_bse * input_strides[4]];
        gix += bse_val * (iy - iy_tnw) * (iz - iz_tnw) * gOut;
        giy += bse_val * (ix - ix_tnw) * (iz - iz_tnw) * gOut;
        giz += bse_val * (ix - ix_tnw) * (iy - iy_tnw) * gOut;
      }
    }

    // Write gradients to grad_grid, multiplied by the coordinate gradient multipliers
    grad_grid[grad_grid_base_offset] = gix_mult * gix;
    grad_grid[grad_grid_base_offset + grad_grid_strides[4]] = giy_mult * giy;
    grad_grid[grad_grid_base_offset + 2 * grad_grid_strides[4]] = giz_mult * giz;

  } else if (interpolation_mode == 1) { // nearest neighbor
    // For nearest neighbor, gradients w.r.t. grid are zero (following CUDA implementation)
    grad_grid[grad_grid_base_offset] = 0.0;
    grad_grid[grad_grid_base_offset + grad_grid_strides[4]] = 0.0;
    grad_grid[grad_grid_base_offset + 2 * grad_grid_strides[4]] = 0.0;
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
INSTANTIATE_GRID_SAMPLER_3D(bfloat);
INSTANTIATE_GRID_SAMPLER_3D_VECTORIZED(bfloat);
#endif