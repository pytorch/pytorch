#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/OpMathType.h>
#include <ATen/native/cuda/GridSampler.h>
#include <ATen/native/GridSamplerUtils.h>
#include <ATen/native/cuda/GridSampler.cuh>
#include <ATen/native/cuda/UpSample.cuh>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/detail/TensorInfo.cuh>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/detail/KernelUtils.h>
#include <ATen/core/TensorBase.h>
#include <ATen/Dispatch.h>
#include <ATen/ceil_div.h>
#include <c10/macros/Macros.h>
#include <cmath>

namespace at::native {

using namespace at::cuda::detail;

using at::native::detail::GridSamplerInterpolation;
using at::native::detail::GridSamplerPadding;

namespace {
  template <typename scalar_t, typename index_t>
  C10_LAUNCH_BOUNDS_1(256)
  __global__ void grid_sampler_2d_kernel(
      const index_t nthreads,
      TensorInfo<const scalar_t, index_t> input,
      TensorInfo<const scalar_t, index_t> grid,
      TensorInfo<scalar_t, index_t> output,
      const GridSamplerInterpolation interpolation_mode,
      const GridSamplerPadding padding_mode,
      bool align_corners) {

    using opmath_t = at::opmath_type<scalar_t>;
    index_t C = input.sizes[1];
    index_t inp_H = input.sizes[2];
    index_t inp_W = input.sizes[3];
    index_t out_H = grid.sizes[1];
    index_t out_W = grid.sizes[2];
    index_t inp_sN = input.strides[0];
    index_t inp_sC = input.strides[1];
    index_t inp_sH = input.strides[2];
    index_t inp_sW = input.strides[3];
    index_t grid_sN = grid.strides[0];
    index_t grid_sH = grid.strides[1];
    index_t grid_sW = grid.strides[2];
    index_t grid_sCoor = grid.strides[3];
    index_t out_sN = output.strides[0];
    index_t out_sC = output.strides[1];
    index_t out_sH = output.strides[2];
    index_t out_sW = output.strides[3];

    CUDA_KERNEL_LOOP_TYPE(index, nthreads, index_t) {
      const index_t w = index % out_W;
      const index_t h = (index / out_W) % out_H;
      const index_t n = index / (out_H * out_W);
      const index_t grid_offset = n * grid_sN + h * grid_sH + w * grid_sW;

      // get the corresponding input x, y co-ordinates from grid
      opmath_t x = grid.data[grid_offset];
      opmath_t y = grid.data[grid_offset + grid_sCoor];

      opmath_t ix = grid_sampler_compute_source_index(x, inp_W, padding_mode, align_corners);
      opmath_t iy = grid_sampler_compute_source_index(y, inp_H, padding_mode, align_corners);

      if (interpolation_mode == GridSamplerInterpolation::Bilinear) {
        // get NE, NW, SE, SW pixel values from (x, y)
        index_t ix_nw = static_cast<index_t>(::floor(ix));
        index_t iy_nw = static_cast<index_t>(::floor(iy));
        index_t ix_ne = ix_nw + 1;
        index_t iy_ne = iy_nw;
        index_t ix_sw = ix_nw;
        index_t iy_sw = iy_nw + 1;
        index_t ix_se = ix_nw + 1;
        index_t iy_se = iy_nw + 1;

        // get surfaces to each neighbor:
        opmath_t nw = (ix_se - ix)    * (iy_se - iy);
        opmath_t ne = (ix    - ix_sw) * (iy_sw - iy);
        opmath_t sw = (ix_ne - ix)    * (iy    - iy_ne);
        opmath_t se = (ix    - ix_nw) * (iy    - iy_nw);

        // calculate bilinear weighted pixel value and set output pixel
        auto inp_ptr_NC = input.data + n * inp_sN;
        auto out_ptr_NCHW = output.data + n * out_sN + h * out_sH + w * out_sW;
        for (index_t c = 0; c < C; ++c, inp_ptr_NC += inp_sC, out_ptr_NCHW += out_sC) {
          opmath_t out_acc = 0;
          if (within_bounds_2d(iy_nw, ix_nw, inp_H, inp_W)) {
            out_acc += inp_ptr_NC[iy_nw * inp_sH + ix_nw * inp_sW] * nw;
          }
          if (within_bounds_2d(iy_ne, ix_ne, inp_H, inp_W)) {
            out_acc += inp_ptr_NC[iy_ne * inp_sH + ix_ne * inp_sW] * ne;
          }
          if (within_bounds_2d(iy_sw, ix_sw, inp_H, inp_W)) {
            out_acc += inp_ptr_NC[iy_sw * inp_sH + ix_sw * inp_sW] * sw;
          }
          if (within_bounds_2d(iy_se, ix_se, inp_H, inp_W)) {
            out_acc += inp_ptr_NC[iy_se * inp_sH + ix_se * inp_sW] * se;
          }
          *out_ptr_NCHW = out_acc;
        }
      } else if (interpolation_mode == GridSamplerInterpolation::Nearest) {
        index_t ix_nearest = static_cast<index_t>(std::nearbyint(ix));
        index_t iy_nearest = static_cast<index_t>(std::nearbyint(iy));

        // assign nearest neighbour pixel value to output pixel
        auto inp_ptr_NC = input.data + n * inp_sN;
        auto out_ptr_NCHW = output.data + n * out_sN + h * out_sH + w * out_sW;
        for (index_t c = 0; c < C; ++c, inp_ptr_NC += inp_sC, out_ptr_NCHW += out_sC) {
          if (within_bounds_2d(iy_nearest, ix_nearest, inp_H, inp_W)) {
            *out_ptr_NCHW = inp_ptr_NC[iy_nearest * inp_sH + ix_nearest * inp_sW];
          } else {
            *out_ptr_NCHW = static_cast<scalar_t>(0);
          }
        }
      } else if (interpolation_mode == GridSamplerInterpolation::Bicubic) {

        ix = grid_sampler_unnormalize(x, inp_W, align_corners);
        iy = grid_sampler_unnormalize(y, inp_H, align_corners);

        opmath_t ix_nw = std::floor(ix);
        opmath_t iy_nw = std::floor(iy);

        const opmath_t tx = ix - ix_nw;
        const opmath_t ty = iy - iy_nw;

        auto inp_ptr_NC = input.data + n * inp_sN;
        auto out_ptr_NCHW = output.data + n * out_sN + h * out_sH + w * out_sW;
        for (index_t c = 0; c < C; ++c, inp_ptr_NC += inp_sC, out_ptr_NCHW += out_sC) {
          opmath_t coefficients[4];

          #pragma unroll 4
          for (index_t i = 0; i < 4; ++i) {
            coefficients[i] = cubic_interp1d(
              get_value_bounded<scalar_t>(inp_ptr_NC, ix_nw - 1, iy_nw - 1 + i, inp_W, inp_H, inp_sW, inp_sH, padding_mode, align_corners),
              get_value_bounded<scalar_t>(inp_ptr_NC, ix_nw + 0, iy_nw - 1 + i, inp_W, inp_H, inp_sW, inp_sH, padding_mode, align_corners),
              get_value_bounded<scalar_t>(inp_ptr_NC, ix_nw + 1, iy_nw - 1 + i, inp_W, inp_H, inp_sW, inp_sH, padding_mode, align_corners),
              get_value_bounded<scalar_t>(inp_ptr_NC, ix_nw + 2, iy_nw - 1 + i, inp_W, inp_H, inp_sW, inp_sH, padding_mode, align_corners),
              tx);
          }

          *out_ptr_NCHW = cubic_interp1d(
            coefficients[0],
            coefficients[1],
            coefficients[2],
            coefficients[3],
            ty);
        }
      }
    }
  }

  template <typename scalar_t, typename index_t>
  C10_LAUNCH_BOUNDS_1(512)
  __global__ void grid_sampler_3d_kernel(
      const index_t nthreads,
      TensorInfo<const scalar_t, index_t> input,
      TensorInfo<const scalar_t, index_t> grid,
      TensorInfo<scalar_t, index_t> output,
      const GridSamplerInterpolation interpolation_mode,
      const GridSamplerPadding padding_mode,
      bool align_corners) {

    using opmath_t = at::opmath_type<scalar_t>;
    index_t C = input.sizes[1];
    index_t inp_D = input.sizes[2];
    index_t inp_H = input.sizes[3];
    index_t inp_W = input.sizes[4];
    index_t out_D = grid.sizes[1];
    index_t out_H = grid.sizes[2];
    index_t out_W = grid.sizes[3];
    index_t inp_sN = input.strides[0];
    index_t inp_sC = input.strides[1];
    index_t inp_sD = input.strides[2];
    index_t inp_sH = input.strides[3];
    index_t inp_sW = input.strides[4];
    index_t grid_sN = grid.strides[0];
    index_t grid_sD = grid.strides[1];
    index_t grid_sH = grid.strides[2];
    index_t grid_sW = grid.strides[3];
    index_t grid_sCoor = grid.strides[4];
    index_t out_sN = output.strides[0];
    index_t out_sC = output.strides[1];
    index_t out_sD = output.strides[2];
    index_t out_sH = output.strides[3];
    index_t out_sW = output.strides[4];

    CUDA_KERNEL_LOOP_TYPE(index, nthreads, index_t) {
      const index_t w = index % out_W;
      const index_t h = (index / out_W) % out_H;
      const index_t d = (index / (out_H * out_W)) % out_D;
      const index_t n = index / (out_D * out_H * out_W);
      const index_t grid_offset = n * grid_sN + d * grid_sD + h * grid_sH + w * grid_sW;

      // get the corresponding input x, y, z co-ordinates from grid
      opmath_t x = grid.data[grid_offset];
      opmath_t y = grid.data[grid_offset + grid_sCoor];
      opmath_t z = grid.data[grid_offset + 2 * grid_sCoor];

      opmath_t ix = grid_sampler_compute_source_index(x, inp_W, padding_mode, align_corners);
      opmath_t iy = grid_sampler_compute_source_index(y, inp_H, padding_mode, align_corners);
      opmath_t iz = grid_sampler_compute_source_index(z, inp_D, padding_mode, align_corners);

      if (interpolation_mode == GridSamplerInterpolation::Bilinear) {
        // get corner pixel values from (x, y, z)
        // for 4d, we used north-east-south-west
        // for 5d, we add top-bottom
        index_t ix_tnw = static_cast<index_t>(::floor(ix));
        index_t iy_tnw = static_cast<index_t>(::floor(iy));
        index_t iz_tnw = static_cast<index_t>(::floor(iz));

        index_t ix_tne = ix_tnw + 1;
        index_t iy_tne = iy_tnw;
        index_t iz_tne = iz_tnw;

        index_t ix_tsw = ix_tnw;
        index_t iy_tsw = iy_tnw + 1;
        index_t iz_tsw = iz_tnw;

        index_t ix_tse = ix_tnw + 1;
        index_t iy_tse = iy_tnw + 1;
        index_t iz_tse = iz_tnw;

        index_t ix_bnw = ix_tnw;
        index_t iy_bnw = iy_tnw;
        index_t iz_bnw = iz_tnw + 1;

        index_t ix_bne = ix_tnw + 1;
        index_t iy_bne = iy_tnw;
        index_t iz_bne = iz_tnw + 1;

        index_t ix_bsw = ix_tnw;
        index_t iy_bsw = iy_tnw + 1;
        index_t iz_bsw = iz_tnw + 1;

        index_t ix_bse = ix_tnw + 1;
        index_t iy_bse = iy_tnw + 1;
        index_t iz_bse = iz_tnw + 1;

        // get surfaces to each neighbor:
        opmath_t tnw = (ix_bse - ix)    * (iy_bse - iy)    * (iz_bse - iz);
        opmath_t tne = (ix    - ix_bsw) * (iy_bsw - iy)    * (iz_bsw - iz);
        opmath_t tsw = (ix_bne - ix)    * (iy    - iy_bne) * (iz_bne - iz);
        opmath_t tse = (ix    - ix_bnw) * (iy    - iy_bnw) * (iz_bnw - iz);
        opmath_t bnw = (ix_tse - ix)    * (iy_tse - iy)    * (iz - iz_tse);
        opmath_t bne = (ix    - ix_tsw) * (iy_tsw - iy)    * (iz - iz_tsw);
        opmath_t bsw = (ix_tne - ix)    * (iy    - iy_tne) * (iz - iz_tne);
        opmath_t bse = (ix    - ix_tnw) * (iy    - iy_tnw) * (iz - iz_tnw);

        auto inp_ptr_NC = input.data + n * inp_sN;
        auto out_ptr_NCDHW = output.data + n * out_sN + d * out_sD + h * out_sH + w * out_sW;
        for (index_t c = 0; c < C; ++c, inp_ptr_NC += inp_sC, out_ptr_NCDHW += out_sC) {
          //   (c, iz_tnw, iy_tnw, ix_tnw) * tnw + (c, iz_tne, iy_tne, ix_tne) * tne
          // + (c, iz_tsw, iy_tsw, ix_tsw) * tsw + (c, iz_tse, iy_tse, ix_tse) * tse
          // + (c, iz_bnw, iy_bnw, ix_bnw) * bnw + (c, iz_bne, iy_bne, ix_bne) * bne
          // + (c, iz_bsw, iy_bsw, ix_bsw) * bsw + (c, iz_bse, iy_bse, ix_bse) * bse
          opmath_t out_acc = 0;
          if (within_bounds_3d(iz_tnw, iy_tnw, ix_tnw, inp_D, inp_H, inp_W)) {
            out_acc += inp_ptr_NC[iz_tnw * inp_sD + iy_tnw * inp_sH + ix_tnw * inp_sW] * tnw;
          }
          if (within_bounds_3d(iz_tne, iy_tne, ix_tne, inp_D, inp_H, inp_W)) {
            out_acc += inp_ptr_NC[iz_tne * inp_sD + iy_tne * inp_sH + ix_tne * inp_sW] * tne;
          }
          if (within_bounds_3d(iz_tsw, iy_tsw, ix_tsw, inp_D, inp_H, inp_W)) {
            out_acc += inp_ptr_NC[iz_tsw * inp_sD + iy_tsw * inp_sH + ix_tsw * inp_sW] * tsw;
          }
          if (within_bounds_3d(iz_tse, iy_tse, ix_tse, inp_D, inp_H, inp_W)) {
            out_acc += inp_ptr_NC[iz_tse * inp_sD + iy_tse * inp_sH + ix_tse * inp_sW] * tse;
          }
          if (within_bounds_3d(iz_bnw, iy_bnw, ix_bnw, inp_D, inp_H, inp_W)) {
            out_acc += inp_ptr_NC[iz_bnw * inp_sD + iy_bnw * inp_sH + ix_bnw * inp_sW] * bnw;
          }
          if (within_bounds_3d(iz_bne, iy_bne, ix_bne, inp_D, inp_H, inp_W)) {
            out_acc += inp_ptr_NC[iz_bne * inp_sD + iy_bne * inp_sH + ix_bne * inp_sW] * bne;
          }
          if (within_bounds_3d(iz_bsw, iy_bsw, ix_bsw, inp_D, inp_H, inp_W)) {
            out_acc += inp_ptr_NC[iz_bsw * inp_sD + iy_bsw * inp_sH + ix_bsw * inp_sW] * bsw;
          }
          if (within_bounds_3d(iz_bse, iy_bse, ix_bse, inp_D, inp_H, inp_W)) {
            out_acc += inp_ptr_NC[iz_bse * inp_sD + iy_bse * inp_sH + ix_bse * inp_sW] * bse;
          }
          *out_ptr_NCDHW = out_acc;
        }
      } else if (interpolation_mode == GridSamplerInterpolation::Nearest) {
        index_t ix_nearest = static_cast<index_t>(std::nearbyint(ix));
        index_t iy_nearest = static_cast<index_t>(std::nearbyint(iy));
        index_t iz_nearest = static_cast<index_t>(std::nearbyint(iz));

        // assign nearest neighbour pixel value to output pixel
        auto inp_ptr_NC = input.data + n * inp_sN;
        auto out_ptr_NCDHW = output.data + n * out_sN + d * out_sD + h * out_sH + w * out_sW;
        for (index_t c = 0; c < C; ++c, inp_ptr_NC += inp_sC, out_ptr_NCDHW += out_sC) {
          if (within_bounds_3d(iz_nearest, iy_nearest, ix_nearest, inp_D, inp_H, inp_W)) {
            *out_ptr_NCDHW = inp_ptr_NC[iz_nearest * inp_sD + iy_nearest * inp_sH + ix_nearest * inp_sW];
          } else {
            *out_ptr_NCDHW = static_cast<scalar_t>(0);
          }
        }
      }
    }
  }

// Note [Passing pointer and offset to fastAtomicAdd]
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// For its internal bounds checking, fastAtomicAdd needs to know where the destination address
// lies relative to the entire tensor, so we pass the base grad_input.data and full offset information,
// including batch * channel offset (NC_offset).

#ifdef USE_ROCM
// Note [ROCm-specific GridSampler backward optimization]
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// The original backward kernel suffers from severe global atomic contention on (chiplet-based) AMD GPUs.
// This optimized implementation for ROCm uses two main strategies:
// 1. Shared memory (LDS) privatization: For `grad_input`, each thread block uses a 2D
//    tile in shared memory as a private accumulator. Atomics are performed on this fast
//    on-chip memory. After the block completes, results are written back to global memory,
//    drastically reducing expensive global atomic operations.
// 2. Channel Chunking: To handle inputs with many channels, we process channels in fixed-size
//    chunks to keep LDS usage bounded and predictable.
// 3. LDS Bank Conflict Avoidance: Padding is added to the inner dimension of the LDS
//    tile to prevent bank conflicts between threads in a warp.
// 4. Compile-time specialization: The kernel is templated on interpolation mode,
//    padding mode, align_corners, and whether the input requires gradients,
//    which were all function arguments originally. The host-side dispatcher
//    launches a specialized kernel version, allowing the compiler to eliminate
//    all mode-related branching inside the kernel.
//
// This optimization is currently enabled for the common Bilinear mode.
// Other modes fall back to the original kernel to ensure correctness.

  // Compile-time specialization to avoid runtime thread divergence.
  // A separate kernel will be created for each combination and pre-compiled.
  // This is expected to be a performance win.
  // Particularly, templatized INPUT_REQ_GRAD makes the compiler completely remove
  // all code related to `grad_input` if users don't need the gradient for `input`.
  template <typename scalar_t, typename index_t, GridSamplerInterpolation INTERP_MODE,
            GridSamplerPadding PADDING_MODE, bool ALIGN_CORNERS, bool INPUT_REQ_GRAD>
  C10_LAUNCH_BOUNDS_1(256)
  __global__ void grid_sampler_2d_backward_kernel_optimized(
      TensorInfo<const scalar_t, index_t> grad_output,
      TensorInfo<const scalar_t, index_t> input,
      TensorInfo<const scalar_t, index_t> grid,
      TensorInfo<scalar_t, index_t> grad_input,
      TensorInfo<scalar_t, index_t> grad_grid) {

    // Typically `float` even if `scalar_t` is `half` or `bflaot16` to ensure
    // all intermediate calculations (like accumulations in LDS) are done in full precision.
    using opmath_t = at::opmath_type<scalar_t>;

    // 2D tile size.
    // Each thread block will process such a tile of the output grid.
    constexpr int TILE_W = 16;
    constexpr int TILE_H = 16;
    // 1-pixel halo for Bilinear interpolation.
    // Bilinear interpolation for an output pixel at (y, x) needs to write gradients to input pixels at (y_in, x_in), (y_in+1, x_in), (y_in, x_in+1), and (y_in+1, x_in+1).
    // This ensures the LDS tile can accommodate contributions for pixels just outside the tile.
    // TODO: Bicubic interpolation samples from a 4x4 area will need more halo.
    constexpr int HALO = 1;
    // Full dimensions of the LDS tile, including the halo.
    constexpr int SMEM_H = TILE_H + 2 * HALO;
    constexpr int SMEM_W = TILE_W + 2 * HALO;
    // 1 more pixel to the inner dimension to avoid LDS bank conflicts.
    constexpr int SMEM_W_PAD = SMEM_W + 1;
    // Process channels in chunks of 4 to avoid using too much LDS.
    constexpr int CCHUNK = 4;

    // Dynamic shared memory layout: [CCHUNK][SMEM_H][SMEM_W_PAD]
    extern __shared__ opmath_t smem_raw[];
    // 3D array of [c_chunk_idx][y][x] to 1D smem_row indexing.
    auto smem_idx = [&](int c_chunk_idx, int y, int x) -> opmath_t& {
      return smem_raw[(c_chunk_idx * SMEM_H + y) * SMEM_W_PAD + x];
    };

    // `blockIdx.z` maps to the batch index of the 3D grid of blocks.
    const index_t n = blockIdx.z;
    const index_t inp_H = input.sizes[2];
    const index_t inp_W = input.sizes[3];
    const index_t out_H = grid.sizes[1];
    const index_t out_W = grid.sizes[2];

    // The top-left coord of the output tile the current block is working on.
    const index_t tile_h_out = blockIdx.y * TILE_H;
    const index_t tile_w_out = blockIdx.x * TILE_W;

    // Channel chunking based loop.
    // c0 is current starting index of the current channel chunk.
    for (index_t c0 = 0; c0 < input.sizes[1]; c0 += CCHUNK) {
      // The number of channels to process for the curretn iteration.
      // The last, partial chunk may be smaller than CCHUNK.
      const int cmax = min((int)CCHUNK, (int)(input.sizes[1] - c0));

      // Cooperatively zero out shared memory for the channel chunk.
      for (int c_idx = 0; c_idx < cmax; ++c_idx) {
        for (int i = threadIdx.y; i < SMEM_H; i += blockDim.y) {
          for (int j = threadIdx.x; j < SMEM_W_PAD; j += blockDim.x) {
            smem_idx(c_idx, i, j) = 0;
          }
        }
      }
      __syncthreads();

      // The unique global output coord for each thread to process.
      const index_t h_out = tile_h_out + threadIdx.y;
      const index_t w_out = tile_w_out + threadIdx.x;

      // Ensure the thread is within the bounds of the output grid.
      if (h_out < out_H && w_out < out_W) {
        const auto grid_offset = n * grid.strides[0] + h_out * grid.strides[1] + w_out * grid.strides[2];
        // The sampling coord (x, y) from the `grid` tensor.
        const opmath_t x = grid.data[grid_offset];
        const opmath_t y = grid.data[grid_offset + grid.strides[3]];

        // The gradient multipliers needed for the `grad_grid` calculation.
        opmath_t gix_mult, giy_mult;
        // The corresponding input coords.
        opmath_t ix = grid_sampler_compute_source_index_set_grad(x, inp_W, PADDING_MODE, ALIGN_CORNERS, &gix_mult);
        opmath_t iy = grid_sampler_compute_source_index_set_grad(y, inp_H, PADDING_MODE, ALIGN_CORNERS, &giy_mult);

        // Thread-local accumulators in registers for `grad_grid` not contended.
        opmath_t gix_agg = 0, giy_agg = 0;

        // NW anchor and fractional parts
        const index_t ix_nw = static_cast<index_t>(std::floor(ix));
        const index_t iy_nw = static_cast<index_t>(std::floor(iy));
        const opmath_t tx = ix - ix_nw;
        const opmath_t ty = iy - iy_nw;

        // NW, NE, SW, SE pixel values from (x, y)
        // The pure geometric bilinear interpolation weights
        const opmath_t nw = (1 - tx) * (1 - ty);
        const opmath_t ne = tx * (1 - ty);
        const opmath_t sw = (1 - tx) * ty;
        const opmath_t se = tx * ty;

        // The inner loop over the channels within the current chunk.
        for (int c_idx = 0; c_idx < cmax; ++c_idx) {
          const index_t c = c0 + c_idx;
          const opmath_t gOut = static_cast<opmath_t>(grad_output.data[n * grad_output.strides[0] + c * grad_output.strides[1] + h_out * grad_output.strides[2] + w_out * grad_output.strides[3]]);
          const scalar_t* inp_ptr_NC = input.data + n * input.strides[0] + c * input.strides[1];

          // Compile-time check for a performance win hopefully.
          // When INPUT_REQ_GRAD is false, the entire `if` block is removed by the compiler.
          if (INPUT_REQ_GRAD) {
            auto accumulate = [&](index_t iy_g, index_t ix_g, opmath_t weight) {
              // Global bounds check, same as the original impelmentation,
              // to check if the target input coord is valid.
              if (within_bounds_2d(iy_g, ix_g, inp_H, inp_W)) {
                // Map global input coord to local LDS coord.
                const index_t s_iy = iy_g - tile_h_out + HALO;
                const index_t s_ix = ix_g - tile_w_out + HALO;
                // Check if the local coord falls within the LDS tile with halo.
                if (s_iy >= 0 && s_iy < SMEM_H && s_ix >= 0 && s_ix < SMEM_W) {
                  // Core optimization: `atomicAdd` to LDS instead of global memory.
                  atomicAdd(reinterpret_cast<float*>(&smem_idx(c_idx, s_iy, s_ix)), static_cast<float>(weight * gOut));
                } else {
                  // Simply ignore the coord outside the tile.
                }
              }
            };
            accumulate(iy_nw, ix_nw, nw);
            accumulate(iy_nw, ix_nw + 1, ne);
            accumulate(iy_nw + 1, ix_nw, sw);
            accumulate(iy_nw + 1, ix_nw + 1, se);
          }

          // Accumulate grad_grid contributions in registers.
          scalar_t v_nw = get_value_bounded(inp_ptr_NC, static_cast<opmath_t>(ix_nw), static_cast<opmath_t>(iy_nw), inp_W, inp_H, input.strides[3], input.strides[2], PADDING_MODE, ALIGN_CORNERS);
          scalar_t v_ne = get_value_bounded(inp_ptr_NC, static_cast<opmath_t>(ix_nw + 1), static_cast<opmath_t>(iy_nw), inp_W, inp_H, input.strides[3], input.strides[2], PADDING_MODE, ALIGN_CORNERS);
          scalar_t v_sw = get_value_bounded(inp_ptr_NC, static_cast<opmath_t>(ix_nw), static_cast<opmath_t>(iy_nw + 1), inp_W, inp_H, input.strides[3], input.strides[2], PADDING_MODE, ALIGN_CORNERS);
          scalar_t v_se = get_value_bounded(inp_ptr_NC, static_cast<opmath_t>(ix_nw + 1), static_cast<opmath_t>(iy_nw + 1), inp_W, inp_H, input.strides[3], input.strides[2], PADDING_MODE, ALIGN_CORNERS);
          gix_agg += (v_ne - v_nw) * (1 - ty) * gOut + (v_se - v_sw) * ty * gOut;
          giy_agg += (v_sw - v_nw) * (1 - tx) * gOut + (v_se - v_ne) * tx * gOut;
        }

        scalar_t* gGrid_ptr_NHW = grad_grid.data + n * grad_grid.strides[0] + h_out * grad_grid.strides[1] + w_out * grad_grid.strides[2];
        gGrid_ptr_NHW[0] = static_cast<scalar_t>(gix_mult * gix_agg);
        gGrid_ptr_NHW[1] = static_cast<scalar_t>(giy_mult * giy_agg);
      }
      __syncthreads();

      if (INPUT_REQ_GRAD) {
        for (int c_idx = 0; c_idx < cmax; ++c_idx) {
          const index_t c = c0 + c_idx;
          const index_t NC_offset = n * grad_input.strides[0] + c * grad_input.strides[1];
          // Cooperatively write back from shared to global memory.
          for (int i = threadIdx.y; i < SMEM_H; i += blockDim.y) {
            for (int j = threadIdx.x; j < SMEM_W; j += blockDim.x) {
              opmath_t val = smem_idx(c_idx, i, j);
              if (val != 0) {
                // Back to the global input coord.
                index_t h_in = tile_h_out - HALO + i;
                index_t w_in = tile_w_out - HALO + j;
                // Atomics to global memory, whose number is reduced from
                // 4*TILE_W*TILE_H to just SMEM_H*SMEM_W per channel chunk.
                safe_add_2d(grad_input.data, h_in, w_in, grad_input.strides[2], grad_input.strides[3],
                            inp_H, inp_W, static_cast<scalar_t>(val), NC_offset, grad_input.numel());
              }
            }
          }
        }
      }
      __syncthreads();
    }
  }

  template <typename scalar_t, typename index_t>
  void launch_grid_sampler_2d_backward_dispatcher(
      const TensorBase &grad_input, const TensorBase &grad_grid,
      const TensorBase &grad_output, const TensorBase &input,
      const TensorBase &grid, GridSamplerInterpolation interpolation_mode,
      GridSamplerPadding padding_mode, bool align_corners,
      bool input_requires_grad) {

    if (interpolation_mode == GridSamplerInterpolation::Bilinear) {
      constexpr int TILE_W = 16;
      constexpr int TILE_H = 16;
      dim3 threads(TILE_W, TILE_H);
      dim3 blocks(
          ceil_div(grid.size(2), (long)TILE_W),
          ceil_div(grid.size(1), (long)TILE_H),
          grid.size(0));  // Launch per-N
      auto stream = at::cuda::getCurrentCUDAStream();

      // Calculate dynamic shared memory size
      constexpr int CCHUNK = 4;
      constexpr int HALO = 1;
      constexpr int SMEM_H = TILE_H + 2 * HALO;
      constexpr int SMEM_W = TILE_W + 2 * HALO;
      constexpr int SMEM_W_PAD = SMEM_W + 1;
      const size_t smem_bytes = CCHUNK * SMEM_H * SMEM_W_PAD * sizeof(at::opmath_type<scalar_t>);

      auto launch = [&](auto pad_mode, auto align_flag, auto req_grad_flag) {
        grid_sampler_2d_backward_kernel_optimized<scalar_t, index_t, GridSamplerInterpolation::Bilinear, pad_mode, align_flag, req_grad_flag>
          <<<blocks, threads, smem_bytes, stream>>>(
            getTensorInfo<const scalar_t, index_t>(grad_output), getTensorInfo<const scalar_t, index_t>(input), getTensorInfo<const scalar_t, index_t>(grid),
            getTensorInfo<scalar_t, index_t>(grad_input), getTensorInfo<scalar_t, index_t>(grad_grid));
      };

      // Dispatcher for template parameters
      if (padding_mode == GridSamplerPadding::Zeros) {
        if (align_corners) {
	        input_requires_grad ? launch(GridSamplerPadding::Zeros, true, true) : launch(GridSamplerPadding::Zeros, true, false);
	      } else {
	        input_requires_grad ? launch(GridSamplerPadding::Zeros, false, true) : launch(GridSamplerPadding::Zeros, false, false);
	      }
      } else {
        // Fallback for other padding modes to the original kernel
        const auto count = grid.size(0) * grid.size(1) * grid.size(2);
        grid_sampler_2d_backward_kernel<scalar_t><<<GET_BLOCKS(count, 256), 256, 0, stream>>>(
            count, getTensorInfo<const scalar_t, index_t>(grad_output), getTensorInfo<const scalar_t, index_t>(input),
            getTensorInfo<const scalar_t, index_t>(grid), input_requires_grad ? getTensorInfo<scalar_t, index_t>(grad_input) : TensorInfo<scalar_t, index_t>(),
            getTensorInfo<scalar_t, index_t>(grad_grid), interpolation_mode, padding_mode, align_corners,
            input_requires_grad ? grad_input.numel() : 0, input_requires_grad);
      }
    } else {
      // Fallback for Nearest and Bicubic modes to the original kernel
      const auto count = grid.size(0) * grid.size(1) * grid.size(2);
      auto stream = at::cuda::getCurrentCUDAStream();
      grid_sampler_2d_backward_kernel<scalar_t><<<GET_BLOCKS(count, 256), 256, 0, stream>>>(
          count, getTensorInfo<const scalar_t, index_t>(grad_output), getTensorInfo<const scalar_t, index_t>(input),
          getTensorInfo<const scalar_t, index_t>(grid), input_requires_grad ? getTensorInfo<scalar_t, index_t>(grad_input) : TensorInfo<scalar_t, index_t>(),
          getTensorInfo<scalar_t, index_t>(grad_grid), interpolation_mode, padding_mode, align_corners,
          input_requires_grad ? grad_input.numel() : 0, input_requires_grad);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
#endif  // USE_ROCM

  template <typename scalar_t, typename index_t>
  C10_LAUNCH_BOUNDS_1(256)
  __global__ void grid_sampler_2d_backward_kernel(
      const index_t nthreads,
      TensorInfo<const scalar_t, index_t> grad_output,
      TensorInfo<const scalar_t, index_t> input,
      TensorInfo<const scalar_t, index_t> grid,
      TensorInfo<scalar_t, index_t> grad_input,  // initialized to zeros (or unused if input_requires_grad is false)
      TensorInfo<scalar_t, index_t> grad_grid,   // initialized to empty
      const GridSamplerInterpolation interpolation_mode,
      const GridSamplerPadding padding_mode,
      bool align_corners,
      const index_t grad_input_memory_span,
      const bool input_requires_grad) {

    index_t C = input.sizes[1];
    index_t inp_H = input.sizes[2];
    index_t inp_W = input.sizes[3];
    index_t out_H = grid.sizes[1];
    index_t out_W = grid.sizes[2];
    index_t inp_sN = input.strides[0];
    index_t inp_sC = input.strides[1];
    index_t inp_sH = input.strides[2];
    index_t inp_sW = input.strides[3];
    index_t grid_sN = grid.strides[0];
    index_t grid_sH = grid.strides[1];
    index_t grid_sW = grid.strides[2];
    index_t grid_sCoor = grid.strides[3];
    index_t gOut_sN = grad_output.strides[0];
    index_t gOut_sC = grad_output.strides[1];
    index_t gOut_sH = grad_output.strides[2];
    index_t gOut_sW = grad_output.strides[3];
    // gInp_* (and NC_offset below) are not really needed if input_requires_grad is false.
    index_t gInp_sN;
    index_t gInp_sC;
    index_t gInp_sH;
    index_t gInp_sW;
    if (input_requires_grad) {
      gInp_sN = grad_input.strides[0];
      gInp_sC = grad_input.strides[1];
      gInp_sH = grad_input.strides[2];
      gInp_sW = grad_input.strides[3];
    }
    index_t gGrid_sW = grad_grid.strides[2];

    CUDA_KERNEL_LOOP_TYPE(index, nthreads, index_t) {
      const index_t w = index % out_W;
      const index_t h = (index / out_W) % out_H;
      const index_t n = index / (out_H * out_W);
      const auto grid_offset = n * grid_sN + h * grid_sH + w * grid_sW;

      // get the corresponding input x, y co-ordinates from grid
      scalar_t x = grid.data[grid_offset];
      scalar_t y = grid.data[grid_offset + grid_sCoor];

      // multipliers for gradients on ix and iy
      scalar_t gix_mult, giy_mult;
      scalar_t ix = grid_sampler_compute_source_index_set_grad(x, inp_W, padding_mode, align_corners, &gix_mult);
      scalar_t iy = grid_sampler_compute_source_index_set_grad(y, inp_H, padding_mode, align_corners, &giy_mult);

      if (interpolation_mode == GridSamplerInterpolation::Bilinear) {
        // get NE, NW, SE, SW pixel values from (x, y)
        index_t ix_nw = static_cast<index_t>(std::floor(ix));
        index_t iy_nw = static_cast<index_t>(std::floor(iy));
        index_t ix_ne = ix_nw + 1;
        index_t iy_ne = iy_nw;
        index_t ix_sw = ix_nw;
        index_t iy_sw = iy_nw + 1;
        index_t ix_se = ix_nw + 1;
        index_t iy_se = iy_nw + 1;

        // get surfaces to each neighbor:
        scalar_t nw = (ix_se - ix)    * (iy_se - iy);
        scalar_t ne = (ix    - ix_sw) * (iy_sw - iy);
        scalar_t sw = (ix_ne - ix)    * (iy    - iy_ne);
        scalar_t se = (ix    - ix_nw) * (iy    - iy_nw);

        scalar_t gix = static_cast<scalar_t>(0), giy = static_cast<scalar_t>(0);
        const scalar_t *gOut_ptr_NCHW = grad_output.data + n * gOut_sN + h * gOut_sH + w * gOut_sW;
        index_t NC_offset = n * gInp_sN;
        const scalar_t *inp_ptr_NC = input.data + n * inp_sN;
        for (index_t c = 0; c < C; ++c, inp_ptr_NC += inp_sC, NC_offset += gInp_sC, gOut_ptr_NCHW += gOut_sC) {
          const scalar_t gOut = *gOut_ptr_NCHW;

          if (input_requires_grad) {
            // calculate and set grad_input. See Note [Passing pointer and offset to fastAtomicAdd].
            safe_add_2d(grad_input.data, iy_nw, ix_nw, gInp_sH, gInp_sW, inp_H, inp_W, nw * gOut, NC_offset, grad_input_memory_span);
            safe_add_2d(grad_input.data, iy_ne, ix_ne, gInp_sH, gInp_sW, inp_H, inp_W, ne * gOut, NC_offset, grad_input_memory_span);
            safe_add_2d(grad_input.data, iy_sw, ix_sw, gInp_sH, gInp_sW, inp_H, inp_W, sw * gOut, NC_offset, grad_input_memory_span);
            safe_add_2d(grad_input.data, iy_se, ix_se, gInp_sH, gInp_sW, inp_H, inp_W, se * gOut, NC_offset, grad_input_memory_span);
          }

          // calculate grad_grid
          if (within_bounds_2d(iy_nw, ix_nw, inp_H, inp_W)) {
            scalar_t nw_val = inp_ptr_NC[iy_nw * inp_sH + ix_nw * inp_sW];
            gix -= nw_val * (iy_se - iy) * gOut;
            giy -= nw_val * (ix_se - ix) * gOut;
          }
          if (within_bounds_2d(iy_ne, ix_ne, inp_H, inp_W)) {
            scalar_t ne_val = inp_ptr_NC[iy_ne * inp_sH + ix_ne * inp_sW];
            gix += ne_val * (iy_sw - iy) * gOut;
            giy -= ne_val * (ix - ix_sw) * gOut;
          }
          if (within_bounds_2d(iy_sw, ix_sw, inp_H, inp_W)) {
            scalar_t sw_val = inp_ptr_NC[iy_sw * inp_sH + ix_sw * inp_sW];
            gix -= sw_val * (iy - iy_ne) * gOut;
            giy += sw_val * (ix_ne - ix) * gOut;
          }
          if (within_bounds_2d(iy_se, ix_se, inp_H, inp_W)) {
            scalar_t se_val = inp_ptr_NC[iy_se * inp_sH + ix_se * inp_sW];
            gix += se_val * (iy - iy_nw) * gOut;
            giy += se_val * (ix - ix_nw) * gOut;
          }
        }

        // assuming grad_grid is contiguous
        // thus we can
        //   1. use index with gGrid_sW to directly compute gGrid_ptr_NHW
        //   2. directly assign to gGrid_ptr_NHW[0], gGrid_ptr_NHW[1]
        scalar_t *gGrid_ptr_NHW = grad_grid.data + index * gGrid_sW;
        gGrid_ptr_NHW[0] = gix_mult * gix;
        gGrid_ptr_NHW[1] = giy_mult * giy;
      } else if (interpolation_mode == GridSamplerInterpolation::Nearest) {
        if (input_requires_grad) {
          index_t ix_nearest = static_cast<index_t>(std::nearbyint(ix));
          index_t iy_nearest = static_cast<index_t>(std::nearbyint(iy));

          // assign nearest neighbour pixel value to output pixel
          const scalar_t *gOut_ptr_NCHW = grad_output.data + n * gOut_sN + h * gOut_sH + w * gOut_sW;
          index_t NC_offset = n * gInp_sN;
          for (index_t c = 0; c < C; ++c, NC_offset += gInp_sC, gOut_ptr_NCHW += gOut_sC) {
            // calculate and set grad_input. See Note [Passing pointer and offset to fastAtomicAdd].
            safe_add_2d(grad_input.data, iy_nearest, ix_nearest, gInp_sH, gInp_sW, inp_H, inp_W, *gOut_ptr_NCHW, NC_offset, grad_input_memory_span);
          }
        }

        // assuming grad_grid is contiguous
        // thus we can
        //   1. use index with gGrid_sW to directly compute gGrid_ptr_NHW
        //   2. directly assign to gGrid_ptr_NHW[0], gGrid_ptr_NHW[1]
        scalar_t *gGrid_ptr_NHW = grad_grid.data + index * gGrid_sW;
        gGrid_ptr_NHW[0] = static_cast<scalar_t>(0);
        gGrid_ptr_NHW[1] = static_cast<scalar_t>(0);
      } else if (interpolation_mode == GridSamplerInterpolation::Bicubic) {

        ix = grid_sampler_unnormalize_set_grad(x, inp_W, align_corners, &gix_mult);
        iy = grid_sampler_unnormalize_set_grad(y, inp_H, align_corners, &giy_mult);

        scalar_t ix_nw = std::floor(ix);
        scalar_t iy_nw = std::floor(iy);

        const scalar_t tx = ix - ix_nw;
        const scalar_t ty = iy - iy_nw;

        scalar_t x_coeffs[4];
        scalar_t y_coeffs[4];
        scalar_t x_coeffs_grad[4];
        scalar_t y_coeffs_grad[4];

        get_cubic_upsampling_coefficients<scalar_t>(x_coeffs, tx);
        get_cubic_upsampling_coefficients<scalar_t>(y_coeffs, ty);
        get_cubic_coefficients_grad<scalar_t>(x_coeffs_grad, tx);
        get_cubic_coefficients_grad<scalar_t>(y_coeffs_grad, ty);

        scalar_t gix = static_cast<scalar_t>(0);
        scalar_t giy = static_cast<scalar_t>(0);

        const scalar_t *gOut_ptr_NCHW = grad_output.data + n * gOut_sN + h * gOut_sH + w * gOut_sW;
        index_t NC_offset = n * gInp_sN;
        const scalar_t *inp_ptr_NC = input.data + n * inp_sN;

        for (index_t c = 0; c < C; ++c, gOut_ptr_NCHW += gOut_sC, NC_offset += gInp_sC, inp_ptr_NC+= inp_sC) {
          const scalar_t gOut = *gOut_ptr_NCHW;

          #pragma unroll 4
          for (index_t i = 0; i < 4; ++i) {
            #pragma unroll 4
            for (index_t j = 0; j < 4; ++j) {

              if (input_requires_grad) {
                // set input gradient. See Note [Passing pointer and offset to fastAtomicAdd].
                add_value_bounded<scalar_t>(grad_input.data, ix_nw - 1 + i, iy_nw - 1 + j, inp_W, inp_H, gInp_sW, gInp_sH,
                  gOut * x_coeffs[i] * y_coeffs[j],
                  padding_mode,
                  align_corners,
                  NC_offset,
                  grad_input_memory_span);
              }

              // set grid gradient
              scalar_t val = get_value_bounded<scalar_t>(inp_ptr_NC, ix_nw - 1 + i, iy_nw - 1 + j,
                inp_W, inp_H, inp_sW, inp_sH, padding_mode, align_corners);

              gix -= val * x_coeffs_grad[i] * y_coeffs[j] * gOut;
              giy -= val * y_coeffs_grad[j] * x_coeffs[i] * gOut;
            }
          }
        }

        scalar_t *gGrid_ptr_NHW = grad_grid.data + index * gGrid_sW;
        gGrid_ptr_NHW[0] = gix_mult * gix;
        gGrid_ptr_NHW[1] = giy_mult * giy;
      }
    }
  }

  template <typename scalar_t, typename index_t>
  C10_LAUNCH_BOUNDS_1(256)
  __global__ void grid_sampler_3d_backward_kernel(
      const index_t nthreads,
      TensorInfo<const scalar_t, index_t> grad_output,
      TensorInfo<const scalar_t, index_t> input,
      TensorInfo<const scalar_t, index_t> grid,
      TensorInfo<scalar_t, index_t> grad_input,  // initialized to zeros (or unused if input_requires_grad is false)
      TensorInfo<scalar_t, index_t> grad_grid,   // initialized to empty
      const GridSamplerInterpolation interpolation_mode,
      const GridSamplerPadding padding_mode,
      bool align_corners,
      const index_t grad_input_memory_span,
      const bool input_requires_grad) {

    index_t C = input.sizes[1];
    index_t inp_D = input.sizes[2];
    index_t inp_H = input.sizes[3];
    index_t inp_W = input.sizes[4];
    index_t out_D = grid.sizes[1];
    index_t out_H = grid.sizes[2];
    index_t out_W = grid.sizes[3];
    index_t inp_sN = input.strides[0];
    index_t inp_sC = input.strides[1];
    index_t inp_sD = input.strides[2];
    index_t inp_sH = input.strides[3];
    index_t inp_sW = input.strides[4];
    index_t grid_sN = grid.strides[0];
    index_t grid_sD = grid.strides[1];
    index_t grid_sH = grid.strides[2];
    index_t grid_sW = grid.strides[3];
    index_t grid_sCoor = grid.strides[4];
    index_t gOut_sN = grad_output.strides[0];
    index_t gOut_sC = grad_output.strides[1];
    index_t gOut_sD = grad_output.strides[2];
    index_t gOut_sH = grad_output.strides[3];
    index_t gOut_sW = grad_output.strides[4];
    // gInp_* (and NC_offset below) are not really needed if input_requires_grad is false.
    int64_t gInp_sN = 0;
    int64_t gInp_sC = 0;
    int64_t gInp_sD = 0;
    int64_t gInp_sH = 0;
    int64_t gInp_sW = 0;
    if (input_requires_grad) {
      gInp_sN = grad_input.strides[0];
      gInp_sC = grad_input.strides[1];
      gInp_sD = grad_input.strides[2];
      gInp_sH = grad_input.strides[3];
      gInp_sW = grad_input.strides[4];
    }
    index_t gGrid_sW = grad_grid.strides[3];

    CUDA_KERNEL_LOOP_TYPE(index, nthreads, index_t) {
      const index_t w = index % out_W;
      const index_t h = (index / out_W) % out_H;
      const index_t d = (index / (out_H * out_W)) % out_D;
      const index_t n = index / (out_D * out_H * out_W);
      const auto grid_offset = n * grid_sN + d * grid_sD + h * grid_sH + w * grid_sW;

      // get the corresponding input x, y, z co-ordinates from grid
      scalar_t ix = grid.data[grid_offset];
      scalar_t iy = grid.data[grid_offset + grid_sCoor];
      scalar_t iz = grid.data[grid_offset + 2 * grid_sCoor];

      // multipliers for gradients on ix, iy, and iz
      scalar_t gix_mult, giy_mult, giz_mult;
      ix = grid_sampler_compute_source_index_set_grad(ix, inp_W, padding_mode, align_corners, &gix_mult);
      iy = grid_sampler_compute_source_index_set_grad(iy, inp_H, padding_mode, align_corners, &giy_mult);
      iz = grid_sampler_compute_source_index_set_grad(iz, inp_D, padding_mode, align_corners, &giz_mult);

      if (interpolation_mode == GridSamplerInterpolation::Bilinear) {
        // get corner pixel values from (x, y, z)
        // for 4d, we used north-east-south-west
        // for 5d, we add top-bottom
        index_t ix_tnw = static_cast<index_t>(std::floor(ix));
        index_t iy_tnw = static_cast<index_t>(std::floor(iy));
        index_t iz_tnw = static_cast<index_t>(std::floor(iz));

        index_t ix_tne = ix_tnw + 1;
        index_t iy_tne = iy_tnw;
        index_t iz_tne = iz_tnw;

        index_t ix_tsw = ix_tnw;
        index_t iy_tsw = iy_tnw + 1;
        index_t iz_tsw = iz_tnw;

        index_t ix_tse = ix_tnw + 1;
        index_t iy_tse = iy_tnw + 1;
        index_t iz_tse = iz_tnw;

        index_t ix_bnw = ix_tnw;
        index_t iy_bnw = iy_tnw;
        index_t iz_bnw = iz_tnw + 1;

        index_t ix_bne = ix_tnw + 1;
        index_t iy_bne = iy_tnw;
        index_t iz_bne = iz_tnw + 1;

        index_t ix_bsw = ix_tnw;
        index_t iy_bsw = iy_tnw + 1;
        index_t iz_bsw = iz_tnw + 1;

        index_t ix_bse = ix_tnw + 1;
        index_t iy_bse = iy_tnw + 1;
        index_t iz_bse = iz_tnw + 1;

        // get surfaces to each neighbor:
        scalar_t tnw = (ix_bse - ix)    * (iy_bse - iy)    * (iz_bse - iz);
        scalar_t tne = (ix    - ix_bsw) * (iy_bsw - iy)    * (iz_bsw - iz);
        scalar_t tsw = (ix_bne - ix)    * (iy    - iy_bne) * (iz_bne - iz);
        scalar_t tse = (ix    - ix_bnw) * (iy    - iy_bnw) * (iz_bnw - iz);
        scalar_t bnw = (ix_tse - ix)    * (iy_tse - iy)    * (iz - iz_tse);
        scalar_t bne = (ix    - ix_tsw) * (iy_tsw - iy)    * (iz - iz_tsw);
        scalar_t bsw = (ix_tne - ix)    * (iy    - iy_tne) * (iz - iz_tne);
        scalar_t bse = (ix    - ix_tnw) * (iy    - iy_tnw) * (iz - iz_tnw);

        scalar_t gix = static_cast<scalar_t>(0), giy = static_cast<scalar_t>(0), giz = static_cast<scalar_t>(0);
        const scalar_t *gOut_ptr_NCDHW = grad_output.data + n * gOut_sN + d * gOut_sD + h * gOut_sH + w * gOut_sW;
        index_t NC_offset;
        if (input_requires_grad) {
          NC_offset = n * gInp_sN;
        }
        const scalar_t *inp_ptr_NC = input.data + n * inp_sN;
        // calculate bilinear weighted pixel value and set output pixel
        for (index_t c = 0; c < C; ++c, gOut_ptr_NCDHW += gOut_sC, NC_offset += gInp_sC, inp_ptr_NC += inp_sC) {
          scalar_t gOut = *gOut_ptr_NCDHW;

          // calculate and set grad_input. See Note [Passing pointer and offset to fastAtomicAdd].
          if (input_requires_grad) {
            safe_add_3d(grad_input.data, iz_tnw, iy_tnw, ix_tnw, gInp_sD, gInp_sH, gInp_sW, inp_D, inp_H, inp_W, tnw * gOut,
                        NC_offset, grad_input_memory_span);
            safe_add_3d(grad_input.data, iz_tne, iy_tne, ix_tne, gInp_sD, gInp_sH, gInp_sW, inp_D, inp_H, inp_W, tne * gOut,
                        NC_offset, grad_input_memory_span);
            safe_add_3d(grad_input.data, iz_tsw, iy_tsw, ix_tsw, gInp_sD, gInp_sH, gInp_sW, inp_D, inp_H, inp_W, tsw * gOut,
                        NC_offset, grad_input_memory_span);
            safe_add_3d(grad_input.data, iz_tse, iy_tse, ix_tse, gInp_sD, gInp_sH, gInp_sW, inp_D, inp_H, inp_W, tse * gOut,
                        NC_offset, grad_input_memory_span);
            safe_add_3d(grad_input.data, iz_bnw, iy_bnw, ix_bnw, gInp_sD, gInp_sH, gInp_sW, inp_D, inp_H, inp_W, bnw * gOut,
                        NC_offset, grad_input_memory_span);
            safe_add_3d(grad_input.data, iz_bne, iy_bne, ix_bne, gInp_sD, gInp_sH, gInp_sW, inp_D, inp_H, inp_W, bne * gOut,
                        NC_offset, grad_input_memory_span);
            safe_add_3d(grad_input.data, iz_bsw, iy_bsw, ix_bsw, gInp_sD, gInp_sH, gInp_sW, inp_D, inp_H, inp_W, bsw * gOut,
                        NC_offset, grad_input_memory_span);
            safe_add_3d(grad_input.data, iz_bse, iy_bse, ix_bse, gInp_sD, gInp_sH, gInp_sW, inp_D, inp_H, inp_W, bse * gOut,
                        NC_offset, grad_input_memory_span);
          }
          // calculate grad_grid
          if (within_bounds_3d(iz_tnw, iy_tnw, ix_tnw, inp_D, inp_H, inp_W)) {
            scalar_t tnw_val = inp_ptr_NC[iz_tnw * inp_sD + iy_tnw * inp_sH + ix_tnw * inp_sW];
            gix -= tnw_val * (iy_bse - iy)    * (iz_bse - iz)    * gOut;
            giy -= tnw_val * (ix_bse - ix)    * (iz_bse - iz)    * gOut;
            giz -= tnw_val * (ix_bse - ix)    * (iy_bse - iy)    * gOut;
          }
          if (within_bounds_3d(iz_tne, iy_tne, ix_tne, inp_D, inp_H, inp_W)) {
            scalar_t tne_val = inp_ptr_NC[iz_tne * inp_sD + iy_tne * inp_sH + ix_tne * inp_sW];
            gix += tne_val * (iy_bsw - iy)    * (iz_bsw - iz)    * gOut;
            giy -= tne_val * (ix    - ix_bsw) * (iz_bsw - iz)    * gOut;
            giz -= tne_val * (ix    - ix_bsw) * (iy_bsw - iy)    * gOut;
          }
          if (within_bounds_3d(iz_tsw, iy_tsw, ix_tsw, inp_D, inp_H, inp_W)) {
            scalar_t tsw_val = inp_ptr_NC[iz_tsw * inp_sD + iy_tsw * inp_sH + ix_tsw * inp_sW];
            gix -= tsw_val * (iy - iy_bne)    * (iz_bne - iz)    * gOut;
            giy += tsw_val * (ix_bne - ix)    * (iz_bne - iz)    * gOut;
            giz -= tsw_val * (ix_bne - ix)    * (iy    - iy_bne) * gOut;
          }
          if (within_bounds_3d(iz_tse, iy_tse, ix_tse, inp_D, inp_H, inp_W)) {
            scalar_t tse_val = inp_ptr_NC[iz_tse * inp_sD + iy_tse * inp_sH + ix_tse * inp_sW];
            gix += tse_val * (iy - iy_bnw)    * (iz_bnw - iz)    * gOut;
            giy += tse_val * (ix    - ix_bnw) * (iz_bnw - iz)    * gOut;
            giz -= tse_val * (ix    - ix_bnw) * (iy    - iy_bnw) * gOut;
          }
          if (within_bounds_3d(iz_bnw, iy_bnw, ix_bnw, inp_D, inp_H, inp_W)) {
            scalar_t bnw_val = inp_ptr_NC[iz_bnw * inp_sD + iy_bnw * inp_sH + ix_bnw * inp_sW];
            gix -= bnw_val * (iy_tse - iy)    * (iz - iz_tse)    * gOut;
            giy -= bnw_val * (ix_tse - ix)    * (iz - iz_tse)    * gOut;
            giz += bnw_val * (ix_tse - ix)    * (iy_tse - iy)    * gOut;
          }
          if (within_bounds_3d(iz_bne, iy_bne, ix_bne, inp_D, inp_H, inp_W)) {
            scalar_t bne_val = inp_ptr_NC[iz_bne * inp_sD + iy_bne * inp_sH + ix_bne * inp_sW];
            gix += bne_val * (iy_tsw - iy)    * (iz - iz_tsw)    * gOut;
            giy -= bne_val * (ix    - ix_tsw) * (iz - iz_tsw)    * gOut;
            giz += bne_val * (ix    - ix_tsw) * (iy_tsw - iy)    * gOut;
          }
          if (within_bounds_3d(iz_bsw, iy_bsw, ix_bsw, inp_D, inp_H, inp_W)) {
            scalar_t bsw_val = inp_ptr_NC[iz_bsw * inp_sD + iy_bsw * inp_sH + ix_bsw * inp_sW];
            gix -= bsw_val * (iy - iy_tne)    * (iz - iz_tne)    * gOut;
            giy += bsw_val * (ix_tne - ix)    * (iz - iz_tne)    * gOut;
            giz += bsw_val * (ix_tne - ix)    * (iy    - iy_tne) * gOut;
          }
          if (within_bounds_3d(iz_bse, iy_bse, ix_bse, inp_D, inp_H, inp_W)) {
            scalar_t bse_val = inp_ptr_NC[iz_bse * inp_sD + iy_bse * inp_sH + ix_bse * inp_sW];
            gix += bse_val * (iy - iy_tnw)    * (iz - iz_tnw)    * gOut;
            giy += bse_val * (ix    - ix_tnw) * (iz - iz_tnw)    * gOut;
            giz += bse_val * (ix    - ix_tnw) * (iy    - iy_tnw) * gOut;
          }
        }

        // assuming grad_grid is contiguous
        // thus we can
        //   1. use index with gGrid_sW to directly compute gGrid_ptr_NDHW
        //   2. directly assign to gGrid_ptr_NDHW[0], gGrid_ptr_NDHW[1], gGrid_ptr_NDHW[2]
        scalar_t *gGrid_ptr_NDHW = grad_grid.data + index * gGrid_sW;
        gGrid_ptr_NDHW[0] = gix_mult * gix;
        gGrid_ptr_NDHW[1] = giy_mult * giy;
        gGrid_ptr_NDHW[2] = giz_mult * giz;
      } else if (interpolation_mode == GridSamplerInterpolation::Nearest) {
        if (input_requires_grad) {
          auto ix_nearest = static_cast<index_t>(std::nearbyint(ix));
          auto iy_nearest = static_cast<index_t>(std::nearbyint(iy));
          auto iz_nearest = static_cast<index_t>(std::nearbyint(iz));

          // assign nearest neighbour pixel value to output pixel
          const scalar_t *gOut_ptr_NCDHW = grad_output.data + n * gOut_sN + d * gOut_sD + h * gOut_sH + w * gOut_sW;
          index_t NC_offset = n * gInp_sN;
          for (index_t c = 0; c < C; ++c, gOut_ptr_NCDHW += gOut_sC, NC_offset += gInp_sC) {
            // calculate and set grad_input. See Note [Passing pointer and offset to fastAtomicAdd].
            safe_add_3d(grad_input.data, iz_nearest, iy_nearest, ix_nearest,
                        gInp_sD, gInp_sH, gInp_sW, inp_D, inp_H, inp_W, *gOut_ptr_NCDHW,
                        NC_offset, grad_input_memory_span);
          }
        }
        // assuming grad_grid is contiguous
        // thus we can
        //   1. use index with gGrid_sW to directly compute gGrid_ptr_NDHW
        //   2. directly assign to gGrid_ptr_NDHW[0], gGrid_ptr_NDHW[1], gGrid_ptr_NDHW[2]
        scalar_t *gGrid_ptr_NDHW = grad_grid.data + index * gGrid_sW;
        gGrid_ptr_NDHW[0] = static_cast<scalar_t>(0);
        gGrid_ptr_NDHW[1] = static_cast<scalar_t>(0);
        gGrid_ptr_NDHW[2] = static_cast<scalar_t>(0);
      }
    }
  }
}  // namespace

void launch_grid_sampler_2d_forward_kernel(
    const TensorBase &output, const TensorBase &input, const TensorBase &grid,
    int64_t interpolation_mode, int64_t padding_mode, bool align_corners) {
  // See NOTE [ grid_sampler Native Functions ].
  // Add checks here in case this is called instead of grid_sampler.
  check_grid_sampler_common(input, grid);
  check_grid_sampler_2d(input, grid);

  auto N = input.size(0);
  auto H = grid.size(1);
  auto W = grid.size(2);
  int64_t count = N * H * W;
  if (count > 0) {
    AT_DISPATCH_FLOATING_TYPES_AND2(
      ScalarType::Half, ScalarType::BFloat16,
      input.scalar_type(), "grid_sampler_2d_cuda", [&] {
      if (canUse32BitIndexMath(input) && canUse32BitIndexMath(grid) &&
          canUse32BitIndexMath(output)) {
        grid_sampler_2d_kernel<scalar_t>
          <<<GET_BLOCKS(count, 256), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
            static_cast<int>(count),
            getTensorInfo<const scalar_t, int>(input),
            getTensorInfo<const scalar_t, int>(grid),
            getTensorInfo<scalar_t, int>(output),
            static_cast<GridSamplerInterpolation>(interpolation_mode),
            static_cast<GridSamplerPadding>(padding_mode),
            align_corners);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      } else {
        grid_sampler_2d_kernel<scalar_t>
          <<<GET_BLOCKS(count, 256), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
            count,
            getTensorInfo<const scalar_t, int64_t>(input),
            getTensorInfo<const scalar_t, int64_t>(grid),
            getTensorInfo<scalar_t, int64_t>(output),
            static_cast<GridSamplerInterpolation>(interpolation_mode),
            static_cast<GridSamplerPadding>(padding_mode),
            align_corners);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      }
    });
  }
}

void launch_grid_sampler_3d_forward_kernel(
    const TensorBase &output, const TensorBase &input, const TensorBase &grid,
    int64_t interpolation_mode, int64_t padding_mode, bool align_corners) {
  // See NOTE [ grid_sampler Native Functions ].
  // Add checks here in case this is called instead of grid_sampler.
  check_grid_sampler_common(input, grid);
  check_grid_sampler_3d(input, grid, interpolation_mode);

  auto N = input.size(0);
  auto D = grid.size(1);
  auto H = grid.size(2);
  auto W = grid.size(3);
  int64_t count = N * D * H * W;
  if (count > 0) {
    AT_DISPATCH_FLOATING_TYPES_AND2(
      ScalarType::Half, ScalarType::BFloat16,
      input.scalar_type(), "grid_sampler_3d_cuda", [&] {
      if (canUse32BitIndexMath(input) && canUse32BitIndexMath(grid) &&
          canUse32BitIndexMath(output)) {
        grid_sampler_3d_kernel<scalar_t>
          <<<GET_BLOCKS(count, 512), 512, 0, at::cuda::getCurrentCUDAStream()>>>(
            static_cast<int>(count),
            getTensorInfo<const scalar_t, int>(input),
            getTensorInfo<const scalar_t, int>(grid),
            getTensorInfo<scalar_t, int>(output),
            static_cast<GridSamplerInterpolation>(interpolation_mode),
            static_cast<GridSamplerPadding>(padding_mode),
            align_corners);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      } else {
        grid_sampler_3d_kernel<scalar_t>
          <<<GET_BLOCKS(count, 512), 512, 0, at::cuda::getCurrentCUDAStream()>>>(
            count,
            getTensorInfo<const scalar_t, int64_t>(input),
            getTensorInfo<const scalar_t, int64_t>(grid),
            getTensorInfo<scalar_t, int64_t>(output),
            static_cast<GridSamplerInterpolation>(interpolation_mode),
            static_cast<GridSamplerPadding>(padding_mode),
            align_corners);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      }
    });
  }
}

void launch_grid_sampler_2d_backward_kernel(
    const TensorBase &grad_input, const TensorBase &grad_grid,
    const TensorBase &grad_output, const TensorBase &input,
    const TensorBase &grid, int64_t interpolation_mode, int64_t padding_mode,
    bool align_corners, std::array<bool, 2> output_mask) {
  // See NOTE [ grid_sampler Native Functions ].
  // Add checks here in case this is called instead of grid_sampler.
  check_grid_sampler_common(input, grid);
  check_grid_sampler_2d(input, grid);

  // See Note [Writing Nondeterministic Operations]
  // Nondeterministic because of atomicAdd usage in the underlying kernel:
  // When multiple threads try to add to the same memory location at once,
  // the order in which they add is not guaranteed, which can lead to
  // tiny, bit-level differences in the final result across different runs.
  globalContext().alertNotDeterministic("grid_sampler_2d_backward_cuda");

#ifdef USE_ROCM
  auto input_requires_grad = output_mask[0];
  // Neither grad_input nor grad_grid is required.
  if (!output_mask[0] && !output_mask[1]) {
    return;
  }
  AT_DISPATCH_FLOATING_TYPES_AND2(
    ScalarType::Half, ScalarType::BFloat16,
    input.scalar_type(), "grid_sampler_2d_backward_cuda", [&] {
    // Small enough for 32-bit integer.
    if (canUse32BitIndexMath(input) && canUse32BitIndexMath(grid) && canUse32BitIndexMath(grad_output)) {
      launch_grid_sampler_2d_backward_dispatcher<scalar_t, int>(
          grad_input, grad_grid, grad_output, input, grid,
          static_cast<GridSamplerInterpolation>(interpolation_mode),
          static_cast<GridSamplerPadding>(padding_mode),
          align_corners, input_requires_grad);
    } else {
      launch_grid_sampler_2d_backward_dispatcher<scalar_t, int64_t>(
          grad_input, grad_grid, grad_output, input, grid,
          static_cast<GridSamplerInterpolation>(interpolation_mode),
          static_cast<GridSamplerPadding>(padding_mode),
          align_corners, input_requires_grad);
    }
  });
#else  // CUDA Path
  auto N = input.size(0);
  auto H = grid.size(1);
  auto W = grid.size(2);

  // If `input` gradient is not required, we skip computing it -- not needing to create
  // the tensor to hold the gradient can markedly increase performance. (`grid` gradient
  // is always computed.)
  auto input_requires_grad = output_mask[0];

  int64_t count = N * H * W;
  if (count > 0) {
    AT_DISPATCH_FLOATING_TYPES_AND2(
      ScalarType::Half, ScalarType::BFloat16,
      input.scalar_type(), "grid_sampler_2d_backward_cuda", [&] {
      if (canUse32BitIndexMath(input) && canUse32BitIndexMath(grid) &&
          canUse32BitIndexMath(grad_output)) {
        grid_sampler_2d_backward_kernel<scalar_t>
          <<<GET_BLOCKS(count, 256), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
            static_cast<int>(count),
            getTensorInfo<const scalar_t, int>(grad_output),
            getTensorInfo<const scalar_t, int>(input),
            getTensorInfo<const scalar_t, int>(grid),
            input_requires_grad ? getTensorInfo<scalar_t, int>(grad_input) : TensorInfo<scalar_t, int>(),
            getTensorInfo<scalar_t, int>(grad_grid),
            static_cast<GridSamplerInterpolation>(interpolation_mode),
            static_cast<GridSamplerPadding>(padding_mode),
            align_corners,
            /*grad_input_memory_span =*/input_requires_grad ? static_cast<int>(grad_input.numel()) : 0,
            input_requires_grad);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      } else {
        grid_sampler_2d_backward_kernel<scalar_t>
          <<<GET_BLOCKS(count, 256), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
            count,
            getTensorInfo<const scalar_t, int64_t>(grad_output),
            getTensorInfo<const scalar_t, int64_t>(input),
            getTensorInfo<const scalar_t, int64_t>(grid),
            input_requires_grad ? getTensorInfo<scalar_t, int64_t>(grad_input) : TensorInfo<scalar_t, int64_t>(),
            getTensorInfo<scalar_t, int64_t>(grad_grid),
            static_cast<GridSamplerInterpolation>(interpolation_mode),
            static_cast<GridSamplerPadding>(padding_mode),
            align_corners,
            /*grad_input_memory_span =*/input_requires_grad ? grad_input.numel() : 0,
            input_requires_grad);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      }
    });
  }
#endif  // USE_ROCM
}

void launch_grid_sampler_3d_backward_kernel(
    const TensorBase &grad_input, const TensorBase &grad_grid,
    const TensorBase& grad_output, const TensorBase& input,
    const TensorBase& grid, int64_t interpolation_mode, int64_t padding_mode,
    bool align_corners, std::array<bool,2> output_mask) {
  // See NOTE [ grid_sampler Native Functions ].
  // Add checks here in case this is called instead of grid_sampler.
  check_grid_sampler_common(input, grid);
  check_grid_sampler_3d(input, grid, interpolation_mode);

  // See Note [Writing Nondeterministic Operations]
  // Nondeterministic because of atomicAdd usage
  globalContext().alertNotDeterministic("grid_sampler_3d_backward_cuda");
  auto N = input.size(0);
  auto D = grid.size(1);
  auto H = grid.size(2);
  auto W = grid.size(3);
  int64_t count = N * D * H * W;
  auto input_requires_grad = output_mask[0];
  if (count > 0) {
    AT_DISPATCH_FLOATING_TYPES_AND2(
      ScalarType::Half, ScalarType::BFloat16,
      input.scalar_type(), "grid_sampler_3d_backward_cuda", [&] {
      if (canUse32BitIndexMath(input) && canUse32BitIndexMath(grid) &&
          canUse32BitIndexMath(grad_output)) {
        grid_sampler_3d_backward_kernel<scalar_t>
          <<<GET_BLOCKS(count, 256), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
            static_cast<int>(count),
            getTensorInfo<const scalar_t, int>(grad_output),
            getTensorInfo<const scalar_t, int>(input),
            getTensorInfo<const scalar_t, int>(grid),
            input_requires_grad ? getTensorInfo<scalar_t, int>(grad_input) : TensorInfo<scalar_t, int>(),
            getTensorInfo<scalar_t, int>(grad_grid),
            static_cast<GridSamplerInterpolation>(interpolation_mode),
            static_cast<GridSamplerPadding>(padding_mode),
            align_corners,
            /*grad_input_memory_span =*/input_requires_grad ? static_cast<int>(grad_input.numel()) : 0,
            input_requires_grad);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      } else {
        grid_sampler_3d_backward_kernel<scalar_t>
          <<<GET_BLOCKS(count, 256), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
            count,
            getTensorInfo<const scalar_t, int64_t>(grad_output),
            getTensorInfo<const scalar_t, int64_t>(input),
            getTensorInfo<const scalar_t, int64_t>(grid),
            input_requires_grad ? getTensorInfo<scalar_t, int64_t>(grad_input) : TensorInfo<scalar_t, int64_t>(),
            getTensorInfo<scalar_t, int64_t>(grad_grid),
            static_cast<GridSamplerInterpolation>(interpolation_mode),
            static_cast<GridSamplerPadding>(padding_mode),
            align_corners,
            /*grad_input_memory_span =*/input_requires_grad ? grad_input.numel() : 0,
            input_requires_grad);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      }
    });
  }
}

}  // namespace at::native
