#include "ATen/ATen.h"
#include "ATen/native/GridSampler.h"
#include "ATen/cuda/CUDAContext.h"
#include "ATen/cuda/CUDAApplyUtils.cuh"
#include "ATen/cuda/detail/TensorInfo.cuh"
#include "ATen/cuda/detail/IndexUtils.cuh"
#include "ATen/cuda/detail/KernelUtils.h"

namespace at { namespace native {

using namespace at::cuda::detail;

namespace {
	static __forceinline__ __device__
	int clip_coordinates(int in, int clip_limit) {
	  return ::min(clip_limit - 1, ::max(in, static_cast<int>(0)));
	}

	static __forceinline__ __device__
	bool within_bounds_2d(int h, int w, int H, int W) {
	  return h >= 0 && h < H && w >= 0 && w < W;
	}

  template<typename scalar_t>
	static __forceinline__ __device__
  void safe_add_2d(scalar_t *data, int64_t h, int64_t w,
                   int64_t sH, int64_t sW, int64_t H, int64_t W,
                   scalar_t delta) {
    if (h >= 0 && h < H && w >= 0 && w < W) {
      atomicAdd(data + h * sH + w * sW, delta);
    }
  }

	template <typename scalar_t>
	__launch_bounds__(1024)
	__global__ void grid_sampler_2d_kernel(
	    const int nthreads,
	    TensorInfo<scalar_t, int> input,
	    TensorInfo<scalar_t, int> grid,
	    TensorInfo<scalar_t, int> output,
	    const int padding_mode) {

	  int C = input.sizes[1];
	  int inp_H = input.sizes[2];
	  int inp_W = input.sizes[3];
	  int out_H = grid.sizes[1];
	  int out_W = grid.sizes[2];
	  int inp_sN = input.strides[0];
	  int inp_sC = input.strides[1];
	  int inp_sH = input.strides[2];
	  int inp_sW = input.strides[3];
	  int grid_sN = grid.strides[0];
	  int grid_sH = grid.strides[1];
	  int grid_sW = grid.strides[2];
	  int grid_sCoor = grid.strides[3];
	  int out_sN = output.strides[0];
	  int out_sC = output.strides[1];
	  int out_sH = output.strides[2];
	  int out_sW = output.strides[3];

	  CUDA_KERNEL_LOOP(index, nthreads) {
	  	const int w = index % out_W;
	  	const int h = (index / out_W) % out_H;
	  	const int n = index / (out_H * out_W);
	  	const int grid_offset = n * grid_sN + h * grid_sH + w * grid_sW;

	    // get the corresponding input x, y co-ordinates from grid
	    scalar_t ix = grid.data[grid_offset];
	    scalar_t iy = grid.data[grid_offset + grid_sCoor];

	    // normalize ix, iy from [-1, 1] to [0, IH-1] & [0, IW-1]
	    float ixf = ((ix + 1.f) / 2) * (inp_W - 1);
	    float iyf = ((iy + 1.f) / 2) * (inp_H - 1);

	    ix = static_cast<scalar_t>(ixf);
	    iy = static_cast<scalar_t>(iyf);

	    // get NE, NW, SE, SW pixel values from (x, y)
	    int ix_nw = static_cast<int>(std::floor(ixf));
	    int iy_nw = static_cast<int>(std::floor(iyf));
	    int ix_ne = ix_nw + 1;
	    int iy_ne = iy_nw;
	    int ix_sw = ix_nw;
	    int iy_sw = iy_nw + 1;
	    int ix_se = ix_nw + 1;
	    int iy_se = iy_nw + 1;

	    // get surfaces to each neighbor:
	    scalar_t nw = (ix_se - ix)    * (iy_se - iy);
	    scalar_t ne = (ix    - ix_sw) * (iy_sw - iy);
	    scalar_t sw = (ix_ne - ix)    * (iy    - iy_ne);
	    scalar_t se = (ix    - ix_nw) * (iy    - iy_nw);

	    // calculate bilinear weighted pixel value and set output pixel
			if (padding_mode == detail::GridSamplerPaddingBorder) {
			  // clip coordinates to image borders
			  ix_nw = clip_coordinates(ix_nw, inp_W);
			  iy_nw = clip_coordinates(iy_nw, inp_H);
			  ix_ne = clip_coordinates(ix_ne, inp_W);
			  iy_ne = clip_coordinates(iy_ne, inp_H);
			  ix_sw = clip_coordinates(ix_sw, inp_W);
			  iy_sw = clip_coordinates(iy_sw, inp_H);
			  ix_se = clip_coordinates(ix_se, inp_W);
			  iy_se = clip_coordinates(iy_se, inp_H);
			}

	    auto inp_ptr_NC = input.data + n * inp_sN;
	    auto out_ptr_NCHW = output.data + n * out_sN + h * out_sH + w * out_sW;
	    for (int c = 0; c < C; ++c, inp_ptr_NC += inp_sC, out_ptr_NCHW += out_sC) {
	      *out_ptr_NCHW = static_cast<scalar_t>(0);
	      if (padding_mode != detail::GridSamplerPaddingZeros || within_bounds_2d(iy_nw, ix_nw, inp_H, inp_W)) {
	        *out_ptr_NCHW += inp_ptr_NC[iy_nw * inp_sH + ix_nw * inp_sW] * nw;
	      }
	      if (padding_mode != detail::GridSamplerPaddingZeros || within_bounds_2d(iy_ne, ix_ne, inp_H, inp_W)) {
	        *out_ptr_NCHW += inp_ptr_NC[iy_ne * inp_sH + ix_ne * inp_sW] * ne;
	      }
	      if (padding_mode != detail::GridSamplerPaddingZeros || within_bounds_2d(iy_sw, ix_sw, inp_H, inp_W)) {
	        *out_ptr_NCHW += inp_ptr_NC[iy_sw * inp_sH + ix_sw * inp_sW] * sw;
	      }
	      if (padding_mode != detail::GridSamplerPaddingZeros || within_bounds_2d(iy_se, ix_se, inp_H, inp_W)) {
	        *out_ptr_NCHW += inp_ptr_NC[iy_se * inp_sH + ix_se * inp_sW] * se;
	      }
	    }
	  }
	}

	template <typename scalar_t>
	__launch_bounds__(1024)
	__global__ void grid_sampler_2d_backward_kernel(
	    const int nthreads,
	    TensorInfo<scalar_t, int> grad_output,
	    TensorInfo<scalar_t, int> input,
	    TensorInfo<scalar_t, int> grid,
	    TensorInfo<scalar_t, int> grad_input,  // initialized to zeros
	    TensorInfo<scalar_t, int> grad_grid,   // initialized to empty
	    const int padding_mode) {

    int C = input.sizes[1];
    int inp_H = input.sizes[2];
    int inp_W = input.sizes[3];
	  int out_H = grid.sizes[1];
	  int out_W = grid.sizes[2];
    int inp_sN = input.strides[0];
    int inp_sC = input.strides[1];
    int inp_sH = input.strides[2];
    int inp_sW = input.strides[3];
    int grid_sN = grid.strides[0];
    int grid_sH = grid.strides[1];
    int grid_sW = grid.strides[2];
    int grid_sCoor = grid.strides[3];
    int gOut_sN = grad_output.strides[0];
    int gOut_sC = grad_output.strides[1];
    int gOut_sH = grad_output.strides[2];
    int gOut_sW = grad_output.strides[3];
    int gInp_sN = grad_input.strides[0];
    int gInp_sC = grad_input.strides[1];
    int gInp_sH = grad_input.strides[2];
    int gInp_sW = grad_input.strides[3];
    int gGrid_sW = grad_grid.strides[2];

	  CUDA_KERNEL_LOOP(index, nthreads) {
	  	const int w = index % out_W;
	  	const int h = (index / out_W) % out_H;
	  	const int n = index / (out_H * out_W);
	  	const int grid_offset = n * grid_sN + h * grid_sH + w * grid_sW;

	    // get the corresponding input x, y co-ordinates from grid
	    scalar_t ix = grid.data[grid_offset];
	    scalar_t iy = grid.data[grid_offset + grid_sCoor];

	    // normalize ix, iy from [-1, 1] to [0, IH-1] & [0, IW-1]
	    float ixf = ((ix + 1.f) / 2) * (inp_W - 1);
	    float iyf = ((iy + 1.f) / 2) * (inp_H - 1);

	    ix = static_cast<scalar_t>(ixf);
	    iy = static_cast<scalar_t>(iyf);

	    // get NE, NW, SE, SW pixel values from (x, y)
	    int ix_nw = static_cast<int>(std::floor(ixf));
	    int iy_nw = static_cast<int>(std::floor(iyf));
	    int ix_ne = ix_nw + 1;
	    int iy_ne = iy_nw;
	    int ix_sw = ix_nw;
	    int iy_sw = iy_nw + 1;
	    int ix_se = ix_nw + 1;
	    int iy_se = iy_nw + 1;

	    // get surfaces to each neighbor:
	    scalar_t nw = (ix_se - ix)    * (iy_se - iy);
	    scalar_t ne = (ix    - ix_sw) * (iy_sw - iy);
	    scalar_t sw = (ix_ne - ix)    * (iy    - iy_ne);
	    scalar_t se = (ix    - ix_nw) * (iy    - iy_nw);

	    int ix_nw_cl, iy_nw_cl, ix_ne_cl, iy_ne_cl, ix_sw_cl, iy_sw_cl, ix_se_cl, iy_se_cl;

	    // calculate bilinear weighted pixel value and set output pixel
			if (padding_mode == detail::GridSamplerPaddingBorder) {
			  // clip coordinates to image borders
			  ix_nw_cl = clip_coordinates(ix_nw, inp_W);
			  iy_nw_cl = clip_coordinates(iy_nw, inp_H);
			  ix_ne_cl = clip_coordinates(ix_ne, inp_W);
			  iy_ne_cl = clip_coordinates(iy_ne, inp_H);
			  ix_sw_cl = clip_coordinates(ix_sw, inp_W);
			  iy_sw_cl = clip_coordinates(iy_sw, inp_H);
			  ix_se_cl = clip_coordinates(ix_se, inp_W);
			  iy_se_cl = clip_coordinates(iy_se, inp_H);
			} else {
	      ix_nw_cl = ix_nw;
	      iy_nw_cl = iy_nw;
	      ix_ne_cl = ix_ne;
	      iy_ne_cl = iy_ne;
	      ix_sw_cl = ix_sw;
	      iy_sw_cl = iy_sw;
	      ix_se_cl = ix_se;
	      iy_se_cl = iy_se;
			}

      scalar_t gix = static_cast<scalar_t>(0), giy = static_cast<scalar_t>(0);
      scalar_t *gOut_ptr_NCHW = grad_output.data + n * gOut_sN + h * gOut_sH + w * gOut_sW;
      scalar_t *gInp_ptr_NC = grad_input.data + n * gInp_sN;
      scalar_t *inp_ptr_NC = input.data + n * inp_sN;
	    for (int c = 0; c < C; ++c, inp_ptr_NC += inp_sC, gInp_ptr_NC += gInp_sC, gOut_ptr_NCHW += gOut_sC) {
        scalar_t gOut = *gOut_ptr_NCHW;

	      // calculate and set grad_input
	      safe_add_2d(gInp_ptr_NC, iy_nw_cl, ix_nw_cl, gInp_sH, gInp_sW, inp_H, inp_W, nw * gOut);
	      safe_add_2d(gInp_ptr_NC, iy_ne_cl, ix_ne_cl, gInp_sH, gInp_sW, inp_H, inp_W, ne * gOut);
	      safe_add_2d(gInp_ptr_NC, iy_sw_cl, ix_sw_cl, gInp_sH, gInp_sW, inp_H, inp_W, sw * gOut);
	      safe_add_2d(gInp_ptr_NC, iy_se_cl, ix_se_cl, gInp_sH, gInp_sW, inp_H, inp_W, se * gOut);

	      // calculate grad_grid
	      if (padding_mode != detail::GridSamplerPaddingZeros || within_bounds_2d(iy_nw_cl, ix_nw_cl, inp_H, inp_W)) {
	      	scalar_t nw_val = inp_ptr_NC[iy_nw_cl * inp_sH + ix_nw_cl * inp_sW];
	      	gix -= nw_val * (iy_se - iy) * gOut;
      		giy -= nw_val * (ix_se - ix) * gOut;
	      }
	      if (padding_mode != detail::GridSamplerPaddingZeros || within_bounds_2d(iy_ne_cl, ix_ne_cl, inp_H, inp_W)) {
	      	scalar_t ne_val = inp_ptr_NC[iy_ne_cl * inp_sH + ix_ne_cl * inp_sW];
	      	gix += ne_val * (iy_sw - iy) * gOut;
      		giy -= ne_val * (ix - ix_sw) * gOut;
	      }
	      if (padding_mode != detail::GridSamplerPaddingZeros || within_bounds_2d(iy_sw_cl, ix_sw_cl, inp_H, inp_W)) {
	      	scalar_t sw_val = inp_ptr_NC[iy_sw_cl * inp_sH + ix_sw_cl * inp_sW];
	      	gix -= sw_val * (iy - iy_ne) * gOut;
      		giy += sw_val * (ix_ne - ix) * gOut;
	      }
	      if (padding_mode != detail::GridSamplerPaddingZeros || within_bounds_2d(iy_se_cl, ix_se_cl, inp_H, inp_W)) {
	      	scalar_t se_val = inp_ptr_NC[iy_se_cl * inp_sH + ix_se_cl * inp_sW];
	        gix += se_val * (iy - iy_nw) * gOut;
	        giy += se_val * (ix - ix_nw) * gOut;
	      }
	    }

	    // un-normalize grad_grid values back to [-1, 1] constraints
	    gix = gix * (inp_W - 1.f) / 2;
	    giy = giy * (inp_H - 1.f) / 2;

	    scalar_t *gGrid_ptr_NHW = grad_grid.data + index * gGrid_sW;  // assuming grad_grid is contiguous
	    gGrid_ptr_NHW[0] = gix;
	    gGrid_ptr_NHW[1] = giy;
	  }
	}
}  // namespace

Tensor grid_sampler_2d_cuda(const Tensor& input, const Tensor& grid,
                            int64_t interpolation_mode, int64_t padding_mode) {
	auto N = input.size(0);
	auto H = grid.size(1);
	auto W = grid.size(2);
	auto output = at::empty({N, input.size(1), H, W}, input.options());
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.type(), "grid_sampler_2d_cuda", [&] {
	  int count = static_cast<int>(N * H * W);
	  grid_sampler_2d_kernel
	    <<<GET_BLOCKS(count), CUDA_NUM_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
	      count,
	      getTensorInfo<scalar_t, int>(input),
	      getTensorInfo<scalar_t, int>(grid),
	      getTensorInfo<scalar_t, int>(output),
	      padding_mode);
  });
  return output;
}

std::tuple<Tensor, Tensor>
grid_sampler_2d_backward_cuda(const Tensor& grad_output, const Tensor& input, const Tensor& grid,
                             	int64_t interpolation_mode, int64_t padding_mode) {
	auto N = input.size(0);
	auto H = grid.size(1);
	auto W = grid.size(2);
	auto grad_input = at::zeros_like(input);
	auto grad_grid = at::empty_like(grid);
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.type(), "grid_sampler_2d_backward_cuda", [&] {
	  int count = static_cast<int>(N * H * W);
	  grid_sampler_2d_backward_kernel
	    <<<GET_BLOCKS(count), CUDA_NUM_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
	      count,
	      getTensorInfo<scalar_t, int>(grad_output),
	      getTensorInfo<scalar_t, int>(input),
	      getTensorInfo<scalar_t, int>(grid),
	      getTensorInfo<scalar_t, int>(grad_input),
	      getTensorInfo<scalar_t, int>(grad_grid),
	      padding_mode);
  });
  return std::make_tuple(grad_input, grad_grid);
}

}}  // namespace at::native