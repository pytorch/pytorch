#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/TensorUtils.h>
#include <ATen/Utils.h>
#include <ATen/cuda/Atomic.cuh>
#include <ATen/cuda/CUDAContext.h>
#include <c10/util/Exception.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/adaptive_avg_pool3d_backward_native.h>
#include <ATen/ops/adaptive_avg_pool3d_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/zeros_like.h>
#endif

#include <algorithm>
#include <cfloat>
#include <cmath>


namespace at {
namespace native {

namespace {

__device__ inline int64_t start_index(int64_t a, int64_t b, int64_t c) {
  return (a / b) * c + ((a % b) * c) / b;
}

__device__ inline int64_t end_index(int64_t a, int64_t b, int64_t c) {
  return 1 + ((a + 1) * c - 1) / b;
}

// 5d tensor B x D x T x H x W
// All kernels view batch dim B and dim D as collapsed.

/*
 * Description:
 *    this function adaptively average pools an input 5D tensor along dimensions
 * 2, 3, and 4 5D input, 5D output
 *
 *    gridDim.y blocks work together on a single 2D output plane specified by
 *    (blockIdx.x + offsetZ).
 */
template <typename scalar_t, typename accscalar_t>
__global__ void adaptiveaveragepool(
    scalar_t *input, scalar_t *output,
    int isizeT, int isizeH, int isizeW,
    int osizeT, int osizeH, int osizeW,
    int64_t istrideD,
    int64_t istrideT, int64_t istrideH, int64_t istrideW,
    int64_t offsetZ) {
  // iterates on output pixels
  int ot, oh, ow;

  // compute offsets based on thread/block ID
  int ostartH = blockIdx.y * blockDim.y + threadIdx.y;
  int oendH = osizeH;
  int ostepH = gridDim.y * blockDim.y;
  int ostartW = threadIdx.x;
  int oendW = osizeW;
  int ostepW = blockDim.x;

  // select output plane
  int64_t o_plane = blockIdx.x + offsetZ;
  ot = o_plane % osizeT; // output frame/time
  int d = o_plane / osizeT; // slice/feature

  // input frame/time range is fixed.
  int istartT = start_index(ot, osizeT, isizeT);
  int iendT = end_index(ot, osizeT, isizeT);
  int kT = iendT - istartT;

  // input offset by slice/feature and earliest relevant frame/time
  scalar_t *input_dt = input + d*istrideD + istartT*istrideT;
  // output offset by slice/feature and frame/time
  scalar_t *output_dt = output + o_plane*osizeH*osizeW;

  // For all output pixels...
  for (oh = ostartH; oh < oendH; oh += ostepH) {
    int istartH = start_index(oh, osizeH, isizeH);
    int iendH = end_index(oh, osizeH, isizeH);
    int kH = iendH - istartH;

    for (ow = ostartW; ow < oendW; ow += ostepW) {
      int istartW = start_index(ow, osizeW, isizeW);
      int iendW = end_index(ow, osizeW, isizeW);
      int kW = iendW - istartW;

      // Compute the average pooling from corresponding input pixels
      scalar_t *ptr_input = input_dt + istartH*istrideH + istartW*istrideW;
      scalar_t *ptr_output = output_dt + oh*osizeW + ow;
      accscalar_t sum = static_cast<accscalar_t>(0);

      int it, ih, iw;
      for (it = 0; it < kT; ++it) {
        for (ih = 0; ih < kH; ++ih) {
          for (iw = 0; iw < kW; ++iw) {
            scalar_t val = ptr_input[ih*istrideH + iw*istrideW];
            sum += static_cast<accscalar_t>(val);
          }
        }
        ptr_input += istrideT; // next input frame
      }
      // Update output
      const accscalar_t divide_factor = static_cast<accscalar_t>(kT * kH * kW);
      *ptr_output = static_cast<scalar_t>(sum / divide_factor);
    }
  }
}

template <typename scalar_t, typename accscalar_t>
void adaptiveaveragepool_loop(
    scalar_t *input_data, scalar_t *output_data,
    int64_t totalZ,
    int isizeT, int isizeH, int isizeW,
    int osizeT, int osizeH, int osizeW,
    int64_t istrideD, int64_t istrideT, int64_t istrideH, int64_t istrideW) {
  int64_t offsetZ = 0;
  dim3 threads(32, 8);
  // each H*W plane is processed by blocksH thread blocks
  int blocksH = std::max((int)(16L / totalZ), 1);
  while (totalZ > 0) {
    dim3 blocks(totalZ > 65535 ? 65535 : totalZ, blocksH);
    adaptiveaveragepool<scalar_t, accscalar_t>
      <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        input_data, output_data,
        isizeT, isizeH, isizeW,
        osizeT, osizeH, osizeW,
        istrideD,
        istrideT, istrideH, istrideW,
        offsetZ);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    totalZ -= 65535;
    offsetZ += 65535;
  }
}

/*
 * Description:
 *    This function computes the gradInput from gradOutput.
 *
 *    gridDim.y blocks work together on a single 2D output plane specified by
 *    (blockIdx.x + offsetZ).
 */
template <typename scalar_t, typename accscalar_t>
__global__ void adaptiveaveragegradinput(
    scalar_t *gradInput, scalar_t *gradOutput,
    int isizeT, int isizeH, int isizeW,
    int osizeT, int osizeH, int osizeW,
    int64_t offsetZ)
{
  // iterators on input pixels
  int it, ih, iw;

  // compute offsets based on thread/block ID
  int istartH = blockIdx.y * blockDim.y + threadIdx.y;
  int iendH = isizeH;
  int istepH = gridDim.y * blockDim.y;
  int istartW = threadIdx.x;
  int iendW = isizeW;
  int istepW = blockDim.x;

  // select input plane
  int64_t i_plane = blockIdx.x + offsetZ;
  it = i_plane % isizeT; // output frame/time
  int d = i_plane / isizeT; // slice/feature

  // output frame/time range is fixed.
  int ostartT = start_index(it, isizeT, osizeT);
  int oendT = end_index(it, isizeT, osizeT);

  // gradInput offset by slice/feature and frame/time.
  scalar_t *gradInput_dt = gradInput + i_plane*isizeH*isizeW;
  // gradOutput offset by slice/feature and earliest relevant frame/time
  scalar_t *gradOutput_dt = gradOutput + (d*osizeT + ostartT)*osizeH*osizeW;

  // For all input pixels...
  for (ih = istartH; ih < iendH; ih += istepH) {
    int ostartH = start_index(ih, isizeH, osizeH);
    int oendH = end_index(ih, isizeH, osizeH);

    for (iw = istartW; iw < iendW; iw += istepW) {
      int ostartW = start_index(iw, isizeW, osizeW);
      int oendW = end_index(iw, isizeW, osizeW);

      // Compute the gradients from corresponding output pixels
      scalar_t *ptr_gradInput = gradInput_dt + ih*isizeW + iw;
      scalar_t *ptr_gradOutput = gradOutput_dt;

      // for all relevant output pixels
      int ot, oh, ow;
      for (ot = ostartT; ot < oendT; ++ot) {
        int kT = end_index(ot, osizeT, isizeT) - start_index(ot, osizeT, isizeT);
        for (oh = ostartH; oh < oendH; ++oh) {
          int kH = end_index(oh, osizeH, isizeH) - start_index(oh, osizeH, isizeH);
          for (ow = ostartW; ow < oendW; ++ow) {
            int kW = end_index(ow, osizeW, isizeW) - start_index(ow, osizeW, isizeW);
            const accscalar_t divide_factor = kW * kH * kT;
            accscalar_t grad_delta = static_cast<accscalar_t>(ptr_gradOutput[oh*osizeW + ow] / divide_factor);
            *ptr_gradInput += static_cast<scalar_t>(grad_delta);
          }
        }
        ptr_gradOutput += osizeH*osizeW; // next output frame
      }
    }
  }
}

template <typename scalar_t, typename accscalar_t>
void adaptiveaveragegradinput_loop(
    scalar_t *gradInput_data, scalar_t *gradOutput_data,
    int64_t totalZ,
    int isizeT, int isizeH, int isizeW,
    int osizeT, int osizeH, int osizeW) {
  int64_t offsetZ = 0;
  dim3 threads(32, 8);
  // each H*W plane is processed by blocksH thread blocks
  int blocksH = std::max((int)(16L / totalZ), 1);
  while (totalZ > 0) {
    dim3 blocks(totalZ > 65535 ? 65535 : totalZ, blocksH);
    adaptiveaveragegradinput<scalar_t, accscalar_t>
      <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        gradInput_data, gradOutput_data,
        isizeT, isizeH, isizeW,
        osizeT, osizeH, osizeW,
        offsetZ);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    totalZ -= 65535;
    offsetZ += 65535;
  }
}

/*
 * Description:
 *    This function computes the gradInput from gradOutput.
 *
 *    gridDim.y blocks work together on a single 2D output plane specified by
 *    (blockIdx.x + offsetZ).
 *
 *    (uses atomic add)
 *
 */
template <typename scalar_t>
__global__ void atomicadaptiveaveragegradinput(
    scalar_t *gradInput, scalar_t *gradOutput,
    int isizeT, int isizeH, int isizeW,
    int osizeT, int osizeH, int osizeW,
    int64_t offsetZ)
{
  // iterators on output pixels
  int ot, oh, ow;

  // compute offsets based on thread/block ID
  int ostartH = blockIdx.y * blockDim.y + threadIdx.y;
  int oendH = osizeH;
  int ostepH = gridDim.y * blockDim.y;
  int ostartW = threadIdx.x;
  int oendW = osizeW;
  int ostepW = blockDim.x;

  // select output plane
  int64_t o_plane = blockIdx.x + offsetZ;
  ot = o_plane % osizeT; // output frame/time
  int d = o_plane / osizeT; // output slice/feature

  // input frame/time range is fixed.
  int istartT = start_index(ot, osizeT, isizeT);
  int iendT = end_index(ot, osizeT, isizeT);
  int kT = iendT - istartT;

  // gradInput offset by slice/feature and earliest relevant frame/time
  scalar_t *gradInput_nt = gradInput + (d*isizeT + istartT)*isizeH*isizeW;
  // gradOutput offset by slice/feature and frame/time
  scalar_t *gradOutput_nt = gradOutput + o_plane*osizeH*osizeW;

  // For all output pixels...
  for (oh = ostartH; oh < oendH; oh += ostepH) {
    int istartH = start_index(oh, osizeH, isizeH);
    int iendH = end_index(oh, osizeH, isizeH);
    int kH = iendH - istartH;

    for (ow = ostartW; ow < oendW; ow += ostepW) {
      int istartW = start_index(ow, osizeW, isizeW);
      int iendW = end_index(ow, osizeW, isizeW);
      int kW = iendW - istartW;

      // Compute the gradients from corresponding input pixels
      scalar_t *ptr_gradInput = gradInput_nt + istartH*isizeW + istartW;
      scalar_t *ptr_gradOutput = gradOutput_nt + oh*osizeW + ow;
      scalar_t grad_delta = *ptr_gradOutput / kT / kH / kW;

      int it, ih, iw;
      for (it = 0; it < kT; ++it) {
        for (ih = 0; ih < kH; ++ih) {
          for (iw = 0; iw < kW; ++iw) {
            gpuAtomicAddNoReturn(&(ptr_gradInput[ih*isizeW + iw]), grad_delta);
          }
        }
        ptr_gradInput += isizeH*isizeW; // next input frame
      }
    }
  }
}

template <typename scalar_t>
void atomicadaptiveaveragegradinput_loop(
    scalar_t* gradInput_data, scalar_t* gradOutput_data,
    int64_t totalZ,
    int isizeT, int isizeH, int isizeW,
    int osizeT, int osizeH, int osizeW) {
  int64_t offsetZ = 0;
  dim3 threads(32, 8);
  int blocksH = std::max((int)(16L / totalZ), 1);
  while (totalZ > 0) {
    dim3 blocks(totalZ > 65535 ? 65535 : totalZ, blocksH);
    atomicadaptiveaveragegradinput<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        gradInput_data, gradOutput_data,
        isizeT, isizeH, isizeW,
        osizeT, osizeH, osizeW,
        offsetZ);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    totalZ -= 65535;
    offsetZ += 65535;
  }
}

// 5D tensor B x D x T x H x w

void adaptive_avg_pool3d_out_cuda_template(
    Tensor& output,
    const Tensor& input_,
    IntArrayRef& output_size) {
  TensorArg output_arg{output, "output", 1};
  TensorArg input_arg{input_, "input_", 2};

  checkAllSameGPU("adaptive_avg_pool3d_cuda", {output_arg, input_arg});

  for (int64_t i = 1; i < input_.ndimension(); i++) {
    TORCH_CHECK(
        input_.size(i) > 0,
        "adaptive_avg_pool3d_cuda(): Expected input to have non-zero size for non-batch dimensions, "
        "but input has sizes ", input_.sizes(),
        " with dimension ", i, " being empty");
  }

  TORCH_CHECK(
      (input_.ndimension() == 4 || input_.ndimension() == 5),
      "adaptive_avg_pool3d_cuda(): Expected 4D or 5D tensor, but got ", input_.sizes());

  // the jit sometimes passes output_size.size() == 1
  TORCH_CHECK(
      output_size.size() == 1 || output_size.size() == 3,
      "adaptive_avg_pool3d: internal error: output_size.size() must be 1 or 3");

  int64_t osizeT = output_size[0];
  int64_t osizeH = output_size[1];
  int64_t osizeW = output_size[2];

  int64_t sizeD, isizeT, isizeH, isizeW;
  int64_t istrideD, istrideT, istrideH, istrideW;
  int64_t totalZ;

  const Tensor& input = input_.ndimension() == 4 ? input_ : input_.contiguous();

  if (input.ndimension() == 4) {
    sizeD = input.size(0);
    isizeT = input.size(1);
    isizeH = input.size(2);
    isizeW = input.size(3);

    istrideD = input.stride(0);
    istrideT = input.stride(1);
    istrideH = input.stride(2);
    istrideW = input.stride(3);

    output.resize_({sizeD, osizeT, osizeH, osizeW});

    totalZ = sizeD * osizeT;
  } else {
    int64_t sizeB = input.size(0);
    sizeD = input.size(1);
    isizeT = input.size(2);
    isizeH = input.size(3);
    isizeW = input.size(4);

    istrideD = input.stride(1);
    istrideT = input.stride(2);
    istrideH = input.stride(3);
    istrideW = input.stride(4);

    output.resize_({sizeB, sizeD, osizeT, osizeH, osizeW});

    totalZ = sizeB * sizeD * osizeT;
  }

  if (output.numel() == 0) {
    return;
  }

  AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16,
      input.scalar_type(), "adaptive_avg_pool3d_cuda", [&] {
        using accscalar_t = at::acc_type<scalar_t, true>;
        scalar_t* input_data = input.data_ptr<scalar_t>();
        scalar_t* output_data = output.data_ptr<scalar_t>();

        adaptiveaveragepool_loop<scalar_t, accscalar_t>(
            input_data, output_data,
            totalZ,
            isizeT, isizeH, isizeW,
            osizeT, osizeH, osizeW,
            istrideD, istrideT, istrideH, istrideW);
      });
}

void adaptive_avg_pool3d_backward_out_cuda_template(
    Tensor& gradInput,
    const Tensor& gradOutput_,
    const Tensor& input) {
  TensorArg grad_input_arg{gradInput, "gradInput", 1};
  TensorArg grad_output_arg{gradOutput_, "gradOutput_", 2};
  TensorArg input_arg{input, "input", 3};

  checkAllSameGPU(
      "adaptive_avg_pool3d_out_cuda",
      {grad_input_arg, grad_output_arg, input_arg});

  const Tensor gradOutput = gradOutput_.contiguous();

  gradInput.resize_as_(input);
  if (gradInput.numel() == 0) {
    return;
  }

  gradInput.zero_();

  int64_t sizeD, isizeT, isizeH, isizeW;
  int64_t osizeT, osizeH, osizeW;
  int64_t totalZ;

  if (input.ndimension() == 4) {
    sizeD = input.size(0);
    isizeT = input.size(1);
    isizeH = input.size(2);
    isizeW = input.size(3);

    osizeT = gradOutput.size(1);
    osizeH = gradOutput.size(2);
    osizeW = gradOutput.size(3);
  } else {
    sizeD = input.size(1);
    isizeT = input.size(2);
    isizeH = input.size(3);
    isizeW = input.size(4);

    osizeT = gradOutput.size(2);
    osizeH = gradOutput.size(3);
    osizeW = gradOutput.size(4);
  }

  bool atomic = (isizeW%osizeW != 0) || (isizeH%osizeH != 0) || (isizeT%osizeT != 0);

  if (input.ndimension() == 4) {
    totalZ = atomic ? sizeD * osizeT : sizeD * isizeT;
  } else {
    int sizeB = input.size(0);
    totalZ = atomic ? sizeB * sizeD * osizeT : sizeB * sizeD * isizeT;
  }

  if (atomic) {
    AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16,
        input.scalar_type(), "adaptive_avg_pool3d_backward_cuda", [&] {
          scalar_t* gradInput_data = gradInput.data_ptr<scalar_t>();
          scalar_t* gradOutput_data = gradOutput.data_ptr<scalar_t>();

          atomicadaptiveaveragegradinput_loop(
              gradInput_data, gradOutput_data,
              totalZ,
              isizeT, isizeH, isizeW,
              osizeT, osizeH, osizeW);
        });
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16,
        input.scalar_type(), "adaptive_avg_pool3d_backward_cuda", [&] {
          using accscalar_t = at::acc_type<scalar_t, true>;

          scalar_t* gradInput_data = gradInput.data_ptr<scalar_t>();
          scalar_t* gradOutput_data = gradOutput.data_ptr<scalar_t>();

          adaptiveaveragegradinput_loop<scalar_t, accscalar_t>(
              gradInput_data, gradOutput_data,
              totalZ,
              isizeT, isizeH, isizeW,
              osizeT, osizeH, osizeW);
        });
  }
}

} // namespace

Tensor& adaptive_avg_pool3d_out_cuda(const Tensor& input,
    IntArrayRef output_size,
    Tensor& output) {
  adaptive_avg_pool3d_out_cuda_template(output, input, output_size);
  return output;
}

Tensor adaptive_avg_pool3d_cuda(
    const Tensor& input,
    IntArrayRef output_size) {
  auto output = at::empty({0}, input.options());
  adaptive_avg_pool3d_out_cuda_template(output, input, output_size);
  return output;
}

Tensor& adaptive_avg_pool3d_backward_out_cuda(const Tensor& gradOutput_,
    const Tensor& input,
    Tensor& gradInput) {
  // See Note [Writing Nondeterministic Operations]
  // Nondeterministic because of atomicAdd usage
  globalContext().alertNotDeterministic("adaptive_avg_pool3d_backward_out_cuda");
  adaptive_avg_pool3d_backward_out_cuda_template(gradInput, gradOutput_, input);
  return gradInput;
}

Tensor adaptive_avg_pool3d_backward_cuda(
    const Tensor& gradOutput_,
    const Tensor& input) {
  // See Note [Writing Nondeterministic Operations]
  // Nondeterministic because of atomicAdd usage
  globalContext().alertNotDeterministic("adaptive_avg_pool3d_backward_cuda");
  auto gradInput = at::zeros_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  adaptive_avg_pool3d_backward_out_cuda_template(gradInput, gradOutput_, input);
  return gradInput;
}

} // namespace native
} // namespace at
