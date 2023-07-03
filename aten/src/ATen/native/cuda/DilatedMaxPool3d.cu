#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/AccumulateType.h>
#include <ATen/ceil_div.h>
#include <ATen/Dispatch.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/NumericUtils.h>
#include <ATen/native/Pool.h>
#include <ATen/cuda/Atomic.cuh>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/NumericLimits.cuh>
#include <ATen/cuda/detail/TensorInfo.cuh>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/detail/KernelUtils.h>
#include <c10/macros/Macros.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/empty.h>
#include <ATen/ops/max_pool3d_with_indices_native.h>
#include <ATen/ops/max_pool3d_with_indices_backward_native.h>
#endif

namespace at::native {
namespace {

__device__ inline int min(int a, int b) {
  return a <= b ? a : b;
}

template <typename scalar_t>
__global__ static void max_pool3d_with_indices_single_out_frame(
  const scalar_t* inputData,
  scalar_t* outputData,
  int64_t* indicesData,
  int features,
  int itime, int iheight, int iwidth,
  int obatch, int otime, int oheight, int owidth,
  int kT, int kH, int kW,
  int dT, int dH, int dW,
  int pT, int pH, int pW,
  int dilationT, int dilationH, int dilationW,
  int offsetZ,
  bool channels_last)
{
  int oColumn = blockIdx.x * blockDim.x + threadIdx.x;
  int oRow = blockIdx.y * blockDim.y + threadIdx.y;
  int oFrame = 0;
  // used only for channels-first indexing
  int64_t slice = 0;
  // used only for channels-last indexing
  int batch = 0;
  int channel = 0;
  if (!channels_last) {
    // indexing order: batch, channel, time
    oFrame = (blockIdx.z * blockDim.z + threadIdx.z + offsetZ) % otime; // output frame/time
    slice = (blockIdx.z * blockDim.z + threadIdx.z + offsetZ) / otime; // output slice/feature
  } else {
    // indexing order: batch, time, channel
    channel = (blockIdx.z * blockDim.z + threadIdx.z + offsetZ) % features; // output feature (channel)
    slice = (blockIdx.z * blockDim.z + threadIdx.z + offsetZ) / features; // output slice (batch + time)
    batch = slice / otime;
    oFrame = slice % otime;
  }

  // For int64_t data type, see https://github.com/pytorch/pytorch/issues/52822
  if (oRow < oheight && oColumn < owidth && oFrame < otime && channel < features && batch < obatch)
  {
    int tStart = oFrame  * dT - pT;
    int hStart = oRow    * dH - pH;
    int wStart = oColumn * dW - pW;
    int tEnd = min(tStart + (kT - 1) * dilationT + 1, itime);
    int hEnd = min(hStart + (kH - 1) * dilationH + 1, iheight);
    int wEnd = min(wStart + (kW - 1) * dilationW + 1, iwidth);

    while(tStart < 0)
      tStart += dilationT;
    while(hStart < 0)
      hStart += dilationH;
    while(wStart < 0)
      wStart += dilationW;

    // maxIndex remains in "channels-first"/contiguous
    int64_t maxIndex = tStart * iheight * iwidth + hStart * iwidth + wStart;

    if (!channels_last) {
        inputData += (int64_t) slice * itime * iheight * iwidth;
    } else {
        inputData += ((int64_t) batch * itime * iheight * iwidth * features) + channel;
    }

    scalar_t max = at::numeric_limits<scalar_t>::lower_bound(); // -Infinity

    for (int t = tStart; t < tEnd; t += dilationT)
    {
      for (int h = hStart; h < hEnd; h += dilationH)
      {
        for (int w = wStart; w < wEnd; w += dilationW)
        {
          scalar_t val;
          int index = t * iheight * iwidth + h * iwidth + w;
          if (!channels_last) {
            val = inputData[index];
          } else {
            int64_t index_channels_last = index*features;
            val = inputData[index_channels_last];
          }

          if ((max < val) || at::_isnan(val))
          {
            max = val;
            maxIndex = index;
          }
        }
      }
    }

    int64_t out_index;
    if (!channels_last) {
      out_index = (int64_t) slice*otime*oheight*owidth + oFrame*oheight*owidth + oRow*owidth + oColumn;
    } else {
      out_index = ((int64_t) batch*otime*oheight*owidth + oFrame*oheight*owidth + oRow*owidth + oColumn)*features + channel;
    }
    outputData[out_index] = max;
    indicesData[out_index] = maxIndex;
  }
}

template <typename scalar_t>
void max_pool3d_with_indices_out_frame(
  const scalar_t* input_data,
  const Tensor& output,
  const Tensor& indices,
  int features,
  int64_t totalZ,
  int itime, int iheight, int iwidth,
  int obatch, int otime, int oheight, int owidth,
  int kT, int kH, int kW,
  int dT, int dH, int dW,
  int pT, int pH, int pW,
  int dilationT, int dilationH, int dilationW,
  bool channels_last)
{
  int offsetZ = 0;
  int threadX = 32;
  int threadY = 8;
  int threadZ = 1;
  int stepZ = 65535;
  if (channels_last) {
    threadX = 2;
    threadY = 4;
    threadZ = 64;
  }
  dim3 block(threadX, threadY, threadZ);

  while (totalZ > 0) {
    dim3 grid(ceil_div(owidth, static_cast<int>(block.x)),
              ceil_div(oheight, static_cast<int>(block.y)),
              totalZ > stepZ*threadZ ? stepZ : ceil_div(totalZ, static_cast<int64_t>(threadZ)));

    max_pool3d_with_indices_single_out_frame
      <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
         input_data,
         output.mutable_data_ptr<scalar_t>(),
         indices.mutable_data_ptr<int64_t>(),
         features,
         itime, iheight, iwidth,
         obatch, otime, oheight, owidth,
         kT, kH, kW,
         dT, dH, dW,
         pT, pH, pW,
         dilationT, dilationH, dilationW,
         offsetZ, channels_last);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    totalZ -= threadZ*stepZ;
    offsetZ += threadZ*stepZ;
  }
}

#undef UPDATE_OUTPUT_KERNEL_WIDTH

template <typename scalar_t>
__global__ static void max_pool3d_with_indices_backward_single_out_frame(
  scalar_t *gradInputData,
  const scalar_t *gradOutputData,
  const int64_t *indicesData,
  int features,
  int itime, int iheight, int iwidth,
  int obatch, int otime, int oheight, int owidth,
  int offsetZ,
  bool channels_last)
{
  int oColumn = blockIdx.x * blockDim.x + threadIdx.x;
  int oRow = blockIdx.y * blockDim.y + threadIdx.y;

  int oFrame = 0;
  // used only for channels-first indexing
  int64_t slice = 0;
  // used only for channels-last indexing
  int batch = 0;
  int channel = 0;
  if (!channels_last) {
    // indexing order: batch, channel, time
    oFrame = (blockIdx.z * blockDim.z + threadIdx.z + offsetZ) % otime; // output frame/time
    slice = (blockIdx.z * blockDim.z + threadIdx.z + offsetZ) / otime; // output slice/feature
  } else {
    // indexing order: batch, time, channel
    channel = (blockIdx.z * blockDim.z + threadIdx.z + offsetZ) % features; // output feature (channel)
    slice = (blockIdx.z * blockDim.z + threadIdx.z + offsetZ) / features; // output slice (batch + time)
    batch = slice / otime;
    oFrame = slice % otime;
  }

  if (oRow < oheight && oColumn < owidth && oFrame < otime && batch < obatch && channel < features)
  {
    int64_t out_index;
    if (!channels_last) {
      out_index = (int64_t) slice*otime*oheight*owidth + oFrame*oheight*owidth + oRow*owidth + oColumn;
    } else {
      out_index = ((int64_t) batch*otime*oheight*owidth + oFrame*oheight*owidth + oRow*owidth + oColumn)*features + channel;
    }
    int64_t maxIndex = indicesData[out_index];
    if (maxIndex != -1) {
      if (!channels_last) {
        gpuAtomicAddNoReturn(&gradInputData[(int64_t) slice * itime  * iheight * iwidth + maxIndex],
          gradOutputData[out_index]);
      } else {
        gpuAtomicAddNoReturn(&gradInputData[((int64_t) batch * itime * iheight * iwidth + maxIndex) * features + channel],
          gradOutputData[out_index]);
      }
    }
  }
}

template <typename scalar_t>
void max_pool3d_with_indices_backward_out_frame(
  scalar_t *gradInputData,
  const Tensor& gradOutput,
  const Tensor& indices,
  int features,
  int64_t totalZ,
  int itime, int iheight, int iwidth,
  int obatch, int otime, int oheight, int owidth,
  bool channels_last)
{
  int offsetZ = 0;
  int threadX = 32;
  int threadY = 8;
  int threadZ = 1;
  int stepZ = 65535;
  if (channels_last) {
    threadX = 2;
    threadY = 4;
    threadZ = 64;
  }
  dim3 block(threadX, threadY, threadZ);

  while (totalZ > 0) {
    dim3 grid(ceil_div(owidth, static_cast<int>(block.x)),
              ceil_div(oheight, static_cast<int>(block.y)),
              totalZ > stepZ*threadZ ? stepZ : ceil_div(totalZ, static_cast<int64_t>(block.z)));

    max_pool3d_with_indices_backward_single_out_frame
      <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        gradInputData,
        gradOutput.const_data_ptr<scalar_t>(),
        indices.const_data_ptr<int64_t>(),
        features,
        itime, iheight, iwidth,
        obatch, otime, oheight, owidth,
        offsetZ,
        channels_last);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    totalZ -= threadZ*stepZ;
    offsetZ += threadZ*stepZ;
  }
}

void max_pool3d_with_indices_out_cuda_template(
           Tensor& output,
           Tensor& indices,
           const Tensor& input,
           IntArrayRef kernel_size,
           IntArrayRef stride,
           IntArrayRef padding,
           IntArrayRef dilation,
           bool ceil_mode)
{
  TensorArg output_arg{ output, "output", 1 };
  TensorArg indices_arg{ indices, "indices", 2 };
  TensorArg input_arg{ input, "input", 3 };

  checkAllSameGPU(__func__,
                  {output_arg, indices_arg, input_arg});

  // #20866, #22032: Guarantee this for the official C++ API?
  TORCH_CHECK(kernel_size.size() == 1 || kernel_size.size() == 3,
    "max_pool3d: kernel_size must either be a single int, or a tuple of three ints")
  const int kT = safe_downcast<int, int64_t>(kernel_size[0]);
  const int kH = kernel_size.size() == 1 ? kT : safe_downcast<int, int64_t>(kernel_size[1]);
  const int kW = kernel_size.size() == 1 ? kT : safe_downcast<int, int64_t>(kernel_size[2]);

  TORCH_CHECK(stride.size() == 0 || stride.size() == 1 || stride.size() == 3,
    "max_pool3d: stride must either be omitted, a single int, or a tuple of three ints")
  const int dT = stride.empty() ? kT : safe_downcast<int, int64_t>(stride[0]);
  const int dH = stride.empty() ? kH :
                 stride.size() == 1 ? dT : safe_downcast<int, int64_t>(stride[1]);
  const int dW = stride.empty() ? kW :
                 stride.size() == 1 ? dT : safe_downcast<int, int64_t>(stride[2]);

  TORCH_CHECK(padding.size() == 1 || padding.size() == 3,
    "max_pool3d: padding must be either be a single int, or a tuple of three ints");
  const int pT = safe_downcast<int, int64_t>(padding[0]);
  const int pH = padding.size() == 1 ? pT : safe_downcast<int, int64_t>(padding[1]);
  const int pW = padding.size() == 1 ? pT : safe_downcast<int, int64_t>(padding[2]);

  TORCH_CHECK(dilation.size() == 1 || dilation.size() == 3,
    "max_pool3d: dilation must be either a single int, or a tuple of three ints");
  const int dilationT = safe_downcast<int, int64_t>(dilation[0]);
  const int dilationH = dilation.size() == 1 ? dilationT : safe_downcast<int, int64_t>(dilation[1]);
  const int dilationW = dilation.size() == 1 ? dilationT : safe_downcast<int, int64_t>(dilation[2]);

  const int64_t nbatch = input.ndimension() == 5 ? input.size(-5) : 1;
  const int64_t nslices = input.size(-4);
  const int64_t itime = input.size(-3);
  const int64_t iheight = input.size(-2);
  const int64_t iwidth = input.size(-1);

  const int64_t otime = pooling_output_shape<int64_t>(itime, kT, pT, dT, dilationT, ceil_mode);
  const int64_t oheight = pooling_output_shape<int64_t>(iheight, kH, pH, dH, dilationH, ceil_mode);
  const int64_t owidth = pooling_output_shape<int64_t>(iwidth, kW, pW, dW, dilationW, ceil_mode);

  pool3d_shape_check(
    input,
    nslices,
    kT, kH, kW,
    dT, dH, dW,
    pT, pH, pW,
    dilationT, dilationH, dilationW,
    itime, iheight, iwidth,
    otime, oheight, owidth,
    "max_pool3d_with_indices_out_cuda_template()");

  bool channels_last = input.ndimension() == 5 && input.suggest_memory_format() == at::MemoryFormat::ChannelsLast3d;
  Tensor _input = input;
  if (input.ndimension() == 4) {
    Tensor input_channels_last_check = input.unsqueeze(0);
    // work around buggy behavior of suggest_memory_format here where
    // suggested format of unsqueezed tensor is contiguous while it is
    // really only contiguous in ChannelsLast3d
    channels_last = (!input_channels_last_check.is_contiguous()) &&
                     input_channels_last_check.is_contiguous(at::MemoryFormat::ChannelsLast3d);
    if (!channels_last) {
      output.resize_({ nslices, otime, oheight, owidth});
      indices.resize_({nslices, otime, oheight, owidth});
    } else {
      _input = input_channels_last_check;
      output.resize_({1, nslices, otime, oheight, owidth}, at::MemoryFormat::ChannelsLast3d);
      indices.resize_({1, nslices, otime, oheight, owidth}, at::MemoryFormat::ChannelsLast3d);
      output = output.squeeze(0);
      indices = indices.squeeze(0);
    }
  } else {
    if (!channels_last) {
      output.resize_({nbatch, nslices, otime, oheight, owidth});
      indices.resize_({nbatch, nslices, otime, oheight, owidth});
    } else {
      output.resize_({nbatch, nslices, otime, oheight, owidth}, at::MemoryFormat::ChannelsLast3d);
      indices.resize_({nbatch, nslices, otime, oheight, owidth}, at::MemoryFormat::ChannelsLast3d);
    }
  }

  if (input.numel() == 0) {
    return;
  }

  Tensor work_input;
  Tensor work_output = output;
  if (!channels_last) {
    work_input = input.contiguous();
  } else {
    work_input = _input.contiguous(at::MemoryFormat::ChannelsLast3d);
  }
  Tensor work_indices = indices;

  AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16,
    input.scalar_type(),
    "max_pool3d_with_indices_out_frame",
    [&]{
      const scalar_t *input_data = work_input.const_data_ptr<scalar_t>();
      const int64_t totalZ = otime * nslices * nbatch;

      max_pool3d_with_indices_out_frame(
        input_data, work_output, work_indices,
        nslices, // features
        totalZ,
        itime, iheight, iwidth,
        nbatch, otime, oheight, owidth,
        kT, kH, kW,
        dT, dH, dW,
        pT, pH, pW,
        dilationT, dilationH, dilationW, channels_last);
    }
  );
}

void max_pool3d_with_indices_backward_out_cuda_template(
           Tensor& gradInput,
           const Tensor& gradOutput,
           const Tensor& input,
           const Tensor& indices,
           IntArrayRef kernel_size,
           IntArrayRef stride,
           IntArrayRef padding,
           IntArrayRef dilation,
           bool ceil_mode)
{
  TensorArg gradInput_arg{ gradInput, "gradInput", 1 };
  TensorArg gradOutput_arg{ gradOutput, "gradOutput", 2 };
  TensorArg input_arg{ input, "input", 3 };
  TensorArg indices_arg{ indices, "indices", 4 };

  checkAllSameGPU(__func__,
                  {gradInput_arg, gradOutput_arg, input_arg, indices_arg});

  // #20866, #22032: Guarantee this for the official C++ API?
  TORCH_CHECK(kernel_size.size() == 1 || kernel_size.size() == 3,
    "max_pool3d: kernel_size must either be a single int, or a tuple of three ints")
  const int kT = safe_downcast<int, int64_t>(kernel_size[0]);
  const int kH = kernel_size.size() == 1 ? kT : safe_downcast<int, int64_t>(kernel_size[1]);
  const int kW = kernel_size.size() == 1 ? kT : safe_downcast<int, int64_t>(kernel_size[2]);

  TORCH_CHECK(stride.size() == 0 || stride.size() == 1 || stride.size() == 3,
    "max_pool3d: stride must either be omitted, a single int, or a tuple of three ints")
  const int dT = stride.empty() ? kT : safe_downcast<int, int64_t>(stride[0]);
  const int dH = stride.empty() ? kH :
                 stride.size() == 1 ? dT : safe_downcast<int, int64_t>(stride[1]);
  const int dW = stride.empty() ? kW :
                 stride.size() == 1 ? dT : safe_downcast<int, int64_t>(stride[2]);

  TORCH_CHECK(padding.size() == 1 || padding.size() == 3,
    "max_pool3d: padding must be either be a single int, or a tuple of three ints");
  const int pT = safe_downcast<int, int64_t>(padding[0]);
  const int pH = padding.size() == 1 ? pT : safe_downcast<int, int64_t>(padding[1]);
  const int pW = padding.size() == 1 ? pT : safe_downcast<int, int64_t>(padding[2]);

  TORCH_CHECK(dilation.size() == 1 || dilation.size() == 3,
    "max_pool3d: dilation must be either a single int, or a tuple of three ints");
  const int dilationT = safe_downcast<int, int64_t>(dilation[0]);
  const int dilationH = dilation.size() == 1 ? dilationT : safe_downcast<int, int64_t>(dilation[1]);
  const int dilationW = dilation.size() == 1 ? dilationT : safe_downcast<int, int64_t>(dilation[2]);

  TORCH_CHECK((input.ndimension() == 4 || input.ndimension() == 5),
    "max_pool2d_with_indices_backward_out_cuda_template(): ",
    "Expected 4D or 5D input tensor, but got ", input.sizes());

  TORCH_CHECK((gradOutput.ndimension() == 4 || gradOutput.ndimension() == 5),
    "max_pool2d_with_indices_backward_out_cuda_template(): ",
    "Expected 4D or 5D gradOutput tensor, but got ", gradOutput.sizes());

  // Resize and initialize result tensor.
  bool channels_last = input.ndimension() == 5 && input.suggest_memory_format() == at::MemoryFormat::ChannelsLast3d;
  Tensor _input = input;
  if (input.ndimension() == 4) {
    Tensor input_channels_last_check = input.unsqueeze(0);
    // work around buggy behavior of suggest_memory_format here where
    // suggested format of unsqueezed tensor is contiguous while it is
    // really only contiguous in ChannelsLast3d
    channels_last = (!input_channels_last_check.is_contiguous()) &&
                     input_channels_last_check.is_contiguous(at::MemoryFormat::ChannelsLast3d);
    if (channels_last) {
      _input = input_channels_last_check;
    }
  }
  if (!channels_last) {
    gradInput.resize_as_(input);
  } else {
    gradInput.resize_as_(_input, at::MemoryFormat::ChannelsLast3d);
  }
  gradInput.zero_();

  const int64_t nbatch = input.ndimension() == 5 ? input.size(-5) : 1;
  const int64_t nslices = input.size(-4);

  const int64_t otime = gradOutput.size(-3);
  const int64_t oheight = gradOutput.size(-2);
  const int64_t owidth = gradOutput.size(-1);

  const int64_t itime = gradInput.size(-3);
  const int64_t iheight = gradInput.size(-2);
  const int64_t iwidth = gradInput.size(-1);

  max_pool3d_backward_shape_check(
    input,
    gradOutput,
    indices,
    nslices,
    kT, kH, kW,
    dT, dH, dW,
    pT, pH, pW,
    dilationT, dilationH, dilationW,
    itime, iheight, iwidth,
    otime, oheight, owidth,
    "max_pool3d_with_indices_backward_out_cuda_template()");

  if (gradOutput.numel() == 0) {
    return;
  }

  Tensor work_grad_input = gradInput;
  Tensor work_grad_output;
  Tensor work_indices;
  if (!channels_last) {
    work_grad_output = gradOutput.contiguous();
    work_indices = indices.contiguous();
  } else {
    if (input.ndimension() == 4) {
      work_grad_output = gradOutput.unsqueeze(0).contiguous(at::MemoryFormat::ChannelsLast3d);
      work_indices = indices.unsqueeze(0).contiguous(at::MemoryFormat::ChannelsLast3d);
    } else {
      work_grad_output = gradOutput.contiguous(at::MemoryFormat::ChannelsLast3d);
      work_indices = indices.contiguous(at::MemoryFormat::ChannelsLast3d);
    }
  }

  AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16, input.scalar_type(),
    "max_pool3d_with_indices_backward_out_frame",
    [&] {
      const int64_t totalZ = otime * nslices * nbatch;
      scalar_t *grad_input_data = work_grad_input.mutable_data_ptr<scalar_t>();

      max_pool3d_with_indices_backward_out_frame(
        grad_input_data, work_grad_output, work_indices,
        nslices,
        totalZ,
        itime, iheight, iwidth,
        nbatch, otime, oheight, owidth,
        channels_last);
    }
  );
}

} // namespace

std::tuple<Tensor&, Tensor&> max_pool3d_with_indices_out_cuda(const Tensor& input,
  IntArrayRef kernel_size,
  IntArrayRef stride,
  IntArrayRef padding,
  IntArrayRef dilation,
  bool ceil_mode,
  Tensor& output,
  Tensor& indices)
{
  max_pool3d_with_indices_out_cuda_template(
    output,
    indices,
    input,
    kernel_size,
    stride,
    padding,
    dilation,
    ceil_mode);
  return std::tuple<Tensor&, Tensor&>(output, indices);
}

std::tuple<Tensor, Tensor> max_pool3d_with_indices_cuda(
  const Tensor& input,
  IntArrayRef kernel_size,
  IntArrayRef stride,
  IntArrayRef padding,
  IntArrayRef dilation,
  bool ceil_mode)
{
  NoNamesGuard guard;

  Tensor output = at::empty({0}, input.options());
  Tensor indices = at::empty({0}, input.options().dtype(kLong));
  max_pool3d_with_indices_out_cuda_template(
    output,
    indices,
    input,
    kernel_size,
    stride,
    padding,
    dilation,
    ceil_mode);

  guard.reset();
  namedinference::propagate_names(output, input);
  namedinference::propagate_names(indices, input);

  return std::tuple<Tensor, Tensor>(output, indices);
}

Tensor& max_pool3d_with_indices_backward_out_cuda(const Tensor& gradOutput,
  const Tensor& input,
  IntArrayRef kernel_size,
  IntArrayRef stride,
  IntArrayRef padding,
  IntArrayRef dilation,
  bool ceil_mode,
  const Tensor& indices,
  Tensor& gradInput)
{
  // See Note [Writing Nondeterministic Operations]
  // Nondeterministic because of atomicAdd usage
  globalContext().alertNotDeterministic("max_pool3d_with_indices_backward_out_cuda");
  max_pool3d_with_indices_backward_out_cuda_template(
    gradInput,
    gradOutput,
    input,
    indices,
    kernel_size,
    stride,
    padding,
    dilation,
    ceil_mode);
  return gradInput;
}

Tensor max_pool3d_with_indices_backward_cuda(
  const Tensor& gradOutput,
  const Tensor& input,
  IntArrayRef kernel_size,
  IntArrayRef stride,
  IntArrayRef padding,
  IntArrayRef dilation,
  bool ceil_mode,
  const Tensor& indices)
{
  // See Note [Writing Nondeterministic Operations]
  // Nondeterministic because of atomicAdd usage
  globalContext().alertNotDeterministic("max_pool3d_with_indices_backward_cuda");
  auto gradInput = at::empty(input.sizes(), input.options());
  max_pool3d_with_indices_backward_out_cuda_template(
    gradInput,
    gradOutput,
    input,
    indices,
    kernel_size,
    stride,
    padding,
    dilation,
    ceil_mode);
  return gradInput;
}

} // at::native
