#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Parallel.h>
#include <ATen/native/cpu/utils.h>

#include <tuple>
#include <vector>

namespace at {

namespace meta {
TORCH_META_FUNC(fractional_max_pool2d) (
  const at::Tensor& input,
  IntArrayRef pool_size,
  IntArrayRef output_size,
  const at::Tensor& randomSamples
) {
  TORCH_CHECK(
      pool_size.size() == 2,
      "fractional_max_pool2d: kernel_size must either be a single Int or tuple of Ints")
  TORCH_CHECK(
      output_size.size() == 2,
      "fractional_max_pool2d: output_size must either be a single Int or tuple of Ints")
  int64_t numBatch = 1;
  int64_t planeDim = 0;
  int64_t heightDim = 1;
  int64_t widthDim = 2;
  int64_t outputH = output_size[0];
  int64_t outputW = output_size[1];
  int64_t poolSizeH = pool_size[0];
  int64_t poolSizeW = pool_size[1];

  int64_t ndims = input.ndimension();
  const auto memory_format = input.suggest_memory_format();
  if (memory_format == at::MemoryFormat::ChannelsLast) {
    TORCH_CHECK(ndims == 4,
        "non-empty 4D (batch mode) tensor expected for input with channels_last layout");
  } else if (memory_format == at::MemoryFormat::Contiguous) {
    TORCH_CHECK((ndims == 3 || ndims == 4),
        "non-empty 3D or 4D (batch mode) tensor expected for input");
  } else {
    TORCH_CHECK(false, "Unsupport memory format. Supports only ChannelsLast, Contiguous");
  }

  if (ndims == 4) {
    numBatch = input.size(0);
    planeDim++;
    heightDim++;
    widthDim++;
  }

  /* sizes */
  int64_t numPlanes = input.size(planeDim);
  int64_t inputH = input.size(heightDim);
  int inputW = input.size(widthDim);

  TORCH_CHECK(outputH + poolSizeH - 1 <= inputH,
    "fractional_max_pool2d(): pool height ", poolSizeH,
    " too large relative to input height ", inputH);
  TORCH_CHECK(outputW + poolSizeW - 1 <= inputW,
    "fractional_max_pool2d(): pool width ", poolSizeW,
    " too large relative to input width ", inputW);

  if (ndims == 3) {
    set_output(0, {numPlanes, outputH, outputW}, input.options());
    /* indices will contain the locations for each output point */
    set_output(1, {numPlanes, outputH, outputW}, input.options().dtype(kLong));
  } else {
    set_output(0, {numBatch, numPlanes, outputH, outputW}, input.options().memory_format(memory_format));
    /* indices will contain the locations for each output point */
    set_output(1, {numBatch, numPlanes, outputH, outputW}, input.options().memory_format(memory_format).dtype(kLong));
  }
}

} // namespace meta

namespace native {
namespace {

template <typename scalar_t>
static std::vector<int> fractional_max_pool2d_generate_intervals(
  scalar_t sample,
  int inputSize,
  int outputSize,
  int poolSize) {
  std::vector<int> sequence(outputSize);
  if (outputSize > 1) {
    scalar_t alpha = static_cast<scalar_t>(inputSize - poolSize) /
      static_cast<scalar_t>(outputSize - 1);

    for (int i = 0; i < outputSize - 1; ++i) {
      sequence[i] =
        static_cast<int>((i + sample) * alpha) - static_cast<int>(sample * alpha);
    }
  }
  sequence[outputSize - 1] = inputSize - poolSize;

  return sequence;
}

template <typename scalar_t>
static void fractional_max_pool2d_contiguous(
    scalar_t* input,
    scalar_t* output,
    int64_t* indices,
    scalar_t* randomSamples,
    int64_t numBatch, int64_t numPlanes,
    int64_t inputW, int64_t inputH,
    int64_t outputW, int64_t outputH,
    int64_t poolSizeW, int64_t poolSizeH) {

  at::parallel_for(0, numBatch * numPlanes, 0, [&](int64_t begin, int64_t end) {
    int64_t n;
    int64_t c;
    data_index_init(begin, n, numBatch, c, numPlanes);

    for (int64_t i = begin; i < end; i++) {
      /* Generate interval sequence */
      scalar_t* randomSamplesForPlane = randomSamples + i * 2;
      auto sequenceW = fractional_max_pool2d_generate_intervals<scalar_t>(
          randomSamplesForPlane[0], inputW, outputW, poolSizeW);
      auto sequenceH = fractional_max_pool2d_generate_intervals<scalar_t>(
          randomSamplesForPlane[1], inputH, outputH, poolSizeH);

      /* local pointers for each plane */
      scalar_t* input_ptr = input + i * inputW * inputH;
      scalar_t* output_ptr = output + i * outputW * outputH;
      int64_t* indices_ptr = indices + i * outputW * outputH;

      for (int64_t oh = 0; oh < outputH; oh++) {
        int64_t ih0 = sequenceH[oh];

        for (int64_t ow = 0; ow < outputW; ow++) {
          int64_t iw0 = sequenceW[ow];

          scalar_t maxVal = -std::numeric_limits<scalar_t>::infinity();
          int64_t maxIndex = ih0 * inputW + iw0;
          for (int64_t ih = ih0; ih < ih0 + poolSizeH; ih++) {
            AT_ASSERT(ih >= 0 && ih < inputH);
            for (int64_t iw = iw0; iw < iw0 + poolSizeW; iw++) {
              AT_ASSERT(iw >= 0 && iw < inputW);

              int index = ih * inputW + iw;
              scalar_t val = input_ptr[index];
              if (val > maxVal || std::isnan(val)) {
                maxVal = val;
                maxIndex = index;
              }
            }
          }
          output_ptr[oh * outputW + ow] = maxVal;
          indices_ptr[oh * outputW + ow] = maxIndex;
        }
      }

      data_index_step(n, numBatch, c, numPlanes);
    }
  });
}

template <typename scalar_t>
static void fractional_max_pool2d_channels_last(
    scalar_t* input,
    scalar_t* output,
    int64_t* indices,
    scalar_t* randomSamples,
    int64_t numBatch, int64_t numPlanes,
    int64_t inputW, int64_t inputH,
    int64_t outputW, int64_t outputH,
    int64_t poolSizeW, int64_t poolSizeH) {

  scalar_t alphaH = (outputH == 1) ? static_cast<scalar_t>(1)
      : static_cast<scalar_t>(inputH - poolSizeH) / static_cast<scalar_t>(outputH - 1);
  scalar_t alphaW = (outputW == 1) ? static_cast<scalar_t>(1)
      : static_cast<scalar_t>(inputW - poolSizeW) / static_cast<scalar_t>(outputW - 1);

  int64_t stride_n = inputH * inputW * numPlanes;
  at::parallel_for(0, numBatch * outputH * outputW, 0, [&](int64_t begin, int64_t end) {
    int64_t n = 0;
    int64_t oh = 0;
    int64_t ow = 0;
    data_index_init(begin, n, numBatch, oh, outputH, ow, outputW);

    for (int64_t i = begin; i < end; i++) {
      for (int64_t c = 0; c < numPlanes; c++) {
        scalar_t* randomSamplesForPlane = randomSamples + n * numPlanes * 2 + c * 2;
        scalar_t sampleW = randomSamplesForPlane[0];
        scalar_t sampleH = randomSamplesForPlane[1];

        int64_t ih0 = (oh == outputH - 1) ? inputH - poolSizeH
            : static_cast<int64_t>((oh + sampleH) * alphaH) - static_cast<int64_t>(sampleH * alphaH);
        int64_t iw0 = (ow == outputW - 1) ? inputW - poolSizeW
            : static_cast<int64_t>((ow + sampleW) * alphaW) - static_cast<int64_t>(sampleW * alphaW);

        scalar_t maxVal = -std::numeric_limits<scalar_t>::infinity();
        int64_t maxIndex = ih0 * inputW + iw0;
        for (int64_t ih = ih0; ih < ih0 + poolSizeH; ih++) {
          AT_ASSERT(ih >= 0 && ih < inputH);
          for (int64_t iw = iw0; iw < iw0 + poolSizeW; iw++) {
            AT_ASSERT(iw >= 0 && iw < inputW);
            int64_t index = ih * inputW + iw;
            scalar_t val = input[n * stride_n + index * numPlanes + c];
            if (val > maxVal || std::isnan(val)) {
              maxVal = val;
              maxIndex = index;
            }
          }
        }
        output[i * numPlanes + c] = maxVal;
        indices[i * numPlanes + c] = maxIndex;
      }

      data_index_step(n, numBatch, oh, outputH, ow, outputW);
    }
  });
}

} // anonymous namespace

TORCH_IMPL_FUNC(fractional_max_pool2d_out_cpu) (
  const at::Tensor& input_,
  IntArrayRef pool_size,
  IntArrayRef output_size,
  const at::Tensor& randomSamples,
  const at::Tensor& output,
  const at::Tensor& indices) {

  int64_t numBatch = 1;
  int64_t planeDim = 0;
  int64_t heightDim = 1;
  int64_t widthDim = 2;
  int64_t outputH = output_size[0]; // output.size(heightDim)
  int64_t outputW = output_size[1]; // output.size(widthDim)
  int64_t poolSizeH = pool_size[0];
  int64_t poolSizeW = pool_size[1];

  /* get contiguous input */
  auto memory_format = input_.suggest_memory_format();
  auto input = input_.contiguous(memory_format);

  int64_t ndims = input.ndimension();

  if (ndims == 4) {
    numBatch = input.size(0);
    planeDim++;
    heightDim++;
    widthDim++;
  }

  /* sizes */
  int64_t numPlanes = input.size(planeDim);
  int64_t inputH = input.size(heightDim);
  int64_t inputW = input.size(widthDim);

  switch (input.suggest_memory_format()) {
    case at::MemoryFormat::Contiguous: {
      AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "fractional_max_pool2d_contiguous", [&] {
        fractional_max_pool2d_contiguous<scalar_t>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            indices.data_ptr<int64_t>(),
            randomSamples.data_ptr<scalar_t>(),
            numBatch, numPlanes,
            inputW, inputH,
            outputW, outputH,
            poolSizeW, poolSizeH);
      });
      break;
    }
    case at::MemoryFormat::ChannelsLast: {
      AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "fractional_max_pool2d_channels_last", [&] {
        fractional_max_pool2d_channels_last<scalar_t>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            indices.data_ptr<int64_t>(),
            randomSamples.data_ptr<scalar_t>(),
            numBatch, numPlanes,
            inputW, inputH,
            outputW, outputH,
            poolSizeW, poolSizeH);
      });
      break;
    }
    default:
      TORCH_CHECK(false, "Unsupported memory format. Supports only ChannelsLast, Contiguous");
  }
}

namespace {

template <typename scalar_t>
static void fractional_max_pool2d_backward_out_single_batch_frame(
  scalar_t* gradInput,
  scalar_t* gradOutput,
  int64_t* indices,
  int numPlanes,
  int inputW, int inputH,
  int outputW, int outputH) {
  at::parallel_for(0, numPlanes, 0, [&](int64_t start, int64_t end) {
    for (auto plane = start; plane < end; plane++) {
      scalar_t* gradInputForPlane = gradInput + plane * inputW * inputH;
      scalar_t* gradOutputForPlane = gradOutput + plane * outputW * outputH;
      int64_t* indicesForPlane = indices + plane * outputW * outputH;

      // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
      int h, w;
      for (h = 0; h < outputH; ++h) {
        for (w = 0; w < outputW; ++w) {
          int outputIndex = h * outputW + w;
          int64_t index = indicesForPlane[outputIndex];
          AT_ASSERT(index >= 0 && index < inputW * inputH);

          gradInputForPlane[index] += gradOutputForPlane[outputIndex];
        }
      }
    }
  });
}

template <typename scalar_t>
static void fractional_max_pool2d_backward_out_frame(
  scalar_t* gradInput,
  scalar_t* gradOutput,
  int64_t* indices,
  int numBatch, int numPlanes,
  int inputW, int inputH,
  int outputW, int outputH) {
    if(numBatch == 1) {
      fractional_max_pool2d_backward_out_single_batch_frame<scalar_t>(
        gradInput, gradOutput, indices,
        numPlanes,
        inputW, inputH, outputW, outputH
      );
      return;
    }
    at::parallel_for(0, numBatch, 0, [&](int64_t start, int64_t end) {
      for (auto batch = start; batch < end; ++batch) {
        fractional_max_pool2d_backward_out_single_batch_frame<scalar_t>(
          gradInput + batch * numPlanes * inputH * inputW,
          gradOutput + batch * numPlanes * outputH * outputW,
          indices + batch * numPlanes * outputH * outputW,
          numPlanes, inputW, inputH, outputW, outputH);
      }
    });
}

Tensor& fractional_max_pool2d_backward_out_cpu_template(
  const at::Tensor& input,
  const at::Tensor& gradOutput_,
  at::Tensor& gradInput,
  IntArrayRef output_size,
  IntArrayRef pool_size /* unused */,
  const at::Tensor& indices_) {

  int numBatch = 1;
  int planeDim = 0;
  int heightDim = 1;
  int widthDim = 2;

  int outputH = output_size[0];
  int outputW = output_size[1];

  int ndims = input.ndimension();
  if (ndims == 4) {
    numBatch = input.size(0);
    planeDim = 1;
    heightDim++;
    widthDim++;
  }

  /* sizes */
  int numPlanes = input.size(planeDim);
  int inputH = input.size(heightDim);
  int inputW = input.size(widthDim);

  /* get contiguous gradOutput */
  auto gradOutput = gradOutput_.contiguous();
  auto indices = indices_.contiguous();

  TORCH_CHECK(outputW == gradOutput.size(widthDim),
    "fractional_max_pool2d_backward(): gradOutput width unexpected");
  TORCH_CHECK(outputH == gradOutput.size(heightDim),
    "fractional_max_pool2d_backward(): gradOutput height unexpected");

  /* resize */
  gradInput.resize_as_(input);
  gradInput.zero_();

  /* backprop */
  AT_DISPATCH_FLOATING_TYPES(
    input.scalar_type(), "fractional_max_pool2d_backward_out_frame", [&] {
      auto gradInput_data = gradInput.data_ptr<scalar_t>();
      auto gradOutput_data = gradOutput.data_ptr<scalar_t>();
      auto indices_data = indices.data_ptr<int64_t>();
      fractional_max_pool2d_backward_out_frame<scalar_t>(
        gradInput_data,
        gradOutput_data,
        indices_data,
        numBatch, numPlanes,
        inputW, inputH,
        outputW, outputH
      );
      }
    );
  return gradInput;
}

} // namespace

Tensor& fractional_max_pool2d_backward_out_cpu(const at::Tensor& gradOutput_,
  const at::Tensor& input,
  IntArrayRef pool_size,
  IntArrayRef output_size,
  const at::Tensor& indices,
  at::Tensor& gradInput)
{
  gradInput.resize_as_(input);
  fractional_max_pool2d_backward_out_cpu_template(
    input,
    gradOutput_,
    gradInput,
    output_size,
    pool_size,
    indices);
  return gradInput;
}

Tensor fractional_max_pool2d_backward_cpu(
  const at::Tensor& gradOutput_,
  const at::Tensor& input,
  IntArrayRef pool_size,
  IntArrayRef output_size,
  const at::Tensor& indices)
{
  Tensor gradInput = at::empty({0}, input.options());
  fractional_max_pool2d_backward_out_cpu_template(
    input,
    gradOutput_,
    gradInput,
    output_size,
    pool_size,
    indices);
  return gradInput;
}

} // at::native
} // at
