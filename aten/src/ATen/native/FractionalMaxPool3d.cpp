#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Parallel.h>
#include <ATen/native/cpu/utils.h>
#include <c10/util/irange.h>

#include <tuple>
#include <vector>

namespace at {

namespace meta {
TORCH_META_FUNC(fractional_max_pool3d) (
  const at::Tensor& input,
  IntArrayRef pool_size,
  IntArrayRef output_size,
  const at::Tensor& randomSamples
) {
  TORCH_CHECK(
      pool_size.size() == 3,
      "fractional_max_pool3d: kernel_size must either be a single Int or tuple of three Ints")
  TORCH_CHECK(
      output_size.size() == 3,
      "fractional_max_pool3d: output_size must either be a single Int or tuple of three Ints")
  int64_t outputT = output_size[0];
  int64_t outputH = output_size[1];
  int64_t outputW = output_size[2];
  int64_t poolSizeT = pool_size[0];
  int64_t poolSizeH = pool_size[1];
  int64_t poolSizeW = pool_size[2];

  int64_t numBatch = 1;
  int64_t planeDim = 0;
  int64_t timeDim = 1;
  int64_t heightDim = 2;
  int64_t widthDim = 3;

  int64_t ndims = input.ndimension();
  const auto memory_format = input.suggest_memory_format();
  if (memory_format == at::MemoryFormat::ChannelsLast3d) {
    TORCH_CHECK(ndims == 5,
        "non-empty 5D (batch mode) tensor expected for input with channels_last_3d layout");
  } else if (memory_format == at::MemoryFormat::Contiguous) {
    TORCH_CHECK((ndims == 4 || ndims == 5),
        "non-empty 4D or 5D (batch mode) tensor expected for input");
  } else {
    TORCH_CHECK(false, "Unsupport memory format. Supports only ChannelsLast3d, Contiguous");
  }

  if (ndims == 5) {
    numBatch = input.size(0);
    planeDim++;
    timeDim++;
    heightDim++;
    widthDim++;
  }

  /* sizes */
  int64_t numPlanes = input.size(planeDim);
  int64_t inputT = input.size(timeDim);
  int64_t inputH = input.size(heightDim);
  int64_t inputW = input.size(widthDim);

  TORCH_CHECK(outputT + poolSizeT - 1 < inputT,
           "fractional_max_pool3d_out(): pool time ", poolSizeT,
           " too large relative to input time ", inputT);
  TORCH_CHECK(outputW + poolSizeW - 1 < inputW,
           "fractional_max_pool3d_out(): pool width ", poolSizeW,
           " too large relative to input width ", inputW);
  TORCH_CHECK(outputH + poolSizeH - 1 < inputH,
           "fractional_max_pool3d_out(): pool height ", poolSizeH,
           " too large relative to input height ", inputH);

  if (ndims == 4) {
    set_output(0, {numPlanes, outputT, outputH, outputW}, input.options());
    /* indices will contain the locations for each output point */
    set_output(1, {numPlanes, outputT, outputH, outputW}, input.options().dtype(kLong));
  } else {
    set_output(0, {numBatch, numPlanes, outputT, outputH, outputW}, input.options().memory_format(memory_format));
    /* indices will contain the locations for each output point */
    set_output(1, {numBatch, numPlanes, outputT, outputH, outputW}, input.options().memory_format(memory_format).dtype(kLong));
  }
}

} // namespace meta

namespace native {
namespace {

template<typename scalar_t>
static std::vector<int> generate_intervals(
  scalar_t sample,
  int64_t inputSize,
  int64_t outputSize,
  int64_t poolSize) {
  std::vector<int> sequence(outputSize);
  if (outputSize > 1) {
    scalar_t alpha = static_cast<scalar_t>(inputSize - poolSize) /
      static_cast<scalar_t>(outputSize - 1);

    for (const auto i : c10::irange(outputSize - 1)) {
      sequence[i] =
        static_cast<int>((i + sample) * alpha) - static_cast<int>(sample * alpha);
    }
  }
  sequence[outputSize - 1] = inputSize - poolSize;

  return sequence;
}

template<typename scalar_t>
static void fractional_max_pool3d_contiguous(
  scalar_t* input,
  scalar_t* output,
  int64_t* indices,
  scalar_t* randomSamples,
  int64_t numBatch, int64_t numPlanes,
  int64_t inputT, int64_t inputH, int64_t inputW,
  int64_t outputT, int64_t outputH, int64_t outputW,
  int64_t poolSizeT, int64_t poolSizeH, int64_t poolSizeW) {

  at::parallel_for(0, numBatch * numPlanes, 0, [&](int64_t begin, int64_t end) {
    int64_t n{0}, c{0};
    data_index_init(begin, n, numBatch, c, numPlanes);

    for (int64_t i = begin; i < end; i++) {
      /* each plane contains 3 random samples,
         one for T, one for W, and one for H */
      scalar_t* randomSamplesForPlane = randomSamples + i * 3;

      /* Generate interval sequence */
      auto sequenceT = generate_intervals<scalar_t>(
          randomSamplesForPlane[0], inputT, outputT, poolSizeT);
      auto sequenceH = generate_intervals<scalar_t>(
          randomSamplesForPlane[1], inputH, outputH, poolSizeH);
      auto sequenceW = generate_intervals<scalar_t>(
          randomSamplesForPlane[2], inputW, outputW, poolSizeW);

      /* local pointers for each plane */
      scalar_t* input_ptr = input + i * inputT * inputH * inputW;
      scalar_t* output_ptr = output + i * outputT * outputH * outputW;
      int64_t* indices_ptr = indices + i * outputT * outputH * outputW;

      for (int64_t ot = 0; ot < outputT; ot++) {
        int64_t it0 = sequenceT[ot];

        for (int64_t oh = 0; oh < outputH; oh++) {
          int64_t ih0 = sequenceH[oh];

          for (int64_t ow = 0; ow < outputW; ow++) {
            int64_t iw0 = sequenceW[ow];

            scalar_t maxVal = -std::numeric_limits<scalar_t>::infinity();
            int64_t maxIndex = it0 * inputH * inputW + ih0 * inputW + iw0;
            for (int64_t it = it0; it < it0 + poolSizeT; it++) {
              AT_ASSERT(it >= 0 && it < inputT);
              for (int64_t ih = ih0; ih < ih0 + poolSizeH; ih++) {
                AT_ASSERT(ih >= 0 && ih < inputH);
                for (int64_t iw = iw0; iw < iw0 + poolSizeW; iw++) {
                  AT_ASSERT(iw >= 0 && iw < inputW);

                  int64_t index =  it * inputH * inputW + ih * inputW + iw;
                  scalar_t val = input_ptr[index];
                  if (val > maxVal || std::isnan(val)) {
                    maxVal = val;
                    maxIndex = index;
                  }
                }
              }
            }
            output_ptr[ot * outputH * outputW + oh * outputW + ow] = maxVal;
            indices_ptr[ot * outputH * outputW + oh * outputW + ow] = maxIndex;
          }
        }
      }

      data_index_step(n, numBatch, c, numPlanes);
    }
  });
}

template<typename scalar_t>
static void fractional_max_pool3d_channels_last(
  scalar_t* input,
  scalar_t* output,
  int64_t* indices,
  scalar_t* randomSamples,
  int64_t numBatch, int64_t numPlanes,
  int64_t inputT, int64_t inputH, int64_t inputW,
  int64_t outputT, int64_t outputH, int64_t outputW,
  int64_t poolSizeT, int64_t poolSizeH, int64_t poolSizeW) {

  scalar_t alphaT = (outputT == 1) ? static_cast<scalar_t>(1)
      : static_cast<scalar_t>(inputT - poolSizeT) / static_cast<scalar_t>(outputT - 1);
  scalar_t alphaH = (outputH == 1) ? static_cast<scalar_t>(1)
      : static_cast<scalar_t>(inputH - poolSizeH) / static_cast<scalar_t>(outputH - 1);
  scalar_t alphaW = (outputW == 1) ? static_cast<scalar_t>(1)
      : static_cast<scalar_t>(inputW - poolSizeW) / static_cast<scalar_t>(outputW - 1);

  int64_t stride_n = inputT * inputH * inputW * numPlanes;
  at::parallel_for(0, numBatch * outputT * outputH * outputW, 0, [&](int64_t begin, int64_t end) {
    int64_t n{0}, ot{0}, oh{0}, ow{0};
    data_index_init(begin, n, numBatch, ot, outputT, oh, outputH, ow, outputW);

    for (int64_t i = begin; i < end; i++) {
      for (int64_t c = 0; c < numPlanes; c++) {
        scalar_t* randomSamplesForPlane = randomSamples + n * numPlanes * 3 + c * 3;
        scalar_t sampleT = randomSamplesForPlane[0];
        scalar_t sampleH = randomSamplesForPlane[1];
        scalar_t sampleW = randomSamplesForPlane[2];

        int64_t it0 = (ot == outputT - 1) ? inputT - poolSizeT
            : static_cast<int64_t>((ot + sampleT) * alphaT) - static_cast<int64_t>(sampleT * alphaT);
        int64_t ih0 = (oh == outputH - 1) ? inputH - poolSizeH
            : static_cast<int64_t>((oh + sampleH) * alphaH) - static_cast<int64_t>(sampleH * alphaH);
        int64_t iw0 = (ow == outputW - 1) ? inputW - poolSizeW
            : static_cast<int64_t>((ow + sampleW) * alphaW) - static_cast<int64_t>(sampleW * alphaW);

        scalar_t maxVal = -std::numeric_limits<scalar_t>::infinity();
        int64_t maxIndex = it0 * inputH * inputW + ih0 * inputW + iw0;
        for (int64_t it = it0; it < it0 + poolSizeT; it++) {
          AT_ASSERT(it >= 0 && it < inputT);
          for (int64_t ih = ih0; ih < ih0 + poolSizeH; ih++) {
            AT_ASSERT(ih >= 0 && ih < inputH);
            for (int64_t iw = iw0; iw < iw0 + poolSizeW; iw++) {
              AT_ASSERT(iw >= 0 && iw < inputW);

              int64_t index =  it * inputH * inputW + ih * inputW + iw;
              scalar_t val = input[n * stride_n + index * numPlanes + c];
              if (val > maxVal || std::isnan(val)) {
                maxVal = val;
                maxIndex = index;
              }
            }
          }
        }
        output[i * numPlanes + c] = maxVal;
        indices[i * numPlanes + c] = maxIndex;
      }

      data_index_step(n, numBatch, ot, outputT, oh, outputH, ow, outputW);
    }
  });
}

template<typename scalar_t>
static void fractional_max_pool3d_backward_out_single_batch_frame(
  scalar_t* gradInput,
  scalar_t* gradOutput,
  int64_t* indices,
  int64_t numPlanes,
  int64_t inputT, int64_t inputH, int64_t inputW,
  int64_t outputT, int64_t outputH, int64_t outputW) {

  at::parallel_for(0, numPlanes, 0, [&](int64_t start, int64_t end) {
    for (auto plane = start; plane < end; plane++) {
      scalar_t* gradInputForPlane = gradInput + plane * inputT * inputH * inputW;
      scalar_t* gradOutputForPlane = gradOutput +
                  plane * outputT * outputH * outputW;
      int64_t* indicesForPlane = indices + plane * outputT * outputH * outputW;

      // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
      int64_t h, w, t;
      for (t = 0; t < outputT; ++t) {
        for (h = 0; h < outputH; ++h) {
          for (w = 0; w < outputW; ++w) {
            int64_t outputIndex = t * outputH * outputW + h * outputW + w;
            int64_t index = indicesForPlane[outputIndex];
            AT_ASSERT(index >= 0 && index < inputT * inputH * inputW);
            gradInputForPlane[index] += gradOutputForPlane[outputIndex];
          }
        }
      }
    }
  });
}

template<typename scalar_t>
static void fractional_max_pool3d_backward_out_frame(
  scalar_t* gradInput,
  scalar_t* gradOutput,
  int64_t* indices,
  int64_t numBatch, int64_t numPlanes,
  int64_t inputT, int64_t inputH, int64_t inputW,
  int64_t outputT, int64_t outputH, int64_t outputW) {
    if(numBatch == 1) {
      fractional_max_pool3d_backward_out_single_batch_frame<scalar_t>(
        gradInput, gradOutput, indices,
        numPlanes,
        inputT, inputH, inputW,
        outputT, outputH, outputW
      );
      return;
    }

    at::parallel_for(0, numBatch, 0, [&](int64_t start, int64_t end) {
      for (auto batch = start; batch < end; ++batch) {
        fractional_max_pool3d_backward_out_single_batch_frame<scalar_t>(
          gradInput + batch * numPlanes * inputW * inputH * inputT,
          gradOutput + batch * numPlanes * outputW * outputH * outputT,
          indices + batch * numPlanes * outputW * outputH * outputT,
          numPlanes,
          inputT, inputH, inputW,
          outputT, outputH, outputW
        );
      }
    });
  }


void fractional_max_pool3d_backward_out_cpu_template(
  const Tensor& input,
  const Tensor& gradOutput_,
  Tensor& gradInput,
  IntArrayRef output_size,
  IntArrayRef pool_size /* unused */,
  const Tensor& indices_) {

  int64_t outputT = output_size[0];
  int64_t outputH = output_size[1];
  int64_t outputW = output_size[2];

  int64_t numBatch = 1;
  int64_t planeDim = 0;
  int64_t timeDim = 1;
  int64_t heightDim = 2;
  int64_t widthDim = 3;

  int64_t ndims = input.ndimension();
  if (ndims == 5) {
    numBatch = input.size(0);
    planeDim = 1;
    heightDim++;
    widthDim++;
    timeDim++;
  }

  /* sizes */
  int64_t numPlanes = input.size(planeDim);
  int64_t inputT = input.size(timeDim);
  int64_t inputH = input.size(heightDim);
  int64_t inputW = input.size(widthDim);

  TORCH_CHECK(outputT == gradOutput_.size(timeDim),
           "fractional_max_pool3d_backward_out(): gradOutput time unexpected");
  TORCH_CHECK(outputH == gradOutput_.size(heightDim),
           "fractional_max_pool3d_backward_out(): ",
           "gradOutput height unexpected");
  TORCH_CHECK(outputW == gradOutput_.size(widthDim),
           "fractional_max_pool3d_backward_out(): gradOutput width unexpected");

  /* get contiguous gradOutput */
  auto gradOutput = gradOutput_.contiguous();
  auto indices = indices_.contiguous();

  /* resize */
  gradInput.resize_as_(input);
  gradInput.zero_();

  /* backprop */
  AT_DISPATCH_FLOATING_TYPES(
    input.scalar_type(),
    "fractional_max_pool3d_backward_out_frame",
    [&]{
      fractional_max_pool3d_backward_out_frame<scalar_t>(
        gradInput.data_ptr<scalar_t>(),
        gradOutput.data_ptr<scalar_t>(),
        indices.data_ptr<int64_t>(),
        numBatch, numPlanes,
        inputT, inputH, inputW,
        outputT, outputH, outputW
      );
    }
  );
}

}// namespace

TORCH_IMPL_FUNC(fractional_max_pool3d_out_cpu) (
  const at::Tensor& input_,
  IntArrayRef pool_size,
  IntArrayRef output_size,
  const at::Tensor& randomSamples,
  const at::Tensor& output,
  const at::Tensor& indices) {

  int64_t outputT = output_size[0];
  int64_t outputH = output_size[1];
  int64_t outputW = output_size[2];
  int64_t poolSizeT = pool_size[0];
  int64_t poolSizeH = pool_size[1];
  int64_t poolSizeW = pool_size[2];

  int64_t numBatch = 1;
  int64_t planeDim = 0;
  int64_t timeDim = 1;
  int64_t heightDim = 2;
  int64_t widthDim = 3;

  /* get contiguous input */
  auto memory_format = input_.suggest_memory_format();
  auto input = input_.contiguous(memory_format);

  int64_t ndims = input.ndimension();
  if (ndims == 5) {
    numBatch = input_.size(0);
    planeDim++;
    timeDim++;
    heightDim++;
    widthDim++;
  }

  /* sizes */
  int64_t numPlanes = input_.size(planeDim);
  int64_t inputT = input_.size(timeDim);
  int64_t inputH = input_.size(heightDim);
  int64_t inputW = input_.size(widthDim);

  switch (input.suggest_memory_format()) {
    case at::MemoryFormat::Contiguous: {
      AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "fractional_max_pool3d_contiguous", [&] {
        fractional_max_pool3d_contiguous<scalar_t>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            indices.data_ptr<int64_t>(),
            randomSamples.data_ptr<scalar_t>(),
            numBatch, numPlanes,
            inputT, inputH, inputW,
            outputT, outputH, outputW,
            poolSizeT, poolSizeH, poolSizeW);
      });
      break;
    }
    case at::MemoryFormat::ChannelsLast3d: {
      AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "fractional_max_pool3d_channels_last", [&] {
        fractional_max_pool3d_channels_last<scalar_t>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            indices.data_ptr<int64_t>(),
            randomSamples.data_ptr<scalar_t>(),
            numBatch, numPlanes,
            inputT, inputH, inputW,
            outputT, outputH, outputW,
            poolSizeT, poolSizeH, poolSizeW);
      });
      break;
    }
    default:
      TORCH_CHECK(false, "Unsupported memory format. Supports only ChannelsLast3d, Contiguous");
  }
}

Tensor& fractional_max_pool3d_backward_out_cpu(const at::Tensor& gradOutput_,
  const at::Tensor& input,
  IntArrayRef pool_size,
  IntArrayRef output_size,
  const at::Tensor& indices,
  at::Tensor& gradInput) {
  fractional_max_pool3d_backward_out_cpu_template(
    input,
    gradOutput_,
    gradInput,
    output_size,
    pool_size,
    indices);
  return gradInput;
}

Tensor fractional_max_pool3d_backward_cpu(
  const at::Tensor& gradOutput_,
  const at::Tensor& input,
  IntArrayRef pool_size,
  IntArrayRef output_size,
  const at::Tensor& indices) {
  Tensor gradInput = at::empty({0}, input.options());
  fractional_max_pool3d_backward_out_cpu_template(
    input,
    gradOutput_,
    gradInput,
    output_size,
    pool_size,
    indices);
  return gradInput;
}

}// native
}// at
