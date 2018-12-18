#include "ATen/ATen.h"
#include "ATen/NativeFunctions.h"
#include <tuple>
#include <vector>

namespace at {
namespace native {
namespace {

static std::vector<int64_t> fractional_max_pool2d_generate_intervals(
  scalar_t sample,
  int64_t inputSize,
  int64_t outputSize,
  int poolSize) {
  scalar_t alpha = (scalar_t) (inputSize - poolSize) / (scalar_t) (outputSize - 1);
  std::vector<int64_t> sequence(outputSize, 0);

  int64_t i;
  for (i = 0; i < outputSize - 1; ++i) {
    sequence[i] =
      (int64_t) ((i + sample) * alpha) - (int64_t) (sample * alpha);
  }
  sequence[outputSize - 1] = inputSize - poolSize;

  return sequence;
}

static void fractional_max_pool2d_out_frame(
  scalar_t* input,
  scalar_t* output,
  int64_t* indices,
  scalar_t* randomSamples,
  int64_t numPlanes,
  int64_t inputW, int64_t inputH,
  int64_t outputW, int64_t outputH,
  int poolSizeW, int poolSizeH) {
  int64_t plane;
#pragma omp parallel for private(plane)
  for (plane = 0; plane < numPlanes; ++plane) {
    /* each plane contains 2 random samples, one for W and one for H */
    scalar_t* randomSamplesForPlane = randomSamples + plane * 2;

    /* Generate interval sequence */
    auto sequenceW = fractional_max_pool2d_generate_intervals(
        randomSamplesForPlane[0], inputW, outputW, poolSizeW);
    auto sequenceH = fractional_max_pool2d_generate_intervals(
        randomSamplesForPlane[1], inputH, outputH, poolSizeH);

    /* loop over output */
    int64_t h, w;

    scalar_t* inputForPlane = input + plane * inputW * inputH;
    scalar_t* outputForPlane = output + plane * outputW * outputH;
    int64_t* indicesForPlane = indices + plane * outputW * outputH;

    for (h = 0; h < outputH; ++h) {
      int64_t inputHStart = sequenceH[h];

      for (w = 0; w < outputW; ++w) {
        int64_t inputWStart = sequenceW[w];

        scalar_t maxVal = -THInf;
        int64_t maxIndex = -1;

        int64_t h2, w2;
        for (h2 = inputHStart; h2 < inputHStart + poolSizeH; ++h2) {
          for (w2 = inputWStart; w2 < inputWStart + poolSizeW; ++w2) {
            AT_ASSERT(h2 >= 0 && h2 < inputH);//?
            AT_ASSERT(w2 >= 0 && w2 < inputW);

            int64_t planeIndex = h2 * inputW + w2;
            scalar_t val = inputForPlane[planeIndex];
            if (val > maxVal) {
              maxVal = val;
              maxIndex = planeIndex;
            }
          }
        }

        AT_ASSERT(maxVal != -THInf);
        AT_ASSERT(maxIndex != -1);

        outputForPlane[h * outputW + w] = maxVal;
        /* +1 to lua index */
        indicesForPlane[h * outputW + w] = maxIndex + TH_INDEX_BASE;
      }
    }

  }
}

void fractional_max_pool2d_out_cpu_template(
  at::Tensor const& input,
  at::Tensor& output,
  IntList output_size,
  IntList pool_size,
  at::Tensor& indices,
  at::Tensor const& randomSamples) {

  int64_t numBatch = 1;
  int planeDim = 0;
  int heightDim = 1;
  int widthDim = 2;
  int outputW = output_size[0];
  int outputH = output_size[1];
  int poolSizeW = pool_size[0];
  int poolSizeH = pool_size[1];

  int64_t numInputDims = input.ndimension();
  AT_CHECK((numInputDims == 3 || numInputDims == 4),
		"non-empty 3D or 4D (batch mode) tensor expected for input, but got: %s");

  if (numInputDims == 4) {
    numBatch = input.size(0);
    planeDim++;
    heightDim++;
    widthDim++;
  }

  /* sizes */
  int64_t numPlanes = input.size(planeDim);
  int64_t inputH = input.size(heightDim);
  int64_t inputW = input.size(widthDim);

  AT_CHECK(outputH + poolSizeH - 1 <= inputH,
    "fractional_max_pool2d(): poolSizeH ", poolSizeH,
    " too large relative to input height ", inputH);
  AT_CHECK(outputW + poolSizeW - 1 <= inputW,
    "fractional_max_pool2d(): poolSizeW ", poolSizeW,
    " too large relative to input width ", inputW);

  /* get contiguous input */
  auto input = input.contiguous();

  if (numInputDims == 3) {
    /* resize output */
    output.resize_({numPlanes, outputH, outputW});
    /* indices will contain the locations for each output point */
    indices.resize_({numPlanes, outputH, outputW});

    AT_DISPATCH_FLOATING_TYPES(input.type(), "fractional_max_pool2d", [&] {
      auto input_data = input.data<scalar_t>();
      auto output_data = output.data<scalar_t>();
      auto indices_data = indices.data<int64_t>();
      auto randomSamples_data = randomSamples.data<scalar_t>();
      fractional_max_pool2d_out_frame(
        input_data,
        output_data,
        indices_data,
        randomSamples_data,
        numPlanes, inputW, inputH, outputW, outputH, poolSizeW, poolSizeH);
      });
  } else {
    output.resize_({numBatch, numPlanes, outputH, outputW});
    /* indices will contain the locations for each output point */
    indices.resize_({numBatch, numPlanes, outputH, outputW});

    int64_t batch;
#pragma omp parallel for private(batch)
    for (batch = 0; batch < numBatch; ++batch) {
      AT_DISPATCH_FLOATING_TYPES(input.type(), "fractional_max_pool2d", [&] {
        auto input_data = input.data<scalar_t>();
        auto output_data = output.data<scalar_t>();
        auto indices_data = indices.data<int64_t>();
        auto randomSamples_data = randomSamples.data<scalar_t>();
        fractional_max_pool2d_out_frame(
          input_data + batch * numPlanes * inputH * inputW,
          output_data + batch * numPlanes * outputH * outputW,
          indices_data + batch * numPlanes * outputH * outputW,
          randomSamples_data + batch * numPlanes * 2,
          numPlanes, inputW, inputH, outputW, outputH, poolSizeW, poolSizeH);
        });
    }
  }

}

static void fractional_max_pool2d_backward_out_frame(
  scalar_t* gradInput,
  scalar_t* gradOutput,
  int64_t* indices,
  int64_t numPlanes,
  int64_t inputW, int64_t inputH,
  int64_t outputW, int64_t outputH) {
  int64_t plane;
#pragma omp parallel for private(plane)
  for (plane = 0; plane < numPlanes; plane++) {
    scalar_t* gradInputForPlane = gradInput + plane * inputW * inputH;
    scalar_t* gradOutputForPlane = gradOutput + plane * outputW * outputH;
    int64_t* indicesForPlane = indices + plane * outputW * outputH;

    int64_t h, w;
    for (h = 0; h < outputH; ++h) {
      for (w = 0; w < outputW; ++w) {
        int64_t outputIndex = h * outputW + w;
        int64_t index = indicesForPlane[outputIndex] - TH_INDEX_BASE;
        AT_ASSERT(index >= 0 && index < inputW * inputH);

        gradInputForPlane[index] += gradOutputForPlane[outputIndex];
      }
    }
  }
}

Tensor& fractional_max_pool2d_backward_out_cpu_template(
  at::Tensor const& input,
  at::Tensor const& gradOutput_,
  at::Tensor& gradInput,
  IntList output_size,
  IntList pool_size,
  at::Tensor const& indices) {

  int64_t numBatch = 1;
  int planeDim = 0;
  int heightDim = 1;
  int widthDim = 2;

  int outputW = output_size[0];
  int outputH = output_size[1];
  int poolSizeW = pool_size[0];
  int poolSizeH = pool_size[1];

  int64_t numInputDims = input.ndimension();
  if (numInputDims == 4) {
    numBatch = input.size(0);
    planeDim = 1;
    heightDim++;
    widthDim++;
  }

  /* sizes */
  int64_t numPlanes = input.size(planeDim);
  int64_t inputH = input.size(heightDim);
  int64_t inputW = input.size(widthDim);

  AT_CHECK(outputW == gradOutput.size(widthDim),
    "fractional_max_pool2d_backward(): gradOutput width unexpected");
  AT_CHECK(outputH == gradOutput.size(heightDim),
    "fractional_max_pool2d_backward(): gradOutput height unexpected");

  /* get contiguous gradOutput */
  auto gradOutput = gradOutput_.contiguous();

  /* resize */
  gradInput = at::zeros_like(input);

  /* backprop */
  if (numInputDims == 3) {
    AT_DISPATCH_FLOATING_TYPES(
      input.type(), "fractional_max_pool2d_backward", [&] {
        auto gradInput_data = gradInput.data<scalar_t>();
        auto gradOutput_data = gradOutput.data<scalar_t>();
        auto indices_data = indices.data<int64_t>();
        fractional_max_pool2d_backward_out_frame<scalar_t>(
          gradInput_data,
          gradOutput_data,
          indices_data,
          numPlanes, inputW, inputH, outputW, outputH);
        }
      );
  } else {
    int64_t batch;
#pragma omp parallel for private(batch)
    for (batch = 0; batch < numBatch; ++batch) {
      AT_DISPATCH_FLOATING_TYPES(
        input.type(), "fractional_max_pool2d_backward", [&] {
          auto gradInput_data = gradInput.data<scalar_t>();
          auto gradOutput_data = gradOutput.data<scalar_t>();
          auto indices_data = indices.data<int64_t>();
          fractional_max_pool2d_backward_out_frame<scalar_t>(
            gradInput_data + batch * numPlanes * inputH * inputW,
            gradOutput_data + batch * numPlanes * outputH * outputW,
            indices_data + batch * numPlanes * outputH * outputW,
            numPlanes, inputW, inputH, outputW, outputH);
          }
        );
    }
  }
  return gradInput;

}

}// namespace

Tensor& fractional_max_pool2d_out_cpu(
  at::Tensor& output,
  at::Tensor& indices,
  at::Tensor const& input,
  IntList pool_size,
  IntList output_size,
  at::Tensor const& randomSamples)
{
  fractional_max_pool2d_out_cpu_template(
    input,
    output,
    output_size,
    pool_size,
    indices,
    randomSamples);
  return output;
}

Tensor fractional_max_pool2d_cpu(
  at::Tensor const& input,
  IntList pool_size,
  IntList output_size,
  at::Tensor const& randomSamples)
{
  Tensor output = at::empty({0}, input.options());
  Tensor indices = at::empty({0}, input.options().dtype(kLong));
  fractional_max_pool2d_out_cpu_template(
    input,
    output,
    output_size,
    pool_size,
    indices,
    randomSamples);
  return output;
}

Tensor& fractional_max_pool2d_backward_out_cpu(
  at::Tensor& gradInput,
  at::Tensor const& gradOutput_,
  at::Tensor const& input,
  IntList pool_size,
  IntList output_size,
  at::Tensor const& indices) {
{
  gradInput.resize_as_(input);
  fractional_max_pool2d_backward_out_cpu_template(
    intput,
    gradOutput_,
    gradInput,
    output_size,
    pool_size,
    indices);
  return gradInput;
}

Tensor fractional_max_pool2d_backward_cpu(
  at::Tensor const& gradOutput_,
  at::Tensor const& input,
  IntList output_size,
  IntList pool_size,
  at::Tensor const& indices)
{
  Tensor gradInput = at::zeros_like(input);
  fractional_max_pool2d_backward_out_cpu_template(
    intput,
    gradOutput_,
    gradInput,
    output_size,
    pool_size,
    indices);
  return gradInput;
}

}// at::native
}// at
