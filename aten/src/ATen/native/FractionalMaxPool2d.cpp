#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Parallel.h>

#include <tuple>
#include <vector>

namespace at {
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
static void fractional_max_pool2d_out_single_batch_frame(
  scalar_t* input,
  scalar_t* output,
  int64_t* indices,
  scalar_t* randomSamples,
  int numPlanes,
  int inputW, int inputH,
  int outputW, int outputH,
  int poolSizeW, int poolSizeH) {
  at::parallel_for(0, numPlanes, 0, [&](int64_t start, int64_t end) {
    for (auto plane = start; plane < end; ++plane) {
      /* each plane contains 2 random samples, one for W and one for H */
      scalar_t* randomSamplesForPlane = randomSamples + plane * 2;

      /* Generate interval sequence */
      auto sequenceW = fractional_max_pool2d_generate_intervals<scalar_t>(
          randomSamplesForPlane[0], inputW, outputW, poolSizeW);
      auto sequenceH = fractional_max_pool2d_generate_intervals<scalar_t>(
          randomSamplesForPlane[1], inputH, outputH, poolSizeH);

      /* loop over output */
      int h, w;

      scalar_t* inputForPlane = input + plane * inputW * inputH;
      scalar_t* outputForPlane = output + plane * outputW * outputH;
      int64_t* indicesForPlane = indices + plane * outputW * outputH;

      for (h = 0; h < outputH; ++h) {
        int inputHStart = sequenceH[h];

        for (w = 0; w < outputW; ++w) {
          int inputWStart = sequenceW[w];

          int h2 = inputHStart, w2 = inputWStart;
          scalar_t maxVal = -std::numeric_limits<scalar_t>::infinity();
          int64_t maxIndex = h2 * inputW + w2;

          for (h2 = inputHStart; h2 < inputHStart + poolSizeH; ++h2) {
            for (w2 = inputWStart; w2 < inputWStart + poolSizeW; ++w2) {
              AT_ASSERT(h2 >= 0 && h2 < inputH);
              AT_ASSERT(w2 >= 0 && w2 < inputW);

              int planeIndex = h2 * inputW + w2;
              scalar_t val = inputForPlane[planeIndex];
              if (val > maxVal || std::isnan(val)) {
                maxVal = val;
                maxIndex = planeIndex;
              }
            }
          }

          outputForPlane[h * outputW + w] = maxVal;
          indicesForPlane[h * outputW + w] = maxIndex;
        }
      }
    }
  });
}

template <typename scalar_t>
static void fractional_max_pool2d_out_frame(
  scalar_t* input,
  scalar_t* output,
  int64_t* indices,
  scalar_t* randomSamples,
  int numBatch, int numPlanes,
  int inputW, int inputH,
  int outputW, int outputH,
  int poolSizeW, int poolSizeH) {
    if(numBatch == 1) {
      fractional_max_pool2d_out_single_batch_frame<scalar_t>(
        input,
        output,
        indices,
        randomSamples,
        numPlanes, inputW, inputH, outputW, outputH, poolSizeW, poolSizeH
      );
      return;
    }
    at::parallel_for(0, numBatch, 0, [&](int64_t start, int64_t end) {
      for (auto batch = start; batch < end; ++batch) {
        fractional_max_pool2d_out_single_batch_frame<scalar_t>(
          input + batch * numPlanes * inputH * inputW,
          output + batch * numPlanes * outputH * outputW,
          indices + batch * numPlanes * outputH * outputW,
          randomSamples + batch * numPlanes * 2,
          numPlanes, inputW, inputH, outputW, outputH, poolSizeW, poolSizeH);
      }
    });
  }

void fractional_max_pool2d_out_cpu_template(
  const at::Tensor& input_,
  at::Tensor& output,
  IntArrayRef output_size,
  IntArrayRef pool_size,
  at::Tensor& indices,
  const at::Tensor& randomSamples) {
  TORCH_CHECK(
      pool_size.size() == 2,
      "fractional_max_pool2d: kernel_size must either be a single Int or tuple of Ints")
  TORCH_CHECK(
      output_size.size() == 2,
      "fractional_max_pool2d: output_size must either be a single Int or tuple of Ints")
  int numBatch = 1;
  int planeDim = 0;
  int heightDim = 1;
  int widthDim = 2;
  int outputH = output_size[0];
  int outputW = output_size[1];
  int poolSizeH = pool_size[0];
  int poolSizeW = pool_size[1];

  /* get contiguous input */
  auto input = input_.contiguous();

  int ndims = input.ndimension();
  TORCH_CHECK(input.numel() > 0 && (ndims == 3 || ndims == 4),
    "non-empty 3D or 4D (batch mode) tensor expected for input, but got: ",
    ndims);

  if (ndims == 4) {
    numBatch = input.size(0);
    planeDim++;
    heightDim++;
    widthDim++;
  }

  /* sizes */
  int numPlanes = input.size(planeDim);
  int inputH = input.size(heightDim);
  int inputW = input.size(widthDim);

  TORCH_CHECK(outputH + poolSizeH - 1 <= inputH,
    "fractional_max_pool2d(): pool height ", poolSizeH,
    " too large relative to input height ", inputH);
  TORCH_CHECK(outputW + poolSizeW - 1 <= inputW,
    "fractional_max_pool2d(): pool width ", poolSizeW,
    " too large relative to input width ", inputW);

  if (ndims == 3) {
    /* resize output */
    output.resize_({numPlanes, outputH, outputW});
    /* indices will contain the locations for each output point */
    indices.resize_({numPlanes, outputH, outputW});
  } else {
    output.resize_({numBatch, numPlanes, outputH, outputW});
    /* indices will contain the locations for each output point */
    indices.resize_({numBatch, numPlanes, outputH, outputW});
  }

  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(),
  "fractional_max_pool2d_out_frame", [&] {
    auto input_data = input.data_ptr<scalar_t>();
    auto output_data = output.data_ptr<scalar_t>();
    auto indices_data = indices.data_ptr<int64_t>();
    auto randomSamples_data = randomSamples.data_ptr<scalar_t>();
    fractional_max_pool2d_out_frame<scalar_t>(
      input_data,
      output_data,
      indices_data,
      randomSamples_data,
      numBatch, numPlanes,
      inputW, inputH,
      outputW, outputH,
      poolSizeW, poolSizeH);
    }
  );
}

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
  const at::Tensor& indices) {

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

std::tuple<Tensor&, Tensor&> fractional_max_pool2d_out_cpu(
    const at::Tensor& input,
    IntArrayRef pool_size,
    IntArrayRef output_size,
    const at::Tensor& randomSamples,
    at::Tensor& output,
    at::Tensor& indices) {
  fractional_max_pool2d_out_cpu_template(
    input,
    output,
    output_size,
    pool_size,
    indices,
    randomSamples);
  return std::tuple<Tensor&, Tensor&>(output, indices);
}

std::tuple<Tensor, Tensor> fractional_max_pool2d_cpu(
  const at::Tensor& input,
  IntArrayRef pool_size,
  IntArrayRef output_size,
  const at::Tensor& randomSamples)
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
  return std::tuple<Tensor, Tensor>(output, indices);
}

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
