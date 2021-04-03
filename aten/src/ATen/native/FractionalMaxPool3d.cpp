#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Parallel.h>

#include <c10/util/irange.h>

#include <tuple>
#include <vector>

namespace at {
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
static void fractional_max_pool3d_out_single_batch_frame(
  scalar_t* input,
  scalar_t* output,
  int64_t* indices,
  scalar_t* randomSamples,
  int64_t numPlanes,
  int64_t inputT, int64_t inputH, int64_t inputW,
  int64_t outputT, int64_t outputH, int64_t outputW,
  int64_t poolSizeT, int64_t poolSizeH, int64_t poolSizeW) {

  at::parallel_for(0, numPlanes, 0, [&](int64_t start, int64_t end) {
    for (auto plane = start; plane < end; ++plane) {
      /* each plane contains 3 random samples,
         one for T, one for W, and one for H */
      scalar_t* randomSamplesForPlane = randomSamples + plane * 3;

      /* Generate interval sequence */
      auto sequenceT = generate_intervals<scalar_t>(
          randomSamplesForPlane[0], inputT, outputT, poolSizeT);
      auto sequenceH = generate_intervals<scalar_t>(
          randomSamplesForPlane[1], inputH, outputH, poolSizeH);
      auto sequenceW = generate_intervals<scalar_t>(
          randomSamplesForPlane[2], inputW, outputW, poolSizeW);

      /* loop over output */
      int64_t t, h, w;

      scalar_t* inputForPlane = input + plane * inputT * inputH * inputW;
      scalar_t* outputForPlane = output + plane * outputT * outputH * outputW;
      int64_t* indicesForPlane = indices + plane * outputT * outputH * outputW;

      for (t = 0; t < outputT; ++t) {
        int64_t inputTStart = sequenceT[t];

        for (h = 0; h < outputH; ++h) {
          int64_t inputHStart = sequenceH[h];

          for (w = 0; w < outputW; ++w) {
            int64_t inputWStart = sequenceW[w];

            int64_t t2 = inputTStart, h2 = inputHStart, w2 = inputWStart;
            scalar_t maxVal = -std::numeric_limits<scalar_t>::infinity();
            int64_t maxIndex = t2 * inputH * inputW + h2 * inputW + w2;

            for (t2 = inputTStart; t2 < inputTStart + poolSizeT; ++t2) {
              for (h2 = inputHStart; h2 < inputHStart + poolSizeH; ++h2) {
                for (w2 = inputWStart; w2 < inputWStart + poolSizeW; ++w2) {
                  AT_ASSERT(t2 >= 0 && t2 < inputT);
                  AT_ASSERT(h2 >= 0 && h2 < inputH);
                  AT_ASSERT(w2 >= 0 && w2 < inputW);

                  int64_t planeIndex = t2 * inputH * inputW + h2 * inputW + w2;
                  scalar_t val = inputForPlane[planeIndex];
                  if (val > maxVal || std::isnan(val)) {
                    maxVal = val;
                    maxIndex = planeIndex;
                  }
                }
              }
            }

            outputForPlane[t * outputH * outputW + h * outputW + w] = maxVal;
            indicesForPlane[t * outputH * outputW + h * outputW + w] = maxIndex;
          }
        }
      }
    }
  });
}

template<typename scalar_t>
static void fractional_max_pool3d_out_frame(
  scalar_t* input,
  scalar_t* output,
  int64_t* indices,
  scalar_t* randomSamples,
  int64_t numBatch, int64_t numPlanes,
  int64_t inputT, int64_t inputH, int64_t inputW,
  int64_t outputT, int64_t outputH, int64_t outputW,
  int64_t poolSizeT, int64_t poolSizeH, int64_t poolSizeW) {
    if(numBatch == 1) {
      fractional_max_pool3d_out_single_batch_frame<scalar_t>(
        input, output, indices, randomSamples,
        numPlanes,
        inputT, inputH, inputW,
        outputT, outputH, outputW,
        poolSizeT, poolSizeH, poolSizeW
      );
      return;
    }

    at::parallel_for(0, numBatch, 0, [&](int64_t start, int64_t end) {
      for (auto batch = start; batch < end; ++batch) {
        fractional_max_pool3d_out_single_batch_frame<scalar_t>(
          input + batch * numPlanes * inputW * inputH * inputT,
          output + batch * numPlanes * outputW * outputH * outputT,
          indices + batch * numPlanes * outputW * outputH * outputT,
          randomSamples + batch * numPlanes * 3,
          numPlanes,
          inputT, inputH, inputW,
          outputT, outputH, outputW,
          poolSizeT, poolSizeH, poolSizeW
        );
      }
    });
  }

void fractional_max_pool3d_out_cpu_template(
  Tensor& output,
  Tensor& indices,
  const Tensor& input_,
  IntArrayRef pool_size,
  IntArrayRef output_size,
  const Tensor& randomSamples) {
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

  int64_t ndims = input_.ndimension();
  TORCH_CHECK(input_.numel() != 0 && (ndims == 4 || ndims == 5),
    "fractional_max_pool3d_out(): non-empty 4D or 5D (batch mode) tensor ",
    " expected for input, but got: ", ndims);

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

  TORCH_CHECK(outputT + poolSizeT - 1 < inputT,
           "fractional_max_pool3d_out(): pool time ", poolSizeT,
           " too large relative to input time ", inputT);
  TORCH_CHECK(outputW + poolSizeW - 1 < inputW,
           "fractional_max_pool3d_out(): pool width ", poolSizeW,
           " too large relative to input width ", inputW);
  TORCH_CHECK(outputH + poolSizeH - 1 < inputH,
           "fractional_max_pool3d_out(): pool height ", poolSizeH,
           " too large relative to input height ", inputH);

  /* get contiguous input */
  auto input = input_.contiguous();

  if (ndims == 4) {
    /* resize output */
    output.resize_({numPlanes, outputT, outputH, outputW});
    /* indices will contain the locations for each output point */
    indices.resize_({numPlanes, outputT, outputH, outputW});
  } else {
    output.resize_({numBatch, numPlanes, outputT, outputH, outputW});
    /* indices will contain the locations for each output point */
    indices.resize_({numBatch, numPlanes, outputT, outputH, outputW});
  }
  AT_DISPATCH_FLOATING_TYPES(
    input.scalar_type(),
    "fractional_max_pool3d_out_frame",
    [&] {
      fractional_max_pool3d_out_frame<scalar_t>(
        input.data_ptr<scalar_t>(),
        output.data_ptr<scalar_t>(),
        indices.data_ptr<int64_t>(),
        randomSamples.data_ptr<scalar_t>(),
        numBatch, numPlanes,
        inputT, inputH, inputW,
        outputT, outputH, outputW,
        poolSizeT, poolSizeH, poolSizeW
      );
    }
  );
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
  const Tensor& indices) {

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

std::tuple<Tensor&, Tensor&> fractional_max_pool3d_out_cpu(const at::Tensor& input,
  IntArrayRef pool_size,
  IntArrayRef output_size,
  const at::Tensor& randomSamples,
  at::Tensor& output,
  at::Tensor& indices) {
  fractional_max_pool3d_out_cpu_template(
    output,
    indices,
    input,
    pool_size,
    output_size,
    randomSamples);
  return std::tuple<Tensor&, Tensor&>(output, indices);
}

std::tuple<Tensor, Tensor> fractional_max_pool3d_cpu(
  const at::Tensor& input,
  IntArrayRef pool_size,
  IntArrayRef output_size,
  const at::Tensor& randomSamples) {
  Tensor output = at::empty(output_size, input.options());
  Tensor indices = at::empty(output_size, at::kLong);
  fractional_max_pool3d_out_cpu_template(
    output,
    indices,
    input,
    pool_size,
    output_size,
    randomSamples);
  return std::tuple<Tensor, Tensor>(output, indices);
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
