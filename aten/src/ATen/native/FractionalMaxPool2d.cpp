#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/TensorMeta.h>
#include <ATen/native/FractionalMaxPooling.h>
#include <c10/util/irange.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/fractional_max_pool2d_backward_native.h>
#include <ATen/ops/fractional_max_pool2d_native.h>
#endif

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
  TORCH_CHECK(ndims == 3 || ndims == 4,
              "fractional_max_pool2d(): Expected 3D or 4D tensor, but got: ", input.sizes());
  for (const auto i : c10::irange(1, ndims)) {
    TORCH_CHECK(input.size(i) > 0,
                "fractional_max_pool2d(): Expected input to have non-zero size for non-batch dimensions, but got",
                input.sizes(), " with dimension ", i, " being empty.");
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
  auto inputW = input.size(widthDim);

  TORCH_CHECK(outputH + poolSizeH - 1 <= inputH,
    "fractional_max_pool2d(): pool height ", poolSizeH,
    " too large relative to input height ", inputH);
  TORCH_CHECK(outputW + poolSizeW - 1 <= inputW,
    "fractional_max_pool2d(): pool width ", poolSizeW,
    " too large relative to input width ", inputW);

  if (ndims == 3) {
    set_output_raw_strided(0, {numPlanes, outputH, outputW}, {}, input.options());
    /* indices will contain the locations for each output point */
    set_output_raw_strided(1, {numPlanes, outputH, outputW}, {}, input.options().dtype(kLong));
  } else {
    set_output_raw_strided(0, {numBatch, numPlanes, outputH, outputW}, {}, input.options());
    /* indices will contain the locations for each output point */
    set_output_raw_strided(1, {numBatch, numPlanes, outputH, outputW}, {}, input.options().dtype(kLong));
  }
}

TORCH_META_FUNC(fractional_max_pool2d_backward)(
  const at::Tensor& gradOutput_,
  const at::Tensor& input,
  IntArrayRef pool_size /* unused */,
  IntArrayRef output_size,
  const at::Tensor& indices) {

  int64_t numBatch = 1;
  int planeDim = 0;
  int heightDim = 1;
  int widthDim = 2;

  auto outputH = output_size[0];
  auto outputW = output_size[1];

  auto ndims = input.ndimension();
  if (ndims == 4) {
    numBatch = input.size(0);
    planeDim = 1;
    heightDim++;
    widthDim++;
  }

  /* sizes */
  auto numPlanes = input.size(planeDim);
  auto inputH = input.size(heightDim);
  auto inputW = input.size(widthDim);

  /* get contiguous gradOutput */
  auto gradOutput = gradOutput_.contiguous();

  auto expectedOutputShape = IntArrayRef(input.sizes().data(), ndims - 2).vec();
  expectedOutputShape.push_back(outputH);
  expectedOutputShape.push_back(outputW);
  TORCH_CHECK(gradOutput.sizes().equals(expectedOutputShape),
    "fractional_max_pool2d_backward(): gradOutput sizes unexpected");
  TORCH_CHECK(indices.sizes().equals(expectedOutputShape),
    "fractional_max_pool2d_backward(): indices sizes unexpected");

  /* resize */
  if (ndims == 3) {
    set_output_raw_strided(0, {numPlanes, inputH, inputW}, {}, input.options());
  } else {
    set_output_raw_strided(0, {numBatch, numPlanes, inputH, inputW}, {}, input.options());
  }
}
} // namespace meta

namespace native {
namespace {

template <typename scalar_t>
static void fractional_max_pool2d_out_single_batch_frame(
  const scalar_t* input,
  scalar_t* output,
  int64_t* indices,
  const scalar_t* randomSamples,
  int numPlanes,
  int inputW, int inputH,
  int outputW, int outputH,
  int poolSizeW, int poolSizeH) {
  at::parallel_for(0, numPlanes, 0, [&](int64_t start, int64_t end) {
    for (const auto plane : c10::irange(start, end)) {
      /* each plane contains 2 random samples, one for W and one for H */
      const scalar_t* randomSamplesForPlane = randomSamples + plane * 2;

      /* Generate interval sequence */
      auto sequenceW = generate_intervals<scalar_t>(
          randomSamplesForPlane[0], inputW, outputW, poolSizeW);
      auto sequenceH = generate_intervals<scalar_t>(
          randomSamplesForPlane[1], inputH, outputH, poolSizeH);

      /* loop over output */
      const scalar_t* inputForPlane = input + plane * inputW * inputH;
      scalar_t* outputForPlane = output + plane * outputW * outputH;
      int64_t* indicesForPlane = indices + plane * outputW * outputH;

      for (int h = 0; h < outputH; ++h) {
        int inputHStart = sequenceH[h];

        for (int w = 0; w < outputW; ++w) {
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
  const scalar_t* input,
  scalar_t* output,
  int64_t* indices,
  const scalar_t* randomSamples,
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
      for (const auto batch : c10::irange(start, end)) {
        fractional_max_pool2d_out_single_batch_frame<scalar_t>(
          input + batch * numPlanes * inputH * inputW,
          output + batch * numPlanes * outputH * outputW,
          indices + batch * numPlanes * outputH * outputW,
          randomSamples + batch * numPlanes * 2,
          numPlanes, inputW, inputH, outputW, outputH, poolSizeW, poolSizeH);
      }
    });
  }

template <typename scalar_t>
static void fractional_max_pool2d_backward_out_single_batch_frame(
  scalar_t* gradInput,
  const scalar_t* gradOutput,
  const int64_t* indices,
  int numPlanes,
  int inputW, int inputH,
  int outputW, int outputH) {
  at::parallel_for(0, numPlanes, 0, [&](int64_t start, int64_t end) {
    for (const auto plane : c10::irange(start, end)) {
      scalar_t* gradInputForPlane = gradInput + plane * inputW * inputH;
      const scalar_t* gradOutputForPlane = gradOutput + plane * outputW * outputH;
      const int64_t* indicesForPlane = indices + plane * outputW * outputH;

      for (int h = 0; h < outputH; ++h) {
        for (int w = 0; w < outputW; ++w) {
          int outputIndex = h * outputW + w;
          int64_t index = indicesForPlane[outputIndex];
          AT_ASSERT(index >= 0 && index < static_cast<int64_t>(inputW) * inputH);

          gradInputForPlane[index] += gradOutputForPlane[outputIndex];
        }
      }
    }
  });
}

template <typename scalar_t>
static void fractional_max_pool2d_backward_out_frame(
  scalar_t* gradInput,
  const scalar_t* gradOutput,
  const int64_t* indices,
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
      for (const auto batch : c10::irange(start, end)) {
        fractional_max_pool2d_backward_out_single_batch_frame<scalar_t>(
          gradInput + batch * numPlanes * inputH * inputW,
          gradOutput + batch * numPlanes * outputH * outputW,
          indices + batch * numPlanes * outputH * outputW,
          numPlanes, inputW, inputH, outputW, outputH);
      }
    });
}

} // anonymous namespace

TORCH_IMPL_FUNC(fractional_max_pool2d_out_cpu) (
  const at::Tensor& input_,
  IntArrayRef pool_size,
  IntArrayRef output_size,
  const at::Tensor& randomSamples_,
  const at::Tensor& output,
  const at::Tensor& indices) {

  fractional_max_pool_check_shape</*ndim*/ 2>(input_, randomSamples_);

  if (output.numel() == 0) {
    return;
  }

  int64_t numBatch = 1;
  int64_t planeDim = 0;
  int64_t heightDim = 1;
  int64_t widthDim = 2;
  int64_t outputH = output_size[0]; // output.size(heightDim)
  int64_t outputW = output_size[1]; // output.size(widthDim)
  int64_t poolSizeH = pool_size[0];
  int64_t poolSizeW = pool_size[1];

  /* get contiguous input and samples */
  auto input = input_.contiguous();
  auto randomSamples = randomSamples_.contiguous();

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

  AT_DISPATCH_FLOATING_TYPES_AND2(
    kBFloat16,
    kHalf,
    input.scalar_type(),
    "fractional_max_pool2d_out_frame", [&] {
      auto input_data = input.const_data_ptr<scalar_t>();
      auto output_data = output.data_ptr<scalar_t>();
      auto indices_data = indices.data_ptr<int64_t>();
      auto randomSamples_data = randomSamples.const_data_ptr<scalar_t>();
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

TORCH_IMPL_FUNC(fractional_max_pool2d_backward_cpu) (
  const at::Tensor& gradOutput_,
  const at::Tensor& input,
  IntArrayRef pool_size,
  IntArrayRef output_size,
  const at::Tensor& indices,
  const at::Tensor& gradInput) {

  gradInput.zero_();

  int64_t numBatch = 1;
  int planeDim = 0;
  int heightDim = 1;
  int widthDim = 2;

  auto outputH = output_size[0];
  auto outputW = output_size[1];

  auto ndims = input.ndimension();
  if (ndims == 4) {
    numBatch = input.size(0);
    planeDim = 1;
    heightDim++;
    widthDim++;
  }

  /* sizes */
  auto numPlanes = input.size(planeDim);
  auto inputH = input.size(heightDim);
  auto inputW = input.size(widthDim);

  /* get contiguous gradOutput */
  auto gradOutput = gradOutput_.contiguous();

  /* backprop */
  AT_DISPATCH_FLOATING_TYPES_AND2(
    kBFloat16,
    kHalf,
    input.scalar_type(), "fractional_max_pool2d_backward_out_frame", [&] {
      auto gradInput_data = gradInput.data_ptr<scalar_t>();
      auto gradOutput_data = gradOutput.const_data_ptr<scalar_t>();
      auto indices_data = indices.const_data_ptr<int64_t>();
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
}

} // at::native
} // at
