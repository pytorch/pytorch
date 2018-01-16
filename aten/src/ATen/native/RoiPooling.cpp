#include "ATen/ATen.h"
#include "ATen/NativeFunctions.h"
#include <tuple>

namespace at {
namespace native {

std::tuple<at::Tensor, at::Tensor> RoiPooling2d_forward_cpu(
	const Tensor& input,
	const Tensor& rois,
	int64_t pooledHeight,
	int64_t pooledWidth,
	double spatialScale)
{
  // Input is the output of the last convolutional layer in the Backbone network, so
  // it should be in the format of NCHW
  AT_ASSERT(input.ndimension() == 4, "Input to RoI Pooling should be a NCHW Tensor");

  // ROIs is the set of region proposals to process. It is a 2D Tensor where the first
  // dim is the # of proposals, and the second dim is the proposal itself in the form
  // [batch_index startW startH endW endH]
  AT_ASSERT(rois.ndimension() == 2, "RoI Proposals should be a 2D Tensor, (batch_sz x proposals)");
  AT_ASSERT(rois.size(1) == 5, "Proposals should be of the form [batch_index startW startH endW enH]");

  auto proposals = rois.size(0);
  auto inputChannels = input.size(1);
  auto inputHeight = input.size(2);
  auto inputWidth = input.size(3);

  // Output Tensor is (num_rois, C, pooledHeight, pooledWidth)
  auto output = input.type().tensor({proposals, inputChannels, pooledHeight, pooledWidth});

  // TODO: need some mechanism for determining train vs. test

  // During training, we need to store the argmaxes for the pooling operation, so
  // the argmaxes Tensor should be the same size as the output Tensor
  auto argmaxes = input.type().toScalarType(kInt).tensor({proposals, inputChannels, pooledHeight, pooledWidth});

  AT_ASSERT(input.is_contiguous(), "input must be contiguous");
  AT_ASSERT(rois.is_contiguous(), "rois must be contiguous");

  auto *rawInput = input.data<float>();
  auto inputChannelStride = inputHeight * inputWidth;
  auto inputBatchStride = inputChannels * inputChannelStride;
  auto *rawRois = rois.data<float>();
  auto roiProposalStride = rois.size(1);

  auto *rawOutput = output.data<float>();
  auto *rawArgmaxes = argmaxes.data<int>();
  auto outputChannelStride = pooledHeight * pooledWidth;

  // Now that our Tensors are properly sized, we can perform the pooling operation.
  // We iterate over each RoI and perform pooling on each channel in the input, to
  // generate a pooledHeight x pooledWidth output for each RoI
  for (auto i = 0; i < proposals; ++i) {
    auto n = static_cast<int>(rawRois[0]);
    auto startWidth = static_cast<int>(std::round(rawRois[1] * spatialScale));
    auto startHeight = static_cast<int>(std::round(rawRois[2] * spatialScale));
    auto endWidth = static_cast<int>(std::round(rawRois[3] * spatialScale));
    auto endHeight = static_cast<int>(std::round(rawRois[4] * spatialScale));

    // TODO: assertions for valid values?
    // TODO: fix malformed ROIs??

    auto roiHeight = endHeight - startHeight;
    auto roiWidth = endWidth - startWidth;

    // Because the Region of Interest can be of variable size, but our output
    // must always be (pooledHeight x pooledWidth), we need to split the RoI
    // into a pooledHeight x pooledWidth grid of tiles

    auto tileHeight = static_cast<float>(roiHeight) / static_cast<float>(pooledHeight);
    auto tileWidth = static_cast<float>(roiWidth) / static_cast<float>(pooledWidth);

    auto *rawInputBatch = rawInput + (n * inputBatchStride);

    // Compute pooling for each of the (pooledHeight x pooledWidth) tiles for each
    // channel in the input
    for (auto ch = 0; ch < inputChannels; ++ch) {
      for (auto ph = 0; ph < pooledHeight; ++ph) {
        for (auto pw = 0; pw < pooledWidth; ++pw) {
          auto tileHStart = static_cast<int64_t>(std::floor(ph * tileHeight));
          auto tileWStart =	static_cast<int64_t>(std::floor(pw * tileWidth));
          auto tileHEnd = static_cast<int64_t>(std::ceil((ph + 1) * tileHeight));
          auto tileWEnd = static_cast<int64_t>(std::ceil((pw + 1) * tileWidth));

          // Add tile offsets to RoI offsets, and clip to input boundaries
          tileHStart = std::min(std::max<int64_t>(tileHStart + startHeight, 0), inputHeight);
          tileWStart = std::min(std::max<int64_t>(tileWStart + startWidth, 0), inputWidth);
          tileHEnd = std::min(std::max<int64_t>(tileHEnd + startHeight, 0), inputHeight);
          tileWEnd = std::min(std::max<int64_t>(tileWEnd + startWidth, 0), inputWidth);

          auto poolIndex = (ph * pooledWidth) + pw;

          // If our pooling region is empty, we set the output to 0, otherwise to
          // the min float so we can calculate the max properly
          auto empty = tileHStart >= tileHEnd || tileWStart >= tileWEnd;
          rawOutput[poolIndex] = empty ? 0 : std::numeric_limits<float>::min();

          // Set to -1 so we don't try to backprop to anywhere
          // TODO: make optional for test
          rawArgmaxes[poolIndex] = -1;

          for (auto th = tileHStart; th < tileHEnd; ++th) {
            for (auto tw = tileWStart; tw < tileWEnd; ++tw) {
              auto index = (th * inputWidth) + tw;
              if (rawInputBatch[index] > rawOutput[poolIndex]) {
                rawOutput[poolIndex] = rawInputBatch[index];
                // TODO: make optional for test
                rawArgmaxes[poolIndex] = index;
              }
            }
          }
        }
      }
      // Increment raw pointers by channel stride
      rawInputBatch += inputChannelStride;
      rawOutput += outputChannelStride;
      // TODO: make optional for test
      rawArgmaxes += outputChannelStride;
    }
    // Increment RoI raw pointer
    rawRois += roiProposalStride;
  }

  return std::make_tuple(output, argmaxes);
}

Tensor RoiPooling2d_backward_cpu(
  const Tensor& input,
  const Tensor& rois,
  int64_t pooledHeight,
  int64_t pooledWidth,
  double spatialScale,
  const Tensor& gradOutput,
  const Tensor& argmaxes) {
  throw std::runtime_error("not implemented");
}

}
}
