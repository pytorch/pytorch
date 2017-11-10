#include "NativeFunctionsCuda.h"

namespace at {
namespace native {

template <typename T>
__global__ void SpatialRoIPooling_forward_kernel(
  const int outputElements,
  const T *input,
  const T *rois,
  const T spatialScale,
  const int inputChannels,
  const int inputHeight,
  const int inputWidth,
  const int pooledHeight,
  const int pooledWidth,
  T *output,
  int *argmaxes)
{
  for (int linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
       linearIndex < outputElements;
       linearIndex += blockDim.x * gridDim.x)
  {
    // Calculate position in output Tensor, i.e. a specific combination
    // of proposal, channel, pool height and pool width
    // TODO: write to improve performance by minimize computation
    int pw = linearIndex % pooledWidth;
    int ph = (linearIndex / pooledWidth) % pooledHeight;
    int ch = (linearIndex / pooledWidth / pooledHeight) % inputChannels;
    int proposal = linearIndex / pooledWidth / pooledHeight / inputChannels;

    // Get particular proposal data
    const T *rois_offset = rois + (proposal * 5);
    int n = rois_offset[0];
    int startWidth = llrintf(rois_offset[1] * spatialScale);
    int startHeight = llrintf(rois_offset[2] * spatialScale);
    int endWidth = llrintf(rois_offset[3] * spatialScale);
    int endHeight = llrintf(rois_offset[4] * spatialScale);

    // TODO: fix malformed RoIs to be 1x1

    int roiHeight = endHeight - startHeight;
    int roiWidth = endWidth - startWidth;

    // Calculate size of tile based on the size of this particular RoI and the
    // output size
    T tileHeight = static_cast<T>(roiHeight) / static_cast<T>(pooledHeight);
    T tileWidth = static_cast<T>(roiWidth) / static_cast<T>(pooledWidth);

    // Calculate offset into the pooled region
    int tileHStart = static_cast<int>(floorf(static_cast<T>(ph) * tileHeight));
    int tileWStart = static_cast<int>(floorf(static_cast<T>(pw) * tileWidth));
    int tileHEnd = static_cast<int>(ceilf(static_cast<T>(ph + 1) * tileHeight));
    int tileWEnd = static_cast<int>(ceilf(static_cast<T>(pw + 1) * tileWidth));

    // Calculate offset into the image itself, based on RoI + pooled offsets,
    // and ensure it falls within image boundaries
    tileHStart = std::min(std::max(tileHStart + startHeight, 0), inputHeight);
    tileWStart = std::min(std::max(tileWStart + startWidth, 0), inputWidth);
    tileHEnd = std::min(std::max(tileHEnd + startHeight, 0), inputHeight);
    tileWEnd = std::min(std::max(tileWEnd + startWidth, 0), inputWidth);

    // If our pooling region is empty, we set the output to 0, otherwise to
    // the min float so we can calculate the max properly
    bool isEmpty = (tileHStart >= tileHEnd) || (tileWStart >= tileWEnd);
    T max = isEmpty ? 0 : std::numeric_limits<T>::min();
    // If nothing is pooled, argmax = -1 causes nothing to be backprop'd
    int maxIdx = -1;

    const T *inputOffset = input + ((n * inputChannels + ch) * inputHeight * inputWidth);
    for (int th = tileHStart; th < tileHEnd; ++th) {
      for (int tw = tileWStart; tw < tileWEnd; ++tw) {
        int index = (th * inputWidth) + tw;
	if (inputOffset[index] > max) {
          max = inputOffset[index];
	  maxIdx = index;
	}
      }
    }
    output[linearIndex] = max;

    // TODO optional argmax
    argmaxes[linearIndex] = maxIdx;
  }
}

std::tuple<Tensor, Tensor> SpatialRoIPooling_forward_cuda(
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

  return std::make_tuple(output, argmaxes);
}

} // at::native
} // at
