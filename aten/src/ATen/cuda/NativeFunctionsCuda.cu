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
    int startWHeight = llrintf(rois_offset[2] * spatialScale);
    int endWidth = llrintf(rois_offset[3] * spatialScale);
    int endHeight = llrintf(rois_offset[4] * spatialScale);
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
