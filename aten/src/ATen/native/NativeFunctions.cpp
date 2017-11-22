#include "ATen/ATen.h"
#include "ATen/NativeFunctions.h"
#include "ATen/WrapDimUtils.h"
#include "ATen/ExpandUtils.h"

namespace at {
namespace native {

Tensor type_as(const Tensor& self, const Tensor& other) {
  return self.toType(other.type());;
}

Tensor expand_as(const Tensor& self, const Tensor& other) {
  return self.expand(other.sizes());
}

std::vector<Tensor> split(const Tensor& self, int64_t split_size, int64_t dim) {
  int64_t dim_size = self.size(dim);
  int64_t num_splits = (dim_size + split_size - 1) / split_size;
  std::vector<Tensor> splits(num_splits);
  int64_t last_split_size = split_size - (split_size * num_splits - dim_size);

  for (int64_t i = 0; i < num_splits; ++i) {
    auto length = i < num_splits - 1 ? split_size : last_split_size;
    splits[i] = self.narrow(dim, i * split_size, length);
  }
  return splits;
}

Tensor slice(const Tensor& self, int64_t start, int64_t end, int64_t step, int64_t dim) {
  int64_t ndim = self.dim();
  AT_ASSERT(ndim > 0, "slice() cannot be applied to a 0-dim tensor.");
  dim = maybe_wrap_dim(dim, ndim);
  auto sizes = std::vector<int64_t>(self.sizes());
  auto strides = std::vector<int64_t>(self.strides());
  if (step <= 0) {
    // TODO: support negative strides
    throw std::runtime_error("slice step must be positive");
  }
  if (start < 0) {
    start += sizes[dim];
  }
  if (end < 0) {
    end += sizes[dim];
  }
  if (start < 0) {
    start = 0;
  } else if (start >= sizes[dim]) {
    start = sizes[dim];
  }
  if (end < start) {
    end = start;
  } else if (end >= sizes[dim]) {
    end = sizes[dim];
  }
  auto storage_offset = self.storage_offset() + start * strides[dim];
  auto len = end - start;
  sizes[dim] = (len + step - 1) / step;  // round-up
  strides[dim] *= step;
  return self.as_strided(sizes, strides, storage_offset);
}

Tensor narrow(const Tensor& self, int64_t dim, int64_t start, int64_t length) {
  AT_ASSERT(self.dim() > 0, "narrow() cannot be applied to a 0-dim tensor.");
  auto cur_size = self.size(dim);
  if (start < 0 || start >= cur_size) {
    runtime_error("start out of range");
  }
  if (length <= 0 || start > cur_size - length) {
    runtime_error("length out of range");
  }
  return at::native::slice(self, start, start + length, 1, dim);
}

Tensor select(const Tensor& self, int64_t dim, int64_t index) {
  int64_t ndim = self.dim();
  AT_ASSERT(ndim > 0, "select() cannot be applied to a 0-dim tensor.");
  dim = maybe_wrap_dim(dim, ndim);
  auto size = self.size(dim);
  if (index < -size || index >= size) {
    std::stringstream ss;
    ss << "select(): index " << index << " out of range for tensor of size ";
    ss << self.sizes() << " at dimension " << dim;
    throw std::runtime_error(ss.str());
  }
  if (index < 0) {
    index += size;
  }
  auto sizes = std::vector<int64_t>(self.sizes());
  auto strides = std::vector<int64_t>(self.strides());
  auto storage_offset = self.storage_offset() + index * strides[dim];
  sizes.erase(sizes.begin() + dim);
  strides.erase(strides.begin() + dim);
  return self.as_strided(sizes, strides, storage_offset);
}

std::vector<Tensor> chunk(const Tensor& self, int64_t chunks, int64_t dim) {
  int64_t split_size = (self.size(dim) + chunks - 1) / chunks;
  // ensure this is dispatched through Tensor/Type, rather than the native function directly.
  return self.split(split_size, dim);
}

int64_t size(const Tensor& self, int64_t dim) {
  dim = maybe_wrap_dim(dim, self.dim());
  // wrap_dim guarantees bounds are correct.
  return self.sizes()[dim];
}

int64_t stride(const Tensor& self, int64_t dim) {
  dim = maybe_wrap_dim(dim, self.dim());
  // wrap_dim guarantees bounds are correct.
  return self.strides()[dim];
}

bool is_same_size(const Tensor& self, const Tensor& other) {
  return self.sizes().equals(other.sizes());
}

bool is_cuda(const Tensor& self) {
  return self.type().is_cuda();
}

bool is_distributed(const Tensor& self) {
  return self.type().is_distributed();
}

bool is_sparse(const Tensor& self) {
  return self.type().is_sparse();
}

Tensor permute(const Tensor& self, IntList dims) {
  auto nDims = self.dim();
  if (dims.size() != (size_t)nDims) {
    runtime_error("number of dims don't match in permute");
  }
  auto oldSizes = self.sizes();
  auto oldStrides = self.strides();
  std::vector<int64_t> newSizes(nDims);
  std::vector<int64_t> newStrides(nDims);
  std::vector<bool> seen(nDims);
  for (int64_t i = 0; i < nDims; i++) {
    auto dim = maybe_wrap_dim(dims[i], nDims);
    if (seen[dim]) {
      runtime_error("repeated dim in permute");
    }
    seen[dim] = true;
    newSizes[i] = oldSizes[dim];
    newStrides[i] = oldStrides[dim];
  }
  return self.as_strided(newSizes, newStrides);
}

Tensor expand(const Tensor& self, IntList size) {
  if (size.size() < (size_t)self.dim()) {
    std::ostringstream ss;
    ss << "expand(" << self.type() << "{" << self.sizes() << "}, size=" << size
       << "): the number of sizes provided (" << size.size() << ") "
       << "must be greater or equal to the number of dimensions in the tensor ("
       << self.dim() << ")";
    throw std::runtime_error(ss.str());
  }

  std::vector<int64_t> expandedSizes;
  std::vector<int64_t> expandedStrides;
  std::tie(expandedSizes, expandedStrides) = inferExpandGeometry(self, size);

  return self.as_strided(expandedSizes, expandedStrides);
}

std::tuple<std::vector<int64_t>, std::vector<int64_t> >
inferSqueezeGeometry(const Tensor &tensor) {
  std::vector<int64_t> sizes;
  std::vector<int64_t> strides;

  for(int64_t d = 0; d < tensor.dim(); d++) {
    if(tensor.sizes()[d] != 1) {
      sizes.push_back(tensor.sizes()[d]);
      strides.push_back(tensor.strides()[d]);
    }
  }

  return std::make_tuple(sizes, strides);
}

std::tuple<std::vector<int64_t>, std::vector<int64_t> >
inferSqueezeGeometry(const Tensor& tensor, int64_t dim) {
  std::vector<int64_t> sizes;
  std::vector<int64_t> strides;

  for(int64_t d = 0; d < tensor.dim(); d++) {
    if(d != dim || tensor.sizes()[dim] != 1) {
      sizes.push_back(tensor.sizes()[d]);
      strides.push_back(tensor.strides()[d]);
    }
  }
  return std::make_tuple(sizes, strides);
}

std::tuple<std::vector<int64_t>, std::vector<int64_t> >
inferUnsqueezeGeometry(const Tensor& tensor, int64_t dim) {
  if (tensor.numel() == 0) {
    throw std::runtime_error("cannot unsqueeze empty tensor");
  }

  std::vector<int64_t> sizes(tensor.sizes());
  std::vector<int64_t> strides(tensor.strides());
  int64_t new_stride = dim >= tensor.dim() - 1 ? 1 : sizes[dim] * strides[dim];
  sizes.insert(sizes.begin() + dim, 1);
  strides.insert(strides.begin() + dim, new_stride);

  return std::make_tuple(sizes, strides);
}

Tensor squeeze(const Tensor& self) {
  auto g = inferSqueezeGeometry(self);
  return self.as_strided(std::get<0>(g), std::get<1>(g));
}

Tensor squeeze(const Tensor& self, int64_t dim) {
  dim = maybe_wrap_dim(dim, self.dim());

  if (self.sizes()[dim] != 1) {
    return self.as_strided(self.sizes().vec(), self.strides().vec());
  }
  auto g = inferSqueezeGeometry(self, dim);
  return self.as_strided(std::get<0>(g), std::get<1>(g));
}

Tensor & squeeze_(Tensor& self) {
  auto g = inferSqueezeGeometry(self);
  return self.as_strided_(std::get<0>(g), std::get<1>(g));
}

Tensor & squeeze_(Tensor& self, int64_t dim) {
  dim = maybe_wrap_dim(dim, self.dim());

  if (self.sizes()[dim] != 1) {
    return self.as_strided_(self.sizes().vec(), self.strides().vec());
  }
  auto g = inferSqueezeGeometry(self, dim);
  return self.as_strided_(std::get<0>(g), std::get<1>(g));
}

Tensor unsqueeze(const Tensor& self, int64_t dim) {
  dim = maybe_wrap_dim(dim, self.dim() + 1);

  auto g = inferUnsqueezeGeometry(self, dim);
  return self.as_strided(std::get<0>(g), std::get<1>(g));
}

Tensor & unsqueeze_(Tensor& self, int64_t dim) {
  dim = maybe_wrap_dim(dim, self.dim() + 1);

  auto g = inferUnsqueezeGeometry(self, dim);
  return self.as_strided_(std::get<0>(g), std::get<1>(g));
}

Tensor stack(TensorList tensors, int64_t dim) {
  if (tensors.size() == 0) {
    throw std::runtime_error("stack expects a non-empty TensorList");
  }
  dim = maybe_wrap_dim(dim, tensors[0].dim() + 1);

  std::vector<Tensor> inputs(tensors.size());
  for (size_t i = 0; i < tensors.size(); ++i) {
    inputs[i] = tensors[i].unsqueeze(dim);
  }
  return at::cat(inputs, dim);
}

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
