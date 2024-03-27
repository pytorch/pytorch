#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/mps/MPSGraphVenturaOps.h>
#include <ATen/native/mps/OperationUtils.h>

#include <ATen/ops/max_unpool2d.h>
#include <ATen/ops/max_unpool2d_native.h>

namespace at::native {
namespace mps {
namespace {
struct CachedGraph : public MPSCachedGraph {
  CachedGraph(MPSGraph* graph) : MPSCachedGraph(graph) {}
  MPSGraphTensor* inputTensor_ = nil;
  MPSGraphTensor* indicesTensor_ = nil;
  MPSGraphTensor* outputTensor_ = nil;
};

std::string getCacheKey(const Tensor& input, const Tensor& indices, const IntArrayRef output_size) {
  return "max_unpooling2d_forward_mps:" + getTensorsStringKey({input, indices}) + "[" + getArrayRefString(output_size) +
      "]";
}

MPSGraphTensor* buildGraph(CachedGraph* cachedGraph, const IntArrayRef output_size) {
  MPSGraph* graph = cachedGraph->graph();
  MPSGraphTensor* inputTensor = cachedGraph->inputTensor_;
  MPSGraphTensor* indicesTensor = cachedGraph->indicesTensor_;

  MPSShape* outputShape = getMPSShape(output_size);

  MPSGraphTensor* outputTensor = [graph constantWithScalar:0.0f shape:outputShape dataType:inputTensor.dataType];

  outputTensor = [graph scatterWithDataTensor:[graph reshapeTensor:outputTensor withShape:@[ @-1 ] name:nil]
                                updatesTensor:[graph reshapeTensor:inputTensor withShape:@[ @-1 ] name:nil]
                                indicesTensor:[graph reshapeTensor:indicesTensor withShape:@[ @-1 ] name:nil]
                                         axis:0
                                         mode:MPSGraphScatterModeSet
                                         name:nil];

  return [graph reshapeTensor:outputTensor withShape:outputShape name:nil];
}

CachedGraph* getGraph(const Tensor& input, const Tensor& indices, const IntArrayRef output_size) {
  @autoreleasepool {
    string key = getCacheKey(input, indices, output_size);
    return LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      newCachedGraph->inputTensor_ = mpsGraphRankedPlaceHolder(mpsGraph, input);
      newCachedGraph->indicesTensor_ = mpsGraphRankedPlaceHolder(mpsGraph, indices);
      newCachedGraph->outputTensor_ = buildGraph(newCachedGraph, output_size);
    });
  }
}

void runGraph(CachedGraph* cachedGraph, const Tensor& input, const Tensor& indices, const Tensor& output) {
  Placeholder inputPlaceholder = Placeholder(cachedGraph->inputTensor_, input);
  Placeholder indicesPlaceholder = Placeholder(cachedGraph->indicesTensor_, indices);

  NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds = @{
    inputPlaceholder.getMPSGraphTensor() : inputPlaceholder.getMPSGraphTensorData(),
    indicesPlaceholder.getMPSGraphTensor() : indicesPlaceholder.getMPSGraphTensorData(),
  };

  Placeholder outputPlaceholder = Placeholder(cachedGraph->outputTensor_, output);
  NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results = @{
    outputPlaceholder.getMPSGraphTensor() : outputPlaceholder.getMPSGraphTensorData(),
  };

  MPSStream* stream = getCurrentMPSStream();
  runMPSGraph(stream, cachedGraph->graph(), feeds, results);
}
} // namespace
} // namespace mps

Tensor& max_unpooling2d_forward_out_mps(const Tensor& input,
                                        const Tensor& indices,
                                        IntArrayRef output_size,
                                        Tensor& output) {
  TORCH_CHECK(indices.scalar_type() == at::ScalarType::Long,
              "elements in indices should be type int64 but got: ",
              indices.scalar_type());
  TORCH_CHECK(output_size.size() == 2,
              "There should be exactly two elements (height, width) in output_size, but got ",
              output_size.size(),
              " elements.");
  TORCH_CHECK((input.ndimension() == 3 || input.ndimension() == 4),
              "Input to max_unpooling2d should be a 3d or 4d Tensor, but got a tensor with ",
              input.ndimension(),
              " dimensions.");
  TORCH_CHECK(input.sizes() == indices.sizes(),
              "Expected shape of indices to be same as that of the input tensor (",
              input.sizes(),
              ") but got indices tensor with shape: ",
              indices.sizes());

  for (const auto i : c10::irange(1, input.ndimension())) {
    TORCH_CHECK(input.size(i) > 0,
                "max_unpooling2d_forward_out_mps(): ",
                "Expected input to have non-zero size for non-batch dimensions, but got ",
                input.sizes(),
                " with dimension ",
                i,
                " being empty.");
  }

  if (!is_macos_13_or_newer()) {
    TORCH_WARN_ONCE("MPS: max_unpooling2d_forward op is supported natively starting from macOS 13.0. ",
                    "Falling back on CPU. This may have performance implications.");

    Tensor output_cpu = at::max_unpool2d(input.to("cpu"), indices.to("cpu"), output_size);
    output.resize_as_(output_cpu);
    output.copy_(output_cpu);

    return output;
  }

  int64_t oheight = output_size[0];
  int64_t owidth = output_size[1];

  if (input.ndimension() == 3) {
    int64_t numChannels = input.size(0);
    output.resize_({numChannels, oheight, owidth});
  } else {
    int64_t numBatch = input.size(0);
    int64_t numChannels = input.size(1);
    output.resize_({numBatch, numChannels, oheight, owidth});
  }

  mps::CachedGraph* graph = mps::getGraph(input, indices, output.sizes());
  mps::runGraph(graph, input, indices, output);

  return output;
}

Tensor max_unpooling2d_forward_mps(const Tensor& input, const Tensor& indices, IntArrayRef output_size) {
  auto output = at::empty({0}, TensorOptions(kMPS));
  at::native::max_unpooling2d_forward_out_mps(input, indices, output_size, output);
  return output;
}

} // namespace at::native