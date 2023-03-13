//  Copyright © 2022 Apple Inc.

#include <ATen/native/mps/OperationUtils.h>
#include <ATen/native/Histogram.h>
#include <ATen/native/Resize.h>

namespace at::native {
namespace mps {

enum BIN_SELECTION_ALGORITHM {
    LINEAR_INTERPOLATION,
    LINEAR_INTERPOLATION_WITH_LOCAL_SEARCH,
    BINARY_SEARCH,
};

static const char* METAL_SUMMARY = R"SUMMARY_METAL(

#include <metal_stdlib>
using namespace metal;

enum BIN_SELECTION_ALGORITHM {
  LINEAR_INTERPOLATION,
  LINEAR_INTERPOLATION_WITH_LOCAL_SEARCH,
  BINARY_SEARCH,
};

// Re-implementation of std::upper_bound with some modifications.
template<typename T>
int64_t upper_bound(constant T * arr, int64_t first, int64_t len, T val) {
  int64_t middle;
  int64_t half_;

  while (len > 0) {
    half_ = len >> 1;
    middle = first + half_;

    if (val < arr[middle]) {
      len = half_;
    } else {
      first = ++middle;
      len = len - half_ - 1;
    }
  }
  return first;
}

// The implementation here is mostly taken from the CPU's implementation with some modifications.
// Please see `aten/src/ATen/native/cpu/HistogramKernel.cpp` for more details.
template<typename T>
kernel void histogramdd(constant void     * input_      [[buffer(0)]],
                  constant T        * weight            [[buffer(1)]],
                  device   T        * local_out         [[buffer(2)]],
                  constant uint     * offsets           [[buffer(3)]],
                  constant size_t   & num_dims          [[buffer(4)]],
                  constant T        * bin_seq           [[buffer(5)]],
                  constant int64_t  * num_bin_edges     [[buffer(6)]],
                  constant T        * leftmost_edge     [[buffer(7)]],
                  constant T        * rightmost_edge    [[buffer(8)]],
                  constant int64_t  * local_out_strides [[buffer(9)]],
                  constant uint8_t  & algorithm         [[buffer(10)]],
                  constant uint8_t  & has_weight        [[buffer(11)]],
                  uint tid [[thread_position_in_grid]]) {
  
  constexpr T eps = 8e-7;
  bool skip_element = false;
  int64_t hist_index = 0;
  int64_t bin_seq_offset = 0;

  for (size_t dim = 0; dim < num_dims; dim++) {
    T element = ((constant T*)input_)[tid * num_dims + dim];

    // Skips elements which fall outside the specified bins and NaN elements
    // Adding an eps to the edges to eliminate precision issues that cause elements accidentally skipped,
    // this is likely due to the minuscule implementation differences between the CPU and MPS's linspace.
    if (!(element >= (leftmost_edge[dim] - eps) && element <= (rightmost_edge[dim] + eps) )) {
        skip_element = true;
        break;
    }
    int64_t pos = -1;
    
    if (algorithm == BIN_SELECTION_ALGORITHM::BINARY_SEARCH) {
      pos = upper_bound(
        bin_seq,
        bin_seq_offset,
        num_bin_edges[dim],
        element
      ) - bin_seq_offset - 1;
    } else if (
      algorithm == BIN_SELECTION_ALGORITHM::LINEAR_INTERPOLATION ||
      algorithm == BIN_SELECTION_ALGORITHM::LINEAR_INTERPOLATION_WITH_LOCAL_SEARCH) {
      pos = static_cast<int64_t>((element - leftmost_edge[dim])
                            * (num_bin_edges[dim] - 1)
                            / (rightmost_edge[dim] - leftmost_edge[dim]));
      if (algorithm == LINEAR_INTERPOLATION_WITH_LOCAL_SEARCH) {
          int64_t pos_min = max(static_cast<int64_t>(0), pos - 1);
          int64_t pos_max = min(pos + 2, num_bin_edges[dim]);
          pos = upper_bound(
            bin_seq,
            bin_seq_offset + pos_min,
            pos_max - pos_min,
            element
          ) - bin_seq_offset - 1;
      }
    }

    if (pos == (num_bin_edges[dim] - 1)) {
      pos -= 1;
    }
    hist_index += local_out_strides[dim + 1] * pos;
    bin_seq_offset += num_bin_edges[dim];
  }
  if (!skip_element) {
    // In the unweighted case, the default weight is 1
    local_out[local_out_strides[0] * tid + hist_index] += has_weight ? weight[tid] : 1;
  }
}


#define REGISTER_HISTOGRAMDD_OP(DTYPE)                        \
template                                                      \
[[host_name("histogramdd_" #DTYPE)]]                          \
kernel void histogramdd<DTYPE>(                               \
  constant void     * input_                  [[buffer(0)]],  \
  constant DTYPE    * weight                  [[buffer(1)]],  \
  device   DTYPE    * local_out               [[buffer(2)]],  \
  constant uint     * offsets                 [[buffer(3)]],  \
  constant size_t   & num_dims                [[buffer(4)]],  \
  constant DTYPE    * bin_seq                 [[buffer(5)]],  \
  constant int64_t  * num_bin_edges           [[buffer(6)]],  \
  constant DTYPE    * leftmost_edge           [[buffer(7)]],  \
  constant DTYPE    * rightmost_edge          [[buffer(8)]],  \
  constant int64_t  * local_out_strides       [[buffer(9)]],  \
  constant uint8_t  & bin_selection_algorithm [[buffer(10)]], \
  constant uint8_t  & has_weight              [[buffer(11)]], \
  uint tid [[thread_position_in_grid]]);

REGISTER_HISTOGRAMDD_OP(float);
//REGISTER_HISTOGRAMDD_OP(half);
//REGISTER_HISTOGRAMDD_OP(int);
//REGISTER_HISTOGRAMDD_OP(long);
//REGISTER_HISTOGRAMDD_OP(short);
//REGISTER_HISTOGRAMDD_OP(char);
//REGISTER_HISTOGRAMDD_OP(uchar);
//REGISTER_HISTOGRAMDD_OP(bool);

)SUMMARY_METAL";

static id<MTLLibrary> compileSummaryOpLibrary(id<MTLDevice> device) {
  static id<MTLLibrary> summaryLibrary = nil;
  if (summaryLibrary) {
    return summaryLibrary;
  }

  NSError *error = nil;
  MTLCompileOptions *options = [[MTLCompileOptions new] autorelease];
  [options setLanguageVersion: MTLLanguageVersion2_3];
  summaryLibrary  = [device newLibraryWithSource:[NSString stringWithCString: METAL_SUMMARY encoding:NSASCIIStringEncoding]
                                       options:options
                                         error:&error];
  TORCH_CHECK(summaryLibrary, "Failed to create metal summary library, error: ", [[error description] UTF8String]);
  return summaryLibrary;
}

static id<MTLComputePipelineState> summaryPipelineState(id<MTLDevice> device, const std::string& kernel) {
  static std::unordered_map<std::string, id<MTLComputePipelineState>> psoCache;
  id<MTLComputePipelineState> pso = psoCache[kernel];
  if (pso) {
    return pso;
  }

  NSError* error = nil;
  id<MTLLibrary> crossLib = compileSummaryOpLibrary(device);
  id<MTLFunction> crossFunc = [crossLib newFunctionWithName:[NSString stringWithUTF8String:kernel.c_str()]];
  TORCH_CHECK(crossFunc, "Failed to create function state object for: ", kernel);
  pso = [device newComputePipelineStateWithFunction:crossFunc error:&error];
  TORCH_CHECK(pso, "Failed to created pipeline state object, error: ", [[error description] UTF8String]);

  psoCache[kernel] = pso;
  return pso;
}

template <typename input_t, BIN_SELECTION_ALGORITHM algorithm>
void histogramdd_kernel_impl(
    Tensor& hist_output,
    const TensorList& bin_edges,
    const Tensor& input,
    const c10::optional<Tensor>& weight) {
  TORCH_CHECK(input.dtype() != at::kDouble, "float64 is not supported on MPS");
  TORCH_INTERNAL_ASSERT(input.dim() == 2);

  constexpr uint8_t bin_selection_algorithm = algorithm;
  const int64_t N = input.size(0);
  const bool has_weight = weight.has_value();

  if (has_weight) {
      TORCH_CHECK(weight.value().is_contiguous(), "histogramdd(): weight should be contiguous on MPS");
      TORCH_INTERNAL_ASSERT(weight.value().dim() == 1 && weight.value().numel() == N);
      TORCH_INTERNAL_ASSERT(weight.value().scalar_type() == input.scalar_type());
  }

  const int64_t D = input.size(1);
  int64_t bin_edges_numel = 0;
  TORCH_INTERNAL_ASSERT(int64_t(bin_edges.size()) == D);
  for (const auto dim : c10::irange(D)) {
      bin_edges_numel += bin_edges[dim].numel();
      TORCH_INTERNAL_ASSERT(bin_edges[dim].is_contiguous());
      TORCH_INTERNAL_ASSERT(hist_output.size(dim) + 1 == bin_edges[dim].numel());
  }

  if (D == 0) {
      // hist is an empty tensor in this case; nothing to do here
      return;
  }

  std::vector<input_t> bin_seq(bin_edges_numel);
  std::vector<int64_t> num_bin_edges(D);
  std::vector<input_t> leftmost_edge(D);
  std::vector<input_t> rightmost_edge(D);
  size_t bin_seq_offset = 0;

  for (const auto dim : c10::irange(D)) {
    for (const auto elem_idx : c10::irange(bin_edges[dim].numel())) {
      bin_seq[bin_seq_offset + elem_idx] = (bin_edges[dim][elem_idx].item().to<input_t>());
    }
    num_bin_edges[dim] = bin_edges[dim].numel();
    leftmost_edge[dim] = bin_seq[bin_seq_offset];
    rightmost_edge[dim] = bin_seq[bin_seq_offset + num_bin_edges[dim] - 1];
    bin_seq_offset += num_bin_edges[dim];
  }


  const uint32_t kernelOffsetNumThreads = input.numel();
  const uint32_t numThreads = N;
  const auto hist_sizes = hist_output.sizes();

  DimVector thread_hist_sizes(hist_sizes.size() + 1); // [n_threads, output_sizes...]
  thread_hist_sizes[0] = numThreads;
  std::copy(hist_sizes.begin(), hist_sizes.end(), thread_hist_sizes.begin() + 1);
  Tensor thread_histograms = at::zeros(
    thread_hist_sizes,
    hist_output.scalar_type(),
    c10::nullopt /* layout */,
    kMPS,
    c10::nullopt /* pin_memory */
  );

  id<MTLDevice> device = MPSDevice::getInstance()->device();
  id<MTLBuffer> inputBuffer  = getMTLBufferStorage(input);
  id<MTLBuffer> outputBuffer = getMTLBufferStorage(thread_histograms);
  id<MTLBuffer> weightBuffer = has_weight ? getMTLBufferStorage(weight.value()) : [[device newBufferWithLength: 0
                                                             options: 0] autorelease];
  size_t weightOffset =  has_weight ? weight.value().storage_offset() * weight.value().element_size() : 0;

  MPSStream* mpsStream = getCurrentMPSStream();
  const uint32_t nDim = input.sizes().size();
  
  dispatch_sync(mpsStream->queue(), ^(){
    @autoreleasepool {
      NSError* error = nil;
      id<MTLCommandBuffer> commandBuffer = mpsStream->commandBuffer();
      id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
      MTLSize gridSize = MTLSizeMake(kernelOffsetNumThreads, 1, 1);
      const IntArrayRef& iterShape = input.sizes();
      std::vector<uint32_t> iterShapeData(iterShape.size());
      std::vector<uint32_t> strides(nDim);

      for (const auto i: c10::irange(iterShape.size())) {
        TORCH_CHECK(i <= UINT32_MAX);
        iterShapeData[i] = (uint32_t)(iterShape[i]);
      }

      for (const auto i: c10::irange(nDim)) {
        strides[i] = input.stride(i);
      }

      id<MTLFunction> kernelDataOffsetFunction = MPSDevice::getInstance()->metalIndexingFunction("kernel_index_offset", nil);
      id<MTLComputePipelineState> kernelDataOffsetPSO = [[device newComputePipelineStateWithFunction: kernelDataOffsetFunction
                                                                                                error: &error] autorelease];
      id<MTLBuffer> kernelDataOffset = [[device newBufferWithLength: kernelOffsetNumThreads * sizeof(uint)
                                                             options: 0] autorelease];
      TORCH_CHECK(kernelDataOffsetPSO, "Failed to create pipeline state object, error: ", [[error description] UTF8String]);
      [computeEncoder setComputePipelineState:kernelDataOffsetPSO];
      [computeEncoder setBytes:strides.data() length:sizeof(uint32_t) * nDim  atIndex:0];
      [computeEncoder setBuffer:kernelDataOffset offset:0 atIndex:1];
      [computeEncoder setBytes:iterShapeData.data() length:sizeof(uint32_t) * iterShape.size() atIndex:2];
      [computeEncoder setBytes:&nDim length:sizeof(uint32_t) atIndex:3];

      NSUInteger kernelOffsetTGSize = kernelDataOffsetPSO.maxTotalThreadsPerThreadgroup;
      if (kernelOffsetTGSize > kernelOffsetNumThreads)
          kernelOffsetTGSize = kernelOffsetNumThreads;

      MTLSize kernelOffsetThreadGroupSize = MTLSizeMake(kernelOffsetTGSize, 1, 1);
      [computeEncoder dispatchThreads: gridSize
                threadsPerThreadgroup: kernelOffsetThreadGroupSize];

      const std::string kernel = "histogramdd_" + scalarToMetalTypeString(input.scalar_type());
      id<MTLComputePipelineState> summaryPSO = summaryPipelineState(device, kernel);
      [computeEncoder setComputePipelineState:summaryPSO];
      [computeEncoder setBuffer:inputBuffer  offset:input.storage_offset() * input.element_size() atIndex:0];
      [computeEncoder setBuffer:weightBuffer  offset:weightOffset atIndex:1];
      [computeEncoder setBuffer:outputBuffer offset:thread_histograms.storage_offset() * thread_histograms.element_size() atIndex:2];
      [computeEncoder setBuffer:kernelDataOffset offset:0 atIndex:3];
      [computeEncoder setBytes:&D length:sizeof(int64_t) atIndex:4];
      [computeEncoder setBytes:bin_seq.data() length:sizeof(input_t) * bin_seq_offset  atIndex:5];
      [computeEncoder setBytes:num_bin_edges.data() length:sizeof(int64_t) * D atIndex:6];
      [computeEncoder setBytes:leftmost_edge.data() length:sizeof(input_t) * D atIndex:7];
      [computeEncoder setBytes:rightmost_edge.data() length:sizeof(input_t) * D atIndex:8];
      [computeEncoder setBytes:thread_histograms.strides().data() length:sizeof(int64_t) * thread_hist_sizes.size() atIndex:9];
      [computeEncoder setBytes:&bin_selection_algorithm length:sizeof(uint8_t) atIndex:10];
      [computeEncoder setBytes:&has_weight length:sizeof(uint8_t) atIndex:11];

      NSUInteger tgSize = summaryPSO.maxTotalThreadsPerThreadgroup;
      if (tgSize > numThreads) {
          tgSize = numThreads;
      }
      gridSize = MTLSizeMake(numThreads, 1, 1);
      MTLSize threadGroupSize = MTLSizeMake(tgSize, 1, 1);
      [computeEncoder dispatchThreads: gridSize
                threadsPerThreadgroup: threadGroupSize];

      [computeEncoder endEncoding];
      mpsStream->commit(true);
    }
  });
  at::sum_out(hist_output, thread_histograms, /*dim=*/{0});
}

void bincount_histc_mps_impl(
    Tensor& self,
    const Tensor& weights,
    const Tensor& output) 
  {
  using namespace mps;

  struct CachedGraph : public MPSCachedGraph
  {
    CachedGraph(MPSGraph *graph) : MPSCachedGraph(graph) {}
    MPSGraphTensor* inputTensor_ = nil;
    MPSGraphTensor* weightsTensor_ = nil;
    MPSGraphTensor* scatterDataTensor_ = nil;
    MPSGraphTensor* outputTensor_ = nil;
  };

  MPSStream* stream = getCurrentMPSStream();
  MPSGraphCache* cache_ = MPSGraphCache::getInstance();
  bool has_weights = weights.defined();

  @autoreleasepool {
    string key = "bincount_mps_impl" + getTensorsStringKey({self, weights});
    CachedGraph* cachedGraph = static_cast<CachedGraph *>(cache_->LookUp(key));
    if(!cachedGraph) {
      MPSCachedGraph *tmpCachedGraph = cache_->CreateCachedGraph(key, ^ MPSCachedGraph * () {

        CachedGraph *newCachedGraph = nil;

        @autoreleasepool {
          // Initialize graph
          MPSGraph* mpsGraph = make_mps_graph();
          newCachedGraph = new CachedGraph(mpsGraph);
          MPSGraphTensor* inputTensor = mpsGraphRankedPlaceHolder(mpsGraph, self);
          MPSGraphTensor *scatterDataTensor = mpsGraphUnrankedPlaceHolder(mpsGraph, getMPSScalarType(output.scalar_type()));

          MPSGraphTensor *updatesTensor = nil;
          if (has_weights) {
            updatesTensor = mpsGraphRankedPlaceHolder(mpsGraph, weights);
          }
          else {
            updatesTensor = [mpsGraph constantWithScalar:1.0f
                                                   shape:getMPSShape(self)
                                                dataType:getMPSDataType(output.scalar_type())];
          }

          MPSGraphTensor *castedInputTensor = inputTensor;
          if (self.scalar_type() == kByte) {
            castedInputTensor = [mpsGraph castTensor:inputTensor
                                              toType:MPSDataTypeInt32
                                                name:@"castInputTensor"];
          }

          MPSGraphTensor *outputTensor = [mpsGraph scatterWithDataTensor:scatterDataTensor
                                                           updatesTensor:updatesTensor
                                                           indicesTensor:castedInputTensor
                                                                     axis:0
                                                                     mode:MPSGraphScatterModeAdd
                                                                     name:nil];

          newCachedGraph->inputTensor_ = inputTensor;
          newCachedGraph->outputTensor_ = outputTensor;
          newCachedGraph->scatterDataTensor_ = scatterDataTensor;
          if (has_weights) {
            newCachedGraph->weightsTensor_ = updatesTensor;
          }
        }
        return newCachedGraph;
      });
      cachedGraph = static_cast<CachedGraph *>(tmpCachedGraph);
    }

    // Create placeholders which use the keys of the CachedGraph to create inputs and outputs of the operation
    Placeholder inputPlaceholder = Placeholder(cachedGraph->inputTensor_, self);
    Placeholder outputPlaceholder = Placeholder(cachedGraph->outputTensor_, output);
    Placeholder scatterPlaceholder = Placeholder(cachedGraph->scatterDataTensor_, output);
    Placeholder weightsPlaceholder = Placeholder();

    // Create dictionary of inputs/feeds and outputs/results
    NSMutableDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds =[NSMutableDictionary dictionary];
    feeds[inputPlaceholder.getMPSGraphTensor()] = inputPlaceholder.getMPSGraphTensorData();
    feeds[scatterPlaceholder.getMPSGraphTensor()] = scatterPlaceholder.getMPSGraphTensorData();
    if(has_weights) {
      weightsPlaceholder = Placeholder(cachedGraph->weightsTensor_, weights);
      feeds[weightsPlaceholder.getMPSGraphTensor()] = weightsPlaceholder.getMPSGraphTensorData();
    }

    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results = @{
      outputPlaceholder.getMPSGraphTensor() : outputPlaceholder.getMPSGraphTensorData()
    };

    // Run the graph
    runMPSGraph(stream, cachedGraph->graph(), feeds, results);
  }
}

Tensor _bincount_histc_mps(
    const Tensor& self,
    const c10::optional<Tensor>& weights_opt,
    const c10::optional<int64_t> minlength_opt,
    const c10::optional<int64_t> nbins_opt,
    const c10::optional<Scalar>& min_opt,
    const c10::optional<Scalar>& max_opt,
    const c10::optional<Tensor>& output_opt)
  {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> weights_maybe_owned = at::borrow_from_optional_tensor(weights_opt);
  const Tensor& weights = *weights_maybe_owned;
  Tensor weights_ = weights;
  bool has_weights = weights.defined();

  Tensor self_ = self;
  int64_t nbins;
  int64_t minlength = 0;

  if (nbins_opt && min_opt && max_opt) {  // histc
    nbins = nbins_opt.value();
    
    if (min_opt.value().toSymInt() != 0 || max_opt.value().toSymInt() != 0){
      auto indices = at::masked_select(self, at::where(at::logical_and(self > min_opt.value(), self < max_opt.value()), 1, 0));
      self_ = at::argsort(indices);
    } else {
      self_ = at::argsort(self);
    }
    
  } else {  // bincount
    TORCH_CHECK(c10::isIntegralType(self.scalar_type(), /*includesBool=*/true));
    TORCH_CHECK(self.dim() == 1 && self.min().item<int64_t>() >= 0, "bincount only supports 1-d non-negative integral inputs.");
    TORCH_CHECK(!(has_weights && (weights.dim() != 1 || weights.size(0) != self.size(0))), "weights should be 1-d and have the same length as input");
    
    if (minlength_opt) {
      minlength = minlength_opt.value();
      TORCH_CHECK(minlength >= 0, "minlength should be >= 0");
    }

    if (self.dim() == 1 && self.numel() == 0) {
      return at::zeros(
          {minlength},
          kLong,
          c10::nullopt /* layout */,
          kMPS,
          c10::nullopt /* pin_memory */);
    }
    nbins = std::max(self.max().item<int64_t>() + 1L, minlength);
  }

  if (output_opt) {
    auto output_maybe_owned = at::borrow_from_optional_tensor(output_opt);
    auto output = *output_maybe_owned;
    resize_output(output, IntArrayRef({nbins}));
    bincount_histc_mps_impl(self_, weights_, output);
    return output;
  } else {
    Tensor output;

    if (has_weights) {
      if(weights.scalar_type() != ScalarType::Float &&
        weights.scalar_type() != ScalarType::Int   &&
        weights.scalar_type() != ScalarType::Half) {
          // Scatter doesn't work for int8/int16 dtypes
          weights_ = weights.to(kInt);
      }
      output = at::zeros(
          {nbins},
          optTypeMetaToScalarType(weights_.options().dtype_opt()),
          weights_.options().layout_opt(),
          weights_.options().device_opt(),
          weights_.options().pinned_memory_opt());
    } else {
      output = at::zeros(
          {nbins},
          (nbins_opt && min_opt && max_opt) ? self.scalar_type() : kLong,
          c10::nullopt /* layout */,
          kMPS,
          c10::nullopt /* pin_memory */);
    }
    bincount_histc_mps_impl(self_, weights_, output);
    return output;
  }
}
template<BIN_SELECTION_ALGORITHM bin_algorithm>
static void histogramdd_out_mps_template(
  const Tensor& self,
  const c10::optional<Tensor>& weight,
  bool density,
  Tensor& hist,
  const TensorList& bin_edges)
{
  hist.fill_(0);

  const int64_t N = self.size(-1);
  const int64_t M = std::accumulate(self.sizes().begin(), self.sizes().end() - 1,
          (int64_t)1, std::multiplies<int64_t>());

  const Tensor reshaped_input = self.reshape({M, N});

  const auto reshaped_weight = weight.has_value()
          ? c10::optional<Tensor>(weight.value().reshape({M}))
          : c10::optional<Tensor>();

  std::vector<Tensor> bin_edges_contig(bin_edges.size());
  for (const auto dim : c10::irange(bin_edges_contig.size())) {
      bin_edges_contig[dim] = bin_edges[dim].contiguous();
  }

  AT_DISPATCH_FLOATING_TYPES(self.scalar_type(), "histogram_mps", [&]() {
    mps::histogramdd_kernel_impl<scalar_t, bin_algorithm>(
      hist,
      bin_edges_contig,
      reshaped_input,
      reshaped_weight
    );
  });

  /* Divides each bin's value by the total count/weight in all bins,
    * and by the bin's volume.
    */
  if (density) {
      const auto hist_sum = hist.sum().item();
      hist.div_(hist_sum);

        /* For each dimension, divides each bin's value
        * by the bin's length in that dimension.
        */
      for (const auto dim : c10::irange(N)) {
          const auto bin_lengths = bin_edges[dim].diff();

          // Used to reshape bin_lengths to align with the corresponding dimension of hist.
          std::vector<int64_t> shape(N, 1);
          shape[dim] = bin_lengths.numel();

          hist.div_(bin_lengths.reshape(shape));
      }
  }
}
} // namespace mps

static void histogramdd_kernel(
  const Tensor& self,
  const c10::optional<Tensor>& weight,
  bool density,
  Tensor& hist,
  const TensorList& bin_edges)
{
  mps::histogramdd_out_mps_template<mps::BINARY_SEARCH>(self, weight, density, hist, bin_edges);
}

static void histogramdd_linear_kernel(const Tensor& self, const c10::optional<Tensor>& weight,
        bool density, Tensor& hist, const TensorList& bin_edges, bool local_search) {

  if (local_search) {
    // histogramdd codepath: both hist and bin_edges are eventually returned as output,
    // so we'll keep them consistent
    mps::histogramdd_out_mps_template<mps::LINEAR_INTERPOLATION_WITH_LOCAL_SEARCH>(
      self, weight, density, hist, bin_edges);
  } else {
    // histc codepath: bin_edges are not returned to the caller
    mps::histogramdd_out_mps_template<mps::LINEAR_INTERPOLATION>(
      self, weight, density, hist, bin_edges);
  }
}

Tensor _bincount_mps(const Tensor& self, const c10::optional<Tensor>& weights_opt, int64_t minlength) {
  return mps::_bincount_histc_mps(
    self,
    weights_opt,
    minlength,
    c10::nullopt /* nbins_opt */,
    c10::nullopt /* min_opt */,
    c10::nullopt /* max_opt */,
    c10::nullopt /* output_opt */
  );
}

Tensor _histc_mps(
    const Tensor& self,
    int64_t nbins,
    const Scalar& min,
    const Scalar& max) {
  return mps::_bincount_histc_mps(
    self,
    c10::nullopt /* weights_opt */,
    c10::nullopt /* minlength_opt */,
    c10::optional<int64_t>(nbins),
    c10::optional<Scalar>(min),
    c10::optional<Scalar>(max),
    c10::nullopt /* output_opt */
  );
}

Tensor& _histc_out_mps(const Tensor& self, int64_t bins, const Scalar& min, const Scalar& max, Tensor& result) {
  mps::_bincount_histc_mps(
    self,
    c10::nullopt /* weights_opt */,
    c10::nullopt /* minlength_opt */,
    c10::optional<int64_t>(bins),
    c10::optional<Scalar>(min),
    c10::optional<Scalar>(max),
    c10::optional<Tensor>(result)
  );
  return result;
}

REGISTER_DISPATCH(histogramdd_stub, &histogramdd_kernel);
REGISTER_DISPATCH(histogramdd_linear_stub, &histogramdd_linear_kernel);
} // namespace at::native
