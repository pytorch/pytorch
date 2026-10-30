//  Copyright © 2022 Apple Inc.
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/TensorIndexing.h>
#include <ATen/mps/MPSProfiler.h>
#include <ATen/native/Resize.h>
#include <ATen/native/mps/OperationUtils.h>
#include <fmt/format.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_unique.h>
#include <ATen/ops/_unique2.h>
#include <ATen/ops/_unique2_native.h>
#include <ATen/ops/_unique_native.h>
#include <ATen/ops/arange.h>
#include <ATen/ops/argsort.h>
#include <ATen/ops/cat.h>
#include <ATen/ops/cumsum.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/full.h>
#include <ATen/ops/masked_select.h>
#include <ATen/ops/ones.h>
#include <ATen/ops/slice.h>
#include <ATen/ops/sort.h>
#include <ATen/ops/unique_consecutive.h>
#include <ATen/ops/unique_consecutive_native.h>
#include <ATen/ops/unique_dim_consecutive.h>
#include <ATen/ops/unique_dim_consecutive_native.h>
#include <ATen/ops/unique_dim_native.h>
#endif

namespace at::native {

#ifndef PYTORCH_JIT_COMPILE_SHADERS
static auto& lib = mps::MetalShaderLibrary::getBundledLibrary();
#else
#include <ATen/native/mps/Unique_metallib.h>
#endif

namespace mps {

struct UniqueCachedGraph : public MPSCachedGraph {
  UniqueCachedGraph(MPSGraph* graph) : MPSCachedGraph(graph) {}
  MPSGraphTensor* inputTensor_ = nil;
  MPSGraphTensor* outputTensor_ = nil;
  MPSGraphTensor* inverseIndicesTensor_ = nil;
  MPSGraphTensor* countsTensor_ = nil;
  MPSGraphTensor* lengthTensor_ = nil;
};

static std::string getUniqueKey(const ScalarType& dtype,
                                const IntArrayRef& base_shape,
                                const bool return_inverse,
                                const bool return_counts,
                                const bool consecutive,
                                std::optional<int64_t> dimOpt) {
  return "_unique2_mps:" + getMPSTypeString(dtype) + "[" + getArrayRefString(base_shape) + "]:[" +
      (dimOpt.has_value() ? std::to_string(dimOpt.value()) : "None") + "]:[" + std::to_string(return_inverse) + "]:[" +
      std::to_string(return_counts) + "]:[" + std::to_string(consecutive) + "]";
}

// dim arg not supported when non consecutive, ie sorted
static std::array<MPSGraphTensor*, 4> buildUniqueGraph(const Tensor& self,
                                                       UniqueCachedGraph* uniqueGraph,
                                                       const bool return_inverse,
                                                       const bool return_counts,
                                                       const bool consecutive,
                                                       std::optional<int64_t> dimOpt) {
  int64_t dim = dimOpt.has_value() ? maybe_wrap_dim(dimOpt.value(), self.dim()) : 0;

  MPSGraph* graph = uniqueGraph->graph();
  MPSGraphTensor* inputTensor = uniqueGraph->inputTensor_;
  MPSShape* shape = [inputTensor shape];
  MPSShape* destShape = shape;
  uint64_t length = [shape[dim] unsignedIntValue];
  MPSDataType dataType = [inputTensor dataType];

  MPSGraphTensor* resultTensor = nil;
  MPSGraphTensor* inverseIndicesTensor = nil;
  MPSGraphTensor* countTensor = nil;
  MPSGraphTensor* lengthTensor = nil;

  const bool needsFlatten = !(dimOpt.has_value() || [shape count] == 1);
  if (needsFlatten) {
    inputTensor = [graph reshapeTensor:inputTensor withShape:@[ @-1 ] name:nil];
    length = 1;
    for (const auto i : c10::irange([shape count])) {
      if (c10::mul_overflows(length, [shape[i] unsignedIntValue], &length)) {
        TORCH_CHECK(false, "RuntimeError: Tensor size overflow");
      }
    }

    destShape = @[ [NSNumber numberWithUnsignedInteger:length] ];
  }

  if (length <= 1) {
    // Trivial case, only 1 element everything is unique
    resultTensor = inputTensor;
    lengthTensor = [graph constantWithScalar:0.0f dataType:MPSDataTypeInt32];
    if (return_inverse) {
      inverseIndicesTensor = [graph constantWithScalar:0.0f dataType:MPSDataTypeInt32];
    }
    if (return_counts) {
      countTensor = [graph constantWithScalar:1.0f dataType:MPSDataTypeInt32];
    }
    return {resultTensor, inverseIndicesTensor, countTensor, lengthTensor};
  }

  // #issue 104398441 sortWithTensor only supports following types, cast if necessary
  if (dataType != MPSDataTypeInt32 && dataType != MPSDataTypeFloat32 && dataType != MPSDataTypeFloat16) {
    dataType = (dataType & MPSDataTypeFloatBit) ? MPSDataTypeFloat32 : MPSDataTypeInt32;
    inputTensor = [graph castTensor:inputTensor toType:dataType name:@"castInputTensor"];
  }

  MPSGraphTensor* sortedInput = nil;
  if (consecutive) {
    sortedInput = inputTensor;
  } else {
    sortedInput = [graph sortWithTensor:inputTensor axis:0 name:nil];
  }

  MPSGraphTensor* frontNMinusOne = [graph sliceTensor:sortedInput dimension:dim start:0 length:length - 1 name:nil];
  MPSGraphTensor* backNMinusOne = [graph sliceTensor:sortedInput dimension:dim start:1 length:length - 1 name:nil];
  MPSGraphTensor* notEqualToPreviousElement = [graph notEqualWithPrimaryTensor:backNMinusOne
                                                               secondaryTensor:frontNMinusOne
                                                                          name:nil];
  MPSGraphTensor* mask = [graph castTensor:notEqualToPreviousElement toType:MPSDataTypeInt32 name:@"castMaskTensor"];

  // If comparing tensors, not scalars, check if entire tensor matches previous element using reductionOr over tensor
  if (dimOpt.has_value() && [shape count] != 1) {
    NSMutableArray* axes = [[NSMutableArray alloc] initWithCapacity:[shape count] - 1];
    for (const auto axis : c10::irange([shape count])) {
      if (static_cast<decltype(dim)>(axis) != dim) {
        [axes addObject:[NSNumber numberWithUnsignedInteger:axis]];
      }
    }
    mask = [graph reductionOrWithTensor:mask axes:axes name:nil];
    mask = [graph squeezeTensor:mask axes:axes name:nil];
    [axes release];
  }

  MPSGraphTensor* scannedIndices = [graph cumulativeSumWithTensor:mask axis:0 name:nil];
  lengthTensor = [graph sliceTensor:scannedIndices dimension:0 start:length - 2 length:1 name:nil];

  MPSGraphTensor* minusOneTensor = [graph constantWithScalar:-1.0f dataType:MPSDataTypeInt32];
  MPSGraphTensor* maskedIndices = [graph selectWithPredicateTensor:mask
                                               truePredicateTensor:scannedIndices
                                              falsePredicateTensor:minusOneTensor
                                                              name:nil];

  MPSGraphTensor* zeroTensor = [graph constantWithScalar:0.0f shape:@[ @1 ] dataType:MPSDataTypeInt32];
  MPSGraphTensor* maskedIndicesWithHead = [graph concatTensors:@[ zeroTensor, maskedIndices ] dimension:0 name:nil];
  MPSGraphTensor* scannedIndicesWithHead = [graph concatTensors:@[ zeroTensor, scannedIndices ] dimension:0 name:nil];

  resultTensor = [graph scatterWithUpdatesTensor:sortedInput
                                   indicesTensor:maskedIndicesWithHead
                                           shape:destShape
                                            axis:dim
                                            mode:MPSGraphScatterModeSet
                                            name:nil];
  // Cast back if necessary
  if ([uniqueGraph->inputTensor_ dataType] != dataType) {
    resultTensor = [graph castTensor:resultTensor toType:[uniqueGraph->inputTensor_ dataType] name:@"castResultTensor"];
  }

  // Compute optional returned tensors if requested
  if (return_inverse) {
    MPSGraphTensor* argSortedInput = nil;
    if (consecutive)
      argSortedInput = [graph coordinateAlongAxis:0
                                        withShape:@[ [NSNumber numberWithUnsignedInteger:length] ]
                                             name:nil];
    else
      argSortedInput = [graph argSortWithTensor:inputTensor axis:0 name:nil];
    inverseIndicesTensor = [graph scatterWithUpdatesTensor:scannedIndicesWithHead
                                             indicesTensor:argSortedInput
                                                     shape:@[ [NSNumber numberWithUnsignedInteger:length] ]
                                                      axis:0
                                                      mode:MPSGraphScatterModeAdd
                                                      name:nil];
    if (needsFlatten)
      inverseIndicesTensor = [graph reshapeTensor:inverseIndicesTensor withShape:shape name:nil];
  }

  if (return_counts) {
    MPSGraphTensor* unitTensor = [graph constantWithScalar:1.0f
                                                     shape:@[ [NSNumber numberWithUnsignedInteger:length] ]
                                                  dataType:MPSDataTypeInt32];
    countTensor = [graph scatterWithUpdatesTensor:unitTensor
                                    indicesTensor:scannedIndicesWithHead
                                            shape:@[ [NSNumber numberWithUnsignedInteger:length] ]
                                             axis:0
                                             mode:MPSGraphScatterModeAdd
                                             name:nil];
  }

  return {resultTensor, inverseIndicesTensor, countTensor, lengthTensor};
}

static UniqueCachedGraph* getUniqueGraph(const Tensor& self,
                                         const bool return_inverse,
                                         const bool return_counts,
                                         const bool consecutive,
                                         std::optional<int64_t> dim) {
  @autoreleasepool {
    std::string key = getUniqueKey(self.scalar_type(), self.sizes(), return_inverse, return_counts, consecutive, dim);
    return LookUpOrCreateCachedGraph<UniqueCachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      newCachedGraph->inputTensor_ = mpsGraphRankedPlaceHolder(mpsGraph, getMPSScalarType(self), getMPSShape(self));
      auto outputTensors = buildUniqueGraph(self, newCachedGraph, return_inverse, return_counts, consecutive, dim);

      newCachedGraph->outputTensor_ = outputTensors[0];
      newCachedGraph->inverseIndicesTensor_ = outputTensors[1];
      newCachedGraph->countsTensor_ = outputTensors[2];
      newCachedGraph->lengthTensor_ = outputTensors[3];
    });
  }
}

static void runUniqueGraph(UniqueCachedGraph* uniqueGraph,
                           const Tensor& input,
                           Tensor& output,
                           Tensor& inverse_indices,
                           Tensor& counts,
                           Tensor& length,
                           bool return_inverse,
                           bool return_counts) {
  Placeholder inputPlaceholder = Placeholder(uniqueGraph->inputTensor_, input);
  auto feeds = dictionaryFromPlaceholders(inputPlaceholder);

  NSMutableDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results = [NSMutableDictionary dictionary];
  Placeholder outputPlaceholder = Placeholder(uniqueGraph->outputTensor_, output);
  Placeholder lengthPlaceholder = Placeholder(uniqueGraph->lengthTensor_, length);
  [results setObject:outputPlaceholder.getMPSGraphTensorData() forKey:outputPlaceholder.getMPSGraphTensor()];
  [results setObject:lengthPlaceholder.getMPSGraphTensorData() forKey:lengthPlaceholder.getMPSGraphTensor()];
  if (return_inverse) {
    Placeholder inverseIndicesPlaceholder = Placeholder(uniqueGraph->inverseIndicesTensor_, inverse_indices);
    [results setObject:inverseIndicesPlaceholder.getMPSGraphTensorData()
                forKey:inverseIndicesPlaceholder.getMPSGraphTensor()];
  }
  if (return_counts) {
    Placeholder countsPlaceholder = Placeholder(uniqueGraph->countsTensor_, counts);
    [results setObject:countsPlaceholder.getMPSGraphTensorData() forKey:countsPlaceholder.getMPSGraphTensor()];
  }

  // Run the graph
  MPSStream* stream = getCurrentMPSStream();
  runMPSGraph(stream, uniqueGraph->graph(), feeds, results);
}

} // namespace mps

static Tensor lexsort_rows_perm_mps(const Tensor& mat_2d) {
  const auto rows = mat_2d.size(0), cols = mat_2d.size(1);
  if (rows <= 1 || cols == 0) {
    return arange(rows, mat_2d.options().dtype(kLong));
  }

  auto perm = arange(rows, mat_2d.options().dtype(kLong));
  for (auto c = cols - 1; c >= 0; --c) {
    // index_select misreads strided inputs spanning more than 2^32 elements.
    auto keys = mat_2d.select(1, c).index({perm});
    const auto idx = argsort(keys, /*stable=*/true, /*dim=*/0, /*descending=*/false);
    perm = perm.index_select(0, idx);
  }
  return perm;
}

// Unique rows of the contiguous (rows, cols) `self`; the flat ops pass a single column.
static std::tuple<Tensor, Tensor, Tensor> _unique_rows_mps(const Tensor& self,
                                                           const bool return_inverse,
                                                           const bool return_counts,
                                                           const bool consecutive) {
  using namespace mps;

  const int64_t rows = self.size(0);
  const int64_t cols = self.size(1);

  // Step 1: sort (unless consecutive).
  TORCH_CHECK_NOT_IMPLEMENTED(consecutive || rows <= std::numeric_limits<int32_t>::max(),
                              "unique: sorting more than 2^31 - 1 elements is not supported on MPS");
  Tensor sorted_values = self;
  Tensor sort_idx;
  if (!consecutive && cols == 1) {
    std::tie(sorted_values, sort_idx) = at::sort(self.view(-1));
  } else if (!consecutive) {
    sort_idx = lexsort_rows_perm_mps(self);
    sorted_values = self.index_select(0, sort_idx);
  }

  // Step 2: boundary mask (int32).
  constexpr int64_t max_threads = std::numeric_limits<uint32_t>::max();
  const bool flat = cols == 1 && rows <= max_threads;
  const int64_t row_threads = std::clamp<int64_t>(cols / 16, 1, 32768);
  Tensor mask = at::empty({rows}, self.options().dtype(kInt));
  if (row_threads > 1) {
    mask.zero_();
  }

  const std::array<int64_t, 2> shape = {rows, cols};
  const std::string type_name = scalarToMetalTypeString(self);
  const std::string mark_kernel = (flat ? "unique_mark_boundaries_" : "unique_mark_rows_") + type_name;
  MPSStream* stream = getCurrentMPSStream();

  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();
      id<MTLComputePipelineState> pso = lib.getPipelineStateForFunc(mark_kernel);
      getMPSProfiler().beginProfileKernel(pso, "unique_mark_boundaries", false, stream);
      [encoder setComputePipelineState:pso];
      mtl_setArgs(encoder, sorted_values, mask);
      if (flat) {
        mtl_dispatch1DJob(encoder, pso, rows);
      } else {
        mtl_setArgs<2>(encoder, shape);
        mtl_dispatch2DJob(encoder, pso, row_threads, std::min(rows, max_threads));
      }
      getMPSProfiler().endProfileKernel(pso, stream);
    }
  });

  // Step 3: inclusive scan of mask.
  const bool scan32 = rows <= std::numeric_limits<int32_t>::max();
  Tensor scan = scan32 ? at::cumsum(mask, /*dim=*/0, kInt) : at::cumsum(mask, /*dim=*/0);

  // Step 4: read num_unique.
  const int64_t num_unique = scan.select(0, rows - 1).item<int64_t>();
  TORCH_INTERNAL_ASSERT(num_unique >= 1);

  // Step 5: allocate outputs.
  Tensor unique_values = at::empty({num_unique, cols}, self.options());
  auto bound_pos = at::empty({return_counts ? num_unique : 0L}, self.options().dtype(kLong));
  auto counts = at::empty({return_counts ? num_unique : 0L}, self.options().dtype(kLong));
  auto inverse = at::empty({return_inverse ? rows : 0L}, self.options().dtype(kLong));
  const auto maybe = [](bool used, const Tensor& t) { return used ? std::optional<Tensor>(t) : std::nullopt; };
  const auto bound_pos_arg = maybe(return_counts, bound_pos);
  const auto sort_idx_arg = maybe(return_inverse && !consecutive, sort_idx);
  const auto inverse_arg = maybe(return_inverse, inverse);
  const std::string emit_kernel = (flat ? "unique_emit_" : "unique_emit_rows_") + type_name + (scan32 ? "_32" : "_64");

  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();
      {
        id<MTLComputePipelineState> pso = lib.getPipelineStateForFunc(emit_kernel);
        getMPSProfiler().beginProfileKernel(pso, "unique_emit", false, stream);
        [encoder setComputePipelineState:pso];
        mtl_setArgs(encoder, sorted_values, mask, scan, unique_values, bound_pos_arg, sort_idx_arg, inverse_arg);
        if (flat) {
          mtl_dispatch1DJob(encoder, pso, rows);
        } else {
          mtl_setArgs<7>(encoder, shape);
          mtl_dispatch2DJob(encoder, pso, row_threads, std::min(rows, max_threads));
        }
        getMPSProfiler().endProfileKernel(pso, stream);
      }

      if (return_counts) {
        id<MTLComputePipelineState> pso = lib.getPipelineStateForFunc("unique_counts");
        getMPSProfiler().beginProfileKernel(pso, "unique_counts", false, stream);
        [encoder setComputePipelineState:pso];
        mtl_setArgs(encoder, bound_pos, counts, num_unique, rows);
        mtl_dispatch1DJob(encoder, pso, std::min(num_unique, max_threads));
        getMPSProfiler().endProfileKernel(pso, stream);
      }
    }
  });

  return std::make_tuple(std::move(unique_values), std::move(inverse), std::move(counts));
}

static std::tuple<Tensor, Tensor, Tensor> _unique_impl_mps(const Tensor& self,
                                                           const bool return_inverse,
                                                           const bool return_counts,
                                                           const bool consecutive,
                                                           std::optional<int64_t> dimOpt) {
  TORCH_CHECK(isIntegralType(self.scalar_type(), /*includeBool=*/true) || mps::supportedFloatingType(self),
              "unique is not supported on MPS for dtype ",
              self.scalar_type());

  if (!dimOpt.has_value()) {
    if (self.numel() == 0) {
      Tensor output = at::empty({0}, self.options());
      Tensor inv = at::empty(return_inverse ? self.sizes() : IntArrayRef{0}, self.options().dtype(kLong));
      Tensor cnt = at::empty({0}, self.options().dtype(kLong));
      return std::make_tuple(std::move(output), std::move(inv), std::move(cnt));
    }
    const auto flat = self.contiguous().view({-1, 1});
    auto [values, inverse, counts] = _unique_rows_mps(flat, return_inverse, return_counts, consecutive);
    return {values.view(-1), return_inverse ? inverse.view(self.sizes()) : inverse, counts};
  }

  if (!consecutive) {
    const int64_t dim = dimOpt.value();
    auto sizes = self.sizes().vec();
    auto zero_dims = std::count(sizes.begin(), sizes.end(), (int64_t)0);
    if (self.size(dim) == 0) {
      TORCH_CHECK(zero_dims == 1, "Number of zero sized dimensions is more than one, so unique cannot be applied ");
      auto output = at::empty(sizes, self.options());
      auto inverse_indices = at::empty({0}, self.options().dtype(kLong));
      auto counts = at::empty({0}, self.options().dtype(kLong));
      return {output, inverse_indices, counts};
    }
    TORCH_CHECK(zero_dims == 0, "There are 0 sized dimensions, and they aren't selected, so unique cannot be applied");

    auto transposed = self.moveaxis(dim, 0);
    auto orig_sizes = transposed.sizes().vec();
    auto input_flat = transposed.contiguous().view({orig_sizes[0], -1});
    auto [values, inverse, counts] = _unique_rows_mps(input_flat, return_inverse, return_counts, consecutive);
    orig_sizes[0] = values.size(0);
    return {values.view(orig_sizes).moveaxis(0, dim), inverse, counts};
  }

  const Tensor& input = self.contiguous();
  // Existing MPSGraph path retained for the unique-along-dim consecutive case.
  // get flat output size
  int64_t totalElems = c10::multiply_integers(input.sizes());

  IntArrayRef outputShape = IntArrayRef(totalElems);
  IntArrayRef inverseIndicesShape = input.sizes();
  IntArrayRef countsShape = IntArrayRef(totalElems);
  int64_t dim = maybe_wrap_dim(dimOpt.value(), self.dim());

  outputShape = input.sizes();
  inverseIndicesShape = IntArrayRef(input.sizes()[dim]);
  countsShape = IntArrayRef(input.sizes()[dim]);
  if (!return_inverse)
    inverseIndicesShape = {};
  if (!return_counts)
    countsShape = {};

  Tensor output = at::empty(outputShape, input.scalar_type(), std::nullopt, kMPS, std::nullopt, std::nullopt);
  Tensor inverse_indices =
      at::empty(inverseIndicesShape, ScalarType::Long, std::nullopt, kMPS, std::nullopt, std::nullopt);
  Tensor counts = at::empty(countsShape, ScalarType::Long, std::nullopt, kMPS, std::nullopt, std::nullopt);
  Tensor length = at::empty({1}, ScalarType::Int, std::nullopt, kMPS, std::nullopt, std::nullopt);

  if (input.numel() == 0) {
    return std::make_tuple(std::move(output), std::move(inverse_indices), std::move(counts));
  }

  @autoreleasepool {
    mps::UniqueCachedGraph* uniqueGraph =
        mps::getUniqueGraph(input, return_inverse, return_counts, consecutive, dimOpt);
    mps::runUniqueGraph(uniqueGraph, input, output, inverse_indices, counts, length, return_inverse, return_counts);
  }

  int64_t lengthScalar = length.item<int64_t>() + 1; // length actually holds max index, add 1
  if (!output.sizes().empty()) {
    output = at::slice(output, dim, 0, lengthScalar);
  }
  if (return_counts)
    counts = at::slice(counts, 0, 0, lengthScalar);

  return std::make_tuple(std::move(output), std::move(inverse_indices), std::move(counts));
}

std::tuple<Tensor, Tensor, Tensor> unique_consecutive_mps(const Tensor& self,
                                                          const bool return_inverse,
                                                          const bool return_counts,
                                                          std::optional<int64_t> dim) {
  return _unique_impl_mps(self, return_inverse, return_counts, true, dim);
}

std::tuple<Tensor, Tensor, Tensor> unique_dim_consecutive_mps(const Tensor& self,
                                                              int64_t dim,
                                                              const bool return_inverse,
                                                              const bool return_counts) {
  return _unique_impl_mps(self, return_inverse, return_counts, true, dim);
}

std::tuple<Tensor, Tensor, Tensor> _unique2_mps(const Tensor& self,
                                                const bool sorted,
                                                const bool return_inverse,
                                                const bool return_counts) {
  return _unique_impl_mps(self, return_inverse, return_counts, false, std::nullopt);
}

std::tuple<Tensor, Tensor> _unique_mps(const Tensor& self, const bool sorted, const bool return_inverse) {
  auto [output, inverse_indices, _] = _unique_impl_mps(self, return_inverse, false, false, std::nullopt);
  return std::make_tuple(std::move(output), std::move(inverse_indices));
}

std::tuple<Tensor, Tensor, Tensor> unique_dim_mps(const Tensor& self,
                                                  int64_t dim,
                                                  const bool /*sorted*/,
                                                  const bool return_inverse,
                                                  const bool return_counts) {
  return _unique_impl_mps(self, return_inverse, return_counts, false, dim);
}

} // namespace at::native
