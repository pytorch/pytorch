//  Copyright © 2022 Apple Inc.
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
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
#include <ATen/ops/nonzero.h>
#include <ATen/ops/ones.h>
#include <ATen/ops/ones_like.h>
#include <ATen/ops/slice.h>
#include <ATen/ops/sort.h>
#include <ATen/ops/unique_consecutive.h>
#include <ATen/ops/unique_consecutive_native.h>
#include <ATen/ops/unique_dim_consecutive.h>
#include <ATen/ops/unique_dim_consecutive_native.h>
#include <ATen/ops/unique_dim_native.h>
#include <ATen/ops/zeros.h>
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

// Fast path for `_unique` / `_unique2` / `unique_consecutive` over a flat
// 1D view of the input. Output shapes match the contract of _unique_impl_mps:
// values are 1D, counts (when requested) is 1D, inverse_indices reshapes to
// the original input shape.
//
// Pipeline:
//   1. (Optional) Sort the flat input. Skipped for `consecutive=true`.
//   2. Mark boundaries: mask[i] = (sorted[i] != sorted[i-1]); mask[0]=1.
//   3. Inclusive scan of mask -> scan[i] is the 1-indexed unique-group id at i.
//   4. Sync once to read num_unique = scan[N-1].
//   5. Emit unique values and boundary positions in parallel.
//   6. Counts from boundary-position differences (no atomics).
//   7. Inverse via the sort permutation (no atomics).
//
// IMPORTANT — dispatch-queue re-entrancy: any `at::*` op (arange, sub, cumsum,
// sort, etc.) acquires the MPS stream's serial queue via its own
// dispatch_sync. Calling one from *inside* a block already running on that
// queue deadlocks libdispatch (SIGTRAP / exit 133, no useful stack). Allocate
// and transform tensors *outside* the block; only the raw
// `lib.getPipelineStateForFunc` + `mtl_setArgs` + `mtl_dispatch1DJob` calls
// belong inside it.
static std::tuple<Tensor, Tensor, Tensor> _unique_flat_mps_fast(const Tensor& self_flat,
                                                                const bool return_inverse,
                                                                const bool return_counts,
                                                                const bool consecutive,
                                                                const IntArrayRef inverse_shape) {
  using namespace mps;

  TORCH_INTERNAL_ASSERT(self_flat.dim() == 1);
  const int64_t numel = self_flat.numel();
  const ScalarType in_dtype = self_flat.scalar_type();

  // at::sort on MPS returns out-of-bounds sort indices for kBool inputs
  // (see also the sort-bool path in Sort.mm); int8/uint8 indices are fine,
  // and a byte reinterpret of int8 would invert the order of negative
  // values, so this stays bool-only rather than covering all 1-byte dtypes.
  // The promotion is a free bitwise view (not a cast): bool's 0/1 bit
  // patterns are valid uint8 values, sort order is preserved, and the
  // kernel pipeline below only keys on the working tensor's dtype. The
  // unique-values output is viewed back to bool at the end.
  const bool sort_via_byte = (in_dtype == kBool);
  Tensor work = sort_via_byte ? self_flat.view(kByte) : self_flat;
  const ScalarType work_dtype = work.scalar_type();

  // Step 1: sort (unless consecutive).
  Tensor sorted_values;
  Tensor sort_idx; // only populated when return_inverse && !consecutive
  if (consecutive) {
    sorted_values = work;
  } else {
    auto sort_result = at::sort(work, /*dim=*/0, /*descending=*/false);
    sorted_values = std::get<0>(sort_result);
    if (return_inverse) {
      sort_idx = std::get<1>(sort_result);
    }
  }

  // Step 2: boundary mask (int32).
  Tensor mask = at::empty({numel}, self_flat.options().dtype(kInt));
  // Step 3: inclusive scan of mask. cumsum preserves int32 dtype.
  Tensor scan;

  // Kernel name suffix tracks the working dtype, not the input dtype, so the
  // bool->byte promotion above flows through correctly.
  const std::string type_name = mps::scalarToMetalTypeString(work_dtype);
  MPSStream* stream = getCurrentMPSStream();

  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();
      id<MTLComputePipelineState> pso =
          lib.getPipelineStateForFunc(fmt::format("unique_mark_boundaries_{}", type_name));
      getMPSProfiler().beginProfileKernel(pso, "unique_mark_boundaries", false);
      [encoder setComputePipelineState:pso];
      mtl_setArgs(encoder, sorted_values, mask, static_cast<uint64_t>(numel));
      mtl_dispatch1DJob(encoder, pso, numel);
      getMPSProfiler().endProfileKernel(pso);
    }
  });

  // cumsum runs as a separate dispatch on MPS; that's fine.
  scan = at::cumsum(mask, /*dim=*/0);
  // cumsum of int32 may upcast to int64 on some paths; track the scan dtype.
  const auto scan_dtype = scan.scalar_type();
  TORCH_CHECK(scan_dtype == kInt || scan_dtype == kLong, "unique: unexpected cumsum output dtype ", scan_dtype);
  const auto scan_suffix = std::to_string(c10::elementSize(scan_dtype) * 8);

  // Step 4: read num_unique.
  const int64_t num_unique = scan.select(0, numel - 1).item<int64_t>();
  TORCH_INTERNAL_ASSERT(num_unique >= 1);

  // Step 5/6/7: allocate outputs. unique_values is allocated in the working
  // dtype (kByte when the input was bool) and cast back at the end so the
  // emit kernel writes through a tensor whose stride matches `sorted_values`.
  Tensor unique_values = at::empty({num_unique}, work.options());
  Tensor bound_pos = at::empty({num_unique}, self_flat.options().dtype(kLong));
  Tensor counts = return_counts ? at::empty({num_unique}, self_flat.options().dtype(kLong))
                                : at::empty({0}, self_flat.options().dtype(kLong));
  Tensor inverse = return_inverse ? at::empty({numel}, self_flat.options().dtype(kLong))
                                  : at::empty({0}, self_flat.options().dtype(kLong));

  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();
      {
        id<MTLComputePipelineState> pso =
            lib.getPipelineStateForFunc(fmt::format("unique_emit_{}_{}", type_name, scan_suffix));
        getMPSProfiler().beginProfileKernel(pso, "unique_emit", false);
        [encoder setComputePipelineState:pso];
        mtl_setArgs(encoder, sorted_values, mask, scan, unique_values, bound_pos, static_cast<uint64_t>(numel));
        mtl_dispatch1DJob(encoder, pso, numel);
        getMPSProfiler().endProfileKernel(pso);
      }

      if (return_counts) {
        id<MTLComputePipelineState> pso = lib.getPipelineStateForFunc("unique_counts");
        getMPSProfiler().beginProfileKernel(pso, "unique_counts", false);
        [encoder setComputePipelineState:pso];
        mtl_setArgs(encoder, bound_pos, counts, static_cast<uint64_t>(num_unique), static_cast<uint64_t>(numel));
        mtl_dispatch1DJob(encoder, pso, num_unique);
        getMPSProfiler().endProfileKernel(pso);
      }

      // For the sorted case we scatter via the sort permutation. For
      // consecutive, the inverse is just (scan - 1), which we compute as a
      // tensor op outside this block to avoid re-entering the queue.
      if (return_inverse && !consecutive) {
        id<MTLComputePipelineState> pso = lib.getPipelineStateForFunc(fmt::format("unique_inverse_{}", scan_suffix));
        getMPSProfiler().beginProfileKernel(pso, "unique_inverse", false);
        [encoder setComputePipelineState:pso];
        mtl_setArgs(encoder, sort_idx, scan, inverse, static_cast<uint64_t>(numel));
        mtl_dispatch1DJob(encoder, pso, numel);
        getMPSProfiler().endProfileKernel(pso);
      }
    }
  });

  if (return_inverse && consecutive) {
    inverse = scan.sub(1).to(kLong);
  }

  // inverse_shape can legitimately be {} (0-d scalar input). Only skip the
  // view when we're not returning an inverse at all.
  if (return_inverse) {
    inverse = inverse.view(inverse_shape);
  }

  if (sort_via_byte) {
    unique_values = unique_values.view(in_dtype);
  }

  return std::make_tuple(std::move(unique_values), std::move(inverse), std::move(counts));
}

static std::tuple<Tensor, Tensor, Tensor> _unique_impl_mps(const Tensor& self,
                                                           const bool return_inverse,
                                                           const bool return_counts,
                                                           const bool consecutive,
                                                           std::optional<int64_t> dimOpt) {
  const Tensor& input = self.contiguous();

  // Fast native-Metal path: when no dim is specified, the operator works on
  // a flat view of the input and the dispatch is one sort (skipped when
  // consecutive) plus four small kernels. Avoids the MPSGraph scatter ops
  // whose runtime explodes when the input contains long runs of duplicates.
  if (!dimOpt.has_value()) {
    // Reject unsupported dtypes up front with a clean error message. This
    // list must stay in sync with the REGISTER_UNIQUE_FOR_T registrations
    // in kernels/Unique.metal.
    const auto in_dtype = input.scalar_type();
    const bool fast_path_supports_dtype = in_dtype == kFloat || in_dtype == kHalf || in_dtype == kBFloat16 ||
        in_dtype == kLong || in_dtype == kInt || in_dtype == kShort || in_dtype == kChar || in_dtype == kByte ||
        in_dtype == kBool;
    TORCH_CHECK(fast_path_supports_dtype, "unique is not supported on MPS for dtype ", in_dtype);
    if (input.numel() == 0) {
      Tensor output = at::empty({0}, input.options());
      Tensor inv = at::empty(return_inverse ? input.sizes() : IntArrayRef{}, input.options().dtype(kLong));
      Tensor cnt = at::empty(return_counts ? IntArrayRef{0} : IntArrayRef{}, input.options().dtype(kLong));
      return std::make_tuple(std::move(output), std::move(inv), std::move(cnt));
    }
    Tensor input_flat = input.view({input.numel()});
    IntArrayRef inv_shape = return_inverse ? input.sizes() : IntArrayRef{};
    return _unique_flat_mps_fast(input_flat, return_inverse, return_counts, consecutive, inv_shape);
  }

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

static Tensor lexsort_rows_perm_mps(const Tensor& mat_2d) {
  const auto rows = mat_2d.size(0), cols = mat_2d.size(1);
  if (rows <= 1 || cols == 0) {
    return arange(rows, mat_2d.options().dtype(kLong));
  }

  auto perm = arange(rows, mat_2d.options().dtype(kLong));
  for (auto c = cols - 1; c >= 0; --c) {
    auto keys = mat_2d.select(1, c).index_select(0, perm);
    const auto idx = argsort(keys, /*stable=*/true, /*dim=*/0, /*descending=*/false);
    perm = perm.index_select(0, idx);
  }
  return perm;
}

static std::tuple<Tensor, Tensor, Tensor> unique_dim_sorted_mps_impl(const Tensor& self,
                                                                     int64_t dim,
                                                                     bool return_inverse,
                                                                     bool return_counts) {
  dim = maybe_wrap_dim(dim, self.dim());

  auto sizes = self.sizes().vec();
  auto num_zero_dims = std::count(sizes.begin(), sizes.end(), (int64_t)0);
  if (self.size(dim) == 0) {
    auto output = at::empty(sizes, self.options());
    auto inverse_indices = at::empty({0}, self.options().dtype(kLong));
    auto counts = at::empty({0}, self.options().dtype(kLong));
    return {output, inverse_indices, counts};
  }

  auto transposed = self.moveaxis(dim, 0);
  auto orig_sizes = transposed.sizes().vec();
  auto rows = transposed.size(0);
  auto input_flat = transposed.contiguous().view({rows, -1});

  auto perm = lexsort_rows_perm_mps(input_flat);
  auto input_sorted = input_flat.index_select(0, perm);

  Tensor is_unique = at::zeros({rows}, self.options().dtype(kBool));
  if (rows > 0) {
    is_unique.narrow(0, 0, 1).fill_(true);
  }
  if (rows > 1) {
    auto a = input_sorted.narrow(0, 1, rows - 1);
    auto b = input_sorted.narrow(0, 0, rows - 1);
    auto row_changed = a.ne(b).any(1);
    is_unique.narrow(0, 1, rows - 1).copy_(row_changed);
  }

  auto unique_pos = nonzero(is_unique).squeeze(1);
  auto group_id = cumsum(is_unique.to(kLong), 0).sub(1);

  auto unique_rows_2d = input_sorted.index_select(0, unique_pos);

  Tensor inverse_indices = empty({0}, self.options().dtype(kLong));
  if (return_inverse) {
    inverse_indices = empty({rows}, self.options().dtype(kLong));
    inverse_indices.index_copy_(0, perm, group_id);
  }

  Tensor counts = empty({0}, self.options().dtype(kLong));
  if (return_counts) {
    const auto num_unique = unique_pos.size(0);
    counts = zeros({num_unique}, self.options().dtype(kLong));
    counts.scatter_add_(0, group_id, ones_like(group_id, group_id.options().dtype(kLong)));
  }

  orig_sizes[0] = unique_rows_2d.size(0);
  auto output = unique_rows_2d.view(orig_sizes).moveaxis(0, dim);

  return std::make_tuple(std::move(output), std::move(inverse_indices), std::move(counts));
}

std::tuple<Tensor, Tensor, Tensor> unique_dim_mps(const Tensor& self,
                                                  int64_t dim,
                                                  const bool /*sorted*/,
                                                  const bool return_inverse,
                                                  const bool return_counts) {
  return unique_dim_sorted_mps_impl(self, dim, return_inverse, return_counts);
}

} // namespace at::native
