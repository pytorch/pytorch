//  Copyright © 2022 Apple Inc.
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/mps/OperationUtils.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/gather_native.h>
#include <ATen/ops/scatter_add_native.h>
#include <ATen/ops/scatter_native.h>
#endif

namespace at::native {

TORCH_IMPL_FUNC(gather_out_mps)
(const Tensor& self_arg, int64_t dim, const Tensor& index, bool sparse_grad, const Tensor& output) {
  using namespace mps;

  if (self_arg.numel() == 0 || index.numel() == 0) {
    return;
  }
  auto self = self_arg.dim() == 0 ? self_arg.view({1}) : self_arg;
  dim = at::maybe_wrap_dim(dim, self.dim());

  TORCH_CHECK(!sparse_grad, "sparse_grad not supported in MPS yet")
  TORCH_CHECK(self.scalar_type() == output.scalar_type(), "gather(): self and output must have the same scalar type");
  TORCH_CHECK(dim >= 0 && dim < self.dim(), "gather(): Indexing dim ", dim, " is out of bounds of tensor");
  TORCH_CHECK(!self.is_complex(), "gather(): Yet not supported for complex");

  struct CachedGraph : public MPSCachedGraph {
    CachedGraph(MPSGraph* graph) : MPSCachedGraph(graph) {}
    MPSGraphTensor* inputTensor_ = nil;
    MPSGraphTensor* indexTensor_ = nil;
    MPSGraphTensor* outputTensor_ = nil;
  };

  @autoreleasepool {
    MPSShape* input_shape = getMPSShape(self);
    MPSShape* index_shape = getMPSShape(index);
    uint32_t num_input_dims = [input_shape count];
    uint32_t num_index_dims = [index_shape count];
    TORCH_CHECK(num_input_dims == num_index_dims, "Input and index must have same rank")

    // Determine if we need to slice into the input tensor
    bool needSlice = false;

    for (const auto i : c10::irange(num_input_dims)) {
      TORCH_CHECK(i == dim || [index_shape[i] intValue] <= [input_shape[i] intValue],
                  "Index dim must not exceed input dim except at gathering axis")
      if (i != dim && [index_shape[i] intValue] < [input_shape[i] intValue])
        needSlice = true;
    }
    auto input_type = getMPSDataType(self);
    auto output_type = getMPSDataType(output);
    if (input_type == MPSDataTypeUInt8) {
      input_type = MPSDataTypeInt8;
    }
    if (output_type == MPSDataTypeUInt8) {
      output_type = MPSDataTypeInt8;
    }
    string key = "gather_out_mps" + getTensorsStringKey({self, index, output}) + ":" + std::to_string(dim);
    auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      MPSGraphTensor* inputTensor = mpsGraphRankedPlaceHolder(mpsGraph, input_type, getMPSShape(self));
      MPSGraphTensor* indexTensor = mpsGraphRankedPlaceHolder(mpsGraph, index);

      MPSGraphTensor* getInput = inputTensor;

      // Slice into the input tensor IF NEEDED
      if (needSlice) {
        NSMutableArray<NSNumber*>* starts = [NSMutableArray<NSNumber*> arrayWithCapacity:num_input_dims];
        NSMutableArray<NSNumber*>* ends = [NSMutableArray<NSNumber*> arrayWithCapacity:num_input_dims];
        NSMutableArray<NSNumber*>* strides = [NSMutableArray<NSNumber*> arrayWithCapacity:num_input_dims];

        for (const auto i : c10::irange(num_input_dims)) {
          // All strides are 1
          strides[i] = @1;
          // All starts are 0
          starts[i] = @0;
          ends[i] = (i != dim) ? index_shape[i] : input_shape[i];
        }

        getInput = [mpsGraph sliceTensor:inputTensor starts:starts ends:ends strides:strides name:nil];
      }

      MPSGraphTensor* castIndexTensor = [mpsGraph castTensor:indexTensor
                                                      toType:MPSDataTypeInt32
                                                        name:(NSString* _Nonnull)nil];

      C10_DIAGNOSTIC_PUSH_AND_IGNORED_IF_DEFINED("-Wobjc-method-access")
      MPSGraphTensor* outputTensor = [mpsGraph gatherAlongAxis:(NSInteger)dim
                                             withUpdatesTensor:getInput
                                                 indicesTensor:castIndexTensor
                                                          name:nil];
      C10_DIAGNOSTIC_POP()
      newCachedGraph->inputTensor_ = inputTensor;
      newCachedGraph->indexTensor_ = indexTensor;
      newCachedGraph->outputTensor_ = outputTensor;
    });

    Placeholder selfPlaceholder = Placeholder(cachedGraph->inputTensor_, self, input_shape, true, input_type);
    Placeholder indexPlaceholder = Placeholder(cachedGraph->indexTensor_, index, index_shape);
    Placeholder outputPlaceholder = Placeholder(cachedGraph->outputTensor_, output, nullptr, false, output_type);

    auto feeds = dictionaryFromPlaceholders(selfPlaceholder, indexPlaceholder);
    runMPSGraph(getCurrentMPSStream(), cachedGraph->graph(), feeds, outputPlaceholder);
  }
}

static void scatter_mps_general(const Tensor& self_arg,
                                int64_t dim,
                                const Tensor& index,
                                const Tensor& src,
                                const Tensor& output,
                                string func_name,
                                const c10::string_view reduce) {
  using namespace mps;

  if (self_arg.numel() == 0 || index.numel() == 0 || src.numel() == 0) {
    return;
  }
  auto self = self_arg.dim() == 0 ? self_arg.view({1}) : self_arg;
  dim = at::maybe_wrap_dim(dim, self.dim());

  TORCH_CHECK(self.scalar_type() == output.scalar_type() && output.scalar_type() == src.scalar_type(),
              "scatter(): self, src and output must have the same scalar type");
  TORCH_CHECK(dim >= 0 && dim < self.dim(), "scatter(): Indexing dim ", dim, " is out of bounds of tensor");
  TORCH_CHECK(!self.is_complex(), "scatter(): Yet not supported for complex");

  struct CachedGraph : public MPSCachedGraph {
    CachedGraph(MPSGraph* graph) : MPSCachedGraph(graph) {}
    MPSGraphTensor* inputTensor_ = nil;
    MPSGraphTensor* indexTensor_ = nil;
    MPSGraphTensor* srcTensor_ = nil;
    MPSGraphTensor* outputTensor_ = nil;
  };

  @autoreleasepool {
    MPSShape* input_shape = getMPSShape(self);
    MPSShape* index_shape = getMPSShape(index);
    MPSShape* src_shape = getMPSShape(src);
    uint32_t num_input_dims = [input_shape count];
    uint32_t num_index_dims = [index_shape count];
    uint32_t num_src_dims = [src_shape count];

    TORCH_CHECK(num_input_dims == num_index_dims && num_index_dims == num_src_dims,
                "Input, index and src must have same rank")

    // Do we need to slice into the src tensor?
    bool needSlice = false;
    bool inputNeedSlice = false;
    bool needsCast = false;

    for (const auto i : c10::irange(num_input_dims)) {
      TORCH_CHECK(i == dim || [index_shape[i] intValue] <= [input_shape[i] intValue],
                  "Index dim must not exceed input dim except at gathering axis")
      TORCH_CHECK([index_shape[i] intValue] <= [src_shape[i] intValue],
                  "Index dim must not exceed input dim except at gathering axis")
      if ([index_shape[i] intValue] < [src_shape[i] intValue])
        needSlice = true;
      if (i != dim && [index_shape[i] intValue] < [input_shape[i] intValue])
        inputNeedSlice = true;
    }
    TORCH_CHECK(reduce != "mean", "Scatter reduce mean mode not yet supported in MPS")

    MPSDataType src_type = getMPSDataType(src);
    if (reduce != "set" || src_type == MPSDataTypeUInt8 || src_type == MPSDataTypeBool) {
      src_type = isFloatingType(src.scalar_type()) ? MPSDataTypeFloat32 : MPSDataTypeInt32;
      needsCast = true;
    }

    string key = func_name + getTensorsStringKey({self, index, src, output}) + ":" + std::to_string(dim) + ":" +
        std::string(reduce);
    auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      MPSGraphTensor* inputTensor = mpsGraphRankedPlaceHolder(mpsGraph, self);
      MPSGraphTensor* indexTensor = mpsGraphRankedPlaceHolder(mpsGraph, index);
      MPSGraphTensor* srcTensor = mpsGraphRankedPlaceHolder(mpsGraph, src);

      MPSGraphTensor* outputTensor = nil;
      MPSGraphTensor* castSrcTensor = srcTensor;
      MPSGraphTensor* castInputTensor = inputTensor;

      if (needsCast) {
        castSrcTensor = [mpsGraph castTensor:srcTensor toType:src_type name:@"cast"];
        castInputTensor = [mpsGraph castTensor:inputTensor toType:src_type name:@"cast"];
      }
      MPSGraphTensor* castIndexTensor = [mpsGraph castTensor:indexTensor toType:MPSDataTypeInt32 name:@"cast"];

      MPSGraphTensor* slicedSrc = castSrcTensor;
      MPSGraphTensor* slicedInput = castInputTensor;

      // Use in case input needs to be smaller to get scatter
      NSMutableArray<NSNumber*>* scatterInputShape = [NSMutableArray arrayWithArray:input_shape];

      // Slice into the src or input tensors IF NEEDED
      if (needSlice || inputNeedSlice) {
        NSMutableArray<NSNumber*>* starts = [NSMutableArray<NSNumber*> arrayWithCapacity:num_input_dims];
        NSMutableArray<NSNumber*>* strides = [NSMutableArray<NSNumber*> arrayWithCapacity:num_input_dims];
        NSMutableArray<NSNumber*>* ends_src = [NSMutableArray<NSNumber*> arrayWithCapacity:num_input_dims];

        for (const auto i : c10::irange(num_input_dims)) {
          strides[i] = @1;
          starts[i] = @0;
          ends_src[i] = index_shape[i];
          scatterInputShape[i] = (i != dim) ? index_shape[i] : input_shape[i];
        }
        if (needSlice) {
          slicedSrc = [mpsGraph sliceTensor:castSrcTensor starts:starts ends:ends_src strides:strides name:nil];
        }
        if (inputNeedSlice) {
          slicedInput = [mpsGraph sliceTensor:castInputTensor
                                       starts:starts
                                         ends:scatterInputShape
                                      strides:strides
                                         name:nil];
        }
      }
      MPSGraphScatterMode scatter_mode = MPSGraphScatterModeSet;

      if (reduce == "sum" || reduce == "add")
        scatter_mode = MPSGraphScatterModeAdd;
      else if (reduce == "prod" || reduce == "multiply")
        scatter_mode = MPSGraphScatterModeMul;
      else if (reduce == "amax")
        scatter_mode = MPSGraphScatterModeMax;
      else if (reduce == "amin")
        scatter_mode = MPSGraphScatterModeMin;

      // Scatter this into the input with set mode
      C10_DIAGNOSTIC_PUSH_AND_IGNORED_IF_DEFINED("-Wobjc-method-access")
      MPSGraphTensor* scatterTensor = [mpsGraph scatterAlongAxis:(NSInteger)dim
                                                  withDataTensor:slicedInput
                                                   updatesTensor:slicedSrc
                                                   indicesTensor:castIndexTensor
                                                            mode:scatter_mode
                                                            name:nil];
      C10_DIAGNOSTIC_POP()
      if (inputNeedSlice) {
        // Make an array of scatter indices tensors
        NSMutableArray<MPSGraphTensor*>* indicesTensors =
            [NSMutableArray<MPSGraphTensor*> arrayWithCapacity:num_input_dims];

        // 1. Concatenate the coord tensors
        // 2. Flatten the values
        // 3. Scatter into input with add mode

        std::vector<int> shape_data(num_input_dims);

        for (const auto i : c10::irange(num_input_dims)) {
          shape_data[i] = {[scatterInputShape[i] intValue]};
        }

        MPSGraphTensor* scatterInputShapeTensor =
            [mpsGraph constantWithData:[NSData dataWithBytes:shape_data.data() length:num_input_dims * sizeof(int)]
                                 shape:@[ [NSNumber numberWithUnsignedInt:num_input_dims] ]
                              dataType:MPSDataTypeInt32];

        for (const auto i : c10::irange(num_input_dims)) {
          MPSGraphTensor* axisTensor = [mpsGraph constantWithScalar:i dataType:MPSDataTypeInt32];
          MPSGraphTensor* scatter_currentIndexTensor = [mpsGraph coordinateAlongAxisTensor:axisTensor
                                                                           withShapeTensor:scatterInputShapeTensor
                                                                                      name:nil];
          scatter_currentIndexTensor = [mpsGraph reshapeTensor:scatter_currentIndexTensor
                                                     withShape:@[ @-1, @1 ]
                                                          name:nil];
          indicesTensors[i] = scatter_currentIndexTensor;
        }

        MPSGraphTensor* scatter_fullIndexTensor = [mpsGraph concatTensors:indicesTensors
                                                                dimension:(NSInteger)1
                                                                     name:nil];

        MPSGraphTensor* flatValuesTensor = [mpsGraph reshapeTensor:scatterTensor withShape:@[ @-1 ] name:nil];

        outputTensor = [mpsGraph scatterNDWithDataTensor:castInputTensor
                                           updatesTensor:flatValuesTensor
                                           indicesTensor:scatter_fullIndexTensor
                                         batchDimensions:0
                                                    mode:MPSGraphScatterModeSet
                                                    name:nil];
      } else {
        outputTensor = scatterTensor;
      }
      newCachedGraph->inputTensor_ = inputTensor;
      newCachedGraph->srcTensor_ = srcTensor;
      newCachedGraph->indexTensor_ = indexTensor;
      newCachedGraph->outputTensor_ =
          needsCast ? castMPSTensor(mpsGraph, outputTensor, output.scalar_type()) : outputTensor;
    });

    Placeholder selfPlaceholder = Placeholder(cachedGraph->inputTensor_, self, input_shape);
    Placeholder srcPlaceholder = Placeholder(cachedGraph->srcTensor_, src, src_shape);
    Placeholder indexPlaceholder = Placeholder(cachedGraph->indexTensor_, index, index_shape);
    Placeholder outputPlaceholder = Placeholder(cachedGraph->outputTensor_, output);

    auto feeds = dictionaryFromPlaceholders(selfPlaceholder, srcPlaceholder, indexPlaceholder);
    runMPSGraph(getCurrentMPSStream(), cachedGraph->graph(), feeds, outputPlaceholder);
  }
}

TORCH_IMPL_FUNC(scatter_src_out_mps)
(const Tensor& self, int64_t dim, const Tensor& index, const Tensor& src, const Tensor& output) {
  scatter_mps_general(self, dim, index, src, output, "scatter_src_out_mps", "set");
}

TORCH_IMPL_FUNC(scatter_value_out_mps)
(const Tensor& self, int64_t dim, const Tensor& index, const Scalar& value, const Tensor& output) {
  Tensor src =
      at::empty(index.sizes(), self.scalar_type(), std::nullopt, kMPS, std::nullopt, self.suggest_memory_format());
  src.fill_(value);
  scatter_mps_general(self, dim, index, const_cast<Tensor&>(src), output, "scatter_value_out_mps", "set");
}

TORCH_IMPL_FUNC(scatter_reduce_out_mps)
(const Tensor& self,
 int64_t dim,
 const Tensor& index,
 const Tensor& src,
 const c10::string_view reduce,
 const Tensor& output) {
  scatter_mps_general(self, dim, index, src, output, "scatter_reduce_out_mps", reduce);
}

TORCH_IMPL_FUNC(scatter_value_reduce_out_mps)
(const Tensor& self,
 int64_t dim,
 const Tensor& index,
 const Scalar& value,
 const c10::string_view reduce,
 const Tensor& output) {
  Tensor src =
      at::empty(index.sizes(), self.scalar_type(), std::nullopt, kMPS, std::nullopt, self.suggest_memory_format());
  src.fill_(value);
  scatter_mps_general(self, dim, index, const_cast<Tensor&>(src), output, "scatter_value_reduce_out_mps", reduce);
}

TORCH_IMPL_FUNC(scatter_add_mps_out)
(const Tensor& self, int64_t dim, const Tensor& index, const Tensor& src, const Tensor& output) {
  scatter_mps_general(self, dim, index, src, output, "scatter_add_mps_out", "add");
}

} // namespace at::native
