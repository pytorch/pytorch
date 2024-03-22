//  Copyright © 2022 Apple Inc.
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/TensorIterator.h>
#include <ATen/mps/MPSAllocatorInterface.h>
#include <ATen/mps/MPSProfiler.h>
#include <ATen/native/mps/MPSGraphSonomaOps.h>
#include <ATen/native/mps/MPSGraphVenturaOps.h>
#include <ATen/native/mps/OperationUtils.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/scalar_tensor.h>
#endif

namespace at::native::mps {

void dispatch_sync_with_rethrow(dispatch_queue_t queue, void (^block)()) {
  __block std::optional<std::exception_ptr> block_exception;
  dispatch_sync(queue, ^() {
    try {
      block();
    } catch (...) {
      block_exception = std::current_exception();
    }
  });
  if (block_exception) {
    std::rethrow_exception(*block_exception);
  }
}

/**
 * Computes distance from lowest to highest element offset in given tensor.
 */
size_t compute_storage_numel_distance(const at::Tensor& t) {
  size_t rc = 1;
  if (t.numel() == 0) {
    return 0;
  }
  for (const auto i : c10::irange(t.dim())) {
    assert(t.size(i) > 0);
    rc += (t.size(i) - 1) * t.stride(i);
  }
  return rc;
}

void runMPSGraph(MPSStream* mpsStream, MPSGraph* mpsGraph, NSDictionary* feeds, NSDictionary* results) {
  mpsStream->executeMPSGraph(mpsGraph, feeds, results, SyncType::COMMIT_ADAPTIVE);
}

static inline void checkSupportsComplex() {
  TORCH_CHECK_TYPE(supportsComplex(), "MPS complex types are only supported on MacOS 14.0 or newer.");
}

static inline void checkSupportsBFloat16() {
  TORCH_CHECK_TYPE(is_macos_13_or_newer(MacOSVersion::MACOS_VER_14_0_PLUS),
                   "MPS bfloat16 type is supported on MacOS 14.0 or newer.");
}

MPSDataType getMPSDataType(ScalarType scalar_type) {
  switch (scalar_type) {
    case ScalarType::Float:
      return MPSDataTypeFloat32;
    case ScalarType::Half:
      return MPSDataTypeFloat16;
    case ScalarType::BFloat16:
      checkSupportsBFloat16();
      return MPSDataTypeBFloat16;
    case ScalarType::Int:
      return MPSDataTypeInt32;
    case ScalarType::Long:
      return MPSDataTypeInt64;
    case ScalarType::Short:
      return MPSDataTypeInt16;
    case ScalarType::Char:
      return MPSDataTypeInt8;
    case ScalarType::Byte:
      return MPSDataTypeUInt8;
    case ScalarType::Bool:
      return MPSDataTypeBool;
    case ScalarType::Double:
      TORCH_CHECK_TYPE(false,
                       "Cannot convert a float64 Tensor to MPS as the MPS framework doesn't support float64. "
                       "Please use float32 instead.")
    case ScalarType::ComplexHalf:
      checkSupportsComplex();
      return MPSDataTypeComplexFloat16;
    case ScalarType::ComplexFloat:
      checkSupportsComplex();
      return MPSDataTypeComplexFloat32;
    default:
      TORCH_CHECK_TYPE(
          false, "Trying to convert ", scalar_type, " to the MPS backend but it does not have support for that dtype.")
  }
}

// #issue 104398441 sortWithTensor and argsortWithTensor has support of
// Int32, Half and Float32 types. These utilities are to help cast to these
// types.
MPSGraphTensor* castToIHFTypes(MPSGraph* mpsGraph,
                               MPSGraphTensor* inputTensor,
                               const Tensor& input,
                               bool includesInt64) {
  MPSDataType dataType = getMPSDataType(input.scalar_type());
  bool condition =
      (dataType != MPSDataTypeInt32) && (dataType != MPSDataTypeFloat32) && (dataType != MPSDataTypeFloat16);
  if (includesInt64) {
    condition = condition && (dataType != MPSDataTypeInt64);
  }
  if (condition) {
    dataType = (dataType & MPSDataTypeFloatBit) ? MPSDataTypeFloat32 : MPSDataTypeInt32;
    return [mpsGraph castTensor:inputTensor toType:dataType name:@"castInputTensor"];
  }
  return inputTensor;
}

// #issue 104398441 sortWithTensor and argsortWithTensor has support of
// Int32, Half and Float32 types. These utilities are to help cast from these
// types.
MPSGraphTensor* castFromIHFTypes(MPSGraph* mpsGraph,
                                 MPSGraphTensor* inputTensor,
                                 const Tensor& input,
                                 bool includesInt64) {
  MPSDataType dataType = getMPSDataType(input.scalar_type());
  bool condition =
      (dataType != MPSDataTypeInt32) && (dataType != MPSDataTypeFloat32) && (dataType != MPSDataTypeFloat16);
  if (includesInt64) {
    condition = condition && (dataType != MPSDataTypeInt64);
  }
  if (condition) {
    inputTensor = [mpsGraph castTensor:inputTensor toType:dataType name:@"castInputTensor"];
  }
  return inputTensor;
}

MPSDataType getMPSScalarType(ScalarType scalar_type) {
  switch (scalar_type) {
    // This is an intentional fallthrough supporting Double for Scalar
    // types as they are casted to Float32 currently.
    case ScalarType::Double:
    case ScalarType::Float:
      return MPSDataTypeFloat32;
    case ScalarType::Half:
      return MPSDataTypeFloat16;
    case ScalarType::BFloat16:
      checkSupportsBFloat16();
      return MPSDataTypeBFloat16;
    case ScalarType::Int:
      return MPSDataTypeInt32;
    case ScalarType::Long:
      return MPSDataTypeInt64;
    case ScalarType::Short:
      return MPSDataTypeInt16;
    case ScalarType::Char:
      return MPSDataTypeInt8;
    case ScalarType::Byte:
      return MPSDataTypeUInt8;
    case ScalarType::Bool:
      return MPSDataTypeBool;
    case ScalarType::ComplexHalf:
      checkSupportsComplex();
      return MPSDataTypeComplexFloat16;
    // This is an intentional fallthrough supporting ComplexDouble for Scalar
    // types as they are casted to Complex64 currently.
    case ScalarType::ComplexDouble:
    case ScalarType::ComplexFloat:
      checkSupportsComplex();
      return MPSDataTypeComplexFloat32;
    default:
      TORCH_CHECK_TYPE(
          false, "Trying to convert ", scalar_type, " to the MPS backend but it does not have support for that dtype.")
  }
}

// use short_name to avoid getting extra long cached graph keys with ops such as cat_out(), etc.
std::string getMPSTypeString(ScalarType scalar_type, bool short_name) {
  switch (scalar_type) {
    case ScalarType::Double:
    case ScalarType::Float:
      return short_name ? "f32" : "Float32";
    case ScalarType::Half:
      return short_name ? "f16" : "Float16";
    case ScalarType::BFloat16:
      return short_name ? "bf16" : "BFloat16";
    case ScalarType::Int:
      return short_name ? "i32" : "Int32";
    case ScalarType::Long:
      return short_name ? "i64" : "Int64";
    case ScalarType::Short:
      return short_name ? "i16" : "Int16";
    case ScalarType::Char:
      return short_name ? "i8" : "Int8";
    case ScalarType::Byte:
      return short_name ? "u8" : "UInt8";
    case ScalarType::Bool:
      return short_name ? "b8" : "Bool";
    case ScalarType::ComplexHalf:
      return short_name ? "c16" : "ComplexFloat16";
    case ScalarType::ComplexFloat:
      return short_name ? "c32" : "ComplexFloat32";
    default:
      return "Undefined";
  }
}

std::string scalarToMetalTypeString(const c10::ScalarType& scalar_type) {
  switch (scalar_type) {
    case ScalarType::Float:
      return "float";
    case ScalarType::Half:
      return "half";
    case ScalarType::BFloat16:
      checkSupportsBFloat16();
      return "bfloat";
    case ScalarType::Int:
      return "int";
    case ScalarType::Long:
      return "long";
    case ScalarType::Short:
      return "short";
    case ScalarType::Char:
      return "char";
    case ScalarType::Byte:
      return "uchar";
    case ScalarType::Bool:
      return "bool";
    default:
      TORCH_CHECK(false, "Undefined type ", scalar_type);
      return "Undefined";
  }
}

static NSArray<NSNumber*>* getTensorAxes(int64_t ndim) {
  auto axes = [NSMutableArray<NSNumber*> arrayWithCapacity:ndim];
  for (const auto i : c10::irange(ndim)) {
    axes[i] = [NSNumber numberWithInteger:i];
  }
  return axes;
}

NSArray<NSNumber*>* getTensorAxes(const Tensor& t) {
  return getTensorAxes(t.dim());
}

static NSArray<NSNumber*>* getTensorAxes(const IntArrayRef& sizes) {
  return getTensorAxes(sizes.size());
}

NSArray<NSNumber*>* getTensorAxes(const IntArrayRef& sizes, at::OptionalIntArrayRef dim) {
  if (dim.has_value() && dim.value().size() != 0) {
    IntArrayRef dimValues = dim.value();
    int ndim = dimValues.size();
    auto axes = [NSMutableArray<NSNumber*> arrayWithCapacity:ndim];
    for (const auto i : c10::irange(ndim)) {
      axes[i] = [NSNumber numberWithInteger:dimValues[i]];
    }

    return axes;
  }

  return getTensorAxes(sizes);
}

std::string getMPSShapeString(MPSShape* shape) {
  std::string str;
  for (NSNumber* elem in shape) {
    str += std::to_string(elem.unsignedLongValue) + ",";
  }
  return str;
}

std::string getArrayRefString(const IntArrayRef s) {
  std::stringstream ss;
  std::copy(s.begin(), s.end(), std::ostream_iterator<int>(ss, ","));
  return ss.str();
}

std::string getTensorsStringKey(const TensorList& tensors, bool short_dtype) {
  std::string str;
  // The key format per tensor would look like ":Float32[1,1,1,10]:"
  for (const Tensor& tensor : tensors) {
    str += ":";
    if (tensor.defined()) {
      str += getMPSTypeString(tensor.scalar_type(), short_dtype) + "[";
      // if tensor is a scalar
      if (tensor.dim() == 0) {
        str += "Scalar";
      } else {
        const NSString* ns_shape_key = [[getMPSShape(tensor) valueForKey:@"description"] componentsJoinedByString:@","];
        str += std::string(ns_shape_key.UTF8String);
      }
      str += "]";
    } else {
      str += "Undefined";
    }
  }
  return str;
}

MPSShape* getMPSShape(const Tensor& t, c10::MemoryFormat memory_format) {
  return getMPSShape(t.sizes(), memory_format);
}

MPSShape* getMPSShape(IntArrayRef sizes, c10::MemoryFormat memory_format) {
  if (memory_format == MemoryFormat::ChannelsLast) {
    TORCH_INTERNAL_ASSERT(sizes.size() == 4, "ChannelsLast memory format must have 4 dimensions!");
    const NSUInteger N = sizes[0];
    const NSUInteger C = sizes[1];
    const NSUInteger H = sizes[2];
    const NSUInteger W = sizes[3];
    return @[ @(N), @(H), @(W), @(C) ];
  }
  const int sz = sizes.size();
  const int sz_ = (sz > 0) ? sz : 1;

  std::vector<NSNumber*> numbers(sz_);

  for (int i = 0; i < sz_; i++) {
    NSInteger sz_i = (i < sz) ? sizes[i] : 1;
    NSNumber* number = [NSNumber numberWithInteger:sz_i];
    numbers[i] = number;
  }
  return [NSArray arrayWithObjects:numbers.data() count:numbers.size()];
}

void printTensorNDArray(const Tensor& t) {
  if (!t.is_mps())
    return;
  if (t.numel() == 0)
    return;
  // Get shape and data type
  auto selfShape = getMPSShape(t);
  auto selfDType = getMPSDataType(t.scalar_type());

  // Initialize data
  id<MTLBuffer> selfBuf = getMTLBufferStorage(t);
  MPSGraphTensorData* tdata = [[[MPSGraphTensorData alloc] initWithMTLBuffer:selfBuf shape:selfShape
                                                                    dataType:selfDType] autorelease];
  C10_CLANG_DIAGNOSTIC_PUSH()
#if C10_CLANG_HAS_WARNING("-Wobjc-method-access")
  C10_CLANG_DIAGNOSTIC_IGNORE("-Wobjc-method-access")
#endif
  [tdata printNDArray];
  C10_CLANG_DIAGNOSTIC_POP()
}

MPSNDArray* ndArrayFromTensor(const Tensor& tensor, MPSShape* shape, MPSDataType mpsType) {
  id<MTLBuffer> buffer = getMTLBufferStorage(tensor);
  MPSGraphTensorData* tmpGraphTensorData = [[[MPSGraphTensorData alloc] initWithMTLBuffer:buffer
                                                                                    shape:shape
                                                                                 dataType:mpsType] autorelease];

  return [tmpGraphTensorData mpsndarray];
}

Placeholder::Placeholder(MPSGraphTensor* mpsGraphTensor,
                         const Tensor& src,
                         MPSShape* mpsShape,
                         bool gatherTensorData,
                         MPSDataType dataType)
    : _tensor(src) {
  TORCH_CHECK(src.is_mps(), "Placeholder storage has not been allocated on MPS device!");
  // extract the pointer to MTLBuffer from the Tensor's storage
  id<MTLBuffer> srcBuf = getMTLBufferStorage(src);
  bool sliceViewTensor = canSliceViewTensor(src, mpsShape);
  // a view tensor could be contiguous (e.g., slice ops) or non-contiguous (e.g., transpose())
  if ((!src.is_contiguous() || (src.storage_offset() && !sliceViewTensor)) && gatherTensorData) {
    Tensor emptyShell = Tensor();
    // use "_tensor" from Placeholder to retain view's output during its usage in other ops
    _tensor = gatherViewTensor(src, emptyShell);
    if (!_tensor.has_storage()) {
      // if we cannot gather, we make the tensor contiguous implicitly, and keep
      // it in placeholder to be able to retrieve it when we return from constructor
      _tensor = src.clone(MemoryFormat::Contiguous);
    }
    srcBuf = getMTLBufferStorage(_tensor);
  }

  // tensor.numel() could be zero, but tensor is valid as long as the buffer size is non-zero.
  // if buffer size is zero in here, it's not a user error. It could be a missing check for
  // tensor.numel() == 0 in our internal implementations of ops.
  TORCH_INTERNAL_ASSERT([srcBuf length] > 0, "Placeholder tensor is empty!");
  if (dataType == MPSDataTypeInvalid) {
    const auto scalar_type = _tensor.scalar_type();
    dataType = _tensor.dim() == 0 ? getMPSScalarType(scalar_type) : getMPSDataType(scalar_type);
  }
  if (src.is_contiguous() && src.storage_offset() && sliceViewTensor) {
    _value = getMPSGraphTensorDataForView(src, mpsShape, dataType);
  } else {
    _value = [[[MPSGraphTensorData alloc] initWithMTLBuffer:srcBuf
                                                      shape:mpsShape ? mpsShape : getMPSShape(_tensor)
                                                   dataType:dataType] autorelease];
  }

  TORCH_INTERNAL_ASSERT(_value);
  _placeholder = mpsGraphTensor;
}

MPSGraphTensorData* getMPSGraphTensorData(MPSGraph* mpsGraph, MPSStream* mpsStream, const Tensor& tensor) {
  auto mpsShape = getMPSShape(tensor);
  auto dataType = getMPSDataType(tensor.scalar_type());

  MPSGraphTensorData* result = nil;
  if (tensor.numel() > 0) {
    id<MTLBuffer> buf = getMTLBufferStorage(tensor);
    result = [[[MPSGraphTensorData alloc] initWithMTLBuffer:buf shape:mpsShape dataType:dataType] autorelease];
  } else {
    // create empty NDArray
    MPSNDArrayDescriptor* desc = [MPSNDArrayDescriptor descriptorWithDataType:dataType shape:mpsShape];
    MPSNDArray* emptyArray = [[[MPSNDArray alloc] initWithDevice:mpsStream->device() descriptor:desc] autorelease];
    result = [[[MPSGraphTensorData alloc] initWithMPSNDArray:emptyArray] autorelease];
  }
  TORCH_INTERNAL_ASSERT(result);
  return result;
}

MPSScalar getMPSScalar(const Scalar& scalar, ScalarType type) {
  switch (type) {
    case ScalarType::Double:
    case ScalarType::Float:
      return {.value.f = scalar.to<float>(), .size = sizeof(float), .type = type};
    case ScalarType::Half:
      return {.value.h = scalar.to<at::Half>(), .size = sizeof(short), .type = type};
    case ScalarType::BFloat16:
      return {.value.bf16 = scalar.to<at::BFloat16>(), .size = sizeof(short), .type = type};
    case ScalarType::Long:
      return {.value.i = scalar.to<int64_t>(), .size = sizeof(int64_t), .type = type};
    case ScalarType::Int:
      return {.value.i = scalar.to<int32_t>(), .size = sizeof(int32_t), .type = type};
    case ScalarType::Short:
      return {.value.i = scalar.to<int16_t>(), .size = sizeof(int16_t), .type = type};
    case ScalarType::Char:
      return {.value.i = scalar.to<int8_t>(), .size = sizeof(int8_t), .type = type};
    case ScalarType::Byte:
      return {.value.i = scalar.to<uint8_t>(), .size = sizeof(uint8_t), .type = type};
    case ScalarType::Bool:
      return {.value.b = scalar.to<bool>(), .size = sizeof(bool), .type = type};
    case ScalarType::ComplexHalf:
      return {.value.ch = scalar.to<c10::complex<at::Half>>(), .size = sizeof(int32_t), .type = type};
    case ScalarType::ComplexFloat:
    case ScalarType::ComplexDouble:
      return {.value.cf = scalar.to<c10::complex<float>>(), .size = sizeof(int64_t), .type = type};
    default:
      TORCH_INTERNAL_ASSERT(false, "Unsupported scalar type '", type, "' on MPS backend.");
  }
}

MPSGraphTensorData* getMPSGraphTensorFromScalar(MPSStream* mpsStream, MPSScalar& scalar) {
  MPSGraphTensorData* result = nullptr;
  // Scalar pools are only supported on devices with unified memory
  if (mpsStream->device().hasUnifiedMemory) {
    scalar.buffer = getIMPSAllocator()->allocScalarBufferWithValue(&scalar.value, scalar.size);
    result = [[[MPSGraphTensorData alloc] initWithMTLBuffer:scalar.getMTLBuffer()
                                                      shape:@[ @1 ]
                                                   dataType:getMPSScalarType(scalar.type)] autorelease];
  } else {
    MPSNDArrayDescriptor* tensorDesc = [MPSNDArrayDescriptor descriptorWithDataType:getMPSScalarType(scalar.type)
                                                                              shape:@[ @1 ]];
    MPSNDArray* tensorNDArray = [[[MPSNDArray alloc] initWithDevice:mpsStream->device()
                                                         descriptor:tensorDesc] autorelease];
    [tensorNDArray writeBytes:&scalar.value strideBytes:nil];
    result = [[[MPSGraphTensorData alloc] initWithMPSNDArray:tensorNDArray] autorelease];
  }
  return result;
}

void resize_tensor(Tensor* output) {
  output->resize_(output->sizes());
}

Tensor wrapped_scalar_tensor_mps(const Scalar& scalar, const Device device) {
  // Copied and modified from aten/stc/ATen/ScalarOps.h
  // as MPS doesn't support float64 tensor.
  Tensor tensor;
  if (scalar.isFloatingPoint()) {
    tensor = at::scalar_tensor(scalar, at::device(device).dtype(at::kFloat));
  } else if (scalar.isBoolean()) {
    tensor = at::scalar_tensor(scalar, at::device(device).dtype(at::kBool));
  } else if (scalar.isComplex()) {
    tensor = at::scalar_tensor(scalar, at::device(device).dtype(at::kComplexDouble));
  } else {
    TORCH_INTERNAL_ASSERT(scalar.isIntegral(false));
    tensor = at::scalar_tensor(scalar, at::device(device).dtype(at::kLong));
  }
  tensor.unsafeGetTensorImpl()->set_wrapped_number(true);
  return tensor;
}

MPSGraph* make_mps_graph() {
  MPSGraph* mpsGraph = [[MPSGraph new] autorelease];
  return mpsGraph;
}

MPSGraphTensor* mpsGraphUnrankedPlaceHolder(MPSGraph* mpsGraph, MPSDataType dataType) {
  return [mpsGraph placeholderWithShape:nil dataType:dataType name:nil];
}

MPSGraphTensor* mpsGraphRankedPlaceHolder(MPSGraph* mpsGraph, MPSDataType dataType, MPSShape* mpsShape) {
  return [mpsGraph placeholderWithShape:mpsShape dataType:dataType name:nil];
}

MPSGraphTensor* mpsGraphRankedPlaceHolder(MPSGraph* mpsGraph, const Tensor& tensor) {
  return [mpsGraph placeholderWithShape:getMPSShape(tensor) dataType:getMPSScalarType(tensor.scalar_type()) name:nil];
}

MPSGraphTensor* mpsGraphScalarPlaceHolder(MPSGraph* mpsGraph, MPSDataType dataType) {
  return [mpsGraph placeholderWithShape:@[ @1 ] dataType:dataType name:nil];
}

MPSGraphTensor* mpsGraphScalarPlaceHolder(MPSGraph* mpsGraph, const Scalar& scalar) {
  return [mpsGraph placeholderWithShape:@[ @1 ] dataType:getMPSScalarType(scalar.type()) name:nil];
}

// this is meant to suppress the availability warning on castTensor
// we pass ScalarType instead of MPSDataType to handle MPSDataTypeBoolean's availability too
MPSGraphTensor* castMPSTensor(MPSGraph* mpsGraph, MPSGraphTensor* tensor, MPSDataType toType) {
  if ([tensor dataType] == toType) {
    return tensor;
  }
  return [mpsGraph castTensor:tensor toType:toType name:@"castTensor"];
}

MPSGraphTensor* castMPSTensor(MPSGraph* mpsGraph, MPSGraphTensor* tensor, ScalarType toType) {
  return [mpsGraph castTensor:tensor toType:getMPSScalarType(toType) name:@"castTensor"];
}

MPSGraphTensor* convertNHWCtoNCHW(MPSGraph* mpsGraph, MPSGraphTensor* tensor) {
  TORCH_INTERNAL_ASSERT(tensor.shape.count == 4, "Tensor must have 4 dimensions!");
  return [mpsGraph transposeTensor:[mpsGraph transposeTensor:tensor dimension:3 withDimension:2 name:nil]
                         dimension:2
                     withDimension:1
                              name:nil];
}

string get_mem_format_string(c10::MemoryFormat memory_format) {
  string mem_format_key;
  switch (memory_format) {
    case at::MemoryFormat::Contiguous:
      mem_format_key = "Contiguous";
      break;
    case at::MemoryFormat::ChannelsLast:
      mem_format_key = "ChannelsLast";
      break;
    default:
      TORCH_CHECK(false, "Invalid memory format", memory_format);
  }

  return mem_format_key;
}

MPSGraphCache* MPSGraphCache::_instance_cache = nullptr;

void MPSGraphCache::profileCachedGraph(const CacheEntry& cacheEntry) const {
  auto& profiler = getMPSProfiler();
  if (profiler.isOperationProfilingEnabled()) {
    std::string graphKey = cacheEntry.key_;
    // for interval-based signpost tracing, we begin the interval here to be able
    // to measure the time it takes to compile the graphs (if graph newly created),
    // and also the time potentially spent on gather/scatter of graph's input tensors
    profiler.beginProfileKernel(cacheEntry.cachedGraph_->graph(), graphKey, true);
  }
}

class MPSGraphCacheCallback : public IMpsAllocatorCallback {
 public:
  MPSGraphCacheCallback() : graph_cache(MPSGraphCache::getInstance()) {}

  void executeMPSAllocatorCallback(void* ptr, EventType event) override {}

 private:
  MPSGraphCache* graph_cache;
};

REGISTER_MPS_ALLOCATOR_CALLBACK("mps_graph_cache_callback", MPSGraphCacheCallback);

id<MTLBuffer> generateKernelDataOffsets(id<MTLComputeCommandEncoder> commandEncoder,
                                        const TensorIteratorBase& iter,
                                        bool use_64bit_index) {
  constexpr uint32_t nOffsets = 3;
  uint32_t numThreads = iter.numel();
  const uint32_t nDim = iter.ndim();
  const IntArrayRef& iterShape = iter.shape();
  std::vector<uint32_t> iterShapeData(iterShape.size());
  std::vector<std::array<uint32_t, nOffsets>> strides(nDim);
  TORCH_INTERNAL_ASSERT(iter.ntensors() >= nOffsets);
  TORCH_CHECK(use_64bit_index || iter.can_use_32bit_indexing(), "Can't be indexed using 32-bit iterator");

  for (const auto i : c10::irange(iterShape.size())) {
    iterShapeData[i] = static_cast<uint32_t>(iterShape[i]);
  }

  for (const auto i : c10::irange(nDim)) {
    for (const auto offset : c10::irange(nOffsets)) {
      strides[i][offset] = static_cast<uint32_t>(iter.strides(offset)[i]);
    }
  }

  id<MTLComputePipelineState> kernelDataOffsetsPSO = MPSDevice::getInstance()->metalIndexingPSO(
      use_64bit_index ? "kernel_index_offsets_64" : "kernel_index_offsets_32");
  const auto elementSize = use_64bit_index ? sizeof(simd_ulong3) : sizeof(simd_uint3);
  id<MTLBuffer> kernelDataOffsets = (id<MTLBuffer>)getIMPSAllocator()->allocate(numThreads * elementSize).get();

  [commandEncoder setComputePipelineState:kernelDataOffsetsPSO];
  [commandEncoder setBytes:strides.data() length:sizeof(uint32_t) * nDim * nOffsets atIndex:0];
  [commandEncoder setBuffer:kernelDataOffsets offset:0 atIndex:1];
  [commandEncoder setBytes:iterShapeData.data() length:sizeof(uint32_t) * iterShape.size() atIndex:2];
  [commandEncoder setBytes:&nDim length:sizeof(uint32_t) atIndex:3];

  mtl_dispatch1DJob(commandEncoder, kernelDataOffsetsPSO, numThreads);

  return kernelDataOffsets;
}

} // namespace at::native::mps
