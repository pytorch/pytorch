//  Copyright © 2022 Apple Inc.
#include <stdexcept>
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/TensorIterator.h>
#include <ATen/mps/MPSAllocatorInterface.h>
#include <ATen/mps/MPSProfiler.h>
#include <ATen/native/mps/MPSGraphSequoiaOps.h>
#include <ATen/native/mps/MPSGraphSonomaOps.h>
#include <ATen/native/mps/MPSGraphVenturaOps.h>
#include <ATen/native/mps/OperationUtils.h>
#include <fmt/format.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/scalar_tensor.h>
#endif

#include <mach-o/dyld.h>
#include <mach-o/getsect.h>

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
size_t compute_storage_numel_distance(const TensorBase& t) {
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
                               const TensorBase& input,
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
                                 const TensorBase& input,
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
    case ScalarType::ComplexHalf:
      return "half2";
    case ScalarType::ComplexFloat:
      return "float2";
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

NSArray<NSNumber*>* getTensorAxes(const TensorBase& t) {
  return getTensorAxes(t.dim());
}

static NSArray<NSNumber*>* getTensorAxes(const IntArrayRef& sizes) {
  return getTensorAxes(sizes.size());
}

NSArray<NSNumber*>* getTensorAxes(const IntArrayRef& sizes, OptionalIntArrayRef dim) {
  if (dim.has_value() && !dim.value().empty()) {
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

std::string getTensorsStringKey(const TensorList& tensors, bool short_dtype, bool exclude_shape) {
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
        if (exclude_shape) {
          str += "[-1]";
        } else {
          str +=
              std::string([[getMPSShape(tensor) valueForKey:@"description"] componentsJoinedByString:@","].UTF8String);
        }
      }
      str += "]";
    } else {
      str += "Undefined";
    }
  }
  return str;
}

Tensor getTensorView(const Tensor& t, MPSShape* shape) {
  std::vector<int64_t> res;
  res.reserve([shape count]);
  for (NSNumber* elem in shape) {
    res.push_back(elem.longLongValue);
  }
  IntArrayRef r = IntArrayRef(res);
  return t.view(res);
}

MPSShape* getMPSShape(const TensorBase& t, c10::MemoryFormat memory_format) {
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

void printTensorNDArray(const TensorBase& t) {
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

MPSNDArray* ndArrayFromTensor(const TensorBase& tensor, MPSShape* shape, MPSDataType mpsType) {
  id<MTLBuffer> buffer = getMTLBufferStorage(tensor);
  MPSGraphTensorData* tmpGraphTensorData = [[[MPSGraphTensorData alloc] initWithMTLBuffer:buffer
                                                                                    shape:shape
                                                                                 dataType:mpsType] autorelease];

  return [tmpGraphTensorData mpsndarray];
}

static std::vector<int64_t> getSortedStrides(const IntArrayRef& s) {
  std::vector<int64_t> idx(s.size());
  iota(idx.begin(), idx.end(), 0);
  sort(idx.begin(), idx.end(), [&s](size_t i1, size_t i2) { return s[i1] > s[i2]; });

  return idx;
}

static std::vector<int64_t> inversePermutation(const std::vector<int64_t>& permuteOrder) {
  auto size = permuteOrder.size();
  std::vector<int64_t> inversePerm(permuteOrder.size());

  for (int i = 0; i < size; i++) {
    inversePerm[permuteOrder[i]] = i;
  }
  return inversePerm;
}

static MPSNDArray* permuteNDArray(MPSNDArray* inArray, const std::vector<int64_t>& permuteOrder_) {
  auto permuteOrder = inversePermutation(permuteOrder_);
  NSUInteger srcRank = [inArray numberOfDimensions];
  if (srcRank != permuteOrder.size()) {
    TORCH_INTERNAL_ASSERT(false);
    return nil;
  }
  std::vector<NSUInteger> dimensionOrder(srcRank);
  std::iota(std::begin(dimensionOrder), std::end(dimensionOrder), 0);
  MPSNDArrayDescriptor* desc = [inArray descriptor];

  for (int64_t i = srcRank - 1; i >= 0; i--) {
    NSUInteger axis = permuteOrder[i];
    auto axisIter = std::find(dimensionOrder.begin(), dimensionOrder.end(), axis);
    NSUInteger axis1 = srcRank - i - 1;
    NSUInteger axis2 = dimensionOrder.end() - axisIter - 1;
    iter_swap(dimensionOrder.begin() + i, axisIter);
    if (axis1 != axis2) {
      [desc transposeDimension:axis1 withDimension:axis2];
    }
  }
  C10_CLANG_DIAGNOSTIC_PUSH()
#if C10_CLANG_HAS_WARNING("-Wnonnull")
  C10_CLANG_DIAGNOSTIC_IGNORE("-Wnonnull")
#endif
  MPSNDArray* result = [inArray arrayViewWithCommandBuffer:nil descriptor:desc aliasing:MPSAliasingStrategyShallAlias];
  C10_CLANG_DIAGNOSTIC_POP()

  TORCH_INTERNAL_ASSERT(result != nil);
  return result;
}

MPSNDArray* getMPSNDArray(const TensorBase& t, MPSShape* sizes, MPSShape* strides) {
  id<MTLBuffer> srcBuf = getMTLBufferStorage(t);

  MPSDataType mpsDataType = getMPSDataType(t.scalar_type());
  MPSNDArrayDescriptor* srcTensorDesc = [MPSNDArrayDescriptor descriptorWithDataType:mpsDataType shape:sizes];
  srcTensorDesc.preferPackedRows = YES;
  MPSNDArray* srcNDArray = [[[MPSNDArray alloc] initWithBuffer:srcBuf
                                                        offset:t.storage_offset() * t.element_size()
                                                    descriptor:srcTensorDesc] autorelease];
  if (strides != nil) {
    srcNDArray = [srcNDArray arrayViewWithShape:sizes strides:strides];
  }
  return srcNDArray;
}

MPSNDArray* getMPSNDArray(const TensorBase& t, const IntArrayRef& sizes, const IntArrayRef& strides) {
  return getMPSNDArray(t, getMPSShape(sizes.empty() ? t.sizes() : sizes), strides.empty() ? nil : getMPSShape(strides));
}

static MPSNDArray* getStridedMPSNDArray(const TensorBase& src, MPSNDArray* srcNDArray) {
  auto strides = src.strides();
  auto sizes = src.sizes();
  auto nStrides = strides.size();
  auto nonZeroStrides = src.strides();
  int64_t crtNonZeroStride = 1;
  bool hasZeroStrides = false;
  auto sortedStridesIndices = getSortedStrides(nonZeroStrides);

  NSMutableArray<NSNumber*>* sortedStridesShape = [NSMutableArray arrayWithCapacity:nStrides];
  NSMutableArray<NSNumber*>* sortedMPSShape = [NSMutableArray arrayWithCapacity:nStrides];
  for (const auto i : c10::irange(nStrides)) {
    sortedStridesShape[i] = [NSNumber numberWithInteger:nonZeroStrides[sortedStridesIndices[i]]];
    sortedMPSShape[i] = [NSNumber numberWithInteger:sizes[sortedStridesIndices[i]]];
  }
  MPSShape* originalSortedMPSShape = sortedMPSShape;
  MPSShape* originalSortedStridesShape = sortedStridesShape;
  bool hasNonZeroStrides = nStrides == 0 ? false : nonZeroStrides[sortedStridesIndices[nStrides - 1]] != 1;
  if (hasNonZeroStrides) {
    originalSortedMPSShape = [sortedMPSShape copy];
    originalSortedStridesShape = [sortedStridesShape copy];
    [sortedStridesShape addObject:[NSNumber numberWithInteger:1]];
    [sortedMPSShape addObject:[NSNumber numberWithInteger:1]];
  }
  if (nStrides == 0) {
    originalSortedMPSShape = getMPSShape(src);
    originalSortedStridesShape = getMPSShape(src.strides());
  }

  srcNDArray = [srcNDArray arrayViewWithShape:sortedMPSShape strides:sortedStridesShape];
  if (hasNonZeroStrides) {
    MPSNDArrayIdentity* identity =
        [[[MPSNDArrayIdentity alloc] initWithDevice:MPSDevice::getInstance()->device()] autorelease];
    srcNDArray = [identity reshapeWithCommandBuffer:nil
                                        sourceArray:srcNDArray
                                              shape:originalSortedMPSShape
                                   destinationArray:nil];
  }
  TORCH_INTERNAL_ASSERT(srcNDArray);

  srcNDArray = permuteNDArray(srcNDArray, sortedStridesIndices);
  TORCH_INTERNAL_ASSERT(srcNDArray);

  return srcNDArray;
}

Placeholder::Placeholder(MPSGraphTensor* mpsGraphTensor, MPSNDArray* mpsNDArray) {
  _placeholder = mpsGraphTensor;
  _value = [[[MPSGraphTensorData alloc] initWithMPSNDArray:mpsNDArray] autorelease];
}

Placeholder::Placeholder(MPSGraphTensor* mpsGraphTensor,
                         const Tensor& src,
                         MPSShape* mpsShape_,
                         bool gatherTensorData,
                         MPSDataType dataType,
                         bool useMPSStridedAPI)
    : _tensor(src) {
  TORCH_CHECK(src.is_mps(), "Placeholder storage has not been allocated on MPS device!");
  // extract the pointer to MTLBuffer from the Tensor's storage
  id<MTLBuffer> srcBuf = getMTLBufferStorage(src);

  static const bool is_macOS_15_0_or_newer = is_macos_13_or_newer(MacOSVersion::MACOS_VER_15_0_PLUS);
  // Use gather kernel to solve strides for macOS < 15.0
  // Starting with macOS 15.0, MPS supports native strides direclty in the kernels
  if (!is_macOS_15_0_or_newer || !useMPSStridedAPI) {
    if ((!src.is_contiguous() || src.storage_offset()) && gatherTensorData) {
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
  }

  // tensor.numel() could be zero, but tensor is valid as long as the buffer size is non-zero.
  // if buffer size is zero in here, it's not a user error. It could be a missing check for
  // tensor.numel() == 0 in our internal implementations of ops.
  TORCH_INTERNAL_ASSERT([srcBuf length] > 0, "Placeholder tensor is empty!");
  if (dataType == MPSDataTypeInvalid) {
    const auto scalar_type = _tensor.scalar_type();
    dataType = _tensor.dim() == 0 ? getMPSScalarType(scalar_type) : getMPSDataType(scalar_type);
  }

  // Tensor is contiguous and has no storage offset.
  // Wrap it directly inside MPSGraphTensorData
  if ((_tensor.is_contiguous() && !_tensor.storage_offset()) || !useMPSStridedAPI || !is_macOS_15_0_or_newer) {
    _value = [[[MPSGraphTensorData alloc] initWithMTLBuffer:srcBuf
                                                      shape:mpsShape_ ? mpsShape_ : getMPSShape(_tensor)
                                                   dataType:dataType] autorelease];
  } else {
    IntArrayRef view_shape;
    if (mpsShape_) {
      _tensor = getTensorView(src, mpsShape_);
    }

    MPSShape* mpsShape = getMPSShape(_tensor);
    MPSShape* mpsStrides = getMPSShape(_tensor.strides());

    auto storage_numel = src.storage().nbytes() / src.element_size();
    MPSNDArrayDescriptor* srcTensorDesc = [MPSNDArrayDescriptor descriptorWithDataType:dataType
                                                                                 shape:@[ @(storage_numel) ]];
    srcTensorDesc.preferPackedRows = YES;
    MPSNDArray* srcNDArray = [[[MPSNDArray alloc] initWithBuffer:srcBuf
                                                          offset:src.storage_offset() * src.element_size()
                                                      descriptor:srcTensorDesc] autorelease];
    TORCH_INTERNAL_ASSERT(srcNDArray);
    if (src.dim() != 0) {
      srcNDArray = getStridedMPSNDArray(_tensor, srcNDArray);
    } else {
      bool needsReshape = false;
      NSMutableArray* mpsExpandedShape = nil;
      NSMutableArray* mpsExpandedStrides = nil;

      if (src.dim() > 0 && src.stride(-1) != 1) {
        needsReshape = true;
        mpsExpandedShape = [NSMutableArray arrayWithArray:mpsShape];
        mpsExpandedStrides = [NSMutableArray arrayWithArray:mpsStrides];
        [mpsExpandedShape addObject:@1];
        [mpsExpandedStrides addObject:@1];
      }
      srcNDArray = [srcNDArray arrayViewWithShape:needsReshape ? mpsExpandedShape : getMPSShape(src)
                                          strides:needsReshape ? mpsExpandedStrides : getMPSShape(src.strides())];
      TORCH_INTERNAL_ASSERT(srcNDArray);

      if (needsReshape) {
        MPSNDArrayIdentity* identity =
            [[[MPSNDArrayIdentity alloc] initWithDevice:MPSDevice::getInstance()->device()] autorelease];
        srcNDArray = [identity reshapeWithCommandBuffer:nil sourceArray:srcNDArray shape:mpsShape destinationArray:nil];
      }
      TORCH_INTERNAL_ASSERT(srcNDArray);
    }
    _value = [[[MPSGraphTensorData alloc] initWithMPSNDArray:srcNDArray] autorelease];
  }

  TORCH_INTERNAL_ASSERT(_value);
  _placeholder = mpsGraphTensor;
}

MPSGraphTensorData* getMPSGraphTensorData(MPSGraph* mpsGraph, MPSStream* mpsStream, const TensorBase& tensor) {
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
      return {.value.h = scalar.to<Half>(), .size = sizeof(short), .type = type};
    case ScalarType::BFloat16:
      return {.value.bf16 = scalar.to<BFloat16>(), .size = sizeof(short), .type = type};
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
      return {.value.ch = scalar.to<c10::complex<Half>>(), .size = sizeof(int32_t), .type = type};
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
    tensor = at::scalar_tensor(scalar, at::device(device).dtype(kFloat));
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

MPSGraphTensor* mpsGraphRankedPlaceHolder(MPSGraph* mpsGraph, const TensorBase& tensor) {
  return [mpsGraph placeholderWithShape:getMPSShape(tensor) dataType:getMPSScalarType(tensor) name:nil];
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

id<MTLLibrary> MetalShaderLibrary::getLibrary() {
  if (C10_UNLIKELY(!library)) {
    TORCH_INTERNAL_ASSERT(nparams == 0);
    library = compileLibrary(shaderSource);
  }
  return library;
}

id<MTLLibrary> MetalShaderLibrary::getLibrary(const std::initializer_list<std::string>& params) {
  TORCH_INTERNAL_ASSERT(nparams == params.size());
  std::string key = "";
  for (auto p : params) {
    key += ":" + p;
  }
  auto lib = libMap[key];
  if (lib) {
    return lib;
  }
  auto it = params.begin();
  switch (nparams) {
    case 1:
      lib = compileLibrary(fmt::format(shaderSource, *it));
      break;
    case 2: {
      auto& first = *it++;
      auto& second = *it;
      lib = compileLibrary(fmt::format(shaderSource, first, second));
      break;
    }
    case 3: {
      auto& first = *it++;
      auto& second = *it++;
      auto& third = *it;
      lib = compileLibrary(fmt::format(shaderSource, first, second, third));
      break;
    }
    default:
      TORCH_INTERNAL_ASSERT(false, "Unsupported number of paramaters ", nparams);
  }
  return libMap[key] = lib;
}

id<MTLLibrary> MetalShaderLibrary::compileLibrary(const std::string& src) {
  static auto fast_math = []() {
    auto val = std::getenv("PYTORCH_MPS_FAST_MATH");
    return val && std::stoi(val) != 0;
  }();
  NSError* error = nil;
  MTLCompileOptions* options = compile_options;
  if (!options) {
    options = [[MTLCompileOptions new] autorelease];
    // Need 3.0 for atomic oprations, 3.1 introduces bfloat support
    [options setLanguageVersion:is_macos_13_or_newer(MacOSVersion::MACOS_VER_14_0_PLUS) ? MTLLanguageVersion3_1
                                                                                        : MTLLanguageVersion3_0];
    if (is_macos_13_or_newer(MacOSVersion::MACOS_VER_15_0_PLUS)) {
      options.mathMode = fast_math ? MTLMathModeFast : MTLMathModeSafe;
      options.mathFloatingPointFunctions =
          fast_math ? MTLMathFloatingPointFunctionsFast : MTLMathFloatingPointFunctionsPrecise;
    } else {
      C10_DIAGNOSTIC_PUSH_AND_IGNORED_IF_DEFINED("-Wdeprecated-declarations")
      [options setFastMathEnabled:fast_math ? YES : NO];
      C10_DIAGNOSTIC_POP()
    }
  }

  const auto str = [NSString stringWithCString:src.c_str() encoding:NSASCIIStringEncoding];
  auto device = MPSDevice::getInstance()->device();
  library = [device newLibraryWithSource:str options:options error:&error];
  TORCH_CHECK(library, "Failed to create metal library, error: ", [[error description] UTF8String]);
  return library;
}

std::pair<id<MTLComputePipelineState>, id<MTLFunction>> MetalShaderLibrary::getLibraryPipelineState(
    id<MTLLibrary> lib,
    const std::string& fname) {
  const auto key = fmt::format("{}:{}", reinterpret_cast<void*>(lib), fname);
  auto found_cpl = cplMap.find(key);
  if (found_cpl != cplMap.end()) {
    return found_cpl->second;
  }

  NSError* error = nil;
  id<MTLFunction> func = [lib newFunctionWithName:[NSString stringWithUTF8String:fname.c_str()]];
  TORCH_CHECK(func, "Failed to create function state object for: ", fname);
  auto cpl = [[lib device] newComputePipelineStateWithFunction:func error:&error];
  TORCH_CHECK(cpl, "Failed to created pipeline state object, error: ", [[error description] UTF8String]);

  cplMap[key] = std::make_pair(cpl, func);
  return cplMap[key];
}

class BundledShaderLibary: public MetalShaderLibrary {
public:
  BundledShaderLibary(): MetalShaderLibrary("") {}
protected:
  id<MTLLibrary> getLibrary() override {
    if (C10_UNLIKELY(!library)) {
      auto device = MPSDevice::getInstance()->device();
      NSError *error = nil;
      auto section_name = is_macos_13_or_newer(MacOSVersion::MACOS_VER_14_0_PLUS) ? "metal_bfloat" : "metal_basic";
      library = [device newLibraryWithData:getSectionData(section_name) error:&error];
      TORCH_CHECK(library, "Failed to create metal library, error: ", [[error description] UTF8String]);
    }
    return library;
  }

  id<MTLLibrary> getLibrary(const std::initializer_list<std::string>& params) override {
   throw std::runtime_error("Should never be called");
  }
private:
  static dispatch_data_t getSectionData(const std::string& name) {
      uint32_t idx = 0;
      for(const auto cnt: c10::irange(_dyld_image_count())) {
          if (strstr(_dyld_get_image_name(cnt), "/libtorch_cpu.dylib")) {
            idx = cnt; break;
          }
      }
      const auto* mach_header = reinterpret_cast<const struct mach_header_64*>(_dyld_get_image_header(idx));
      unsigned long mtl_lib_size = 0;
      const auto* mtl_lib_data = getsectiondata(mach_header, "__TEXT", name.c_str(), &mtl_lib_size);
      if (mtl_lib_data == nullptr) {
        throw std::runtime_error("Can't find metal library section " + name);
      }
      return dispatch_data_create(mtl_lib_data, mtl_lib_size, dispatch_get_main_queue(), ^() {});
    }
};

MetalShaderLibrary& MetalShaderLibrary::getBundledLibrary() {
    static BundledShaderLibary l;
    return l;
}

} // namespace at::native::mps
