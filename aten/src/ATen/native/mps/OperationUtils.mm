//  Copyright © 2022 Apple Inc.
#include <ATen/core/TensorBase.h>
#include <ATen/native/mps/MetalShaderLibrary.h>
#include <c10/metal/common.h>
#include <functional>
#include <stdexcept>
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/TensorIterator.h>
#include <ATen/mps/MPSAllocatorInterface.h>
#include <ATen/mps/MPSProfiler.h>
#include <ATen/mps/MPSStream.h>
#include <ATen/native/mps/MPSGraphSequoiaOps.h>
#include <ATen/native/mps/OperationUtils.h>
#include <fmt/format.h>
#include <fmt/ranges.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/scalar_tensor.h>
#endif

#include <c10/util/env.h>
#include <mach-o/dyld.h>
#include <mach-o/getsect.h>

@implementation MPSGraph (PyTorchFixups)
- (MPSGraphTensor*)minimumWithNaNPropagationAndIntFallbackWithPrimaryTensor:(MPSGraphTensor*)primaryTensor
                                                            secondaryTensor:(MPSGraphTensor*)secondaryTensor
                                                                       name:(NSString*)name {
  // As of MacOS-15.1 m..imumWithNanPropagation is only defined for floating types and calling it with integral
  // arguments results in
  //  /AppleInternal/Library/BuildRoots/c7c74b64-74b4-11ef-aeda-9635a580fe0d/Library/Caches/com.apple.xbs/Sources/MetalPerformanceShaders/MPSCore/Utility/MPSKernelDAG.mm:805:
  //  failed assertion `Error getting visible function: (null) Function isNaN_u8_i8 was not found in the library'
  if (([primaryTensor dataType] & MPSDataTypeFloatBit) == 0) {
    return [self minimumWithPrimaryTensor:primaryTensor secondaryTensor:secondaryTensor name:name];
  }
  return [self minimumWithNaNPropagationWithPrimaryTensor:primaryTensor secondaryTensor:secondaryTensor name:name];
}

- (MPSGraphTensor*)maximumWithNaNPropagationAndIntFallbackWithPrimaryTensor:(MPSGraphTensor*)primaryTensor
                                                            secondaryTensor:(MPSGraphTensor*)secondaryTensor
                                                                       name:(NSString*)name {
  // As of MacOS-15.1 m..imumWithNanPropagation is only defined for floating types and calling it with integral
  // arguments results in
  //  /AppleInternal/Library/BuildRoots/c7c74b64-74b4-11ef-aeda-9635a580fe0d/Library/Caches/com.apple.xbs/Sources/MetalPerformanceShaders/MPSCore/Utility/MPSKernelDAG.mm:805:
  //  failed assertion `Error getting visible function: (null) Function isNaN_u8_i8 was not found in the library'
  if (([primaryTensor dataType] & MPSDataTypeFloatBit) == 0) {
    return [self maximumWithPrimaryTensor:primaryTensor secondaryTensor:secondaryTensor name:name];
  }
  return [self maximumWithNaNPropagationWithPrimaryTensor:primaryTensor secondaryTensor:secondaryTensor name:name];
}
@end

namespace at::native::mps {
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

MPSDataType getMPSDataType(ScalarType scalar_type) {
  switch (scalar_type) {
    case ScalarType::Float:
      return MPSDataTypeFloat32;
    case ScalarType::Half:
      return MPSDataTypeFloat16;
    case ScalarType::BFloat16:
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
      return MPSDataTypeComplexFloat16;
    case ScalarType::ComplexFloat:
      return MPSDataTypeComplexFloat32;
    // Unsigned types
    case ScalarType::UInt64:
      return MPSDataTypeUInt64;
    case ScalarType::UInt32:
      return MPSDataTypeUInt32;
    case ScalarType::UInt16:
      return MPSDataTypeUInt16;
    default:
      TORCH_CHECK_TYPE(
          false, "Trying to convert ", scalar_type, " to the MPS backend but it does not have support for that dtype.")
  }
}

// #issue 104398441 sortWithTensor and argsortWithTensor has support of
// Int32, Half and Float32 types. These utilities are to help cast to these
// types.
MPSGraphTensor* castToIHFTypes(MPSGraph* mpsGraph, MPSGraphTensor* inputTensor, const TensorBase& input) {
  MPSDataType dataType = getMPSDataType(input.scalar_type());
  bool condition = (dataType != MPSDataTypeInt32) && (dataType != MPSDataTypeFloat32) &&
      (dataType != MPSDataTypeFloat16) && (dataType != MPSDataTypeInt64);
  if (condition) {
    dataType = (dataType & MPSDataTypeFloatBit) ? MPSDataTypeFloat32 : MPSDataTypeInt32;
    return [mpsGraph castTensor:inputTensor toType:dataType name:@"castInputTensor"];
  }
  return inputTensor;
}

// #issue 104398441 sortWithTensor and argsortWithTensor has support of
// Int32, Half and Float32 types. These utilities are to help cast from these
// types.
MPSGraphTensor* castFromIHFTypes(MPSGraph* mpsGraph, MPSGraphTensor* inputTensor, const TensorBase& input) {
  MPSDataType dataType = getMPSDataType(input.scalar_type());
  bool condition = (dataType != MPSDataTypeInt32) && (dataType != MPSDataTypeFloat32) &&
      (dataType != MPSDataTypeFloat16) && (dataType != MPSDataTypeInt64);
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
      return MPSDataTypeComplexFloat16;
    // This is an intentional fallthrough supporting ComplexDouble for Scalar
    // types as they are casted to Complex64 currently.
    case ScalarType::ComplexDouble:
    case ScalarType::ComplexFloat:
      return MPSDataTypeComplexFloat32;
    // Unsigned types
    case ScalarType::UInt64:
      return MPSDataTypeUInt64;
    case ScalarType::UInt32:
      return MPSDataTypeUInt32;
    case ScalarType::UInt16:
      return MPSDataTypeUInt16;
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
    // Unsigned types
    case ScalarType::UInt64:
      return short_name ? "u64" : "UInt64";
    case ScalarType::UInt32:
      return short_name ? "u32" : "UInt32";
    case ScalarType::UInt16:
      return short_name ? "u16" : "UInt16";
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
    // Unsigned types
    case ScalarType::UInt64:
      return "ulong";
    case ScalarType::UInt32:
      return "uint";
    case ScalarType::UInt16:
      return "ushort";
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
  return fmt::to_string(fmt::join(s, ","));
}

std::string to_hex_key(float f) {
  return fmt::format("{:a}", f);
}

std::string getTensorsStringKey(const TensorList& tensors, bool short_dtype, bool exclude_shape) {
  fmt::basic_memory_buffer<char, 100> buffer;
  auto buf_iterator = std::back_inserter(buffer);

  for (const Tensor& tensor : tensors) {
    fmt::format_to(buf_iterator, ":");
    if (tensor.defined()) {
      fmt::format_to(buf_iterator, "{}[", getMPSTypeString(tensor.scalar_type(), short_dtype));
      if (tensor.dim() == 0) {
        fmt::format_to(buf_iterator, "Scalar");
      } else {
        if (exclude_shape) {
          fmt::format_to(buf_iterator, "-1");
        } else {
          fmt::format_to(buf_iterator, "{}", getArrayRefString(tensor.sizes()));
        }
      }
      fmt::format_to(buf_iterator, "]");
      if (tensor.is_conj()) {
        fmt::format_to(buf_iterator, "_conj");
      }
    } else {
      fmt::format_to(buf_iterator, "Undefined");
    }
  }
  return fmt::to_string(buffer);
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

// Should be called before initWithBuffer to prevent hard crashes with
// '[MPSNDArray initWithDevice:descriptor:isTextureBacked:] Error: NDArray dimension length > INT_MAX'
static void check_mps_shape(MPSShape* shape) {
  for (NSNumber* elem in shape) {
    const auto val = [elem longValue];
    TORCH_CHECK(val <= std::numeric_limits<int32_t>::max(), "MPSGaph does not support tensor dims larger than INT_MAX");
  }
}

bool isTooLargeForMPSGraph(const Tensor& tensor, bool useMPSStridedAPI) {
  static const bool is_macOS_15_0_or_newer = is_macos_13_or_newer(MacOSVersion::MACOS_VER_15_0_PLUS);
  if ((!tensor.is_contiguous() || tensor.storage_offset()) && useMPSStridedAPI && is_macOS_15_0_or_newer) {
    auto storage_numel = tensor.storage().nbytes() / tensor.element_size() - tensor.storage_offset();
    if (storage_numel > std::numeric_limits<int32_t>::max()) {
      return true;
    }
  }
  for (auto size : tensor.sizes()) {
    if (size > std::numeric_limits<int32_t>::max()) {
      return true;
    }
  }
  return false;
}

MPSNDArray* getMPSNDArray(const TensorBase& t, MPSShape* sizes, MPSShape* strides) {
  id<MTLBuffer> srcBuf = getMTLBufferStorage(t);

  MPSDataType mpsDataType = getMPSDataType(t.scalar_type());
  MPSNDArrayDescriptor* srcTensorDesc = [MPSNDArrayDescriptor descriptorWithDataType:mpsDataType shape:sizes];
  srcTensorDesc.preferPackedRows = YES;
  check_mps_shape(sizes);
  MPSNDArray* srcNDArray = [[[MPSNDArray alloc] initWithBuffer:srcBuf
                                                        offset:t.storage_offset() * t.element_size()
                                                    descriptor:srcTensorDesc] autorelease];
  if (strides != nil) {
    srcNDArray = getStridedMPSNDArray(t, srcNDArray);
  }
  return srcNDArray;
}

MPSNDArray* getMPSNDArray(const TensorBase& t, const IntArrayRef& sizes, const IntArrayRef& strides) {
  return getMPSNDArray(t, getMPSShape(sizes.empty() ? t.sizes() : sizes), strides.empty() ? nil : getMPSShape(strides));
}

MPSNDArray* getStridedMPSNDArray(const TensorBase& src, MPSNDArray* srcNDArray) {
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
  // Starting with macOS 15.0, MPS supports native strides directly in the kernels
  if (!is_macOS_15_0_or_newer || !useMPSStridedAPI) {
    if ((!src.is_contiguous() || src.storage_offset()) && gatherTensorData) {
      Tensor emptyShell = Tensor();
      // use "_tensor" from Placeholder to retain view's output during its usage in other ops
      // And preserve conjugated property here
      if (!src.is_conj()) {
        _tensor = gatherViewTensor(src, emptyShell);
      } else {
        _tensor = gatherViewTensor(src.conj(), emptyShell).conj();
      }
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
    auto shape = mpsShape_ ? mpsShape_ : getMPSShape(_tensor);
    check_mps_shape(shape);
    _value = [[[MPSGraphTensorData alloc] initWithMTLBuffer:srcBuf shape:shape dataType:dataType] autorelease];
  } else {
    IntArrayRef view_shape;
    if (mpsShape_) {
      _tensor = getTensorView(src, mpsShape_);
    }

    MPSShape* mpsShape = getMPSShape(_tensor);
    MPSShape* mpsStrides = getMPSShape(_tensor.strides());
    check_mps_shape(mpsShape);

    auto storage_numel = src.storage().nbytes() / src.element_size() - src.storage_offset();
    TORCH_CHECK(storage_numel <= std::numeric_limits<int32_t>::max(),
                "MPSGaph does not support tensor dims larger than INT_MAX");
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
      return {.size = sizeof(float), .type = type, .value.f = scalar.to<float>()};
    case ScalarType::Half:
      return {.size = sizeof(short), .type = type, .value.h = scalar.to<Half>()};
    case ScalarType::BFloat16:
      return {.size = sizeof(short), .type = type, .value.bf16 = scalar.to<BFloat16>()};
    case ScalarType::Long:
      return {.size = sizeof(int64_t), .type = type, .value.i = scalar.to<int64_t>()};
    case ScalarType::Int:
      return {.size = sizeof(int32_t), .type = type, .value.i = scalar.to<int32_t>()};
    case ScalarType::Short:
      return {.size = sizeof(int16_t), .type = type, .value.i = scalar.to<int16_t>()};
    case ScalarType::Char:
      return {.size = sizeof(int8_t), .type = type, .value.i = scalar.to<int8_t>()};
    case ScalarType::Byte:
      return {.size = sizeof(uint8_t), .type = type, .value.i = scalar.to<uint8_t>()};
    case ScalarType::Bool:
      return {.size = sizeof(bool), .type = type, .value.b = scalar.to<bool>()};
    case ScalarType::ComplexHalf:
      return {.size = sizeof(int32_t), .type = type, .value.ch = scalar.to<c10::complex<Half>>()};
    case ScalarType::ComplexFloat:
    case ScalarType::ComplexDouble:
      return {.size = sizeof(int64_t), .type = type, .value.cf = scalar.to<c10::complex<float>>()};
    // Unsigned types
    case ScalarType::UInt32:
      return {.size = sizeof(uint32_t), .type = type, .value.i = scalar.to<uint32_t>()};
    case ScalarType::UInt16:
      return {.size = sizeof(uint16_t), .type = type, .value.i = scalar.to<uint16_t>()};
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
    tensor = at::scalar_tensor(scalar, at::device(device).dtype(at::kComplexFloat));
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

std::string get_mem_format_string(c10::MemoryFormat memory_format) {
  std::string mem_format_key;
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

MPSKernelCache* MPSKernelCache::_instance_cache = nullptr;

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

// MetalShaderLibrary implementation
MetalShaderLibrary::~MetalShaderLibrary() {
  for (const auto& it : cplMap) {
    auto [cpl, func] = it.second;
    [cpl release];
    [func release];
  }
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
      lib = compileLibrary(fmt::format(fmt::runtime(shaderSource), *it));
      break;
    case 2: {
      auto& first = *it++;
      auto& second = *it;
      lib = compileLibrary(fmt::format(fmt::runtime(shaderSource), first, second));
      break;
    }
    case 3: {
      auto& first = *it++;
      auto& second = *it++;
      auto& third = *it;
      lib = compileLibrary(fmt::format(fmt::runtime(shaderSource), first, second, third));
      break;
    }
    default:
      TORCH_INTERNAL_ASSERT(false, "Unsupported number of parameters ", nparams);
  }
  return libMap[key] = lib;
}

id<MTLLibrary> MetalShaderLibrary::compileLibrary(const std::string& src) {
  static auto fast_math = []() {
    auto const val = c10::utils::get_env("PYTORCH_MPS_FAST_MATH");
    return val.has_value() && val != "0";
  }();
  NSError* error = nil;
  MTLCompileOptions* options = compile_options;
  if (!options) {
    options = [[MTLCompileOptions new] autorelease];
    if (is_macos_13_or_newer(MacOSVersion::MACOS_VER_26_0_PLUS)) {
      // Metal-4.0 allows tensor template arguments
      [options setLanguageVersion:MTLLanguageVersion4_0];
    } else if (is_macos_13_or_newer(MacOSVersion::MACOS_VER_15_0_PLUS)) {
      // Metal-3.2 allows lambdas in shader code
      [options setLanguageVersion:MTLLanguageVersion3_2];
    } else {
      [options setLanguageVersion:MTLLanguageVersion3_1];
    }
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
  if (library == nil) {
    if ([error domain] == MTLLibraryErrorDomain && [error code] == MTLLibraryErrorCompileFailure) {
      throw c10::SyntaxError([[error localizedDescription] UTF8String]);
    }
    TORCH_CHECK(false, "Failed to create metal library, error: ", [[error description] UTF8String]);
  }
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

std::vector<std::string> MetalShaderLibrary::getFunctionNames() {
  if (C10_UNLIKELY(!library && nparams > 0)) {
    throw std::runtime_error("Library must be initialized first");
  }
  std::vector<std::string> rc;
  @autoreleasepool {
    NSArray<NSString*>* names = [getLibrary() functionNames];
    rc.reserve([names count]);
    for (auto idx : c10::irange([names count])) {
      rc.emplace_back([[names objectAtIndex:idx] UTF8String]);
    }
  }
  return rc;
}

std::shared_ptr<MetalKernelFunction> MetalShaderLibrary::getKernelFunction(const std::string& name) {
  auto [cpl, func] = getLibraryPipelineState(getLibrary(), name);
  return std::make_shared<MetalKernelFunction>(cpl, func);
}

MetalKernelFunction* MetalShaderLibrary::getCachedKernelFunctionPtr(const std::string& name) {
  // Check if kernel is already cached
  auto it = kernelCache.find(name);
  if (it != kernelCache.end()) {
    return it->second.get();
  }

  // Create new kernel function and cache it
  auto [cpl, func] = getLibraryPipelineState(getLibrary(), name);
  auto kernel = std::make_unique<MetalKernelFunction>(cpl, func);
  MetalKernelFunction* raw_ptr = kernel.get();
  kernelCache[name] = std::move(kernel);

  return raw_ptr;
}

class BundledShaderLibary : public MetalShaderLibrary {
 public:
  BundledShaderLibary() : MetalShaderLibrary("") {}

 protected:
  id<MTLLibrary> getLibrary() override {
    if (C10_UNLIKELY(!library)) {
      auto device = MPSDevice::getInstance()->device();
      NSError* error = nil;
      library = [device newLibraryWithData:getSectionData("metal_basic") error:&error];
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
    for (const auto cnt : c10::irange(_dyld_image_count())) {
      if (strstr(_dyld_get_image_name(cnt), "/libtorch_cpu.dylib")) {
        idx = cnt;
        break;
      }
    }
    const auto* mach_header = reinterpret_cast<const struct mach_header_64*>(_dyld_get_image_header(idx));
    unsigned long mtl_lib_size = 0;
    const auto* mtl_lib_data = getsectiondata(mach_header, "__TEXT", name.c_str(), &mtl_lib_size);
    if (mtl_lib_data == nullptr) {
      throw std::runtime_error("Can't find metal library section " + name);
    }
    return dispatch_data_create(mtl_lib_data,
                                mtl_lib_size,
                                dispatch_get_main_queue(),
                                ^(){
                                });
  }
};

void MetalShaderLibrary::exec_unary_kernel(TensorIteratorBase& iter,
                                           const std::string& name,
                                           std::optional<c10::Scalar> alpha,
                                           std::optional<c10::ScalarType> scalar_arg_type) {
  // Decompose 64-bit tensor into 32-bit ones
  if (!iter.can_use_32bit_indexing()) {
    for (auto&& sub_iter : iter.with_32bit_indexing()) {
      exec_unary_kernel(sub_iter, name, alpha, scalar_arg_type);
    }
    return;
  }

  auto inputTensor = iter.input(0);
  auto outputTensor = iter.output(0);
  uint32_t length = iter.numel();
  if (length == 0) {
    return;
  }
  using namespace mps;
  const auto alpha_type = scalar_arg_type.has_value() ? scalar_arg_type.value() : iter.common_dtype();
  auto kernel_name = fmt::format("{}_{}_{}_{}{}",
                                 name,
                                 iter.is_contiguous() ? "dense" : "strided",
                                 scalarToMetalTypeString(outputTensor),
                                 scalarToMetalTypeString(inputTensor),
                                 alpha.has_value() ? fmt::format("_{}", scalarToMetalTypeString(alpha_type)) : "");
  @autoreleasepool {
    auto cplState = getPipelineStateForFunc(kernel_name);

    MPSStream* mpsStream = getCurrentMPSStream();
    dispatch_sync(mpsStream->queue(), ^() {
      auto computeEncoder = mpsStream->commandEncoder();

      getMPSProfiler().beginProfileKernel(cplState, name, {inputTensor});

      [computeEncoder setComputePipelineState:cplState];
      bind_iter_tensors(computeEncoder, iter);
      if (!iter.is_contiguous()) {
        mtl_setArgs<2>(computeEncoder,
                       outputTensor.sizes(),
                       inputTensor.strides(),
                       outputTensor.strides(),
                       inputTensor.ndimension());
      }
      if (alpha) {
        mtl_setBytes(computeEncoder, getMPSScalar(*alpha, alpha_type), iter.is_contiguous() ? 2 : 6);
      }
      mtl_dispatch1DJob(computeEncoder, cplState, length);

      getMPSProfiler().endProfileKernel(cplState);
    });
  }
}

void MetalShaderLibrary::exec_binary_kernel(TensorIteratorBase& iter,
                                            const std::string& name,
                                            std::optional<c10::Scalar> alpha,
                                            std::optional<c10::ScalarType> scalar_arg_type) {
  // TODO: Figure a better place to downcast double scalars (probably in tensor iterator itself?)
  // Right now running something like 1.0-torch.rand(5, device='mps') will create iterator with
  // double as common dtype (because Python floating point are always 64-bit values)
  TORCH_CHECK(iter.output().scalar_type() != at::kDouble, "float64 is not supported on MPS");

  // Skip for empty iterators
  if (iter.numel() == 0) {
    return;
  }

  // Decompose 64-bit tensor into 32-bit ones
  if (!iter.can_use_32bit_indexing()) {
    for (auto&& sub_iter : iter.with_32bit_indexing()) {
      exec_binary_kernel(sub_iter, name, alpha, scalar_arg_type);
    }
    return;
  }

  auto convert_double_scalar = [](Tensor& t) {
    if (t.dim() != 0) {
      return;
    }
    if (t.scalar_type() == kDouble) {
      t = t.to(kFloat);
    } else if (t.scalar_type() == kComplexDouble) {
      t = t.to(kComplexFloat);
    }
  };

  Tensor input = iter.input(0);
  Tensor other = iter.input(1);
  Tensor out = iter.output();

  convert_double_scalar(input);
  convert_double_scalar(other);

  MPSStream* mpsStream = getCurrentMPSStream();
  const auto cast_needed = input.scalar_type() != other.scalar_type();
  const auto suffix = iter.is_contiguous() ? "dense" : "strided";
  bool use_broadcast_kernel = false;
  bool use_scalar_kernel = false;
  bool broadcast_on_lhs = false;
  bool scalar_on_lhs = false;
  int64_t broadcast_numel = 0;

  const bool input_is_full = input.numel() == out.numel();
  const bool other_is_full = other.numel() == out.numel();
  const bool input_is_bc = is_dense_broadcastable(input, out) && !input_is_full;
  const bool other_is_bc = is_dense_broadcastable(other, out) && !other_is_full;

  if (input_is_bc && other_is_full && other.is_contiguous() && out.is_contiguous()) {
    broadcast_numel = input.numel();
    if (broadcast_numel == 1) {
      use_scalar_kernel = true;
      scalar_on_lhs = true;
    } else {
      use_broadcast_kernel = true;
      broadcast_on_lhs = true;
    }
  } else if (other_is_bc && input_is_full && input.is_contiguous() && out.is_contiguous()) {
    broadcast_numel = other.numel();
    if (broadcast_numel == 1) {
      use_scalar_kernel = true;
      scalar_on_lhs = false;
    } else {
      use_broadcast_kernel = true;
      broadcast_on_lhs = false;
    }
  }

  const auto alpha_type = scalar_arg_type.has_value() ? scalar_arg_type.value() : iter.common_dtype();
  const auto alpha_suffix = alpha.has_value() ? fmt::format("_{}", scalarToMetalTypeString(alpha_type)) : "";

  std::string kernel_name;
  if (use_scalar_kernel) {
    const auto& tensor_operand = scalar_on_lhs ? other : input;
    const auto lhs_suffix = scalar_on_lhs ? "_lhs" : "";
    if (cast_needed) {
      kernel_name =
          fmt::format("{}_dense_scalar{}_cast_{}{}", name, lhs_suffix, scalarToMetalTypeString(out), alpha_suffix);
    } else {
      kernel_name = fmt::format("{}_dense_scalar{}_{}_{}{}",
                                name,
                                lhs_suffix,
                                scalarToMetalTypeString(out),
                                scalarToMetalTypeString(tensor_operand),
                                alpha_suffix);
    }
  } else if (use_broadcast_kernel) {
    const auto& tensor_operand = broadcast_on_lhs ? other : input;
    if (cast_needed) {
      kernel_name = fmt::format("{}_dense_broadcast{}_cast_{}{}",
                                name,
                                broadcast_on_lhs ? "_rhs" : "",
                                scalarToMetalTypeString(out),
                                alpha_suffix);
    } else {
      kernel_name = fmt::format("{}_dense_broadcast{}_{}_{}{}",
                                name,
                                broadcast_on_lhs ? "_rhs" : "",
                                scalarToMetalTypeString(out),
                                scalarToMetalTypeString(tensor_operand),
                                alpha_suffix);
    }
  } else {
    // TODO: Implicitly pass both input and output types to non-cast kernels
    kernel_name = cast_needed ? fmt::format("{}_{}_cast_{}{}", name, suffix, scalarToMetalTypeString(out), alpha_suffix)
                              : fmt::format("{}_{}_{}_{}{}",
                                            name,
                                            suffix,
                                            scalarToMetalTypeString(out),
                                            scalarToMetalTypeString(input),
                                            alpha_suffix);
  }
  dispatch_sync_with_rethrow(mpsStream->queue(), ^() {
    @autoreleasepool {
      auto computeEncoder = mpsStream->commandEncoder();
      auto binaryPSO = getPipelineStateForFunc(kernel_name);
      // this function call is a no-op if MPS Profiler is not enabled
      getMPSProfiler().beginProfileKernel(binaryPSO, kernel_name, {input, other});
      [computeEncoder setComputePipelineState:binaryPSO];
      bind_iter_tensors(computeEncoder, iter);
      if (use_scalar_kernel) {
        if (cast_needed) {
          std::array<uint32_t, 4> sizes_types = {static_cast<uint32_t>(c10::elementSize(input.scalar_type())),
                                                 static_cast<uint32_t>(c10::elementSize(other.scalar_type())),
                                                 static_cast<uint32_t>(input.scalar_type()),
                                                 static_cast<uint32_t>(other.scalar_type())};
          if (alpha) {
            mtl_setArgs<3>(computeEncoder, getMPSScalar(*alpha, alpha_type), sizes_types);
          } else {
            mtl_setArgs<3>(computeEncoder, sizes_types);
          }
        } else if (alpha) {
          mtl_setArgs<3>(computeEncoder, getMPSScalar(*alpha, alpha_type));
        }
      } else if (use_broadcast_kernel) {
        mtl_setArgs<3>(computeEncoder, broadcast_numel);
        if (cast_needed) {
          std::array<uint32_t, 4> sizes_types = {static_cast<uint32_t>(c10::elementSize(input.scalar_type())),
                                                 static_cast<uint32_t>(c10::elementSize(other.scalar_type())),
                                                 static_cast<uint32_t>(input.scalar_type()),
                                                 static_cast<uint32_t>(other.scalar_type())};
          if (alpha) {
            mtl_setArgs<4>(computeEncoder, getMPSScalar(*alpha, alpha_type), sizes_types);
          } else {
            mtl_setArgs<4>(computeEncoder, sizes_types);
          }
        } else if (alpha) {
          mtl_setArgs<4>(computeEncoder, getMPSScalar(*alpha, alpha_type));
        }
      } else {
        if (iter.is_contiguous()) {
          if (alpha) {
            mtl_setBytes(computeEncoder, getMPSScalar(*alpha, alpha_type), 3);
          }
          if (cast_needed) {
            std::array<int, 4> size_and_types = {static_cast<int>(c10::elementSize(input.scalar_type())),
                                                 static_cast<int>(c10::elementSize(other.scalar_type())),
                                                 static_cast<int>(input.scalar_type()),
                                                 static_cast<int>(other.scalar_type())};
            mtl_setBytes(computeEncoder, size_and_types, alpha ? 4 : 3);
          }
        } else {
          // Please note that shapes and strides of the iterator might be
          // different than that of its operands, for example binary op
          // between 4x4 tensor and scalar will result in 1D 16 element iterator
          std::array<int, 4> ndim_and_types = {iter.ndim(),
                                               static_cast<int>(input.scalar_type()),
                                               static_cast<int>(other.scalar_type()),
                                               static_cast<int>(out.scalar_type())};
          if (alpha) {
            mtl_setArgs<3>(computeEncoder,
                           getMPSScalar(*alpha, alpha_type),
                           iter.shape(),
                           iter.strides(0),
                           iter.strides(1),
                           iter.strides(2),
                           ndim_and_types);
          } else {
            mtl_setArgs<3>(
                computeEncoder, iter.shape(), iter.strides(0), iter.strides(1), iter.strides(2), ndim_and_types);
          }
        }
      }
      mtl_dispatch1DJob(computeEncoder, binaryPSO, iter.numel());
      getMPSProfiler().endProfileKernel(binaryPSO);
    }
  });
}

void MetalShaderLibrary::exec_ternary_kernel(TensorIteratorBase& iter, const std::string& name) {
  // TODO: Figure a better place to downcast double scalars (probably in tensor iterator itself?)
  // Right now running something like 1.0-torch.rand(5, device='mps') will create iterator with
  // double as common dtype (because Python floating point are always 64-bit values)
  TORCH_CHECK(iter.output().scalar_type() != at::kDouble, "float64 is not supported on MPS");

  // Skip for empty iterators
  if (iter.numel() == 0) {
    return;
  }

  // Decompose 64-bit tensor into 32-bit ones
  if (!iter.can_use_32bit_indexing()) {
    for (auto&& sub_iter : iter.with_32bit_indexing()) {
      exec_binary_kernel(sub_iter, name);
    }
    return;
  }

  auto convert_double_scalar = [](Tensor& t) {
    if (t.dim() != 0) {
      return;
    }
    if (t.scalar_type() == kDouble) {
      t = t.to(kFloat);
    } else if (t.scalar_type() == kComplexDouble) {
      t = t.to(kComplexFloat);
    }
  };

  Tensor input = iter.input(0);
  Tensor other1 = iter.input(1);
  Tensor other2 = iter.input(2);
  Tensor out = iter.output();

  convert_double_scalar(input);
  convert_double_scalar(other1);
  convert_double_scalar(other2);

  MPSStream* mpsStream = getCurrentMPSStream();
  const auto cast_needed =
      (input.scalar_type() != other1.scalar_type()) || (input.scalar_type() != other2.scalar_type());
  const auto suffix = iter.is_contiguous() ? "dense" : "strided";
  // TODO: Implicitly pass both input and output types to non-cast kernels
  const auto kernel_name = cast_needed
      ? fmt::format("{}_{}_cast_{}", name, suffix, scalarToMetalTypeString(out))
      : fmt::format("{}_{}_{}_{}", name, suffix, scalarToMetalTypeString(out), scalarToMetalTypeString(input));
  dispatch_sync_with_rethrow(mpsStream->queue(), ^() {
    @autoreleasepool {
      auto computeEncoder = mpsStream->commandEncoder();
      auto binaryPSO = getPipelineStateForFunc(kernel_name);
      // this function call is a no-op if MPS Profiler is not enabled
      getMPSProfiler().beginProfileKernel(binaryPSO, kernel_name, {input, other1, other2});
      [computeEncoder setComputePipelineState:binaryPSO];
      // Set input and output tensors
      bind_iter_tensors(computeEncoder, iter);
      // Iterator is contiguous if all of its elements are dense in storage,
      // i.e. it's true for both row-first and column-first tensors
      if (iter.is_contiguous()) {
        if (cast_needed) {
          std::array<int, 3> sizes = {static_cast<int>(c10::elementSize(input.scalar_type())),
                                      static_cast<int>(c10::elementSize(other1.scalar_type())),
                                      static_cast<int>(c10::elementSize(other2.scalar_type()))};
          std::array<int, 3> types = {static_cast<int>(input.scalar_type()),
                                      static_cast<int>(other1.scalar_type()),
                                      static_cast<int>(other2.scalar_type())};
          mtl_setArgs<4>(computeEncoder, sizes, types);
        }
      } else {
        // Please note that shapes and strides of the iterator might be
        // different than that of its operands, for example binary op
        // between 4x4 tensor and scalar will result in 1D 16 element iterator
        std::array<int, 4> types = {static_cast<int>(input.scalar_type()),
                                    static_cast<int>(other1.scalar_type()),
                                    static_cast<int>(other2.scalar_type()),
                                    static_cast<int>(out.scalar_type())};
        mtl_setArgs<4>(computeEncoder,
                       iter.shape(),
                       iter.strides(0),
                       iter.strides(1),
                       iter.strides(2),
                       iter.strides(3),
                       iter.ndim(),
                       types);
      }
      mtl_dispatch1DJob(computeEncoder, binaryPSO, iter.numel());
      getMPSProfiler().endProfileKernel(binaryPSO);
    }
  });
}

MetalShaderLibrary& MetalShaderLibrary::getBundledLibrary() {
  static BundledShaderLibary l;
  return l;
}

// DynamicMetalShaderLibrary implementation
DynamicMetalShaderLibrary::~DynamicMetalShaderLibrary() {
  [library release];
}

// MetalKernelFunction implementation
MetalKernelFunction::MetalKernelFunction(MTLComputePipelineState_t cps_, MTLFunction_t f_)
    : cps([cps_ retain]), func([f_ retain]) {}

MetalKernelFunction::~MetalKernelFunction() {
  [cps release];
  [func release];
}

void MetalKernelFunction::runCommandBlock(std::function<void(void)> run) {
  dispatch_sync_with_rethrow(getCurrentMPSStream()->queue(), ^() {
    @autoreleasepool {
      run();
    }
  });
}

void MetalKernelFunction::startEncoding() {
  encoder = getCurrentMPSStream()->commandEncoder();
  [encoder setComputePipelineState:cps];
}

void MetalKernelFunction::dispatch(uint64_t length, std::optional<uint64_t> group_size) {
  const auto max_tg_size = getMaxThreadsPerThreadgroup();
  const auto group_size_val = group_size.value_or(std::min(length, max_tg_size));
  TORCH_CHECK_VALUE(group_size_val <= max_tg_size, "Threadgroup size exceeds ", max_tg_size, " limit");
  [encoder dispatchThreads:MTLSizeMake(length, 1, 1) threadsPerThreadgroup:MTLSizeMake(group_size_val, 1, 1)];
}

void MetalKernelFunction::dispatch(c10::ArrayRef<uint64_t> length, c10::OptionalArrayRef<uint64_t> group_size) {
  TORCH_CHECK(!length.empty() && length.size() < 4, "Dispatch dimensions must be less than 3 and non-empty");
  TORCH_CHECK(!group_size.has_value() || group_size->size() == length.size(),
              "size and group_size must have same number of dimensions");
  const auto max_tg_size = getMaxThreadsPerThreadgroup();
  const auto group_size_length = group_size.has_value() ? group_size->size() : 0;
  auto tg_size = MTLSizeMake(group_size_length > 0 ? group_size->at(0) : max_tg_size,
                             group_size_length > 1 ? group_size->at(1) : 1,
                             group_size_length > 2 ? group_size->at(2) : 1);
  TORCH_CHECK_VALUE(tg_size.width * tg_size.height * tg_size.depth <= max_tg_size,
                    "Threadgroup size exceeds ",
                    max_tg_size,
                    " limit");
  [encoder dispatchThreads:MTLSizeMake(length[0], length.size() > 1 ? length[1] : 1, length.size() == 3 ? length[2] : 1)
      threadsPerThreadgroup:tg_size];
}

void MetalKernelFunction::setArg(unsigned idx, const at::TensorBase& t) {
  mtl_setBuffer(encoder, t, idx);
}

void MetalKernelFunction::setArg(unsigned idx, const void* ptr, uint64_t size) {
  TORCH_CHECK(size > 0);
  [encoder setBytes:ptr length:size atIndex:idx];
}

void MetalKernelFunction::setErrorBufferIndex(unsigned idx) {
  auto stream = ::at::mps::getCurrentMPSStream();
  [encoder setBuffer:stream->getErrorBuffer() offset:0 atIndex:idx];
}

uint64_t MetalKernelFunction::getMaxThreadsPerThreadgroup() const {
  return [cps maxTotalThreadsPerThreadgroup];
}

uint64_t MetalKernelFunction::getThreadExecutionWidth() const {
  return [cps threadExecutionWidth];
}

uint64_t MetalKernelFunction::getStaticThreadGroupMemoryLength() const {
  return [cps staticThreadgroupMemoryLength];
}

void* get_tensor_gpu_address(const at::TensorBase& t) {
  return reinterpret_cast<void*>(getMTLBufferStorage(t).gpuAddress + t.storage_offset() * t.element_size());
}

} // namespace at::native::mps

// Check that c10::metal::ScalarType is strict subset (with matching values) of c10::ScalarType
#define DTYPE_CHECKER(_n, _v) \
  static_assert(static_cast<int>(::c10::ScalarType::_n) == static_cast<int>(::c10::metal::ScalarType::_n));
C10_METAL_ALL_TYPES_FUNCTOR(DTYPE_CHECKER)
