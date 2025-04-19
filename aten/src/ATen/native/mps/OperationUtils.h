//  Copyright © 2022 Apple Inc.

#pragma once

#include <initializer_list>
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/Tensor.h>
#include <ATen/Utils.h>
#include <ATen/mps/MPSStream.h>
#include <ATen/native/mps/MetalShaderLibrary.h>
#include <ATen/native/mps/TensorFactory.h>
#include <c10/core/ScalarType.h>
#include <torch/library.h>
#include <unordered_map>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/zeros.h>
#include <ATen/ops/zeros_like.h>
#endif

#include <MetalPerformanceShaders/MetalPerformanceShaders.h>

@interface MPSGraph (PyTorchFixups)
- (MPSGraphTensor*)minimumWithNaNPropagationAndIntFallbackWithPrimaryTensor:(MPSGraphTensor*)primaryTensor
                                                            secondaryTensor:(MPSGraphTensor*)secondaryTensor
                                                                       name:(NSString*)name;

- (MPSGraphTensor*)maximumWithNaNPropagationAndIntFallbackWithPrimaryTensor:(MPSGraphTensor*)primaryTensor
                                                            secondaryTensor:(MPSGraphTensor*)secondaryTensor
                                                                       name:(NSString*)name;
@end

// Fwd declarations
namespace at {
struct TensorIteratorBase;
}
using namespace at::mps;

namespace at::native::mps {

void dispatch_sync_with_rethrow(dispatch_queue_t queue, void (^block)());

struct MPSScalar {
  id<MTLBuffer> getMTLBuffer() const {
    return __builtin_bit_cast(id<MTLBuffer>, buffer.get());
  }

  size_t size = 0;
  ScalarType type = ScalarType::Undefined;
  c10::DataPtr buffer; // stores MTLBuffer (frees buffer if MPSScalar instance goes out of scope)
  union {
    float f; // MPS doesn't support 'double'
    at::Half h;
    int64_t i;
    bool b;
    c10::complex<float> cf;
    c10::complex<at::Half> ch;
    at::BFloat16 bf16;
  } value{};
};

void runMPSGraph(MPSStream* mpsStream, MPSGraph* mpsGraph, NSDictionary* feeds, NSDictionary* results);

MPSDataType getMPSDataType(ScalarType scalar_type);
static inline MPSDataType getMPSDataType(const TensorBase& t) {
  return getMPSDataType(t.scalar_type());
}
MPSDataType getMPSScalarType(ScalarType scalar_type);
static inline MPSDataType getMPSScalarType(const TensorBase& t) {
  return getMPSScalarType(t.scalar_type());
}
MPSScalar getMPSScalar(const Scalar& scalar, ScalarType type);
std::string getMPSTypeString(ScalarType scalar_type, bool short_name = false);
static inline std::string getMPSTypeString(const TensorBase& t, bool short_name = false) {
  return getMPSTypeString(t.scalar_type(), short_name);
}
std::string scalarToMetalTypeString(const c10::ScalarType& scalar_type);
static inline std::string scalarToMetalTypeString(const TensorBase& t) {
  return scalarToMetalTypeString(t.scalar_type());
}
NSArray<NSNumber*>* getTensorAxes(const TensorBase& t);
NSArray<NSNumber*>* getTensorAxes(const IntArrayRef& sizes, at::OptionalIntArrayRef dim);
std::string getMPSShapeString(MPSShape* shape);
std::string getTensorsStringKey(const TensorList& tensors, bool short_dtype = true, bool exclude_shape = false);
std::string getArrayRefString(const IntArrayRef s);
// use has_storage() on the returned tensor to determine if src actually is a view
Tensor gatherViewTensor(const Tensor& src, Tensor& dst);
Tensor& scatterViewTensor(const Tensor& src, Tensor& output);
MPSGraphTensor* castToIHFTypes(MPSGraph* mpsGraph,
                               MPSGraphTensor* inputTensor,
                               const TensorBase& input,
                               bool includesInt64 = false);
MPSGraphTensor* castFromIHFTypes(MPSGraph* mpsGraph,
                                 MPSGraphTensor* inputTensor,
                                 const TensorBase& input,
                                 bool includesInt64 = false);

MPSNDArray* getMPSNDArray(const TensorBase& t, const IntArrayRef& sizes = {}, const IntArrayRef& strides = {});
MPSNDArray* getMPSNDArray(const TensorBase& t, MPSShape* sizes = nil, MPSShape* strides = nil);
// The MPSShape could vary based on memory format
Tensor getTensorView(const Tensor& t, MPSShape* shape);
MPSShape* getMPSShape(const TensorBase& t, c10::MemoryFormat memory_format = MemoryFormat::Contiguous);
MPSShape* getMPSShape(IntArrayRef sizes, c10::MemoryFormat memory_format = MemoryFormat::Contiguous);

static inline id<MTLBuffer> getMTLBufferStorage(const TensorBase& tensor) {
  return __builtin_bit_cast(id<MTLBuffer>, tensor.storage().data());
}

class Placeholder {
 public:
  Placeholder() : _placeholder(nullptr), _value(nullptr), _tensor(Tensor()) {}
  Placeholder(MPSGraphTensor* mpsGraphTensor) : _placeholder(mpsGraphTensor), _value(nullptr), _tensor(Tensor()) {}
  Placeholder(MPSGraphTensor* mpsGraphTensor, MPSNDArray* mpsNDArray);
  Placeholder(MPSGraphTensor* mpsGraphTensor,
              const Tensor& self,
              MPSShape* mpsShape = nullptr,
              bool gatherTensorData = true,
              MPSDataType dataType = MPSDataTypeInvalid,
              bool useMPSStridedAPI = true);
  MPSGraphTensor* getMPSGraphTensor() {
    return _placeholder;
  }
  MPSGraphTensorData* getMPSGraphTensorData() {
    return _value;
  }
  bool isIntermediate() {
    return _value == nullptr;
  }

 private:
  MPSGraphTensor* _placeholder;
  MPSGraphTensorData* _value;
  Tensor _tensor;
};

void resize_tensor(Tensor* output);
Tensor wrapped_scalar_tensor_mps(const Scalar& scalar, const Device device);
MPSGraphTensor* convertNHWCtoNCHW(MPSGraph* mpsGraph, MPSGraphTensor* tensor);
MPSGraphTensor* castMPSTensor(MPSGraph* mpsGraph, MPSGraphTensor* tensor, ScalarType toType);
MPSGraphTensor* castMPSTensor(MPSGraph* mpsGraph, MPSGraphTensor* tensor, MPSDataType toType);
MPSGraphTensorData* getMPSGraphTensorData(MPSGraph* mpsGraph, MPSStream* mpsStream, const TensorBase& tensor);
MPSGraphTensorData* getMPSGraphTensorFromScalar(MPSStream* mpsStream, MPSScalar& scalar);

MPSGraph* make_mps_graph();
void printTensorNDArray(const TensorBase& t);
MPSNDArray* ndArrayFromTensor(const TensorBase& tensor, MPSShape* shape, MPSDataType mpsType);

MPSGraphTensor* mpsGraphUnrankedPlaceHolder(MPSGraph* mpsGraph, MPSDataType dataType);
MPSGraphTensor* mpsGraphRankedPlaceHolder(MPSGraph* mpsGraph, MPSDataType dataType, MPSShape* mpsShape);
MPSGraphTensor* mpsGraphRankedPlaceHolder(MPSGraph* mpsGraph, const TensorBase& tensor);
MPSGraphTensor* mpsGraphScalarPlaceHolder(MPSGraph* mpsGraph, MPSDataType dataType);
MPSGraphTensor* mpsGraphScalarPlaceHolder(MPSGraph* mpsGraph, const Scalar& scalar);

string get_mem_format_string(c10::MemoryFormat memory_format);

using MPSCacheKey = uint64_t;

// derive this class to cache a graph and its inputs/outputs
// can be used to store any NSObject
struct MPSCachedGraph {
  MPSCachedGraph(NSObject* object) : _object([object retain]) {}
  virtual ~MPSCachedGraph() {
    [_object release];
    _object = nullptr;
  }

  template <typename T>
  inline T* as() {
    return static_cast<T*>(this);
  }

  MPSGraph* graph() const {
    return (MPSGraph*)_object;
  }
  NSObject* object() const {
    return _object;
  }

 private:
  NSObject* _object = nullptr;
};

struct MPSUnaryCachedGraph : public MPSCachedGraph {
  MPSUnaryCachedGraph(MPSGraph* graph) : MPSCachedGraph(graph) {}
  MPSGraphTensor* inputTensor_ = nil;
  MPSGraphTensor* outputTensor_ = nil;
};

struct MPSUnaryGradCachedGraph : public MPSCachedGraph {
  MPSUnaryGradCachedGraph(MPSGraph* graph) : MPSCachedGraph(graph) {}
  MPSGraphTensor* gradOutputTensor_ = nil;
  MPSGraphTensor* inputTensor_ = nil;
  MPSGraphTensor* outputTensor_ = nil; // some backward input is actually the forward's output
  MPSGraphTensor* gradInputTensor_ = nil;
};

struct MPSBinaryCachedGraph : public MPSCachedGraph {
  MPSBinaryCachedGraph(MPSGraph* graph) : MPSCachedGraph(graph) {}
  MPSGraphTensor* inputTensor_ = nil;
  MPSGraphTensor* otherTensor_ = nil;
  MPSGraphTensor* outputTensor_ = nil;
};

struct MPSBinaryGradCachedGraph : public MPSCachedGraph {
  MPSBinaryGradCachedGraph(MPSGraph* graph) : MPSCachedGraph(graph) {}
  MPSGraphTensor* gradOutputTensor_ = nil;
  MPSGraphTensor* inputTensor_ = nil;
  MPSGraphTensor* otherTensor_ = nil;
  MPSGraphTensor* gradInputTensor_ = nil;
};

// TODO: Improve the overall design of MPSGraphCache.
// https://github.com/pytorch/pytorch/issues/77176
// Cache holding various keys mapped to graphs
struct MPSGraphCache {
  typedef MPSCachedGraph* (^CreateCachedGraphBlock)();

  struct CacheEntry {
    CacheEntry(const std::string& key, MPSCachedGraph* cachedGraph) : cachedGraph_(cachedGraph), key_(key) {}
    MPSCachedGraph* cachedGraph_ = nullptr;
    std::string key_;
  };

 public:
  static MPSGraphCache* getInstance() {
    if (_instance_cache == nullptr) {
      _instance_cache = new MPSGraphCache();
    }
    return _instance_cache;
  }

  ~MPSGraphCache() {
    dispatch_release(serialQueue_);

    for (const auto& i : cache_) {
      delete i.second.cachedGraph_;
    }
  }

  // Disallow the copy constructor and operator= functions
  MPSGraphCache(const MPSGraphCache&) = delete;
  void operator=(const MPSGraphCache&) = delete;

  MPSCachedGraph* CreateCachedGraph(const std::string& key, CreateCachedGraphBlock createCacheBlock) {
    __block MPSCachedGraph* cachedGraph = nil;

    MPSCacheKey hash = std::hash<std::string>{}(key);

    dispatch_sync_with_rethrow(serialQueue_, ^() {
      // verify the cached entry doesn't already exist
      if (cache_.count(hash) != 0) {
        auto& entry = cache_.at(hash);
        TORCH_INTERNAL_ASSERT_DEBUG_ONLY(key == entry.key_, "Key collision in the MPS cached graph!\n");
        cachedGraph = entry.cachedGraph_;
      } else {
        cachedGraph = createCacheBlock();
        CacheEntry entry(key, cachedGraph);
        cache_.emplace(hash, entry);
        profileCachedGraph(entry);
      }
    });
    return cachedGraph;
  }

  template <typename T>
  inline T* CreateCachedGraphAs(const std::string& key, CreateCachedGraphBlock createCacheBlock) {
    return static_cast<T*>(CreateCachedGraph(key, createCacheBlock));
  }

  MPSCachedGraph* LookUp(const std::string& key) const {
    __block MPSCachedGraph* cachedGraph = nullptr;

    MPSCacheKey hash = std::hash<std::string>{}(key);

    dispatch_sync(serialQueue_, ^() {
      if (cache_.count(hash) != 0) {
        auto& entry = cache_.at(hash);
        TORCH_INTERNAL_ASSERT_DEBUG_ONLY(key == entry.key_, "Key collision in the MPS cached graph!\n");
        cachedGraph = entry.cachedGraph_;
        profileCachedGraph(entry);
      }
    });
    return cachedGraph;
  }

  template <typename T>
  inline T* LookUpAs(const std::string& key) const {
    return static_cast<T*>(LookUp(key));
  }

 private:
  MPSGraphCache() {
    serialQueue_ = dispatch_queue_create("cache queue", DISPATCH_QUEUE_SERIAL);
  }
  // this is defined in OperationUtils.mm to not include
  // MPSProfiler.h in header OperationUtils.h
  void profileCachedGraph(const CacheEntry& cacheEntry) const;

  static MPSGraphCache* _instance_cache;
  std::unordered_map<MPSCacheKey, CacheEntry> cache_;
  dispatch_queue_t serialQueue_ = nullptr;
};

// Common template for creating graph with a specified cache if missing
template <typename T>
inline T* LookUpOrCreateCachedGraph(const std::string& key, std::function<void(MPSGraph*, T*)> instantiate) {
  auto cache_ = MPSGraphCache::getInstance();
  if (auto rc = cache_->LookUpAs<T>(key)) {
    return rc;
  }
  return cache_->CreateCachedGraphAs<T>(key, ^mps::MPSCachedGraph*() {
    T* newCachedGraph = nil;
    @autoreleasepool {
      // Initialize graph
      auto mpsGraph = mps::make_mps_graph();
      newCachedGraph = new T(mpsGraph);
      instantiate(mpsGraph, newCachedGraph);
    }
    return newCachedGraph;
  });
}

// Common math operations
MPSGraphTensor* log1p(MPSGraph* mpsGraph, MPSGraphTensor* inputTensor);

#define MPS_CHECK_INT64_OP_SUPPORTED(input_tensor, mac_os_13_3_plus, op_name)                                            \
  if (!mac_os_13_3_plus && input_tensor.scalar_type() == kLong) {                                                        \
    TORCH_WARN_ONCE(                                                                                                     \
        "MPS: no support for int64 for ",                                                                                \
        op_name,                                                                                                         \
        ", downcasting to a smaller data type (int32/float32). Native support for int64 has been added in macOS 13.3."); \
  }

/**
 * Returns distance from lowest to highest element offset in given tensor.
 */
size_t compute_storage_numel_distance(const TensorBase& t);

/**
 * Checks whether tensor is mapped to a contiguous area in the storage.
 */
inline bool is_dense_in_storage(const TensorBase& t) {
  return compute_storage_numel_distance(t) == static_cast<size_t>(t.numel());
}

template <typename encoder_t,
          typename = std::enable_if_t<std::is_same_v<id<MTLComputeCommandEncoder>, encoder_t> ||
                                      std::is_same_v<id<MTLArgumentEncoder>, encoder_t>>>
static inline void mtl_setBuffer(encoder_t encoder, const TensorBase& t, unsigned idx) {
  if (C10_UNLIKELY(t.device().type() == kCPU)) {
    if constexpr (std::is_same_v<id<MTLComputeCommandEncoder>, encoder_t>) {
      TORCH_CHECK(t.dim() == 0, "Passed CPU tensor to MPS op");
      [encoder setBytes:t.storage().data() length:t.element_size() atIndex:idx];
    } else {
      TORCH_CHECK(false, "Passed CPU tensor to MPS op");
    }
    return;
  }
  [encoder setBuffer:getMTLBufferStorage(t) offset:t.storage_offset() * t.element_size() atIndex:idx];
}

// Implementation of setBytes for containers vs trivially copiable types must be separate
// Containers like `std::array` could have been uploaded directly, but `c10::ArrayRef`,
// while trivially copiable, includes padding  which if copied as Metal shader parameters
// might overwrite other values
template <
    typename T,
    typename = std::enable_if_t<std::is_integral_v<T> || std::is_same_v<T, float> ||
                                (std::is_class_v<T> && std::is_trivially_copyable_v<T> && !detail::has_size_type_v<T>)>>
static inline void mtl_setBytes(id<MTLComputeCommandEncoder> encoder, const T val, unsigned idx) {
  [encoder setBytes:&val length:sizeof(T) atIndex:idx];
}

template <typename Container, typename = std::enable_if_t<detail::has_size_type_v<Container>>>
static inline void mtl_setBytes(id<MTLComputeCommandEncoder> encoder, const Container& values, unsigned idx) {
  [encoder setBytes:values.data() length:sizeof(typename Container::value_type) * values.size() atIndex:idx];
}

static inline void mtl_setBytes(id<MTLComputeCommandEncoder> encoder, const MPSScalar& s, unsigned idx) {
  [encoder setBytes:&s.value length:s.size atIndex:idx];
}

namespace detail {
template <typename T>
inline void mtl_setArg(id<MTLComputeCommandEncoder> encoder, const T& val, unsigned idx) {
  mtl_setBytes(encoder, val, idx);
}

inline void mtl_setArg(id<MTLComputeCommandEncoder> encoder, id<MTLBuffer> val, unsigned idx) {
  [encoder setBuffer:val offset:0 atIndex:idx];
}

template <>
inline void mtl_setArg(id<MTLComputeCommandEncoder> encoder, const Tensor& val, unsigned idx) {
  mtl_setBuffer(encoder, val, idx);
}

template <>
inline void mtl_setArg(id<MTLComputeCommandEncoder> encoder, const std::optional<Tensor>& val, unsigned idx) {
  if (val.has_value()) {
    mtl_setBuffer(encoder, val.value(), idx);
  }
}

template <>
inline void mtl_setArg(id<MTLComputeCommandEncoder> encoder, const TensorBase& val, unsigned idx) {
  mtl_setBuffer(encoder, val, idx);
}
// MPS does not support doubles, so cast it down to float before passing as an argument
template <>
inline void mtl_setArg(id<MTLComputeCommandEncoder> encoder, const double& val, unsigned idx) {
  float val_f = static_cast<float>(val);
  mtl_setBytes(encoder, val_f, idx);
}
} // namespace detail

template <unsigned idx = 0, typename T>
static inline void mtl_setArgs(id<MTLComputeCommandEncoder> encoder, const T& val) {
  detail::mtl_setArg(encoder, val, idx);
}

template <unsigned idx = 0, typename T, typename... Args>
static inline void mtl_setArgs(id<MTLComputeCommandEncoder> encoder, const T& val, Args&&... args) {
  detail::mtl_setArg(encoder, val, idx);
  mtl_setArgs<idx + 1>(encoder, std::forward<Args>(args)...);
}

static inline void mtl_dispatch1DJob(id<MTLComputeCommandEncoder> encoder,
                                     id<MTLComputePipelineState> cplState,
                                     NSUInteger length) {
  static_assert(sizeof(NSUInteger) == sizeof(uint64_t));
  const auto maxThreadsPerGroup = [cplState maxTotalThreadsPerThreadgroup];
  auto size = MTLSizeMake(length, 1, 1);
  auto threadGroupSize = MTLSizeMake(std::min(maxThreadsPerGroup, length), 1, 1);
  [encoder dispatchThreads:size threadsPerThreadgroup:threadGroupSize];
}

id<MTLBuffer> generateKernelDataOffsets(id<MTLComputeCommandEncoder> commandEncoder,
                                        const TensorIteratorBase& iter,
                                        bool use_64bit_index = false);

inline NSDictionary* dictionaryFromPlaceholders(Placeholder& p1) {
  return @{p1.getMPSGraphTensor() : p1.getMPSGraphTensorData()};
}

inline NSDictionary* dictionaryFromPlaceholders(Placeholder& p1, Placeholder& p2) {
  return @{
    p1.getMPSGraphTensor() : p1.getMPSGraphTensorData(),
    p2.getMPSGraphTensor() : p2.getMPSGraphTensorData(),
  };
}

inline NSDictionary* dictionaryFromPlaceholders(Placeholder& p1, Placeholder& p2, Placeholder& p3) {
  return @{
    p1.getMPSGraphTensor() : p1.getMPSGraphTensorData(),
    p2.getMPSGraphTensor() : p2.getMPSGraphTensorData(),
    p3.getMPSGraphTensor() : p3.getMPSGraphTensorData(),
  };
}

inline NSDictionary* dictionaryFromPlaceholders(Placeholder& p1, Placeholder& p2, Placeholder& p3, Placeholder& p4) {
  return @{
    p1.getMPSGraphTensor() : p1.getMPSGraphTensorData(),
    p2.getMPSGraphTensor() : p2.getMPSGraphTensorData(),
    p3.getMPSGraphTensor() : p3.getMPSGraphTensorData(),
    p4.getMPSGraphTensor() : p4.getMPSGraphTensorData(),
  };
}

inline void runMPSGraph(MPSStream* stream, MPSGraph* graph, NSDictionary* feeds, Placeholder& result) {
  runMPSGraph(stream, graph, feeds, dictionaryFromPlaceholders(result));
}

inline bool supportsComplex() {
  return is_macos_13_or_newer(MacOSVersion::MACOS_VER_14_0_PLUS);
}

// MPS yet to support double types, but starting from MacOS 14, supports bfloat16
inline bool supportedFloatingType(ScalarType dtype) {
  return dtype == kFloat || dtype == kHalf || dtype == kBFloat16;
}

inline bool supportedFloatingType(const TensorBase& t) {
  return supportedFloatingType(t.scalar_type());
}

inline bool supportedFloatingOrComplexType(ScalarType dtype) {
  if (dtype == kComplexFloat || dtype == kComplexHalf) {
    return supportsComplex();
  }
  return supportedFloatingType(dtype);
}
inline bool supportedFloatingOrComplexType(const TensorBase& t) {
  return supportedFloatingOrComplexType(t.scalar_type());
}

inline void checkSupportsBFloat16() {
  TORCH_CHECK_TYPE(is_macos_13_or_newer(MacOSVersion::MACOS_VER_14_0_PLUS),
                   "MPS bfloat16 type is supported on MacOS 14.0 or newer.");
}

inline bool needsGather(const TensorBase& t) {
  static const bool is_macOS_15_0_or_newer = is_macos_13_or_newer(MacOSVersion::MACOS_VER_15_0_PLUS);
  return !is_macOS_15_0_or_newer && (!t.is_contiguous() || t.storage_offset());
}

} // namespace at::native::mps
