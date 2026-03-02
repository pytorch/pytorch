//  Copyright © 2022 Apple Inc.

#pragma once

#include <initializer_list>
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/Tensor.h>
#include <ATen/TensorIterator.h>
#include <ATen/Utils.h>
#include <ATen/mps/MPSProfiler.h>
#include <ATen/mps/MPSStream.h>
#include <ATen/native/mps/MetalShaderLibrary.h>
#include <ATen/native/mps/TensorFactory.h>
#include <c10/core/ScalarType.h>
#include <fmt/format.h>
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

using namespace at::mps;

namespace at::native::mps {

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
std::string to_hex_key(float);
std::string getArrayRefString(const IntArrayRef s);
// use has_storage() on the returned tensor to determine if src actually is a view
Tensor gatherViewTensor(const Tensor& src, Tensor& dst);
Tensor& scatterViewTensor(const Tensor& src, Tensor& output);
MPSGraphTensor* castToIHFTypes(MPSGraph* mpsGraph, MPSGraphTensor* inputTensor, const TensorBase& input);
MPSGraphTensor* castFromIHFTypes(MPSGraph* mpsGraph, MPSGraphTensor* inputTensor, const TensorBase& input);

MPSNDArray* getStridedMPSNDArray(const TensorBase& src, MPSNDArray* srcNDArray);
MPSNDArray* getMPSNDArray(const TensorBase& t, const IntArrayRef& sizes = {}, const IntArrayRef& strides = {});
MPSNDArray* getMPSNDArray(const TensorBase& t, MPSShape* sizes = nil, MPSShape* strides = nil);
// The MPSShape could vary based on memory format
Tensor getTensorView(const Tensor& t, MPSShape* shape);
MPSShape* getMPSShape(const TensorBase& t, c10::MemoryFormat memory_format = MemoryFormat::Contiguous);
MPSShape* getMPSShape(IntArrayRef sizes, c10::MemoryFormat memory_format = MemoryFormat::Contiguous);

// Determines whether a tensor is too large to use MPSGraph
bool isTooLargeForMPSGraph(const Tensor& tensor, bool useMPSStridedAPI = true);

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

MPSGraphTensor* mpsGraphUnrankedPlaceHolder(MPSGraph* mpsGraph, MPSDataType dataType);
MPSGraphTensor* mpsGraphRankedPlaceHolder(MPSGraph* mpsGraph, MPSDataType dataType, MPSShape* mpsShape);
MPSGraphTensor* mpsGraphRankedPlaceHolder(MPSGraph* mpsGraph, const TensorBase& tensor);
MPSGraphTensor* mpsGraphScalarPlaceHolder(MPSGraph* mpsGraph, MPSDataType dataType);
MPSGraphTensor* mpsGraphScalarPlaceHolder(MPSGraph* mpsGraph, const Scalar& scalar);

std::string get_mem_format_string(c10::MemoryFormat memory_format);

using MPSCacheKey = uint64_t;

struct MPSCachedKernel {
  MPSCachedKernel(NSObject* object) : _object([object retain]) {}
  virtual ~MPSCachedKernel() {
    [_object release];
    _object = nullptr;
  }

  // Delete copy constructor and assignment
  MPSCachedKernel(const MPSCachedKernel&) = delete;
  void operator=(const MPSCachedKernel&) = delete;

  template <typename T>
  inline T* kernel() const {
    return (T*)_object;
  }

 private:
  NSObject* _object = nullptr;
};

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

struct MPSKernelCache {
  typedef MPSCachedKernel* (^CreateCachedKernelBlock)();

  struct CacheEntry {
    CacheEntry(const std::string& key, MPSCachedKernel* cachedKernel) : cachedKernel_(cachedKernel), key_(key) {}
    MPSCachedKernel* cachedKernel_ = nullptr;
    std::string key_;
  };

 public:
  static MPSKernelCache* getInstance() {
    if (_instance_cache == nullptr) {
      _instance_cache = new MPSKernelCache();
    }
    return _instance_cache;
  }

  ~MPSKernelCache() {
    dispatch_release(serialQueue_);
    for (const auto& i : cache_) {
      delete i.second.cachedKernel_;
    }
  }

  // Disallow the copy constructor and operator= functions
  MPSKernelCache(const MPSKernelCache&) = delete;
  void operator=(const MPSKernelCache&) = delete;

  MPSCachedKernel* CreateCachedKernel(const std::string& key, CreateCachedKernelBlock createCacheBlock) {
    __block MPSCachedKernel* cachedKernel = nil;
    MPSCacheKey hash = std::hash<std::string>{}(key);
    dispatch_sync_with_rethrow(serialQueue_, ^() {
      if (cache_.count(hash) != 0) {
        auto& entry = cache_.at(hash);
        TORCH_INTERNAL_ASSERT_DEBUG_ONLY(key == entry.key_, "Key collision in the MPS cached kernel!\n");
        cachedKernel = entry.cachedKernel_;
      } else {
        cachedKernel = createCacheBlock();
        CacheEntry entry(key, cachedKernel);
        cache_.emplace(hash, entry);
      }
    });
    return cachedKernel;
  }
  template <typename T>
  inline T* CreateCachedKernelAs(const std::string& key, CreateCachedKernelBlock createCacheBlock) {
    return static_cast<T*>(CreateCachedKernel(key, createCacheBlock));
  }

  MPSCachedKernel* LookUp(const std::string& key) const {
    __block MPSCachedKernel* cachedKernel = nil;

    MPSCacheKey hash = std::hash<std::string>{}(key);
    dispatch_sync_with_rethrow(serialQueue_, ^() {
      if (cache_.count(hash) != 0) {
        auto& entry = cache_.at(hash);
        TORCH_INTERNAL_ASSERT_DEBUG_ONLY(key == entry.key_, "Key collision in the MPS cached kernel!\n");
        cachedKernel = entry.cachedKernel_;
      }
    });
    return cachedKernel;
  }

  template <typename T>
  inline T* LookUpAs(const std::string& key) const {
    return static_cast<T*>(LookUp(key));
  }

 private:
  MPSKernelCache() {
    serialQueue_ = dispatch_queue_create("kernel cache queue", DISPATCH_QUEUE_SERIAL);
  }

  static MPSKernelCache* _instance_cache;
  std::unordered_map<MPSCacheKey, CacheEntry> cache_;
  dispatch_queue_t serialQueue_ = nullptr;
};

// Common template for creating cached kernel if missing
template <typename T>
inline T* LookUpOrCreateCachedKernel(const std::string& key, std::function<MPSKernel*()> instantiate) {
  auto cache_ = MPSKernelCache::getInstance();
  if (auto rc = cache_->LookUpAs<T>(key)) {
    return rc;
  }
  return cache_->CreateCachedKernelAs<T>(key, ^mps::MPSCachedKernel*() {
    auto k_ = new mps::MPSCachedKernel(instantiate());
    return k_;
  });
}

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
      // MPS does not support doubles, silently downcast CPU scalar to float
      if (C10_UNLIKELY(t.scalar_type() == kDouble)) {
        auto val = static_cast<float>(*reinterpret_cast<const double*>(t.const_data_ptr()));
        [encoder setBytes:&val length:sizeof(val) atIndex:idx];
        return;
      }
      if (C10_UNLIKELY(t.scalar_type() == kComplexDouble)) {
        auto val = static_cast<c10::complex<float>>(*reinterpret_cast<const c10::complex<double>*>(t.const_data_ptr()));
        [encoder setBytes:&val length:sizeof(val) atIndex:idx];
        return;
      }
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

static size_t iter_tensor_offset(TensorIteratorBase& iter, unsigned idx) {
  // At the moment, MPS storage data is not the real GPU pointer, but rather a pointer to id<MTLBuffer> object
  // But TensorIterator constructs data_ptr as if base was just a raw pointer
  // Workaround this problem by computing an offset from the start of the tensor, which works for both
  // tensor views and sliced 64-bit iterators
  return reinterpret_cast<size_t>(iter.data_ptr(idx)) -
      reinterpret_cast<size_t>(iter.tensor_base(idx).storage().data());
}

static inline void bind_iter_tensors(id<MTLComputeCommandEncoder> encoder,
                                     TensorIteratorBase& iter,
                                     std::optional<size_t> ntensors = std::nullopt) {
  for (auto idx : c10::irange(ntensors.value_or(iter.ntensors()))) {
    auto& t = iter.tensor_base(idx);
    // Handle CPU scalars
    if (C10_UNLIKELY(t.device().type() == kCPU)) {
      mtl_setBuffer(encoder, t, idx);
      continue;
    }
    auto offs = iter_tensor_offset(iter, idx);
    [encoder setBuffer:getMTLBufferStorage(t) offset:offs atIndex:idx];
  }
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

// MPS yet to support double types, but starting from MacOS 14, supports bfloat16
inline bool supportedFloatingType(ScalarType dtype) {
  return dtype == kFloat || dtype == kHalf || dtype == kBFloat16;
}

inline bool supportedFloatingType(const TensorBase& t) {
  return supportedFloatingType(t.scalar_type());
}

inline bool supportedFloatingOrComplexType(ScalarType dtype) {
  if (dtype == kComplexFloat || dtype == kComplexHalf) {
    return true;
  }
  return supportedFloatingType(dtype);
}
inline bool supportedFloatingOrComplexType(const TensorBase& t) {
  return supportedFloatingOrComplexType(t.scalar_type());
}

inline bool needsGather(const TensorBase& t) {
  static const bool is_macOS_15_0_or_newer = is_macos_13_or_newer(MacOSVersion::MACOS_VER_15_0_PLUS);
  return !is_macOS_15_0_or_newer && (!t.is_contiguous() || t.storage_offset());
}

template <typename T>
void MetalShaderLibrary::exec_unary_kernel_with_params(TensorIteratorBase& iter,
                                                       const std::string& name,
                                                       T params,
                                                       const std::string& params_type_name) {
  using namespace at::mps;
  // Decompose 64-bit tensor into 32-bit ones
  if (!iter.can_use_32bit_indexing()) {
    for (auto&& sub_iter : iter.with_32bit_indexing()) {
      exec_unary_kernel_with_params(sub_iter, name, params, params_type_name);
    }
    return;
  }

  auto inputTensor = iter.input(0);
  auto outputTensor = iter.output(0);
  uint32_t length = iter.numel();
  if (length == 0) {
    return;
  }
  auto kernel_name = fmt::format("{}_{}_{}_{}{}",
                                 name,
                                 iter.is_contiguous() ? "dense" : "strided",
                                 scalarToMetalTypeString(outputTensor),
                                 scalarToMetalTypeString(inputTensor),
                                 fmt::format("_{}", params_type_name));
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
      detail::mtl_setArg(computeEncoder, params, iter.is_contiguous() ? 2 : 6);
      mtl_dispatch1DJob(computeEncoder, cplState, length);

      getMPSProfiler().endProfileKernel(cplState);
    });
  }
}

template <typename T>
void MetalShaderLibrary::exec_binary_kernel_with_params(TensorIteratorBase& iter,
                                                        const std::string& name,
                                                        T params,
                                                        const std::string& params_type_name) {
  using namespace mps;
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
      exec_binary_kernel_with_params(sub_iter, name, params, params_type_name);
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
  // TODO: Implicitly pass both input and output types to non-cast kernels
  const auto kernel_name = cast_needed
      ? fmt::format("{}_{}_cast_{}_{}", name, suffix, scalarToMetalTypeString(out), params_type_name)
      : fmt::format("{}_{}_{}_{}_{}",
                    name,
                    suffix,
                    scalarToMetalTypeString(out),
                    scalarToMetalTypeString(input),
                    params_type_name);
  dispatch_sync_with_rethrow(mpsStream->queue(), ^() {
    @autoreleasepool {
      auto computeEncoder = mpsStream->commandEncoder();
      auto binaryPSO = getPipelineStateForFunc(kernel_name);
      // this function call is a no-op if MPS Profiler is not enabled
      getMPSProfiler().beginProfileKernel(binaryPSO, kernel_name, {input, other});
      [computeEncoder setComputePipelineState:binaryPSO];
      // Set input and output tensors
      bind_iter_tensors(computeEncoder, iter);
      // Iterator is contiguous if all of its elements are dense in storage,
      // i.e. it's true for both row-first and column-first tensors
      if (iter.is_contiguous()) {
        detail::mtl_setArg(computeEncoder, params, 3);
        if (cast_needed) {
          std::array<int, 4> size_and_types = {static_cast<int>(c10::elementSize(input.scalar_type())),
                                               static_cast<int>(c10::elementSize(other.scalar_type())),
                                               static_cast<int>(input.scalar_type()),
                                               static_cast<int>(other.scalar_type())};
          mtl_setBytes(computeEncoder, size_and_types, 4);
        }
      } else {
        // Please note that shapes and strides of the iterator might be
        // different than that of its operands, for example binary op
        // between 4x4 tensor and scalar will result in 1D 16 element iterator
        std::array<int, 4> ndim_and_types = {iter.ndim(),
                                             static_cast<int>(input.scalar_type()),
                                             static_cast<int>(other.scalar_type()),
                                             static_cast<int>(out.scalar_type())};
        mtl_setArgs<3>(
            computeEncoder, params, iter.shape(), iter.strides(0), iter.strides(1), iter.strides(2), ndim_and_types);
      }
      mtl_dispatch1DJob(computeEncoder, binaryPSO, iter.numel());
      getMPSProfiler().endProfileKernel(binaryPSO);
    }
  });
}

// Checks if one tensor is broadcastable into another
static bool is_dense_broadcastable(const Tensor& from, const Tensor& into) {
  if (!from.is_contiguous() || !into.is_contiguous()) {
    return false;
  }
  bool checking_squeezable_dims = false;
  for (const auto dim : c10::irange(from.ndimension())) {
    if (checking_squeezable_dims) {
      if (from.size(-dim - 1) == 1) {
        continue;
      }
      return false;
    }
    checking_squeezable_dims = from.size(-dim - 1) != into.size(-dim - 1);
  }
  return true;
}

} // namespace at::native::mps
