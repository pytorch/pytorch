//  Copyright © 2022 Apple Inc.

#pragma once

#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/Tensor.h>
#include <ATen/Utils.h>
#include <ATen/mps/MPSStream.h>
#include <ATen/native/mps/TensorFactory.h>
#include <c10/util/Optional.h>
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

#ifdef __OBJC__
#include <MetalPerformanceShaders/MetalPerformanceShaders.h>
#endif

using namespace at::mps;

namespace at {
namespace native {
namespace mps {

struct MPSScalar {
  id<MTLBuffer> getMTLBuffer() const { return __builtin_bit_cast(id<MTLBuffer>, buffer.get()); }

  size_t size = 0;
  ScalarType type = ScalarType::Undefined;
  c10::DataPtr buffer; // stores MTLBuffer (frees buffer if MPSScalar instance goes out of scope)
  union {
    float f; // MPS doesn't support 'double'
    at::Half h;
    int64_t i;
    bool b;
  } value {};
};

void runMPSGraph(
    MPSStream* mpsStream,
    MPSGraph* mpsGraph,
    NSDictionary* feeds,
    NSDictionary* results);

MPSDataType getMPSDataType(ScalarType scalar_type);
static inline MPSDataType getMPSDataType(const Tensor& t) {
  return getMPSDataType(t.scalar_type());
}
MPSDataType getMPSScalarType(ScalarType scalar_type);
static inline MPSDataType getMPSScalarType(const Tensor& t) {
  return getMPSScalarType(t.scalar_type());
}
MPSScalar   getMPSScalar(const Scalar& scalar, ScalarType type);
std::string getMPSTypeString(ScalarType scalar_type, bool short_name = false);
static inline std::string getMPSTypeString(const Tensor& t, bool short_name = false) {
  return getMPSTypeString(t.scalar_type(), short_name);
}
std::string scalarToMetalTypeString(const c10::ScalarType& scalar_type);
NSArray<NSNumber*>* getTensorAxes(const Tensor& t);
NSArray<NSNumber*>* getTensorAxes(const IntArrayRef& sizes, at::OptionalIntArrayRef dim);
std::string getMPSShapeString(MPSShape* shape);
std::string getTensorsStringKey(const TensorList& tensors, bool short_dtype = true);
std::string getArrayRefString(const IntArrayRef s);
// use has_storage() on the returned tensor to determine if src actually is a view
Tensor gatherViewTensor(const at::Tensor& src, at::Tensor& dst);
Tensor& scatterViewTensor(const at::Tensor& src, at::Tensor& output);
bool canSliceViewTensor(const Tensor& src, MPSShape *mpsShape);
MPSGraphTensorData* getMPSGraphTensorDataForView(const Tensor& src, MPSShape *mpsShape, const MPSDataType mpsDataType);
MPSGraphTensor* castToIHFTypes(MPSGraph* mpsGraph, MPSGraphTensor* inputTensor, const Tensor& input, bool includesInt64 = false);
MPSGraphTensor* castFromIHFTypes(MPSGraph* mpsGraph, MPSGraphTensor* inputTensor, const Tensor& input, bool includesInt64 = false);

// The MPSShape could vary based on memory format
MPSShape* getMPSShape(const Tensor& t, c10::MemoryFormat memory_format = MemoryFormat::Contiguous);
MPSShape* getMPSShape(IntArrayRef sizes, c10::MemoryFormat memory_format = MemoryFormat::Contiguous);

static inline id<MTLBuffer> getMTLBufferStorage(const at::Tensor& tensor) {
  return __builtin_bit_cast(id<MTLBuffer>, tensor.storage().data());
}

class Placeholder {
 public:
  Placeholder() : _placeholder(nullptr), _value(nullptr), _tensor(Tensor()) {}
  Placeholder(MPSGraphTensor* mpsGraphTensor) : _placeholder(mpsGraphTensor), _value(nullptr), _tensor(Tensor()) {}
  Placeholder(MPSGraphTensor* mpsGraphTensor, const Tensor& self, MPSShape *mpsShape = nullptr,
              bool gatherTensorData = true, MPSDataType dataType = MPSDataTypeInvalid);
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
MPSGraphTensor* trunc_tensor(MPSGraph* mpsGraph, MPSGraphTensor* inputTensor);
MPSGraphTensor* convertNHWCtoNCHW(MPSGraph *mpsGraph, MPSGraphTensor* tensor);
MPSGraphTensor* castMPSTensor(MPSGraph *mpsGraph, MPSGraphTensor* tensor, ScalarType toType);
MPSGraphTensor* castMPSTensor(MPSGraph *mpsGraph, MPSGraphTensor* tensor, MPSDataType toType);
MPSGraphTensorData *getMPSGraphTensorData(MPSGraph* mpsGraph, MPSStream* mpsStream, const Tensor& tensor);
MPSGraphTensorData* getMPSGraphTensorFromScalar(MPSStream* mpsStream, MPSScalar& scalar);

MPSGraph* make_mps_graph();
void printTensorNDArray(const Tensor& t);
MPSNDArray* ndArrayFromTensor(const Tensor& tensor, MPSShape *shape, MPSDataType mpsType);

MPSGraphTensor* mpsGraphUnrankedPlaceHolder(MPSGraph *mpsGraph, MPSDataType dataType);
MPSGraphTensor* mpsGraphRankedPlaceHolder(MPSGraph *mpsGraph, MPSDataType dataType, MPSShape* mpsShape);
MPSGraphTensor* mpsGraphRankedPlaceHolder(MPSGraph *mpsGraph, const Tensor& tensor);
MPSGraphTensor* mpsGraphScalarPlaceHolder(MPSGraph *mpsGraph, MPSDataType dataType);
MPSGraphTensor* mpsGraphScalarPlaceHolder(MPSGraph *mpsGraph, const Scalar& scalar);

string get_mem_format_string(c10::MemoryFormat memory_format);

using MPSCacheKey = uint64_t;

// derive this class to cache a graph and its inputs/outputs
// can be used to store any NSObject
struct MPSCachedGraph
{
  MPSCachedGraph(NSObject *object) : _object([object retain]) {}
  virtual ~MPSCachedGraph() {
   [_object release];
   _object = nullptr;
  }

  template<typename T>
  inline T* as() {
    return static_cast<T*>(this);
  }

  MPSGraph *graph() const { return (MPSGraph *)_object; }
  NSObject *object() const { return _object; }
private:
  NSObject *_object = nullptr;
};

struct MPSUnaryCachedGraph : public MPSCachedGraph
{
  MPSUnaryCachedGraph(MPSGraph *graph) : MPSCachedGraph(graph) {}
  MPSGraphTensor *inputTensor_ = nil;
  MPSGraphTensor *outputTensor_ = nil;
};

struct MPSUnaryGradCachedGraph : public MPSCachedGraph
{
  MPSUnaryGradCachedGraph(MPSGraph *graph) : MPSCachedGraph(graph) {}
  MPSGraphTensor *gradOutputTensor_ = nil;
  MPSGraphTensor *inputTensor_ = nil;
  MPSGraphTensor *outputTensor_ = nil; // some backward input is actually the forward's output
  MPSGraphTensor *gradInputTensor_ = nil;
};

struct MPSBinaryCachedGraph : public MPSCachedGraph
{
  MPSBinaryCachedGraph(MPSGraph *graph) : MPSCachedGraph(graph) {}
  MPSGraphTensor *inputTensor_ = nil;
  MPSGraphTensor *otherTensor_ = nil;
  MPSGraphTensor *outputTensor_ = nil;
};

struct MPSBinaryGradCachedGraph : public MPSCachedGraph
{
  MPSBinaryGradCachedGraph(MPSGraph *graph) : MPSCachedGraph(graph) {}
  MPSGraphTensor *gradOutputTensor_ = nil;
  MPSGraphTensor *inputTensor_ = nil;
  MPSGraphTensor *otherTensor_ = nil;
  MPSGraphTensor *gradInputTensor_ = nil;
};

// TODO: Improve the overall design of MPSGraphCache.
// https://github.com/pytorch/pytorch/issues/77176
// Cache holding various keys mapped to graphs
struct MPSGraphCache
{
  typedef MPSCachedGraph * (^CreateCachedGraphBlock)();

  struct CacheEntry {
    CacheEntry(const std::string& key, MPSCachedGraph *cachedGraph) : cachedGraph_(cachedGraph), key_(key) {}
    MPSCachedGraph* cachedGraph_ = nullptr;
    std::string key_;
  };

 public:

  static MPSGraphCache* getInstance() {
    if(_instance_cache == nullptr) {
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

    dispatch_sync(serialQueue_, ^() {

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

  template<typename T>
  inline T* CreateCachedGraphAs(const std::string& key, CreateCachedGraphBlock createCacheBlock) {
    return static_cast<T *>(CreateCachedGraph(key, createCacheBlock));
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

  template<typename T>
  inline T* LookUpAs(const std::string& key) const {
    return static_cast<T *>(LookUp(key));
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
template<typename T>
inline T* LookUpOrCreateCachedGraph(const std::string& key, std::function<void(MPSGraph*, T*)> instantiate) {
  auto cache_ = MPSGraphCache::getInstance();
  if (auto rc  = cache_->LookUpAs<T>(key)) {
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

#define MPS_CHECK_INT64_OP_SUPPORTED(input_tensor, mac_os_13_3_plus, op_name)                                           \
  if (!mac_os_13_3_plus && input_tensor.scalar_type() == kLong) {                                                       \
     TORCH_WARN_ONCE("MPS: no support for int64 for ", op_name,                                                         \
     ", downcasting to a smaller data type (int32/float32). Native support for int64 has been added in macOS 13.3.");   \
  }

} // namespace mps
} // namespace native
} // namespace at
