//  Copyright © 2022 Apple Inc.

#include <ATen/ATen.h>
#include <ATen/Tensor.h>
#include <ATen/Utils.h>
#include <ATen/mps/MPSStream.h>
#include <ATen/native/mps/TensorFactory.h>
#include <c10/core/ScalarType.h>
#include <torch/library.h>
#include <unordered_map>

#ifdef __OBJC__
#include <MetalPerformanceShaders/MetalPerformanceShaders.h>
#endif

using namespace at::mps;

namespace at {
namespace native {
namespace mps {

struct TORCH_CUDA_CPP_API MPSGeneratorImpl : public c10::GeneratorImpl {
  MPSGeneratorImpl(DeviceIndex device_index = -1);
  ~MPSGeneratorImpl() = default;

  void set_current_seed(uint64_t seed) override;
  uint64_t current_seed() const override;
  uint64_t seed() override;
  void set_state(const c10::TensorImpl& new_state) override;
  c10::intrusive_ptr<c10::TensorImpl> get_state() const override;
  static DeviceType device_type();

private:
  MPSGeneratorImpl* clone_impl() const override;
  uint64_t seed_ = default_rng_seed_val;
};

const Generator& getDefaultMPSGenerator();

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
MPSDataType getMPSScalarType(ScalarType scalar_type);
MPSScalar   getMPSScalar(const Scalar& scalar, ScalarType type);
std::string getMPSTypeString(ScalarType scalar_type);
std::string getMPSShapeString(MPSShape* shape);
std::string getTensorsStringKey(const TensorList& tensors, bool use_scalar_value = false);
std::string getArrayRefString(const IntArrayRef s);
// use has_storage() on the returned tensor to determine if src actually is a view
Tensor gatherViewTensor(const at::Tensor& src, at::Tensor& dst);
Tensor& scatterViewTensor(const at::Tensor& src, at::Tensor& output);

MPSShape* getMPSShape(const Tensor& t);
MPSShape* getMPSShape(IntArrayRef sizes);
MPSShape* getMPSShape(c10::MaybeOwned<Tensor> t);

static inline id<MTLBuffer> getMTLBufferStorage(const at::Tensor& tensor) {
  return __builtin_bit_cast(id<MTLBuffer>, tensor.storage().data());
}

class Placeholder {
 public:
  Placeholder() : _placeholder(nullptr), _value(nullptr), _tensor(Tensor()) {}
  Placeholder(MPSGraphTensor* mpsGraphTensor) : _placeholder(mpsGraphTensor), _value(nullptr), _tensor(Tensor()) {}
  Placeholder(MPSGraphTensor* mpsGraphTensor, const Tensor& self, MPSShape *mpsShape = nullptr);
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
MPSGraphTensor* trunc_tensor(MPSGraph* mpsGraph, MPSGraphTensor* inputTensor);
MPSGraphTensor* castMPSTensor(MPSGraph *mpsGraph, MPSGraphTensor* tensor, ScalarType toType);
MPSGraphTensorData *getMPSGraphTensorData(MPSGraph* mpsGraph, MPSStream* mpsStream, const Tensor& tensor);
MPSGraphTensorData* getMPSGraphTensorFromScalar(MPSStream* mpsStream, MPSScalar& scalar);

MPSGraph* make_mps_graph();
void printTensorNDArray(const Tensor& t);

MPSGraphTensor* mpsGraphUnrankedPlaceHolder(MPSGraph *mpsGraph, MPSDataType dataType);
MPSGraphTensor* mpsGraphRankedPlaceHolder(MPSGraph *mpsGraph, MPSDataType dataType, MPSShape* mpsShape);
MPSGraphTensor* mpsGraphRankedPlaceHolder(MPSGraph *mpsGraph, const Tensor& tensor);
MPSGraphTensor* mpsGraphScalarPlaceHolder(MPSGraph *mpsGraph, const Scalar& scalar);

string get_mem_format_string(c10::MemoryFormat memory_format);

using MPSCacheKey = uint64_t;

// derive this class to cache a graph and its inputs/ouputs
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


// TODO: Improve the overall design of MPSGraphCache.
// https://github.com/pytorch/pytorch/issues/77176
// Cache holding various keys mapped to graphs
struct MPSGraphCache
{
  typedef MPSCachedGraph * (^CreateCachedGraphBlock)();

  struct CacheEntry {
    CacheEntry(std::string key, MPSCachedGraph *cachedGraph) : cachedGraph_(cachedGraph), key_(key) {}
    MPSCachedGraph* cachedGraph_ = nullptr;
    std::string key_ = nullptr;
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

    for (auto i : cache_) {
      delete i.second.cachedGraph_;
    }
  }

  // Disallow the copy constructor and operator= functions
  MPSGraphCache(const MPSGraphCache&) = delete;
  void operator=(const MPSGraphCache&) = delete;

  MPSCachedGraph* CreateCachedGraph(const std::string& key, CreateCachedGraphBlock createCacheBlock, void* view_ptr = nullptr) {

    __block MPSCachedGraph * result = nil;

    MPSCacheKey hash = std::hash<std::string>{}(key);

    dispatch_sync(serialQueue_, ^() {

      // verify the cached entry doesn't already exist
      if (cache_.count(hash) != 0) {
        auto& entry = cache_.at(hash);
        TORCH_INTERNAL_ASSERT_DEBUG_ONLY(key == entry.key_, "Key collision in the MPS cached graph!\n");
        result = entry.cachedGraph_;
      }
      else {
        result = createCacheBlock();
        CacheEntry entry(key, result);
        cache_.emplace(hash, entry);
        if (view_ptr) {
          views_list.insert(std::make_pair(view_ptr, hash));
        }
      }
    });
    return result;
  }

  template<typename T>
  inline T* CreateCachedGraphAs(const std::string& key, CreateCachedGraphBlock createCacheBlock, void* view_ptr = nullptr) {
    return static_cast<T *>(CreateCachedGraph(key, createCacheBlock, view_ptr));
  }

  MPSCachedGraph* LookUp(const std::string& key) const {

    __block MPSCachedGraph* result = nullptr;

    MPSCacheKey hash = std::hash<std::string>{}(key);

    dispatch_sync(serialQueue_, ^() {

      if (cache_.count(hash) != 0) {
        auto& entry = cache_.at(hash);
        TORCH_INTERNAL_ASSERT_DEBUG_ONLY(key == entry.key_, "Key collision in the MPS cached graph!\n");
        result = entry.cachedGraph_;
      }
    });
    return result;
  }

  template<typename T>
  inline T* LookUpAs(const std::string& key) const {
    return static_cast<T *>(LookUp(key));
  }

  void FindAndRemoveViewEntry(void* ptr) {
    // this may find multiple view entries with the same buffer pointers
    auto views_range = views_list.equal_range(ptr);
    if (views_range.first == views_range.second)
      return;
    for (auto view_it = views_range.first; view_it != views_range.second; ++view_it) {
      MPSCacheKey hash = view_it->second;
      // find the cache entry associated with the hash
      auto cache_it = cache_.find(hash);
      if (cache_it != cache_.end()) {
        cache_.erase(cache_it);
        delete cache_it->second.cachedGraph_;
      }
    }
    // this erase-by-key will remove all pairs in the list with the same key
    views_list.erase(ptr);
  }

 private:
  MPSGraphCache() {
    serialQueue_ = dispatch_queue_create("cache queue", DISPATCH_QUEUE_SERIAL);
  }

  static MPSGraphCache* _instance_cache;
  std::unordered_map<MPSCacheKey, CacheEntry> cache_;
  // list of buffers associated with view entries in the cache
  // note that multiple view cache entries could use the same buffer pointer
  std::unordered_multimap<void*, MPSCacheKey> views_list;
  dispatch_queue_t serialQueue_ = nullptr;

};

} // namespace mps
} // namespace native
} // namespace at
