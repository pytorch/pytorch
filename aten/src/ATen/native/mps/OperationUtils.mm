//  Copyright © 2022 Apple Inc.

#include <ATen/native/mps/OperationUtils.h>
#include <ATen/mps/MPSAllocator.h>

namespace at {
namespace native {
namespace mps {

uint64_t MPSGeneratorImpl::seed() {
  auto random = c10::detail::getNonDeterministicRandom(true);
  this->set_current_seed(random);
  return random;
}
uint64_t MPSGeneratorImpl::current_seed() const {
  return seed_;
}

void MPSGeneratorImpl::set_current_seed(uint64_t seed) {
  seed_ = seed;
}

MPSGeneratorImpl::MPSGeneratorImpl(DeviceIndex device_index)
  : c10::GeneratorImpl{Device(DeviceType::MPS, device_index),
              DispatchKeySet(c10::DispatchKey::MPS)} {
}

const Generator& getDefaultMPSGenerator() {
  static auto gen = make_generator<MPSGeneratorImpl>(0);
  gen.seed();
  return gen;
}
DeviceType MPSGeneratorImpl::device_type() {
  return DeviceType::MPS;
}
c10::intrusive_ptr<c10::TensorImpl> MPSGeneratorImpl::get_state() const {
  static const size_t seed_size = sizeof(uint64_t);
  static const size_t offset_size = sizeof(int64_t);
  static const size_t total_size = seed_size + offset_size;

  auto state_tensor = at::detail::empty_cpu({(int64_t)total_size}, ScalarType::Byte, c10::nullopt, c10::nullopt, c10::nullopt, c10::nullopt);

  return state_tensor.getIntrusivePtr();
}

void MPSGeneratorImpl::set_state(const c10::TensorImpl& new_state) {
  static const size_t seed_size = sizeof(uint64_t);

  detail::check_rng_state(new_state);

  uint64_t input_seed;
  auto new_rng_state = new_state.data<uint8_t>();
  memcpy(&input_seed, new_rng_state, seed_size);
  this->set_current_seed(input_seed);
}

MPSGeneratorImpl* MPSGeneratorImpl::clone_impl() const {
  auto gen = new MPSGeneratorImpl(0);
  gen->set_current_seed(this->seed_);
  return gen;
}

std::string getStridedKey(const Tensor& self, const IntArrayRef sz,
                          const IntArrayRef strides, int64_t offset) {
  // TODO: move storage_offset to a PlaceholderTensor and strides to a
  // tensor too, to avoid too many cache entries.
  return std::to_string((uintptr_t)self.storage().data()) +
              ":" + mps::getArrayRefString(sz) +
              ":" + mps::getArrayRefString(strides) +
              ":" + std::to_string(offset) +
              ":" + getMPSTypeString(self.scalar_type());
}

void runMPSGraph(
    MPSStream* mpsStream,
    MPSGraph* mpsGraph,
    NSDictionary* feeds,
    NSDictionary* results) {
  dispatch_sync(mpsStream->queue(), ^() {
    @autoreleasepool {
      mpsStream->commit(true);
      id<MTLCommandQueue> commandQueue = mpsStream->commandQueue();
      MPSGraphExecutionDescriptor *executionDescriptor = [[MPSGraphExecutionDescriptor new] autorelease];

      executionDescriptor.completionHandler = ^(NSDictionary<MPSGraphTensor *,
                                                MPSGraphTensorData *> * resultsDictionary,
                                                NSError * _Nullable error) {
      };

      [mpsGraph runAsyncWithMTLCommandQueue:commandQueue
                                feeds:feeds
                     targetOperations:nil
                    resultsDictionary:results
                  executionDescriptor:executionDescriptor];

    }
  });
}

MPSDataType getMPSDataType(ScalarType scalar_type) {
  switch (scalar_type) {
    case ScalarType::Float:
      return MPSDataTypeFloat32;
    case ScalarType::Half:
      return MPSDataTypeFloat16;
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
      TORCH_CHECK_TYPE(false, "Cannot convert a float64 Tensor to MPS as the MPS framework doesn't support float64. "
                       "Please use float32 instead.")
    default:
      TORCH_CHECK_TYPE(false, "Trying to convert ", scalar_type, " to the MPS backend but it does not have support for that dtype.")
  }
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
    default:
      TORCH_CHECK_TYPE(false, "Trying to convert ", scalar_type, " to the MPS backend but it does not have support for that dtype.")
  }
}

std::string getMPSTypeString(ScalarType scalar_type) {
  switch (scalar_type) {
    case ScalarType::Double:
    case ScalarType::Float:
      return "MPSDataTypeFloat32";
    case ScalarType::Half:
      return "MPSDataTypeFloat16";
    case ScalarType::Int:
      return "MPSDataTypeInt32";
    case ScalarType::Long:
      return "MPSDataTypeInt64";
    case ScalarType::Short:
      return "MPSDataTypeInt16";
    case ScalarType::Byte:
      return "MPSDataTypeInt8";
    case ScalarType::Bool:
      return "MPSDataTypeBool";
    default:
      return "Undefined";
  }
}

std::string getMPSShapeString(MPSShape* shape) {
    std::string str;
    for(NSNumber *elem in shape) {
        str += std::to_string(elem.unsignedLongValue) + ",";
    }
    return str;
}

std::string getArrayRefString(const IntArrayRef s) {
  std::stringstream ss;
  std::copy(s.begin(), s.end(), std::ostream_iterator<int>(ss, ","));
  return ss.str();
}

std::string getTensorsStringKey(const TensorList& tensors, bool use_scalar_value) {
    std::string str;
    // The key format per tensor would look like ":MPSDataTypeFloat32[1,1,1,10]:"
    for (const Tensor& tensor: tensors) {
      str += ":";
      if (tensor.defined()) {
        str += getMPSTypeString(tensor.scalar_type()) + "[";
        // if tensor is a scalar
        if (tensor.dim() == 0) {
          str += (use_scalar_value ? std::to_string(getMPSScalarValue(tensor)) : "Scalar");
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

double getMPSScalarValue(const Tensor& t) {
  assert (t.dim() == 0);  // only applicable for scalar types
  auto other_value = t.item();
  return other_value.to<double>();
}

MPSShape* getMPSShape(const Tensor& t) {
  const int sz = t.dim();
  const int sz_ = (sz > 0) ? sz : 1;

  NSNumber* numbers[sz_];

  for (int i = 0; i < sz_; i++)
  {
    NSInteger sz_i = (i < sz) ? t.size(i) : 1;

    NSNumber* number = [NSNumber numberWithInt:sz_i];
    numbers[i] = number;
  }
  return [NSArray arrayWithObjects:numbers count:sz_];
}

MPSShape* getMPSShape(c10::MaybeOwned<Tensor> t) {
  const Tensor& t_ = *t;
  return getMPSShape(t_);
}

MPSShape* getMPSShape(IntArrayRef sizes) {
  const int sz = sizes.size();
  const int sz_ = (sz > 0) ? sz : 1;

  NSNumber* numbers[sz_];

  for (int i = 0; i < sz_; i++)
  {
    NSInteger sz_i = (i < sz) ? sizes[i] : 1;

    NSNumber* number = [NSNumber numberWithInt:sz_i];
    numbers[i] = number;
  }
  return [NSArray arrayWithObjects:numbers count:sz_];
}

void printTensorNDArray(const Tensor& t) {
  if (!t.is_mps()) return;
  if(t.numel() == 0) return;
  // Get shape and data type
  auto selfShape = getMPSShape(t);
  auto selfDType = getMPSDataType(t.scalar_type());

  // Initialize data
  id<MTLBuffer> selfBuf = __builtin_bit_cast(id<MTLBuffer>, t.storage().data());
  MPSGraphTensorData* tdata = [[[MPSGraphTensorData alloc] initWithMTLBuffer:selfBuf
                                                            shape:selfShape
                                                         dataType:selfDType] autorelease];
  [tdata printNDArray];
}

MPSCachedGraph* _getCachedGraph(const at::Tensor& src) {
  MPSGraphCache* cache_ = MPSGraphCache::getInstance();
  string key = getStridedKey(src, src.sizes(), src.strides(), src.storage_offset());
  MPSCachedGraph* cachedGraph = cache_->LookUp(key);

  return cachedGraph;
}

id<MTLBuffer> _gatherViewTensor(const at::Tensor& src, id<MTLBuffer> sourceBuffer, MPSCachedGraph* mpsCachedGraph, Tensor& output) {
  TORCH_CHECK(mpsCachedGraph != nil);

  MPSStream* stream = getCurrentMPSStream();

  struct CachedGraph : public MPSCachedGraph
  {
    CachedGraph(MPSGraph *graph) : MPSCachedGraph(graph) {}
    MPSGraphTensor* inputTensor_ = nil;
    MPSGraphTensor* outputTensor_ = nil;
  };

  CachedGraph* cachedGraph = static_cast<CachedGraph *>(mpsCachedGraph);

  @autoreleasepool {
    MPSGraphTensor* inputTensor = cachedGraph->inputTensor_;
    MPSGraphTensorData* inputTensorData = [[[MPSGraphTensorData alloc] initWithMTLBuffer: sourceBuffer
                                                                        shape: [inputTensor shape]
                                                                        dataType: [inputTensor dataType]] autorelease];
    id<MTLBuffer> resultBuffer = __builtin_bit_cast(id<MTLBuffer>, output.storage().data());
    MPSGraphTensorData* outputTensorData = [[[MPSGraphTensorData alloc] initWithMTLBuffer: resultBuffer
                                                                        shape: getMPSShape(src.sizes())
                                                                        dataType: getMPSDataType(src.scalar_type())] autorelease];
    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds = @{
      inputTensor : inputTensorData
    };

    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results = @{
      cachedGraph->outputTensor_ : outputTensorData
    };

    runMPSGraph(stream, cachedGraph->graph(), feeds, results);
    return resultBuffer;
  }
}

id<MTLBuffer> gatherViewTensor(const at::Tensor& src, id<MTLBuffer> sourceBuffer) {
  MPSCachedGraph* mpsCachedGraph = _getCachedGraph(src);
  if (mpsCachedGraph) {
    Tensor output = at::native::empty_mps(
                    src.sizes(),
                    src.scalar_type(),
                    c10::nullopt,
                    kMPS,
                    c10::nullopt,
                    c10::nullopt);

    _gatherViewTensor(src, sourceBuffer, mpsCachedGraph, output);
    return __builtin_bit_cast(id<MTLBuffer>, output.storage().data());
  }

  return nil;
}

id<MTLBuffer> gatherViewTensorWithAllocatedMem(const at::Tensor& src, id<MTLBuffer> sourceBuffer, Tensor& output, MPSCachedGraph* mpsCachedGraph) {
  TORCH_CHECK(mpsCachedGraph != nil);

  _gatherViewTensor(src, sourceBuffer, mpsCachedGraph, output);
  return __builtin_bit_cast(id<MTLBuffer>, output.storage().data());
}

Placeholder::Placeholder(MPSGraphTensor* mpsGraphTensor, const Tensor& src, MPSShape *mpsShape)
{
  Tensor src_ = src;
  TORCH_CHECK(src_.is_mps(), "Placeholder storage has not been allocated on MPS device!");
    // extract the pointer to MTLBuffer from the Tensor's storage
  id<MTLBuffer> srcBuf = __builtin_bit_cast(id<MTLBuffer>, src.storage().data());
  if (src.is_view()) {
    MPSCachedGraph* cachedGraph = _getCachedGraph(src);
    if (cachedGraph) {
      allocateViewTensor(src);
      id<MTLBuffer> gatherTensor = gatherViewTensorWithAllocatedMem(src, srcBuf, _viewOutput, cachedGraph);
      if (gatherTensor) {
        srcBuf = gatherTensor;
      }
    } else {
      src_ = src.contiguous();
      srcBuf = __builtin_bit_cast(id<MTLBuffer>, src_.storage().data());
    }
  }
  // tensor.numel() could be zero, but tensor is valid as long as the buffer size is non-zero.
  // if buffer size is zero in here, it's not a user error. It could be a missing check for
  // tensor.numel() == 0 in our internal implementations of ops.
  TORCH_INTERNAL_ASSERT([srcBuf length] > 0, "Placeholder tensor is empty!");

  const MPSDataType mpsDataType = src_.dim() == 0 ? getMPSScalarType(src_.scalar_type()) : getMPSDataType(src_.scalar_type());
  if (!mpsShape)
    mpsShape = getMPSShape(src_);

  _value = [[[MPSGraphTensorData alloc] initWithMTLBuffer:srcBuf
                                                    shape:mpsShape
                                                 dataType:mpsDataType] autorelease];
  TORCH_INTERNAL_ASSERT(_value);
  _placeholder = mpsGraphTensor;
}

MPSGraphTensorData *getMPSGraphTensorData(MPSGraph* mpsGraph,
                                          MPSStream* mpsStream,
                                          const Tensor& tensor) {
  auto mpsShape = getMPSShape(tensor);
  auto dataType = getMPSDataType(tensor.scalar_type());

  MPSGraphTensorData *result = nil;
  if (tensor.numel() > 0) {
    id<MTLBuffer> buf = __builtin_bit_cast(id<MTLBuffer>, tensor.storage().data());
    result = [[[MPSGraphTensorData alloc] initWithMTLBuffer:buf
                                                    shape:mpsShape
                                                 dataType:dataType]
                                                 autorelease];
  } else {
    // create empty NDArray
    MPSNDArrayDescriptor *desc = [MPSNDArrayDescriptor descriptorWithDataType:dataType
                                                                        shape:mpsShape];
    MPSNDArray *emptyArray = [[[MPSNDArray alloc]
                              initWithDevice:mpsStream->device() descriptor:desc] autorelease];
    result = [[[MPSGraphTensorData alloc] initWithMPSNDArray:emptyArray] autorelease];
  }
  assert(result);
  return result;
}

MPSGraphTensorData* getMPSGraphTensorFromScalar(MPSStream* mpsStream, const Scalar& scalar, MPSDataType dataType) {
  union {
    float f; // MPS doesn't support 'double'
    at::Half h;
    int64_t i;
    bool b;
  } v;
  switch (dataType) {
    case MPSDataTypeFloat32:
      v.f = scalar.to<float>();
      break;
    case MPSDataTypeFloat16:
      v.h = scalar.to<at::Half>();
      break;
    case MPSDataTypeInt64:
      v.i = scalar.to<int64_t>();
      break;
    case MPSDataTypeInt32:
      v.i = scalar.to<int32_t>();
      break;
    case MPSDataTypeInt16:
      v.i = scalar.to<int16_t>();
      break;
    case MPSDataTypeInt8:
      v.i = scalar.to<int8_t>();
      break;
    case MPSDataTypeBool:
      v.b = scalar.to<bool>();
      break;
    default:
      TORCH_INTERNAL_ASSERT(false, "Unsupported scalar type on MPS backend.")
  }

  MPSNDArrayDescriptor *tensorDesc = [MPSNDArrayDescriptor descriptorWithDataType:dataType shape:@[@1]];
  MPSNDArray *tensorNDArray = [[[MPSNDArray alloc] initWithDevice:mpsStream->device() descriptor:tensorDesc] autorelease];
  [tensorNDArray writeBytes:&v strideBytes:nil];
  MPSGraphTensorData* result = [[[MPSGraphTensorData alloc] initWithMPSNDArray:tensorNDArray] autorelease];
  return result;
}

void resize_tensor(Tensor* output) {
  output->resize_(output->sizes());
}

MPSGraph* make_mps_graph() {
  MPSGraph* mpsGraph = [[MPSGraph new] autorelease];
  mpsGraph.options = MPSGraphOptionsNone;
  return mpsGraph;
}

MPSGraphTensor* mpsGraphUnrankedPlaceHolder(MPSGraph *mpsGraph, MPSDataType dataType) {
  return [mpsGraph placeholderWithShape:nil
                               dataType:dataType
                                   name:nil];
}

MPSGraphTensor* mpsGraphRankedPlaceHolder(MPSGraph *mpsGraph, MPSDataType dataType, MPSShape* mpsShape) {
  return [mpsGraph placeholderWithShape:mpsShape
                               dataType:dataType
                                   name:nil];
}

MPSGraphTensor* mpsGraphRankedPlaceHolder(MPSGraph *mpsGraph, const Tensor& tensor) {
    return [mpsGraph placeholderWithShape:getMPSShape(tensor)
                                 dataType:getMPSScalarType(tensor.scalar_type())
                                     name:nil];
}

// this is meant to suppress the availability warning on castTensor
// we pass ScalarType instead of MPSDataType to handle MPSDataTypeBoolean's availability too
MPSGraphTensor* castMPSTensor(MPSGraph *mpsGraph, MPSGraphTensor* tensor, ScalarType toType) {
  return [mpsGraph castTensor:tensor toType:getMPSScalarType(toType) name:@"castTensor"];
}

string get_mem_format_string(c10::MemoryFormat memory_format) {
  string mem_format_key;
  switch(memory_format) {
    case at::MemoryFormat::Contiguous:
      mem_format_key = "Contiguous";
      break;
    case at::MemoryFormat::ChannelsLast:
      mem_format_key = "ChannelsLast";
      break;
    default:
      assert(0 && "Invalid memory format\n");
  }

  return mem_format_key;
}

MPSGraphCache* MPSGraphCache::_instance_cache = nullptr;

class MPSGraphCacheCallback : public IMpsAllocatorCallback {
public:
  MPSGraphCacheCallback() : graph_cache(MPSGraphCache::getInstance()) { }

  void executeMPSAllocatorCallback(void* ptr, EventType event) override { }
private:
  MPSGraphCache* graph_cache;
};

REGISTER_MPS_ALLOCATOR_CALLBACK("mps_graph_cache_callback", MPSGraphCacheCallback);

} // namespace mps
} // namespace native
} // namespace at
