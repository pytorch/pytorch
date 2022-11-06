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

void runMPSGraph(MPSStream* mpsStream, MPSGraph* mpsGraph, NSDictionary* feeds, NSDictionary* results) {
  mpsStream->executeMPSGraph(mpsGraph, feeds, results, SyncType::COMMIT_ADAPTIVE);
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
      return "Float32";
    case ScalarType::Half:
      return "Float16";
    case ScalarType::Int:
      return "Int32";
    case ScalarType::Long:
      return "Int64";
    case ScalarType::Short:
      return "Int16";
    case ScalarType::Char:
      return "Int8";
    case ScalarType::Byte:
      return "UInt8";
    case ScalarType::Bool:
      return "Bool";
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
          str += (use_scalar_value ? std::to_string(tensor.item().to<double>()) : "Scalar");
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

MPSShape* getMPSShape(const Tensor& t) {
  return getMPSShape(t.sizes());
}

MPSShape* getMPSShape(c10::MaybeOwned<Tensor> t) {
  const Tensor& t_ = *t;
  return getMPSShape(t_);
}

MPSShape* getMPSShape(IntArrayRef sizes) {
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
  if (!t.is_mps()) return;
  if(t.numel() == 0) return;
  // Get shape and data type
  auto selfShape = getMPSShape(t);
  auto selfDType = getMPSDataType(t.scalar_type());

  // Initialize data
  id<MTLBuffer> selfBuf = getMTLBufferStorage(t);
  MPSGraphTensorData* tdata = [[[MPSGraphTensorData alloc] initWithMTLBuffer:selfBuf
                                                            shape:selfShape
                                                         dataType:selfDType] autorelease];
  C10_CLANG_DIAGNOSTIC_PUSH()
  #if C10_CLANG_HAS_WARNING("-Wobjc-method-access")
  C10_CLANG_DIAGNOSTIC_IGNORE("-Wobjc-method-access")
  #endif
  [tdata printNDArray];
  C10_CLANG_DIAGNOSTIC_POP()
}

Placeholder::Placeholder(MPSGraphTensor* mpsGraphTensor, const Tensor& src, MPSShape *mpsShape) : _tensor(src)
{
  TORCH_CHECK(src.is_mps(), "Placeholder storage has not been allocated on MPS device!");
  // extract the pointer to MTLBuffer from the Tensor's storage
  id<MTLBuffer> srcBuf = getMTLBufferStorage(src);
  // a view tensor could be contiguous (e.g., slice ops) or non-contiguous (e.g., transpose())
  if (src.is_view() || !src.is_contiguous()) {
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

  const MPSDataType mpsDataType = _tensor.dim() == 0 ? getMPSScalarType(_tensor.scalar_type()) : getMPSDataType(_tensor.scalar_type());
  if (!mpsShape)
    mpsShape = getMPSShape(_tensor);

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
    id<MTLBuffer> buf = getMTLBufferStorage(tensor);
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

MPSScalar getMPSScalar(const Scalar& scalar, ScalarType type) {
  switch (type) {
    case ScalarType::Double:
    case ScalarType::Float: return {.value.f = scalar.to<float>()   , .size = sizeof(float)  , .type = type};
    case ScalarType::Half:  return {.value.h = scalar.to<at::Half>(), .size = sizeof(short)  , .type = type};
    case ScalarType::Long:  return {.value.i = scalar.to<int64_t>() , .size = sizeof(int64_t), .type = type};
    case ScalarType::Int:   return {.value.i = scalar.to<int32_t>() , .size = sizeof(int32_t), .type = type};
    case ScalarType::Short: return {.value.i = scalar.to<int16_t>() , .size = sizeof(int16_t), .type = type};
    case ScalarType::Char:  return {.value.i = scalar.to<int8_t>()  , .size = sizeof(int8_t) , .type = type};
    case ScalarType::Byte:  return {.value.i = scalar.to<uint8_t>() , .size = sizeof(uint8_t), .type = type};
    case ScalarType::Bool:  return {.value.b = scalar.to<bool>()    , .size = sizeof(bool)   , .type = type};
    default:
      TORCH_INTERNAL_ASSERT(false, "Unsupported scalar type '", type, "' on MPS backend.");
  }
}

MPSGraphTensorData* getMPSGraphTensorFromScalar(MPSStream* mpsStream, MPSScalar& scalar) {
  MPSGraphTensorData *result = nullptr;
  // Scalar pools are only supported on devices with unified memory
  if (mpsStream->device().hasUnifiedMemory) {
    scalar.buffer = at::mps::allocate_scalar_buffer(&scalar.value, scalar.size);
    result = [[[MPSGraphTensorData alloc] initWithMTLBuffer: scalar.getMTLBuffer()
                                                      shape: @[@1]
                                                   dataType: getMPSScalarType(scalar.type)] autorelease];
  } else {
    MPSNDArrayDescriptor *tensorDesc = [MPSNDArrayDescriptor descriptorWithDataType:getMPSScalarType(scalar.type) shape:@[@1]];
    MPSNDArray *tensorNDArray = [[[MPSNDArray alloc] initWithDevice:mpsStream->device() descriptor:tensorDesc] autorelease];
    [tensorNDArray writeBytes:&scalar.value strideBytes:nil];
    result = [[[MPSGraphTensorData alloc] initWithMPSNDArray:tensorNDArray] autorelease];
  }
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

MPSGraphTensor* mpsGraphScalarPlaceHolder(MPSGraph *mpsGraph, MPSDataType dataType) {
    return [mpsGraph placeholderWithShape:@[@1]
                                 dataType:dataType
                                     name:nil];
}

MPSGraphTensor* mpsGraphScalarPlaceHolder(MPSGraph *mpsGraph, const Scalar& scalar) {
    return [mpsGraph placeholderWithShape:@[@1]
                                 dataType:getMPSScalarType(scalar.type())
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