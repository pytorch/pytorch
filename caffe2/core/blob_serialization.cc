#include "caffe2/core/blob_serialization.h"

#include <mutex>
#include <sstream>
#include <utility>

#include <c10/util/irange.h>
#include <c10/util/string_view.h>

#include "caffe2/core/blob.h"
#include "caffe2/core/common.h"
#include "caffe2/utils/proto_utils.h"
#ifdef USE_FBGEMM
#include "fbgemm/FbgemmConvert.h"
#endif

C10_DEFINE_int(
    caffe2_tensor_chunk_size,
    1000000,
    "Chunk size to split tensor data into");

C10_DEFINE_int(
    caffe2_max_tensor_serializer_threads,
    16,
    "Maximal number of threads that can be used for tensor serialization");

C10_DEFINE_bool(
    caffe2_serialize_fp16_as_bytes,
    false,
    "Serialize FLOAT16 tensors using byte_data field");

C10_DEFINE_bool(
    caffe2_serialize_using_bytes_as_holder,
    false,
    "Serialize BOOL, UINT8, INT8, UINT16, INT16, FLOAT16 tensors using byte_data field instead of int32");

namespace caffe2 {
namespace {

// This is a simplified copy of folly::Range.
// This is similar to c10::ArrayRef but it can point to non-const data.
template<typename Iter>
class Range {
 public:
  using value_type = typename std::remove_reference<
      typename std::iterator_traits<Iter>::reference>::type;

  Range(Iter b, Iter e) : begin_{b}, end_{e} {}
  Range(Iter b, size_t size) : begin_{b}, end_{b + size} {}

  CAFFE2_NODISCARD constexpr Iter data() const {
    return begin_;
  }
  CAFFE2_NODISCARD constexpr Iter begin() const {
    return begin_;
  }
  CAFFE2_NODISCARD constexpr Iter end() const {
    return end_;
  }
  CAFFE2_NODISCARD constexpr size_t size() const {
    return end_ - begin_;
  }

  value_type& operator[](size_t n) const {
    assert(n < size());
    return begin_[n];
  }

 private:
  Iter begin_;
  Iter end_;
};

/**
 * Return a mutable Range pointing to a portion of the tensor's data field.
 *
 * Returns a Range pointing to the elements starting at the specified start
 * index, and including the specified number of elements.
 */
template <typename T>
Range<T*> GetMutableTensorDataRange(
    Tensor& tensor,
    size_t start,
    size_t numElements) {
  CAFFE_ENFORCE(
      // NOLINTNEXTLINE(clang-diagnostic-sign-compare)
      start + numElements <= tensor.numel(),
      "Requested invalid mutable tensor range [",
      start,
      ", ",
      start + numElements,
      ") with total tensor size ",
      tensor.numel());
  return Range<T*>(tensor.template mutable_data<T>() + start, numElements);
}

template <typename T>
c10::ArrayRef<T> GetTensorDataRange(
    const Tensor& tensor,
    size_t start,
    size_t numElements) {
  CAFFE_ENFORCE(
      // NOLINTNEXTLINE(clang-diagnostic-sign-compare)
      start + numElements <= tensor.numel(),
      "Requested invalid tensor range [",
      start,
      ", ",
      start + numElements,
      ") with total tensor size ",
      tensor.numel());
  return c10::ArrayRef<T>(tensor.template data<T>() + start, numElements);
}

template <typename T>
bool EnableByteEncoding() {
  // if typeSize == 1, endianness does not matter. Else check for endianness.
  if (sizeof(T) > 1 && !kIsLittleEndian) {
    return false;
  }
  return FLAGS_caffe2_serialize_using_bytes_as_holder;
}

bool EnableByteEncodingFloat16() {
  if (!kIsLittleEndian) {
    return false;
  }
  // Check if special casing for float is enabled if
  // caffe2_serialize_using_bytes_as_holder is not enabled.
  return FLAGS_caffe2_serialize_using_bytes_as_holder ||
      FLAGS_caffe2_serialize_fp16_as_bytes;
}

size_t EstimatePerElementSize(
    const Tensor& tensor,
    const BlobSerializationOptions& options) {
  const TensorProto::DataType data_type = TypeMetaToDataType(tensor.dtype());
  switch (data_type) {
    case TensorProto_DataType_FLOAT:
#ifdef USE_FBGEMM
      if (options.float_format() ==
          BlobSerializationOptions_FloatFormat_FLOAT_BFLOAT16) {
        // Each element is serialized as a 2-byte bfloat16
        return sizeof(uint16_t);
      }
#endif
      return sizeof(float);
    case TensorProto_DataType_INT32:
      // protobuf will use varint encoding, so it won't be a fixed field width
      // per integer, and will use between 1 and 5 bytes.  Just return 4 bytes
      // as an estimate.  With randomized data the actual value may be higher
      // than this, since around half the numbers will have the high bit set and
      // would require 5 bytes to encode.
      return sizeof(int32_t);
    case TensorProto_DataType_INT64:
      // Same varint reasoning as for the INT32 case.
      return sizeof(int64_t);
    case TensorProto_DataType_STRING:
      // We unfortunately cannot estimate the size well for strings, without
      // knowing the individual element lengths.  Just return 50 bytes per
      // string as a guess.
      return 50;
    case TensorProto_DataType_BOOL:
      // Depending on EnableByteEncoding() this is either serialized in
      // byte_data or int32_data, but in either case it takes 1 byte per element
      // (since bool values will only take 1 byte when varint encoded in
      // int32_data).
      return 1;
    case TensorProto_DataType_UINT8:
      if (EnableByteEncoding<uint8_t>()) {
        return 1;
      } else {
        // Unfortunately when storing uint8_t values in int32_data any values
        // over 127 will require 2 bytes to store due to varint encoding.
        // With random data we would expect around 1.5 bytes per element.  Round
        // up to 2.
        return 2;
      }
    case TensorProto_DataType_INT8:
      if (EnableByteEncoding<int8_t>()) {
        return 1;
      } else {
        // Unfortunately when storing int8_t values in int32_data any negative
        // values will require 2 bytes to store due to varint encoding.  With
        // random data we would expect around 1.5 bytes per element.  Round up
        // to 2.
        return 2;
      }
    case TensorProto_DataType_UINT16:
      if (EnableByteEncoding<uint16_t>()) {
        return 2;
      } else {
        // With random data, varint encoding will end up requiring closer to 3
        // bytes per element.
        return 3;
      }
    case TensorProto_DataType_INT16:
      if (EnableByteEncoding<int16_t>()) {
        return 2;
      } else {
        // With random data, varint encoding will end up requiring closer to 3
        // bytes per element.
        return 3;
      }
    case TensorProto_DataType_FLOAT16:
      if (EnableByteEncodingFloat16()) {
        return 2;
      } else {
        // The data will be stored as uint16_t values in the int32_data.
        // Due to varint encoding many values may require 3 bytes.
        return 3;
      }
    case TensorProto_DataType_DOUBLE:
      return sizeof(double);
    case TensorProto_DataType_UNDEFINED:
      return tensor.itemsize();
    case TensorProto_DataType_BYTE:
    case TensorProto_DataType_ZERO_COLLISION_HASH:
    case TensorProto_DataType_REBATCHING_BUFFER:
      // These data types should never be hit during serialization
      LOG(ERROR) << "unexpected tensor data type during serialization size "
                    "estimation: "
                 << static_cast<int>(data_type);
      return 0;
  }

  LOG(ERROR) << "unknown tensor data type during serialization size "
                "estimation: "
             << static_cast<int>(data_type);
  return 0;
}

} // namespace

/**
 * @brief StringSerializer is the serializer for String.
 *
 * StringSerializer takes in a blob that contains a String, and serializes it
 * into a BlobProto protocol buffer.
 */
class StringSerializer : public BlobSerializerBase {
 public:
  StringSerializer() = default;
  ~StringSerializer() override = default;
  /**
   * Serializes a Blob. Note that this blob has to contain Tensor,
   * otherwise this function produces a fatal error.
   */
  void Serialize(
      const void* pointer,
      TypeMeta typeMeta,
      const string& name,
      SerializationAcceptor acceptor) override {
    CAFFE_ENFORCE(typeMeta.Match<std::string>());

    BlobProto blob_proto;
    blob_proto.set_name(name);
    blob_proto.set_type("std::string");
    blob_proto.set_content(*static_cast<const std::string*>(pointer));
    acceptor(name, SerializeBlobProtoAsString_EnforceCheck(blob_proto));
  }

  size_t EstimateSerializedBlobSize(
      const void* pointer,
      TypeMeta,
      c10::string_view name,
      const BlobSerializationOptions&) override {
    auto* str = static_cast<const std::string*>(pointer);
    // Add 20 for the "std::string" type field plus other overhead for the
    // BlobProto message serialization.
    return name.size() + str->size() + 20;
  }
};

/**
 * @brief StringDeserializer is the deserializer for Strings.
 *
 */
class StringDeserializer : public BlobDeserializerBase {
 public:
  void Deserialize(const BlobProto& proto, Blob* blob) override {
    *blob->GetMutable<std::string>() = proto.content();
  }
};

namespace {
void SerializeBlob(
    const void* pointer,
    TypeMeta typeMeta,
    const string& name,
    BlobSerializerBase::SerializationAcceptor acceptor,
    const BlobSerializationOptions& options) {
  std::unique_ptr<BlobSerializerBase> serializer(
      CreateSerializer(typeMeta.id()));
  CAFFE_ENFORCE(serializer, "No known serializer for ", typeMeta.name());
  serializer->SerializeWithOptions(pointer, typeMeta, name, std::move(acceptor), options);
}

std::string
SerializeBlob(const void* pointer, TypeMeta typeMeta, const string& name) {
  std::string data;
  BlobSerializerBase::SerializationAcceptor acceptor =
      [&data](const std::string&, const std::string& blob_str) {
        DCHECK(data.empty()); // should be called once with kNoChunking
        data = blob_str;
      };
  BlobSerializationOptions options;
  options.set_chunk_size(kNoChunking);
  SerializeBlob(pointer, typeMeta, name, acceptor, options);
  return data;
}
} // namespace

void SerializeBlob(
    const Blob& blob,
    const string& name,
    BlobSerializerBase::SerializationAcceptor acceptor,
    const BlobSerializationOptions& options) {
  SerializeBlob(blob.GetRaw(), blob.meta(), name, std::move(acceptor), options);
}

void SerializeBlob(
    const Blob& blob,
    const string& name,
    BlobSerializerBase::SerializationAcceptor acceptor) {
  BlobSerializationOptions options;
  SerializeBlob(blob.GetRaw(), blob.meta(), name, std::move(acceptor), options);
}

std::string SerializeBlob(const Blob& blob, const string& name) {
  return SerializeBlob(blob.GetRaw(), blob.meta(), name);
}

size_t EstimateSerializedBlobSize(
    const Blob& blob,
    c10::string_view name,
    const BlobSerializationOptions& options) {
  std::unique_ptr<BlobSerializerBase> serializer{
      CreateSerializer(blob.meta().id())};
  if (!serializer) {
    LOG(ERROR) << "No known serializer for " << blob.meta().name();
    return 0;
  }
  return serializer->EstimateSerializedBlobSize(
      blob.GetRaw(), blob.meta(), name, options);
}

void TensorSerializer::Serialize(
    const void* pointer,
    TypeMeta typeMeta,
    const string& name,
    BlobSerializerBase::SerializationAcceptor acceptor) {
  BlobSerializationOptions options;
  this->SerializeWithOptions(pointer, typeMeta, name, acceptor, options);
}

void TensorSerializer::SerializeWithOptions(
    const void* pointer,
    TypeMeta typeMeta,
    const string& name,
    BlobSerializerBase::SerializationAcceptor acceptor,
    const BlobSerializationOptions& options) {
  CAFFE_ENFORCE(typeMeta.Match<Tensor>());
  const auto& tensor = *static_cast<const Tensor*>(pointer);
  auto chunk_size = options.chunk_size();
  if (chunk_size == kNoChunking) {
    chunk_size = tensor.numel() + 1; // to account for empty tensors
  } else if (chunk_size == kDefaultChunkSize) {
    chunk_size = FLAGS_caffe2_tensor_chunk_size;
  }

  auto processChunk = [&](int64_t chunkStart) {
    BlobProto blob_proto;
    blob_proto.set_name(name);
    blob_proto.set_type(kTensorBlobType);
    TensorProto& proto = *blob_proto.mutable_tensor();
    proto.set_name(name);
    this->Serialize(
        tensor,
        name,
        blob_proto.mutable_tensor(),
        options,
        chunkStart,
        chunk_size);
    acceptor(
        c10::str(name, kChunkIdSeparator, chunkStart / chunk_size),
        SerializeBlobProtoAsString_EnforceCheck(blob_proto));
  };

#ifndef __ANDROID__
  // Poorman's IOBound ThreadPool
  SimpleQueue<size_t> chunkQueue;
  auto task = [&]() {
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    size_t chunkStart;
    while (chunkQueue.Pop(&chunkStart)) {
      processChunk(chunkStart);
    }
  };
  std::vector<std::future<void>> futures;
  if (tensor.numel() > chunk_size) {
    futures.reserve(FLAGS_caffe2_max_tensor_serializer_threads);
    for (const auto i : c10::irange(FLAGS_caffe2_max_tensor_serializer_threads)) {
      (void)i;
      futures.emplace_back(std::async(std::launch::async, task));
    }
  }
#endif

  VLOG(1) << "Serializing blob " << name;
  // Serialize whole vector. If vector is empty, it's shape still needs to be
  // serialized in empty proto
  for (size_t chunkBegin = 0;
       // NOLINTNEXTLINE(clang-diagnostic-sign-compare)
       chunkBegin < std::max(tensor.numel(), static_cast<int64_t>(1));
       chunkBegin += chunk_size) {
    VLOG(2) << "Starting a chunk at " << chunkBegin;
#ifndef __ANDROID__
    if (tensor.numel() > chunk_size) {
      chunkQueue.Push(chunkBegin);
    } else {
      // Sync mode for small tensors
      processChunk(chunkBegin);
    }
#else
    // Since Android does not have std::future, we will always do sync mode
    processChunk(chunkBegin);
#endif
  }

#ifndef __ANDROID__
  chunkQueue.NoMoreJobs();
  for (auto& fut : futures) {
    fut.get();
  }
#endif
}

size_t TensorSerializer::EstimateSerializedBlobSize(
      const void* pointer,
      TypeMeta typeMeta,
      c10::string_view name,
      const BlobSerializationOptions& options) {
  CAFFE_ENFORCE(typeMeta.Match<Tensor>());
  const auto& tensor = *static_cast<const Tensor*>(pointer);

  auto chunk_size = options.chunk_size();
  if (chunk_size == kNoChunking) {
    chunk_size = tensor.numel() + 1; // to account for empty tensors
  } else if (chunk_size == kDefaultChunkSize) {
    chunk_size = FLAGS_caffe2_tensor_chunk_size;
  }

  // There is a small amount of fixed overhead per chunk to serialize the
  // fixed TensorProto message data independent from the chunk contents.
  // This normally appears to be around 50 bytes.
  // The blob name is also written out in the BlobProto for each chunk.
  constexpr size_t protobuf_overhead_per_chunk = 50;
  size_t num_chunks = (tensor.numel() + (chunk_size - 1)) / chunk_size;
  size_t overhead = num_chunks * (name.size() + protobuf_overhead_per_chunk);

  return overhead + tensor.numel() * EstimatePerElementSize(tensor, options);
}

namespace {

template <typename T, typename S = T>
void SerializeUsingBytesOrInt32(
    bool enableByteEncoding,
    c10::ArrayRef<S> input,
    BaseContext& context,
    TensorProto& proto) {
  if (enableByteEncoding) {
    const auto bufSize = sizeof(T) * input.size();
    auto* byteData = reinterpret_cast<const uint8_t*>(input.data());
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
    unique_ptr<uint8_t[]> buffer(new uint8_t[bufSize]);
    context.template CopyToCPU<uint8_t>(bufSize, byteData, buffer.get());
    context.FinishDeviceComputation();
    proto.set_byte_data(buffer.get(), bufSize);
  } else {
    detail::CopyToProtoWithCast(
        input.size(),
        reinterpret_cast<const T*>(input.data()),
        proto.mutable_int32_data(),
        &context);
  }
}

/**
 * SerializeParams is just a helper class to consolidate the parameters
 * required for serializing tensor data so they can be passed around more
 * easily.
 *
 * It also contains some helper functions to perform some operations on the
 * parameters that are shared by multiple serialization functions.
 */
template<typename T>
struct SerializeParams {
  SerializeParams(
      c10::ArrayRef<T> in,
      TensorProto& proto,
      BaseContext& ctx,
      const BlobSerializationOptions& opts)
      : input{in}, tensor_proto{proto}, context{ctx}, options{opts} {}

  void SetDataFormat(TensorProto::SerializationFormat format) const {
    tensor_proto.set_data_format(format);
  }

  void CopyToRepeatedField(google::protobuf::RepeatedField<T>* field) const {
    detail::CopyToProtoAsIs(input.size(), input.data(), field, &context);
  }

  c10::ArrayRef<T> input;
  TensorProto& tensor_proto;
  BaseContext& context;
  const BlobSerializationOptions& options;
};

void SerializeTensorData(const SerializeParams<int64_t>& params) {
  params.CopyToRepeatedField(params.tensor_proto.mutable_int64_data());
}

void SerializeTensorData(const SerializeParams<int32_t>& params) {
  params.CopyToRepeatedField(params.tensor_proto.mutable_int32_data());
}

template <typename T>
typename std::enable_if<
    std::is_same<T, bool>::value || std::is_same<T, uint8_t>::value ||
        std::is_same<T, int8_t>::value || std::is_same<T, uint16_t>::value ||
        std::is_same<T, int16_t>::value,
    void>::type
SerializeTensorData(const SerializeParams<T>& params) {
  SerializeUsingBytesOrInt32<T>(
      EnableByteEncoding<T>(),
      params.input,
      params.context,
      params.tensor_proto);
}

void SerializeTensorData(const SerializeParams<at::Half>& params) {
  SerializeUsingBytesOrInt32<uint16_t>(
      EnableByteEncodingFloat16(),
      params.input,
      params.context,
      params.tensor_proto);
}

#ifdef USE_FBGEMM
namespace {
// Unfortunately we can't include folly/lang/Bits.h here,
// so provide our own byte-swapping code.
fbgemm::bfloat16 ByteSwap(fbgemm::bfloat16 n) {
#ifdef _MSC_VER
  return _byteswap_ushort(n);
#else
  return __builtin_bswap16(n);
#endif
}

void ByteSwapArray(
    const fbgemm::bfloat16* src,
    fbgemm::bfloat16* dest,
    size_t num_elements) {
  // Note that we support src and dest pointing to the same location.
  // We currently only use this function on big-endian machines, so it isn't
  // worth trying to build a fancier SIMD version.
  for (size_t n = 0; n < num_elements; ++n) {
    dest[n] = ByteSwap(src[n]);
  }
}
} // namespace
#endif // USE_FBGEMM

void SerializeTensorData(const SerializeParams<float>& params) {
  // The FLOAT_BFLOAT16 option requests doing a conversion to bfloat16.  This
  // reduces the serialized data size at the cost of some lost precision.
  // We currently only support doing this when compiled with fbgemm.
#ifdef USE_FBGEMM
  if (params.options.float_format() ==
      BlobSerializationOptions_FloatFormat_FLOAT_BFLOAT16) {
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
    std::unique_ptr<float[]> tmp_buffer;
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    const float* src;
    if (params.context.device() == CPU) {
      src = params.input.data();
    } else {
      tmp_buffer.reset(new float[params.input.size()]);
      params.context.CopyToCPU(
          params.input.size(), params.input.data(), tmp_buffer.get());
    }

    params.SetDataFormat(TensorProto_SerializationFormat_FMT_BFLOAT16);
    // TODO: it would be nice if we could use
    // folly::resizeWithoutInitialization() here
    params.tensor_proto.mutable_raw_data()->resize(
        params.input.size() * sizeof(fbgemm::bfloat16));

    Range<fbgemm::bfloat16*> dest(
        reinterpret_cast<fbgemm::bfloat16*>(
            &(*params.tensor_proto.mutable_raw_data())[0]),
        params.input.size());

    // NOLINTNEXTLINE(clang-analyzer-core.CallAndMessage)
    fbgemm::FloatToBfloat16_simd(src, dest.data(), params.input.size());

    // Note: technically a platform can have different integer from floating
    // point endianness, and we ideally should check floating point endianness
    // here.  However, the fbgemm code doesn't appear to make this distinction,
    // and at least in the Bfloat16ToFloat_ref() code it appears to assume that
    // floating point and integer endianness are the same.
    if (!kIsLittleEndian) {
      ByteSwapArray(dest.data(), dest.data(), dest.size());
    }
    return;
  }
#endif

  params.SetDataFormat(TensorProto_SerializationFormat_FMT_PROTOBUF);
  params.CopyToRepeatedField(params.tensor_proto.mutable_float_data());
}

void SerializeTensorData(const SerializeParams<double>& params) {
  params.CopyToRepeatedField(params.tensor_proto.mutable_double_data());
}

void SerializeTensorData(const SerializeParams<std::string>& params) {
  params.tensor_proto.mutable_string_data()->Reserve(params.input.size());
  for (const std::string& element : params.input) {
    params.tensor_proto.add_string_data(element);
  }
}

#define SERIALIZE_TYPE_CASE(proto_type, type)                   \
  case TensorProto_DataType_##proto_type: {                     \
    SerializeTensorData(SerializeParams<type>(                  \
        GetTensorDataRange<type>(input, chunkBegin, chunkSize), \
        proto,                                                  \
        *context,                                               \
        options));                                              \
    return;                                                     \
  }

} // namespace

void TensorSerializer::Serialize(
    const Tensor& input,
    const string& name,
    TensorProto* proto_ptr,
    const BlobSerializationOptions& options,
    size_t chunkBegin,
    int32_t chunkSize) {
  CAFFE_ENFORCE(
      // NOLINTNEXTLINE(clang-diagnostic-sign-compare)
      chunkBegin <= input.numel(),
      "Chunk begin is out of tensor: ",
      chunkBegin,
      ' ',
      input.numel());
  // NOLINTNEXTLINE(clang-diagnostic-sign-compare)
  if (chunkBegin + chunkSize > input.numel()) {
    chunkSize = input.numel() - chunkBegin;
  }

  if (chunkSize != 0) {
    CAFFE_ENFORCE(
        input.raw_data(),
        "The input does not have data input yet. This is probably because you "
        "created a tensor of non-zero shape but never filled its data via "
        "mutable_data() calls. This means that it makes no sense to serialize "
        "the tensor content.");
  } else if (!input.dtype_initialized()) {
    C10_LOG_EVERY_MS(WARNING, 1000)
        << "You're trying to serialize tensor with zero numel and no dtype. "
        << "This is a legacy behavior and it WILL BREAK. Contact PyTorch team "
        << "for details. Offending blob name: " << name;
  }

  TensorProto& proto = *proto_ptr;
  proto.mutable_segment()->set_begin(chunkBegin);
  proto.mutable_segment()->set_end(chunkBegin + chunkSize);

  for (const auto i : c10::irange(input.dim())) {
    proto.add_dims(input.size(i));
  }
  StoreDeviceDetail(input, &proto);

  const TensorProto::DataType data_type = TypeMetaToDataType(input.dtype());
  proto.set_data_type(data_type);
  // TODO: use CUDAGuard here instead of context and employ explicit sync
  // copy
  auto context = CreateContext(input.GetDevice());
  switch (data_type) {
    SERIALIZE_TYPE_CASE(FLOAT, float)
    SERIALIZE_TYPE_CASE(INT32, int32_t)
    SERIALIZE_TYPE_CASE(STRING, std::string)
    SERIALIZE_TYPE_CASE(BOOL, bool)
    SERIALIZE_TYPE_CASE(UINT8, uint8_t)
    SERIALIZE_TYPE_CASE(INT8, int8_t)
    SERIALIZE_TYPE_CASE(UINT16, uint16_t)
    SERIALIZE_TYPE_CASE(INT16, int16_t)
    SERIALIZE_TYPE_CASE(INT64, int64_t)
    SERIALIZE_TYPE_CASE(FLOAT16, at::Half)
    SERIALIZE_TYPE_CASE(DOUBLE, double)
    case TensorProto_DataType_BYTE:
      LOG(FATAL) << "This should not happen. When serializing, "
                    "BYTE is deprecated and moved to UINT8.";
      return;
    case TensorProto_DataType_UNDEFINED:
      proto.mutable_string_data()->Reserve(chunkSize);
      if (chunkSize > 0) {
        const char* raw_data = static_cast<const char*>(input.raw_data());
        for (const auto i : c10::irange(chunkBegin, chunkBegin + chunkSize)) {
          proto.add_string_data(SerializeBlob(
              raw_data + i * input.itemsize(), input.dtype(), ""));
        }
      }
      return;
    case TensorProto_DataType_ZERO_COLLISION_HASH:
      CAFFE_ENFORCE(
        false,
        "Serialization for zero collision hash type is supported by "
        "specialized serializer ZeroCollisionIdHashSerializer");
      return;
    case TensorProto_DataType_REBATCHING_BUFFER:
      CAFFE_ENFORCE(
        false,
        "Serialization for REBATCHING_BUFFER type is supported by "
        "specialized serializer RebatchingBufferSerialier");
      return;

      // Note: we intentially do not provide "default:" so if any new data types
      // are added, the compiler should warn the user to add the case here.
  }

  CAFFE_ENFORCE(false, "unexpected data type during tensor serialization");
}

int GetGPUIDForPointer(const void* ptr);

void TensorSerializer::StoreDeviceDetail(
    const Tensor& input,
    TensorProto* proto) {
  ExtractDeviceOption(proto->mutable_device_detail(), input.GetDevice());
}
// The actual serialization registry objects.
C10_DEFINE_TYPED_REGISTRY(
    BlobSerializerRegistry,
    TypeIdentifier,
    BlobSerializerBase,
    std::unique_ptr);

C10_DEFINE_REGISTRY(BlobDeserializerRegistry, BlobDeserializerBase);

void DeserializeBlob(const string& content, Blob* result) {
  BlobProto blob_proto;
  CAFFE_ENFORCE(
      blob_proto.ParseFromString(content),
      "Cannot parse content into a BlobProto.");
  DeserializeBlob(blob_proto, result);
}

void DeserializeBlob(const BlobProto& blob_proto, Blob* result) {
  if (blob_proto.type() == kTensorBlobType) {
    // This is a tensor object. Depending on the device type, we will
    // use the corresponding TensorDeserializer.
    auto deserializer = CreateDeserializer(
        "Tensor" +
        DeviceTypeName(blob_proto.tensor().device_detail().device_type()));
    // Tensor's deserializer should always be registered, but we will double
    // check if it is not null anyway.
    CAFFE_ENFORCE(deserializer.get());
    deserializer->Deserialize(blob_proto, result);
  } else {
    auto deserializer = CreateDeserializer(blob_proto.type());
    CAFFE_ENFORCE(
        deserializer.get(),
        "No registered deserializer for type ",
        blob_proto.type());
    deserializer->Deserialize(blob_proto, result);
  }
}

// === Local helper functions ===
// Get dimensions from Tensor proto
c10::IntArrayRef DimsFromTensorProto(const TensorProto& proto) {
  return c10::IntArrayRef(proto.dims().data(), proto.dims().size());
}

// Get number of elements from Tensor proto
int64_t NumelFromTensorProto(const TensorProto& tensor_proto) {
  int64_t numel = 1;
  for (const int64_t d : tensor_proto.dims()) {
    numel *= d;
  }
  return numel;
}

// Get data type from Tensor proto
TypeMeta GetDataType(const TensorProto& tensor_proto) {
  TypeMeta dtype;
  if (tensor_proto.data_type() != TensorProto_DataType_UNDEFINED) {
    dtype = DataTypeToTypeMeta(tensor_proto.data_type());
  } else {
    Blob temp_blob;
    DeserializeBlob(tensor_proto.string_data(0), &temp_blob);
    dtype = temp_blob.meta();
  }
  return dtype;
}

// Get TensorOptions from Tensor proto
// Assumes TensorProto is not empty
static at::TensorOptions TensorOptionsFromProto(
    const TensorProto& tensor_proto) {
  return at::dtype(GetDataType(tensor_proto))
      .device(OptionToDevice(tensor_proto.device_detail()));
}

std::unique_ptr<BaseContext> ContextFromProto(
    const TensorProto& tensor_proto) {
  auto device = OptionToDevice(tensor_proto.device_detail());
  return CreateContext(device);
}

// === Local helper functions ===

Tensor EmptyTensorFromProto(const TensorProto& tensor_proto) {
  auto context = ContextFromProto(tensor_proto);
  context->SwitchToDevice();
  if (NumelFromTensorProto(tensor_proto) == 0 &&
      tensor_proto.data_type() == TensorProto_DataType_UNDEFINED) {
    // TODO: remove when serialization of dtype uninitialized tensor is removed
    return caffe2::empty(
        {0},
        at::dtype<float>().device(
            OptionToDevice(tensor_proto.device_detail())));
  } else {
    return caffe2::empty(
        DimsFromTensorProto(tensor_proto),
        TensorOptionsFromProto(tensor_proto));
  }
}

void TensorDeserializer::Deserialize(const BlobProto& blob_proto, Blob* blob) {
  const auto& tensor_proto = blob_proto.tensor();
  auto context = ContextFromProto(tensor_proto);
  context->SwitchToDevice();
  if (NumelFromTensorProto(tensor_proto) == 0 &&
      tensor_proto.data_type() == TensorProto_DataType_UNDEFINED) {
    // TODO: remove after empty Tensor serialization is forbidden
    VLOG(1) << "Deseriralizing an empty Tensor.";
    BlobGetMutableTensor(
        blob,
        {0},
        at::dtype<float>().device(
            OptionToDevice(tensor_proto.device_detail())));
  } else {
    DeserializeToTensor(
        tensor_proto,
        BlobGetMutableTensor(
            blob,
            DimsFromTensorProto(tensor_proto),
            TensorOptionsFromProto(tensor_proto)));
  }
}

namespace {

template <typename T, typename D = T>
void DeserializeFromBytesOrInt32(
    const TensorProto& tensor_proto,
    Range<D*> dest,
    BaseContext& context) {
  if (tensor_proto.has_byte_data()) {
    auto typeSize = sizeof(T);
    CAFFE_ENFORCE(
        kIsLittleEndian || typeSize == 1,
        "Serialization with bytes not supported on big endian platform.");
    size_t numElems = tensor_proto.byte_data().size();
    if (tensor_proto.data_type() == TensorProto_DataType_UINT8) {
      if (tensor_proto.has_segment()) {
        const auto& segment = tensor_proto.segment();
        numElems = segment.end() - segment.begin();
      }
    }
    CAFFE_ENFORCE_EQ(
        typeSize * dest.size(), numElems, "Incorrect proto field size.");
    const uint8_t* protoData =
        reinterpret_cast<const uint8_t*>(tensor_proto.byte_data().data());
    context.template CopyToCPU<D>(
        dest.size(),
        reinterpret_cast<const D*>(protoData),
        dest.data());
  } else {
    // Backward compatibility with models which used int32_data field
    detail::CopyFromProtoWithCast(
        dest.size(),
        tensor_proto.int32_data(),
        reinterpret_cast<T*>(dest.data()),
        &context);
  }
}

/**
 * DeserializeParams is just a helper class to consolidate the parameters
 * required for deserializing tensor data so they can be passed around more
 * easily.
 *
 * It also contains some helper functions to perform some operations on the
 * parameters that are shared by multiple deserialization functions.
 */
template<typename T>
struct DeserializeParams {
  DeserializeParams(Range<T*> dst, const TensorProto& proto, BaseContext& ctx)
      : dest{dst}, tensor_proto{proto}, context{ctx} {}

  void LiteralCopy(c10::string_view src) const {
    // Simply copy the data as-is from src to dest
    CAFFE_ENFORCE_EQ(
        dest.size() * sizeof(T),
        src.size(),
        "incorrect data size when deserializing blob: ",
        dest.size(),
        " * ",
        sizeof(T),
        " != ",
        src.size());
    context.CopyBytesFromCPU(src.size(), src.data(), dest.data());
  }

  void CopyFromRepeatedField(
      const google::protobuf::RepeatedField<T>& field) const {
    detail::CopyFromProtoAsIs(dest.size(), field, dest.data(), &context);
  }

  void CopyFromBytesOrInt32() const {
    DeserializeFromBytesOrInt32<T>(tensor_proto, dest, context);
  }

  Range<T*> dest;
  const TensorProto& tensor_proto;
  BaseContext& context;
};

/**
 * DeserializeTensorData() is specialized for each supported combination of
 * SerializationFormat and output type.
 *
 * The default implementation throws an exception, but this function can be
 * specialized to support different combinations.
 */
template <TensorProto::SerializationFormat, typename T>
void DeserializeTensorData(const DeserializeParams<T>& params) {
  CAFFE_ENFORCE(
      false,
      "unsupported serialization format ",
      static_cast<int>(params.tensor_proto.data_format()),
      " when deserializing float data");
}

#define DESERIALIZE_IMPL(type, data_type)                                   \
  template <>                                                               \
  void                                                                      \
  DeserializeTensorData<TensorProto_SerializationFormat_##data_type, type>( \
      const DeserializeParams<type>& params)

DESERIALIZE_IMPL(int64_t, FMT_PROTOBUF) {
  params.CopyFromRepeatedField(params.tensor_proto.int64_data());
}

DESERIALIZE_IMPL(int32_t, FMT_PROTOBUF) {
  params.CopyFromRepeatedField(params.tensor_proto.int32_data());
}

DESERIALIZE_IMPL(uint16_t, FMT_PROTOBUF) {
  params.CopyFromBytesOrInt32();
}

DESERIALIZE_IMPL(int16_t, FMT_PROTOBUF) {
  params.CopyFromBytesOrInt32();
}

DESERIALIZE_IMPL(uint8_t, FMT_PROTOBUF) {
  params.CopyFromBytesOrInt32();
}

DESERIALIZE_IMPL(int8_t, FMT_PROTOBUF) {
  params.CopyFromBytesOrInt32();
}

DESERIALIZE_IMPL(bool, FMT_PROTOBUF) {
  params.CopyFromBytesOrInt32();
}

void DeserializeLegacyByteData(
    TensorProto::SerializationFormat format,
    const DeserializeParams<uint8_t>& params) {
  // The BYTE format should only be used for very old blobs that don't
  // have a data_format field in the first place.  Let's log this case but
  // continue attempting deserialization anyway.
  CAFFE_ENFORCE_EQ(
      format,
      TensorProto_SerializationFormat_FMT_PROTOBUF,
      "found serialized blob with BYTE data type but unexpected data format ",
      static_cast<int>(format));

  params.LiteralCopy(params.tensor_proto.byte_data());
}

DESERIALIZE_IMPL(at::Half, FMT_PROTOBUF) {
  DeserializeFromBytesOrInt32<uint16_t, at::Half>(
      params.tensor_proto, params.dest, params.context);
}

DESERIALIZE_IMPL(float, FMT_PROTOBUF) {
  params.CopyFromRepeatedField(params.tensor_proto.float_data());
}

DESERIALIZE_IMPL(float, FMT_BFLOAT16) {
#ifdef USE_FBGEMM
  CAFFE_ENFORCE_EQ(
      params.dest.size() * sizeof(fbgemm::bfloat16),
      params.tensor_proto.raw_data().size(),
      "incorrect data size in serialized bfloat16 data");
  auto raw_src = reinterpret_cast<const fbgemm::bfloat16*>(
      params.tensor_proto.raw_data().data());

  // If we are on a big-endian machine, byte-swap the serialized data.
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  const fbgemm::bfloat16* src;
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
  std::unique_ptr<fbgemm::bfloat16[]> bswap_buffer;
  if (kIsLittleEndian) {
    src = raw_src;
  } else {
    bswap_buffer.reset(new fbgemm::bfloat16[params.dest.size()]);
    ByteSwapArray(raw_src, bswap_buffer.get(), params.dest.size());
    src = bswap_buffer.get();
  }

  // If we are on a non-CPU device, we need an intermediate CPU buffer for the
  // bfloat16 to float conversion.
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
  std::unique_ptr<float[]> tmp_buffer;
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  float* dest;
  if (params.context.device() == CPU) {
    dest = params.dest.data();
  } else {
    tmp_buffer.reset(new float[params.dest.size()]);
    dest = tmp_buffer.get();
  }

  fbgemm::Bfloat16ToFloat_simd(src, dest, params.dest.size());
  if (params.context.device() != CPU) {
    params.context.CopyFromCPU(params.dest.size(), dest, params.dest.data());
  }
#else
  // We cannot load serialized bfloat16 data without fbgemm.
  CAFFE_ENFORCE(
      false, "cannot perform bfloat16 to float conversion without fbgemm");
#endif
}

DESERIALIZE_IMPL(double, FMT_PROTOBUF) {
  params.CopyFromRepeatedField(params.tensor_proto.double_data());
}

DESERIALIZE_IMPL(std::string, FMT_PROTOBUF) {
  CAFFE_ENFORCE_EQ(
      params.dest.size(),
      params.tensor_proto.string_data().size(),
      "incorrect data size in serialized data: ",
      params.dest.size(),
      " != ",
      params.tensor_proto.string_data().size());
  for (const auto i : c10::irange(params.dest.size())) {
    params.dest[i] = params.tensor_proto.string_data(i);
  }
}

#define DESERIALIZE_FORMAT_CASE(format)                                 \
  case TensorProto_SerializationFormat_##format: {                      \
    DeserializeTensorData<TensorProto_SerializationFormat_##format, T>( \
        params);                                                        \
    return;                                                             \
  }

template <typename T>
void DeserializeTensorBody(
    TensorProto::SerializationFormat format,
    Range<T*> dest,
    const TensorProto& tensor_proto,
    BaseContext& context) {
  DeserializeParams<T> params(dest, tensor_proto, context);
  switch (format) {
    DESERIALIZE_FORMAT_CASE(FMT_PROTOBUF);
    DESERIALIZE_FORMAT_CASE(FMT_BFLOAT16);
  }

  // This can happen if the blob was serialized by a newer version of the code
  // using some new format value that we don't understand.
  CAFFE_ENFORCE(
      false,
      "unsupported serialization format " + c10::str(static_cast<int>(format)));
}

#define DESERIALIZE_TYPE_CASE(proto_type, type)                          \
  case TensorProto_DataType_##proto_type: {                              \
    DeserializeTensorBody(                                               \
        format,                                                          \
        GetMutableTensorDataRange<type>(*tensor, chunkBegin, chunkSize), \
        tensor_proto,                                                    \
        context);                                                        \
    return;                                                              \
  }

void DeserializeTensor(
    const TensorProto& tensor_proto,
    Tensor* tensor,
    BaseContext& context) {
  int64_t chunkBegin = 0;
  auto chunkEnd = tensor->numel();
  if (tensor_proto.has_segment()) {
    chunkBegin = tensor_proto.segment().begin();
    chunkEnd = tensor_proto.segment().end();
  }
  CAFFE_ENFORCE(
      0 <= chunkBegin && chunkBegin <= chunkEnd && chunkEnd <= tensor->numel(),
      "Invalid chunk ",
      chunkBegin,
      ' ',
      chunkEnd,
      " with total tensor size ",
      tensor->numel());
  auto chunkSize = chunkEnd - chunkBegin;

  if (!tensor_proto.has_data_type()) {
    // If the data_type field is not set, this either means it was not present
    // in the serialized data, or it was set to an enum value that we don't know
    // about.  This likely means that the serialized data was written by a
    // different version of the software using a new data type value that we
    // don't understand.
    throw std::runtime_error(
        "Cannot deserialize tensor: unrecognized data type");
  }

  // If the data_format field is not present this is an older buffer
  // serialized with the FMT_PROTOBUF format.
  auto format = tensor_proto.has_data_format()
      ? static_cast<TensorProto::SerializationFormat>(
            tensor_proto.data_format())
      : TensorProto_SerializationFormat_FMT_PROTOBUF;

  switch (tensor_proto.data_type()) {
    DESERIALIZE_TYPE_CASE(FLOAT, float);
    DESERIALIZE_TYPE_CASE(INT32, int32_t);
    DESERIALIZE_TYPE_CASE(STRING, std::string);
    DESERIALIZE_TYPE_CASE(BOOL, bool);
    DESERIALIZE_TYPE_CASE(UINT8, uint8_t);
    DESERIALIZE_TYPE_CASE(INT8, int8_t);
    DESERIALIZE_TYPE_CASE(UINT16, uint16_t);
    DESERIALIZE_TYPE_CASE(INT16, int16_t);
    DESERIALIZE_TYPE_CASE(INT64, int64_t);
    DESERIALIZE_TYPE_CASE(FLOAT16, at::Half);
    DESERIALIZE_TYPE_CASE(DOUBLE, double);
    case TensorProto_DataType_BYTE:
      // BYTE is special, since it is a legacy data type value that effectively
      // means the same thing as UINT8, except that it used to be serialized in
      // a different format.  Recent code always writes out byte data with the
      // UINT8 type, never BYTE, but let's leave legacy deserialization code in
      // place for now just in case we ever encounter an old blob using this
      // format.
      DeserializeLegacyByteData(
          format,
          DeserializeParams<uint8_t>{
              GetMutableTensorDataRange<uint8_t>(
                  *tensor, chunkBegin, chunkSize),
              tensor_proto,
              context});
      return;
    case TensorProto_DataType_UNDEFINED: {
      Blob temp_blob;
      void* raw_ptr = nullptr;
      for (const auto i : c10::irange(chunkSize)) {
        DeserializeBlob(tensor_proto.string_data(i), &temp_blob);
        if (i == 0) {
          raw_ptr = tensor->raw_mutable_data(temp_blob.meta());
        }
        temp_blob.meta().copy()(
            temp_blob.GetRaw(),
            static_cast<char*>(raw_ptr) +
                (i + chunkBegin) * temp_blob.meta().itemsize(),
            1);
      }
    } return;
    case TensorProto_DataType_ZERO_COLLISION_HASH:
      CAFFE_ENFORCE(
          false,
          "Deserialization for zero collision hash type is supported by "
          "specialized deserializer ZeroCollisionIdHashDeserializer");
      return;
    case TensorProto_DataType_REBATCHING_BUFFER:
      CAFFE_ENFORCE(
          false,
          "Deserialization for REBATCHING_BUFFER type is supported by "
          "specialized serializer RebatchingBufferDeserialier");
      return;
      // Note: we intentially do not provide "default:" so if any new data types
  }

  // We should never reach here unless there is a bug and protobuf somehow
  // returns an unexpected value.  protobuf should filter out all unknown enum
  // values, and the has_data_type() check above will catch that case.
  CAFFE_ENFORCE(
      false,
      "Deserialization for REBATCHING_BUFFER type is supported by "
      "specialized serializer RebatchingBufferDeserialier");
}

} // namespace

void TensorDeserializer::DeserializeToTensor(
    const TensorProto& tensor_proto,
    Tensor* tensor) {
  CAFFE_ENFORCE(
      tensor->storage_initialized() && tensor->dtype_initialized(),
      "Tensor must be initialized before passed into Deserialize function.");
  // We create a local context for deserializing. Since Caffe2 contexts are
  // usually lightweight, this should not involve too much overhead.
  auto context = ContextFromProto(tensor_proto);
  context->SwitchToDevice();
  DeserializeTensor(tensor_proto, tensor, *context);
  context->FinishDeviceComputation();
}

Tensor TensorDeserializer::Deserialize(const TensorProto& tensor_proto) {
  auto tensor = EmptyTensorFromProto(tensor_proto);
  DeserializeToTensor(tensor_proto, &tensor);
  return tensor;
}

////////////////////////////////////////////////////////////////////////////////
// Serialization Helpers
////////////////////////////////////////////////////////////////////////////////

std::string SerializeAsString_EnforceCheck(
    const google::protobuf::MessageLite& msg,
    const char* error_location) {
  std::string serialize_output;
  bool result = msg.SerializeToString(&serialize_output);
  if (!error_location) {
    CAFFE_ENFORCE(result, "protobuf::SerializeToString failed");
  } else {
    CAFFE_ENFORCE(
        result, "protobuf::SerializeToString failed for ", error_location);
  }
  return serialize_output;
}

namespace {
// Serialize Tensor
REGISTER_BLOB_SERIALIZER((TypeMeta::Id<Tensor>()), TensorSerializer);
REGISTER_BLOB_DESERIALIZER(TensorCPU, TensorDeserializer);
// Serialize std::string
REGISTER_BLOB_SERIALIZER((TypeMeta::Id<std::string>()), StringSerializer);
REGISTER_BLOB_DESERIALIZER(std::string, StringDeserializer);
} // namespace
} // namespace caffe2
