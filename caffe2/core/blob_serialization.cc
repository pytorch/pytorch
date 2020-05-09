#include "caffe2/core/blob_serialization.h"

#include <mutex>
#include <sstream>

#include "caffe2/core/blob.h"
#include "caffe2/utils/proto_utils.h"

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
    "Serialize BOOL, UINT8, INT8, UINT16, INT16, INT64, FLOAT16 tensors using byte_data field instead of int32");

#ifdef _MSC_VER
// It's MSVC, so we just have to guess ... and allow an override
#ifdef FOLLY_ENDIAN_BE
constexpr auto kIsLittleEndian = false;
#else
constexpr auto kIsLittleEndian = true;
#endif
#else
constexpr auto kIsLittleEndian = __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__;
#endif

namespace caffe2 {
/**
 * @brief StringSerializer is the serializer for String.
 *
 * StringSerializer takes in a blob that contains a String, and serializes it
 * into a BlobProto protocol buffer.
 */
class StringSerializer : public BlobSerializerBase {
 public:
  StringSerializer() {}
  ~StringSerializer() override {}
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
    int chunk_size) {
  std::unique_ptr<BlobSerializerBase> serializer(
      CreateSerializer(typeMeta.id()));
  CAFFE_ENFORCE(serializer, "No known serializer for ", typeMeta.name());
  serializer->SerializeWithChunkSize(
      pointer, typeMeta, name, acceptor, chunk_size);
}

std::string
SerializeBlob(const void* pointer, TypeMeta typeMeta, const string& name) {
  std::string data;
  BlobSerializerBase::SerializationAcceptor acceptor =
      [&data](const std::string&, const std::string& blob_str) {
        DCHECK(data.empty()); // should be called once with kNoChunking
        data = blob_str;
      };
  SerializeBlob(pointer, typeMeta, name, acceptor, kNoChunking);
  return data;
}
} // namespace

void SerializeBlob(
    const Blob& blob,
    const string& name,
    BlobSerializerBase::SerializationAcceptor acceptor,
    int chunk_size) {
  SerializeBlob(blob.GetRaw(), blob.meta(), name, acceptor, chunk_size);
}

std::string SerializeBlob(const Blob& blob, const string& name) {
  return SerializeBlob(blob.GetRaw(), blob.meta(), name);
}

void TensorSerializer::Serialize(
    const void* pointer,
    TypeMeta typeMeta,
    const string& name,
    BlobSerializerBase::SerializationAcceptor acceptor) {
  this->SerializeWithChunkSize(
      pointer, typeMeta, name, acceptor, kDefaultChunkSize);
}

void TensorSerializer::SerializeWithChunkSize(
    const void* pointer,
    TypeMeta typeMeta,
    const string& name,
    BlobSerializerBase::SerializationAcceptor acceptor,
    int chunk_size) {
  CAFFE_ENFORCE(typeMeta.Match<Tensor>());
  const auto& tensor = *static_cast<const Tensor*>(pointer);
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
        tensor, name, blob_proto.mutable_tensor(), chunkStart, chunk_size);
    acceptor(
        c10::str(name, kChunkIdSeparator, chunkStart / chunk_size),
        SerializeBlobProtoAsString_EnforceCheck(blob_proto));
  };

#ifndef __ANDROID__
  // Poorman's IOBound ThreadPool
  SimpleQueue<size_t> chunkQueue;
  auto task = [&]() {
    size_t chunkStart;
    while (chunkQueue.Pop(&chunkStart)) {
      processChunk(chunkStart);
    }
  };
  std::vector<std::future<void>> futures;
  if (tensor.numel() > chunk_size) {
    futures.reserve(FLAGS_caffe2_max_tensor_serializer_threads);
    for (int i = 0; i < FLAGS_caffe2_max_tensor_serializer_threads; ++i) {
      futures.emplace_back(std::async(std::launch::async, task));
    }
  }
#endif

  VLOG(1) << "Serializing blob " << name;
  // Serialize whole vector. If vector is empty, it's shape still needs to be
  // serialized in empty proto
  for (size_t chunkBegin = 0;
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

static bool EnableByteEncoding(
    const TensorProto::DataType& dataType,
    const size_t& typeSize) {
  // if typeSize == 1, endianness does not matter. Else check for endianness.
  bool ret = false;
  bool safeForEndianness = (typeSize == 1 || kIsLittleEndian);
  if (safeForEndianness) {
    ret = FLAGS_caffe2_serialize_using_bytes_as_holder;
    // Check if special casing for float is enabled if
    // caffe2_serialize_using_bytes_as_holder is not enabled.
    if (!ret) {
      ret =
          (dataType == TensorProto_DataType_FLOAT16 &&
           FLAGS_caffe2_serialize_fp16_as_bytes);
    }
  }
  return ret;
}

template <typename T, typename S = T>
static void SerializeUsingBytesOrInt32(
    const Tensor& input,
    const TensorProto::DataType& dataType,
    size_t chunkBegin,
    int32_t chunkSize,
    BaseContext* context,
    TensorProto& proto) {
  const auto typeSize = sizeof(T);
  if (EnableByteEncoding(dataType, typeSize)) {
    const auto bufSize = typeSize * chunkSize;
    auto* byteData =
        reinterpret_cast<const uint8_t*>(input.template data<S>() + chunkBegin);
    unique_ptr<uint8_t[]> buffer(new uint8_t[bufSize]);
    context->template CopyToCPU<uint8_t>(bufSize, byteData, buffer.get());
    context->FinishDeviceComputation();
    proto.set_byte_data(buffer.release(), bufSize);
  } else {
    detail::CopyToProtoWithCast(
        chunkSize,
        reinterpret_cast<const T*>(input.template data<S>()) + chunkBegin,
        proto.mutable_int32_data(),
        context);
  }
}

void TensorSerializer::Serialize(
    const Tensor& input,
    const string& name,
    TensorProto* proto_ptr,
    size_t chunkBegin,
    int32_t chunkSize) {
  CAFFE_ENFORCE(
      chunkBegin <= input.numel(),
      "Chunk begin is out of tensor: ",
      chunkBegin,
      ' ',
      input.numel());
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

  for (int i = 0; i < input.dim(); ++i) {
    proto.add_dims(input.size(i));
  }
  const TensorProto::DataType data_type = TypeMetaToDataType(input.dtype());
  proto.set_data_type(data_type);
  StoreDeviceDetail(input, &proto);
  // TODO: use CUDAGuard here instead of context and employ explicit sync
  // copy
  auto uniq_ptr = CreateContext(input.GetDevice());
  // A lot of copypaste is error prone. Should we create a macro for this?
  switch (data_type) {
    case TensorProto_DataType_FLOAT:
      detail::CopyToProtoAsIs(
          chunkSize,
          input.template data<float>() + chunkBegin,
          proto.mutable_float_data(),
          uniq_ptr.get());
      break;
    case TensorProto_DataType_INT32:
      detail::CopyToProtoAsIs(
          chunkSize,
          input.template data<int>() + chunkBegin,
          proto.mutable_int32_data(),
          uniq_ptr.get());
      break;
    case TensorProto_DataType_BYTE:
      LOG(FATAL) << "This should not happen. When serializing, "
                    "BYTE is deprecated and moved to UINT8.";
      break;
    case TensorProto_DataType_STRING: {
      proto.mutable_string_data()->Reserve(chunkSize);
      const string* content = input.template data<string>();
      for (int i = chunkBegin; i < chunkBegin + chunkSize; ++i) {
        proto.add_string_data(content[i]);
      }
      break;
    }
    case TensorProto_DataType_BOOL:
      SerializeUsingBytesOrInt32<bool>(
          input, data_type, chunkBegin, chunkSize, uniq_ptr.get(), proto);
      break;
    case TensorProto_DataType_UINT8:
      SerializeUsingBytesOrInt32<uint8_t>(
          input, data_type, chunkBegin, chunkSize, uniq_ptr.get(), proto);
      break;
    case TensorProto_DataType_INT8:
      SerializeUsingBytesOrInt32<int8_t>(
          input, data_type, chunkBegin, chunkSize, uniq_ptr.get(), proto);
      break;
    case TensorProto_DataType_UINT16:
      SerializeUsingBytesOrInt32<uint16_t>(
          input, data_type, chunkBegin, chunkSize, uniq_ptr.get(), proto);
      break;
    case TensorProto_DataType_INT16:
      SerializeUsingBytesOrInt32<int16_t>(
          input, data_type, chunkBegin, chunkSize, uniq_ptr.get(), proto);
      break;
    case TensorProto_DataType_INT64:
      detail::CopyToProtoAsIs(
          chunkSize,
          input.template data<int64_t>() + chunkBegin,
          proto.mutable_int64_data(),
          uniq_ptr.get());
      break;
    case TensorProto_DataType_FLOAT16:
      SerializeUsingBytesOrInt32<uint16_t, at::Half>(
          input, data_type, chunkBegin, chunkSize, uniq_ptr.get(), proto);
      break;
    case TensorProto_DataType_DOUBLE:
      detail::CopyToProtoAsIs(
          chunkSize,
          input.template data<double>() + chunkBegin,
          proto.mutable_double_data(),
          uniq_ptr.get());
      break;
    case TensorProto_DataType_UNDEFINED: {
      proto.mutable_string_data()->Reserve(chunkSize);
      if (chunkSize > 0) {
        const char* raw_data = static_cast<const char*>(input.raw_data());
        for (int i = chunkBegin; i < chunkBegin + chunkSize; ++i) {
          proto.add_string_data(SerializeBlob(
              raw_data + i * input.itemsize(), input.dtype(), ""));
        }
      }
    } break;
    case TensorProto_DataType_ZERO_COLLISION_HASH: {
      CAFFE_ENFORCE(
          false,
          "Serialization for zero collision hash type is supported by specialized serializer ZeroCollisionIdHashSerializer");
    } break;
      // Note: we intentially do not provide "default:" so if any new data types
      // are added, the compiler should warn the user to add the case here.
  }
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
static std::vector<int64_t> DimsFromTensorProto(const TensorProto& proto) {
  std::vector<int64_t> dims;
  dims.reserve(proto.dims().size());
  for (const int64_t d : proto.dims()) {
    dims.push_back(d);
  }
  return dims;
}

// Get number of elements from Tensor proto
static int64_t NumelFromTensorProto(const TensorProto& tensor_proto) {
  int64_t numel = 1;
  for (const int64_t d : tensor_proto.dims()) {
    numel *= d;
  }
  return numel;
}

// Get data type from Tensor proto
static TypeMeta GetDataType(const TensorProto& tensor_proto) {
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

static std::unique_ptr<BaseContext> ContextFromProto(
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
  auto tensor_proto = blob_proto.tensor();
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

template <typename T, typename D = T>
void DeserializeFromBytesOrInt32(
    const TensorProto& tensor_proto,
    size_t chunkBegin,
    int32_t chunkSize,
    BaseContext* context,
    Tensor* tensor) {
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
        typeSize * chunkSize, numElems, "Incorrect proto field size.");
    const uint8_t* protoData =
        reinterpret_cast<const uint8_t*>(tensor_proto.byte_data().data());
    context->template CopyToCPU<D>(
        chunkSize,
        reinterpret_cast<const D*>(protoData),
        tensor->template mutable_data<D>() + chunkBegin);
  } else {
    // Backward compatibility with models which used int32_data field
    detail::CopyFromProtoWithCast(
        chunkSize,
        tensor_proto.int32_data(),
        reinterpret_cast<T*>(tensor->template mutable_data<D>()) + chunkBegin,
        context);
  }
}

void TensorDeserializer::DeserializeToTensor(
    const TensorProto& tensor_proto,
    Tensor* tensor) {
  CAFFE_ENFORCE(
      tensor->storage_initialized() && tensor->dtype_initialized(),
      "Tensor must be initialized before passed into Deserialize function.");
  // We create a local context for deserializing. Since Caffe2 contexts are
  // usually lightweight, this should not involve too much overhead.
  auto uniq_ptr = ContextFromProto(tensor_proto);
  // since CopyFromProtoAsIs accepts BaseContext*
  auto context = uniq_ptr.get();
  context->SwitchToDevice();

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

  switch (tensor_proto.data_type()) {
    case TensorProto_DataType_FLOAT:
      detail::CopyFromProtoAsIs(
          chunkSize,
          tensor_proto.float_data(),
          tensor->template mutable_data<float>() + chunkBegin,
          context);
      break;
    case TensorProto_DataType_INT32:
      detail::CopyFromProtoAsIs(
          chunkSize,
          tensor_proto.int32_data(),
          tensor->template mutable_data<int>() + chunkBegin,
          context);
      break;
    case TensorProto_DataType_BYTE:
      // Since BYTE stores the data in a string field instead of a repreated
      // field we will have it special cased.
      CAFFE_ENFORCE_EQ(
          chunkSize,
          tensor_proto.byte_data().size(),
          "Incorrect proto field size.");
      context->template CopyToCPU<uint8_t>(
          chunkSize,
          reinterpret_cast<const uint8_t*>(tensor_proto.byte_data().data()),
          tensor->template mutable_data<uint8_t>() + chunkBegin);
      break;
    case TensorProto_DataType_STRING:
      // Special handing of string because it is a non-fundamental type.
      {
        string* content = tensor->template mutable_data<string>();
        for (int i = 0; i < chunkSize; ++i) {
          content[i + chunkBegin] = tensor_proto.string_data(i);
        }
      }
      break;
    case TensorProto_DataType_BOOL:
      DeserializeFromBytesOrInt32<bool>(
          tensor_proto, chunkBegin, chunkSize, context, tensor);
      break;
    case TensorProto_DataType_UINT8:
      DeserializeFromBytesOrInt32<uint8_t>(
          tensor_proto, chunkBegin, chunkSize, context, tensor);
      break;
    case TensorProto_DataType_INT8:
      DeserializeFromBytesOrInt32<int8_t>(
          tensor_proto, chunkBegin, chunkSize, context, tensor);
      break;
    case TensorProto_DataType_UINT16:
      DeserializeFromBytesOrInt32<uint16_t>(
          tensor_proto, chunkBegin, chunkSize, context, tensor);
      break;
    case TensorProto_DataType_INT16:
      DeserializeFromBytesOrInt32<int16_t>(
          tensor_proto, chunkBegin, chunkSize, context, tensor);
      break;
    case TensorProto_DataType_INT64:
      detail::CopyFromProtoAsIs(
          chunkSize,
          tensor_proto.int64_data(),
          tensor->template mutable_data<int64_t>() + chunkBegin,
          context);
      break;
    case TensorProto_DataType_FLOAT16:
      DeserializeFromBytesOrInt32<uint16_t, at::Half>(
          tensor_proto, chunkBegin, chunkSize, context, tensor);
      break;
    case TensorProto_DataType_DOUBLE:
      detail::CopyFromProtoAsIs(
          chunkSize,
          tensor_proto.double_data(),
          tensor->template mutable_data<double>() + chunkBegin,
          context);
      break;
    case TensorProto_DataType_UNDEFINED: {
      Blob temp_blob;
      void* raw_ptr = nullptr;
      for (int i = 0; i < chunkSize; ++i) {
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
    } break;
    case TensorProto_DataType_ZERO_COLLISION_HASH: {
      CAFFE_ENFORCE(
          false,
          "Deserialization for zero collision hash type is supported by specialized deserializer ZeroCollisionIdHashDeserializer");
    } break;
      // Note: we intentially do not provide "default:" so if any new data types
  }
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
