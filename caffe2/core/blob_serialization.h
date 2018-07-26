#ifndef CAFFE2_CORE_BLOB_SERIALIZATION_H_
#define CAFFE2_CORE_BLOB_SERIALIZATION_H_

#include <limits>
#include <future>

#include <google/protobuf/repeated_field.h>

#include "caffe2/core/blob.h"
#include "caffe2/core/blob_serializer_base.h"
#include "caffe2/core/tensor.h"
#include "caffe2/core/typeid.h"
#include "caffe2/core/types.h"
#include "caffe2/utils/simple_queue.h"

CAFFE2_DECLARE_int(caffe2_tensor_chunk_size);
CAFFE2_DECLARE_int(caffe2_max_tensor_serializer_threads);
CAFFE2_DECLARE_bool(caffe2_serialize_fp16_as_bytes);

namespace caffe2 {

constexpr auto kTensorBlobType = "Tensor";
// String used to separate chunk id from the blob name when storing in DB
constexpr auto kChunkIdSeparator = "#%";

// The Blob serialization registry and serializer creator functions.
CAFFE_DECLARE_TYPED_REGISTRY(
    BlobSerializerRegistry,
    CaffeTypeId,
    BlobSerializerBase,
    std::unique_ptr);
#define REGISTER_BLOB_SERIALIZER(id, ...) \
  CAFFE_REGISTER_TYPED_CLASS(BlobSerializerRegistry, id, __VA_ARGS__)
// Creates an operator with the given operator definition.
inline unique_ptr<BlobSerializerBase> CreateSerializer(CaffeTypeId id) {
  return BlobSerializerRegistry()->Create(id);
}

/**
 * @brief TensorSerializer is the serializer for Tensors.
 *
 * TensorSerializer takes in a blob that contains a Tensor, and serializes it
 * into a TensorProto protocol buffer.
 */
template <class Context>
class TensorSerializer : public BlobSerializerBase {
 public:
  TensorSerializer() : context_() {}
  ~TensorSerializer() override {}
  /**
   * Serializes a Blob. Note that this blob has to contain Tensor<Context>,
   * otherwise this function produces a fatal error.
   */
  void Serialize(
      const Blob& blob,
      const string& name,
      SerializationAcceptor acceptor) override;
  void SerializeWithChunkSize(
      const Blob& blob,
      const string& name,
      SerializationAcceptor acceptor,
      int chunk_size) override;

  void Serialize(const Tensor<Context>& tensor, const string& name,
                 TensorProto* proto, size_t chunkBegin, int32_t chunkSize);

 private:
  // A utility function to store the device context detauls.
  void StoreDeviceDetail(const Tensor<Context>& input, TensorProto* proto);
  Context context_;
};

/**
 * @brief BlobDeserializerBase is an abstract class that deserializes a blob
 * from a BlobProto or a TensorProto.
 */
class BlobDeserializerBase {
 public:
  virtual ~BlobDeserializerBase() {}

  // Deserializes from a BlobProto object.
  virtual void Deserialize(const BlobProto& proto, Blob* blob) = 0;
};

CAFFE_DECLARE_REGISTRY(BlobDeserializerRegistry, BlobDeserializerBase);
#define REGISTER_BLOB_DESERIALIZER(name, ...) \
  CAFFE_REGISTER_CLASS(BlobDeserializerRegistry, name, __VA_ARGS__)
// Creates an operator with the given operator definition.
inline unique_ptr<BlobDeserializerBase> CreateDeserializer(const string& type) {
  return BlobDeserializerRegistry()->Create(type);
}

/**
 * @brief TensorDeserializer is the deserializer for Tensors.
 *
 * The device that the deserialized Tensor will live under is determined by the
 * device_detail field. If you want to specify the device of the deserialized
 * tensor, change the TensorProto's corresponding fields before calling
 * Deserialize.
 */
template <class Context>
class TensorDeserializer : public BlobDeserializerBase {
 public:
  void Deserialize(const BlobProto& proto, Blob* blob) override;
  void Deserialize(const TensorProto& proto, Tensor<Context>* tensor);
};

////////////////////////////////////////////////////////////////////////////////
// Implementations
////////////////////////////////////////////////////////////////////////////////

namespace detail {
template <typename SrcType, typename DstType, class Context>
inline void CopyToProtoAsIs(
    const size_t size,
    const SrcType* src,
    google::protobuf::RepeatedField<DstType>* field,
    Context* context) {
  static_assert(
      sizeof(SrcType) == sizeof(DstType),
      "The source type and dest type cannot be copied as-is. Did "
      "you mean CopyToProtoWithCast?");
  field->Reserve(size);
  for (int i = 0; i < size; ++i) {
    field->Add(0);
  }
  context->template Copy<SrcType, Context, CPUContext>(
      size, src, reinterpret_cast<SrcType*>(field->mutable_data()));
  // Make sure that we finish the copy into the protobuf.
  context->FinishDeviceComputation();
}

template <typename SrcType, typename DstType, class Context>
inline void CopyToProtoWithCast(
    const size_t size,
    const SrcType* src,
    google::protobuf::RepeatedField<DstType>* field,
    Context* context) {
  // TODO: we are having one unnecessary copy here if the context is already
  // CPUContext. Remove it if it is performance critical.
  unique_ptr<SrcType[]> buffer(new SrcType[size]);
  context->template Copy<SrcType, Context, CPUContext>(
      size, src, buffer.get());
  context->FinishDeviceComputation();
  field->Reserve(size);
  for (int i = 0; i < size; ++i) {
    field->Add(static_cast<DstType>(buffer[i]));
  }
}

template <typename SrcType, typename DstType, class Context>
inline void CopyFromProtoAsIs(
    const size_t size,
    const google::protobuf::RepeatedField<SrcType>& field,
    DstType* dst,
    Context* context) {
  static_assert(
      sizeof(SrcType) == sizeof(DstType),
      "The source type and dest type cannot be copied as-is. Did "
      "you mean CopyFromProtoWithCast?");
  CAFFE_ENFORCE_EQ(size, field.size(), "Incorrect proto field size.");
  context->template Copy<DstType, CPUContext, Context>(
      size, reinterpret_cast<const DstType*>(field.data()), dst);
}

template <typename SrcType, typename DstType, class Context>
inline void CopyFromProtoWithCast(
    const size_t size,
    const google::protobuf::RepeatedField<SrcType>& field,
    DstType* dst,
    Context* context) {
  CAFFE_ENFORCE_EQ(size, field.size(), "Incorrect proto field size.");
  // TODO: we are having one unnecessary copy here if the context is already
  // CPUContext. Remove it if it is performance critical.
  unique_ptr<DstType[]> buffer(new DstType[size]);
  const SrcType* src = field.data();
  for (int i = 0; i < size; ++i) {
    buffer[i] = static_cast<DstType>(src[i]);
  }
  context->template Copy<DstType, CPUContext, Context>(size, buffer.get(), dst);
}

}  // namespace detail

template <class Context>
void TensorSerializer<Context>::Serialize(
    const Blob& blob,
    const string& name,
    BlobSerializerBase::SerializationAcceptor acceptor) {
  this->SerializeWithChunkSize(blob, name, acceptor, kDefaultChunkSize);
}

template <class Context>
void TensorSerializer<Context>::SerializeWithChunkSize(
    const Blob& blob,
    const string& name,
    BlobSerializerBase::SerializationAcceptor acceptor,
    int chunk_size) {
  CAFFE_ENFORCE(blob.IsType<Tensor<Context>>());
  const auto& tensor = blob.template Get<Tensor<Context>>();
  if (chunk_size == kNoChunking) {
    chunk_size = tensor.size() + 1; // to account for empty tensors
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
        MakeString(name, kChunkIdSeparator, chunkStart / chunk_size),
        blob_proto.SerializeAsString());
  };

#ifndef __ANDROID__
  std::vector<std::future<void>> futures;
  // Poorman's IOBound ThreadPool
  SimpleQueue<size_t> chunkQueue;
  auto task = [&]() {
    size_t chunkStart;
    while (chunkQueue.Pop(&chunkStart)) {
      processChunk(chunkStart);
    }
  };
  if (tensor.size() > chunk_size) {
    for (int i = 0; i < FLAGS_caffe2_max_tensor_serializer_threads; ++i) {
      futures.emplace_back(std::async(std::launch::async, task));
    }
  }
#endif

  VLOG(1) << "Serializing blob " << name;
  // Serialize whole vector. If vector is empty, it's shape still needs to be
  // serialized in empty proto
  for (size_t chunkBegin = 0;
       chunkBegin < std::max(tensor.size(), static_cast<TIndex>(1));
       chunkBegin += chunk_size) {
    VLOG(2) << "Starting a chunk at " << chunkBegin;
#ifndef __ANDROID__
    if (tensor.size() > chunk_size) {
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

template <class Context>
void TensorSerializer<Context>::Serialize(
    const Tensor<Context>& input,
    const string& /*name*/,
    TensorProto* proto_ptr,
    size_t chunkBegin,
    int32_t chunkSize) {
  CAFFE_ENFORCE(
      chunkBegin <= input.size(),
      "Chunk begin is out of tensor: ",
      chunkBegin,
      ' ',
      input.size());
  if (chunkBegin + chunkSize > input.size()) {
    chunkSize = input.size() - chunkBegin;
  }

  CAFFE_ENFORCE(
      input.raw_data() || chunkSize == 0,
      "The input does not have data input yet. This is probably because you "
      "created a tensor of non-zero shape but never filled its data via "
      "mutable_data() calls. This means that it makes no sense to serialize "
      "the tensor content.");

  TensorProto& proto = *proto_ptr;
  proto.mutable_segment()->set_begin(chunkBegin);
  proto.mutable_segment()->set_end(chunkBegin + chunkSize);

  for (int i = 0; i < input.ndim(); ++i) {
    proto.add_dims(input.dim(i));
  }
  const TensorProto::DataType data_type = TypeMetaToDataType(input.meta());
  proto.set_data_type(data_type);
  StoreDeviceDetail(input, &proto);

  // A lot of copypaste is error prone. Should we create a macro for this?
  switch (data_type) {
  case TensorProto_DataType_FLOAT:
    detail::CopyToProtoAsIs(
        chunkSize,
        input.template data<float>() + chunkBegin,
        proto.mutable_float_data(),
        &this->context_);
    break;
  case TensorProto_DataType_INT32:
    detail::CopyToProtoAsIs(
        chunkSize,
        input.template data<int>() + chunkBegin,
        proto.mutable_int32_data(),
        &this->context_);
    break;
  case TensorProto_DataType_BYTE:
    LOG(FATAL) << "This should not happen. When serializing, "
                  "BYTE is deprecated and moved to UINT8.";
    break;
  case TensorProto_DataType_STRING:
    {
      proto.mutable_string_data()->Reserve(chunkSize);
      const string* content = input.template data<string>();
      for (int i = chunkBegin; i < chunkBegin + chunkSize; ++i) {
        proto.add_string_data(content[i]);
      }
      break;
    }
  case TensorProto_DataType_BOOL:
    detail::CopyToProtoWithCast(
        chunkSize,
        input.template data<bool>() + chunkBegin,
        proto.mutable_int32_data(),
        &this->context_);
    break;
  case TensorProto_DataType_UINT8:
    detail::CopyToProtoWithCast(
        chunkSize,
        input.template data<uint8_t>() + chunkBegin,
        proto.mutable_int32_data(),
        &this->context_);
    break;
  case TensorProto_DataType_INT8:
    detail::CopyToProtoWithCast(
        chunkSize,
        input.template data<int8_t>() + chunkBegin,
        proto.mutable_int32_data(),
        &this->context_);
    break;
  case TensorProto_DataType_UINT16:
    detail::CopyToProtoWithCast(
        chunkSize,
        input.template data<uint16_t>() + chunkBegin,
        proto.mutable_int32_data(),
        &this->context_);
    break;
  case TensorProto_DataType_INT16:
    detail::CopyToProtoWithCast(
        chunkSize,
        input.template data<int16_t>() + chunkBegin,
        proto.mutable_int32_data(),
        &this->context_);
    break;
  case TensorProto_DataType_INT64:
    detail::CopyToProtoAsIs(
        chunkSize,
        input.template data<int64_t>() + chunkBegin,
        proto.mutable_int64_data(),
        &this->context_);
    break;
  case TensorProto_DataType_FLOAT16: {
    if (FLAGS_caffe2_serialize_fp16_as_bytes) {
      const int kValue = 1;
      CAFFE_ENFORCE_EQ(
          reinterpret_cast<const char*>(&kValue)[0],
          1,
          "Serialization of FLOAT16 on big endian platform "
          "is not written yet.");
      unique_ptr<char[]> buffer(new char[2 * chunkSize]);
      this->context_.template Copy<char, Context, CPUContext>(
          2 * chunkSize,
          reinterpret_cast<const char*>(
              input.template data<float16>() + chunkBegin),
          buffer.get());
      this->context_.FinishDeviceComputation();
      proto.set_byte_data(buffer.release(), 2 * chunkSize);
    } else {
      detail::CopyToProtoWithCast(
          chunkSize,
          reinterpret_cast<const uint16_t*>(input.template data<float16>()) +
              chunkBegin,
          proto.mutable_int32_data(),
          &this->context_);
    }
  } break;
  case TensorProto_DataType_DOUBLE:
    detail::CopyToProtoAsIs(
        chunkSize,
        input.template data<double>() + chunkBegin,
        proto.mutable_double_data(),
        &this->context_);
    break;
  case TensorProto_DataType_UNDEFINED: {
    proto.mutable_string_data()->Reserve(chunkSize);
    Blob temp_blob;
    const char* raw_data = static_cast<const char*>(input.raw_data());
    for (int i = chunkBegin; i < chunkBegin + chunkSize; ++i) {
      temp_blob.ShareExternal(
          const_cast<char*>(raw_data + i * input.itemsize()), input.meta());
      proto.add_string_data(temp_blob.Serialize(""));
    }
  } break;
    // Note: we intentially do not provide "default:" so if any new data types
    // are added, the compiler should warn the user to add the case here.
  }
}

template <class Context>
void TensorDeserializer<Context>::Deserialize(
    const BlobProto& blob_proto,
    Blob* blob) {
  Deserialize(blob_proto.tensor(), blob->GetMutable<Tensor<Context>>());
}

template <class Context>
void TensorDeserializer<Context>::Deserialize(
    const TensorProto& proto,
    Tensor<Context>* tensor) {
  // We create a local context for deserializing. Since Caffe2 contexts are
  // usually lightweighted, this should not involve too much overhead.
  Context context(proto.device_detail());
  context.SwitchToDevice(0);
  vector<TIndex> dims;
  for (const TIndex d : proto.dims()) {
    dims.push_back(d);
  }
  tensor->Resize(dims);

  int64_t chunkBegin = 0;
  auto chunkEnd = tensor->size();
  if (proto.has_segment()) {
    chunkBegin = proto.segment().begin();
    chunkEnd = proto.segment().end();
  }
  CAFFE_ENFORCE(
      0 <= chunkBegin && chunkBegin <= chunkEnd && chunkEnd <= tensor->size(),
      "Invalid chunk ",
      chunkBegin,
      ' ',
      chunkEnd,
      " with total tensor size ",
      tensor->size());
  auto chunkSize = chunkEnd - chunkBegin;

  switch (proto.data_type()) {
    case TensorProto_DataType_FLOAT:
      detail::CopyFromProtoAsIs(
          chunkSize,
          proto.float_data(),
          tensor->template mutable_data<float>() + chunkBegin,
          &context);
      break;
    case TensorProto_DataType_INT32:
      detail::CopyFromProtoAsIs(
          chunkSize,
          proto.int32_data(),
          tensor->template mutable_data<int>() + chunkBegin,
          &context);
      break;
    case TensorProto_DataType_BYTE:
      // Since BYTE stores the data in a string field instead of a repreated
      // field we will have it special cased.
      CAFFE_ENFORCE_EQ(
          chunkSize, proto.byte_data().size(), "Incorrect proto field size.");
      context.template Copy<uint8_t, Context, CPUContext>(
          chunkSize,
          reinterpret_cast<const uint8_t*>(proto.byte_data().data()),
          tensor->template mutable_data<uint8_t>() + chunkBegin);
      break;
    case TensorProto_DataType_STRING:
      // Special handing of string because it is a non-fundamental type.
      {
        string* content = tensor->template mutable_data<string>();
        for (int i = 0; i < chunkSize; ++i) {
          content[i + chunkBegin] = proto.string_data(i);
        }
      }
      break;
    case TensorProto_DataType_BOOL:
      detail::CopyFromProtoWithCast(
          chunkSize,
          proto.int32_data(),
          tensor->template mutable_data<bool>() + chunkBegin,
          &context);
      break;
    case TensorProto_DataType_UINT8:
      detail::CopyFromProtoWithCast(
          chunkSize,
          proto.int32_data(),
          tensor->template mutable_data<uint8_t>() + chunkBegin,
          &context);
      break;
    case TensorProto_DataType_INT8:
      detail::CopyFromProtoWithCast(
          chunkSize,
          proto.int32_data(),
          tensor->template mutable_data<int8_t>() + chunkBegin,
          &context);
      break;
    case TensorProto_DataType_UINT16:
      detail::CopyFromProtoWithCast(
          chunkSize,
          proto.int32_data(),
          tensor->template mutable_data<uint16_t>() + chunkBegin,
          &context);
      break;
    case TensorProto_DataType_INT16:
      detail::CopyFromProtoWithCast(
          chunkSize,
          proto.int32_data(),
          tensor->template mutable_data<int16_t>() + chunkBegin,
          &context);
      break;
    case TensorProto_DataType_INT64:
      detail::CopyFromProtoAsIs(
          chunkSize,
          proto.int64_data(),
          tensor->template mutable_data<int64_t>() + chunkBegin,
          &context);
      break;
    case TensorProto_DataType_FLOAT16:
      if (proto.has_byte_data()) {
        const int kValue = 1;
        CAFFE_ENFORCE_EQ(
            reinterpret_cast<const char*>(&kValue)[0],
            1,
            "Serialization of FLOAT16 on big endian platform "
            "is not written yet.");
        CAFFE_ENFORCE_EQ(
            2 * chunkSize,
            proto.byte_data().size(),
            "Incorrect proto field size.");
        context.template Copy<float16, Context, CPUContext>(
            chunkSize,
            reinterpret_cast<const float16*>(proto.byte_data().data()),
            tensor->template mutable_data<float16>() + chunkBegin);
      } else {
        // Backward compatibility with models which used int32_data field
        detail::CopyFromProtoWithCast(
            chunkSize,
            proto.int32_data(),
            reinterpret_cast<uint16_t*>(
                tensor->template mutable_data<float16>()) +
                chunkBegin,
            &context);
      }
      break;
    case TensorProto_DataType_DOUBLE:
      detail::CopyFromProtoAsIs(
          chunkSize,
          proto.double_data(),
          tensor->template mutable_data<double>() + chunkBegin,
          &context);
      break;
    case TensorProto_DataType_UNDEFINED: {
      Blob temp_blob;
      void* raw_ptr = nullptr;
      for (int i = 0; i < chunkSize; ++i) {
        temp_blob.Deserialize(proto.string_data(i));
        if (i == 0) {
          raw_ptr = tensor->raw_mutable_data(temp_blob.meta());
        }
        temp_blob.meta().copy()(
            temp_blob.GetRaw(),
            static_cast<char*>(raw_ptr) +
                (i + chunkBegin) * temp_blob.meta().itemsize(),
            1);
      }
    }
  }
  context.FinishDeviceComputation();
}

}  // namespace caffe2

#endif  // CAFFE2_CORE_BLOB_SERIALIZATION_H_
