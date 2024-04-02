#include "caffe2/operators/index_ops.h"
#include <atomic>
#include <limits>
#include <mutex>
#include <sstream>
#include <unordered_map>
#include <vector>
#include "caffe2/core/blob_serialization.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/tensor.h"

namespace caffe2 {

// TODO(azzolini): support sizes larger than int32
template <class T>
// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
class IndexCreateOp : public Operator<CPUContext> {
 public:
  template <class... Args>
  explicit IndexCreateOp(Args&&... args)
      : Operator(std::forward<Args>(args)...),
        maxElements_(OperatorBase::GetSingleArgument<int>(
            "max_elements",
            std::numeric_limits<int>::max())) {}

  bool RunOnDevice() override {
    *OperatorBase::Output<std::unique_ptr<IndexBase>>(0) =
        std::unique_ptr<IndexBase>(new Index<T>(maxElements_));
    return true;
  }

 private:
  int64_tValue maxElements_;
};

class IndexGetOp : public Operator<CPUContext> {
 public:
  template <class... Args>
  explicit IndexGetOp(Args&&... args) : Operator(std::forward<Args>(args)...) {}

  bool RunOnDevice() override {
    return DispatchHelper<IndexKeyTypes>::call(this, Input(1));
  }
  template <typename T>
  bool DoRunWithType() {
    auto& base = OperatorBase::Input<std::unique_ptr<IndexBase>>(0);
    auto* dict = dynamic_cast_if_rtti<Index<T>*>(base.get());
    CAFFE_ENFORCE(dict, "Wrong dictionary type given input keys.");
    const auto& keys = Input(1);

    auto* values = Output(0, keys.sizes(), at::dtype<int64_tValue>());
    dict->Get(
        keys.data<T>(),
        values->template mutable_data<int64_tValue>(),
        keys.numel());
    return true;
  }
};

class IndexLoadOp : public Operator<CPUContext> {
 public:
  template <class... Args>
  explicit IndexLoadOp(Args&&... args)
      : Operator(std::forward<Args>(args)...),
        skipFirstEntry_(
            OperatorBase::GetSingleArgument<int>("skip_first_entry", 0)) {}

  bool RunOnDevice() override {
    return DispatchHelper<IndexKeyTypes>::call(this, Input(1));
  }
  template <typename T>
  bool DoRunWithType() {
    auto& base = OperatorBase::Input<std::unique_ptr<IndexBase>>(0);
    auto* dict = dynamic_cast_if_rtti<Index<T>*>(base.get());
    CAFFE_ENFORCE(dict, "Wrong dictionary type given input keys.");
    const auto& keys = Input(1);
    const auto* keys_data = keys.data<T>();
    auto keys_size = keys.numel();
    if (skipFirstEntry_) {
      CAFFE_ENFORCE(keys.numel() > 0);
      ++keys_data;
      --keys_size;
    }
    return dict->Load(keys_data, keys_size);
  }

 private:
  bool skipFirstEntry_;
};

class IndexStoreOp : public Operator<CPUContext> {
 public:
  template <class... Args>
  explicit IndexStoreOp(Args&&... args)
      : Operator(std::forward<Args>(args)...) {}

  bool RunOnDevice() override {
    auto& base = OperatorBase::Input<std::unique_ptr<IndexBase>>(0);
    return DispatchHelper<IndexKeyTypes>::call(this, base->Type());
  }

  template <typename T>
  bool DoRunWithType() {
    auto& base = OperatorBase::Input<std::unique_ptr<IndexBase>>(0);
    auto* dict = dynamic_cast_if_rtti<Index<T>*>(base.get());
    CAFFE_ENFORCE(dict);
    return dict->Store(Output(0));
  }
};

class IndexFreezeOp : public Operator<CPUContext> {
 public:
  template <class... Args>
  explicit IndexFreezeOp(Args&&... args)
      : Operator(std::forward<Args>(args)...) {}

  bool RunOnDevice() override {
    auto& base = OperatorBase::Input<std::unique_ptr<IndexBase>>(0);
    base->Freeze();
    return true;
  }
};

class IndexSizeOp : public Operator<CPUContext> {
 public:
  template <class... Args>
  explicit IndexSizeOp(Args&&... args)
      : Operator(std::forward<Args>(args)...) {}

  bool RunOnDevice() override {
    auto& base = OperatorBase::Input<std::unique_ptr<IndexBase>>(0);

    auto* out = Output(0, std::vector<int64_t>{}, at::dtype<int64_tValue>());
    *out->template mutable_data<int64_tValue>() = base->Size();
    return true;
  }
};

REGISTER_CPU_OPERATOR(IntIndexCreate, IndexCreateOp<int32_t>);
REGISTER_CPU_OPERATOR(LongIndexCreate, IndexCreateOp<int64_t>);
REGISTER_CPU_OPERATOR(StringIndexCreate, IndexCreateOp<std::string>);

REGISTER_CPU_OPERATOR(IndexGet, IndexGetOp);
REGISTER_CPU_OPERATOR(IndexLoad, IndexLoadOp);
REGISTER_CPU_OPERATOR(IndexStore, IndexStoreOp);
REGISTER_CPU_OPERATOR(IndexFreeze, IndexFreezeOp);
REGISTER_CPU_OPERATOR(IndexSize, IndexSizeOp);

OPERATOR_SCHEMA(IntIndexCreate)
    .NumInputs(0)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Creates a dictionary that maps int32 keys to consecutive integers
from 1 to max_elements. Zero is reserved for unknown keys.
)DOC")
    .Arg("max_elements", "Max number of elements, including the zero entry.")
    .Output(0, "handler", "Pointer to an Index instance.")
    .ScalarType(TensorProto_DataType_UNDEFINED);

OPERATOR_SCHEMA(LongIndexCreate)
    .NumInputs(0)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Creates a dictionary that maps int64 keys to consecutive integers
from 1 to max_elements. Zero is reserved for unknown keys.
)DOC")
    .Arg("max_elements", "Max number of elements, including the zero entry.")
    .Output(0, "handler", "Pointer to an Index instance.")
    .ScalarType(TensorProto_DataType_UNDEFINED);

OPERATOR_SCHEMA(StringIndexCreate)
    .NumInputs(0)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Creates a dictionary that maps string keys to consecutive integers
from 1 to max_elements. Zero is reserved for unknown keys.
)DOC")
    .Arg("max_elements", "Max number of elements, including the zero entry.")
    .Output(0, "handle", "Pointer to an Index instance.")
    .ScalarType(TensorProto_DataType_UNDEFINED);

OPERATOR_SCHEMA(IndexGet)
    .NumInputs(2)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Given an index handle and a tensor of keys, return an Int tensor of same shape
containing the indices for each of the keys. If the index is frozen, unknown
entries are given index 0. Otherwise, new entries are added into the index.
If an insert is necessary but max_elements has been reached, fail.
)DOC")
    .Input(0, "handle", "Pointer to an Index instance.")
    .Input(1, "keys", "Tensor of keys to be looked up.")
    .Output(0, "indices", "Indices for each of the keys.")
    .ScalarType(TensorProto::INT64);

OPERATOR_SCHEMA(IndexFreeze)
    .NumInputs(1)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Freezes the given index, disallowing creation of new index entries.
Should not be called concurrently with IndexGet.
)DOC")
    .Input(0, "handle", "Pointer to an Index instance.")
    .Output(0, "handle", "The input handle.")
    .EnforceInplace({{0, 0}})
    .ScalarType(TensorProto_DataType_UNDEFINED);

OPERATOR_SCHEMA(IndexLoad)
    .NumInputs(2)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Loads the index from the given 1-D tensor. Elements in the tensor will be given
consecutive indexes starting at 1. Fails if tensor contains repeated elements.
)DOC")
    .Input(0, "handle", "Pointer to an Index instance.")
    .Input(1, "items", "1-D tensor with elements starting with index 1.")
    .Output(0, "handle", "The input handle.")
    .EnforceInplace({{0, 0}})
    .Arg(
        "skip_first_entry",
        "If set, skips the first entry of the tensor. This allows "
        "to load tensors that are aligned with an embedding, where the first "
        "entry corresponds to the default 0 index entry.")
    .ScalarType(TensorProto_DataType_UNDEFINED);

OPERATOR_SCHEMA(IndexStore)
    .NumInputs(1)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Stores the keys of this index in a 1-D tensor. Since element 0 is reserved
for unknowns, the first element of the output tensor will be element of index 1.
)DOC")
    .Input(0, "handle", "Pointer to an Index instance.")
    .Output(0, "items", "1-D tensor with elements starting with index 1.");

OPERATOR_SCHEMA(IndexSize)
    .NumInputs(1)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Returns the number of entries currently present in the index.
)DOC")
    .Input(0, "handle", "Pointer to an Index instance.")
    .Output(0, "items", "Scalar int64 tensor with number of entries.");

NO_GRADIENT(IndexGetOp);
NO_GRADIENT(IntIndexCreate);
NO_GRADIENT(LongIndexCreate);
NO_GRADIENT(StringIndexCreate);
SHOULD_NOT_DO_GRADIENT(IndexFreeze);
SHOULD_NOT_DO_GRADIENT(IndexLoad);
SHOULD_NOT_DO_GRADIENT(IndexStore);
SHOULD_NOT_DO_GRADIENT(IndexSize);

class IndexSerializer : public BlobSerializerBase {
 public:
  // NOLINTNEXTLINE(modernize-use-equals-default)
  IndexSerializer() {}
  // NOLINTNEXTLINE(modernize-use-equals-default)
  ~IndexSerializer() override {}

  void Serialize(
      const void* pointer,
      TypeMeta typeMeta,
      const string& name,
      SerializationAcceptor acceptor) override {
    CAFFE_ENFORCE(typeMeta.Match<std::unique_ptr<IndexBase>>());
    const auto& base = *static_cast<const std::unique_ptr<IndexBase>*>(pointer);
    Blob tensor_blob;
    auto* tensor_out = BlobGetMutableTensor(&tensor_blob, CPU);

    if (base->Type().Match<std::string>()) {
      doStore<std::string>(base, tensor_out);
    } else if (base->Type().Match<int32_t>()) {
      doStore<int32_t>(base, tensor_out);
    } else if (base->Type().Match<int64_t>()) {
      doStore<int64_t>(base, tensor_out);
    } else {
      CAFFE_THROW("Index of this type can't be serialized.");
    }

    CAFFE_ENFORCE(
        tensor_out->numel() <= std::numeric_limits<int32_t>::max(),
        "Index too large to be serialized.");
    BlobProto blob_proto;
    TensorSerializer ser;
    ser.Serialize(
        *tensor_out, name, blob_proto.mutable_tensor(), 0, tensor_out->numel());
    blob_proto.set_name(name);
    blob_proto.set_type("std::unique_ptr<caffe2::IndexBase>");

    std::ostringstream os;
    os << base->maxElements() << " " << base->isFrozen();
    blob_proto.set_content(os.str());

    acceptor(name, SerializeBlobProtoAsString_EnforceCheck(blob_proto));
  }

 private:
  template <typename T>
  void doStore(const std::unique_ptr<IndexBase>& base, Tensor* tensor_out) {
    auto* dict = dynamic_cast_if_rtti<Index<T>*>(base.get());
    CAFFE_ENFORCE(dict, "Wrong dictionary type.");
    dict->Store(tensor_out);
  }
};

class IndexDeserializer : public BlobDeserializerBase {
 public:
  void Deserialize(const BlobProto& proto, Blob* blob) override {
    TensorDeserializer deser;
    Blob tensor_blob;
    deser.Deserialize(proto, &tensor_blob);

    std::istringstream is(proto.content());
    int64_t maxElements{std::numeric_limits<int64_t>::max()};
    bool isFrozen{false};
    is >> maxElements >> isFrozen;

    auto& tensor_in = tensor_blob.template Get<Tensor>();
    auto* base = blob->template GetMutable<std::unique_ptr<IndexBase>>();

    if (tensor_in.IsType<std::string>()) {
      doLoad<std::string>(base, maxElements, tensor_in);
    } else if (tensor_in.IsType<int32_t>()) {
      doLoad<int32_t>(base, maxElements, tensor_in);
    } else if (tensor_in.IsType<int64_t>()) {
      doLoad<int64_t>(base, maxElements, tensor_in);
    } else {
      CAFFE_THROW("Index of this type cannot be deserialized.");
    }

    if (isFrozen) {
      (*base)->Freeze();
    }
  }

 private:
  template <typename T>
  void doLoad(
      std::unique_ptr<IndexBase>* base,
      int64_t maxElements,
      const Tensor& tensor_in) {
    base->reset(new Index<T>(maxElements));
    auto* dict = dynamic_cast_if_rtti<Index<T>*>(base->get());
    dict->Load(tensor_in.data<T>(), tensor_in.numel());
  }
};

CAFFE_KNOWN_TYPE(std::unique_ptr<caffe2::IndexBase>);

REGISTER_BLOB_SERIALIZER(
    (TypeMeta::Id<std::unique_ptr<caffe2::IndexBase>>()),
    IndexSerializer);
REGISTER_BLOB_DESERIALIZER(
    std::unique_ptr<caffe2::IndexBase>,
    IndexDeserializer);

} // namespace caffe2
