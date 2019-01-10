#include "counter_ops.h"

#include "caffe2/core/blob_serialization.h"

namespace caffe2 {
namespace {
/**
 *  @brief CounterSerializer is the serializer for Counter type.
 *
 * CounterSerializer takes in a blob that contains a Counter, and serializes
 * it into a BlobProto protocol buffer. At the moment only int64_t counters are
 * supported (since it's the only once that is really used).
 *
 */
class CounterSerializer : public BlobSerializerBase {
 public:
  CounterSerializer() {}
  ~CounterSerializer() {}

  void Serialize(
      const Blob& blob,
      const string& name,
      SerializationAcceptor acceptor) override {
    CAFFE_ENFORCE(blob.IsType<std::unique_ptr<Counter<int64_t>>>());

    BlobProto blob_proto;
    blob_proto.set_name(name);
    blob_proto.set_type("std::unique_ptr<Counter<int64_t>>");
    TensorProto& proto = *blob_proto.mutable_tensor();
    proto.set_name(name);
    proto.set_data_type(TensorProto_DataType_INT64);
    proto.add_dims(1);
    proto.add_int64_data(
        blob.template Get<std::unique_ptr<Counter<int64_t>>>()->retrieve());
    acceptor(name, blob_proto.SerializeAsString());
  }
};

/**
 * @brief CounterDeserializer is the deserializer for Counters.
 *
 */
class CounterDeserializer : public BlobDeserializerBase {
 public:
  void Deserialize(const BlobProto& proto, Blob* blob) override {
    auto tensorProto = proto.tensor();
    CAFFE_ENFORCE_EQ(tensorProto.dims_size(), 1, "Unexpected size of dims");
    CAFFE_ENFORCE_EQ(tensorProto.dims(0), 1, "Unexpected value of dims");
    CAFFE_ENFORCE_EQ(
        tensorProto.data_type(),
        TensorProto_DataType_INT64,
        "Only int64_t counters supported");
    CAFFE_ENFORCE_EQ(
        tensorProto.int64_data_size(), 1, "Unexpected size of data");
    *blob->GetMutable<std::unique_ptr<Counter<int64_t>>>() =
        caffe2::make_unique<Counter<int64_t>>(tensorProto.int64_data(0));
  }
};
}

// TODO(jiayq): deprecate these ops & consolidate them with
// IterOp/AtomicIterOp

REGISTER_CPU_OPERATOR(CreateCounter, CreateCounterOp<int64_t, CPUContext>);
REGISTER_CPU_OPERATOR(ResetCounter, ResetCounterOp<int64_t, CPUContext>);
REGISTER_CPU_OPERATOR(CountDown, CountDownOp<int64_t, CPUContext>);
REGISTER_CPU_OPERATOR(
    CheckCounterDone,
    CheckCounterDoneOp<int64_t, CPUContext>);
REGISTER_CPU_OPERATOR(CountUp, CountUpOp<int64_t, CPUContext>);
REGISTER_CPU_OPERATOR(RetrieveCount, RetrieveCountOp<int64_t, CPUContext>);

OPERATOR_SCHEMA(CreateCounter)
    .NumInputs(0)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Creates a count-down counter with initial value specified by the 'init_count'
argument.
)DOC")
    .Output(0, "counter", "A blob pointing to an instance of a new counter.")
    .Arg("init_count", "Initial count for the counter, must be >= 0.");

OPERATOR_SCHEMA(ResetCounter)
    .NumInputs(1)
    .NumOutputs(0, 1)
    .SetDoc(R"DOC(
Resets a count-down counter with initial value specified by the 'init_count'
argument.
)DOC")
    .Input(0, "counter", "A blob pointing to an instance of a new counter.")
    .Output(0, "previous_value", "(optional) Previous value of the counter.")
    .Arg("init_count", "Resets counter to this value, must be >= 0.");

OPERATOR_SCHEMA(CountDown)
    .NumInputs(1)
    .NumOutputs(1)
    .SetDoc(R"DOC(
If the internal count value > 0, decreases count value by 1 and outputs false,
otherwise outputs true.
)DOC")
    .Input(0, "counter", "A blob pointing to an instance of a counter.")
    .Output(0, "done", "false unless the internal count is zero.");

OPERATOR_SCHEMA(CheckCounterDone)
    .NumInputs(1)
    .NumOutputs(1)
    .SetDoc(R"DOC(
If the internal count value <= 0, outputs true, otherwise outputs false,
)DOC")
    .Input(0, "counter", "A blob pointing to an instance of a counter.")
    .Output(0, "done", "true if the internal count is zero or negative.");

OPERATOR_SCHEMA(CountUp)
    .NumInputs(1)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Increases count value by 1 and outputs the previous value atomically
)DOC")
    .Input(0, "counter", "A blob pointing to an instance of a counter.")
    .Output(0, "previous_count", "count value BEFORE this operation");

OPERATOR_SCHEMA(RetrieveCount)
    .NumInputs(1)
    .NumOutputs(1)
    .ScalarType(TensorProto::INT64)
    .SetDoc(R"DOC(
Retrieve the current value from the counter.
)DOC")
    .Input(0, "counter", "A blob pointing to an instance of a counter.")
    .Output(0, "count", "current count value.");

SHOULD_NOT_DO_GRADIENT(CreateCounter);
SHOULD_NOT_DO_GRADIENT(ResetCounter);
SHOULD_NOT_DO_GRADIENT(CountDown);
SHOULD_NOT_DO_GRADIENT(CountUp);
SHOULD_NOT_DO_GRADIENT(RetrieveCount);

CAFFE_KNOWN_TYPE(std::unique_ptr<Counter<int64_t>>);
REGISTER_BLOB_SERIALIZER(
    (TypeMeta::Id<std::unique_ptr<Counter<int64_t>>>()),
    CounterSerializer);
REGISTER_BLOB_DESERIALIZER(
    std::unique_ptr<Counter<int64_t>>,
    CounterDeserializer);

} // namespace caffe2
