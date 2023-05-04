#include "counter_ops.h"
#include "caffe2/core/blob_serialization.h"

namespace caffe2 {

const char* githubLinks = R"DOC(
  Github Links:
  - https://github.com/pytorch/pytorch/blob/main/caffe2/operators/counter_ops.cc

)DOC";

const char* kCountExample = R"DOC(
<details>

<summary> <b>Example</b> </summary>

**Code**

```
workspace.ResetWorkspace()

createcounter_op = core.CreateOperator(
    "CreateCounter",
    [],
    ["counter"],
    init_count=5
)

retrievecount_op = core.CreateOperator(
    "RetrieveCount",
    ["counter"],
    ["count"]
)

checkcounterdone_op = core.CreateOperator(
    "CheckCounterDone",
    ["counter"],
    ["done"]
)

countup_op = core.CreateOperator(
    "CountUp",
    ["counter"],
    ["previous_count"],
)

countdown_op = core.CreateOperator(
    "CountDown",
    ["counter"],
    ["done"],
)

resetcounter_op = core.CreateOperator(
    "ResetCounter",
    ["counter"],
    ["previous_count"],
    init_count=3
)


// Create counter
workspace.RunOperatorOnce(createcounter_op)
print("'counter' pointer:", workspace.FetchBlob("counter"))


// Retrieve initial counter value
workspace.RunOperatorOnce(retrievecount_op)
print("Initial 'count':", workspace.FetchBlob("count"))


// Check if counter is done
workspace.RunOperatorOnce(checkcounterdone_op)
print("Initial 'done' value:", workspace.FetchBlob("done"))


// Test CountUp operator
print("\nTesting CountUp operator...")
for i in range(5):
    workspace.RunOperatorOnce(countup_op)
    print("'previous_count' after CountUp:", workspace.FetchBlob("previous_count"))

workspace.RunOperatorOnce(retrievecount_op)
print("'count' value after CountUp test:", workspace.FetchBlob("count"))


// Test CountDown operator
print("\nTesting CountDown operator...")
for i in range(11):
    workspace.RunOperatorOnce(countdown_op)
    workspace.RunOperatorOnce(retrievecount_op)
    print("'count' value after CountDown: {}\t'done' value: {}".format(workspace.FetchBlob("count"), workspace.FetchBlob("done")))
```

**Result**

```
'counter' pointer: counter, a C++ native class of type std::__1::unique_ptr<caffe2::Counter<long long>, std::__1::default_delete<caffe2::Counter<long long> > >.
Initial 'count': 5
Initial 'done' value: False

Testing CountUp operator...
'previous_count' after CountUp: 5
'previous_count' after CountUp: 6
'previous_count' after CountUp: 7
'previous_count' after CountUp: 8
'previous_count' after CountUp: 9
'count' value after CountUp test: 10

Testing CountDown operator...
'count' value after CountDown: 9        'done' value: False
'count' value after CountDown: 8        'done' value: False
'count' value after CountDown: 7        'done' value: False
'count' value after CountDown: 6        'done' value: False
'count' value after CountDown: 5        'done' value: False
'count' value after CountDown: 4        'done' value: False
'count' value after CountDown: 3        'done' value: False
'count' value after CountDown: 2        'done' value: False
'count' value after CountDown: 1        'done' value: False
'count' value after CountDown: 0        'done' value: False
'count' value after CountDown: -1        'done' value: True
```

</details>

)DOC";

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
  // NOLINTNEXTLINE(modernize-use-equals-default)
  CounterSerializer() {}
  // NOLINTNEXTLINE(modernize-use-equals-default)
  ~CounterSerializer() override {}

  void Serialize(
      const void* pointer,
      TypeMeta typeMeta,
      const string& name,
      SerializationAcceptor acceptor) override {
    CAFFE_ENFORCE(typeMeta.Match<std::unique_ptr<Counter<int64_t>>>());

    BlobProto blob_proto;
    blob_proto.set_name(name);
    blob_proto.set_type("std::unique_ptr<Counter<int64_t>>");
    TensorProto& proto = *blob_proto.mutable_tensor();
    proto.set_name(name);
    proto.set_data_type(TensorProto_DataType_INT64);
    proto.add_dims(1);
    proto.add_int64_data(
        (*static_cast<const std::unique_ptr<Counter<int64_t>>*>(pointer))
            ->retrieve());
    acceptor(name, SerializeBlobProtoAsString_EnforceCheck(blob_proto));
  }
};

/**
 * @brief CounterDeserializer is the deserializer for Counters.
 *
 */
class CounterDeserializer : public BlobDeserializerBase {
 public:
  void Deserialize(const BlobProto& proto, Blob* blob) override {
    // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
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
        std::make_unique<Counter<int64_t>>(tensorProto.int64_data(0));
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
Creates a count-down counter with initial value specified by the `init_count`
argument.

)DOC" + (string) githubLinks + (string) kCountExample)
    .Output(
        0,
        "counter",
        "*(type: Tensor`<ptr>`)* A blob pointing to an instance of a new counter.")
    .Arg(
        "init_count",
        "*(type: int; default: 0)* Initial count for the counter, must be >= 0.");

OPERATOR_SCHEMA(ResetCounter)
    .NumInputs(1)
    .NumOutputs(0, 1)
    .SetDoc(R"DOC(
Resets a count-down counter with initial value specified by the `init_count`
argument.
)DOC" + (string) githubLinks + (string) kCountExample)
    .Input(
        0,
        "counter",
        "*(type: Tensor`<ptr>`)* A blob pointing to an instance of a counter.")
    .Output(
        0,
        "previous_value",
        "*(type: int)* [OPTIONAL] count value BEFORE this operation.")
    .Arg(
        "init_count",
        "*(type: int; default: 0)* Resets counter to this value, must be >= 0.");

OPERATOR_SCHEMA(CountDown)
    .NumInputs(1)
    .NumOutputs(1)
    .SetDoc(R"DOC(
If the internal count value > 0, decreases count value by 1 and outputs False,
otherwise outputs True.
)DOC" + (string) githubLinks + (string) kCountExample)
    .Input(
        0,
        "counter",
        "*(type: Tensor`<ptr>`)* A blob pointing to an instance of a counter.")
    .Output(
        0,
        "done",
        "*(type: bool)* False unless the internal count is zero.");

OPERATOR_SCHEMA(CheckCounterDone)
    .NumInputs(1)
    .NumOutputs(1)
    .SetDoc(R"DOC(
If the internal count value <= 0, outputs true, otherwise outputs false.
)DOC" + (string) githubLinks + (string) kCountExample)
    .Input(
        0,
        "counter",
        "*(type: Tensor`<ptr>`)* A blob pointing to an instance of a counter.")
    .Output(
        0,
        "done",
        "*(type: bool)* True if the internal count is zero or negative, otherwise False.");

OPERATOR_SCHEMA(CountUp)
    .NumInputs(1)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Increases count value by 1 and outputs the previous value atomically.
)DOC" + (string) githubLinks + (string) kCountExample)
    .Input(
        0,
        "counter",
        "*(type: Tensor`<ptr>`)* A blob pointing to an instance of a counter.")
    .Output(
        0,
        "previous_count",
        "*(type: int)* Count value BEFORE this operation.");

OPERATOR_SCHEMA(RetrieveCount)
    .NumInputs(1)
    .NumOutputs(1)
    .ScalarType(TensorProto::INT64)
    .SetDoc(R"DOC(
Retrieve the current value from the counter as an integer.
)DOC" + (string) githubLinks + (string) kCountExample)
    .Input(
        0,
        "counter",
        "*(type: Tensor`<ptr>`)* A blob pointing to an instance of a counter.")
    .Output(
        0,
        "count",
        "*(type: int)* Current count value.");

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
