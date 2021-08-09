#include "caffe2/sgd/iter_op.h"

#ifdef CAFFE2_USE_MKLDNN
#include <caffe2/ideep/operators/operator_fallback_ideep.h>
#include <caffe2/ideep/utils/ideep_operator.h>
#endif

namespace caffe2 {

void MutexSerializer::Serialize(
    const void* pointer,
    TypeMeta typeMeta,
    const string& name,
    BlobSerializerBase::SerializationAcceptor acceptor) {
  CAFFE_ENFORCE(typeMeta.Match<std::unique_ptr<std::mutex>>());
  BlobProto blob_proto;
  blob_proto.set_name(name);
  blob_proto.set_type("std::unique_ptr<std::mutex>");
  blob_proto.set_content("");
  acceptor(name, SerializeBlobProtoAsString_EnforceCheck(blob_proto));
}

void MutexDeserializer::Deserialize(const BlobProto& /* unused */, Blob* blob) {
  *blob->GetMutable<std::unique_ptr<std::mutex>>() =
      std::make_unique<std::mutex>();
}

REGISTER_CPU_OPERATOR(Iter, IterOp<CPUContext>);
REGISTER_CPU_OPERATOR(AtomicIter, AtomicIterOp<CPUContext>);

#ifdef CAFFE2_USE_MKLDNN
REGISTER_IDEEP_OPERATOR(AtomicIter, IDEEPFallbackOp<AtomicIterOp<CPUContext>>);
#endif

REGISTER_BLOB_SERIALIZER(
    (TypeMeta::Id<std::unique_ptr<std::mutex>>()),
    MutexSerializer);
REGISTER_BLOB_DESERIALIZER(std::unique_ptr<std::mutex>, MutexDeserializer);

OPERATOR_SCHEMA(Iter)
    .NumInputs(0, 1)
    .NumOutputs(1)
    .EnforceInplace({{0, 0}})
    .SetDoc(R"DOC(
Stores a singe integer, that gets incremented on each call to Run().
Useful for tracking the iteration count during SGD, for example.
)DOC");

OPERATOR_SCHEMA(AtomicIter)
    .NumInputs(2)
    .NumOutputs(1)
    .EnforceInplace({{1, 0}})
    .IdenticalTypeAndShapeOfInput(1)
    .SetDoc(R"DOC(
Similar to Iter, but takes a mutex as the first input to make sure that
updates are carried out atomically. This can be used in e.g. Hogwild sgd
algorithms.
)DOC")
    .Input(0, "mutex", "The mutex used to do atomic increment.")
    .Input(1, "iter", "The iter counter as an int64_t TensorCPU.");

NO_GRADIENT(Iter);
NO_GRADIENT(AtomicIter);
} // namespace caffe2
