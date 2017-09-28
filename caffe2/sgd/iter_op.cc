/**
 * Copyright (c) 2016-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "caffe2/sgd/iter_op.h"

namespace caffe2 {

void MutexSerializer::Serialize(
    const Blob& blob,
    const string& name,
    BlobSerializerBase::SerializationAcceptor acceptor) {
  CAFFE_ENFORCE(blob.IsType<std::unique_ptr<std::mutex>>());
  BlobProto blob_proto;
  blob_proto.set_name(name);
  blob_proto.set_type("std::unique_ptr<std::mutex>");
  blob_proto.set_content("");
  acceptor(name, blob_proto.SerializeAsString());
}

void MutexDeserializer::Deserialize(const BlobProto& /* unused */, Blob* blob) {
  *blob->GetMutable<std::unique_ptr<std::mutex>>() =
      caffe2::make_unique<std::mutex>();
}

REGISTER_CPU_OPERATOR(Iter, IterOp<CPUContext>);
REGISTER_CPU_OPERATOR(AtomicIter, AtomicIterOp<CPUContext>);

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
    .SetDoc(R"DOC(
Similar to Iter, but takes a mutex as the first input to make sure that
updates are carried out atomically. This can be used in e.g. Hogwild sgd
algorithms.
)DOC")
    .Input(0, "mutex", "The mutex used to do atomic increment.")
    .Input(1, "iter", "The iter counter as an int64_t TensorCPU.");

NO_GRADIENT(Iter);
NO_GRADIENT(AtomicIter);
}  // namespace caffe2
