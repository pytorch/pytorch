#include "caffe2/operators/tensor_protos_db_input.h"

namespace caffe2 {
REGISTER_CPU_OPERATOR(TensorProtosDBInput, TensorProtosDBInput<CPUContext>);

OPERATOR_SCHEMA(TensorProtosDBInput)
  .NumInputs(1)
  .NumOutputs(1, INT_MAX)
  .SetDoc(R"DOC(
TensorProtosDBInput is a simple input operator that basically reads things
from a db where each key-value pair stores an index as key, and a TensorProtos
object as value. These TensorProtos objects should have the same size, and they
will be grouped into batches of the given size. The DB Reader is provided as
input to the operator and it returns as many output tensors as the size of the
TensorProtos object. Each output will simply be a tensor containing a batch of
data with size specified by the 'batch_size' argument containing data from the
corresponding index in the TensorProtos objects in the DB.
)DOC")
  .Arg("batch_size", "(int, default 0) the number of samples in a batch. The "
       "default value of 0 means that the operator will attempt to insert the "
       "entire data in a single output blob.")
  .Input(0, "data", "A pre-initialized DB reader. Typically, this is obtained "
         "by calling CreateDB operator with a db_name and a db_type. The "
         "resulting output blob is a DB Reader tensor")
  .Output(0, "output", "The output tensor in which the batches of data are "
          "returned. The number of output tensors is equal to the size of "
          "(number of TensorProto's in) the TensorProtos objects stored in the "
          "DB as values. Each output tensor will be of size specified by the "
          "'batch_size' argument of the operator");

NO_GRADIENT(TensorProtosDBInput);
}  // namespace caffe2
