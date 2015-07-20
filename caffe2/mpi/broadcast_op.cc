#include <mpi.h>

#include "caffe2/core/operator.h"
#include "caffe2/mpi/mpi_common.h"

namespace caffe2 {

// BroadcastOp does Broadcast using MPI.
template <typename dtype, class DeviceContext>
class BroadcastOp final : public Operator<dtype, DeviceContext> {
 public:
  USE_OPERATOR_BASE_FUNCTIONS;
  BroadcastOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<dtype, DeviceContext>(operator_def, ws),
        root_(OperatorBase::template GetSingleArgument<int>("root", 0)) {
    CHECK_EQ(operator_def.input(0), operator_def.output(0))
        << "Broadcast is an in-place operator.";
  }
  ~BroadcastOp() {}

  bool RunOnDevice() {
    auto* output = Output(0);
    CHECK_GT(output->size(), 0);
    MPI_Bcast(static_cast<void*>(output->mutable_data()), output->size(),
              MPIDataTypeWrapper<dtype>::type(), root_, MPI_COMM_WORLD);
    return true;
  }

 protected:
  int root_;
  // Input: X. Output: X.
  // Note that Broadcast works in-place by definition.
  INPUT_OUTPUT_STATS(1, 1, 1, 1);
  DISABLE_COPY_AND_ASSIGN(BroadcastOp);
};

namespace {
REGISTER_CPU_OPERATOR(Broadcast, BroadcastOp<float, CPUContext>);

}

}  // namespace caffe2
