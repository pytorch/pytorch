#include <mpi.h>

#include "caffe2/core/operator.h"
#include "caffe2/mpi/mpi_common.h"

namespace caffe2 {

// AllreduceOp does Allreduce using MPI. Currently, only SUM is supported.
template <typename dtype, class DeviceContext>
class AllreduceOp final : public Operator<dtype, DeviceContext> {
 public:
  USE_OPERATOR_BASE_FUNCTIONS;
  USE_SIMPLE_CTOR_DTOR(AllreduceOp);

  bool RunOnDevice() {
    auto& input = Input(0);
    auto* output = Output(0);
    output->ReshapeLike(input);
    MPI_Allreduce(const_cast<dtype*>(input.data()),
                  output->mutable_data(), input.size(),
                  MPIDataTypeWrapper<dtype>::type(), MPI_SUM, MPI_COMM_WORLD);
    return true;
  }

 protected:
  // Input: X; Output: X_reduced.
  INPUT_OUTPUT_STATS(1, 1, 1, 1);
  DISABLE_COPY_AND_ASSIGN(AllreduceOp);
};

namespace {
REGISTER_CPU_OPERATOR(Allreduce, AllreduceOp<float, CPUContext>);
// Note: Allreduce does not work on CUDA devices as of OpenMPI 1.8.4 yet. In the
// future we can simply initialize it here.
}

}  // namespace caffe2
