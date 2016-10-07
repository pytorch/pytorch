#include <mpi.h>

#include "caffe2/core/init.h"
#include "caffe2/core/operator.h"
#include "caffe2/proto/caffe2.pb.h"
#include "caffe2/utils/proto_utils.h"
#include "caffe2/core/logging.h"

CAFFE2_DEFINE_string(plan, "", "The given path to the plan protobuffer.");

int main(int argc, char** argv) {
  caffe2::SetUsageMessage("Runs a caffe2 plan that has MPI operators in it.");
  int mpi_ret;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &mpi_ret);
  if (mpi_ret != MPI_THREAD_MULTIPLE &&
      mpi_ret != MPI_THREAD_SERIALIZED) {
    std::cerr << "Caffe2 MPI requires the underlying MPI to support the "
                 "MPI_THREAD_SERIALIZED or MPI_THREAD_MULTIPLE mode.\n";
    return 1;
  }
  caffe2::GlobalInit(&argc, &argv);
  LOG(INFO) << "Loading plan: " << caffe2::FLAGS_plan;
  caffe2::PlanDef plan_def;
  CAFFE_ENFORCE(ReadProtoFromFile(caffe2::FLAGS_plan, &plan_def));
  std::unique_ptr<caffe2::Workspace> workspace(new caffe2::Workspace());
  workspace->RunPlan(plan_def);

  // This is to allow us to use memory leak checks.
  google::protobuf::ShutdownProtobufLibrary();
  MPI_Finalize();
  return 0;
}
