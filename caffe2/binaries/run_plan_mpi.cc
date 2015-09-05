#include <mpi.h>

#include "caffe2/core/init.h"
#include "caffe2/core/operator.h"
#include "caffe2/proto/caffe2.pb.h"
#include "caffe2/utils/proto_utils.h"
#include "glog/logging.h"

DEFINE_string(plan, "", "The given path to the plan protobuffer.");

int main(int argc, char** argv) {
  int mpi_ret;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &mpi_ret);
  if (mpi_ret != MPI_THREAD_MULTIPLE) {
    LOG(ERROR) << "Caffe2 MPI requires the underlying MPI to support the "
                  "MPI_THREAD_MULTIPLE mode";
    return 1;
  }
  caffe2::GlobalInit(&argc, &argv);
  gflags::SetUsageMessage("Runs a given plan.");
  LOG(INFO) << "Loading plan: " << FLAGS_plan;
  caffe2::PlanDef plan_def;
  CHECK(ReadProtoFromFile(FLAGS_plan, &plan_def));
  std::unique_ptr<caffe2::Workspace> workspace(new caffe2::Workspace());
  workspace->RunPlan(plan_def);

  // This is to allow us to use memory leak checks.
  google::protobuf::ShutdownProtobufLibrary();
  gflags::ShutDownCommandLineFlags();
  MPI_Finalize();
  return 0;
}
