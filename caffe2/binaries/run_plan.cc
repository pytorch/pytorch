#include "caffe2/core/operator.h"
#include "caffe2/proto/caffe2.pb.h"
#include "caffe2/utils/proto_utils.h"
#include "caffe2/binaries/gflags_namespace.h"
#include "glog/logging.h"

DEFINE_string(plan, "", "The given path to the plan protobuffer.");

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  gflags::SetUsageMessage("Runs a given plan.");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  LOG(INFO) << "Loading plan: " << FLAGS_plan;
  caffe2::PlanDef plan_def;
  CHECK(ReadProtoFromFile(FLAGS_plan, &plan_def));
  std::unique_ptr<caffe2::Workspace> workspace(new caffe2::Workspace());
  workspace->RunPlan(plan_def);

  // This is to allow us to use memory leak checks.
  google::protobuf::ShutdownProtobufLibrary();
  gflags::ShutDownCommandLineFlags();
  return 0;
}
