#include "caffe2/core/init.h"
#include "caffe2/core/operator.h"
#include "caffe2/proto/caffe2.pb.h"
#include "caffe2/utils/proto_utils.h"
#include "caffe2/core/logging.h"

CAFFE2_DEFINE_string(plan, "", "The given path to the plan protobuffer.");

int main(int argc, char** argv) {
  caffe2::GlobalInit(&argc, &argv);
  if (caffe2::FLAGS_plan.size() == 0) {
    LOG(ERROR) << "No plan specified. Use --plan=/path/to/plan.";
    return 0;
  }
  LOG(INFO) << "Loading plan: " << caffe2::FLAGS_plan;
  caffe2::PlanDef plan_def;
  CAFFE_ENFORCE(ReadProtoFromFile(caffe2::FLAGS_plan, &plan_def));
  std::unique_ptr<caffe2::Workspace> workspace(new caffe2::Workspace());
  workspace->RunPlan(plan_def);

  // This is to allow us to use memory leak checks.
  google::protobuf::ShutdownProtobufLibrary();
  return 0;
}
