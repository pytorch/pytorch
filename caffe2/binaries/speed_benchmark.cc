#include "caffe2/core/init.h"
#include "caffe2/core/operator.h"
#include "caffe2/proto/caffe2.pb.h"
#include "caffe2/utils/proto_utils.h"
#include "caffe2/core/logging.h"

CAFFE2_DEFINE_string(net, "", "The given net to benchmark.");
CAFFE2_DEFINE_string(init_net, "",
                     "The given net to initialize any parameters.");
CAFFE2_DEFINE_int(warmup, 0, "The number of iterations to warm up.");
CAFFE2_DEFINE_int(iter, 10, "The number of iterations to run.");
CAFFE2_DEFINE_bool(run_individual, false, "Whether to benchmark individual operators.");

int main(int argc, char** argv) {
  caffe2::GlobalInit(&argc, &argv);
  std::unique_ptr<caffe2::Workspace> workspace(new caffe2::Workspace());
  // Run initialization network.
  caffe2::NetDef net_def;
  CAFFE_ENFORCE(ReadProtoFromFile(caffe2::FLAGS_init_net, &net_def));
  CAFFE_ENFORCE(workspace->RunNetOnce(net_def));
  CAFFE_ENFORCE(ReadProtoFromFile(caffe2::FLAGS_net, &net_def));
  caffe2::NetBase* net = workspace->CreateNet(net_def);
  CHECK_NOTNULL(net);
  CAFFE_ENFORCE(net->Run());
  net->TEST_Benchmark(caffe2::FLAGS_warmup, caffe2::FLAGS_iter, caffe2::FLAGS_run_individual);
  return 0;
}
