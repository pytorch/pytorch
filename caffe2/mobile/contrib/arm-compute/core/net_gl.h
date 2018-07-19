#ifndef CAFFE2_CORE_NET_GL_H_
#define CAFFE2_CORE_NET_GL_H_

#include <vector>

#include "caffe2/core/common.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/net.h"
#include "caffe2/core/registry.h"
#include "caffe2/core/tensor.h"
#include "caffe2/core/workspace.h"
#include "caffe2/proto/caffe2.pb.h"

namespace caffe2 {

// This is the very basic structure you need to run a network with
// ARM's compute library
class GLNet : public NetBase {
 private:
  bool first_run_ = true;
  Workspace* ws_;
  // record output blob for sync step in operator level benchmarking
  std::vector<string> output_blobs_;
  // record operator type and only sync after gpu op
  std::vector<bool> opengl_device_;
 public:
  GLNet(const std::shared_ptr<const NetDef>& net_def, Workspace* ws);
  bool SupportsAsync() override {
    return false;
  }

  vector<float> TEST_Benchmark(
      const int warmup_runs,
      const int main_runs,
      const bool run_individual) override;

  /*
   * This returns a list of pointers to objects stored in unique_ptrs.
   * Used by Observers.
   *
   * Think carefully before using.
   */
  vector<OperatorBase*> GetOperators() const override {
    vector<OperatorBase*> op_list;
    for (auto& op : operators_) {
      op_list.push_back(op.get());
    }
    return op_list;
  }

 protected:
  bool Run();
  bool RunAsync();
  bool DoRunAsync() override {
    return Run();
  }

  vector<unique_ptr<OperatorBase>> operators_;

  DISABLE_COPY_AND_ASSIGN(GLNet);
};

} // namespace caffe2

#endif // CAFFE2_CORE_NET_SIMPLE_H_
