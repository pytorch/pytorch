#include "gl_operator_test.h"

namespace caffe2 {

TEST(OPENGLOperatorTest, Concat) {

  for (auto Cs: std::vector<std::vector<int>>{
      {4, 4},
      {4, 4, 4},
      {6, 6, 6},
      {16, 8, 4},
      {32, 8, 16, 4},
    }) {
    Workspace ws;
    int batchSize = 1;
    int H = 8;
    int W = 8;
    for (int i = 0; i < Cs.size(); ++i) {
      PopulateCPUBlob(&ws, true, std::string("cpu_X") + caffe2::to_string(i), {batchSize, Cs[i], H, W});
    }

  NetDef cpu_net;
  {
    OperatorDef* def = AddOp(&cpu_net, "Concat", {}, {"ref_Y", "cpu_dummy"});
      for (int i = 0; i < Cs.size(); ++i ) {
        def->add_input(std::string("cpu_X") + caffe2::to_string(i));
      }
  }

  NetDef gpu_net;
  gpu_net.set_type("opengl");
  {
    OperatorDef* def = AddOp(&gpu_net, "Concat", {}, {"gpu_Y", "gpu_dummy"});
    MAKE_OPENGL_OPERATOR(def);
    for (int i = 0; i < Cs.size(); ++i ) {
      def->add_input(std::string("cpu_X") + caffe2::to_string(i));
    }
  }

  compareNetResult(ws, cpu_net, gpu_net);

  }
}

} // namespace caffe2
