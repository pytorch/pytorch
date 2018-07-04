#include "gl_operator_test.h"

namespace caffe2 {

TEST(OPENGLOperatorTest, Softmax) {

  Workspace ws;
  int N = 1;
  int D = 128;
  PopulateCPUBlob(&ws, true, "cpu_X", {N, D}, 1);

  NetDef cpu_net;
  {
    AddOp(&cpu_net, "Softmax", {"cpu_X"}, {"ref_Y"});
  }

  NetDef gpu_net;
  gpu_net.set_type("opengl");
  {
    OperatorDef* def = AddOp(&gpu_net, "Softmax", {"cpu_X"}, {"gpu_Y"});
    MAKE_OPENGL_OPERATOR(def);
  }

  compareNetResult(ws, cpu_net, gpu_net);

}

} // namespace caffe2
