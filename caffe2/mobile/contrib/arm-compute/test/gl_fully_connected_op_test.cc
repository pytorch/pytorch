#include "gl_operator_test.h"

namespace caffe2 {

TEST(OPENGLOperatorTest, FC) {

  Workspace ws;
  int batchSize = 1;
  int CIn = 4;
  int H = 8;
  int W = 8;
  int COut = 16;

  PopulateCPUBlob(&ws, true, "cpu_X", {batchSize, CIn, H, W});
  PopulateCPUBlob(&ws, true, "cpu_W", {COut, CIn * H * W});
  PopulateCPUBlob(&ws, true, "cpu_B", {COut});

  constexpr float tol = 0.2;

  NetDef cpu_net;
  {
    AddOp(&cpu_net, "FC", {"cpu_X", "cpu_W", "cpu_B"}, {"ref_Y"});
  }

  NetDef gpu_net;
  gpu_net.set_type("opengl");
  {
    OperatorDef* def = AddOp(&gpu_net, "FC", {"cpu_X", "cpu_W", "cpu_B"}, {"gpu_Y"});
    MAKE_OPENGL_OPERATOR(def);
  }

  // will work after the next release of ACL
  // compareNetResult(ws, cpu_net, gpu_net, "ref_Y", "gpu_Y", tol, true);
}

} // namespace caffe2
