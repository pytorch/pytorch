#include "gl_operator_test.h"

namespace caffe2 {

constexpr float tol = 5.0e-2;

TEST(OPENGLOperatorTest, NormPlanarYUV) {

  Workspace ws;
  int batchSize = 1;
  int channels = 8;

  PopulateCPUBlob(&ws, true, "cpu_X", {batchSize, channels, 8, 13});

  PopulateCPUBlob(&ws, true, "cpu_mean", {1, channels});
  PopulateCPUBlob(&ws, true, "cpu_stddev", {1, channels}, 1, 0.5);

  NetDef cpu_net;
  {
    OperatorDef* def = AddOp(&cpu_net, "NormalizePlanarYUV", {"cpu_X", "cpu_mean", "cpu_stddev"}, {"ref_Y"});
  }

  NetDef gpu_net;
  gpu_net.set_type("opengl");
  {
    OperatorDef* def = AddOp(&gpu_net, "NormalizePlanarYUV", {"cpu_X", "cpu_mean", "cpu_stddev"}, {"gpu_Y"});
    MAKE_OPENGL_OPERATOR(def);
  }

  compareNetResult4D(ws, cpu_net, gpu_net);
}

} // namespace caffe2
