#include "gl_operator_test.h"

namespace caffe2 {

TEST(OPENGLOperatorTest, ResizeNearest) {

  Workspace ws;
  float height_scale = 2;
  float width_scale = 2;
  int N = 1;
  int CIn = 7;

  PopulateCPUBlob(&ws, true, "cpu_X", {N, CIn, 37, 89});

  NetDef cpu_net;
  {
    OperatorDef* def = AddOp(&cpu_net, "ResizeNearest", {"cpu_X"}, {"ref_Y"});
    ADD_ARG((*def), "height_scale", f, height_scale);
    ADD_ARG((*def), "width_scale", f, width_scale);
  }

  NetDef gpu_net;
  gpu_net.set_type("opengl");
  {
    OperatorDef* def = AddOp(&gpu_net, "ResizeNearest", {"cpu_X"}, {"gpu_Y"});
    MAKE_OPENGL_OPERATOR(def);
    ADD_ARG((*def), "height_scale", f, height_scale);
    ADD_ARG((*def), "width_scale", f, width_scale);
  }

  compareNetResult4D(ws, cpu_net, gpu_net);

}

} // namespace caffe2
