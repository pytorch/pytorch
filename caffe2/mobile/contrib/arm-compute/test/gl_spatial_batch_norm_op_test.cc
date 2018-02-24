#include "gl_operator_test.h"

namespace caffe2 {

TEST(OPENGLOperatorTest, SpatialBN) {

  Workspace ws;
  int batchSize = 1;
  int channels = 8;

  PopulateCPUBlob(&ws, true, "cpu_X", {3, channels, 8, 13});
  PopulateCPUBlob(&ws, true, "cpu_scale", {channels});
  PopulateCPUBlob(&ws, true, "cpu_bias", {channels});
  PopulateCPUBlob(&ws, true, "cpu_mean", {channels});
  PopulateCPUBlob(&ws, true, "cpu_var", {channels}, 1, 0.5);

  NetDef cpu_net;
  {
    OperatorDef* def = AddOp(&cpu_net, "SpatialBN", {"cpu_X", "cpu_scale", "cpu_bias", "cpu_mean", "cpu_var"}, {"ref_Y"});
    ADD_ARG((*def), OpSchema::Arg_IsTest, i, 1);
  }

  NetDef gpu_net;
  gpu_net.set_type("opengl");
  {
    OperatorDef* def = AddOp(&gpu_net, "SpatialBN", {"cpu_X", "cpu_scale", "cpu_bias", "cpu_mean", "cpu_var"}, {"gpu_Y"});
    MAKE_OPENGL_OPERATOR(def);
    ADD_ARG((*def), OpSchema::Arg_IsTest, i, 1);
  }

  compareNetResult4D(ws, cpu_net, gpu_net, "ref_Y", "gpu_Y", 0.01);

}

} // namespace caffe2
