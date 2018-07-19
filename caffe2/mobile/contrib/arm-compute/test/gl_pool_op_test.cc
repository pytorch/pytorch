#include "gl_operator_test.h"

namespace caffe2 {

TEST(OPENGLOperatorTest, AveragePool) {
  Workspace ws;
  PopulateCPUBlob(&ws, true, "cpu_X", {1, 1, 8, 8});

  NetDef cpu_net;
  {
    OperatorDef* def = AddOp(&cpu_net, "AveragePool", {"cpu_X"}, {"ref_Y"});
    ADD_ARG((*def), "kernel", i, 2);
    ADD_ARG((*def), "pad", i, 0);
    ADD_ARG((*def), "stride", i, 2);
    ADD_ARG((*def), "order", s, "NCHW");
  }

  NetDef gpu_net;
  gpu_net.set_type("opengl");
  {
    OperatorDef* def = AddOp(&gpu_net, "AveragePool", {"cpu_X"}, {"gpu_Y"});
    ADD_ARG((*def), "kernel", i, 2);
    ADD_ARG((*def), "pad", i, 0);
    ADD_ARG((*def), "stride", i, 2);
    ADD_ARG((*def), "order", s, "NCHW");
    MAKE_OPENGL_OPERATOR(def);
  }

  compareNetResult(ws, cpu_net, gpu_net);

}

TEST(OPENGLOperatorTest, MaxPool) {
  Workspace ws;
  PopulateCPUBlob(&ws, true, "cpu_X", {1, 1, 8, 8});

  NetDef cpu_net;
  {
    OperatorDef* def = AddOp(&cpu_net, "MaxPool", {"cpu_X"}, {"ref_Y"});
    ADD_ARG((*def), "kernel", i, 2);
    ADD_ARG((*def), "pad", i, 0);
    ADD_ARG((*def), "stride", i, 2);
    ADD_ARG((*def), "order", s, "NCHW");
  }

  NetDef gpu_net;
  gpu_net.set_type("opengl");
  {
    OperatorDef* def = AddOp(&gpu_net, "MaxPool", {"cpu_X"}, {"gpu_Y"});
    ADD_ARG((*def), "kernel", i, 2);
    ADD_ARG((*def), "pad", i, 0);
    ADD_ARG((*def), "stride", i, 2);
    ADD_ARG((*def), "order", s, "NCHW");
    MAKE_OPENGL_OPERATOR(def);
  }

  compareNetResult(ws, cpu_net, gpu_net);

}

TEST(OPENGLOperatorTest, AverageGlobalPool) {
  Workspace ws;
  PopulateCPUBlob(&ws, true, "cpu_X", {1, 1, 8, 8});

  NetDef cpu_net;
  {
    OperatorDef* def = AddOp(&cpu_net, "AveragePool", {"cpu_X"}, {"ref_Y"});
    ADD_ARG((*def), "global_pooling", i, 1);
    ADD_ARG((*def), "pad", i, 0);
    ADD_ARG((*def), "stride", i, 1);
    ADD_ARG((*def), "order", s, "NCHW");
  }

  NetDef gpu_net;
  gpu_net.set_type("opengl");
  {
    OperatorDef* def = AddOp(&gpu_net, "AveragePool", {"cpu_X"}, {"gpu_Y"});
    ADD_ARG((*def), "global_pooling", i, 1);
    ADD_ARG((*def), "pad", i, 0);
    ADD_ARG((*def), "stride", i, 1);
    ADD_ARG((*def), "order", s, "NCHW");
    MAKE_OPENGL_OPERATOR(def);
  }

  compareNetResult(ws, cpu_net, gpu_net);

}

} // namespace caffe2
