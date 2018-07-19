#include "gl_operator_test.h"
#include "caffe2/core/timer.h"

namespace caffe2 {

constexpr float tol = 5.0e-2;

// {MaxPool, Relu, Add} followed by pad 1 conv
TEST(OPENGLOperatorTest, ConvMaxPoolConv) {

  Workspace ws;
  auto channel_in = 16;
  auto channel_out = 16;
  auto spatial = 32;
  auto kern = 3;

  PopulateCPUBlob(&ws, true, "cpu_X", {1, channel_in, spatial, spatial}, 1337);
  PopulateCPUBlob(&ws, true, "W", {channel_out, channel_in, kern, kern}, 1337);
  PopulateCPUBlob(&ws, false, "b", {channel_out}, 0);
  PopulateCPUBlob(&ws, true, "W2", {channel_out, channel_in, kern, kern});
  PopulateCPUBlob(&ws, true, "b2", {channel_out});

#define ADD_CONV_ARGS                                                          \
  {                                                                            \
    ADD_ARG((*def), "kernel", i, kern);                                           \
    ADD_ARG((*def), "stride", i, 1);                                              \
    ADD_ARG((*def), "pad", i, 1);                                                 \
    ADD_ARG((*def), "order", s, "NCHW");                                          \
  }

  NetDef cpu_net;
  {
    OperatorDef* def = AddOp(&cpu_net, "Conv", {"cpu_X", "W", "b"}, {"ref_Y"});
    def->set_name("cpu_conv");
    ADD_CONV_ARGS;
  }
  {
    OperatorDef* def = AddOp(&cpu_net, "MaxPool", {"ref_Y"}, {"ref_maxpool"});
    ADD_ARG((*def), "kernel", i, 2);
    ADD_ARG((*def), "pad", i, 0);
    ADD_ARG((*def), "stride_w", i, 2);
    ADD_ARG((*def), "stride_h", i, 2);
    ADD_ARG((*def), "order", s, "NCHW");
  }
  {
    OperatorDef* def = AddOp(&cpu_net, "Conv", {"ref_maxpool", "W2", "b2"}, {"ref_Y2"});
    ADD_CONV_ARGS;
  }

  NetDef gpu_net;
  gpu_net.set_type("opengl");
  {
    OperatorDef* def = AddOp(&gpu_net, "Conv", {"cpu_X", "W", "b"}, {"gpu_Y"});
    MAKE_OPENGL_OPERATOR(def);
    ADD_CONV_ARGS;
  }
  {
    OperatorDef* def = AddOp(&gpu_net, "MaxPool", {"gpu_Y"}, {"gpu_maxpool"});
    ADD_ARG((*def), "kernel", i, 2);
    ADD_ARG((*def), "pad", i, 0);
    ADD_ARG((*def), "stride_w", i, 2);
    ADD_ARG((*def), "stride_h", i, 2);
    ADD_ARG((*def), "order", s, "NCHW");
    MAKE_OPENGL_OPERATOR(def);
  }
  {
    OperatorDef* def = AddOp(&gpu_net, "Conv", {"gpu_maxpool", "W2", "b2"}, {"gpu_Y2"});
    MAKE_OPENGL_OPERATOR(def);
    ADD_CONV_ARGS;
  }

#undef ADD_CONV_ARGS

  // will work after next release of ACL
  // compareNetResult4D(ws, cpu_net, gpu_net, "ref_Y2", "gpu_Y2", tol);
}

TEST(OPENGLOperatorTest, ConvReluConv) {

  Workspace ws;
  auto channel_in = 16;
  auto channel_out = 16;
  auto spatial = 32;
  auto kern = 3;

  PopulateCPUBlob(&ws, true, "cpu_X", {1, channel_in, spatial, spatial}, 1337);
  PopulateCPUBlob(&ws, true, "W", {channel_out, channel_in, kern, kern}, 1337);
  PopulateCPUBlob(&ws, false, "b", {channel_out}, 0);
  PopulateCPUBlob(&ws, true, "W2", {channel_out, channel_in, kern, kern});
  PopulateCPUBlob(&ws, true, "b2", {channel_out});

#define ADD_CONV_ARGS                                                          \
  {                                                                            \
    ADD_ARG((*def), "kernel", i, kern);                                           \
    ADD_ARG((*def), "stride", i, 1);                                              \
    ADD_ARG((*def), "pad", i, 1);                                                 \
    ADD_ARG((*def), "order", s, "NCHW");                                          \
  }

  NetDef cpu_net;
  {
    OperatorDef* def = AddOp(&cpu_net, "Conv", {"cpu_X", "W", "b"}, {"ref_Y"});
    def->set_name("cpu_conv");
    ADD_CONV_ARGS;
  }
  {
    OperatorDef* def = AddOp(&cpu_net, "Relu", {"ref_Y"}, {"ref_relu"});
  }
  {
    OperatorDef* def = AddOp(&cpu_net, "Conv", {"ref_relu", "W2", "b2"}, {"ref_Y2"});
    ADD_CONV_ARGS;
  }

  NetDef gpu_net;
  gpu_net.set_type("opengl");
  {
    OperatorDef* def = AddOp(&gpu_net, "Conv", {"cpu_X", "W", "b"}, {"gpu_Y"});
    MAKE_OPENGL_OPERATOR(def);
    ADD_CONV_ARGS;
  }
  {
    OperatorDef* def = AddOp(&gpu_net, "Relu", {"gpu_Y"}, {"gpu_relu"});
    MAKE_OPENGL_OPERATOR(def);
  }
  {
    OperatorDef* def = AddOp(&gpu_net, "Conv", {"gpu_relu", "W2", "b2"}, {"gpu_Y2"});
    MAKE_OPENGL_OPERATOR(def);
    ADD_CONV_ARGS;
  }

#undef ADD_CONV_ARGS

  // will work after next release of ACL
  // compareNetResult4D(ws, cpu_net, gpu_net, "ref_Y2", "gpu_Y2", tol);

}

TEST(OPENGLOperatorTest, ConvAddConv) {

  Workspace ws;
  auto channel_in = 16;
  auto channel_out = 16;
  auto spatial = 32; // --> 2x2 w no padding, all values 9
  auto kern = 3;

  PopulateCPUBlob(&ws, true, "cpu_X", {1, channel_in, spatial, spatial}, 1337);
  PopulateCPUBlob(&ws, true, "W", {channel_out, channel_in, kern, kern}, 1337);
  PopulateCPUBlob(&ws, false, "b", {channel_out}, 0);
  PopulateCPUBlob(&ws, true, "W2", {channel_out, channel_in, kern, kern});
  PopulateCPUBlob(&ws, true, "b2", {channel_out});
  PopulateCPUBlob(&ws, true, "cpu_Y", {1, channel_in, spatial, spatial}, 1337);

#define ADD_CONV_ARGS                                                          \
  {                                                                            \
    ADD_ARG((*def), "kernel", i, kern);                                           \
    ADD_ARG((*def), "stride", i, 1);                                              \
    ADD_ARG((*def), "pad", i, 1);                                                 \
    ADD_ARG((*def), "order", s, "NCHW");                                          \
  }

  NetDef cpu_net;
  {
    OperatorDef* def = AddOp(&cpu_net, "Conv", {"cpu_X", "W", "b"}, {"ref_Y"});
    def->set_name("cpu_conv");
    ADD_CONV_ARGS;
  }
  {
    OperatorDef* def = AddOp(&cpu_net, "Add", {"ref_Y", "cpu_Y"}, {"ref_add"});
  }
  {
    OperatorDef* def = AddOp(&cpu_net, "Conv", {"ref_add", "W2", "b2"}, {"ref_Y2"});
    ADD_CONV_ARGS;
  }

  NetDef gpu_net;
  gpu_net.set_type("opengl");
  {
    OperatorDef* def = AddOp(&gpu_net, "Conv", {"cpu_X", "W", "b"}, {"gpu_Y"});
    MAKE_OPENGL_OPERATOR(def);
    ADD_CONV_ARGS;
  }
  {
    OperatorDef* def = AddOp(&gpu_net, "Add", {"gpu_Y", "cpu_Y"}, {"gpu_add"});
    MAKE_OPENGL_OPERATOR(def);
  }
  {
    OperatorDef* def = AddOp(&gpu_net, "Conv", {"gpu_add", "W2", "b2"}, {"gpu_Y2"});
    MAKE_OPENGL_OPERATOR(def);
    ADD_CONV_ARGS;
  }
#undef ADD_CONV_ARGS

  // will work after next release of ACL
  // compareNetResult4D(ws, cpu_net, gpu_net, "ref_Y2", "gpu_Y2", tol);

}
} // namespace caffe2
