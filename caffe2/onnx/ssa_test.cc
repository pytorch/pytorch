#include "caffe2/core/common.h"
#include "caffe2/onnx/onnx_exporter.h"

#include <gtest/gtest.h>

TEST(SsaTest, ConvReluInplace) {
  caffe2::NetDef net;
  auto* op = net.add_op();
  op->set_type("Conv");
  op->add_input("X");
  op->add_input("W");
  op->add_input("b");
  op->add_output("Y");
  op = net.add_op();
  op->set_type("Relu");
  op->add_input("Y");
  op->add_output("Y");

  caffe2::onnx::SsaRewrite(nullptr, &net);
}
