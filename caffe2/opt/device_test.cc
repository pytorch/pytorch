#include "caffe2/core/common.h"
#include "caffe2/opt/converter.h"
#include "caffe2/opt/device.h"

#include <gtest/gtest.h>

using namespace nom::repr;

#define ADD_ARG(_op, _name, _type, _val)    \
  {                                         \
    caffe2::Argument* arg = _op->add_arg(); \
    arg->set_name(_name);                   \
    arg->set_##_type(_val);                 \
  }

TEST(DeviceTest, InsertCopies) {
  caffe2::NetDef net;
  for (auto i = 0; i < 9; ++i) {
    if (i % 3 == 0) {
      caffe2::OperatorDef* def = net.add_op();
      def->set_type("Conv");
      def->add_input("X");
      def->add_input("W" + c10::to_string(i));
      def->add_input("b" + c10::to_string(i));
      ADD_ARG(def, "kernel", i, 3);
      ADD_ARG(def, "stride", i, 1);
      ADD_ARG(def, "pad", i, 0);
      ADD_ARG(def, "order", s, "NCHW");
      def->add_output("X");
      def->mutable_device_option()->set_device_type(caffe2::PROTO_CPU);
    } else {
      caffe2::OperatorDef* def = net.add_op();
      def->set_type("Relu");
      def->add_input("X");
      def->add_output("X");
      def->mutable_device_option()->set_device_type(caffe2::PROTO_CPU);
    }
  }
  auto nn = caffe2::convertToNNModule(net);

  for (auto node : nn.dataFlow.getMutableNodes()) {
    if (nn::is<Relu>(node)) {
      auto annot = nn::get<NeuralNetOperator>(node)->getMutableAnnotation();
      auto c2_annot = dyn_cast<caffe2::Caffe2Annotation>(annot);
      c2_annot->setDeviceType(caffe2::PROTO_OPENCL);
    }
  }

  caffe2::opt::insertCopies(
      &nn,
      [](NNGraph::NodeRef node) {
        // Ignore all tensors
        if (!nn::is<NeuralNetOperator>(node)) {
          return true;
        }
        auto annot = nn::get<NeuralNetOperator>(node)->getMutableAnnotation();
        NOM_REQUIRE_OR_RET_FALSE(annot);
        auto c2_annot = dyn_cast<caffe2::Caffe2Annotation>(annot);
        NOM_REQUIRE_OR_RET_FALSE(c2_annot);
        return c2_annot->getDeviceType() == caffe2::PROTO_OPENCL;
      },
      [](NNGraph& g) {
        return g.createNode(std::make_unique<GenericOperator>());
      },
      [](NNGraph& g) {
        return g.createNode(std::make_unique<GenericOperator>());
      });

  auto proto = caffe2::convertToCaffe2Proto(nn, net);

  // Conv -> Relu -> Relu
  // becomes
  // Conv -> Generic -> Relu -> Relu -> Generic
  // thus
  // 9 ops of this pattern becomes 15
  EXPECT_EQ(proto.op().size(), 15);
}
