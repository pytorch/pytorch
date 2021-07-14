#include <gtest/gtest.h>
#include "caffe2/opt/converter.h"
#include "caffe2/opt/distributed.h"

caffe2::NetDef fakeNet() {
  caffe2::NetDef net;

  {
    caffe2::OperatorDef* def = net.add_op();
    def->set_type("Fake");
    def->add_input("X");
    def->add_output("Y");
  }
  {
    caffe2::OperatorDef* def = net.add_op();
    def->set_type("Fake");
    def->add_input("Y");
    def->add_output("Z");
  }
  {
    caffe2::OperatorDef* def = net.add_op();
    def->set_type("Fake");
    def->add_input("Z");
    def->add_input("X");
    def->add_output("W");
  }
  net.add_external_input("X");
  net.add_external_output("Y");
  net.add_external_output("W");

  return net;
}

// Common usage
using namespace nom::repr;

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(Converter, DeclareExport) {
  auto net = fakeNet();
  caffe2::injectDataEdgeIndicators(&net);
  auto nn = caffe2::convertToNNModule(net);


  // This is in nom::repr
  auto inputs = nn::filter<Declare>(nn);
  auto outputs = nn::filter<Export>(nn);

  auto count = 0;
  for (const auto& declareNode : inputs) {
    count++;
    // This call fails an assertion if it isn't true
    auto delcare_op = nn::get<Declare>(declareNode);
    // String version of name can be extracted like this
    EXPECT_EQ(delcare_op->getName(), "Declare");

    // What used to be external_input (note that getOutputs returns a vector)
    auto inputNode = nn::getOutputs(declareNode).at(0);

    // Key idea is that we are working with nodes that hold things,
    // so nn::get<T> is very commonly used
    auto input = nn::get<Tensor>(inputNode);
    // We only had one external input in the original net,
    // so this should be true
    EXPECT_EQ(input->getName(), "X");
  }
  // Only 1 external input
  EXPECT_EQ(count, 1);

  // Reset for external output
  count = 0;
  for (const auto& exportNode : outputs) {
    count++;
  }
  // 2 external outputs
  EXPECT_EQ(count, 2);
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(Distributed, InsertDeviceOptions) {
  auto net = fakeNet();
  caffe2::injectDataEdgeIndicators(&net);
  auto nn = caffe2::convertToNNModule(net);
  caffe2::DeviceOption d;
  d.set_device_type(1337);
  caffe2::addBlobDeviceOptions({{"X", d}, {"Y", d}, {"W", d}}, &nn);

  for (auto& ns : {nn::filter<Declare>(nn), nn::filter<Export>(nn)}) {
    for (auto& node : ns) {
      auto op = nn::get<NeuralNetOperator>(node);
      auto annot = dyn_cast<caffe2::Caffe2Annotation>(op->getAnnotation());
      // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
      auto d = annot->getDeviceOption();
      EXPECT_EQ(d.device_type(), 1337);
    }
  }
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(Distributed, InsertDeviceOptionsFailureCase) {
  auto net = fakeNet();
  caffe2::injectDataEdgeIndicators(&net);
  auto nn = caffe2::convertToNNModule(net);
  caffe2::DeviceOption d;
  d.set_device_type(1337);
  // We can only use correct blob names, expect failure otherwise
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  EXPECT_THROW(
      {
        caffe2::addBlobDeviceOptions(
            {{"X", d}, {"Y", d}, {"W", d}, {"FAKE", d}}, &nn);
      },
      std::exception);
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(Converter, InjectDataEdgeIndicators) {
  auto net = fakeNet();

  auto nn = caffe2::convertToNNModule(net);
  caffe2::injectDataEdgeIndicators(&nn);
  auto new_net = caffe2::convertToCaffe2Proto(nn);

  EXPECT_EQ(new_net.op_size(), 3 + 1 + 2); // Inserted 1 Declare and 2 Export

  auto declare_count = 0;
  auto export_count = 0;
  for (const auto& op : new_net.op()) {
    declare_count += op.type() == "Declare";
    export_count += op.type() == "Export";
  }
  EXPECT_EQ(declare_count, 1);
  EXPECT_EQ(export_count, 2);

  // Remove them from the network
  EXPECT_EQ(new_net.external_input_size(), 0);
  EXPECT_EQ(new_net.external_output_size(), 0);

  auto new_nn = caffe2::convertToNNModule(new_net);
  caffe2::removeDataEdgeIndicators(&new_nn);
  new_net = caffe2::convertToCaffe2Proto(new_nn);

  for (const auto& op : new_net.op()) {
    EXPECT_NE(op.type(), "Declare");
    EXPECT_NE(op.type(), "Export");
  }

  EXPECT_EQ(new_net.external_input_size(), 1);
  EXPECT_EQ(new_net.external_output_size(), 2);
}

// Main usage
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(Converter, OverloadedConvertToNNModule) {
  auto net = fakeNet();
  caffe2::DeviceOption d;
  d.set_device_type(1337);
  auto nn = caffe2::convertToNNModule(net, {{"X", d}, {"Y", d}, {"W", d}});

  for (auto& ns : {nn::filter<Declare>(nn), nn::filter<Export>(nn)}) {
    for (auto& node : ns) {
      auto op = nn::get<NeuralNetOperator>(node);
      auto annot = dyn_cast<caffe2::Caffe2Annotation>(op->getAnnotation());
      // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
      auto d = annot->getDeviceOption();
      EXPECT_EQ(d.device_type(), 1337);
    }
  }
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(Converter, OverloadedConvertToNNModuleFailure) {
  auto net = fakeNet();
  caffe2::DeviceOption d;
  d.set_device_type(1337);
  // We can only use correct blob names, expect failure otherwise
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  EXPECT_THROW(
      {
        auto nn = caffe2::convertToNNModule(
            net, {{"X", d}, {"Y", d}, {"W", d}, {"FAKE", d}});
      },
      std::exception);
}
