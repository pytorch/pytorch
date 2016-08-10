#ifndef CAFFE2_OPERATORS_ELEMENTWISE_OP_TEST_H_
#define CAFFE2_OPERATORS_ELEMENTWISE_OP_TEST_H_

#include <iostream>
#include <string>
#include <vector>

#include "caffe2/operators/elementwise_op.h"
#include "gtest/gtest.h"

template <typename Context>
void CopyVector(const int N, const bool* x, bool* y);

template <typename Context>
void FillTensor(
    caffe2::Workspace* ws,
    const std::string& name,
    const std::vector<caffe2::TIndex>& shape,
    const std::vector<uint8_t>& values) {
  auto* blob = ws->CreateBlob(name);
  auto* tensor = blob->GetMutable<caffe2::Tensor<Context>>();
  tensor->Resize(shape);
  auto* mutable_data = tensor->template mutable_data<bool>();
  const bool* data = reinterpret_cast<const bool*>(values.data());
  CopyVector<Context>(values.size(), data, mutable_data);
}

template <typename Context>
caffe2::OperatorDef CreateOperatorDef() {
  caffe2::OperatorDef def;
  return def;
}

template <typename Context>
caffe2::OperatorDef DefineOperator(const std::string& op_type) {
  caffe2::OperatorDef def = CreateOperatorDef<Context>();
  def.set_name("test");
  def.set_type(op_type);
  def.add_input("X");
  def.add_input("Y");
  def.add_output("Z");
  return def;
}

template <typename Context>
void elementwiseAnd() {
  const int N = 4;
  const int M = 2;
  caffe2::Workspace ws;
  auto def = DefineOperator<Context>("And");
  { // equal size
    FillTensor<Context>(&ws, "X", {N}, {true, false, true, false});
    FillTensor<Context>(&ws, "Y", {N}, {true, true, false, false});
    std::unique_ptr<caffe2::OperatorBase> op(caffe2::CreateOperator(def, &ws));
    EXPECT_NE(nullptr, op.get());
    EXPECT_TRUE(op->Run());
    auto* blob = ws.GetBlob("Z");
    EXPECT_NE(nullptr, blob);
    caffe2::TensorCPU Z(blob->Get<caffe2::Tensor<Context>>());
    EXPECT_EQ(Z.size(), N);
    std::vector<bool> result{true, false, false, false};
    for (size_t i = 0; i < Z.size(); ++i) {
      EXPECT_EQ(Z.template data<bool>()[i], result[i]);
    }
  }
  { // broadcast
    auto* arg = def.add_arg();
    arg->set_name("broadcast");
    arg->set_i(1);
    FillTensor<Context>(
        &ws, "X", {M, N}, {true, false, true, false, true, false, true, false});
    FillTensor<Context>(&ws, "Y", {N}, {true, true, false, false});
    std::unique_ptr<caffe2::OperatorBase> op(caffe2::CreateOperator(def, &ws));
    EXPECT_NE(nullptr, op.get());
    EXPECT_TRUE(op->Run());
    auto* blob = ws.GetBlob("Z");
    EXPECT_NE(nullptr, blob);
    caffe2::TensorCPU Z(blob->Get<caffe2::Tensor<Context>>());
    EXPECT_EQ(Z.size(), M * N);
    std::vector<bool> result{
        true, false, false, false, true, false, false, false};
    for (size_t i = 0; i < Z.size(); ++i) {
      EXPECT_EQ(Z.template data<bool>()[i], result[i]);
    }
  }
}

template <typename Context>
void elementwiseOr() {
  const int N = 4;
  const int M = 2;
  caffe2::Workspace ws;
  auto def = DefineOperator<Context>("Or");
  { // equal size
    FillTensor<Context>(&ws, "X", {N}, {true, false, true, false});
    FillTensor<Context>(&ws, "Y", {N}, {true, true, false, false});
    std::unique_ptr<caffe2::OperatorBase> op(caffe2::CreateOperator(def, &ws));
    EXPECT_NE(nullptr, op.get());
    EXPECT_TRUE(op->Run());
    auto* blob = ws.GetBlob("Z");
    EXPECT_NE(nullptr, blob);
    caffe2::TensorCPU Z(blob->Get<caffe2::Tensor<Context>>());
    EXPECT_EQ(Z.size(), N);
    std::vector<bool> result{true, true, true, false};
    for (size_t i = 0; i < Z.size(); ++i) {
      EXPECT_EQ(Z.template data<bool>()[i], result[i]);
    }
  }
  { // broadcast
    auto* arg = def.add_arg();
    arg->set_name("broadcast");
    arg->set_i(1);
    FillTensor<Context>(
        &ws, "X", {M, N}, {true, false, true, false, true, false, true, false});
    FillTensor<Context>(&ws, "Y", {N}, {true, true, false, false});
    std::unique_ptr<caffe2::OperatorBase> op(caffe2::CreateOperator(def, &ws));
    EXPECT_NE(nullptr, op.get());
    EXPECT_TRUE(op->Run());
    auto* blob = ws.GetBlob("Z");
    EXPECT_NE(nullptr, blob);
    caffe2::TensorCPU Z(blob->Get<caffe2::Tensor<Context>>());
    EXPECT_EQ(Z.size(), M * N);
    std::vector<bool> result{true, true, true, false, true, true, true, false};
    for (size_t i = 0; i < Z.size(); ++i) {
      EXPECT_EQ(Z.template data<bool>()[i], result[i]);
    }
  }
}

template <typename Context>
void elementwiseXor() {
  const int N = 4;
  const int M = 2;
  caffe2::Workspace ws;
  auto def = DefineOperator<Context>("Xor");
  { // equal size
    FillTensor<Context>(&ws, "X", {N}, {true, false, true, false});
    FillTensor<Context>(&ws, "Y", {N}, {true, true, false, false});
    std::unique_ptr<caffe2::OperatorBase> op(caffe2::CreateOperator(def, &ws));
    EXPECT_NE(nullptr, op.get());
    EXPECT_TRUE(op->Run());
    auto* blob = ws.GetBlob("Z");
    EXPECT_NE(nullptr, blob);
    caffe2::TensorCPU Z(blob->Get<caffe2::Tensor<Context>>());
    EXPECT_EQ(Z.size(), N);
    std::vector<bool> result{false, true, true, false};
    for (size_t i = 0; i < Z.size(); ++i) {
      EXPECT_EQ(Z.template data<bool>()[i], result[i]);
    }
  }
  { // broadcast
    auto* arg = def.add_arg();
    arg->set_name("broadcast");
    arg->set_i(1);
    FillTensor<Context>(
        &ws, "X", {M, N}, {true, false, true, false, true, false, true, false});
    FillTensor<Context>(&ws, "Y", {N}, {true, true, false, false});
    std::unique_ptr<caffe2::OperatorBase> op(caffe2::CreateOperator(def, &ws));
    EXPECT_NE(nullptr, op.get());
    EXPECT_TRUE(op->Run());
    auto* blob = ws.GetBlob("Z");
    EXPECT_NE(nullptr, blob);
    caffe2::TensorCPU Z(blob->Get<caffe2::Tensor<Context>>());
    EXPECT_EQ(Z.size(), M * N);
    std::vector<bool> result{
        false, true, true, false, false, true, true, false};
    for (size_t i = 0; i < Z.size(); ++i) {
      EXPECT_EQ(Z.template data<bool>()[i], result[i]);
    }
  }
}

template <typename Context>
void elementwiseNot() {
  const int N = 2;
  caffe2::Workspace ws;
  caffe2::OperatorDef def = CreateOperatorDef<Context>();
  def.set_name("test");
  def.set_type("Not");
  def.add_input("X");
  def.add_output("Y");
  FillTensor<Context>(&ws, "X", {N}, {true, false});
  std::unique_ptr<caffe2::OperatorBase> op(caffe2::CreateOperator(def, &ws));
  EXPECT_NE(nullptr, op.get());
  EXPECT_TRUE(op->Run());
  auto* blob = ws.GetBlob("Y");
  EXPECT_NE(nullptr, blob);
  caffe2::TensorCPU Y(blob->Get<caffe2::Tensor<Context>>());
  EXPECT_EQ(Y.size(), N);
  std::vector<bool> result{false, true};
  for (size_t i = 0; i < Y.size(); ++i) {
    EXPECT_EQ(Y.template data<bool>()[i], result[i]);
  }
}

#endif // CAFFE2_OPERATORS_ELEMENTWISE_OP_TEST_H_
