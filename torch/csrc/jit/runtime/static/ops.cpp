#include <torch/csrc/jit/runtime/static/ops.h>
#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

bool canRunOutOfPlace(Node* n) {
  auto str = std::string(n->kind().toQualString());
  if ((str == "aten::add") || (str == "aten::mul") || (str == "aten::addmm") ||
      (str == "aten::bmm") || (str == "aten::sigmoid") ||
      (str == "aten::cat")) {
    return true;
  }
  return false;
}

std::function<void(StaticRuntime::ConstantMap&)> getOutOfPlaceOperation(
    Node* n) {
  auto create_empty_from = [](const at::Tensor& t) {
    return at::empty({0}, t.options());
  };

  if (n->kind() == c10::Symbol::fromQualString("aten::add")) {
    auto out = n->outputs().at(0);
    auto in0 = n->inputs().at(0);
    auto in1 = n->inputs().at(1);
    auto in2 = n->inputs().at(2);
    return [=](StaticRuntime::ConstantMap& ws) {
      auto in0_t = ws.at(in0).toTensor();
      auto in1_t = ws.at(in1).toTensor();
      auto in2_s = ws.at(in2).toScalar();
      if (!ws.count(out)) {
        ws.emplace(out, create_empty_from(in0_t));
      }
      auto out_t = ws.at(out).toTensor();
      at::native::add_out(out_t, in0_t, in1_t, in2_s);
    };
  } else if (n->kind() == c10::Symbol::fromQualString("aten::mul")) {
    auto out = n->outputs().at(0);
    auto in0 = n->inputs().at(0);
    auto in1 = n->inputs().at(1);
    return [=](StaticRuntime::ConstantMap& ws) {
      auto in0_t = ws.at(in0).toTensor();
      auto in1_t = ws.at(in1).toTensor();
      if (!ws.count(out)) {
        ws.emplace(out, create_empty_from(in0_t));
      }
      auto out_t = ws.at(out).toTensor();
      at::native::mul_out(out_t, in0_t, in1_t);
    };
  } else if (n->kind() == c10::Symbol::fromQualString("aten::addmm")) {
    auto out = n->outputs().at(0);
    auto in0 = n->inputs().at(0);
    auto in1 = n->inputs().at(1);
    auto in2 = n->inputs().at(2);
    auto in3 = n->inputs().at(3);
    auto in4 = n->inputs().at(4);
    return [=](StaticRuntime::ConstantMap& ws) {
      auto in0_t = ws.at(in0).toTensor();
      auto in1_t = ws.at(in1).toTensor();
      auto in2_t = ws.at(in2).toTensor();
      auto in3_s = ws.at(in3).toScalar();
      auto in4_s = ws.at(in3).toScalar();
      if (!ws.count(out)) {
        ws.emplace(out, create_empty_from(in0_t));
      }
      auto out_t = ws.at(out).toTensor();
      at::native::addmm_cpu_out(out_t, in0_t, in1_t, in2_t, in3_s, in4_s);
    };
  } else if (n->kind() == c10::Symbol::fromQualString("aten::clamp")) {
    auto out = n->outputs().at(0);
    auto in0 = n->inputs().at(0);
    auto in1 = n->inputs().at(1);
    auto in2 = n->inputs().at(2);
    return [=](StaticRuntime::ConstantMap& ws) {
      auto in0_t = ws.at(in0).toTensor();
      auto in1_s = ws.at(in1).toScalar();
      auto in2_s = ws.at(in2).toScalar();
      if (!ws.count(out)) {
        ws.emplace(out, create_empty_from(in0_t));
      }
      auto out_t = ws.at(out).toTensor();
      at::native::clamp_out(out_t, in0_t, in1_s, in2_s);
    };
  } else if (n->kind() == c10::Symbol::fromQualString("aten::bmm")) {
    auto out = n->outputs().at(0);
    auto in0 = n->inputs().at(0);
    auto in1 = n->inputs().at(1);
    return [=](StaticRuntime::ConstantMap& ws) {
      auto in0_t = ws.at(in0).toTensor();
      auto in1_t = ws.at(in1).toTensor();
      if (!ws.count(out)) {
        ws.emplace(out, create_empty_from(in0_t));
      }
      auto out_t = ws.at(out).toTensor();
      at::native::bmm_out_cpu(out_t, in0_t, in1_t);
    };
  } else if (n->kind() == c10::Symbol::fromQualString("aten::cat")) {
    auto out = n->outputs().at(0);
    auto in0 = n->inputs().at(0);
    auto in1 = n->inputs().at(1);
    return [=](StaticRuntime::ConstantMap& ws) {
      auto in0_tl = ws.at(in0).toTensorVector();
      auto in1_i = ws.at(in1).toInt();
      if (!ws.count(out)) {
        ws.emplace(out, create_empty_from(in0_tl[0]));
      }
      auto out_t = ws.at(out).toTensor();
      at::native::cat_out(out_t, in0_tl, in1_i);
    };
  } else if (n->kind() == c10::Symbol::fromQualString("aten::sigmoid")) {
    auto out = n->outputs().at(0);
    auto in0 = n->inputs().at(0);
    return [=](StaticRuntime::ConstantMap& ws) {
      auto in0_t = ws.at(in0).toTensor();
      if (!ws.count(out)) {
        ws.emplace(out, create_empty_from(in0_t));
      }
      auto out_t = ws.at(out).toTensor();
      at::native::sigmoid_out(out_t, in0_t);
    };
  }

  return [](StaticRuntime::ConstantMap&) { TORCH_CHECK(0); };
}

} // namespace jit
} // namespace torch
