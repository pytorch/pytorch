#include <torch/csrc/jit/runtime/static/ops.h>
#include <ATen/NativeFunctions.h>
#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {
namespace {
inline at::Tensor create_empty_from(const at::Tensor& t) {
  return at::empty({0}, t.options());
}
} // namespace

bool canRunOutOfPlace(Node* n) {
  static std::unordered_set<std::string> out_of_place_nodes{"aten::add",
                                                            "aten::mul",
                                                            "aten::addmm",
                                                            "aten::bmm",
                                                            "aten::sigmoid",
                                                            "aten::cat"};
  auto str = std::string(n->kind().toQualString());
  return out_of_place_nodes.count(str) > 0;
}

// TODO: expand to include all view producing ops, mostly in
// https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/TensorShape.cpp
bool canRunNatively(Node* n) {
  static std::unordered_set<std::string> native_nodes{"aten::transpose",
                                                      "aten::flatten"};
  auto str = std::string(n->kind().toQualString());
  return native_nodes.count(str) > 0;
}

std::function<void(const ProcessedNode*, std::vector<IValue>&)>
getOutOfPlaceOperation(Node* n) {
  if (n->kind() == c10::Symbol::fromQualString("aten::add")) {
    return [=](const ProcessedNode* p_node, std::vector<IValue>& reg) {
      auto in0_t = p_node->Input(0, reg).toTensor();
      auto in1_t = p_node->Input(1, reg).toTensor();
      auto in2_s = p_node->Input(2, reg).toScalar();
      if (p_node->Output(0, reg).isNone()) {
        p_node->Output(0, reg) = create_empty_from(in0_t);
      }
      auto out_t = p_node->Output(0, reg).toTensor();
      at::native::add_out(out_t, in0_t, in1_t, in2_s);
    };
  } else if (n->kind() == c10::Symbol::fromQualString("aten::mul")) {
    return [=](const ProcessedNode* p_node, std::vector<IValue>& reg) {
      auto in0_t = p_node->Input(0, reg).toTensor();
      auto in1_t = p_node->Input(1, reg).toTensor();
      if (p_node->Output(0, reg).isNone()) {
        p_node->Output(0, reg) = create_empty_from(in0_t);
      }
      auto out_t = p_node->Output(0, reg).toTensor();
      at::native::mul_out(out_t, in0_t, in1_t);
    };
  } else if (n->kind() == c10::Symbol::fromQualString("aten::addmm")) {
    return [=](const ProcessedNode* p_node, std::vector<IValue>& reg) {
      auto in0_t = p_node->Input(0, reg).toTensor();
      auto in1_t = p_node->Input(1, reg).toTensor();
      auto in2_t = p_node->Input(2, reg).toTensor();
      auto in3_s = p_node->Input(3, reg).toScalar();
      auto in4_s = p_node->Input(4, reg).toScalar();
      if (p_node->Output(0, reg).isNone()) {
        p_node->Output(0, reg) = create_empty_from(in0_t);
      }
      auto out_t = p_node->Output(0, reg).toTensor();
      at::native::addmm_cpu_out(out_t, in0_t, in1_t, in2_t, in3_s, in4_s);
    };
  } else if (n->kind() == c10::Symbol::fromQualString("aten::clamp")) {
    return [=](const ProcessedNode* p_node, std::vector<IValue>& reg) {
      auto in0_t = p_node->Input(0, reg).toTensor();
      auto in1_s = p_node->Input(1, reg).toScalar();
      auto in2_s = p_node->Input(2, reg).toScalar();
      if (p_node->Output(0, reg).isNone()) {
        p_node->Output(0, reg) = create_empty_from(in0_t);
      }
      auto out_t = p_node->Output(0, reg).toTensor();
      at::native::clamp_out(out_t, in0_t, in1_s, in2_s);
    };
  } else if (n->kind() == c10::Symbol::fromQualString("aten::bmm")) {
    return [=](const ProcessedNode* p_node, std::vector<IValue>& reg) {
      auto in0_t = p_node->Input(0, reg).toTensor();
      auto in1_t = p_node->Input(1, reg).toTensor();
      if (p_node->Output(0, reg).isNone()) {
        p_node->Output(0, reg) = create_empty_from(in0_t);
      }
      auto out_t = p_node->Output(0, reg).toTensor();
      at::native::bmm_out_cpu(out_t, in0_t, in1_t);
    };
  } else if (n->kind() == c10::Symbol::fromQualString("aten::cat")) {
    return [=](const ProcessedNode* p_node, std::vector<IValue>& reg) {
      auto in0_tl = p_node->Input(0, reg).toTensorVector();
      auto in1_i = p_node->Input(1, reg).toInt();
      if (p_node->Output(0, reg).isNone()) {
        p_node->Output(0, reg) = create_empty_from(in0_tl[0]);
      }
      auto out_t = p_node->Output(0, reg).toTensor();
      at::native::_cat_out_cpu(out_t, in0_tl, in1_i);
    };
  } else if (n->kind() == c10::Symbol::fromQualString("aten::sigmoid")) {
    return [=](const ProcessedNode* p_node, std::vector<IValue>& reg) {
      auto in0_t = p_node->Input(0, reg).toTensor();
      if (p_node->Output(0, reg).isNone()) {
        p_node->Output(0, reg) = create_empty_from(in0_t);
      }
      auto out_t = p_node->Output(0, reg).toTensor();
      at::native::sigmoid_out(out_t, in0_t);
    };
  }

  return [](const ProcessedNode*, std::vector<IValue>&) { TORCH_CHECK(0); };
}

std::function<void(const ProcessedNode*, std::vector<IValue>&)>
getNativeOperation(Node* n) {
  if (n->kind() == c10::Symbol::fromQualString("aten::transpose")) {
    return [=](const ProcessedNode* p_node, std::vector<IValue>& reg) {
      auto in0_t = p_node->Input(0, reg).toTensor();
      auto in1_i = p_node->Input(1, reg).toInt();
      auto in2_i = p_node->Input(2, reg).toInt();
      p_node->Output(0, reg) = at::native::transpose(in0_t, in1_i, in2_i);
    };
  } else if (n->kind() == c10::Symbol::fromQualString("aten::flatten")) {
    return [=](const ProcessedNode* p_node, std::vector<IValue>& reg) {
      auto in0_t = p_node->Input(0, reg).toTensor();
      auto in1_i = p_node->Input(1, reg).toInt();
      auto in2_i = p_node->Input(2, reg).toInt();
      p_node->Output(0, reg) = at::native::flatten(in0_t, in1_i, in2_i);
    };
  }
  return [](const ProcessedNode*, std::vector<IValue>&) { TORCH_CHECK(0); };
}

} // namespace jit
} // namespace torch
