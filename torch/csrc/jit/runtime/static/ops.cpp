#include <torch/csrc/jit/runtime/static/ops.h>
#include <ATen/NativeFunctions.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/runtime/vararg_functions.h>

namespace torch {
namespace jit {
namespace {
inline at::Tensor create_empty_from(const at::Tensor& t) {
  return at::empty({0}, t.options());
}
} // namespace

bool canRunOutOfPlace(Node* n) {
  // In alphabetical order
  const static std::unordered_set<std::string> out_of_place_nodes{
      "aten::add",
      "aten::addmm",
      "aten::bmm",
      "aten::cat",
      "aten::clamp",
      "aten::leaky_relu",
      "aten::logit",
      "aten::mul",
      "aten::nan_to_num",
      "aten::relu",
      "aten::sigmoid",
      "aten::stack",
      "aten::tanh"};
  auto str = std::string(n->kind().toQualString());
  return out_of_place_nodes.count(str) > 0;
}

// TODO: expand to include all view producing ops, mostly in
// https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/TensorShape.cpp
bool canRunNatively(Node* n) {
  // In alphabetical order
  const static std::unordered_set<std::string> native_nodes{
      "aten::flatten",
      "aten::transpose",
      "prim::ListConstruct",
      "prim::ListUnpack",
      "prim::TupleConstruct"};
  auto str = std::string(n->kind().toQualString());
  return native_nodes.count(str) > 0;
}

std::function<void(const ProcessedNode*, std::vector<IValue>&)>
getOutOfPlaceOperation(Node* n) {
  if (n->kind() == c10::Symbol::fromQualString("aten::add")) {
    return [](const ProcessedNode* p_node, std::vector<IValue>& reg) {
      auto in0_t = p_node->Input(0, reg).toTensor();
      auto in1_t = p_node->Input(1, reg).toTensor();
      auto in2_s = p_node->Input(2, reg).toScalar();
      if (p_node->Output(0, reg).isNone()) {
        p_node->Output(0, reg) = create_empty_from(in0_t);
      }
      auto out_t = p_node->Output(0, reg).toTensor();
      out_t.resize_({0});
      at::native::add_out(out_t, in0_t, in1_t, in2_s);
    };
  } else if (n->kind() == c10::Symbol::fromQualString("aten::mul")) {
    return [](const ProcessedNode* p_node, std::vector<IValue>& reg) {
      auto in0_t = p_node->Input(0, reg).toTensor();
      auto in1_t = p_node->Input(1, reg).toTensor();
      if (p_node->Output(0, reg).isNone()) {
        p_node->Output(0, reg) = create_empty_from(in0_t);
      }
      auto out_t = p_node->Output(0, reg).toTensor();
      out_t.resize_({0});
      at::native::mul_out(out_t, in0_t, in1_t);
    };
  } else if (n->kind() == c10::Symbol::fromQualString("aten::addmm")) {
    return [](const ProcessedNode* p_node, std::vector<IValue>& reg) {
      auto in0_t = p_node->Input(0, reg).toTensor();
      auto in1_t = p_node->Input(1, reg).toTensor();
      auto in2_t = p_node->Input(2, reg).toTensor();
      auto in3_s = p_node->Input(3, reg).toScalar();
      auto in4_s = p_node->Input(4, reg).toScalar();
      if (p_node->Output(0, reg).isNone()) {
        p_node->Output(0, reg) = create_empty_from(in0_t);
      }
      auto out_t = p_node->Output(0, reg).toTensor();
      out_t.resize_({0});
      at::native::addmm_cpu_out(out_t, in0_t, in1_t, in2_t, in3_s, in4_s);
    };
  } else if (n->kind() == c10::Symbol::fromQualString("aten::clamp")) {
    return [](const ProcessedNode* p_node, std::vector<IValue>& reg) {
      auto in0_t = p_node->Input(0, reg).toTensor();
      auto in1_s = p_node->Input(1, reg).toScalar();
      auto in2_s = p_node->Input(2, reg).toScalar();
      if (p_node->Output(0, reg).isNone()) {
        p_node->Output(0, reg) = create_empty_from(in0_t);
      }
      auto out_t = p_node->Output(0, reg).toTensor();
      out_t.resize_({0});
      at::native::clamp_out(out_t, in0_t, in1_s, in2_s);
    };
  } else if (n->kind() == c10::Symbol::fromQualString("aten::bmm")) {
    return [](const ProcessedNode* p_node, std::vector<IValue>& reg) {
      auto in0_t = p_node->Input(0, reg).toTensor();
      auto in1_t = p_node->Input(1, reg).toTensor();
      if (p_node->Output(0, reg).isNone()) {
        p_node->Output(0, reg) = create_empty_from(in0_t);
      }
      auto out_t = p_node->Output(0, reg).toTensor();
      out_t.resize_({0});
      at::native::bmm_out_cpu(out_t, in0_t, in1_t);
    };
  } else if (n->kind() == c10::Symbol::fromQualString("aten::nan_to_num")) {
    return [](const ProcessedNode* p_node, std::vector<IValue>& reg) {
      auto input_size = p_node->input_regs().size();
      auto in0_t = p_node->Input(0, reg).toTensor();
      double in1_d = input_size > 1 ? p_node->Input(1, reg).toDouble() : 0;
      double in2_d = input_size > 2 ? p_node->Input(2, reg).toDouble()
                                    : std::numeric_limits<double>::infinity();
      double in3_d = input_size > 3 ? p_node->Input(3, reg).toDouble()
                                    : -std::numeric_limits<double>::infinity();
      if (p_node->Output(0, reg).isNone()) {
        p_node->Output(0, reg) = create_empty_from(in0_t);
      }
      auto out_t = p_node->Output(0, reg).toTensor();
      out_t.resize_({0});
      at::native::nan_to_num_out(out_t, in0_t, in1_d, in2_d, in3_d);
    };
  } else if (n->kind() == c10::Symbol::fromQualString("aten::cat")) {
    return [](const ProcessedNode* p_node, std::vector<IValue>& reg) {
      auto in0_tl = p_node->Input(0, reg).toTensorVector();
      auto in1_i = p_node->Input(1, reg).toInt();
      if (p_node->Output(0, reg).isNone()) {
        p_node->Output(0, reg) = create_empty_from(in0_tl[0]);
      }
      auto out_t = p_node->Output(0, reg).toTensor();
      out_t.resize_({0});
      at::native::_cat_out_cpu(out_t, in0_tl, in1_i);
    };
  } else if (n->kind() == c10::Symbol::fromQualString("aten::tanh")) {
    return [](const ProcessedNode* p_node, std::vector<IValue>& reg) {
      auto in0_t = p_node->Input(0, reg).toTensor();
      if (p_node->Output(0, reg).isNone()) {
        p_node->Output(0, reg) = create_empty_from(in0_t);
      }
      auto out_t = p_node->Output(0, reg).toTensor();
      out_t.resize_({0});
      at::native::tanh_out(out_t, in0_t);
    };
  } else if (n->kind() == c10::Symbol::fromQualString("aten::stack")) {
    return [](const ProcessedNode* p_node, std::vector<IValue>& reg) {
      auto inputs = p_node->Input(0, reg).toTensorVector();
      auto dim = p_node->Input(1, reg).toInt();
      if (p_node->Output(0, reg).isNone()) {
        p_node->Output(0, reg) = create_empty_from(inputs[0]);
      }
#ifndef NDEBUG
      at::IntArrayRef entry_shape = inputs[0].sizes();
      for (auto i = 1; i < inputs.size(); i++) {
        TORCH_CHECK(
            inputs[i].sizes() == entry_shape,
            "stack expects each tensor to be equal size, but got ",
            entry_shape,
            " at entry 0 and ",
            inputs[i].sizes(),
            " at entry ",
            i);
      }
#endif
      for (auto i = 0; i < inputs.size(); i++) {
        inputs[i] = inputs[i].unsqueeze(dim);
      }
      auto out_t = p_node->Output(0, reg).toTensor();
      out_t.resize_({0});
      at::native::_cat_out_cpu(out_t, inputs, dim);
    };
  } else if (n->kind() == c10::Symbol::fromQualString("aten::sigmoid")) {
    return [](const ProcessedNode* p_node, std::vector<IValue>& reg) {
      auto in0_t = p_node->Input(0, reg).toTensor();
      if (p_node->Output(0, reg).isNone()) {
        p_node->Output(0, reg) = create_empty_from(in0_t);
      }
      auto out_t = p_node->Output(0, reg).toTensor();
      out_t.resize_({0});
      at::native::sigmoid_out(out_t, in0_t);
    };
  } else if (n->kind() == c10::Symbol::fromQualString("aten::leaky_relu")) {
    auto in1 = toIValue(n->inputs()[1]);
    if (in1) {
      auto in1_s = in1->toScalar();
      return [in1_s](const ProcessedNode* p_node, std::vector<IValue>& reg) {
        auto in0_t = p_node->Input(0, reg).toTensor();
        if (p_node->Output(0, reg).isNone()) {
          p_node->Output(0, reg) = create_empty_from(in0_t);
        }
        auto out_t = p_node->Output(0, reg).toTensor();
        at::native::leaky_relu_out(out_t, in0_t, in1_s);
      };
    } else {
      return [](const ProcessedNode* p_node, std::vector<IValue>& reg) {
        auto in0_t = p_node->Input(0, reg).toTensor();
        auto in1_s = p_node->Input(1, reg).toScalar();
        if (p_node->Output(0, reg).isNone()) {
          p_node->Output(0, reg) = create_empty_from(in0_t);
        }
        auto out_t = p_node->Output(0, reg).toTensor();
        at::native::leaky_relu_out(out_t, in0_t, in1_s);
      };
    }
  } else if (n->kind() == c10::Symbol::fromQualString("aten::relu")) {
    return [](const ProcessedNode* p_node, std::vector<IValue>& reg) {
      auto in0_t = p_node->Input(0, reg).toTensor();
      if (p_node->Output(0, reg).isNone()) {
        p_node->Output(0, reg) = create_empty_from(in0_t);
      }
      auto out_t = p_node->Output(0, reg).toTensor();
      out_t.resize_({0});
      at::native::threshold_out(out_t, in0_t, 0, 0);
    };
  } else if (n->kind() == c10::Symbol::fromQualString("aten::logit")) {
    return [](const ProcessedNode* p_node, std::vector<IValue>& reg) {
      auto in0_t = p_node->Input(0, reg).toTensor();
      auto in1_d = p_node->Input(1, reg).toDouble();
      if (p_node->Output(0, reg).isNone()) {
        p_node->Output(0, reg) = create_empty_from(in0_t);
      }
      auto out_t = p_node->Output(0, reg).toTensor();
      out_t.resize_({0});
      at::native::logit_out(out_t, in0_t, in1_d);
    };
  } else if (n->kind() == c10::Symbol::fromQualString("aten::clone")) {
    return [](const ProcessedNode* p_node, std::vector<IValue>& reg) {
      auto in0_t = p_node->Input(0, reg).toTensor();
      if (p_node->Output(0, reg).isNone()) {
        p_node->Output(0, reg) = create_empty_from(in0_t);
      }
      auto out_t = p_node->Output(0, reg).toTensor();
      at::native::resize_as_(out_t, in0_t, c10::nullopt);
      at::native::copy_(out_t, in0_t, false);
    };
  }
  return [](const ProcessedNode*, std::vector<IValue>&) { TORCH_CHECK(0); };
}

std::function<void(const ProcessedNode*, std::vector<IValue>&)>
getNativeOperation(Node* n) {
  if (n->kind() == c10::Symbol::fromQualString("aten::transpose")) {
    return [](const ProcessedNode* p_node, std::vector<IValue>& reg) {
      auto in0_t = p_node->Input(0, reg).toTensor();
      auto in1_i = p_node->Input(1, reg).toInt();
      auto in2_i = p_node->Input(2, reg).toInt();
      p_node->Output(0, reg) = at::native::transpose(in0_t, in1_i, in2_i);
    };
  } else if (n->kind() == c10::Symbol::fromQualString("aten::flatten")) {
    return [](const ProcessedNode* p_node, std::vector<IValue>& reg) {
      auto in0_t = p_node->Input(0, reg).toTensor();
      auto in1_i = p_node->Input(1, reg).toInt();
      auto in2_i = p_node->Input(2, reg).toInt();
      p_node->Output(0, reg) = at::native::flatten(in0_t, in1_i, in2_i);
    };
  } else if (n->kind() == prim::TupleConstruct) {
    return [](const ProcessedNode* p_node, std::vector<IValue>& reg) {
      // prepare inputs
      std::vector<IValue> stack;
      const size_t size = p_node->input_regs().size();
      stack.reserve(size);
      for (size_t i = 0; i < size; i++) {
        stack.emplace_back(p_node->Input(i, reg));
      }
      // run op
      auto* node = p_node->get_node();
      const auto& type = node->output()->type()->expect<TupleType>();
      if (type->name().has_value()) {
        namedTupleConstruct(stack, type, node->inputs().size());
      } else {
        tupleConstruct(stack, node->inputs().size());
      }
      // put output back
      p_node->Output(0, reg) = std::move(stack[0]);
    };
  } else if (n->kind() == prim::ListConstruct) {
    return [](const ProcessedNode* p_node, std::vector<IValue>& reg) {
      // prepare inputs
      std::vector<IValue> stack;
      const size_t size = p_node->input_regs().size();
      stack.reserve(size);
      for (size_t i = 0; i < size; i++) {
        stack.emplace_back(p_node->Input(i, reg));
      }
      // run op
      listConstruct(
          stack,
          p_node->get_node()->output()->type()->expect<ListType>(),
          p_node->input_regs().size());
      // put output back
      p_node->Output(0, reg) = std::move(stack[0]);
    };
  } else if (n->kind() == prim::ListUnpack) {
    return [](const ProcessedNode* p_node, std::vector<IValue>& reg) {
      // prepare inputs
      std::vector<IValue> stack;
      const size_t size = p_node->input_regs().size();
      stack.reserve(size);
      for (size_t i = 0; i < size; i++) {
        stack.emplace_back(p_node->Input(i, reg));
      }
      // run op
      size_t num_outputs = p_node->output_regs().size();
      listUnpack(stack, num_outputs);
      // put output back
      DCHECK_EQ(stack.size(), num_outputs);
      for (auto i = 0; i < num_outputs; i++) {
        p_node->Output(i, reg) = std::move(stack[i]);
      }
    };
  }
  return [](const ProcessedNode*, std::vector<IValue>&) { TORCH_CHECK(0); };
}

} // namespace jit
} // namespace torch
