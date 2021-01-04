#include <torch/csrc/jit/runtime/static/ops.h>
#include <ATen/NativeFunctions.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/runtime/vararg_functions.h>

namespace torch {
namespace jit {

C10_DEFINE_REGISTRY(SROperatorRegistry, SROperatorFunctor);

bool canRunOutOfPlace(Node* n) {
  auto op_name = std::string(n->kind().toQualString());
  return SROperatorRegistry()->Has(op_name);
}

bool canReuseInputs(Node* n) {
  auto op_name = std::string(n->kind().toQualString());
  DCHECK(SROperatorRegistry()->Has(op_name));
  return SROperatorRegistry()->Create(op_name)->CanReuseInput();
}

bool canReuseOutputs(Node* n) {
  auto op_name = std::string(n->kind().toQualString());
  DCHECK(SROperatorRegistry()->Has(op_name));
  return SROperatorRegistry()->Create(op_name)->CanReuseOutput();
}

// TODO: expand to include all view producing ops, mostly in
// https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/TensorShape.cpp
bool canRunNatively(Node* n) {
  // In alphabetical order
  const static std::unordered_set<std::string> native_nodes{
      "aten::flatten",
      "aten::narrow",
      "aten::reshape",
      "aten::slice",
      "aten::transpose",
      "aten::to",
      "prim::ListConstruct",
      "prim::ListUnpack",
      "prim::TupleConstruct"};
  auto str = std::string(n->kind().toQualString());
  if (!native_nodes.count(str)) {
    return false;
  }
  if (str == "aten::to") {
    return n->inputs().size() == 5;
  }
  return true;
}

// TODO: PLEASE DON'T COPY PASTE THIS, this is copy pasted
// generated code to unblock, need to make this nicer
struct static_add final : public at::native::structured_add_out {
  static_add(at::Tensor& output) : output_(output) {}
  void set_output(
      int64_t output_idx,
      at::IntArrayRef sizes,
      at::IntArrayRef strides,
      at::TensorOptions options,
      at::DimnameList names) override {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(output_idx == 0);
    // NB: do NOT use resize_output as it will complain if not zero sized.
    at::native::resize_(output_, sizes);
    if (!strides.empty()) {
      TORCH_INTERNAL_ASSERT(!options.memory_format_opt().has_value());
      output_.as_strided_(sizes, strides);
    } else if (options.memory_format_opt().has_value()) {
      output_.unsafeGetTensorImpl()->empty_tensor_restride(
          *options.memory_format_opt());
    }
  }
  const at::Tensor& maybe_get_output(int64_t output_idx) override {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(output_idx == 0);
    return output_;
  }
  at::Tensor& output_;
};

REGISTER_OPERATOR_FUNCTOR(aten::add, aten_add, [](Node* n) -> SROperator {
  return [](const ProcessedNode* p_node, std::vector<IValue>& reg) {
    auto in0_t = p_node->Input(0, reg).toTensor();
    auto in1_t = p_node->Input(1, reg).toTensor();
    auto in2_s = p_node->Input(2, reg).toScalar();
    if (p_node->Output(0, reg).isNone()) {
      p_node->Output(0, reg) = create_empty_from(in0_t);
    }
    auto out_t = p_node->Output(0, reg).toTensor();
    static_add op{out_t};
    op.meta(in0_t, in1_t, in2_s);
    op.impl(out_t, in0_t, in1_t, in2_s);
  };
});

REGISTER_OPERATOR_FUNCTOR(aten::mul, aten_mul, [](Node* n) -> SROperator {
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
});

REGISTER_OPERATOR_FUNCTOR(aten::addmm, aten_addmm, [](Node* n) -> SROperator {
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
});

REGISTER_OPERATOR_FUNCTOR(aten::clamp, aten_clamp, [](Node* n) -> SROperator {
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
});

REGISTER_OPERATOR_FUNCTOR(aten::bmm, aten_bmm, [](Node* n) -> SROperator {
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
});

REGISTER_OPERATOR_FUNCTOR(
    aten::nan_to_num,
    aten_nan_to_num,
    [](Node* n) -> SROperator {
      return [](const ProcessedNode* p_node, std::vector<IValue>& reg) {
        auto input_size = p_node->input_regs().size();
        auto in0_t = p_node->Input(0, reg).toTensor();
        double in1_d = input_size > 1 ? p_node->Input(1, reg).toDouble() : 0;
        double in2_d = input_size > 2 ? p_node->Input(2, reg).toDouble()
                                      : std::numeric_limits<double>::infinity();
        double in3_d = input_size > 3
            ? p_node->Input(3, reg).toDouble()
            : -std::numeric_limits<double>::infinity();
        if (p_node->Output(0, reg).isNone()) {
          p_node->Output(0, reg) = create_empty_from(in0_t);
        }
        auto out_t = p_node->Output(0, reg).toTensor();
        out_t.resize_({0});
        at::native::nan_to_num_out(out_t, in0_t, in1_d, in2_d, in3_d);
      };
    });
REGISTER_OPERATOR_FUNCTOR(aten::cat, aten_cat, [](Node* n) -> SROperator {
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
});
REGISTER_OPERATOR_FUNCTOR(aten::tanh, aten_tanh, [](Node* n) -> SROperator {
  return [](const ProcessedNode* p_node, std::vector<IValue>& reg) {
    auto in0_t = p_node->Input(0, reg).toTensor();
    if (p_node->Output(0, reg).isNone()) {
      p_node->Output(0, reg) = create_empty_from(in0_t);
    }
    auto out_t = p_node->Output(0, reg).toTensor();
    out_t.resize_({0});
    at::native::tanh_out(out_t, in0_t);
  };
});

// Split out into a function to appease MSVC's pre-processor
SROperator aten_stack(Node* n) {
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
}

REGISTER_OPERATOR_FUNCTOR(aten::stack, aten_stack, aten_stack);

REGISTER_OPERATOR_FUNCTOR(
    aten::sigmoid,
    aten_sigmoid,
    [](Node* n) -> SROperator {
      return [](const ProcessedNode* p_node, std::vector<IValue>& reg) {
        auto in0_t = p_node->Input(0, reg).toTensor();
        if (p_node->Output(0, reg).isNone()) {
          p_node->Output(0, reg) = create_empty_from(in0_t);
        }
        auto out_t = p_node->Output(0, reg).toTensor();
        out_t.resize_({0});
        at::native::sigmoid_out(out_t, in0_t);
      };
    });
REGISTER_OPERATOR_FUNCTOR(
    aten::leaky_relu,
    aten_leaky_relu,
    [](Node* n) -> SROperator {
      auto in1 = toIValue(n->inputs()[1]);
      if (in1) {
        auto in1_s = in1->toScalar();
        return [=](const ProcessedNode* p_node, std::vector<IValue>& reg) {
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
    });
REGISTER_OPERATOR_FUNCTOR(aten::relu, aten_relu, [](Node* n) -> SROperator {
  return [](const ProcessedNode* p_node, std::vector<IValue>& reg) {
    auto in0_t = p_node->Input(0, reg).toTensor();
    if (p_node->Output(0, reg).isNone()) {
      p_node->Output(0, reg) = create_empty_from(in0_t);
    }
    auto out_t = p_node->Output(0, reg).toTensor();
    out_t.resize_({0});
    at::native::threshold_out(out_t, in0_t, 0, 0);
  };
});
REGISTER_OPERATOR_FUNCTOR(aten::logit, aten_logit, [](Node* n) -> SROperator {
  return [](const ProcessedNode* p_node, std::vector<IValue>& reg) {
    auto in0_t = p_node->Input(0, reg).toTensor();
    double in1_d = p_node->input_regs().size() > 1
        ? p_node->Input(1, reg).toDouble()
        : -1.0;
    if (p_node->Output(0, reg).isNone()) {
      p_node->Output(0, reg) = create_empty_from(in0_t);
    }
    auto out_t = p_node->Output(0, reg).toTensor();
    out_t.resize_({0});
    at::native::logit_out(out_t, in0_t, in1_d);
  };
});
REGISTER_OPERATOR_FUNCTOR(aten::clone, aten_clone, [](Node* n) -> SROperator {
  return [](const ProcessedNode* p_node, std::vector<IValue>& reg) {
    auto in0_t = p_node->Input(0, reg).toTensor();
    if (p_node->Output(0, reg).isNone()) {
      p_node->Output(0, reg) = create_empty_from(in0_t);
    }
    auto out_t = p_node->Output(0, reg).toTensor();
    at::native::resize_as_(out_t, in0_t, c10::nullopt);
    at::native::copy_(out_t, in0_t, false);
  };
});

std::function<void(const ProcessedNode*, std::vector<IValue>&)>
getOutOfPlaceOperation(Node* n) {
  auto op_name = n->kind().toQualString();
  if (SROperatorRegistry()->Has(op_name)) {
    return SROperatorRegistry()->Create(op_name)->Generate(n);
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
  } else if (n->kind() == c10::Symbol::fromQualString("aten::permute")) {
    return [](const ProcessedNode* p_node, std::vector<IValue>& reg) {
      auto in0_t = p_node->Input(0, reg).toTensor();
      auto in1_iv = p_node->Input(1, reg).toIntVector();
      p_node->Output(0, reg) = at::native::permute(in0_t, in1_iv);
    };
  } else if (n->kind() == c10::Symbol::fromQualString("aten::reshape")) {
    return [](const ProcessedNode* p_node, std::vector<IValue>& reg) {
      auto in0_t = p_node->Input(0, reg).toTensor();
      auto in1_iv = p_node->Input(1, reg).toIntVector();
      p_node->Output(0, reg) = at::native::reshape(in0_t, in1_iv);
    };
  } else if (n->kind() == c10::Symbol::fromQualString("aten::slice")) {
    return [](const ProcessedNode* p_node, std::vector<IValue>& reg) {
      auto in0_t = p_node->Input(0, reg).toTensor();
      auto in1_i = p_node->Input(1, reg).toInt();
      auto in2_i = p_node->Input(2, reg).toInt();
      auto in3_i = p_node->Input(3, reg).toInt();
      auto in4_i = p_node->Input(4, reg).toInt();
      p_node->Output(0, reg) =
          at::native::slice(in0_t, in1_i, in2_i, in3_i, in4_i);
    };
  } else if (n->kind() == c10::Symbol::fromQualString("aten::narrow")) {
    return [](const ProcessedNode* p_node, std::vector<IValue>& reg) {
      auto self = p_node->Input(0, reg).toTensor(); // self
      auto dim = p_node->Input(1, reg).toInt(); // dim
      int64_t start = 0;
      if (p_node->Input(2, reg).isScalar()) {
        start = p_node->Input(2, reg).toInt();
      } else {
        auto t = p_node->Input(2, reg).toTensor();
        start = t.item<int64_t>();
      }
      auto length = p_node->Input(3, reg).toInt(); // length
      TORCH_CHECK(
          self.dim() > 0, "narrow() cannot be applied to a 0-dim tensor.");
      auto cur_size = self.size(dim);
      if (start != cur_size && start < 0) { // start being the end is valid, but
                                            // not a valid dim specification.
        start = at::maybe_wrap_dim(start, cur_size);
      }
      TORCH_CHECK(
          length >= 0 && start <= cur_size - length,
          "start (",
          start,
          ") + length (",
          length,
          ") exceeds dimension size (",
          cur_size,
          ").");
      p_node->Output(0, reg) =
          at::native::slice(self, dim, start, start + length, 1);
    };
  } else if (n->kind() == c10::Symbol::fromQualString("aten::to")) {
    return [](const ProcessedNode* p_node, std::vector<IValue>& reg) {
      DCHECK(p_node->input_regs().size() == 5);
      auto in0_t = p_node->Input(0, reg).toTensor();
      auto in1_i = p_node->Input(1, reg).toScalarType();
      auto in2_i = p_node->Input(2, reg).toBool();
      auto in3_i = p_node->Input(3, reg).toBool();
      if (p_node->Input(4, reg).isNone()) {
        p_node->Output(0, reg) =
            at::native::to(in0_t, in1_i, in2_i, in3_i, c10::nullopt);
      } else {
        auto in4_o = p_node->Input(4, reg).toMemoryFormat();
        p_node->Output(0, reg) =
            at::native::to(in0_t, in1_i, in2_i, in3_i, in4_o);
      }
    };
  }
  return [](const ProcessedNode*, std::vector<IValue>&) { TORCH_CHECK(0); };
}

} // namespace jit
} // namespace torch
