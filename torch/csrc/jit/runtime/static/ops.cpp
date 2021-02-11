#include <torch/csrc/jit/runtime/static/ops.h>

#include <ATen/CPUFunctions.h>
#include <ATen/InferSize.h>
#include <ATen/NativeFunctions.h>
#include <ATen/TensorUtils.h>
#include <ATen/native/IndexingUtils.h>
#include <ATen/native/TensorAdvancedIndexing.h>
#include <ATen/native/quantized/cpu/qembeddingbag.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/runtime/vararg_functions.h>

namespace at {
namespace native {
// The out variants of view ops can't be moved to aten because they don't
// exactly follow the semantics of the aten ops. aten::reshape/flatten create
// views, t, that are tracked by autograd and t.is_view() returns true. Here
// t.is_view() would return false instead.
at::Tensor& reshape_out(
    at::Tensor& out,
    const at::Tensor& self,
    const std::vector<int64_t>& proposed_shape,
    bool infer_size = true) {
  auto shape = infer_size ? at::infer_size(proposed_shape, self.numel())
                          : proposed_shape;
  auto stride = at::detail::computeStride(self.sizes(), self.strides(), shape);

  if (stride.has_value()) {
    // create view
    if (!out.defined() || !out.storage().is_alias_of(self.storage())) {
      auto impl = c10::make_intrusive<c10::TensorImpl>(
          c10::Storage(self.storage()), self.key_set(), self.dtype());
      out = at::Tensor(std::move(impl));
    }

    c10::TensorImpl* impl = out.unsafeGetTensorImpl();
    impl->set_storage_offset(self.storage_offset());
    impl->set_sizes_and_strides(shape, *stride);
  } else {
    // copy over tensor
    if (!out.defined()) {
      out = at::native::empty_like(
          self, self.options(), at::MemoryFormat::Contiguous);
    }
    // copy first and set shape/strides later. It doesn't work the other way
    // around.
    at::native::copy_(out, self);
    stride = at::detail::computeStride(out.sizes(), out.strides(), shape);
    c10::TensorImpl* impl = out.unsafeGetTensorImpl();
    impl->set_sizes_and_strides(shape, *stride);
  }
  // namedinference::propagate_names(output, self);
  return out;
}

at::Tensor& flatten_out(
    at::Tensor& out,
    const at::Tensor& self,
    int64_t start_dim,
    int64_t end_dim) {
  start_dim =
      start_dim < 0 ? c10::maybe_wrap_dim(start_dim, self.dim()) : start_dim;
  end_dim = end_dim < 0 ? c10::maybe_wrap_dim(end_dim, self.dim()) : end_dim;
  TORCH_CHECK(
      start_dim <= end_dim,
      "flatten() has invalid args: start_dim cannot come after end_dim");

  if (self.dim() == 0) {
    return reshape_out(out, self, {1}, false);
  }

  if (start_dim == end_dim) {
    out = self;
    return out;
  }

  // We don't want to infer_size on the entire shape, because that can give us
  // an extra degree of freedom we don't want; for example, consider shape [0,
  // 1, 3, 0], with start_dim=1, end_dim=2. It's clear we want result shape [0,
  // 3, 0] but passing [0, -1, 0] to infer_size means the -1 can take on any
  // value and satisfy the constraints.
  auto iter = self.sizes().data();
  auto slice_numel = std::accumulate(
      iter + start_dim,
      iter + end_dim + 1,
      static_cast<int64_t>(1),
      std::multiplies<int64_t>());

  std::vector<int64_t> shape;
  shape.reserve(self.dim() - end_dim + start_dim);
  for (int64_t i = 0; i < start_dim; i++) {
    shape.push_back(self.sizes()[i]);
  }
  shape.push_back(slice_numel);
  for (int64_t i = end_dim + 1; i < self.dim(); i++) {
    shape.push_back(self.sizes()[i]);
  }
  return reshape_out(out, self, shape, false);
}
} // namespace native
} // namespace at

namespace torch {
namespace jit {

C10_DEFINE_REGISTRY(SROperatorRegistry, SROperatorFunctor);
// View ops with out variants are registered separately
C10_DEFINE_REGISTRY(SRViewOperatorRegistry, SROperatorFunctor);

bool canRunOutOfPlace(Node* n) {
  auto op_name = std::string(n->kind().toQualString());
  return SROperatorRegistry()->Has(op_name) ||
      SRViewOperatorRegistry()->Has(op_name);
}

// The inputs/outputs of view ops do not participate in memory reuse
bool canReuseInputsOutputs(Node* n) {
  auto op_name = std::string(n->kind().toQualString());
  return !SRViewOperatorRegistry()->Has(op_name);
}

bool isViewOp(Node* n) {
  auto op_name = std::string(n->kind().toQualString());
  return SRViewOperatorRegistry()->Has(op_name);
}

bool canReuseInputs(Node* n) {
  auto op_name = std::string(n->kind().toQualString());
  if (SROperatorRegistry()->Has(op_name)) {
    return SROperatorRegistry()->Create(op_name)->CanReuseInput();
  }
  return false;
}

bool canReuseOutputs(Node* n) {
  auto op_name = std::string(n->kind().toQualString());
  if (SROperatorRegistry()->Has(op_name)) {
    return SROperatorRegistry()->Create(op_name)->CanReuseOutput();
  }
  return false;
}

// TODO: expand to include all view producing ops, mostly in
// https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/TensorShape.cpp
bool canRunNatively(Node* n) {
  // In alphabetical order
  const static std::unordered_set<std::string> native_nodes{
      "aten::flatten",
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

REGISTER_OPERATOR_FUNCTOR(aten::add, aten_add, [](Node* n) -> SROperator {
  return [](ProcessedNode* p_node) {
    auto& in0_t = p_node->Input(0).toTensor();
    auto& in1_t = p_node->Input(1).toTensor();
    auto in2_s = p_node->Input(2).toScalar();
    if (p_node->Output(0).isNone()) {
      p_node->Output(0) = create_empty_from(in0_t);
    }
    auto& out_t = p_node->Output(0).toTensor();
    fastResizeToZero(out_t);
    at::cpu::add_out(out_t, in0_t, in1_t, in2_s);
  };
});

REGISTER_OPERATOR_FUNCTOR(aten::mul, aten_mul, [](Node* n) -> SROperator {
  return [](ProcessedNode* p_node) {
    auto& in0_t = p_node->Input(0).toTensor();
    auto& in1_t = p_node->Input(1).toTensor();
    if (p_node->Output(0).isNone()) {
      p_node->Output(0) = create_empty_from(in0_t);
    }
    auto& out_t = p_node->Output(0).toTensor();
    fastResizeToZero(out_t);
    at::native::mul_out(out_t, in0_t, in1_t);
  };
});

REGISTER_OPERATOR_FUNCTOR(aten::addmm, aten_addmm, [](Node* n) -> SROperator {
  return [](ProcessedNode* p_node) {
    auto& in0_t = p_node->Input(0).toTensor();
    auto& in1_t = p_node->Input(1).toTensor();
    auto& in2_t = p_node->Input(2).toTensor();
    auto in3_s = p_node->Input(3).toScalar();
    auto in4_s = p_node->Input(4).toScalar();
    if (p_node->Output(0).isNone()) {
      p_node->Output(0) = create_empty_from(in0_t);
    }
    auto& out_t = p_node->Output(0).toTensor();
    fastResizeToZero(out_t);
    at::native::addmm_cpu_out(out_t, in0_t, in1_t, in2_t, in3_s, in4_s);
  };
});

REGISTER_OPERATOR_FUNCTOR(aten::clamp, aten_clamp, [](Node* n) -> SROperator {
  return [](ProcessedNode* p_node) {
    auto& in0_t = p_node->Input(0).toTensor();
    auto in1_s = p_node->Input(1).toScalar();
    auto in2_s = p_node->Input(2).toScalar();
    if (p_node->Output(0).isNone()) {
      p_node->Output(0) = create_empty_from(in0_t);
    }
    auto& out_t = p_node->Output(0).toTensor();
    fastResizeToZero(out_t);
    at::native::clamp_out(out_t, in0_t, in1_s, in2_s);
  };
});

REGISTER_OPERATOR_FUNCTOR(aten::bmm, aten_bmm, [](Node* n) -> SROperator {
  return [](ProcessedNode* p_node) {
    auto& in0_t = p_node->Input(0).toTensor();
    auto& in1_t = p_node->Input(1).toTensor();
    if (p_node->Output(0).isNone()) {
      p_node->Output(0) = create_empty_from(in0_t);
    }
    auto& out_t = p_node->Output(0).toTensor();
    fastResizeToZero(out_t);
    at::native::bmm_out_cpu(out_t, in0_t, in1_t);
  };
});

REGISTER_OPERATOR_FUNCTOR(
    aten::nan_to_num,
    aten_nan_to_num,
    [](Node* n) -> SROperator {
      return [](ProcessedNode* p_node) {
        auto input_size = p_node->inputs().size();
        auto& in0_t = p_node->Input(0).toTensor();
        double in1_d = input_size > 1 ? p_node->Input(1).toDouble() : 0;
        double in2_d = input_size > 2 ? p_node->Input(2).toDouble()
                                      : std::numeric_limits<double>::infinity();
        double in3_d = input_size > 3
            ? p_node->Input(3).toDouble()
            : -std::numeric_limits<double>::infinity();
        if (p_node->Output(0).isNone()) {
          p_node->Output(0) = create_empty_from(in0_t);
        }
        auto& out_t = p_node->Output(0).toTensor();
        fastResizeToZero(out_t);
        at::native::nan_to_num_out(out_t, in0_t, in1_d, in2_d, in3_d);
      };
    });
REGISTER_OPERATOR_FUNCTOR(aten::cat, aten_cat, [](Node* n) -> SROperator {
  return [](ProcessedNode* p_node) {
    auto in0_tl = p_node->Input(0).toTensorVector();
    auto in1_i = p_node->Input(1).toInt();
    if (p_node->Output(0).isNone()) {
      p_node->Output(0) = create_empty_from(in0_tl[0]);
    }
    auto& out_t = p_node->Output(0).toTensor();
    fastResizeToZero(out_t);
    at::native::_cat_out_cpu(out_t, in0_tl, in1_i);
  };
});
REGISTER_OPERATOR_FUNCTOR(aten::tanh, aten_tanh, [](Node* n) -> SROperator {
  return [](ProcessedNode* p_node) {
    auto& in0_t = p_node->Input(0).toTensor();
    if (p_node->Output(0).isNone()) {
      p_node->Output(0) = create_empty_from(in0_t);
    }
    auto& out_t = p_node->Output(0).toTensor();
    fastResizeToZero(out_t);
    at::native::tanh_out(out_t, in0_t);
  };
});

// Split out into a function to appease MSVC's pre-processor
SROperator aten_stack(Node* n) {
  return [](ProcessedNode* p_node) {
    auto inputs = p_node->Input(0).toTensorVector();
    auto dim = p_node->Input(1).toInt();
    if (p_node->Output(0).isNone()) {
      p_node->Output(0) = create_empty_from(inputs[0]);
    }
    auto& out_t = p_node->Output(0).toTensor();
    fastResizeToZero(out_t);
    at::native::_stack_out_cpu(inputs, dim, out_t);
  };
}

REGISTER_OPERATOR_FUNCTOR(aten::stack, aten_stack, aten_stack);

REGISTER_OPERATOR_FUNCTOR(
    aten::sigmoid,
    aten_sigmoid,
    [](Node* n) -> SROperator {
      return [](ProcessedNode* p_node) {
        auto& in0_t = p_node->Input(0).toTensor();
        if (p_node->Output(0).isNone()) {
          p_node->Output(0) = create_empty_from(in0_t);
        }
        auto& out_t = p_node->Output(0).toTensor();
        fastResizeToZero(out_t);
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
        return [=](ProcessedNode* p_node) {
          auto& in0_t = p_node->Input(0).toTensor();
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = create_empty_from(in0_t);
          }
          auto& out_t = p_node->Output(0).toTensor();
          at::native::leaky_relu_out(out_t, in0_t, in1_s);
        };
      } else {
        return [](ProcessedNode* p_node) {
          auto& in0_t = p_node->Input(0).toTensor();
          auto in1_s = p_node->Input(1).toScalar();
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = create_empty_from(in0_t);
          }
          auto& out_t = p_node->Output(0).toTensor();
          at::native::leaky_relu_out(out_t, in0_t, in1_s);
        };
      }
    });
REGISTER_OPERATOR_FUNCTOR(aten::relu, aten_relu, [](Node* n) -> SROperator {
  return [](ProcessedNode* p_node) {
    auto& in0_t = p_node->Input(0).toTensor();
    if (p_node->Output(0).isNone()) {
      p_node->Output(0) = create_empty_from(in0_t);
    }
    auto& out_t = p_node->Output(0).toTensor();
    fastResizeToZero(out_t);
    at::native::threshold_out(out_t, in0_t, 0, 0);
  };
});
REGISTER_OPERATOR_FUNCTOR(aten::logit, aten_logit, [](Node* n) -> SROperator {
  return [](ProcessedNode* p_node) {
    auto& in0_t = p_node->Input(0).toTensor();
    double in1_d =
        p_node->inputs().size() > 1 ? p_node->Input(1).toDouble() : -1.0;
    if (p_node->Output(0).isNone()) {
      p_node->Output(0) = create_empty_from(in0_t);
    }
    auto& out_t = p_node->Output(0).toTensor();
    fastResizeToZero(out_t);
    at::native::logit_out(out_t, in0_t, in1_d);
  };
});
REGISTER_OPERATOR_FUNCTOR(aten::clone, aten_clone, [](Node* n) -> SROperator {
  return [](ProcessedNode* p_node) {
    auto& in0_t = p_node->Input(0).toTensor();
    if (p_node->Output(0).isNone()) {
      p_node->Output(0) = create_empty_from(in0_t);
    }
    auto& out_t = p_node->Output(0).toTensor();
    at::native::resize_as_(out_t, in0_t, c10::nullopt);
    at::native::copy_(out_t, in0_t, false);
  };
});
REGISTER_OPERATOR_FUNCTOR_OPT(
    quantized::embedding_bag_byte_rowwise_offsets,
    quantized_embedding_bag_byte_rowwise_offsets,
    false, // don't reuse byte inputs
    true,
    [](Node* n) -> SROperator {
      return [](ProcessedNode* p_node) {
        auto& weight = p_node->Input(0).toTensor();
        auto& indices = p_node->Input(1).toTensor();
        auto offsets = p_node->Input(2).toOptional<at::Tensor>();
        auto pruned_weights = p_node->Input(5).toBool();
        auto per_sample_weights = p_node->Input(6).toOptional<at::Tensor>();
        auto compressed_indices_mapping =
            p_node->Input(7).toOptional<at::Tensor>();
        auto include_last_offset = p_node->Input(8).toBool();
        if (p_node->Output(0).isNone()) {
          p_node->Output(0) =
              at::empty({0}, weight.options().dtype(at::kFloat));
        }
        auto& out_t = p_node->Output(0).toTensor();
        fastResizeToZero(out_t);
        return at::native::embedding_bag_byte_rowwise_offsets_out(
            out_t,
            weight,
            indices,
            offsets,
            false, // unused scale_grad_by_freq
            0, // unused mode
            pruned_weights,
            per_sample_weights,
            compressed_indices_mapping,
            include_last_offset);
      };
    });
REGISTER_OPERATOR_FUNCTOR_OPT(
    quantized::embedding_bag_4bit_rowwise_offsets,
    embedding_bag_4bit_rowwise_offsets,
    false, // don't reuse byte inputs
    true,
    [](Node* n) -> SROperator {
      return [](ProcessedNode* p_node) {
        auto& weight = p_node->Input(0).toTensor();
        auto& indices = p_node->Input(1).toTensor();
        auto offsets = p_node->Input(2).toOptional<at::Tensor>();
        auto pruned_weights = p_node->Input(5).toBool();
        auto per_sample_weights = p_node->Input(6).toOptional<at::Tensor>();
        auto compressed_indices_mapping =
            p_node->Input(7).toOptional<at::Tensor>();
        auto include_last_offset = p_node->Input(8).toBool();
        if (p_node->Output(0).isNone()) {
          p_node->Output(0) =
              at::empty({0}, weight.options().dtype(at::kFloat));
        }
        auto& out_t = p_node->Output(0).toTensor();
        fastResizeToZero(out_t);
        return at::native::embedding_bag_byte_rowwise_offsets_out(
            out_t,
            weight,
            indices,
            offsets,
            false, // unused scale_grad_by_freq
            0, // unused mode
            pruned_weights,
            per_sample_weights,
            compressed_indices_mapping,
            include_last_offset);
      };
    });

// The out variant takes precedence over native
REGISTER_OPERATOR_FUNCTOR(aten::narrow, aten_narrow, [](Node* n) -> SROperator {
  return [](ProcessedNode* p_node) {
    auto& self = p_node->Input(0).toTensor(); // self
    auto dim = p_node->Input(1).toInt(); // dim
    int64_t start = 0;
    if (p_node->Input(2).isScalar()) {
      start = p_node->Input(2).toInt();
    } else {
      auto& t = p_node->Input(2).toTensor();
      start = t.item<int64_t>();
    }
    auto length = p_node->Input(3).toInt(); // length

    if (p_node->Output(0).isNone()) {
      p_node->Output(0) = create_empty_from(self);
    }
    auto& output = p_node->Output(0).toTensor();
    fastResizeToZero(output);
    at::native::narrow_copy_dense_cpu_out(self, dim, start, length, output);
  };
});
REGISTER_OPERATOR_FUNCTOR(aten::index, aten_index, [](Node* n) -> SROperator {
  return [](ProcessedNode* p_node) {
    const auto& in0_t = p_node->Input(0).toTensor();
    auto in1_l =
        at::native::toListOfOptionalTensors(p_node->Input(1).toListRef());
    if (p_node->Output(0).isNone()) {
      p_node->Output(0) = create_empty_from(in0_t);
    }
    auto& out_t = p_node->Output(0).toTensor();
    fastResizeToZero(out_t);
    at::native::index_out(out_t, in0_t, in1_l);
  };
});

// Out variants for view ops are registered to a separate registry because
// their outputs (views) can't participate in memory reuse.
REGISTER_VIEW_OPERATOR_FUNCTOR(
    aten::reshape,
    aten_reshape,
    [](Node* n) -> SROperator {
      return [](ProcessedNode* p_node) {
        auto& self = p_node->Input(0).toTensor(); // self
        auto proposed_shape = p_node->Input(1).toIntVector(); // shape

        if (p_node->Output(0).isNone()) {
          p_node->Output(0) = at::Tensor();
        }
        auto& out = p_node->Output(0).toTensor();
        at::native::reshape_out(out, self, proposed_shape, true);
      };
    });

REGISTER_VIEW_OPERATOR_FUNCTOR(
    aten::flatten,
    aten_flatten,
    [](Node* n) -> SROperator {
      return [](ProcessedNode* p_node) {
        DCHECK(p_node->inputs().size() == 3);
        auto& self = p_node->Input(0).toTensor();
        auto start_dim = p_node->Input(1).toInt();
        auto end_dim = p_node->Input(2).toInt();

        if (p_node->Output(0).isNone()) {
          p_node->Output(0) = at::Tensor();
        }
        auto& out = p_node->Output(0).toTensor();
        at::native::flatten_out(out, self, start_dim, end_dim);
      };
    });

std::function<void(ProcessedNode*)> getOutOfPlaceOperation(Node* n) {
  auto op_name = n->kind().toQualString();
  if (SROperatorRegistry()->Has(op_name)) {
    return SROperatorRegistry()->Create(op_name)->Generate(n);
  }
  if (SRViewOperatorRegistry()->Has(op_name)) {
    return SRViewOperatorRegistry()->Create(op_name)->Generate(n);
  }

  return [](ProcessedNode*) { TORCH_CHECK(0); };
}

std::function<void(ProcessedNode*)> getNativeOperation(Node* n) {
  if (n->kind() == c10::Symbol::fromQualString("aten::transpose")) {
    return [](ProcessedNode* p_node) {
      auto& in0_t = p_node->Input(0).toTensor();
      auto in1_i = p_node->Input(1).toInt();
      auto in2_i = p_node->Input(2).toInt();
      p_node->Output(0) = at::native::transpose(in0_t, in1_i, in2_i);
    };
  } else if (n->kind() == c10::Symbol::fromQualString("aten::flatten")) {
    return [](ProcessedNode* p_node) {
      DCHECK(p_node->inputs().size() == 3);
      auto& in0_t = p_node->Input(0).toTensor();
      auto in1_i = p_node->Input(1).toInt();
      auto in2_i = p_node->Input(2).toInt();
      p_node->Output(0) = at::native::flatten(in0_t, in1_i, in2_i);
    };
  } else if (n->kind() == prim::TupleConstruct) {
    return [](ProcessedNode* p_node) {
      // prepare inputs
      std::vector<IValue> stack;
      const size_t size = p_node->inputs().size();
      stack.reserve(size);
      for (size_t i = 0; i < size; i++) {
        stack.emplace_back(p_node->Input(i));
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
      p_node->Output(0) = std::move(stack[0]);
    };
  } else if (n->kind() == prim::ListConstruct) {
    return [](ProcessedNode* p_node) {
      // prepare inputs
      std::vector<IValue> stack;
      const size_t size = p_node->inputs().size();
      stack.reserve(size);
      for (size_t i = 0; i < size; i++) {
        stack.emplace_back(p_node->Input(i));
      }
      // run op
      listConstruct(
          stack,
          p_node->get_node()->output()->type()->expectRef<ListType>(),
          p_node->inputs().size());
      // put output back
      p_node->Output(0) = std::move(stack[0]);
    };
  } else if (n->kind() == prim::ListUnpack) {
    return [](ProcessedNode* p_node) {
      // prepare inputs
      std::vector<IValue> stack;
      const size_t size = p_node->inputs().size();
      stack.reserve(size);
      for (size_t i = 0; i < size; i++) {
        stack.emplace_back(p_node->Input(i));
      }
      // run op
      size_t num_outputs = p_node->outputs().size();
      listUnpack(stack, num_outputs);
      // put output back
      DCHECK_EQ(stack.size(), num_outputs);
      for (auto i = 0; i < num_outputs; i++) {
        p_node->Output(i) = std::move(stack[i]);
      }
    };
  } else if (n->kind() == c10::Symbol::fromQualString("aten::permute")) {
    return [](ProcessedNode* p_node) {
      auto& in0_t = p_node->Input(0).toTensor();
      auto in1_iv = p_node->Input(1).toIntVector();
      p_node->Output(0) = at::native::permute(in0_t, in1_iv);
    };
  } else if (n->kind() == c10::Symbol::fromQualString("aten::reshape")) {
    return [](ProcessedNode* p_node) {
      auto& in0_t = p_node->Input(0).toTensor();
      auto in1_iv = p_node->Input(1).toIntVector();
      p_node->Output(0) = at::native::reshape(in0_t, in1_iv);
    };
  } else if (n->kind() == c10::Symbol::fromQualString("aten::slice")) {
    return [](ProcessedNode* p_node) {
      auto& in0_t = p_node->Input(0).toTensor();
      auto in1_i = p_node->Input(1).toInt();
      auto in2_i = p_node->Input(2).toInt();
      auto in3_i = p_node->Input(3).toInt();
      auto in4_i = p_node->Input(4).toInt();
      p_node->Output(0) = at::native::slice(in0_t, in1_i, in2_i, in3_i, in4_i);
    };
  } else if (n->kind() == c10::Symbol::fromQualString("aten::narrow")) {
    return [](ProcessedNode* p_node) {
      auto& self = p_node->Input(0).toTensor(); // self
      auto dim = p_node->Input(1).toInt(); // dim
      int64_t start = 0;
      if (p_node->Input(2).isScalar()) {
        start = p_node->Input(2).toInt();
      } else {
        auto& t = p_node->Input(2).toTensor();
        start = t.item<int64_t>();
      }
      auto length = p_node->Input(3).toInt(); // length
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
      p_node->Output(0) =
          at::native::slice(self, dim, start, start + length, 1);
    };
  } else if (n->kind() == c10::Symbol::fromQualString("aten::to")) {
    return [](ProcessedNode* p_node) {
      DCHECK(p_node->inputs().size() == 5);
      auto& in0_t = p_node->Input(0).toTensor();
      auto in1_i = p_node->Input(1).toScalarType();
      auto in2_i = p_node->Input(2).toBool();
      auto in3_i = p_node->Input(3).toBool();
      if (p_node->Input(4).isNone()) {
        p_node->Output(0) =
            at::native::to(in0_t, in1_i, in2_i, in3_i, c10::nullopt);
      } else {
        auto in4_o = p_node->Input(4).toMemoryFormat();
        p_node->Output(0) = at::native::to(in0_t, in1_i, in2_i, in3_i, in4_o);
      }
    };
  }
  return [](ProcessedNode*) { TORCH_CHECK(0); };
}

} // namespace jit
} // namespace torch
