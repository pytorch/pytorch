#include <torch/csrc/jit/runtime/static/ops.h>

#include <ATen/CPUFunctions.h>
#include <ATen/InferSize.h>
#include <ATen/NativeFunctions.h>
#include <ATen/ScalarOps.h>
#include <ATen/TensorUtils.h>
#include <ATen/native/EmbeddingBag.h>
#include <ATen/native/Fill.h>
#include <ATen/native/IndexingUtils.h>
#include <ATen/native/Resize.h>
#include <ATen/native/SharedReduceOps.h>
#include <ATen/native/TensorAdvancedIndexing.h>
#include <ATen/native/layer_norm.h>
#include <ATen/native/quantized/cpu/fbgemm_utils.h>
#include <ATen/native/quantized/cpu/qembeddingbag.h>
#include <ATen/native/quantized/cpu/qembeddingbag_prepack.h>
#include <c10/core/ScalarType.h>
#include <c10/util/irange.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/runtime/static/impl.h>
#include <torch/csrc/jit/runtime/static/te_wrapper.h>
#include <torch/csrc/jit/runtime/vararg_functions.h>
#include <torch/csrc/jit/tensorexpr/ir.h>
#include <torch/csrc/jit/tensorexpr/ir_simplifier.h>
#include <torch/csrc/jit/tensorexpr/llvm_codegen.h>
#include <torch/csrc/jit/tensorexpr/loopnest.h>
#include <mutex>
#include <unordered_map>

C10_DEFINE_bool(
    static_runtime_enable_fast_math,
    true,
    "If on, static runtime may use use optimizations that cause accurary loss "
    "vs the jit interpreter");

namespace at {
namespace native {

void repeat_out(at::Tensor& result, const Tensor& self, IntArrayRef repeats) {
  TORCH_CHECK(
      repeats.size() >= static_cast<size_t>(self.dim()),
      "Number of dimensions of repeat dims can not be smaller than number of dimensions of tensor");

  // Add new leading dimensions to the tensor if the
  // number of target dimensions is larger than the
  // number of source dimensions.
  int64_t num_new_dimensions = repeats.size() - self.dim();
  DimVector padded_size(num_new_dimensions, 1);
  padded_size.insert(
      padded_size.end(), self.sizes().begin(), self.sizes().end());
  DimVector target_size(repeats.size());
  bool zero_tensor = false;
  for (const auto idx : c10::irange(repeats.size())) {
    if (repeats[idx] == 0) {
      zero_tensor = true;
    }
    target_size[idx] = padded_size[idx] * repeats[idx];
  }

  // return an empty tensor if one of the repeat dimensions is zero
  at::native::resize_(result, target_size, c10::nullopt);
  if (zero_tensor) {
    return;
  }

  Tensor xtensor = at::native::expand(self, padded_size);
  Tensor urtensor = at::native::alias(result);
  for (const auto i : c10::irange(xtensor.dim())) {
    // can't unfold with step 0, so make sure step is at least 1
    // (it doesn't matter what it is in that case, because the size is 0).
    urtensor = urtensor.unfold(
        i, xtensor.size(i), std::max<int64_t>(xtensor.size(i), 1));
  }

  at::native::copy_(urtensor, xtensor.expand_as(urtensor));
}

// copy version of view ops
at::Tensor& reshape_copy_out(
    at::Tensor& out,
    const at::Tensor& self,
    const std::vector<int64_t>& proposed_shape,
    bool infer_size) {
  auto shape = infer_size ? at::infer_size(proposed_shape, self.numel())
                          : proposed_shape;
  at::native::resize_(out, shape, c10::nullopt);

  auto self_contig = self.expect_contiguous();

  size_t nbytes = self.nbytes();
  if (nbytes == 0) {
    return out;
  }

  const void* self_data = self_contig->data_ptr();
  void* out_data = out.data_ptr();
  memcpy(out_data, self_data, nbytes);

  return out;
}

at::Tensor& flatten_copy_out(
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
    return reshape_copy_out(out, self, {1}, false);
  }

  if (start_dim == end_dim) {
    auto shape = self.sizes().vec();
    return reshape_copy_out(out, self, shape, false);
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
      // NOLINTNEXTLINE(modernize-use-transparent-functors)
      std::multiplies<int64_t>());

  std::vector<int64_t> shape;
  shape.reserve(self.dim() - end_dim + start_dim);
  for (const auto i : c10::irange(start_dim)) {
    shape.push_back(self.sizes()[i]);
  }
  shape.push_back(slice_numel);
  for (int64_t i = end_dim + 1; i < self.dim(); i++) {
    shape.push_back(self.sizes()[i]);
  }
  return reshape_copy_out(out, self, shape, false);
}

at::Tensor& to_copy_out(
    Tensor& out,
    const Tensor& self,
    bool non_blocking,
    bool copy_strides) {
  if (copy_strides) {
    at::native::resize_impl_cpu_(
        out.unsafeGetTensorImpl(), self.sizes(), self.strides());
  } else {
    at::native::resize_(out, self.sizes(), c10::nullopt);
  }
  at::native::copy_(out, self, non_blocking);
  return out;
}

Tensor& linear_out(
    Tensor& output,
    const Tensor& input,
    const Tensor& weight,
    const c10::optional<Tensor>& bias_opt) {
  TORCH_CHECK(!input.is_mkldnn());

  auto bias = bias_opt.has_value()
      ? c10::MaybeOwned<Tensor>::borrowed(*bias_opt)
      : c10::MaybeOwned<Tensor>::owned(c10::in_place);

  if (input.dim() == 2 && bias->defined()) {
    // Fused op is marginally faster.
    return at::cpu::addmm_out(output, *bias, input, weight.t());
  }
  at::native::matmul_out(input, weight.t(), output);
  if (bias->defined()) {
    at::cpu::add_(output, *bias);
  }
  return output;
}

Tensor& c2_argmin_out(
    Tensor& output,
    const Tensor& input,
    const int64_t dim,
    const bool keepdim) {
  const auto ndim = input.dim();
  int64_t dim_ = maybe_wrap_dim(dim, ndim);
  TORCH_CHECK(dim_ >= 0 && dim_ < ndim);

  const auto in_dims = input.sizes();

  c10::SmallVector<int64_t, 5> out_dims;
  out_dims.reserve(ndim);
  int prev_size = 1;
  int next_size = 1;
  for (int i = 0; i < dim_; ++i) {
    out_dims.push_back(in_dims[i]);
    prev_size *= in_dims[i];
  }
  if (keepdim) {
    out_dims.push_back(1);
  }
  for (auto i = dim_ + 1; i < ndim; ++i) {
    out_dims.push_back(in_dims[i]);
    next_size *= in_dims[i];
  }
  at::native::resize_(output, out_dims, c10::nullopt);

  const auto n = in_dims[dim_];

  if (next_size == 1) {
    AT_DISPATCH_ALL_TYPES_AND2(
        kHalf, kBFloat16, input.scalar_type(), "argmin_input", [&]() {
          const auto in_ptr = input.data_ptr<scalar_t>();
          const auto out_ptr = output.data_ptr<int64_t>();
          // input is a [prev_size, n] tensor.
          // output is a [prev_size,] tensor.
          // Thus, access is contiguous/coalesced.
          for (int i = 0; i < prev_size; ++i) {
            auto v = std::min_element(
                in_ptr + i * n,
                in_ptr + (i + 1) * n,
                [](scalar_t a, scalar_t b) {
                  // if a is nan, then a is *less* than b with LessOrNan
                  // semantics
                  if (at::_isnan(a)) {
                    return true;
                  }
                  // if a is not nan and b is nan, then a is not less than b
                  // with LessOrNan semantics otherwise, act normally. If `b` is
                  // NaN then a < b will always return false, so this is
                  // equivalent to the first snippet.
                  return a < b;
                });
            out_ptr[i] = std::distance(in_ptr + i * n, v);
          }
        });
  } else {
    AT_DISPATCH_ALL_TYPES_AND2(
        kHalf, kBFloat16, input.scalar_type(), "argmin_input", [&]() {
          const auto less_or_nan = native::detail::LessOrNan<scalar_t>{};

          const auto in_ptr = input.data_ptr<scalar_t>();
          const auto out_ptr = output.data_ptr<int64_t>();

          std::memset(out_ptr, 0, prev_size * next_size * sizeof(int64_t));

          for (int i = 0; i < prev_size; ++i) {
            const scalar_t* cur_in_ptr = in_ptr + i * n * next_size + next_size;
            for (int k = 1; k < n; ++k) {
              for (int j = 0; j < next_size; ++j) {
                int64_t* cur_out_ptr = out_ptr + i * next_size + j;
                if (less_or_nan(
                        *cur_in_ptr,
                        in_ptr
                            [i * n * next_size + *cur_out_ptr * next_size + j],
                        *cur_out_ptr,
                        k)) {
                  *cur_out_ptr = k;
                }
                ++cur_in_ptr;
              }
            }
          }
        });
  }
  return output;
}
} // namespace native
} // namespace at

namespace torch {
namespace jit {

C10_DEFINE_REGISTRY(SROperatorRegistry, SROperatorFunctor);

bool opIsRegistered(const c10::Symbol& op_name) {
  const std::string name(op_name.toQualString());
  return SROperatorRegistry()->Has(name);
}

bool disableUnsafeMathOp(const char* op_name) {
  if (FLAGS_static_runtime_enable_fast_math) {
    return false;
  }
  // This list contains ops that use caffe2 math library or use NNC that does
  // not guarantee bit exactness vs the jit interpreter. Note aten::relu is not
  // included even though it uses NNC because the results of relu should always
  // match.
  static const FastSet<std::string> fast_ops{
      "aten::add", "aten::tanh", "aten::sigmoid", "aten::logit"};
  return fast_ops.count(op_name) > 0;
}

std::function<void(ProcessedNode*)> getOutOfPlaceOperation(Node* n) {
  auto op_name = n->kind().toQualString();
  if (SROperatorRegistry()->Has(op_name) && !disableUnsafeMathOp(op_name)) {
    return SROperatorRegistry()->Create(op_name)->Generate(n);
  }

  return nullptr;
}

// Returns true if the node represents an op with variadic arguments.
bool hasVarArgs(Node* n) {
  if (n->kind() == prim::VarConcat || n->kind() == prim::VarStack) {
    return true;
  }
  return false;
}

bool canReuseInputsOutputs(
    Node* n,
    const FastMap<Node*, bool>& node_has_out_variant) {
  auto it = node_has_out_variant.find(n);
  if (it != node_has_out_variant.end()) {
    return it->second;
  }
  return getOutOfPlaceOperation(n) != nullptr;
}

// returns true if the producers of the inputs
// to this operations are out of place.
// This means the IValues will not change run to run
bool inputsCanRunOutOfPlace(
    Node* n,
    const FastMap<Node*, bool>& node_has_out_variant) {
  for (auto* input : n->inputs()) {
    if (!canReuseInputsOutputs(input->node(), node_has_out_variant)) {
      return false;
    }
  }
  return true;
}

bool isOptimizableContainerType(
    Node* n,
    const FastMap<Node*, bool>& node_has_out_variant) {
  const auto& type = n->output()->type();
  bool is_supported_type = false;
  if (type->kind() == TypeKind::ListType) {
    const auto& list_type = type->expectRef<ListType>();
    is_supported_type =
        list_type.getElementType()->kind() == TypeKind::TensorType;
  } else if (type->kind() == TypeKind::TupleType) {
    const auto& tuple_type = type->expectRef<TupleType>();
    auto types = tuple_type.containedTypes();
    const auto& iter =
        std::find_if(types.begin(), types.end(), [](const TypePtr& elem) {
          return elem->kind() == TypeKind::TensorType;
        });
    is_supported_type = iter != types.end();
  }
  return is_supported_type && inputsCanRunOutOfPlace(n, node_has_out_variant);
}

REGISTER_OPERATOR_FUNCTOR(
    prim::ListConstruct,
    prim_ListConstruct,
    [](Node* n) -> SROperator {
      const auto& type = n->output()->type()->expectRef<ListType>();
      bool can_optimize = isOptimizableContainerType(n, FastMap<Node*, bool>());
      return [can_optimize, &type](ProcessedNode* p_node) {
        const auto& out_l = p_node->Output(0);
        if (!out_l.isNone() && can_optimize) {
          return;
        }
        const size_t size = p_node->inputs().size();
        c10::List<IValue> vals(type.getElementType());
        vals.reserve(size);
        for (const auto i : c10::irange(size)) {
          vals.push_back(p_node->Input(i));
        }
        p_node->Output(0) = std::move(vals);
      };
    });

REGISTER_OPERATOR_FUNCTOR(
    prim::TupleConstruct,
    prim_TupleConstruct,
    [](Node* n) -> SROperator {
      bool can_optimize = isOptimizableContainerType(n, FastMap<Node*, bool>());
      return [can_optimize](ProcessedNode* p_node) {
        const auto& out_l = p_node->Output(0);
        if (!out_l.isNone() && can_optimize) {
          return;
        }
        // prepare inputs
        const size_t size = p_node->inputs().size();
        std::vector<IValue> vals;
        vals.reserve(size);
        for (const auto i : c10::irange(size)) {
          vals.push_back(p_node->Input(i));
        }
        p_node->Output(0) = c10::ivalue::Tuple::create(std::move(vals));
      };
    });

REGISTER_OPERATOR_FUNCTOR(aten::abs, aten_abs, [](Node* n) -> SROperator {
  if (!n->matches(torch::schema("aten::abs(Tensor self) -> Tensor"))) {
    LogAndDumpSchema(n);
    return nullptr;
  }
  return [](ProcessedNode* p_node) {
    const auto& in0_t = p_node->Input(0).toTensor();
    if (p_node->Output(0).isNone()) {
      p_node->Output(0) = at::native::abs(in0_t);
    } else {
      auto& out_t = p_node->Output(0).toTensor();
      fastResizeToZero(out_t);
      at::native::abs_out(in0_t, out_t);
    }
  };
});

REGISTER_OPERATOR_FUNCTOR(aten::mul, aten_mul, [](Node* n) -> SROperator {
  if (!n->matches(torch::schema(
          "aten::mul.Tensor(Tensor self, Tensor other) -> Tensor"))) {
    LogAndDumpSchema(n);
    return nullptr;
  }

  return [](ProcessedNode* p_node) {
    const auto& in0_t = p_node->Input(0).toTensor();
    const auto& in1_t = p_node->Input(1).toTensor();
    if (p_node->Output(0).isNone()) {
      p_node->Output(0) = at::cpu::mul(in0_t, in1_t);
    } else {
      auto& out_t = p_node->Output(0).toTensor();
      fastResizeToZero(out_t);
      at::cpu::mul_out(out_t, in0_t, in1_t);
    }
  };
});

REGISTER_OPERATOR_FUNCTOR(aten::addmm, aten_addmm, [](Node* n) -> SROperator {
  if (!n->matches(torch::schema(
          "aten::addmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1) -> Tensor"))) {
    LogAndDumpSchema(n);
    return nullptr;
  }
  return [](ProcessedNode* p_node) {
    const auto& in0_t = p_node->Input(0).toTensor();
    const auto& in1_t = p_node->Input(1).toTensor();
    const auto& in2_t = p_node->Input(2).toTensor();
    const auto in3_s = p_node->Input(3).toScalar();
    const auto in4_s = p_node->Input(4).toScalar();
    if (p_node->Output(0).isNone()) {
      p_node->Output(0) = at::cpu::addmm(in0_t, in1_t, in2_t, in3_s, in4_s);
    } else {
      auto& out_t = p_node->Output(0).toTensor();
      fastResizeToZero(out_t);
      at::cpu::addmm_out(out_t, in0_t, in1_t, in2_t, in3_s, in4_s);
    }
  };
});

REGISTER_OPERATOR_FUNCTOR(aten::clamp, aten_clamp, [](Node* n) -> SROperator {
  if (!n->matches(torch::schema(
          "aten::clamp(Tensor self, Scalar? min=None, Scalar? max=None) -> Tensor")) &&
      !n->matches(torch::schema(
          "aten::clamp.Tensor(Tensor self, Tensor? min=None, Tensor? max=None) -> Tensor"))) {
    LogAndDumpSchema(n);
    return nullptr;
  }

  return [](ProcessedNode* p_node) {
    const auto& in0_t = p_node->Input(0).toTensor();

    if (p_node->Output(0).isNone()) {
      p_node->Output(0) = create_empty_from(in0_t);
    }
    auto& out_t = p_node->Output(0).toTensor();
    fastResizeToZero(out_t);

    if (p_node->Input(1).isTensor()) {
      auto in1_t = p_node->Input(1).toOptional<at::Tensor>();
      auto in2_t = p_node->Input(2).toOptional<at::Tensor>();
      at::native::clamp_out(in0_t, in1_t, in2_t, out_t);
    } else {
      auto in1_s = p_node->Input(1).toOptional<at::Scalar>();
      auto in2_s = p_node->Input(2).toOptional<at::Scalar>();
      at::cpu::clamp_out(out_t, in0_t, in1_s, in2_s);
    }
  };
});

REGISTER_OPERATOR_FUNCTOR(aten::bmm, aten_bmm, [](Node* n) -> SROperator {
  if (!n->matches(
          torch::schema("aten::bmm(Tensor self, Tensor mat2) -> Tensor"))) {
    LogAndDumpSchema(n);
    return nullptr;
  }
  return [](ProcessedNode* p_node) {
    const auto& in0_t = p_node->Input(0).toTensor();
    const auto& in1_t = p_node->Input(1).toTensor();
    if (p_node->Output(0).isNone()) {
      p_node->Output(0) = create_empty_from(in0_t);
    }
    auto& out_t = p_node->Output(0).toTensor();

    fastResizeToZero(out_t);
    at::cpu::bmm_out(out_t, in0_t, in1_t);
  };
});

REGISTER_OPERATOR_FUNCTOR(aten::nan_to_num, aten_nan_to_num, [](Node* n) -> SROperator {
  if (!n->matches(torch::schema(
          "aten::nan_to_num(Tensor self, float? nan=None, float? posinf=None, float? neginf=None) -> Tensor"))) {
    LogAndDumpSchema(n);
    return nullptr;
  }
  return [](ProcessedNode* p_node) {
    const auto& in0_t = p_node->Input(0).toTensor();
    const auto in1_d = p_node->Input(1).toOptional<double>();
    const auto in2_d = p_node->Input(2).toOptional<double>();
    const auto in3_d = p_node->Input(3).toOptional<double>();
    if (p_node->Output(0).isNone()) {
      p_node->Output(0) = at::native::nan_to_num(in0_t, in1_d, in2_d, in3_d);
    } else {
      auto& out_t = p_node->Output(0).toTensor();
      fastResizeToZero(out_t);
      at::native::nan_to_num_out(in0_t, in1_d, in2_d, in3_d, out_t);
    }
  };
});

// Split out into a function to appease MSVC's pre-processor
SROperator aten_stack(Node* n) {
  if (!n->matches(torch::schema(
          "aten::stack(Tensor[] tensors, int dim=0) -> Tensor"))) {
    LogAndDumpSchema(n);
    return nullptr;
  }
  return [](ProcessedNode* p_node) {
    const auto inputs = p_node->Input(0).toTensorVector();
    const auto dim = p_node->Input(1).toInt();
    if (p_node->Output(0).isNone()) {
      p_node->Output(0) = at::native::_stack_cpu(inputs, dim);
    } else {
      auto& out_t = p_node->Output(0).toTensor();
      fastResizeToZero(out_t);
      at::native::_stack_out_cpu(inputs, dim, out_t);
    }
  };
}

REGISTER_OPERATOR_FUNCTOR(aten::stack, aten_stack, aten_stack);

REGISTER_OPERATOR_FUNCTOR(
    prim::VarStack,
    prim_VarStack,
    [](Node* n) -> SROperator {
      return [](ProcessedNode* p_node) {
        const size_t num_inputs = p_node->inputs().size();

        std::vector<at::Tensor> inputs(num_inputs - 1);
        for (size_t i = 0; i < num_inputs - 1; ++i) {
          inputs[i] = p_node->Input(i).toTensor();
        }

        const auto dim = p_node->Input(num_inputs - 1).toInt();
        if (p_node->Output(0).isNone()) {
          p_node->Output(0) = at::native::_stack_cpu(inputs, dim);
        } else {
          auto& out_t = p_node->Output(0).toTensor();
          fastResizeToZero(out_t);
          at::native::_stack_out_cpu(inputs, dim, out_t);
        }
      };
    });

REGISTER_OPERATOR_FUNCTOR(aten::leaky_relu, aten_leaky_relu, [](Node* n) -> SROperator {
  if (!n->matches(torch::schema(
          "aten::leaky_relu(Tensor self, Scalar negative_slope=0.01) -> Tensor"))) {
    LogAndDumpSchema(n);
    return nullptr;
  }
  return [](ProcessedNode* p_node) {
    const auto& in0_t = p_node->Input(0).toTensor();
    const auto in1_s = p_node->Input(1).toScalar();
    if (p_node->Output(0).isNone()) {
      p_node->Output(0) = at::cpu::leaky_relu(in0_t, in1_s);
    } else {
      auto& out_t = p_node->Output(0).toTensor();
      at::cpu::leaky_relu_out(out_t, in0_t, in1_s);
    }
  };
});

REGISTER_OPERATOR_FUNCTOR(aten::relu, aten_relu, [](Node* n) -> SROperator {
  if (!n->matches(torch::schema("aten::relu(Tensor self) -> Tensor"))) {
    LogAndDumpSchema(n);
    return nullptr;
  }
  auto te = createRelu();
  return [te](ProcessedNode* p_node) {
    const auto& in0_t = p_node->Input(0).toTensor();
    if (p_node->Output(0).isNone()) {
      p_node->Output(0) = create_empty_from(in0_t);
    }
    auto& out_t = p_node->Output(0).toTensor();
    if (!te->supports(in0_t)) {
      fastResizeToZero(out_t);
      at::cpu::threshold_out(out_t, in0_t, 0, 0);
    } else {
      at::native::resize_(out_t, in0_t.sizes(), c10::nullopt);
      int64_t nn = in0_t.numel();
      te->call({out_t.data_ptr(), in0_t.data_ptr(), &nn});
    }
  };
});

REGISTER_OPERATOR_FUNCTOR(aten::tanh, aten_tanh, [](Node* n) -> SROperator {
  if (!n->matches(torch::schema("aten::tanh(Tensor self) -> Tensor"))) {
    LogAndDumpSchema(n);
    return nullptr;
  }
  auto te = createTanh();
  return [te](ProcessedNode* p_node) {
    const auto& in0_t = p_node->Input(0).toTensor();
    if (p_node->Output(0).isNone()) {
      p_node->Output(0) = create_empty_from(in0_t);
    }
    auto& out_t = p_node->Output(0).toTensor();
    if (!te->supports(in0_t)) {
      fastResizeToZero(out_t);
      at::cpu::tanh_out(out_t, in0_t);
    } else {
      at::native::resize_(out_t, in0_t.sizes(), c10::nullopt);
      int64_t nn = in0_t.numel();
      te->call({out_t.data_ptr(), in0_t.data_ptr(), &nn});
    }
  };
});

REGISTER_OPERATOR_FUNCTOR(
    aten::sigmoid,
    aten_sigmoid,
    [](Node* n) -> SROperator {
      if (!n->matches(torch::schema("aten::sigmoid(Tensor self) -> Tensor"))) {
        LogAndDumpSchema(n);
        return nullptr;
      }
      auto te = createSigmoid();
      return [te](ProcessedNode* p_node) {
        const auto& in0_t = p_node->Input(0).toTensor();
        if (p_node->Output(0).isNone()) {
          p_node->Output(0) = create_empty_from(in0_t);
        }
        auto& out_t = p_node->Output(0).toTensor();
        if (!te->supports(in0_t)) {
          fastResizeToZero(out_t);
          at::cpu::sigmoid_out(out_t, in0_t);
        } else {
          at::native::resize_(out_t, in0_t.sizes(), c10::nullopt);
          int64_t nn = in0_t.numel();
          te->call({out_t.data_ptr(), in0_t.data_ptr(), &nn});
        }
      };
    });

REGISTER_OPERATOR_FUNCTOR(aten::logit, aten_logit, [](Node* n) -> SROperator {
  if (!n->matches(torch::schema(
          "aten::logit(Tensor self, float? eps=None) -> Tensor"))) {
    LogAndDumpSchema(n);
    return nullptr;
  }
  c10::optional<float> clamp = c10::nullopt;
  if (n->inputs()[1]->node()->kind() == prim::Constant) {
    auto clamp_d = toIValue(n->inputs()[1])->toOptional<double>();
    clamp = clamp_d
        ? c10::make_optional<float>(static_cast<float>(clamp_d.value()))
        : c10::nullopt;
  }
  auto te = clamp ? createLogit() : nullptr;
  float clamp_value = clamp ? *clamp : 0.0f;
  return [te, clamp_value](ProcessedNode* p_node) {
    const auto& in0_t = p_node->Input(0).toTensor();
    if (p_node->Output(0).isNone()) {
      p_node->Output(0) = create_empty_from(in0_t);
    }
    auto& out_t = p_node->Output(0).toTensor();
    if (!te || !te->supports(in0_t)) {
      const auto& in0_t = p_node->Input(0).toTensor();
      const auto in1_d = p_node->Input(1).toOptional<double>();
      fastResizeToZero(out_t);
      at::native::logit_out(in0_t, in1_d, out_t);
    } else {
      at::native::resize_(out_t, in0_t.sizes(), c10::nullopt);
      int64_t nn = in0_t.numel();
      float c = clamp_value;
      te->call({out_t.data_ptr(), in0_t.data_ptr(), &nn, &c});
    }
  };
});

// TODO(T98923825): Uncomment this once the bug in this gets fixed.
/*
REGISTER_OPERATOR_FUNCTOR(aten::clone, aten_clone, [](Node* n) -> SROperator {
  if (!n->matches(torch::schema(
          "aten::clone(Tensor self, *, MemoryFormat? memory_format=None) ->
Tensor"))) { LogAndDumpSchema(n); return nullptr;
  }
  return [](ProcessedNode* p_node) {
    const auto& src = p_node->Input(0).toTensor();
    const auto& optional_memory_format =
        p_node->Input(1).toOptional<c10::MemoryFormat>();
    auto memory_format =
        optional_memory_format.value_or(c10::MemoryFormat::Preserve);

    if (p_node->Output(0).isNone()) {
      if (memory_format == c10::MemoryFormat::Preserve &&
          src.is_non_overlapping_and_dense()) {
        // Copy all strides
        p_node->Output(0) =
            at::empty_strided(src.sizes(), src.strides(), src.options());
      } else {
        memory_format = src.suggest_memory_format();
        p_node->Output(0) = create_empty_from(src, memory_format);
      }
    }
    auto& out_t = p_node->Output(0).toTensor();
    at::native::resize_impl_cpu_(
        out_t.unsafeGetTensorImpl(), src.sizes(), src.strides());
    at::native::copy_(out_t, src, false);
  };
});
*/

REGISTER_OPERATOR_FUNCTOR(
    quantized::embedding_bag_byte_rowwise_offsets,
    quantized_embedding_bag_byte_rowwise_offsets,
    [](Node* n) -> SROperator {
      if (!n->matches(torch::schema(
              "quantized::embedding_bag_byte_rowwise_offsets(Tensor weight, Tensor indices, Tensor? offsets=None, bool scale_grad_by_freq=False, int mode=0, bool pruned_weights=False, Tensor? per_sample_weights=None, Tensor? compressed_indices_mapping=None, bool include_last_offset=False) -> Tensor"))) {
        LogAndDumpSchema(n);
        return nullptr;
      }
      return [](ProcessedNode* p_node) {
        const auto& weight = p_node->Input(0).toTensor();
        const auto& indices = p_node->Input(1).toTensor();
        const auto offsets = p_node->Input(2).toOptional<at::Tensor>();
        const auto pruned_weights = p_node->Input(5).toBool();
        const auto per_sample_weights =
            p_node->Input(6).toOptional<at::Tensor>();
        const auto compressed_indices_mapping =
            p_node->Input(7).toOptional<at::Tensor>();
        const auto include_last_offset = p_node->Input(8).toBool();
        if (p_node->Output(0).isNone()) {
          p_node->Output(0) = create_empty_from(weight, at::kFloat);
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

REGISTER_OPERATOR_FUNCTOR(
    quantized::embedding_bag_4bit_rowwise_offsets,
    embedding_bag_4bit_rowwise_offsets,
    [](Node* n) -> SROperator {
      if (!n->matches(torch::schema(
              "quantized::embedding_bag_4bit_rowwise_offsets(Tensor weight, Tensor indices, Tensor? offsets=None, bool scale_grad_by_freq=False, int mode=0, bool pruned_weights=False, Tensor? per_sample_weights=None, Tensor? compressed_indices_mapping=None, bool include_last_offset=False) -> Tensor"))) {
        LogAndDumpSchema(n);
        return nullptr;
      }
      return [](ProcessedNode* p_node) {
        const auto& weight = p_node->Input(0).toTensor();
        const auto& indices = p_node->Input(1).toTensor();
        const auto offsets = p_node->Input(2).toOptional<at::Tensor>();
        const auto pruned_weights = p_node->Input(5).toBool();
        const auto per_sample_weights =
            p_node->Input(6).toOptional<at::Tensor>();
        const auto compressed_indices_mapping =
            p_node->Input(7).toOptional<at::Tensor>();
        const auto include_last_offset = p_node->Input(8).toBool();
        if (p_node->Output(0).isNone()) {
          p_node->Output(0) = create_empty_from(weight, at::kFloat);
        }
        auto& out_t = p_node->Output(0).toTensor();
        fastResizeToZero(out_t);
        return at::native::embedding_bag_4bit_rowwise_offsets_out(
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

REGISTER_OPERATOR_FUNCTOR(
    quantized::embedding_bag_byte_prepack,
    embedding_bag_byte_prepack,
    [](Node* n) -> SROperator {
      if (!n->matches(torch::schema(
              "quantized::embedding_bag_byte_prepack(Tensor weight) -> Tensor"))) {
        LogAndDumpSchema(n);
        return nullptr;
      }
      return [](ProcessedNode* p_node) {
        const auto& weight = p_node->Input(0).toTensor();
        if (p_node->Output(0).isNone()) {
          p_node->Output(0) = at::native::qembeddingbag_byte_prepack(weight);
          return;
        }
        auto& out_t = p_node->Output(0).toTensor();
        fastResizeToZero(out_t);
        at::native::qembeddingbag_byte_prepack_out(out_t, weight);
      };
    });

// The out variant takes precedence over native
REGISTER_OPERATOR_FUNCTOR(aten::narrow_copy, aten_narrow_copy, [](Node* n) -> SROperator {
  if (!n->matches(torch::schema(
          "aten::narrow_copy(Tensor self, int dim, int start, int length) -> Tensor"))) {
    LogAndDumpSchema(n);
    return nullptr;
  }
  return [](ProcessedNode* p_node) {
    const auto& self = p_node->Input(0).toTensor(); // self
    const auto dim = p_node->Input(1).toInt(); // dim
    int64_t start = 0;
    if (p_node->Input(2).isScalar()) {
      start = p_node->Input(2).toInt();
    } else {
      auto& t = p_node->Input(2).toTensor();
      start = t.item<int64_t>();
    }
    auto length = p_node->Input(3).toInt(); // length

    if (p_node->Output(0).isNone()) {
      p_node->Output(0) =
          at::native::narrow_copy_dense_cpu(self, dim, start, length);
    } else {
      auto& output = p_node->Output(0).toTensor();
      fastResizeToZero(output);
      at::native::narrow_copy_dense_cpu_out(self, dim, start, length, output);
    }
  };
});
REGISTER_OPERATOR_FUNCTOR(aten::index, aten_index, [](Node* n) -> SROperator {
  if (!n->matches(torch::schema(
          "aten::index.Tensor(Tensor self, Tensor?[] indices) -> Tensor"))) {
    LogAndDumpSchema(n);
    return nullptr;
  }
  return [](ProcessedNode* p_node) {
    const auto& in0_t = p_node->Input(0).toTensor();
    const auto in1_l =
        at::native::toListOfOptionalTensors(p_node->Input(1).toListRef());
    if (p_node->Output(0).isNone()) {
      p_node->Output(0) = at::native::index(in0_t, in1_l);
    } else {
      auto& out_t = p_node->Output(0).toTensor();
      fastResizeToZero(out_t);
      at::native::index_out(out_t, in0_t, in1_l);
    }
  };
});
REGISTER_OPERATOR_FUNCTOR(aten::pow, aten_pow, [](Node* n) -> SROperator {
  if (!n->matches(torch::schema(
          "aten::pow.Tensor_Tensor(Tensor self, Tensor exponent) -> Tensor")) &&
      !n->matches(torch::schema(
          "aten::pow.Scalar(Scalar self, Tensor exponent) -> Tensor")) &&
      !n->matches(torch::schema(
          "aten::pow.Tensor_Scalar(Tensor self, Scalar exponent) -> Tensor"))) {
    LogAndDumpSchema(n);
    return nullptr;
  }
  return [](ProcessedNode* p_node) {
    if (p_node->Output(0).isNone()) {
      c10::ScalarType dtype;
      if (p_node->Input(0).isTensor()) {
        const auto& in0_t = p_node->Input(0).toTensor();
        if (p_node->Input(1).isTensor()) {
          dtype = at::native::result_type(in0_t, p_node->Input(1).toTensor());
          p_node->Output(0) = create_empty_from(in0_t, dtype);
        } else {
          dtype = at::native::result_type(in0_t, p_node->Input(1).toScalar());
          p_node->Output(0) = at::native::empty_like(
              in0_t,
              dtype,
              in0_t.options().layout_opt(),
              in0_t.options().device_opt(),
              in0_t.options().pinned_memory_opt(),
              at::MemoryFormat::Preserve);
        }
      } else {
        const auto& in1_t = p_node->Input(1).toTensor();
        dtype = at::native::result_type(p_node->Input(0).toScalar(), in1_t);
        p_node->Output(0) = at::native::empty_like(
            in1_t,
            dtype,
            in1_t.options().layout_opt(),
            in1_t.options().device_opt(),
            in1_t.options().pinned_memory_opt(),
            at::MemoryFormat::Preserve);
      }
    }
    auto& out_t = p_node->Output(0).toTensor();
    fastResizeToZero(out_t);
    if (p_node->Input(0).isTensor()) {
      if (p_node->Input(1).isTensor()) {
        at::cpu::pow_out(
            out_t, p_node->Input(0).toTensor(), p_node->Input(1).toTensor());
      } else {
        at::cpu::pow_out(
            out_t, p_node->Input(0).toTensor(), p_node->Input(1).toScalar());
      }
    } else {
      at::cpu::pow_out(
          out_t, p_node->Input(0).toScalar(), p_node->Input(1).toTensor());
    }
  };
});
// out variant takes precedence over native
// NB: This impl doesn't work for cpu->cuda copy/cast or vice versa.
REGISTER_OPERATOR_FUNCTOR(
    static_runtime::to_copy,
    aten_to_copy,
    [](Node* n) -> SROperator {
      // support 4- or 5-arg for adindexer/adfinder models
      // Keep TORCH_CHECK here because there is no alternative for fallback
      TORCH_CHECK(n->inputs().size() == 4 || n->inputs().size() == 5);
      return [](ProcessedNode* p_node) {
        const auto& self = p_node->Input(0).toTensor();
        // ignore input 3 (copy)
        auto non_blocking = p_node->Input(2).toBool(); // non_blocking
        // handle memory format
        bool copy_strides = false;
        c10::optional<c10::MemoryFormat> memory_format = c10::nullopt;
        if (p_node->inputs().size() == 5) {
          memory_format = p_node->Input(4).toOptional<c10::MemoryFormat>();
        }
        memory_format = memory_format.value_or(c10::MemoryFormat::Preserve);

        if (p_node->Output(0).isNone()) {
          // handle dtype, layout, and device
          at::ScalarType dtype;
          c10::Layout layout = self.layout();
          c10::Device device = self.device();
          if (p_node->Input(1).isTensor()) {
            const auto& other = p_node->Input(1).toTensor();
            dtype = other.scalar_type();
            layout = other.layout();
            device = other.device();
          } else {
            dtype = p_node->Input(1).toScalarType();
          }

          if (memory_format == c10::MemoryFormat::Preserve) {
            if (self.is_non_overlapping_and_dense()) {
              memory_format = c10::nullopt;
              copy_strides = true;
            } else {
              memory_format = self.suggest_memory_format();
            }
          }

          // See Note [Explicit nullopt MemoryFormat argument]
          // Can't use size {0} if memory_format is ChannelLast
          p_node->Output(0) = at::detail::empty_cpu(
              self.sizes(),
              dtype,
              layout,
              self.device(),
              c10::nullopt,
              memory_format);
        }

        copy_strides = copy_strides ||
            (memory_format == c10::MemoryFormat::Preserve &&
             self.is_non_overlapping_and_dense());

        auto& out_t = p_node->Output(0).toTensor();
        fastResizeToZero(out_t);
        at::native::to_copy_out(out_t, self, non_blocking, copy_strides);
      };
    });

// Out variants for view ops are registered to a separate registry because
// their outputs (views) can't participate in memory reuse.
REGISTER_OPERATOR_FUNCTOR(
    static_runtime::reshape_copy,
    aten_reshape,
    [](Node* n) -> SROperator {
      TORCH_CHECK(n->inputs().size() == 2);
      return [](ProcessedNode* p_node) {
        const auto& self = p_node->Input(0).toTensor(); // self
        const auto proposed_shape = p_node->Input(1).toIntVector(); // shape

        if (p_node->Output(0).isNone()) {
          p_node->Output(0) = create_empty_from(self);
        }
        auto& out = p_node->Output(0).toTensor();
        at::native::reshape_copy_out(out, self, proposed_shape, true);
      };
    });

REGISTER_OPERATOR_FUNCTOR(
    static_runtime::flatten_copy,
    aten_flatten,
    [](Node* n) -> SROperator {
      TORCH_CHECK(n->inputs().size() == 3);
      return [](ProcessedNode* p_node) {
        const auto& self = p_node->Input(0).toTensor();
        const auto start_dim = p_node->Input(1).toInt();
        const auto end_dim = p_node->Input(2).toInt();

        if (p_node->Output(0).isNone()) {
          p_node->Output(0) = create_empty_from(self);
        }
        auto& out = p_node->Output(0).toTensor();
        at::native::flatten_copy_out(out, self, start_dim, end_dim);
      };
    });

REGISTER_OPERATOR_FUNCTOR(aten::sum, aten_sum, [](Node* n) -> SROperator {
  if (n->inputs().size() != 2 && n->inputs().size() != 4) {
    return nullptr;
  }
  if (!n->matches(torch::schema(
          "aten::sum(Tensor self, *, ScalarType? dtype=None) -> Tensor")) &&
      !n->matches(torch::schema(
          "aten::sum.dim_IntList(Tensor self, int[1] dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor"))) {
    LogAndDumpSchema(n);
    return nullptr;
  }
  return [](ProcessedNode* p_node) {
    const at::Tensor& self = p_node->Input(0).toTensor();

    c10::optional<at::ScalarType> dtype = c10::nullopt;
    if (p_node->inputs().size() == 2) {
      // sum(Tensor self, *, ScalarType? dtype=None) -> Tensor
      dtype = p_node->Input(1).toOptional<at::ScalarType>();
    }

    std::vector<int64_t> dim = {};
    bool keepdim = false;
    if (p_node->inputs().size() == 4) {
      // sum.dim_IntList(Tensor self, int[1] dim, bool keepdim=False, *,
      // ScalarType? dtype=None) -> Tensor
      dim = p_node->Input(1).toIntList().vec();
      keepdim = p_node->Input(2).toBool();
      dtype = p_node->Input(3).toOptional<at::ScalarType>();
    }

    if (p_node->Output(0).isNone()) {
      p_node->Output(0) = at::cpu::sum(self, dim, keepdim, dtype);
    } else {
      auto& output = p_node->Output(0).toTensor();
      fastResizeToZero(output);
      at::cpu::sum_out(output, self, dim, keepdim, dtype);
    }
  };
});

REGISTER_OPERATOR_FUNCTOR(aten::embedding_bag, aten_embedding_bag, [](Node* n) -> SROperator {
  // TODO: Support only 9 args once the old signature has been removed.
  if (!n->matches(torch::schema(
          "aten::embedding_bag(Tensor weight, Tensor indices, Tensor offsets, bool scale_grad_by_freq=False, int mode=0, bool sparse=False, Tensor? per_sample_weights=None, bool include_last_offset=False) -> (Tensor, Tensor, Tensor, Tensor)")) &&
      !n->matches(torch::schema(
          "aten::embedding_bag.padding_idx(Tensor weight, Tensor indices, Tensor offsets, bool scale_grad_by_freq, int mode, bool sparse, Tensor? per_sample_weights, bool include_last_offset, int? padding_idx) -> (Tensor, Tensor, Tensor, Tensor)"))) {
    LogAndDumpSchema(n);
    return nullptr;
  }
  return [](ProcessedNode* p_node) {
    const auto& weight = p_node->Input(0).toTensor();
    const auto& indices = p_node->Input(1).toTensor();
    const auto& offsets = p_node->Input(2).toTensor();
    auto scale_grad_by_freq = p_node->Input(3).toBool();
    auto mode = p_node->Input(4).to<int64_t>();
    auto sparse = p_node->Input(5).toBool();
    auto per_sample_weights = p_node->Input(6).toOptional<at::Tensor>();
    auto include_last_offset = p_node->Input(7).toBool();
    c10::optional<int64_t> padding_idx;
    if (p_node->inputs().size() == 9) {
      if (p_node->Input(8).isNone()) {
        padding_idx = c10::nullopt;
      } else {
        padding_idx = p_node->Input(8).toInt();
      }
    }

    at::native::check_arguments(
        weight,
        indices,
        offsets,
        mode,
        per_sample_weights,
        include_last_offset);

    std::ignore = scale_grad_by_freq;
    std::ignore = sparse;

    if (p_node->Output(0).isNone()) {
      p_node->Output(0) = at::empty(
          {include_last_offset ? offsets.sizes()[0] - 1 : offsets.sizes()[0],
           weight.sizes()[1]},
          weight.options());
    } else {
      at::native::resize_(
          p_node->Output(0).toTensor(),
          {include_last_offset ? offsets.sizes()[0] - 1 : offsets.sizes()[0],
           weight.sizes()[1]},
          c10::nullopt);
    }
    at::Tensor& output = p_node->Output(0).toTensor();

    if (p_node->Output(1).isNone()) {
      p_node->Output(1) = at::empty({0}, offsets.options());
    }
    at::Tensor& offset2bag = p_node->Output(1).toTensor();
    at::native::make_offset2bag_out(
        offset2bag,
        output,
        weight,
        indices,
        offsets,
        mode,
        per_sample_weights,
        padding_idx.value_or(-1));

    if (p_node->Output(2).isNone()) {
      p_node->Output(2) = at::empty(offsets.sizes(), offsets.options());
    }
    at::Tensor& bag_size = p_node->Output(2).toTensor();
    at::native::make_bag_size_out(
        bag_size, offsets, indices, mode, include_last_offset, false);

    if (p_node->Output(3).isNone()) {
      p_node->Output(3) = at::empty(bag_size.sizes(), offsets.options());
    }
    at::Tensor& max_indices = p_node->Output(3).toTensor();
    at::native::make_max_indices_out(
        max_indices,
        weight,
        indices,
        offsets,
        bag_size,
        mode,
        include_last_offset);

    at::native::_embedding_bag_cpu_impl_out(
        output,
        offset2bag,
        bag_size,
        max_indices,
        weight,
        indices,
        offsets,
        mode,
        per_sample_weights,
        include_last_offset,
        padding_idx.value_or(-1));
  };
});

REGISTER_OPERATOR_FUNCTOR(aten::repeat, aten_repeat, [](Node* n) -> SROperator {
  if (!n->matches(torch::schema(
          "aten::repeat(Tensor self, int[] repeats) -> Tensor"))) {
    LogAndDumpSchema(n);
    return nullptr;
  }
  return [](ProcessedNode* p_node) {
    const auto& self = p_node->Input(0).toTensor();
    const auto repeats = p_node->Input(1).toIntVector();

    if (p_node->Output(0).isNone()) {
      p_node->Output(0) = at::native::repeat(self, repeats);
    } else {
      at::Tensor& output = p_node->Output(0).toTensor();
      at::native::repeat_out(output, self, repeats);
    }
  };
});

REGISTER_OPERATOR_FUNCTOR(aten::sign, aten_sign, [](Node* n) -> SROperator {
  if (!n->matches(torch::schema("aten::sign.Tensor(Tensor input) -> Tensor"))) {
    LogAndDumpSchema(n);
    return nullptr;
  }
  return [](ProcessedNode* p_node) {
    const auto& in0_t = p_node->Input(0).toTensor();
    if (p_node->Output(0).isNone()) {
      p_node->Output(0) = at::cpu::sign(in0_t);
    } else {
      auto& out_t = p_node->Output(0).toTensor();
      fastResizeToZero(out_t);

      at::cpu::sign_out(out_t, in0_t);
    }
  };
});

REGISTER_OPERATOR_FUNCTOR(aten::div, aten_div, [](Node* n) -> SROperator {
  if (!n->matches(torch::schema(
          "aten::div.Tensor(Tensor self, Tensor other) -> Tensor")) &&
      !n->matches(torch::schema(
          "aten::div.Tensor_mode(Tensor self, Tensor other, *, str? rounding_mode) -> Tensor")) &&
      !n->matches(torch::schema(
          "aten::div.Scalar(Tensor self, Scalar other) -> Tensor")) &&
      !n->matches(torch::schema(
          "aten::div.Scalar_mode(Tensor self, Scalar other, *, str? rounding_mode) -> Tensor"))) {
    LogAndDumpSchema(n);
    return nullptr;
  }
  return [](ProcessedNode* p_node) {
    const auto& in0_t = p_node->Input(0).toTensor();
    c10::optional<c10::string_view> rounding_mode = c10::nullopt;
    if (p_node->inputs().size() > 2) {
      rounding_mode = p_node->Input(2).toOptional<c10::string_view>();
    }
    const auto& in1_t = p_node->Input(1).isTensor()
        ? p_node->Input(1).toTensor()
        : at::native::wrapped_scalar_tensor(p_node->Input(1).toScalar());

    if (p_node->Output(0).isNone()) {
      p_node->Output(0) = at::cpu::div(in0_t, in1_t, rounding_mode);
    } else {
      auto& out_t = p_node->Output(0).toTensor();
      fastResizeToZero(out_t);

      at::cpu::div_out(out_t, in0_t, in1_t, rounding_mode);
    }
  };
});

REGISTER_OPERATOR_FUNCTOR(aten::log, aten_log, [](Node* n) -> SROperator {
  if (!n->matches(torch::schema("aten::log.Tensor(Tensor input) -> Tensor"))) {
    LogAndDumpSchema(n);
    return nullptr;
  }
  return [](ProcessedNode* p_node) {
    const auto& in0_t = p_node->Input(0).toTensor();
    if (p_node->Output(0).isNone()) {
      p_node->Output(0) = at::cpu::log(in0_t);
    } else {
      auto& out_t = p_node->Output(0).toTensor();
      fastResizeToZero(out_t);

      at::cpu::log_out(out_t, in0_t);
    }
  };
});

REGISTER_OPERATOR_FUNCTOR(aten::sub, aten_sub, [](Node* n) -> SROperator {
  if (!n->matches(torch::schema(
          "aten::sub.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor")) &&
      !n->matches(torch::schema(
          "aten::sub.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> Tensor"))) {
    LogAndDumpSchema(n);
    return nullptr;
  }
  return [](ProcessedNode* p_node) {
    const auto& in0_t = p_node->Input(0).toTensor();
    const auto alpha = p_node->Input(2).toScalar();

    const auto& in1_t = p_node->Input(1).isTensor()
        ? p_node->Input(1).toTensor()
        : at::native::wrapped_scalar_tensor(p_node->Input(1).toScalar());

    if (p_node->Output(0).isNone()) {
      p_node->Output(0) = at::cpu::sub(in0_t, in1_t, alpha);
    } else {
      auto& out_t = p_node->Output(0).toTensor();
      fastResizeToZero(out_t);

      at::cpu::sub_out(out_t, in0_t, in1_t, alpha);
    }
  };
});

// TODO: support clamp_min.Tensor(Tensor self, Tensor min) -> Tensor
REGISTER_OPERATOR_FUNCTOR(
    aten::clamp_min,
    aten_clamp_min,
    [](Node* n) -> SROperator {
      if (!n->matches(torch::schema(
              "aten::clamp_min(Tensor self, Scalar min) -> Tensor"))) {
        LogAndDumpSchema(n);
        return nullptr;
      }
      return [](ProcessedNode* p_node) {
        const auto& in0_t = p_node->Input(0).toTensor();
        const auto in1_s = p_node->Input(1).toScalar();
        if (p_node->Output(0).isNone()) {
          p_node->Output(0) = at::native::clamp_min(in0_t, in1_s);
        } else {
          auto& out_t = p_node->Output(0).toTensor();
          fastResizeToZero(out_t);
          at::native::clamp_min_out(in0_t, in1_s, out_t);
        }
      };
    });

REGISTER_OPERATOR_FUNCTOR(aten::argmin, aten_argmin, [](Node* n) -> SROperator {
  if (!n->matches(torch::schema(
          "aten::argmin(Tensor self, int? dim=None, bool keepdim=False) -> Tensor"))) {
    LogAndDumpSchema(n);
    return nullptr;
  }
  return [](ProcessedNode* p_node) {
    const auto& in0_t = p_node->Input(0).toTensor();
    const auto dim = p_node->Input(1).toOptional<int64_t>();
    const auto keepdim = p_node->Input(2).toBool();
    if (p_node->Output(0).isNone()) {
      p_node->Output(0) = at::cpu::argmin(in0_t, dim, keepdim);
    } else {
      auto& out_t = p_node->Output(0).toTensor();
      fastResizeToZero(out_t);
      if (in0_t.is_contiguous() && dim.has_value()) {
        at::native::c2_argmin_out(out_t, in0_t, dim.value(), keepdim);
        return;
      }
      at::cpu::argmin_out(out_t, in0_t, dim, keepdim);
    }
  };
});

REGISTER_OPERATOR_FUNCTOR(aten::softmax, aten_softmax, [](Node* n) -> SROperator {
  if (!n->matches(torch::schema(
          "aten::softmax(Tensor self, int dim, ScalarType? dtype=None) -> Tensor"))) {
    LogAndDumpSchema(n);
    return nullptr;
  }
  return [](ProcessedNode* p_node) {
    const auto& in_t = p_node->Input(0).toTensor();
    const auto& dim = p_node->Input(1).toInt();
    const auto& dtype = p_node->Input(2).toOptional<c10::ScalarType>();
    if (p_node->Output(0).isNone()) {
      p_node->Output(0) = at::native::softmax(in_t, dim, dtype);
    } else {
      auto& out_t = p_node->Output(0).toTensor();
      fastResizeToZero(out_t);

      auto half_to_float = in_t.scalar_type() == at::ScalarType::Half &&
          dtype == at::ScalarType::Float;
      at::cpu::_softmax_out(out_t, in_t, dim, half_to_float);
    }
  };
});

REGISTER_OPERATOR_FUNCTOR(
    static_runtime::layer_norm,
    aten_layer_norm,
    [](Node* n) -> SROperator {
      if (!n->matches(torch::schema(
              "static_runtime::layer_norm(Tensor input, int[] normalized_shape, Tensor? weight=None, Tensor? bias=None, float eps=1e-05, bool cudnn_enable=True) -> (Tensor,Tensor,Tensor)"))) {
        LogAndDumpSchema(n);
        return nullptr;
      }
      return [](ProcessedNode* p_node) {
        // ignore Input(5): `bool cudnn_enable=True`
        const auto& input = p_node->Input(0).toTensor();
        const auto normalized_shape = p_node->Input(1).toIntVector();
        auto weight_opt = p_node->Input(2).toOptional<at::Tensor>();
        auto bias_opt = p_node->Input(3).toOptional<at::Tensor>();
        float eps = p_node->Input(4).toDouble();

        c10::MaybeOwned<at::Tensor> weight_maybe_owned =
            at::borrow_from_optional_tensor(weight_opt);
        const at::Tensor& weight = *weight_maybe_owned;
        c10::MaybeOwned<at::Tensor> bias_maybe_owned =
            at::borrow_from_optional_tensor(bias_opt);
        const at::Tensor& bias = *bias_maybe_owned;

        auto M_N = at::native::_check_layer_norm_inputs(
            input, normalized_shape, weight, bias);
        auto M = M_N.first;
        auto N = M_N.second;
        auto X = input.expect_contiguous();
        auto gamma = weight.expect_contiguous();
        auto beta = bias.expect_contiguous();

        if (p_node->Output(0).isNone()) {
          p_node->Output(0) = at::native::empty_like(
              *X,
              c10::nullopt /* dtype */,
              c10::nullopt /* layout */,
              c10::nullopt /* device */,
              c10::nullopt /* pin_memory */,
              at::MemoryFormat::Contiguous);
        } else {
          at::native::resize_(
              p_node->Output(0).toTensor(), X->sizes(), c10::nullopt);
        }
        if (p_node->Output(1).isNone()) {
          p_node->Output(1) = create_empty_from({M}, *X);
        } else {
          at::native::resize_(p_node->Output(1).toTensor(), {M}, c10::nullopt);
        }
        if (p_node->Output(2).isNone()) {
          p_node->Output(2) = create_empty_from({M}, *X);
        } else {
          at::native::resize_(p_node->Output(2).toTensor(), {M}, c10::nullopt);
        }
        at::Tensor& output = p_node->Output(0).toTensor();
        at::Tensor mean = p_node->Output(1).toTensor();
        at::Tensor rstd = p_node->Output(2).toTensor();
        at::native::layer_norm_cpu_out(
            output,
            mean,
            rstd,
            input,
            normalized_shape,
            *gamma,
            *beta,
            eps,
            M,
            N);
      };
    });

REGISTER_OPERATOR_FUNCTOR(aten::norm, aten_norm, [](Node* n) -> SROperator {
  if (!n->matches(torch::schema(
          "aten::norm.ScalarOpt_dtype(Tensor self, Scalar? p, *, ScalarType dtype) -> Tensor")) &&
      !n->matches(torch::schema(
          "aten::norm.ScalarOpt_dim_dtype(Tensor self, Scalar? p, int[1] dim, bool keepdim, *, ScalarType dtype) -> Tensor")) &&
      !n->matches(torch::schema(
          "aten::norm.ScalarOpt_dim(Tensor self, Scalar? p, int[1] dim, bool keepdim=False) -> Tensor"))) {
    LogAndDumpSchema(n);
    return nullptr;
  }

  return [](ProcessedNode* p_node) {
    const auto& in0_t = p_node->Input(0).toTensor();

    if (p_node->Output(0).isNone()) {
      p_node->Output(0) = create_empty_from(in0_t);
    }
    auto& out_t = p_node->Output(0).toTensor();
    fastResizeToZero(out_t);

    const size_t num_inp = p_node->inputs().size();
    const auto in1_s = p_node->Input(1).toOptional<at::Scalar>();
    if (num_inp == 3) {
      at::cpu::norm_outf(
          in0_t,
          in1_s,
          c10::IntArrayRef{},
          false,
          p_node->Input(2).toScalarType(),
          out_t);
      return;
    }

    if (num_inp > 4) {
      at::cpu::norm_outf(
          in0_t,
          in1_s,
          p_node->Input(2).toIntVector(), // dim
          p_node->Input(3).toBool(), // keepdim
          p_node->Input(4).toScalarType(), // dtype
          out_t);
      return;
    }
    at::cpu::norm_outf(
        in0_t,
        in1_s,
        p_node->Input(2).toIntVector(), // dim
        p_node->Input(3).toBool(), // keepdim
        out_t);
  };
});

REGISTER_OPERATOR_FUNCTOR(aten::matmul, aten_matmul, [](Node* n) -> SROperator {
  if (!n->matches(
          torch::schema("aten::matmul(Tensor self, Tensor other) -> Tensor"))) {
    LogAndDumpSchema(n);
    return nullptr;
  }
  return [](ProcessedNode* p_node) {
    const auto& in0_t = p_node->Input(0).toTensor();
    const auto& in1_t = p_node->Input(1).toTensor();

    if (p_node->Output(0).isNone()) {
      p_node->Output(0) = at::native::matmul(in0_t, in1_t);
    } else {
      auto& out_t = p_node->Output(0).toTensor();
      fastResizeToZero(out_t);
      at::native::matmul_out(in0_t, in1_t, out_t);
    }
  };
});

REGISTER_OPERATOR_FUNCTOR(quantized::linear, quantized_linear, [](Node* n) -> SROperator {
  if (!n->matches(torch::schema(
          "quantized::linear(Tensor X, __torch__.torch.classes.quantized.LinearPackedParamsBase W_prepack, float Y_scale_i, int Y_zero_point_i) -> Tensor Y"))) {
    LogAndDumpSchema(n);
    return nullptr;
  }
  const auto w = toIValue(n->inputs()[1]);
  c10::intrusive_ptr<LinearPackedParamsBase> packed_weight;
  if (w) {
    packed_weight = w->toCustomClass<LinearPackedParamsBase>();
  }
  return [packed_weight](ProcessedNode* p_node) {
    const auto& input = p_node->Input(0).toTensor();
    const auto output_scale = p_node->Input(2).toDouble();
    const auto output_zero_point = p_node->Input(3).toInt();

    if (p_node->Output(0).isNone()) {
      p_node->Output(0) = at::native::empty_affine_quantized(
          {0},
          c10::kQUInt8,
          c10::nullopt,
          c10::kCPU,
          false,
          output_scale,
          output_zero_point,
          c10::nullopt);
    }
    auto& out_t = p_node->Output(0).toTensor();
    fastResizeToZero(out_t);

    if (packed_weight) {
      packed_weight->apply_out(input, output_scale, output_zero_point, out_t);
    } else {
      // Weights could be quantized on the fly
      auto packed_weight_tmp =
          p_node->Input(1).toCustomClass<LinearPackedParamsBase>();
      packed_weight_tmp->apply_out(
          input, output_scale, output_zero_point, out_t);
    }
  };
});

REGISTER_OPERATOR_FUNCTOR(
    fb::quantized_linear,
    fb_quantized_linear,
    [](Node* n) -> SROperator {
      if (!n->matches(torch::schema(
              "fb::quantized_linear(Tensor X, __torch__.torch.classes.quantized.LinearPackedParamsBase w_prepack, Tensor Y_scale_i, Tensor Y_zero_point_i) -> Tensor"))) {
        LogAndDumpSchema(n);
        return nullptr;
      }
      const auto w = toIValue(n->inputs()[1]);
      c10::intrusive_ptr<LinearPackedParamsBase> packed_weight;
      if (w) {
        packed_weight = w->toCustomClass<LinearPackedParamsBase>();
      }
      return [packed_weight](ProcessedNode* p_node) {
        const auto& input = p_node->Input(0).toTensor();
        const auto output_scale = p_node->Input(2).toTensor().item().toFloat();
        const auto output_zero_point =
            p_node->Input(3).toTensor().item().toLong();

        if (p_node->Output(0).isNone()) {
          p_node->Output(0) = at::native::empty_affine_quantized(
              {0},
              c10::kQUInt8,
              c10::nullopt,
              c10::kCPU,
              false,
              output_scale,
              output_zero_point,
              c10::nullopt);
        }
        auto& out_t = p_node->Output(0).toTensor();
        fastResizeToZero(out_t);

        if (packed_weight) {
          packed_weight->apply_out(
              input, output_scale, output_zero_point, out_t);
        } else {
          // Weights could be quantized on the fly
          auto packed_weight_tmp =
              p_node->Input(1).toCustomClass<LinearPackedParamsBase>();
          packed_weight_tmp->apply_out(
              input, output_scale, output_zero_point, out_t);
        }
      };
    });

REGISTER_OPERATOR_FUNCTOR(aten::full, aten_full, [](Node* n) -> SROperator {
  if (!n->matches(torch::schema(
          "aten::full(int[] size, Scalar fill_value, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor"))) {
    LogAndDumpSchema(n);
    return nullptr;
  }
  return [](ProcessedNode* p_node) {
    const auto& size = p_node->Input(0).toIntVector();
    const auto fill_value = p_node->Input(1).toScalar();
    if (p_node->Output(0).isNone()) {
      const auto dtype = p_node->Input(2).toOptional<c10::ScalarType>();
      const auto layout = p_node->Input(3).toOptional<c10::Layout>();
      const auto device = p_node->Input(4).toOptional<c10::Device>();
      const auto pin_memory = p_node->Input(5).toOptional<bool>();
      p_node->Output(0) =
          at::native::full(size, fill_value, dtype, layout, device, pin_memory);
    } else {
      p_node->Output(0) =
          at::native::full_out(size, fill_value, p_node->Output(0).toTensor());
    }
  };
});

REGISTER_OPERATOR_FUNCTOR(aten::full_like, aten_full_like, [](Node* n) -> SROperator {
  if (!n->matches(torch::schema(
          "aten::full_like(Tensor self, Scalar fill_value, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor"))) {
    LogAndDumpSchema(n);
    return nullptr;
  }
  return [](ProcessedNode* p_node) {
    const auto in1_s = p_node->Input(1).toScalar();
    const auto& in0_t = p_node->Input(0).toTensor();
    if (p_node->Output(0).isNone()) {
      const auto dtype = p_node->Input(2).toOptional<c10::ScalarType>();
      const auto layout = p_node->Input(3).toOptional<c10::Layout>();
      const auto device = p_node->Input(4).toOptional<c10::Device>();
      const auto pin_memory = p_node->Input(5).toOptional<bool>();
      const auto memory_format =
          p_node->Input(6).toOptional<c10::MemoryFormat>();

      p_node->Output(0) = at::native::empty_like(
          in0_t, dtype, layout, device, pin_memory, memory_format);
    }
    auto& out_t = p_node->Output(0).toTensor();
    at::native::resize_(out_t, in0_t.sizes(), c10::nullopt);
    at::native::fill_out(out_t, in1_s);
  };
});

REGISTER_OPERATOR_FUNCTOR(aten::linear, aten_linear, [](Node* n) -> SROperator {
  if (!n->matches(torch::schema(
          "aten::linear(Tensor input, Tensor weight, Tensor? bias=None) -> Tensor"))) {
    LogAndDumpSchema(n);
    return nullptr;
  }

  return [](ProcessedNode* p_node) {
    const auto& in0_t = p_node->Input(0).toTensor();
    const auto& in1_t = p_node->Input(1).toTensor();
    auto in2_t = p_node->Input(2).toOptional<at::Tensor>();

    if (p_node->Output(0).isNone()) {
      p_node->Output(0) = at::native::linear(in0_t, in1_t, in2_t);
    } else {
      auto& out_t = p_node->Output(0).toTensor();
      fastResizeToZero(out_t);
      at::native::linear_out(out_t, in0_t, in1_t, in2_t);
    }
  };
});

REGISTER_OPERATOR_FUNCTOR(aten::fmod, aten_fmod, [](Node* n) -> SROperator {
  if (!n->matches(torch::schema(
          "aten::fmod.Scalar(Tensor self, Scalar other) -> Tensor")) &&
      !n->matches(torch::schema(
          "aten::fmod.Tensor(Tensor self, Tensor other) -> Tensor"))) {
    LogAndDumpSchema(n);
    return nullptr;
  }
  return [](ProcessedNode* p_node) {
    const auto& in0_t = p_node->Input(0).toTensor();
    const auto& in1_t = p_node->Input(1).isTensor()
        ? p_node->Input(1).toTensor()
        : at::native::wrapped_scalar_tensor(p_node->Input(1).toScalar());

    if (p_node->Output(0).isNone()) {
      p_node->Output(0) = at::cpu::fmod(in0_t, in1_t);
    } else {
      auto& out_t = p_node->Output(0).toTensor();
      fastResizeToZero(out_t);

      at::cpu::fmod_out(out_t, in0_t, in1_t);
    }
  };
});

REGISTER_OPERATOR_FUNCTOR(aten::linalg_norm, aten_linalg_norm, [](Node* n) -> SROperator {
  if (!n->matches(torch::schema(
          "aten::linalg_norm(Tensor self, Scalar? ord=None, int[1]? dim=None, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor")) &&
      !n->matches(torch::schema(
          "aten::linalg_norm.ord_str(Tensor self, str ord, int[1]? dim=None, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor"))) {
    LogAndDumpSchema(n);
    return nullptr;
  }
  return [](ProcessedNode* p_node) {
    const auto& input = p_node->Input(0).toTensor();
    const auto dim = p_node->Input(2).toIntVector();
    const auto keepdim = p_node->Input(3).toBool();
    const auto dtype = p_node->Input(4).toOptional<c10::ScalarType>();

    if (p_node->Output(0).isNone()) {
      if (p_node->Input(1).isScalar()) {
        p_node->Output(0) = at::native::linalg_norm(
            input,
            p_node->Input(1).toOptional<at::Scalar>(),
            dim,
            keepdim,
            dtype);
      } else {
        p_node->Output(0) = at::native::linalg_norm(
            input, p_node->Input(1).toStringView(), dim, keepdim, dtype);
      }
      return;
    }

    auto& output = p_node->Output(0).toTensor();
    fastResizeToZero(output);

    if (p_node->Input(1).isScalar()) {
      at::native::linalg_norm_out(
          input,
          p_node->Input(1).toOptional<at::Scalar>(),
          dim,
          keepdim,
          dtype,
          output);
    } else {
      at::native::linalg_norm_out(
          input, p_node->Input(1).toStringRef(), dim, keepdim, dtype, output);
    }
  };
});

REGISTER_OPERATOR_FUNCTOR(aten::cat, aten_cat, [](Node* n) -> SROperator {
  if (!n->matches(
          torch::schema("aten::cat(Tensor[] tensors, int dim=0) -> Tensor"))) {
    LogAndDumpSchema(n);
    return nullptr;
  }
  return [](ProcessedNode* p_node) {
    const auto inputs = p_node->Input(0).toTensorVector();
    const auto dim = p_node->Input(1).toInt();
    if (p_node->Output(0).isNone()) {
      p_node->Output(0) = at::native::_cat_cpu(inputs, dim);
      return;
    }

    auto& output = p_node->Output(0).toTensor();
    fastResizeToZero(output);
    at::native::_cat_out_cpu(inputs, dim, output);
  };
});

REGISTER_OPERATOR_FUNCTOR(aten::cumsum, aten_cumsum, [](Node* n) -> SROperator {
  if (!n->matches(torch::schema(
          "aten::cumsum(Tensor self, int dim, ScalarType? dtype=None) -> Tensor"))) {
    LogAndDumpSchema(n);
    return nullptr;
  }
  return [](ProcessedNode* p_node) {
    const auto& input = p_node->Input(0).toTensor();
    const auto dim = p_node->Input(1).toInt();
    const auto dtype = p_node->Input(2).toOptional<c10::ScalarType>();

    if (p_node->Output(0).isNone()) {
      p_node->Output(0) = at::cpu::cumsum(input, dim, dtype);
      return;
    }

    auto& output = p_node->Output(0).toTensor();
    fastResizeToZero(output);
    at::cpu::cumsum_out(output, input, dim, dtype);
  };
});

REGISTER_OPERATOR_FUNCTOR(
    aten::nonzero,
    aten_nonzero,
    [](Node* n) -> SROperator {
      if (!n->matches(torch::schema("aten::nonzero(Tensor self) -> Tensor"))) {
        LogAndDumpSchema(n);
        return nullptr;
      }
      return [](ProcessedNode* p_node) {
        const auto& input = p_node->Input(0).toTensor();
        if (p_node->Output(0).isNone()) {
          p_node->Output(0) = at::native::nonzero_cpu(input);
          return;
        }

        auto& output = p_node->Output(0).toTensor();
        fastResizeToZero(output);
        at::native::nonzero_out_cpu(input, output);
      };
    });

namespace {

void check_cat_no_zero_dim(const std::vector<at::Tensor>& tensors) {
  for (const auto i : c10::irange(tensors.size())) {
    auto& t = tensors[i];
    TORCH_CHECK(
        t.dim() > 0,
        "zero-dimensional tensor (at position ",
        i,
        ") cannot be concatenated");
  }
}

} // namespace

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_OPERATOR_FUNCTOR(
    prim::VarConcat,
    prim_VarConcat,
    [](Node* n) -> SROperator {
      return [](ProcessedNode* p_node) {
        const size_t num_inputs = p_node->inputs().size();
        std::vector<at::Tensor> inputs(num_inputs - 1);
        for (const auto i : c10::irange(num_inputs - 1)) {
          inputs[i] = p_node->Input(i).toTensor();
        }
        auto dim = p_node->Input(num_inputs - 1).toInt();
        if (p_node->Output(0).isNone()) {
          p_node->Output(0) = at::cat(inputs, dim);
        } else {
          check_cat_no_zero_dim(inputs);
          dim = legacy_cat_wrap_dim(dim, inputs);
          auto& out_t = p_node->Output(0).toTensor();
          fastResizeToZero(out_t);
          at::native::_cat_out_cpu(inputs, dim, out_t);
        }
      };
    });

namespace {

// This template and its specialization help us avoid compiler warnings
// about taking the absolute value of an unsigned type in signed_log1p
template <class T>
T abs_if_signed(T val) {
  return std::abs(val);
}

template <>
unsigned char abs_if_signed<unsigned char>(unsigned char val) {
  return val;
}

// Computes f(x) = sign(x) * ln(|1 + x|) for each x in the input tensor
void signed_log1p_out(at::Tensor& out, const at::Tensor& input) {
  at::native::resize_(out, input.sizes(), c10::nullopt);

  const auto input_contig = input.expect_contiguous();
  auto output_contig = out.expect_contiguous();

  AT_DISPATCH_ALL_TYPES(input.scalar_type(), "signed_log1p_kernel", [&]() {
    const auto input_data = input_contig->data_ptr<scalar_t>();
    auto output_data = output_contig->data_ptr<float>();
    const auto N = input.numel();

    for (const auto i : c10::irange(N)) {
      const int sign = input_data[i] < 0 ? -1 : 1;
      output_data[i] = std::log1p(abs_if_signed(input_data[i])) * sign;
    }
  });
}

at::Tensor signed_log1p(const at::Tensor& input) {
  auto out = create_empty_from(input);
  signed_log1p_out(out, input);
  return out;
}

} // namespace

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_OPERATOR_FUNCTOR(
    static_runtime::signed_log1p,
    static_runtime_signed_log1p,
    [](Node* n) -> SROperator {
      if (!n->matches(torch::schema(
              "static_runtime::signed_log1p(Tensor x) -> Tensor"))) {
        LogAndDumpSchema(n);
        return nullptr;
      }
      auto te = createSignedLog1p();
      return [te](ProcessedNode* p_node) {
        const auto& input = p_node->Input(0).toTensor();
        if (p_node->Output(0).isNone()) {
          p_node->Output(0) = create_empty_from(input);
        }
        auto& out = p_node->Output(0).toTensor();
        if (!te || !te->supports(input)) {
          fastResizeToZero(out);
          signed_log1p_out(out, input);
          return;
        }
        at::native::resize_(out, input.sizes(), c10::nullopt);
        int64_t nn = input.numel();
        te->call({out.data_ptr(), input.data_ptr(), &nn});
      };
    });

REGISTER_OPERATOR_FUNCTOR(
    aten::remainder,
    aten_remainder,
    [](Node* n) -> SROperator {
      if (n->matches(torch::schema(
              "aten::remainder.Tensor(Tensor self, Tensor other) -> Tensor"))) {
        return [](ProcessedNode* p_node) {
          const auto& self = p_node->Input(0).toTensor();
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) =
                at::cpu::remainder(self, p_node->Input(1).toTensor());
          } else {
            auto& out = p_node->Output(0).toTensor();
            fastResizeToZero(out);
            at::cpu::remainder_out(out, self, p_node->Input(1).toTensor());
          }
        };
      }
      if (n->matches(torch::schema(
              "aten::remainder.Scalar(Tensor self, Scalar other) -> Tensor"))) {
        return [](ProcessedNode* p_node) {
          const auto& self = p_node->Input(0).toTensor();
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) =
                at::native::remainder(self, p_node->Input(1).toScalar());
          } else {
            auto& out = p_node->Output(0).toTensor();
            fastResizeToZero(out);
            at::native::remainder_out(self, p_node->Input(1).toScalar(), out);
          }
        };
      }

      // Unrecognized overload
      LogAndDumpSchema(n);
      return nullptr;
    });
} // namespace jit
} // namespace torch
