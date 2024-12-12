#include <torch/csrc/jit/runtime/static/ops.h>

#include <ATen/CPUFunctions.h>
#include <ATen/InferSize.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Parallel.h>
#include <ATen/ScalarOps.h>
#include <ATen/TensorUtils.h>
#include <ATen/cpu/vec/functional.h>
#include <ATen/cpu/vec/vec.h>
#include <ATen/native/Fill.h>
#include <ATen/native/IndexingUtils.h>
#include <ATen/native/NonSymbolicBC.h>
#include <ATen/native/Resize.h>
#include <ATen/native/SharedReduceOps.h>
#include <ATen/native/TensorAdvancedIndexing.h>
#include <ATen/native/TensorConversions.h>
#include <ATen/native/cpu/SerialStackImpl.h>
#include <ATen/native/layer_norm.h>
#include <ATen/native/quantized/cpu/fbgemm_utils.h>
#include <ATen/native/quantized/cpu/qembeddingbag.h>
#include <ATen/native/quantized/cpu/qembeddingbag_prepack.h>
#include <ATen/quantized/QTensorImpl.h>
#include <ATen/quantized/Quantizer.h>
#include <c10/core/ScalarType.h>
#include <c10/core/WrapDimMinimal.h>
#include <c10/util/irange.h>
#include <torch/csrc/jit/ir/constants.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/passes/symbolic_shape_runtime_fusion.h>
#include <torch/csrc/jit/runtime/static/impl.h>
#include <torch/csrc/jit/runtime/static/processed_node_wrapper.h>
#include <torch/csrc/jit/runtime/static/te_wrapper.h>
#include <torch/csrc/jit/runtime/vararg_functions.h>
#include <torch/csrc/jit/tensorexpr/ir.h>
#include <torch/csrc/jit/tensorexpr/ir_simplifier.h>
#include <torch/csrc/jit/tensorexpr/llvm_codegen.h>
#include <torch/csrc/jit/tensorexpr/loopnest.h>
#include <iterator>

#include <ATen/CompositeExplicitAutogradFunctions.h>

// clang-format off
C10_DEFINE_bool(
    static_runtime_enable_fast_math,
    true,
    "If on, static runtime may use use optimizations that cause accuracy loss "
    "vs the jit interpreter")

namespace at::native {
static void repeat_out(
    at::Tensor& result,
    const Tensor& self,
    IntArrayRef repeats) {
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
  at::native::resize_(result, target_size, std::nullopt);
  if (zero_tensor) {
    return;
  }

  Tensor xtensor = at::compositeexplicitautograd::expand(self, padded_size);
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
    const at::DimVector& proposed_shape,
    bool infer_size) {
  const auto& shape = infer_size
      ? at::infer_size_dv(proposed_shape, self.numel())
      : proposed_shape;
  at::native::resize_(out, shape, std::nullopt);

  auto self_contig = self.expect_contiguous();

  size_t nbytes = self.nbytes();
  if (nbytes == 0) {
    return out;
  }

  const void* self_data = self_contig->const_data_ptr();
  void* out_data = out.mutable_data_ptr();
  memcpy(out_data, self_data, nbytes);

  return out;
}

static at::Tensor& flatten_copy_out(
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
    return reshape_copy_out(out, self, at::DimVector{1}, false);
  }

  if (start_dim == end_dim) {
    auto shape = at::DimVector{self.sizes()};
    return reshape_copy_out(out, self, shape, false);
  }

  // We don't want to infer_size on the entire shape, because that can give us
  // an extra degree of freedom we don't want; for example, consider shape [0,
  // 1, 3, 0], with start_dim=1, end_dim=2. It's clear we want result shape
  // [0, 3, 0] but passing [0, -1, 0] to infer_size means the -1 can take on
  // any value and satisfy the constraints.
  auto iter = self.sizes().data();
  auto slice_numel = std::accumulate(
      iter + start_dim,
      iter + end_dim + 1,
      static_cast<int64_t>(1),
      // NOLINTNEXTLINE(modernize-use-transparent-functors)
      std::multiplies<int64_t>());

  at::DimVector shape;
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

namespace {

// This is annoying and sily, but it's solving a real problem: the
// _MSC_VER version causes an ICE on our old clang5 builds. The
// non-_MSC_VER version is a syntax error according to MSVC. Use the
// appropriate version depending on if we're MSVC or not.

#define TO_COPY_OUT_FAST_PATH_LOGIC(out, self, self_t)             \
  do {                                                             \
    const auto N = self.numel();                                   \
    const auto self_data = self.const_data_ptr<self_t>();          \
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(                        \
        kHalf,                                                     \
        kBFloat16,                                                 \
        kBool,                                                     \
        out.scalar_type(),                                         \
        "to_copy_out_inner_loop",                                  \
        [&]() {                                                    \
          const auto out_data = out.mutable_data_ptr<scalar_t>();  \
          for (const auto idx : c10::irange(N)) {                  \
            /* NOLINTNEXTLINE(bugprone-signed-char-misuse) */      \
            out_data[idx] = static_cast<scalar_t>(self_data[idx]); \
          }                                                        \
        });                                                        \
  } while (0)

#ifdef _MSC_VER
template <typename T>
void to_copy_out_fast_path(Tensor& out, const Tensor& self) {
  TO_COPY_OUT_FAST_PATH_LOGIC(out, self, T);
}

#define TO_COPY_OUT_FAST_PATH_BODY(out, self) \
  to_copy_out_fast_path<scalar_t>(out, self)
#else
#define TO_COPY_OUT_FAST_PATH_BODY(out, self) \
  using self_t = scalar_t;                    \
  TO_COPY_OUT_FAST_PATH_LOGIC(out, self, self_t)
#endif
} // namespace

at::Tensor& to_copy_out(
    Tensor& out,
    const Tensor& self,
    bool non_blocking,
    bool copy_strides,
    std::optional<MemoryFormat> memory_format) {
  if (copy_strides) {
    at::native::resize_impl_cpu_(
        out.unsafeGetTensorImpl(), self.sizes(), self.strides());
  } else {
    at::native::resize_(out, self.sizes(), std::nullopt);
  }
  auto is_unsupported_dtype = [](ScalarType t) {
#define TORCH_OPS_UNSUPPORTED_TYPE(_, type) \
  case k##type:                             \
    return true;
    switch (t) {
      AT_FORALL_QINT_TYPES(TORCH_OPS_UNSUPPORTED_TYPE)
      AT_FORALL_COMPLEX_TYPES(TORCH_OPS_UNSUPPORTED_TYPE)
      default:
        return false;
    }
#undef TORCH_OPS_UNSUPPORTED_TYPE
  };
  // Fast path: can we just copy the data ourselves? Avoids creating a
  // TensorIterator in at::native::copy_, which is relatively
  // expensive.
  if (self.is_contiguous() && !non_blocking &&
      // Did the user request us to make a copy that isn't contiguous?
      (memory_format == std::nullopt ||
       memory_format == c10::MemoryFormat::Preserve ||
       memory_format == c10::MemoryFormat::Contiguous) &&
      // CopyKernel.cpp handles this case specially, so let's not mess
      // with it.
      !self.is_neg() && !is_unsupported_dtype(self.dtype().toScalarType()) &&
      !is_unsupported_dtype(out.dtype().toScalarType()) &&
      !(
          // FBGEMM optimization might kick in, don't interfere with
          // that.
          (self.dtype() == kFloat && out.dtype() == kHalf) ||
          (self.dtype() == kHalf && out.dtype() == kFloat))) {
    AT_DISPATCH_ALL_TYPES_AND3(
        kHalf, kBFloat16, kBool, self.scalar_type(), "to_copy_out", [&]() {
          TO_COPY_OUT_FAST_PATH_BODY(out, self);
        });
    return out;
  }
  at::native::copy_(out, self, non_blocking);
  return out;
}

static Tensor& linear_out(
    Tensor& output,
    const Tensor& input,
    const Tensor& weight,
    const std::optional<Tensor>& bias_opt) {
  TORCH_CHECK(!input.is_onednn());

  auto bias = bias_opt.has_value()
      ? c10::MaybeOwned<Tensor>::borrowed(*bias_opt)
      : c10::MaybeOwned<Tensor>::owned(std::in_place);

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

static Tensor& c2_argmin_out(
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
  at::native::resize_(output, out_dims, std::nullopt);

  const auto n = in_dims[dim_];

  if (next_size == 1) {
    AT_DISPATCH_ALL_TYPES_AND2(
        kHalf, kBFloat16, input.scalar_type(), "argmin_input", [&]() {
          const auto in_ptr = input.const_data_ptr<scalar_t>();
          const auto out_ptr = output.mutable_data_ptr<int64_t>();
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
                  // with LessOrNan semantics otherwise, act normally. If `b`
                  // is NaN then a < b will always return false, so this is
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

          const auto in_ptr = input.const_data_ptr<scalar_t>();
          const auto out_ptr = output.mutable_data_ptr<int64_t>();

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

static at::Tensor& dequantize_copy_out(Tensor& out, const Tensor& self) {
  if (C10_UNLIKELY(!self.is_quantized())) {
    // fallback to dequantize_cpu equivalent case: make sure out is at::kFloat
    DCHECK(out.scalar_type() == kFloat);
    return at::native::to_copy_out(out, self, false, false, std::nullopt);
  }
  return get_qtensorimpl(self)->quantizer()->dequantize_out(out, self);
}
} // namespace at::native

namespace torch::jit {

C10_DEFINE_REGISTRY(SROperatorRegistry, SROperatorFunctor)

bool opIsRegistered(const c10::Symbol& op_name) {
  const std::string name(op_name.toQualString());
  return SROperatorRegistry()->Has(name);
}

static bool disableUnsafeMathOp(const char* op_name) {
  if (FLAGS_static_runtime_enable_fast_math) {
    return false;
  }
  // This list contains ops that use caffe2 math library or use NNC that does
  // not guarantee bit exactness vs the jit interpreter. Note aten::relu is not
  // included even though it uses NNC because the results of relu should always
  // match.
  static const c10::FastSet<std::string> fast_ops{
      "aten::add", "aten::tanh", "aten::sigmoid", "aten::logit"};
  return fast_ops.count(op_name) > 0;
}

SROperator getOutOfPlaceOperation(Node* n) {
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
    const c10::FastMap<Node*, bool>& node_has_out_variant) {
  auto it = node_has_out_variant.find(n);
  if (it != node_has_out_variant.end()) {
    return it->second;
  }
  return getOutOfPlaceOperation(n) != nullptr;
}

// returns true if the producers of the inputs
// to this operations are out of place.
// This means the IValues will not change run to run
static bool inputsCanRunOutOfPlace(
    Node* n,
    const c10::FastMap<Node*, bool>& node_has_out_variant) {
  for (auto* input : n->inputs()) {
    if (!canReuseInputsOutputs(input->node(), node_has_out_variant)) {
      return false;
    }
  }
  return true;
}

bool isOptimizableContainerType(
    Node* n,
    const c10::FastMap<Node*, bool>& node_has_out_variant) {
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

static void listConstructSlowPath(
    const ListType& list_type,
    const size_t size,
    ProcessedNode* p_node) {
  c10::List<IValue> vals(list_type.getElementType());
  vals.reserve(size);
  for (const auto i : c10::irange(size)) {
    vals.push_back(p_node->Input(i));
  }
  p_node->Output(0) = vals;
}

bool sr_schema_check_kind(torch::jit::Node* node, c10::Symbol node_kind) {
  auto is_match = node->kind() == node_kind;
  if (!is_match) {
    torch::jit::LogAndDumpSchema(node);
  }
  return is_match;
}

REGISTER_OPERATOR_FUNCTOR(
    prim::ListConstruct,
    prim_ListConstruct,
    [](Node* n) -> SROperator {
      if (!sr_schema_check_kind(n, prim::ListConstruct)) {
        return nullptr;
      }
      const bool can_optimize =
          isOptimizableContainerType(n, c10::FastMap<Node*, bool>());
      const auto& type = n->output()->type()->expectRef<ListType>();
      const size_t size = n->inputs().size();
      if (!can_optimize) {
        return [&type, size](ProcessedNode* p_node) {
          DCHECK(p_node->num_inputs() == size);
          listConstructSlowPath(type, size, p_node);
        };
      }
      return [&type, size](ProcessedNode* p_node) {
        DCHECK(p_node->num_inputs() == size);
        const auto& out_l = p_node->Output(0);
        if (!out_l.isNone()) {
          return;
        }
        listConstructSlowPath(type, size, p_node);
      };
    })

static void tupleConstructSlowPath(const size_t size, ProcessedNode* p_node) {
  // prepare inputs
  switch (size) {
    case 1:
      p_node->Output(0) = c10::ivalue::Tuple::create(p_node->Input(0));
      break;
    case 2:
      p_node->Output(0) =
          c10::ivalue::Tuple::create(p_node->Input(0), p_node->Input(1));
      break;
    case 3:
      p_node->Output(0) = c10::ivalue::Tuple::create(
          p_node->Input(0), p_node->Input(1), p_node->Input(2));
      break;
    default: {
      std::vector<IValue> vals;
      vals.reserve(size);
      for (const auto i : c10::irange(size)) {
        vals.push_back(p_node->Input(i));
      }
      p_node->Output(0) = c10::ivalue::Tuple::create(std::move(vals));
      break;
    }
  }
}

REGISTER_OPERATOR_FUNCTOR(
    prim::TupleConstruct,
    prim_TupleConstruct,
    [](Node* n) -> SROperator {
      if (!sr_schema_check_kind(n, prim::TupleConstruct)) {
        return nullptr;
      }
      const bool can_optimize =
          isOptimizableContainerType(n, c10::FastMap<Node*, bool>());
      const size_t size = n->inputs().size();
      if (!can_optimize) {
        return [size](ProcessedNode* p_node) {
          DCHECK(p_node->num_inputs() == size);
          tupleConstructSlowPath(size, p_node);
        };
      }
      return [size](ProcessedNode* p_node) {
        DCHECK(p_node->num_inputs() == size);
        const auto& out_l = p_node->Output(0);
        if (!out_l.isNone()) {
          return;
        }
        tupleConstructSlowPath(size, p_node);
      };
    })

REGISTER_OPERATOR_FUNCTOR(aten::abs, aten_abs, [](Node* n) -> SROperator {
  if (!n->matches(torch::schema("aten::abs(Tensor self) -> Tensor"))) {
    LogAndDumpSchema(n);
    return nullptr;
  }
  return [](ProcessedNode* p_node) {
    const auto& in0_t = p_node->Input(0).toTensor();
    if (p_node->Output(0).isNone()) {
      p_node->Output(0) = at::native::abs(in0_t);
      return;
    }
    auto& out_t = p_node->Output(0).toTensor();
    fastResizeToZero(out_t);
    at::native::abs_out(in0_t, out_t);
  };
})

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
      return;
    }
    auto& out_t = p_node->Output(0).toTensor();
    fastResizeToZero(out_t);
    at::cpu::mul_out(out_t, in0_t, in1_t);
  };
})

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
      return;
    }
    auto& out_t = p_node->Output(0).toTensor();
    fastResizeToZero(out_t);
    at::cpu::addmm_out(out_t, in0_t, in1_t, in2_t, in3_s, in4_s);
  };
})

#ifdef FBCODE_CAFFE2
// Disable externally to avoid MSVC errors in open-source CI

REGISTER_OPERATOR_FUNCTOR(
    static_runtime::clamp_nan_to_num,
    static_runtime_clamp_nan_to_num,
    [](Node* n) -> SROperator {
      if (!sr_schema_check(
              n,
              "static_runtime::clamp_nan_to_num(Tensor input, Scalar? min, Scalar? max, float? nan, float? posinf, float? posinf) -> Tensor")) {
        return nullptr;
      }
      auto clamp_min_ival_opt = toIValue(n->input(1));
      auto clamp_max_ival_opt = toIValue(n->input(2));
      TORCH_CHECK(
          clamp_min_ival_opt.has_value() && clamp_max_ival_opt.has_value());

      auto clamp_min_opt = clamp_min_ival_opt->toOptional<at::Scalar>();
      auto clamp_max_opt = clamp_max_ival_opt->toOptional<at::Scalar>();
      TORCH_CHECK(clamp_min_opt.has_value() && clamp_max_opt.has_value());

      return [te = createClampNanToNum(),
              clamp_min = clamp_min_opt->to<float>(),
              clamp_max =
                  clamp_max_opt->to<float>()](ProcessedNode* p_node) mutable {
        const auto& in0_t = p_node->Input(0).toTensor();
        if (p_node->Output(0).isNone()) {
          p_node->Output(0) = create_empty_from(in0_t);
        }
        auto& out_t = p_node->Output(0).toTensor();
        fastResizeToZero(out_t);
        auto in3_s = p_node->Input(3).toOptional<double>();

        if (!te || !te->checkInput<float>(in0_t)) {
          at::cpu::nan_to_num_out(
              out_t,
              at::cpu::clamp(in0_t, clamp_min, clamp_max),
              in3_s,
              std::nullopt,
              std::nullopt);
          return;
        }
        at::native::resize_(out_t, in0_t.sizes(), std::nullopt);

        auto output_size = in0_t.numel();

        // This might be UB if in3_s is absurdly large, but most implementations
        // just turn it into `inf` in that case. The PyTorch core nan_to_num
        // kernel just static_cast()s the limits to the destination type, so
        // we'll ignore overflow issues here as well.
        auto nan = in3_s.has_value() ? static_cast<float>(*in3_s) : 0.f;

        te->call(
            {out_t.data_ptr(),
             in0_t.data_ptr(),
             &clamp_min,
             &clamp_max,
             &nan,
             &output_size});
      };
    })

#endif

REGISTER_OPERATOR_FUNCTOR(aten::clamp, aten_clamp, [](Node* n) -> SROperator {
  if (n->matches(torch::schema(
          "aten::clamp(Tensor self, Scalar? min=None, Scalar? max=None) -> Tensor"))) {
    return [te = createClamp()](ProcessedNode* p_node) {
      const auto& in0_t = p_node->Input(0).toTensor();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = create_empty_from(in0_t);
      }
      auto& out_t = p_node->Output(0).toTensor();
      fastResizeToZero(out_t);
      auto in1_s = p_node->Input(1).toOptional<at::Scalar>();
      auto in2_s = p_node->Input(2).toOptional<at::Scalar>();
      if (!te->checkInput<float>(in0_t)) {
        at::cpu::clamp_out(out_t, in0_t, in1_s, in2_s);
        return;
      }
      at::native::resize_(out_t, in0_t.sizes(), std::nullopt);
      auto output_size = in0_t.numel();
      auto min = in1_s.has_value() ? in1_s->toFloat()
                                   : -std::numeric_limits<float>::infinity();
      auto max = in2_s.has_value() ? in2_s->toFloat()
                                   : std::numeric_limits<float>::infinity();
      te->call({out_t.data_ptr(), in0_t.data_ptr(), &min, &max, &output_size});
    };
  }
  if (n->matches(
          "aten::clamp.Tensor(Tensor self, Tensor? min=None, Tensor? max=None) -> Tensor")) {
    return [](ProcessedNode* p_node) {
      const auto& in0_t = p_node->Input(0).toTensor();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = create_empty_from(in0_t);
      }
      auto& out_t = p_node->Output(0).toTensor();
      fastResizeToZero(out_t);
      auto in1_t = p_node->Input(1).toOptional<at::Tensor>();
      auto in2_t = p_node->Input(2).toOptional<at::Tensor>();
      at::cpu::clamp_out(out_t, in0_t, in1_t, in2_t);
    };
  }
  LogAndDumpSchema(n);
  return nullptr;
})

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
})

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
      return;
    }
    auto& out_t = p_node->Output(0).toTensor();
    fastResizeToZero(out_t);
    at::native::nan_to_num_out(in0_t, in1_d, in2_d, in3_d, out_t);
  };
})

namespace {

void varStackSerialOut(
    at::Tensor& result,
    int64_t dim,
    const ProcessedNodeInputWrapper& inputs) {
  auto result_sizes = inputs[0].sizes().vec();
  result_sizes.insert(result_sizes.begin() + dim, inputs.size());
  at::native::resize_(result, result_sizes);

  AT_DISPATCH_FLOATING_TYPES(
      result.scalar_type(), "varstack_serial_kernel", [&]() {
        at::native::detail::
            stack_serial_kernel_impl<scalar_t, ProcessedNodeInputWrapper>(
                result, inputs, dim);
      });
}

std::vector<at::Tensor> unsqueezeVarStackInputs(
    const ProcessedNodeInputWrapper& inputs,
    const int64_t dim) {
  std::vector<at::Tensor> result;
  result.reserve(inputs.size());
  for (const auto i : c10::irange(inputs.size())) {
    result.push_back(at::native::unsqueeze(inputs[i], dim));
  }
  return result;
}

void varstackNonserialOut(
    at::Tensor& result,
    const int64_t dim,
    const ProcessedNodeInputWrapper& inputs) {
  std::vector<at::Tensor> inputs_unsqueezed =
      unsqueezeVarStackInputs(inputs, dim);
  fastResizeToZero(result);
  at::cpu::cat_outf(inputs_unsqueezed, dim, result);
}

void varStackFastOut(
    at::Tensor& out,
    int64_t dim,
    const ProcessedNodeInputWrapper& inputs) {
  DCHECK(out.is_contiguous());
  const auto num_inputs = static_cast<int64_t>(inputs.size());
  TORCH_CHECK(num_inputs > 0, "stack expects a non-empty list of tensors");

  const auto first_tensor_shape = inputs[0].sizes();
  for (const auto i : c10::irange(1, num_inputs)) {
    const auto shape = inputs[i].sizes();
    TORCH_CHECK(
        shape == first_tensor_shape,
        "Stack expects each tensor to be the same size, but got ",
        first_tensor_shape,
        " at position 0 and ",
        shape,
        " at position ",
        i);
  }

  const std::array<int64_t, 2> output_size = (dim == 0 || dim == -2)
      ? std::array<int64_t, 2>{num_inputs, 1}
      : std::array<int64_t, 2>{1, num_inputs};

  at::native::resize_(out, output_size, std::nullopt);

  AT_DISPATCH_ALL_TYPES(out.scalar_type(), "varStackFastOut", [&]() {
    auto* out_data = out.mutable_data_ptr<scalar_t>();
    for (const auto i : c10::irange(num_inputs)) {
      auto& tensor = inputs[i];
      auto* input_ptr = tensor.const_data_ptr<scalar_t>();
      out_data[i] = *input_ptr;
    }
  });
}

bool inputsAreScalars(const ProcessedNodeInputWrapper& inputs) {
  // All stack inputs should have the same size, so we only check
  // the first one. If this isn't true, an exception will be thrown
  // in the VarStack implementation
  const auto& first_tensor = inputs[0];
  return first_tensor.sizes()[0] == 1 && first_tensor.dim() == 1;
}

void varStackOut(ProcessedNode& pnode, int64_t dim) {
  const auto num_inputs = pnode.num_inputs();
  TORCH_CHECK(num_inputs > 1, "stack expects a non-empty list of tensors");
  dim = c10::maybe_wrap_dim(dim, pnode.Input(0).toTensor().dim() + 1);

  auto inputs = ProcessedNodeInputWrapper(pnode);
  auto& output = pnode.Output(0).toTensor();

  if (output.is_contiguous() && inputsAreScalars(inputs)) {
    varStackFastOut(output, dim, inputs);
    return;
  }

  bool can_use_serial = at::native::detail::CanUseNativeSerialStack<
      ProcessedNodeInputWrapper,
      /*skip_overlap_check*/ true>::call(output, inputs, dim);

  if (can_use_serial) {
    varStackSerialOut(output, dim, inputs);
    return;
  }
  varstackNonserialOut(output, dim, inputs);
}

} // namespace

// Split out into a function to appease MSVC's pre-processor
static SROperator aten_stack(Node* n) {
  if (!n->matches(torch::schema(
          "aten::stack(Tensor[] tensors, int dim=0) -> Tensor"))) {
    LogAndDumpSchema(n);
    return nullptr;
  }
  return [](ProcessedNode* p_node) {
    const auto inputs = p_node->Input(0).toTensorVector();
    TORCH_CHECK(!inputs.empty(), "stack expects non-empty tensor list");
    const auto dim = p_node->Input(1).toInt();
    if (p_node->Output(0).isNone()) {
      p_node->Output(0) = at::native::_stack_cpu(inputs, dim);
      return;
    }
    auto& out_t = p_node->Output(0).toTensor();
    fastResizeToZero(out_t);
    at::native::_stack_out_cpu(inputs, dim, out_t);
  };
}

REGISTER_OPERATOR_FUNCTOR(aten::stack, aten_stack, aten_stack)

REGISTER_OPERATOR_FUNCTOR(
    prim::VarStack,
    prim_VarStack,
    [](Node* n) -> SROperator {
      if (!sr_schema_check_kind(n, prim::VarStack)) {
        return nullptr;
      }
      return [](ProcessedNode* p_node) {
        const size_t num_inputs = p_node->num_inputs();
        const auto dim = p_node->Input(num_inputs - 1).toInt();

        if (p_node->Output(0).isNone()) {
          p_node->Output(0) = create_empty_from(p_node->Input(0).toTensor());
        }
        varStackOut(*p_node, dim);
      };
    })

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
      return;
    }
    auto& out_t = p_node->Output(0).toTensor();
    at::cpu::leaky_relu_out(out_t, in0_t, in1_s);
  };
})

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
    if (!te->checkInput<float>(in0_t)) {
      fastResizeToZero(out_t);
      at::cpu::threshold_out(out_t, in0_t, 0, 0);
      return;
    }
    at::native::resize_(out_t, in0_t.sizes(), std::nullopt);
    int64_t nn = in0_t.numel();
    te->call({out_t.data_ptr(), in0_t.data_ptr(), &nn});
  };
})

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
    if (!te->checkInput<float>(in0_t)) {
      fastResizeToZero(out_t);
      at::cpu::tanh_out(out_t, in0_t);
      return;
    }
    at::native::resize_(out_t, in0_t.sizes(), std::nullopt);
    int64_t nn = in0_t.numel();
    te->call({out_t.data_ptr(), in0_t.data_ptr(), &nn});
  };
})

REGISTER_OPERATOR_FUNCTOR(
    prim::TensorExprDynamicGroup,
    prim_TensorExprDynamicGroup,
    [](Node* n) -> SROperator {
      if (!sr_schema_check_kind(n, prim::TensorExprDynamicGroup)) {
        return nullptr;
      }
      auto graph = n->g(attr::Subgraph);
      Code code(graph, "");
      return [code](ProcessedNode* p_node) {
        auto num_outputs = p_node->num_outputs();
        Stack stack;
        if (p_node->Output(0).isNone()) {
          stack.reserve(p_node->num_inputs());
        } else {
          stack.reserve(p_node->num_inputs() + num_outputs);
          for (const auto& o : p_node->outputs()) {
            stack.emplace_back(o);
          }
        }
        for (auto i : c10::irange(p_node->num_inputs())) {
          stack.emplace_back(p_node->Input(i));
        }
        runTensorExprDynamicGroup(code, stack);
        if (p_node->Output(0).isNone()) {
          TORCH_INTERNAL_ASSERT(
              stack.size() == num_outputs,
              "Unexpected # of outputs on stack after executing TensorExprDynamicGroup");
          for (auto i : c10::irange(num_outputs)) {
            p_node->Output(i) = std::move(stack[i]);
          }
        }
      };
    })

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
        if (!te->checkInput<float>(in0_t)) {
          fastResizeToZero(out_t);
          at::cpu::sigmoid_out(out_t, in0_t);
          return;
        }
        at::native::resize_(out_t, in0_t.sizes(), std::nullopt);
        int64_t nn = in0_t.numel();
        te->call({out_t.data_ptr(), in0_t.data_ptr(), &nn});
      };
    })

REGISTER_OPERATOR_FUNCTOR(aten::logit, aten_logit, [](Node* n) -> SROperator {
  if (!n->matches(torch::schema(
          "aten::logit(Tensor self, float? eps=None) -> Tensor"))) {
    LogAndDumpSchema(n);
    return nullptr;
  }
  std::optional<double> clamp = std::nullopt;
  if (n->inputs()[1]->node()->kind() == prim::Constant) {
    auto clamp_d = toIValue(n->inputs()[1])->toOptional<double>();
    clamp = clamp_d;
  }
  auto te = clamp ? createLogit() : nullptr;
  return [te, clamp](ProcessedNode* p_node) {
    const auto& in0_t = p_node->Input(0).toTensor();
    if (p_node->Output(0).isNone()) {
      p_node->Output(0) = create_empty_from(in0_t);
    }
    auto& out_t = p_node->Output(0).toTensor();
    if (!te || !te->checkInput<float>(in0_t)) {
      const auto& in0_t = p_node->Input(0).toTensor();
      const auto in1_d = p_node->Input(1).toOptional<double>();
      fastResizeToZero(out_t);
      at::native::logit_out(in0_t, in1_d, out_t);
      return;
    }
    at::native::resize_(out_t, in0_t.sizes(), std::nullopt);
    int64_t nn = in0_t.numel();
    float c = clamp.value() ? static_cast<float>(clamp.value()) : 0;
    te->call({out_t.data_ptr(), in0_t.data_ptr(), &nn, &c});
  };
})

REGISTER_OPERATOR_FUNCTOR(aten::clone, aten_clone, [](Node* n) -> SROperator {
  if (!n->matches(torch::schema(
          "aten::clone(Tensor self, *, MemoryFormat? memory_format=None) ->Tensor"))) {
    LogAndDumpSchema(n);
    return nullptr;
  }
  return [](ProcessedNode* p_node) {
    const auto& src = p_node->Input(0).toTensor();
    const auto& optional_memory_format =
        p_node->Input(1).toOptional<c10::MemoryFormat>();
    auto memory_format =
        optional_memory_format.value_or(c10::MemoryFormat::Preserve);
    /*
      disable out_variant of clone for case with stride = 0 and
      memory formats other than preserve. Perform dynamic allocation
      instead of memory reuse for simpler implementation. We could,
      in principle, figure out copy of strides.
    */
    if ((at::has_internal_overlap(src.unsafeGetTensorImpl()) ==
         at::MemOverlap::Yes) ||
        (memory_format != c10::MemoryFormat::Preserve)) {
      p_node->Output(0) = at::native::clone(src, memory_format);
      return;
    }
    if (p_node->Output(0).isNone()) {
      if (src.is_non_overlapping_and_dense()) {
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
})

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
    })

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
    })

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
    })

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
      return;
    }
    auto& output = p_node->Output(0).toTensor();
    fastResizeToZero(output);
    at::native::narrow_copy_dense_cpu_out(self, dim, start, length, output);
  };
})
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
      p_node->Output(0) = at::cpu::index(in0_t, in1_l);
      return;
    }
    auto& out_t = p_node->Output(0).toTensor();
    fastResizeToZero(out_t);
    at::cpu::index_out(out_t, in0_t, in1_l);
  };
})

REGISTER_OPERATOR_FUNCTOR(
    aten::index_select,
    aten_index_select,
    [](Node* n) -> SROperator {
      if (!n->matches(torch::schema(
              "aten::index_select(Tensor self, int dim, Tensor index) -> Tensor"))) {
        LogAndDumpSchema(n);
        return nullptr;
      }
      return [](ProcessedNode* p_node) {
        const auto& self = p_node->Input(0).toTensor();
        const auto dim = p_node->Input(1).toInt();
        const auto& index = p_node->Input(2).toTensor();
        if (p_node->Output(0).isNone()) {
          p_node->Output(0) = at::native::index_select_cpu_(self, dim, index);
          return;
        }
        auto& out = p_node->Output(0).toTensor();
        fastResizeToZero(out);
        at::native::index_select_out_cpu_(self, dim, index, out);
      };
    })

REGISTER_OPERATOR_FUNCTOR(aten::pow, aten_pow, [](Node* n) -> SROperator {
  if (n->matches(torch::schema(
          "aten::pow.Tensor_Tensor(Tensor self, Tensor exponent) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      if (p_node->Output(0).isNone()) {
        const auto& in0_t = p_node->Input(0).toTensor();
        auto dtype =
            at::native::result_type(in0_t, p_node->Input(1).toTensor());
        p_node->Output(0) = create_empty_from(in0_t, dtype);
      }
      auto& out_t = p_node->Output(0).toTensor();
      fastResizeToZero(out_t);
      at::cpu::pow_out(
          out_t, p_node->Input(0).toTensor(), p_node->Input(1).toTensor());
    };
  }
  if (n->matches(torch::schema(
          "aten::pow.Scalar(Scalar self, Tensor exponent) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      if (p_node->Output(0).isNone()) {
        const auto& in1_t = p_node->Input(1).toTensor();
        auto dtype =
            at::native::result_type(p_node->Input(0).toScalar(), in1_t);
        p_node->Output(0) = at::native::empty_like(
            in1_t,
            dtype,
            in1_t.options().layout_opt(),
            in1_t.options().device_opt(),
            in1_t.options().pinned_memory_opt(),
            at::MemoryFormat::Preserve);
      }
      auto& out_t = p_node->Output(0).toTensor();
      fastResizeToZero(out_t);
      at::cpu::pow_out(
          out_t, p_node->Input(0).toScalar(), p_node->Input(1).toTensor());
    };
  }
  if (n->matches(torch::schema(
          "aten::pow.Tensor_Scalar(Tensor self, Scalar exponent) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      if (p_node->Output(0).isNone()) {
        const auto& in0_t = p_node->Input(0).toTensor();
        auto dtype =
            at::native::result_type(in0_t, p_node->Input(1).toScalar());
        p_node->Output(0) = at::native::empty_like(
            in0_t,
            dtype,
            in0_t.options().layout_opt(),
            in0_t.options().device_opt(),
            in0_t.options().pinned_memory_opt(),
            at::MemoryFormat::Preserve);
      }
      auto& out_t = p_node->Output(0).toTensor();
      fastResizeToZero(out_t);
      at::cpu::pow_out(
          out_t, p_node->Input(0).toTensor(), p_node->Input(1).toScalar());
    };
  }
  LogAndDumpSchema(n);
  return nullptr;
})

namespace {

// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
struct ToArgs {
  std::optional<at::ScalarType> dtype;
  c10::Layout layout;
  bool know_to_will_alias = false;
  std::optional<c10::MemoryFormat> memory_format;
};

template <bool has_constant_non_tensor_dtype_and_flags, bool has_memory_format>
ToArgs extract_to_args(ProcessedNode* p_node) {
  ToArgs result;
  if (!has_constant_non_tensor_dtype_and_flags && p_node->Input(1).isTensor()) {
    const auto& other = p_node->Input(1).toTensor();
    result.dtype = other.scalar_type();
    result.layout = other.layout();
    TORCH_DCHECK_EQ(other.device().type(), c10::DeviceType::CPU);
  } else {
    const auto& self = p_node->Input(0).toTensor();
    result.dtype = p_node->Input(1).toOptional<at::ScalarType>();
    result.layout = self.layout();
    // Static runtime only works with CPU tensors; don't need to read this.
    TORCH_DCHECK_EQ(self.device().type(), c10::DeviceType::CPU);
    result.know_to_will_alias = has_constant_non_tensor_dtype_and_flags &&
        (!result.dtype.has_value() ||
         result.dtype.value() == self.dtype().toScalarType());
  }
  if (has_memory_format) {
    TORCH_DCHECK_EQ(p_node->num_inputs(), 5);
    result.memory_format = p_node->Input(4).toOptional<c10::MemoryFormat>();
    result.know_to_will_alias = result.know_to_will_alias &&
        (result.memory_format.value_or(c10::MemoryFormat::Preserve) ==
         c10::MemoryFormat::Preserve);
  }

  return result;
}

template <bool has_constant_non_tensor_dtype_and_flags, bool has_memory_format>
struct CheckToWillAlias {
  static bool call(
      ProcessedNode* p_node,
      const at::Tensor& self,
      const ToArgs& to_args) {
    // The to_maybe_copy_out operator functor should have detected a
    // constant true `copy` argument and used to_copy instead.
    bool copy = false;
    if (has_constant_non_tensor_dtype_and_flags) {
      DCHECK(!p_node->Input(3).toBool());
      copy = false;
    } else {
      copy = p_node->Input(3).toBool();
    }
    return !copy &&
        (to_args.know_to_will_alias ||
         at::native::to_will_alias(
             self,
             to_args.dtype,
             to_args.layout,
             c10::Device{c10::DeviceType::CPU},
             copy,
             has_memory_format ? to_args.memory_format
                               : c10::MemoryFormat::Preserve));
  }
};

template <>
struct CheckToWillAlias<true, false> {
  // Special case! First, there is no memory format to check. Second,
  // we know that the layout and device will match self, so we only
  // need to check the dtype.
  static bool call(ProcessedNode* p_node, const at::Tensor& self) {
    DCHECK(!p_node->Input(3).toBool()); // !copy
    const auto dtype_opt = p_node->Input(1).toOptional<at::ScalarType>();
    return !dtype_opt.has_value() || *dtype_opt == self.dtype().toScalarType();
  }
};

// Force inlining so we don't have to branch on whether args is null
// at runtime.
template <bool has_constant_non_tensor_dtype_and_flags, bool has_memory_format>
C10_ALWAYS_INLINE void to_copy_functor_impl(
    ProcessedNode* p_node,
    const ToArgs* args) {
  const auto& self = p_node->Input(0).toTensor();
  // ignore input 3 (copy)
  auto non_blocking = p_node->Input(2).toBool(); // non_blocking
  // handle memory format
  bool copy_strides = false;

  std::optional<c10::MemoryFormat> memory_format = c10::MemoryFormat::Preserve;
  std::optional<ToArgs> my_args;
  if (!args) {
    my_args = extract_to_args<
        has_constant_non_tensor_dtype_and_flags,
        has_memory_format>(p_node);
    args = &my_args.value();
  }
  if (has_memory_format) {
    memory_format = args->memory_format.value_or(c10::MemoryFormat::Preserve);
  }

  if (memory_format == c10::MemoryFormat::Preserve) {
    if (self.is_non_overlapping_and_dense()) {
      memory_format = std::nullopt;
      copy_strides = true;
    } else {
      memory_format = self.suggest_memory_format();
    }
  }

  bool need_to_allocate_output = true;
  if (p_node->Output(0).isTensor()) {
    const auto& existing_output = p_node->Output(0).toTensor();
    if ((!has_constant_non_tensor_dtype_and_flags &&
         (existing_output.dtype() != args->dtype ||
          existing_output.layout() != args->layout ||
          existing_output.device() != self.device())) ||
        (has_memory_format &&
         !existing_output.is_contiguous(
             memory_format.value_or(c10::MemoryFormat::Contiguous)))) {
      need_to_allocate_output = true;
    } else {
      need_to_allocate_output = false;
    }
  }

  // See Note [Explicit nullopt MemoryFormat argument]
  // Can't use size {0} if memory_format is ChannelLast
  if (need_to_allocate_output) {
    p_node->Output(0) = at::detail::empty_cpu(
        self.sizes(),
        args->dtype,
        args->layout,
        self.device(),
        std::nullopt,
        memory_format);
  } else {
    if (has_memory_format) {
      memory_format = p_node->Input(4).toOptional<c10::MemoryFormat>().value_or(
          c10::MemoryFormat::Preserve);
    } else {
      memory_format = c10::MemoryFormat::Preserve;
    }
  }

  copy_strides = copy_strides ||
      (memory_format == c10::MemoryFormat::Preserve &&
       self.is_non_overlapping_and_dense());

  auto& out_t = p_node->Output(0).toTensor();
  fastResizeToZero(out_t);
  at::native::to_copy_out(
      out_t, self, non_blocking, copy_strides, memory_format);
}

template <bool has_constant_non_tensor_dtype_and_flags, bool has_memory_format>
void to_copy_functor(ProcessedNode* p_node) {
  to_copy_functor_impl<
      has_constant_non_tensor_dtype_and_flags,
      has_memory_format>(p_node, nullptr);
}

template <bool has_constant_non_tensor_dtype_and_flags, bool has_memory_format>
void to_maybe_copy_out_functor(ProcessedNode* p_node) {
  // It would be great if we could avoid checking our arguments every
  // time. However, we need to make account for the possibility that
  // the dtype (and layout, memory format, etc.) of self changed
  // between iterations.
  ToArgs args = extract_to_args<
      has_constant_non_tensor_dtype_and_flags,
      has_memory_format>(p_node);
  const auto& self = p_node->Input(0).toTensor();
  if (CheckToWillAlias<
          has_constant_non_tensor_dtype_and_flags,
          has_memory_format>::call(p_node, self, args)) {
    // Don't write our Tensor output. This leaves it None if it
    // was never allocated (and there is a special case in the
    // memory planner to not start managing in this case), but
    // if we are oscillating between aliasing and needing to
    // copy, we should just leave our output in place so as not
    // to confuse the memory planner.
    p_node->Output(1) = false;
  } else {
    p_node->Output(1) = true;
    to_copy_functor_impl<
        has_constant_non_tensor_dtype_and_flags,
        has_memory_format>(p_node, &args);
  }
}

// Take advantage of CheckToWillAlias not requiring the args in this
// case.
template <>
void to_maybe_copy_out_functor<true, false>(ProcessedNode* p_node) {
  const auto& self = p_node->Input(0).toTensor();
  if (CheckToWillAlias<true, false>::call(p_node, self)) {
    p_node->Output(1) = false;
  } else {
    p_node->Output(1) = true;
    auto args = extract_to_args<true, false>(p_node);
    to_copy_functor_impl<true, false>(p_node, &args);
  }
}

bool node_has_constant_non_tensor_dtype_and_flags(Node* n) {
  const auto* input1 = n->inputs()[1];
  return input1->type()->kind() != TypeKind::TensorType &&
      input1->node()->kind() == prim::Constant &&
      n->inputs()[2]->node()->kind() == prim::Constant &&
      n->inputs()[3]->node()->kind() == prim::Constant;
}

auto get_to_copy_functor(
    bool has_constant_non_tensor_dtype_and_flags,
    bool has_memory_format) {
  if (has_constant_non_tensor_dtype_and_flags) {
    if (has_memory_format) {
      return to_copy_functor<true, true>;
    } else {
      return to_copy_functor<true, false>;
    }
  } else {
    if (has_memory_format) {
      return to_copy_functor<false, true>;
    } else {
      return to_copy_functor<false, false>;
    }
  }
}

} // namespace

REGISTER_OPERATOR_FUNCTOR(
    static_runtime::to_maybe_copy_out,
    aten_to_maybe_copy,
    [](Node* n) -> SROperator {
      // support 4- or 5-arg for adindexer/adfinder models
      // Keep TORCH_CHECK here because there is no alternative for fallback
      if (!sr_schema_check(
              n,
              "static_runtime::to_maybe_copy_out.prim_dtype(Tensor self, int? dtype=None, bool non_blocking=False, bool copy=False) -> (Tensor, bool)",
              "static_runtime::to_maybe_copy_out.dtype(Tensor self, ScalarType dtype, bool non_blocking=False, bool copy=False, MemoryFormat? memory_format=None) -> (Tensor, bool)",
              "static_runtime::to_maybe_copy_out.other(Tensor self, Tensor other, bool non_blocking=False, bool copy=False, MemoryFormat? memory_format=None) -> (Tensor, bool)")) {
        return nullptr;
      }
      TORCH_CHECK(n->inputs().size() == 4 || n->inputs().size() == 5);
      const bool has_constant_non_tensor_dtype_and_flags =
          node_has_constant_non_tensor_dtype_and_flags(n);
      const bool has_memory_format = n->inputs().size() == 5;

      // If we are going to copy always, just use the to_copy path so
      // that the to_maybe_copy path can assume that won't happen.
      if (has_constant_non_tensor_dtype_and_flags) {
        const auto copyArg =
            torch::jit::constant_as<bool>(n->inputs()[3]->node()->output());
        DCHECK(copyArg.has_value());
        if (*copyArg) {
          return get_to_copy_functor(
              has_constant_non_tensor_dtype_and_flags, has_memory_format);
        }
      }
      if (has_constant_non_tensor_dtype_and_flags) {
        if (has_memory_format) {
          return to_maybe_copy_out_functor<true, true>;
        } else {
          return to_maybe_copy_out_functor<true, false>;
        }
      } else {
        if (has_memory_format) {
          return to_maybe_copy_out_functor<false, true>;
        } else {
          return to_maybe_copy_out_functor<false, false>;
        }
      }
    })

// out variant takes precedence over native
// NB: This impl doesn't work for cpu->cuda copy/cast or vice versa.
REGISTER_OPERATOR_FUNCTOR(
    static_runtime::to_copy,
    static_runtime_to_copy,
    [](Node* n) -> SROperator {
      // support 4- or 5-arg for adindexer/adfinder models
      // Keep TORCH_CHECK here because there is no alternative for fallback
      if (!sr_schema_check(
              n,
              "static_runtime::to_copy.prim_dtype(Tensor self, int? dtype=None, bool non_blocking=False, bool copy=False) -> Tensor",
              "static_runtime::to_copy.dtype(Tensor self, ScalarType dtype, bool non_blocking=False, bool copy=False, MemoryFormat? memory_format=None) -> Tensor",
              "static_runtime::to_copy.other(Tensor self, Tensor other, bool non_blocking=False, bool copy=False, MemoryFormat? memory_format=None) -> Tensor")) {
        return nullptr;
      }
      TORCH_CHECK(n->inputs().size() == 4 || n->inputs().size() == 5);
      const bool has_constant_non_tensor_dtype_and_flags =
          node_has_constant_non_tensor_dtype_and_flags(n);
      const bool has_memory_format = n->inputs().size() == 5;
      return get_to_copy_functor(
          has_constant_non_tensor_dtype_and_flags, has_memory_format);
    })

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_OPERATOR_FUNCTOR(
    static_runtime::dequantize_copy,
    aten_dequantize_copy,
    [](Node* n) -> SROperator {
      if (!n->matches(torch::schema(
              "static_runtime::dequantize_copy.self(Tensor self) -> Tensor"))) {
        // please implement static runtime support for aten::dequantize with
        // TensorList
        LogAndDumpSchema(n);
        return nullptr;
      }
      return [](ProcessedNode* p_node) {
        const auto& self = p_node->Input(0).toTensor();
        if (p_node->Output(0).isNone()) {
          p_node->Output(0) =
              create_empty_from(self, at::kFloat, self.suggest_memory_format());
        }

        auto& out_t = p_node->Output(0).toTensor();
        fastResizeToZero(out_t);
        at::native::dequantize_copy_out(out_t, self);
      };
    })

// Out variants for view ops are registered to a separate registry because
// their outputs (views) can't participate in memory reuse.
REGISTER_OPERATOR_FUNCTOR(
    static_runtime::reshape_copy,
    aten_reshape,
    [](Node* n) -> SROperator {
      if (!sr_schema_check(
              n,
              "static_runtime::reshape_copy(Tensor self, int[] shape) -> Tensor")) {
        return nullptr;
      }
      TORCH_CHECK(n->inputs().size() == 2);
      return [](ProcessedNode* p_node) {
        const auto& self = p_node->Input(0).toTensor(); // self
        const auto proposed_shape = p_node->Input(1).toDimVector(); // shape

        if (p_node->Output(0).isNone()) {
          p_node->Output(0) = create_empty_from(self);
        }
        auto& out = p_node->Output(0).toTensor();
        at::native::reshape_copy_out(out, self, proposed_shape, true);
      };
    })

REGISTER_OPERATOR_FUNCTOR(
    static_runtime::flatten_copy,
    aten_flatten,
    [](Node* n) -> SROperator {
      if (!sr_schema_check(
              n,
              "static_runtime::flatten_copy.using_ints(Tensor self, int start_dim=0, int end_dim=-1) -> Tensor")) {
        return nullptr;
      }
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
    })

REGISTER_OPERATOR_FUNCTOR(aten::sum, aten_sum, [](Node* n) -> SROperator {
  if (n->inputs().size() != 2 && n->inputs().size() != 4) {
    return nullptr;
  }
  if (n->matches(torch::schema(
          "aten::sum(Tensor self, *, ScalarType? dtype=None) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const at::Tensor& self = p_node->Input(0).toTensor();
      auto dtype = p_node->Input(1).toOptional<at::ScalarType>();
      std::vector<int64_t> dim = {};
      bool keepdim = false;
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::sum(self, dim, keepdim, dtype);
      } else {
        auto& output = p_node->Output(0).toTensor();
        fastResizeToZero(output);
        at::cpu::sum_out(output, self, dim, keepdim, dtype);
      }
    };
  }
  if (n->matches(torch::schema(
          "aten::sum.dim_IntList(Tensor self, int[1]? dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const at::Tensor& self = p_node->Input(0).toTensor();
      auto dim = p_node->Input(1).toDimVector();
      auto keepdim = p_node->Input(2).toBool();
      auto dtype = p_node->Input(3).toOptional<at::ScalarType>();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::sum(self, dim, keepdim, dtype);
      } else {
        auto& output = p_node->Output(0).toTensor();
        fastResizeToZero(output);
        at::cpu::sum_out(output, self, dim, keepdim, dtype);
      }
    };
  }
  LogAndDumpSchema(n);
  return nullptr;
})

REGISTER_OPERATOR_FUNCTOR(aten::mean, aten_mean, [](Node* n) -> SROperator {
  if (n->matches(torch::schema(
          "aten::mean.dim(Tensor self, int[1]? dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      const auto dim = p_node->Input(1).toDimVector();
      const bool keepdim = p_node->Input(2).toBool();
      const auto dtype = p_node->Input(3).toOptional<at::ScalarType>();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = create_empty_from(
            self, dtype.value_or(self.dtype().toScalarType()));
      }
      auto& output = p_node->Output(0).toTensor();
      fastResizeToZero(output);
      at::cpu::mean_out(output, self, dim, keepdim, dtype);
    };
  }

  if (n->matches(torch::schema(
          "aten::mean(Tensor self, *, ScalarType? dtype=None) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      const auto dtype = p_node->Input(1).toOptional<at::ScalarType>();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = create_empty_from(
            self, dtype.value_or(self.dtype().toScalarType()));
      }
      auto& output = p_node->Output(0).toTensor();
      fastResizeToZero(output);
      at::cpu::mean_out(output, self, /*dim=*/{}, /*keepdim=*/false, dtype);
    };
  }

  LogAndDumpSchema(n);
  return nullptr;
})

REGISTER_OPERATOR_FUNCTOR(aten::repeat, aten_repeat, [](Node* n) -> SROperator {
  if (!n->matches(torch::schema(
          "aten::repeat(Tensor self, int[] repeats) -> Tensor"))) {
    LogAndDumpSchema(n);
    return nullptr;
  }
  return [](ProcessedNode* p_node) {
    const auto& self = p_node->Input(0).toTensor();
    const auto repeats = p_node->Input(1).toDimVector();

    if (p_node->Output(0).isNone()) {
      p_node->Output(0) = at::native::repeat(self, repeats);
      return;
    }
    at::Tensor& output = p_node->Output(0).toTensor();
    at::native::repeat_out(output, self, repeats);
  };
})

REGISTER_OPERATOR_FUNCTOR(aten::max, aten_max, [](Node* n) -> SROperator {
  if (n->matches(torch::schema(
          "aten::max.other(Tensor self, Tensor other) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      const auto& other = p_node->Input(1).toTensor();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::native::max(self, other);
        return;
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::native::max_out(self, other, out);
    };
  }

  if (n->matches(torch::schema(
          "aten::max.dim(Tensor self, int dim, bool keepdim=False) -> (Tensor values, Tensor indices)"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      auto dim = p_node->Input(1).toInt();
      const auto keepdim = p_node->Input(2).toBool();

      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = create_empty_from(self);
      }

      if (p_node->Output(1).isNone()) {
        p_node->Output(1) = create_empty_from(self, at::kLong);
      }

      auto& values = p_node->Output(0).toTensor();
      auto& indices = p_node->Output(1).toTensor();
      fastResizeToZero(values);
      fastResizeToZero(indices);
      at::cpu::max_out(values, indices, self, dim, keepdim);
    };
  }

  if (n->matches(torch::schema("aten::max(Tensor self) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = create_empty_from(self);
      }
      auto& value = p_node->Output(0).toTensor();
      fastResizeToZero(value);
      at::cpu::amax_out(value, self);
    };
  }

  LogAndDumpSchema(n);
  return nullptr;
})

REGISTER_OPERATOR_FUNCTOR(aten::sign, aten_sign, [](Node* n) -> SROperator {
  if (!n->matches(torch::schema("aten::sign.Tensor(Tensor input) -> Tensor"))) {
    LogAndDumpSchema(n);
    return nullptr;
  }
  return [](ProcessedNode* p_node) {
    const auto& in0_t = p_node->Input(0).toTensor();
    if (p_node->Output(0).isNone()) {
      p_node->Output(0) = at::cpu::sign(in0_t);
      return;
    }
    auto& out_t = p_node->Output(0).toTensor();
    fastResizeToZero(out_t);
    at::cpu::sign_out(out_t, in0_t);
  };
})

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

  return [te = createDiv()](ProcessedNode* p_node) {
    const auto& in0_t = p_node->Input(0).toTensor();
    std::optional<std::string_view> rounding_mode = std::nullopt;
    if (p_node->num_inputs() > 2) {
      rounding_mode = p_node->Input(2).toOptional<std::string_view>();
    }
    const auto& in1_t = p_node->Input(1).isTensor()
        ? p_node->Input(1).toTensor()
        : at::native::wrapped_scalar_tensor(p_node->Input(1).toScalar());

    if (p_node->Output(0).isNone()) {
      p_node->Output(0) = create_empty_from(in0_t);
    }
    auto& out_t = p_node->Output(0).toTensor();

    if (in0_t.sizes() == in1_t.sizes() &&
        in0_t.scalar_type() == in1_t.scalar_type() &&
        in0_t.strides() == in1_t.strides() && in0_t.is_contiguous() &&
        in0_t.scalar_type() == at::kFloat) {
      int64_t dim = in0_t.numel();
      int i_rounding_mode = 0;
      if (rounding_mode && !rounding_mode.value().empty()) {
        const char peek_rounding_mode = rounding_mode.value().at(0);
        if (peek_rounding_mode == 't') {
          // trunc after div
          i_rounding_mode = 1;
        } else if (peek_rounding_mode == 'f') {
          // floor after div
          i_rounding_mode = 2;
        }
      }
      at::native::resize_(out_t, in0_t.sizes());
      te->call(
          {out_t.data_ptr(),
           in0_t.data_ptr(),
           in1_t.data_ptr(),
           &i_rounding_mode,
           &dim});
    } else {
      fastResizeToZero(out_t);
      at::cpu::div_out(out_t, in0_t, in1_t, rounding_mode);
    }
  };
})

REGISTER_OPERATOR_FUNCTOR(aten::log, aten_log, [](Node* n) -> SROperator {
  if (!n->matches(torch::schema("aten::log.Tensor(Tensor input) -> Tensor"))) {
    LogAndDumpSchema(n);
    return nullptr;
  }
  return [](ProcessedNode* p_node) {
    const auto& in0_t = p_node->Input(0).toTensor();
    if (p_node->Output(0).isNone()) {
      p_node->Output(0) = at::cpu::log(in0_t);
      return;
    }
    auto& out_t = p_node->Output(0).toTensor();
    fastResizeToZero(out_t);
    at::cpu::log_out(out_t, in0_t);
  };
})

REGISTER_OPERATOR_FUNCTOR(aten::sub, aten_sub, [](Node* n) -> SROperator {
  if (n->matches(torch::schema(
          "aten::sub.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& in0_t = p_node->Input(0).toTensor();
      const auto& in1_t = p_node->Input(1).toTensor();
      const auto alpha = p_node->Input(2).toScalar();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::sub(in0_t, in1_t, alpha);
        return;
      }
      auto& out_t = p_node->Output(0).toTensor();
      fastResizeToZero(out_t);
      at::cpu::sub_out(out_t, in0_t, in1_t, alpha);
    };
  }
  if (n->matches(torch::schema(
          "aten::sub.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& in0_t = p_node->Input(0).toTensor();
      const auto& in1_t =
          at::native::wrapped_scalar_tensor(p_node->Input(1).toScalar());
      const auto alpha = p_node->Input(2).toScalar();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::sub(in0_t, in1_t, alpha);
        return;
      }
      auto& out_t = p_node->Output(0).toTensor();
      fastResizeToZero(out_t);
      at::cpu::sub_out(out_t, in0_t, in1_t, alpha);
    };
  }
  LogAndDumpSchema(n);
  return nullptr;
})

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
          p_node->Output(0) = at::cpu::clamp_min(in0_t, in1_s);
          return;
        }
        auto& out_t = p_node->Output(0).toTensor();
        fastResizeToZero(out_t);
        at::cpu::clamp_min_out(out_t, in0_t, in1_s);
      };
    })

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
      return;
    }
    auto& out_t = p_node->Output(0).toTensor();
    fastResizeToZero(out_t);
    if (in0_t.is_contiguous() && dim.has_value()) {
      at::native::c2_argmin_out(out_t, in0_t, dim.value(), keepdim);
      return;
    }
    at::cpu::argmin_out(out_t, in0_t, dim, keepdim);
  };
})

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
      return;
    }
    auto& out_t = p_node->Output(0).toTensor();
    fastResizeToZero(out_t);
    auto half_to_float = in_t.scalar_type() == at::ScalarType::Half &&
        dtype == at::ScalarType::Float;
    at::cpu::_softmax_out(out_t, in_t, dim, half_to_float);
  };
})

namespace {

c10::MaybeOwned<at::Tensor> borrow_from_optional_tensor_ivalue(
    const IValue& iv) {
  if (iv.isNone()) {
    return c10::MaybeOwned<at::Tensor>::owned(std::in_place);
  }
  return c10::MaybeOwned<at::Tensor>::borrowed(iv.toTensor());
}

} // namespace

REGISTER_OPERATOR_FUNCTOR(aten::layer_norm, aten_layer_norm, [](Node* n) -> SROperator {
  if (!sr_schema_check(
          n,
          "aten::layer_norm(Tensor input, int[] normalized_shape, Tensor? weight=None, Tensor? bias=None, float eps=1e-05, bool cudnn_enable=True) -> Tensor")) {
    return nullptr;
  }
  return [](ProcessedNode* p_node) {
    // ignore Input(5): `bool cudnn_enable=True`
    const auto& input = p_node->Input(0).toTensor();
    const auto normalized_shape = p_node->Input(1).toDimVector();
    float eps = p_node->Input(4).toDouble();

    c10::MaybeOwned<at::Tensor> weight_maybe_owned =
        borrow_from_optional_tensor_ivalue(p_node->Input(2));
    const at::Tensor& weight = *weight_maybe_owned;
    c10::MaybeOwned<at::Tensor> bias_maybe_owned =
        borrow_from_optional_tensor_ivalue(p_node->Input(3));
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
          std::nullopt /* dtype */,
          std::nullopt /* layout */,
          std::nullopt /* device */,
          std::nullopt /* pin_memory */,
          at::MemoryFormat::Contiguous);
    } else {
      at::native::resize_(
          p_node->Output(0).toTensor(), X->sizes(), std::nullopt);
    }
    at::Tensor& output = p_node->Output(0).toTensor();
    at::native::layer_norm_cpu_out(output, *X, *gamma, *beta, eps, M, N);
  };
})

REGISTER_OPERATOR_FUNCTOR(aten::norm, aten_norm, [](Node* n) -> SROperator {
  if (n->matches(torch::schema(
          "aten::norm.ScalarOpt_dtype(Tensor self, Scalar? p, *, ScalarType dtype) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& in0_t = p_node->Input(0).toTensor();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = create_empty_from(in0_t);
      }
      auto& out_t = p_node->Output(0).toTensor();
      fastResizeToZero(out_t);
      const auto in1_s = p_node->Input(1).toOptional<at::Scalar>();
      at::cpu::norm_outf(
          in0_t,
          in1_s,
          c10::IntArrayRef{},
          false,
          p_node->Input(2).toScalarType(),
          out_t);
    };
  }
  if (n->matches(torch::schema(
          "aten::norm.ScalarOpt_dim_dtype(Tensor self, Scalar? p, int[1] dim, bool keepdim, *, ScalarType dtype) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& in0_t = p_node->Input(0).toTensor();

      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = create_empty_from(in0_t);
      }
      auto& out_t = p_node->Output(0).toTensor();
      fastResizeToZero(out_t);

      const auto in1_s = p_node->Input(1).toOptional<at::Scalar>();
      at::cpu::norm_outf(
          in0_t,
          in1_s,
          p_node->Input(2).toDimVector(), // dim
          p_node->Input(3).toBool(), // keepdim
          p_node->Input(4).toScalarType(), // dtype
          out_t);
    };
  }
  if (n->matches(torch::schema(
          "aten::norm.ScalarOpt_dim(Tensor self, Scalar? p, int[1] dim, bool keepdim=False) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& in0_t = p_node->Input(0).toTensor();

      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = create_empty_from(in0_t);
      }
      auto& out_t = p_node->Output(0).toTensor();
      fastResizeToZero(out_t);

      const auto in1_s = p_node->Input(1).toOptional<at::Scalar>();
      at::cpu::norm_outf(
          in0_t,
          in1_s,
          p_node->Input(2).toDimVector(), // dim
          p_node->Input(3).toBool(), // keepdim
          out_t);
    };
  }
  LogAndDumpSchema(n);
  return nullptr;
})

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
      return;
    }
    auto& out_t = p_node->Output(0).toTensor();
    fastResizeToZero(out_t);
    at::native::matmul_out(in0_t, in1_t, out_t);
  };
})

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
          std::nullopt,
          c10::kCPU,
          false,
          output_scale,
          output_zero_point,
          std::nullopt);
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
})

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
              std::nullopt,
              c10::kCPU,
              false,
              output_scale,
              output_zero_point,
              std::nullopt);
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
    })

namespace {

template <bool has_relu>
void apply_dynamic_out_functor(
    c10::intrusive_ptr<LinearPackedParamsBase> packed_weight,
    const at::Tensor& input,
    at::Tensor& out,
    bool reduce_range);

template <>
void apply_dynamic_out_functor<false>(
    c10::intrusive_ptr<LinearPackedParamsBase> packed_weight,
    const at::Tensor& input,
    at::Tensor& out,
    bool reduce_range) {
  packed_weight->apply_dynamic_out(input, out, reduce_range);
}

template <>
void apply_dynamic_out_functor<true>(
    c10::intrusive_ptr<LinearPackedParamsBase> packed_weight,
    const at::Tensor& input,
    at::Tensor& out,
    bool reduce_range) {
  // The implementation of PackedLinearWeightFp16::apply_dynamic_impl does not
  // handle relu. Currently, it ignores the `ReluFused` template parameter.
  // So, we explicitly do the relu here.
  packed_weight->apply_dynamic_out(input, out, reduce_range);
  out.relu_();
}

template <bool has_relu>
SROperator quantized_linear_dynamic_fp16_impl(Node* n) {
  const auto weight = toIValue(n->inputs()[1]);
  c10::intrusive_ptr<LinearPackedParamsBase> packed_weight;
  if (weight) {
    packed_weight = weight->toCustomClass<LinearPackedParamsBase>();
  }
  if (packed_weight) {
    return [packed_weight](ProcessedNode* p_node) {
      const auto& input = p_node->Input(0).toTensor();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = create_empty_from(input, at::kFloat);
      }
      auto& out_t = p_node->Output(0).toTensor();
      fastResizeToZero(out_t);
      apply_dynamic_out_functor<has_relu>(packed_weight, input, out_t, false);
    };
  } else {
    return [](ProcessedNode* p_node) {
      const auto& input = p_node->Input(0).toTensor();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = create_empty_from(input, at::kFloat);
      }
      auto& out_t = p_node->Output(0).toTensor();
      fastResizeToZero(out_t);
      // Weights could be quantized on the fly
      auto packed_weight_tmp =
          p_node->Input(1).toCustomClass<LinearPackedParamsBase>();
      apply_dynamic_out_functor<has_relu>(
          packed_weight_tmp, input, out_t, false);
    };
  }
}

} // namespace

REGISTER_OPERATOR_FUNCTOR(
    quantized::linear_dynamic_fp16,
    quantized_linear_dynamic_fp16,
    [](Node* n) -> SROperator {
      if (!n->matches(torch::schema(
              "quantized::linear_dynamic_fp16(Tensor X, __torch__.torch.classes."
              "quantized.LinearPackedParamsBase W_prepack) -> Tensor Y"))) {
        LogAndDumpSchema(n);
        return nullptr;
      }
      return quantized_linear_dynamic_fp16_impl<false>(n);
    })

REGISTER_OPERATOR_FUNCTOR(
    quantized::linear_relu_dynamic_fp16,
    quantized_linear_relu_dynamic_fp16,
    [](Node* n) -> SROperator {
      if (!n->matches(torch::schema(
              "quantized::linear_relu_dynamic_fp16(Tensor X, __torch__.torch.classes."
              "quantized.LinearPackedParamsBase W_prepack) -> Tensor Y"))) {
        LogAndDumpSchema(n);
        return nullptr;
      }
      return quantized_linear_dynamic_fp16_impl<true>(n);
    })

// device & pin_memory matter only when CUDA is enabled.
static bool hasTensorWithOptions(
    const IValue& ivalue,
    std::optional<c10::ScalarType> dtype,
    std::optional<c10::Layout> layout) {
  if (!ivalue.isTensor()) {
    return false;
  }
  const auto& tensor = ivalue.toTensor();
  if (dtype == tensor.dtype().toScalarType() &&
      layout == tensor.options().layout_opt()) {
    return true;
  }
  VLOG(1) << "tensor exists, but tensor options were different";
  return false;
}

static bool hasTensorWithOptions(
    const IValue& ivalue,
    std::optional<c10::ScalarType> dtype,
    std::optional<c10::Layout> layout,
    std::optional<c10::MemoryFormat> memory_format) {
  return hasTensorWithOptions(ivalue, dtype, layout) &&
      (memory_format == ivalue.toTensor().options().memory_format_opt());
}

REGISTER_OPERATOR_FUNCTOR(aten::full, aten_full, [](Node* n) -> SROperator {
  if (!n->matches(torch::schema(
          "aten::full(int[] size, Scalar fill_value, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor"))) {
    LogAndDumpSchema(n);
    return nullptr;
  }
  return [](ProcessedNode* p_node) {
    const auto& size = p_node->Input(0).toDimVector();
    const auto fill_value = p_node->Input(1).toScalar();
    const auto dtype = p_node->Input(2).toOptional<c10::ScalarType>();
    const auto layout = p_node->Input(3).toOptional<c10::Layout>();
    if (!hasTensorWithOptions(p_node->Output(0), dtype, layout)) {
      const auto device = p_node->Input(4).toOptional<c10::Device>();
      const auto pin_memory = p_node->Input(5).toOptional<bool>();
      p_node->Output(0) =
          at::native::full(size, fill_value, dtype, layout, device, pin_memory);
      return;
    }
    p_node->Output(0) =
        at::native::full_out(size, fill_value, p_node->Output(0).toTensor());
  };
})

REGISTER_OPERATOR_FUNCTOR(aten::full_like, aten_full_like, [](Node* n) -> SROperator {
  if (!n->matches(torch::schema(
          "aten::full_like(Tensor self, Scalar fill_value, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor"))) {
    LogAndDumpSchema(n);
    return nullptr;
  }
  return [](ProcessedNode* p_node) {
    const auto in1_s = p_node->Input(1).toScalar();
    const auto& in0_t = p_node->Input(0).toTensor();
    const auto dtype = p_node->Input(2).toOptional<c10::ScalarType>();
    const auto layout = p_node->Input(3).toOptional<c10::Layout>();
    if (!hasTensorWithOptions(p_node->Output(0), dtype, layout)) {
      const auto device = p_node->Input(4).toOptional<c10::Device>();
      const auto pin_memory = p_node->Input(5).toOptional<bool>();
      const auto memory_format =
          p_node->Input(6).toOptional<c10::MemoryFormat>();

      p_node->Output(0) = at::native::empty_like(
          in0_t, dtype, layout, device, pin_memory, memory_format);
    }
    auto& out_t = p_node->Output(0).toTensor();
    at::native::resize_(out_t, in0_t.sizes(), std::nullopt);
    at::native::fill_out(out_t, in1_s);
  };
})

REGISTER_OPERATOR_FUNCTOR(aten::ones, aten_ones, [](Node* n) -> SROperator {
  if (!n->matches(torch::schema(
          "aten::ones(int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor"))) {
    LogAndDumpSchema(n);
    return nullptr;
  }
  return [](ProcessedNode* p_node) {
    const auto size = p_node->Input(0).toDimVector();
    if (p_node->Output(0).isNone()) {
      const auto dtype = p_node->Input(1).toOptional<c10::ScalarType>();
      const auto layout = p_node->Input(2).toOptional<c10::Layout>();
      const auto device = p_node->Input(3).toOptional<c10::Device>();
      const auto pin_memory = p_node->Input(4).toOptional<bool>();
      p_node->Output(0) =
          at::native::ones(size, dtype, layout, device, pin_memory);
      return;
    }
    auto& out_t = p_node->Output(0).toTensor();
    fastResizeToZero(out_t);
    at::native::ones_out(size, out_t);
  };
})

REGISTER_OPERATOR_FUNCTOR(aten::ones_like, aten_ones_like, [](Node* n) -> SROperator {
  if (!n->matches(torch::schema(
          "aten::ones_like(Tensor self, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor"))) {
    LogAndDumpSchema(n);
    return nullptr;
  }
  return [](ProcessedNode* p_node) {
    const auto& self = p_node->Input(0).toTensor();
    const auto dtype = p_node->Input(1).toOptional<c10::ScalarType>();
    const auto layout = p_node->Input(2).toOptional<c10::Layout>();
    const auto device = p_node->Input(3).toOptional<c10::Device>();
    const auto pin_memory = p_node->Input(4).toOptional<bool>();
    const auto memory_format = p_node->Input(5).toOptional<c10::MemoryFormat>();
    if (!hasTensorWithOptions(
            p_node->Output(0), dtype, layout, memory_format)) {
      p_node->Output(0) = at::native::ones_like(
          self, dtype, layout, device, pin_memory, memory_format);
      return;
    }
    auto& out_t = p_node->Output(0).toTensor();
    fastResizeToZero(out_t);
    at::native::ones_out(self.sizes(), out_t);
  };
})

REGISTER_OPERATOR_FUNCTOR(aten::zeros, aten_zeros, [](Node* n) -> SROperator {
  if (!n->matches(torch::schema(
          "aten::zeros(int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor"))) {
    LogAndDumpSchema(n);
    return nullptr;
  }
  return [](ProcessedNode* p_node) {
    const auto size = p_node->Input(0).toDimVector();
    const auto dtype = p_node->Input(1).toOptional<c10::ScalarType>();
    const auto layout = p_node->Input(2).toOptional<c10::Layout>();
    if (!hasTensorWithOptions(p_node->Output(0), dtype, layout)) {
      p_node->Output(0) = at::compositeexplicitautograd::zeros(
          size, dtype, layout, std::nullopt, std::nullopt);
      return;
    }
    auto& out_t = p_node->Output(0).toTensor();
    fastResizeToZero(out_t);
    at::compositeexplicitautograd::zeros_out(out_t, size);
  };
})

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
      return;
    }
    auto& out_t = p_node->Output(0).toTensor();
    fastResizeToZero(out_t);
    at::native::linear_out(out_t, in0_t, in1_t, in2_t);
  };
})

REGISTER_OPERATOR_FUNCTOR(aten::linalg_norm, aten_linalg_norm, [](Node* n) -> SROperator {
  if (n->matches(torch::schema(
          "aten::linalg_norm(Tensor self, Scalar? ord=None, int[1]? dim=None, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& input = p_node->Input(0).toTensor();
      const auto dim = p_node->Input(2).toDimVector();
      const auto keepdim = p_node->Input(3).toBool();
      const auto dtype = p_node->Input(4).toOptional<c10::ScalarType>();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::native::linalg_norm(
            input,
            p_node->Input(1).toOptional<at::Scalar>(),
            dim,
            keepdim,
            dtype);
        return;
      }
      auto& output = p_node->Output(0).toTensor();
      fastResizeToZero(output);
      at::native::linalg_norm_out(
          input,
          p_node->Input(1).toOptional<at::Scalar>(),
          dim,
          keepdim,
          dtype,
          output);
    };
  }
  if (n->matches(torch::schema(
          "aten::linalg_norm.ord_str(Tensor self, str ord, int[1]? dim=None, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& input = p_node->Input(0).toTensor();
      const auto dim = p_node->Input(2).toDimVector();
      const auto keepdim = p_node->Input(3).toBool();
      const auto dtype = p_node->Input(4).toOptional<c10::ScalarType>();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::native::linalg_norm(
            input, p_node->Input(1).toStringView(), dim, keepdim, dtype);
        return;
      }
      auto& output = p_node->Output(0).toTensor();
      fastResizeToZero(output);
      at::native::linalg_norm_out(
          input, p_node->Input(1).toStringRef(), dim, keepdim, dtype, output);
    };
  }
  LogAndDumpSchema(n);
  return nullptr;
})

REGISTER_OPERATOR_FUNCTOR(aten::cat, aten_cat, [](Node* n) -> SROperator {
  if (!n->matches(
          torch::schema("aten::cat(Tensor[] tensors, int dim=0) -> Tensor"))) {
    LogAndDumpSchema(n);
    return nullptr;
  }
  return [](ProcessedNode* p_node) {
    const auto inputs = p_node->Input(0).toTensorVector();
    TORCH_CHECK(!inputs.empty(), "concat expects non-empty tensor list");
    const auto dim = p_node->Input(1).toInt();
    if (p_node->Output(0).isNone()) {
      p_node->Output(0) = at::cpu::cat(inputs, dim);
      return;
    }
    auto& output = p_node->Output(0).toTensor();
    fastResizeToZero(output);
    at::cpu::cat_outf(inputs, dim, output);
  };
})

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
})

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
    })

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_OPERATOR_FUNCTOR(
    prim::VarConcat,
    prim_VarConcat,
    [](Node* n) -> SROperator {
      if (!sr_schema_check_kind(n, prim::VarConcat)) {
        return nullptr;
      }
      return [](ProcessedNode* p_node) {
        const size_t num_inputs = p_node->num_inputs();
        std::vector<at::Tensor> inputs(num_inputs - 1);
        for (const auto i : c10::irange(num_inputs - 1)) {
          inputs[i] = p_node->Input(i).toTensor();
        }
        auto dim = p_node->Input(num_inputs - 1).toInt();
        if (p_node->Output(0).isNone()) {
          p_node->Output(0) = at::cpu::cat(inputs, dim);
          return;
        }
        auto& out_t = p_node->Output(0).toTensor();
        fastResizeToZero(out_t);
        at::cpu::cat_outf(inputs, dim, out_t);
      };
    })

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
  at::native::resize_(out, input.sizes(), std::nullopt);

  const auto input_contig = input.expect_contiguous();
  auto output_contig = out.expect_contiguous();

  AT_DISPATCH_ALL_TYPES(input.scalar_type(), "signed_log1p_kernel", [&]() {
    const auto input_data = input_contig->const_data_ptr<scalar_t>();
    auto output_data = output_contig->mutable_data_ptr<float>();
    const auto N = input.numel();

    for (const auto i : c10::irange(N)) {
      const int sign = input_data[i] < 0 ? -1 : 1;
      output_data[i] = std::log1p(abs_if_signed(input_data[i])) * sign;
    }
  });
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
        if (!te || !te->checkInput<float>(input)) {
          fastResizeToZero(out);
          signed_log1p_out(out, input);
          return;
        }
        at::native::resize_(out, input.sizes(), std::nullopt);
        int64_t nn = input.numel();
        te->call({out.data_ptr(), input.data_ptr(), &nn});
      };
    })

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
            return;
          }
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          at::cpu::remainder_out(out, self, p_node->Input(1).toTensor());
        };
      }
      if (n->matches(torch::schema(
              "aten::remainder.Scalar(Tensor self, Scalar other) -> Tensor"))) {
        return [](ProcessedNode* p_node) {
          const auto& self = p_node->Input(0).toTensor();
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) =
                at::native::remainder(self, p_node->Input(1).toScalar());
            return;
          }
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          at::native::remainder_out(self, p_node->Input(1).toScalar(), out);
        };
      }

      // Unrecognized overload
      LogAndDumpSchema(n);
      return nullptr;
    })

REGISTER_OPERATOR_FUNCTOR(aten::where, aten_where, [](Node* n) -> SROperator {
  if (n->matches(torch::schema(
          "aten::where.self(Tensor condition, Tensor self, Tensor other) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& cond = p_node->Input(0).toTensor();
      const auto& self = p_node->Input(1).toTensor();
      const auto& other = p_node->Input(2).toTensor();

      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = create_empty_from(self);
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::native::where_self_out(cond, self, other, out);
    };
  }

  LogAndDumpSchema(n);
  return nullptr;
})

REGISTER_OPERATOR_FUNCTOR(
    prim::NumToTensor,
    prim_NumToTensor,
    [](Node* n) -> SROperator {
      if (n->matches(
              torch::schema("prim::NumToTensor.Scalar(Scalar s) -> Tensor")) ||
          n->matches(
              torch::schema("prim::NumToTensor.bool(bool a) -> Tensor"))) {
        return [](ProcessedNode* pnode) {
          const auto scalar = pnode->Input(0).toScalar();
          if (pnode->Output(0).isNone()) {
            pnode->Output(0) = at::scalar_to_tensor(scalar);
            return;
          }
          auto& out = pnode->Output(0).toTensor();
          at::detail::scalar_fill(out, scalar);
        };
      }
      LogAndDumpSchema(n);
      return nullptr;
    })

REGISTER_OPERATOR_FUNCTOR(
    quantized::embedding_bag_byte_unpack,
    quantized_embedding_bag_byte_unpack,
    [](Node* n) -> SROperator {
      if (!sr_schema_check(
              n,
              "quantized::embedding_bag_byte_unpack(Tensor weight) -> Tensor")) {
        return nullptr;
      }
      return [](ProcessedNode* pnode) {
        auto& weight = pnode->Input(0).toTensor();
        if (pnode->Output(0).isNone()) {
          pnode->Output(0) = at::empty(
              {},
              weight.options().dtype(at::kFloat),
              weight.suggest_memory_format());
        }
        auto& out = pnode->Output(0).toTensor();
        at::native::qembeddingbag_byte_unpack_out(out, weight);
      };
    })

} // namespace torch::jit
