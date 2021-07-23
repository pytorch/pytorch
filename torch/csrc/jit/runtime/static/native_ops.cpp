#include <torch/csrc/jit/runtime/static/ops.h>

#include <ATen/CPUFunctions.h>
#include <ATen/NativeFunctions.h>
#include <ATen/ScalarOps.h>
#include <ATen/TensorUtils.h>
#include <ATen/native/IndexingUtils.h>
#include <ATen/native/Resize.h>
#include <ATen/native/TensorAdvancedIndexing.h>
#include <c10/util/irange.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/runtime/vararg_functions.h>

namespace torch {
namespace jit {

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
C10_DEFINE_REGISTRY(SRNativeOperatorRegistry, SROperatorFunctor);

bool nativeOpIsRegistered(const c10::Symbol& op_name) {
  const std::string name(op_name.toQualString());
  return SRNativeOperatorRegistry()->Has(name);
}

std::function<void(ProcessedNode*)> getNativeOperation(Node* n) {
  auto op_name = n->kind().toQualString();
  if (SRNativeOperatorRegistry()->Has(op_name)) {
    return SRNativeOperatorRegistry()->Create(op_name)->Generate(n);
  }
  return nullptr;
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_NATIVE_OPERATOR_FUNCTOR(
    prim::TupleConstruct,
    prim_TupleConstruct,
    [](Node* n) -> SROperator {
      return [](ProcessedNode* p_node) {
        // prepare inputs
        std::vector<IValue> stack;
        const size_t size = p_node->inputs().size();
        stack.reserve(size);
        for (const auto i : c10::irange(size)) {
          stack.emplace_back(p_node->Input(i));
        }
        // run op
        auto* node = p_node->node();
        const auto& type = node->output()->type()->expect<TupleType>();
        if (type->name().has_value()) {
          namedTupleConstruct(stack, type, node->inputs().size());
        } else {
          tupleConstruct(stack, node->inputs().size());
        }
        // put output back
        p_node->Output(0) = std::move(stack[0]);
      };
    });

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_NATIVE_OPERATOR_FUNCTOR(
    prim::DictConstruct,
    prim_DictConstruct,
    [](Node* n) -> SROperator {
      return [](ProcessedNode* p_node) {
        // prepare inputs
        std::vector<IValue> stack;
        const size_t size = p_node->inputs().size();
        stack.reserve(size);
        for (const auto i : c10::irange(size)) {
          stack.emplace_back(p_node->Input(i));
        }
        // run op
        auto* node = p_node->node();
        dictConstruct(
            stack,
            node->output()->type()->expectRef<DictType>(),
            node->inputs().size());
        // put output back
        p_node->Output(0) = std::move(stack[0]);
      };
    });

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_NATIVE_OPERATOR_FUNCTOR(
    aten::__getitem__,
    aten_getitem,
    [](Node* n) -> SROperator {
      if (n->inputs().size() != 2) {
        return nullptr;
      }
      // TODO: make __getitem__ work for other container types
      if (n->input(0)->type()->castRaw<DictType>() == nullptr) {
        return nullptr;
      }
      return [](ProcessedNode* p_node) {
        auto dict = p_node->Input(0).toGenericDict();
        auto key = p_node->Input(1);
        auto value = dict.find(key);
        TORCH_CHECK(value != dict.end(), "Key not in dict: ", key);
        p_node->Output(0) = value->value();
      };
    });

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_NATIVE_OPERATOR_FUNCTOR(
    prim::ListConstruct,
    prim_ListConstruct,
    [](Node* n) -> SROperator {
      return [](ProcessedNode* p_node) {
        // prepare inputs
        std::vector<IValue> stack;
        const size_t size = p_node->inputs().size();
        stack.reserve(size);
        for (const auto i : c10::irange(size)) {
          stack.emplace_back(p_node->Input(i));
        }
        // run op
        listConstruct(
            stack,
            p_node->node()->output()->type()->expectRef<ListType>(),
            p_node->inputs().size());
        // put output back
        p_node->Output(0) = std::move(stack[0]);
      };
    });

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_NATIVE_OPERATOR_FUNCTOR(
    prim::ListUnpack,
    prim_ListUnpack,
    [](Node* n) -> SROperator {
      return [](ProcessedNode* p_node) {
        // prepare inputs
        std::vector<IValue> stack;
        const size_t size = p_node->inputs().size();
        stack.reserve(size);
        for (const auto i : c10::irange(size)) {
          stack.emplace_back(p_node->Input(i));
        }
        // run op
        size_t num_outputs = p_node->outputs().size();
        listUnpack(stack, num_outputs);
        // put output back
        DCHECK_EQ(stack.size(), num_outputs);
        // NOLINTNEXTLINE(clang-diagnostic-sign-compare)
        for (auto i = 0; i < num_outputs; i++) {
          p_node->Output(i) = std::move(stack[i]);
        }
      };
    });

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_NATIVE_OPERATOR_FUNCTOR(
    prim::GetAttr,
    prim_GetAttr,
    [](Node* n) -> SROperator {
      return [](ProcessedNode* p_node) {
        auto module = p_node->Input(0).toObject();
        Node* node = p_node->node();
        const auto type = node->input()->type()->expect<ClassType>();
        const auto& field = node->s(attr::name);
        const auto slot = type->getAttributeSlot(field);
        p_node->Output(0) = module->getSlot(slot);
      };
    });

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_NATIVE_OPERATOR_FUNCTOR(
    prim::SetAttr,
    prim_SetAttr,
    [](Node* n) -> SROperator {
      return [](ProcessedNode* p_node) {
        auto module = p_node->Input(0).toObject();
        Node* node = p_node->node();
        const auto type = node->inputs()[0]->type()->expect<ClassType>();
        const auto& field = node->s(attr::name);
        const auto slot = type->getAttributeSlot(field);
        module->setSlot(slot, p_node->Input(1));
      };
    });

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_NATIVE_OPERATOR_FUNCTOR(
    aten::transpose,
    aten_transpose,
    [](Node* n) -> SROperator {
      if (!n->matches(torch::schema(
              "aten::transpose.int(Tensor(a) self, int dim0, int dim1) -> Tensor(a)"))) {
        LogAndDumpSchema(n);
        return nullptr;
      }
      return [](ProcessedNode* p_node) {
        const auto& in0_t = p_node->Input(0).toTensor();
        const auto in1_i = p_node->Input(1).toInt();
        const auto in2_i = p_node->Input(2).toInt();
        p_node->Output(0) = at::native::transpose(in0_t, in1_i, in2_i);
      };
    });

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_NATIVE_OPERATOR_FUNCTOR(aten::flatten, aten_flatten, [](Node* n) -> SROperator {
  if (!n->matches(torch::schema(
          "aten::flatten.using_ints(Tensor(a) self, int start_dim=0, int end_dim=-1) -> Tensor(a)"))) {
    LogAndDumpSchema(n);
    return nullptr;
  }
  return [](ProcessedNode* p_node) {
    const auto& in0_t = p_node->Input(0).toTensor();
    const auto in1_i = p_node->Input(1).toInt();
    const auto in2_i = p_node->Input(2).toInt();
    p_node->Output(0) = at::native::flatten(in0_t, in1_i, in2_i);
  };
});

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_NATIVE_OPERATOR_FUNCTOR(
    aten::permute,
    aten_permute,
    [](Node* n) -> SROperator {
      if (!n->matches(torch::schema(
              "aten::permute(Tensor(a) self, int[] dims) -> Tensor(a)"))) {
        LogAndDumpSchema(n);
        return nullptr;
      }
      return [](ProcessedNode* p_node) {
        const auto& in0_t = p_node->Input(0).toTensor();
        const auto in1_iv = p_node->Input(1).toIntVector();
        p_node->Output(0) = at::native::permute(in0_t, in1_iv);
      };
    });

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_NATIVE_OPERATOR_FUNCTOR(
    aten::reshape,
    aten_reshape,
    [](Node* n) -> SROperator {
      if (!n->matches(torch::schema(
              "aten::reshape(Tensor(a) self, int[] shape) -> Tensor(a)"))) {
        LogAndDumpSchema(n);
        return nullptr;
      }
      return [](ProcessedNode* p_node) {
        const auto& in0_t = p_node->Input(0).toTensor();
        const auto in1_iv = p_node->Input(1).toIntVector();
        p_node->Output(0) = at::native::reshape(in0_t, in1_iv);
      };
    });

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_NATIVE_OPERATOR_FUNCTOR(aten::slice, aten_slice, [](Node* n) -> SROperator {
  if (!n->matches(torch::schema(
          "aten::slice.Tensor(Tensor(a) self, int dim=0, int? start=0, int? end=9223372036854775807, int step=1) -> Tensor(a)"))) {
    LogAndDumpSchema(n);
    return nullptr;
  }
  return [](ProcessedNode* p_node) {
    const auto& in0_t = p_node->Input(0).toTensor();
    const auto in1_i = p_node->Input(1).toInt();
    const auto in2_i = p_node->Input(2).toInt();
    const auto in3_i = p_node->Input(3).toInt();
    const auto in4_i = p_node->Input(4).toInt();
    p_node->Output(0) = at::native::slice(in0_t, in1_i, in2_i, in3_i, in4_i);
  };
});

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_NATIVE_OPERATOR_FUNCTOR(aten::narrow, aten_narrow, [](Node* n) -> SROperator {
  if (!n->matches(torch::schema(
          "aten::narrow(Tensor(a) self, int dim, int start, int length) -> Tensor(a)")) &&
      !n->matches(torch::schema(
          "aten::narrow.Tensor(Tensor(a) self, int dim, Tensor start, int length) -> Tensor(a)"))) {
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
    const auto length = p_node->Input(3).toInt(); // length
    TORCH_CHECK(
        self.dim() > 0, "narrow() cannot be applied to a 0-dim tensor.");
    auto cur_size = self.sizes()[dim];
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
    p_node->Output(0) = at::native::slice(self, dim, start, start + length, 1);
  };
});

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_NATIVE_OPERATOR_FUNCTOR(aten::to, aten_to, [](Node* n) -> SROperator {
  if (!n->matches(torch::schema(
          "aten::to.other(Tensor(a) self, Tensor other, bool non_blocking=False, bool copy=False, MemoryFormat? memory_format=None) -> Tensor(a)")) &&
      !n->matches(torch::schema(
          "aten::to.dtype(Tensor(a) self, ScalarType dtype, bool non_blocking=False, bool copy=False, MemoryFormat? memory_format=None) -> Tensor(a)"))) {
    LogAndDumpSchema(n);
    return nullptr;
  }
  return [](ProcessedNode* p_node) {
    const auto& in0_t = p_node->Input(0).toTensor();
    const auto in2_i = p_node->Input(2).toBool();
    const auto in3_i = p_node->Input(3).toBool();
    const auto in4_o = p_node->Input(4).toOptional<at::MemoryFormat>();
    if (p_node->Input(1).isTensor()) {
      // to.other(Tensor(a) self, Tensor other, bool non_blocking=False, bool
      // copy=False, MemoryFormat? memory_format=None) -> Tensor(a)
      const auto in1_t = p_node->Input(1).toTensor();
      p_node->Output(0) = at::native::to(in0_t, in1_t, in2_i, in3_i, in4_o);
    } else {
      // to.dtype(Tensor(a) self, ScalarType dtype, bool non_blocking=False,
      // bool copy=False, MemoryFormat? memory_format=None) -> Tensor(a)
      const auto in1_i = p_node->Input(1).toScalarType();
      p_node->Output(0) = at::native::to(in0_t, in1_i, in2_i, in3_i, in4_o);
    }
    // in case that Output(0) is an alias of in0_t, copy the tensor.
    if (p_node->Output(0).toTensor().unsafeGetTensorImpl() ==
        in0_t.unsafeGetTensorImpl()) {
      p_node->Output(0) = in0_t.clone();
    }
  };
});

} // namespace jit
} // namespace torch
