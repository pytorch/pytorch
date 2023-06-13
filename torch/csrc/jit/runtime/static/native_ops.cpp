#include <torch/csrc/jit/passes/inliner.h>
#include <torch/csrc/jit/runtime/static/impl.h>
#include <torch/csrc/jit/runtime/static/ops.h>

#include <ATen/CPUFunctions.h>
#include <ATen/NativeFunctions.h>
#include <ATen/ScalarOps.h>
#include <ATen/TensorUtils.h>
#include <ATen/native/IndexingUtils.h>
#include <ATen/native/NonSymbolicBC.h>
#include <ATen/native/Resize.h>
#include <ATen/native/TensorAdvancedIndexing.h>
#include <c10/util/intrusive_ptr.h>
#include <c10/util/irange.h>
#include <c10/util/ssize.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/mobile/promoted_prim_ops.h>
#include <torch/csrc/jit/runtime/register_ops_utils.h>
#include <torch/csrc/jit/runtime/vararg_functions.h>

namespace {
constexpr auto createBorrowedIValue =
    c10::MaybeOwnedTraits<c10::IValue>::createBorrow;
} // namespace
namespace torch {
namespace jit {

namespace {

std::vector<IValue> boxInputs(const ProcessedNode& pnode) {
  std::vector<IValue> result;
  for (const auto i : c10::irange(pnode.num_inputs())) {
    result.push_back(pnode.Input(i));
  }
  return result;
}

} // namespace

C10_DEFINE_REGISTRY(SRNativeOperatorRegistry, SROperatorFunctor);

bool nativeOpIsRegistered(const c10::Symbol& op_name) {
  const std::string name(op_name.toQualString());
  return SRNativeOperatorRegistry()->Has(name);
}

SROperator getNativeOperation(Node* n) {
  auto op_name = n->kind().toQualString();
  if (SRNativeOperatorRegistry()->Has(op_name)) {
    return SRNativeOperatorRegistry()->Create(op_name)->Generate(n);
  }
  return nullptr;
}

REGISTER_NATIVE_OPERATOR_FUNCTOR(
    prim::TupleConstruct,
    prim_TupleConstruct,
    [](Node* n) -> SROperator {
      if (!sr_schema_check_kind(n, prim::TupleConstruct)) {
        return nullptr;
      }
      return [](ProcessedNode* p_node) {
        // prepare inputs
        auto stack = boxInputs(*p_node);
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

REGISTER_NATIVE_OPERATOR_FUNCTOR(
    prim::TupleUnpack,
    prim_TupleUnpack,
    [](Node* n) -> SROperator {
      if (!sr_schema_check_kind(n, prim::TupleUnpack)) {
        return nullptr;
      }
      return [](ProcessedNode* p_node) {
        const auto& elems = p_node->Input(0).toTupleRef().elements();
        const size_t num_outputs = p_node->outputs().size();
        TORCH_CHECK(
            num_outputs == elems.size(),
            "Number of outputs must match number of tuple elements.")
        for (size_t i = 0; i < num_outputs; ++i) {
          p_node->Output(i) = elems[i];
        }
      };
    });

REGISTER_NATIVE_OPERATOR_FUNCTOR(
    prim::DictConstruct,
    prim_DictConstruct,
    [](Node* n) -> SROperator {
      if (!sr_schema_check_kind(n, prim::DictConstruct)) {
        return nullptr;
      }
      auto dict_type = n->output()->type()->expect<DictType>();
      const auto num_inputs = n->inputs().size();
      TORCH_DCHECK_EQ(num_inputs % 2, 0);
      return [dict_type = std::move(dict_type),
              num_inputs,
              dict_size = num_inputs / 2](ProcessedNode* p_node) {
        auto result = c10::impl::GenericDict(
            dict_type->containedType(0), dict_type->containedType(1));
        result.reserve(dict_size);
        for (size_t i = 0; i < num_inputs; i += 2) {
          const auto& key = p_node->Input(i);
          const auto& value = p_node->Input(i + 1);
          result.insert_or_assign(key, value);
        }
        p_node->Output(0) = result;
      };
    });

// See [Borrowed IValue Outputs]
REGISTER_NATIVE_OPERATOR_FUNCTOR(
    static_runtime::dict_unpack,
    static_runtime_dict_unpack,
    [](Node* n) -> SROperator {
      if (!sr_schema_check(n, "static_runtime::dict_unpack(...) -> ...")) {
        return nullptr;
      }
      return [](ProcessedNode* p_node) {
        DCHECK(
            static_cast<size_t>(p_node->num_inputs() - 1) ==
            p_node->outputs().size());
        auto dict = p_node->Input(0).toGenericDict();
        const auto num_inputs = p_node->num_inputs();
        for (size_t i = 1; i < num_inputs; ++i) {
          const auto& key = p_node->Input(i);
          auto value = dict.find(key);
          TORCH_CHECK(value != dict.end(), "Key not in dict: ", key);
          p_node->Output(i - 1) = createBorrowedIValue(value->value());
        }
      };
    });

REGISTER_NATIVE_OPERATOR_FUNCTOR(aten::__getitem__, aten_getitem, [](Node* n) -> SROperator {
  if (!sr_schema_check(
          n,
          // TODO: "aten::__getitem__.str(str s, int index) -> str",
          "aten::__getitem__.t(t[](a) list, int idx) -> t(*)",
          "aten::__getitem__.Dict_str(Dict(str, t) self, str key) -> t(*)",
          "aten::__getitem__.Dict_int(Dict(int, t) self, int key) -> t(*)",
          "aten::__getitem__.Dict_bool(Dict(bool, t) self, bool key) -> t(*)",
          "aten::__getitem__.Dict_float(Dict(float, t) self, float key) -> t(*)",
          "aten::__getitem__.Dict_complex(Dict(complex, t) self, complex key) -> t(*)",
          "aten::__getitem__.Dict_Tensor(Dict(Tensor, t) self, Tensor key) -> t(*)")) {
    return nullptr;
  }

  if (n->inputs().size() != 2) {
    return nullptr;
  }

  if (n->input(0)->type()->castRaw<DictType>()) {
    return [](ProcessedNode* p_node) {
      auto dict = p_node->Input(0).toGenericDict();
      const auto& key = p_node->Input(1);
      auto value = dict.find(key);
      TORCH_CHECK(value != dict.end(), "Key not in dict: ", key);
      p_node->Output(0) = value->value();
    };
  } else if (n->input(0)->type()->castRaw<ListType>()) {
    return [](ProcessedNode* p_node) {
      const auto& list = p_node->Input(0).toList();
      auto idx = p_node->Input(1).toInt();
      p_node->Output(0) = getItem(list, idx);
    };
  }

  // TODO(T98581096): make __getitem__ work for other container types
  return nullptr;
});

REGISTER_NATIVE_OPERATOR_FUNCTOR(
    prim::ListConstruct,
    prim_ListConstruct,
    [](Node* n) -> SROperator {
      if (!sr_schema_check_kind(n, prim::ListConstruct)) {
        return nullptr;
      }
      return [](ProcessedNode* p_node) {
        // prepare inputs
        auto stack = boxInputs(*p_node);
        // run op
        listConstruct(
            stack,
            p_node->node()->output()->type()->expectRef<ListType>(),
            p_node->num_inputs());
        // put output back
        p_node->Output(0) = std::move(stack[0]);
      };
    });

REGISTER_NATIVE_OPERATOR_FUNCTOR(
    prim::ListUnpack,
    prim_ListUnpack,
    [](Node* n) -> SROperator {
      if (!sr_schema_check_kind(n, prim::ListUnpack)) {
        return nullptr;
      }
      const auto num_outputs = n->outputs().size();
      return [num_outputs](ProcessedNode* p_node) {
        const auto list = p_node->Input(0).toListRef();
        TORCH_CHECK(
            list.size() == num_outputs,
            "Expected ",
            num_outputs,
            " elements in list but got ",
            list.size());
        for (const auto i : c10::irange(num_outputs)) {
          p_node->Output(i) = list[i];
        }
      };
    });

REGISTER_NATIVE_OPERATOR_FUNCTOR(
    aten::append,
    aten_append,
    [](Node* n) -> SROperator {
      if (!sr_schema_check(
              n, "aten::append.t(t[](a!) self, t(c -> *) el) -> t[](a!)")) {
        return nullptr;
      }
      return [](ProcessedNode* p_node) {
        auto list = p_node->Input(0).toList();
        list.push_back(p_node->Input(1));
      };
    });

REGISTER_NATIVE_OPERATOR_FUNCTOR(
    aten::list,
    aten_list,
    [](Node* n) -> SROperator {
      if (n->matches(torch::schema("aten::list(str t) -> str[]"))) {
        return [](ProcessedNode* p_node) {
          const auto str = p_node->Input(0).toStringRef();
          c10::List<std::string> chars;
          chars.reserve(str.size());
          for (auto c : str) {
            chars.emplace_back(1, c);
          }
          p_node->Output(0) = std::move(chars);
        };
      }

      if (n->matches(torch::schema("aten::list.t(t[] l) -> t[]"))) {
        return [](ProcessedNode* p_node) {
          const auto input = p_node->Input(0).toList();
          p_node->Output(0) = input.copy();
        };
      }

      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_NATIVE_OPERATOR_FUNCTOR(
    aten::numel,
    aten_numel,
    [](Node* n) -> SROperator {
      if (!sr_schema_check(n, "aten::numel(Tensor self) -> int")) {
        return nullptr;
      }
      return [](ProcessedNode* p_node) {
        const auto& arg = p_node->Input(0).toTensor();
        p_node->Output(0) = arg.numel();
      };
    });

REGISTER_NATIVE_OPERATOR_FUNCTOR(
    aten::cpu,
    aten_cpu,
    [](Node* n) -> SROperator {
      if (!sr_schema_check(n, "aten::cpu(Tensor self) -> Tensor")) {
        return nullptr;
      }
      return [](ProcessedNode* p_node) {
        const auto& arg = p_node->Input(0).toTensor();
        p_node->Output(0) = arg.cpu();
      };
    });

REGISTER_NATIVE_OPERATOR_FUNCTOR(
    aten::__range_length,
    aten_range_length,
    [](Node* n) -> SROperator {
      if (!sr_schema_check(
              n, "aten::__range_length(int lo, int hi, int step) -> int")) {
        return nullptr;
      }
      return [](ProcessedNode* p_node) {
        auto lo = p_node->Input(0).toInt();
        auto hi = p_node->Input(1).toInt();
        auto step = p_node->Input(2).toInt();
        // error handling when step_val == 0 during runtime
        if (step == 0) {
          throw std::runtime_error("range() arg 3 must not be zero");
        }
        if (step > 0 && lo < hi) {
          p_node->Output(0) = 1 + (hi - 1 - lo) / step;
        } else if (step < 0 && lo > hi) {
          p_node->Output(0) = 1 + (lo - 1 - hi) / (0 - step);
        } else {
          p_node->Output(0) = 0;
        }
      };
    });

REGISTER_NATIVE_OPERATOR_FUNCTOR(aten::index_put, aten_index_put, [](Node* n) -> SROperator {
  if (n->matches(torch::schema(
          "aten::index_put(Tensor self, Tensor[] indices, Tensor values, bool accumulate=False) -> Tensor")) ||
      n->matches(torch::schema(
          "aten::index_put(Tensor self, Tensor?[] indices, Tensor values, bool accumulate=False) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      const auto& indices =
          at::native::toListOfOptionalTensors(p_node->Input(1).toListRef());
      const auto& values = p_node->Input(2).toTensor();
      const auto accumulate = p_node->Input(3).toBool();
      p_node->Output(0) =
          at::native::index_put(self, indices, values, accumulate);
    };
  }

  LogAndDumpSchema(n);
  return nullptr;
});

REGISTER_NATIVE_OPERATOR_FUNCTOR(
    aten::item,
    aten_item,
    [](Node* n) -> SROperator {
      if (!sr_schema_check(n, "aten::item(Tensor self) -> Scalar")) {
        return nullptr;
      }
      return [](ProcessedNode* p_node) {
        const auto& self = p_node->Input(0).toTensor();
        p_node->Output(0) = at::native::item(self);
      };
    });

REGISTER_NATIVE_OPERATOR_FUNCTOR(
    prim::GetAttr,
    prim_GetAttr,
    [](Node* n) -> SROperator {
      if (!sr_schema_check_kind(n, prim::GetAttr)) {
        return nullptr;
      }
      return [](ProcessedNode* p_node) {
        auto& module = p_node->Input(0).toObjectRef();
        Node* node = p_node->node();
        const auto& type = node->input()->type()->expectRef<ClassType>();
        const auto& field = node->s(attr::name);
        const auto slot = type.getAttributeSlot(field);
        p_node->Output(0) = module.getSlot(slot);
      };
    });

REGISTER_NATIVE_OPERATOR_FUNCTOR(
    prim::SetAttr,
    prim_SetAttr,
    [](Node* n) -> SROperator {
      if (!sr_schema_check_kind(n, prim::SetAttr)) {
        return nullptr;
      }
      return [](ProcessedNode* p_node) {
        auto& module = p_node->Input(0).toObjectRef();
        Node* node = p_node->node();
        const auto& type = node->inputs()[0]->type()->expectRef<ClassType>();
        const auto& field = node->s(attr::name);
        const auto slot = type.getAttributeSlot(field);
        module.setSlot(slot, p_node->Input(1));
      };
    });

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
        const auto in1_iv = p_node->Input(1).toDimVector();
        p_node->Output(0) = at::native::permute(in0_t, in1_iv);
      };
    });

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
        const auto in1_iv = p_node->Input(1).toDimVector();
        p_node->Output(0) = at::native::reshape(in0_t, in1_iv);
      };
    });

REGISTER_NATIVE_OPERATOR_FUNCTOR(aten::slice, aten_slice, [](Node* n) -> SROperator {
  if (!n->matches(torch::schema(
          "aten::slice.Tensor(Tensor(a) self, int dim=0, int? start=0, int? end=9223372036854775807, int step=1) -> Tensor(a)"))) {
    LogAndDumpSchema(n);
    return nullptr;
  }
  return [](ProcessedNode* p_node) {
    const auto& in0_t = p_node->Input(0).toTensor();
    const auto in1_i = p_node->Input(1).toInt();
    const auto in2_i = p_node->Input(2).toOptional<int64_t>();
    const auto in3_i = p_node->Input(3).toOptional<int64_t>();
    const auto in4_i = p_node->Input(4).toInt();
    p_node->Output(0) = at::native::slice(in0_t, in1_i, in2_i, in3_i, in4_i);
  };
});

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

REGISTER_NATIVE_OPERATOR_FUNCTOR(aten::to, aten_to, [](Node* n) -> SROperator {
  if (n->matches(torch::schema(
          "aten::to.other(Tensor(a) self, Tensor other, bool non_blocking=False, bool copy=False, MemoryFormat? memory_format=None) -> Tensor(a)"))) {
    return [](ProcessedNode* p_node) {
      const auto& in0_t = p_node->Input(0).toTensor();
      const auto& in1_t = p_node->Input(1).toTensor();
      const auto in2_i = p_node->Input(2).toBool();
      const auto in3_i = p_node->Input(3).toBool();
      const auto in4_o = p_node->Input(4).toOptional<at::MemoryFormat>();
      p_node->Output(0) = at::native::to(in0_t, in1_t, in2_i, in3_i, in4_o);
    };
  }
  if (n->matches(torch::schema(
          "aten::to.dtype(Tensor(a) self, ScalarType dtype, bool non_blocking=False, bool copy=False, MemoryFormat? memory_format=None) -> Tensor(a)"))) {
    return [](ProcessedNode* p_node) {
      const auto& in0_t = p_node->Input(0).toTensor();
      const auto in1_i = p_node->Input(1).toScalarType();
      const auto in2_i = p_node->Input(2).toBool();
      const auto in3_i = p_node->Input(3).toBool();
      const auto in4_o = p_node->Input(4).toOptional<at::MemoryFormat>();
      p_node->Output(0) = at::native::to(in0_t, in1_i, in2_i, in3_i, in4_o);
    };
  }
  if (n->matches(torch::schema(
          "aten::to.prim_dtype(Tensor(a) self, int? dtype, bool non_blocking=False, bool copy=False) -> Tensor(a|b)"))) {
    return [](ProcessedNode* p_node) {
      const auto& in0_t = p_node->Input(0).toTensor();
      const auto in1_i = p_node->Input(1).toOptional<at::ScalarType>();
      const auto in2_i = p_node->Input(2).toBool();
      const auto in3_i = p_node->Input(3).toBool();
      // To mimick the behavior of the JIT interpreter, if both dtype
      // and copy are not set, we return self. Otherwise, we assume
      // that dtype is set.
      if (!in1_i && !in3_i) {
        p_node->Output(0) = in0_t;
      } else {
        TORCH_CHECK(
            in1_i,
            "dytpe cannot be None when copy is True for aten::to.prim_dtype");
        p_node->Output(0) = at::native::to(in0_t, *in1_i, in2_i, in3_i);
      }
    };
  }
  LogAndDumpSchema(n);
  return nullptr;
});

REGISTER_NATIVE_OPERATOR_FUNCTOR(
    aten::detach,
    aten_detach,
    [](Node* n) -> SROperator {
      if (!n->matches(
              torch::schema("aten::detach(Tensor(a) self) -> Tensor(a)"))) {
        LogAndDumpSchema(n);
        return nullptr;
      }
      return [](ProcessedNode* p_node) {
        const auto& in0_t = p_node->Input(0).toTensor();
        p_node->Output(0) = at::native::alias(in0_t);
      };
    });

REGISTER_NATIVE_OPERATOR_FUNCTOR(
    aten::expand_as,
    aten_expand_as,
    [](Node* n) -> SROperator {
      if (!n->matches(torch::schema(
              "aten::expand_as(Tensor(a) self, Tensor other) -> Tensor(a)"))) {
        LogAndDumpSchema(n);
        return nullptr;
      }
      return [](ProcessedNode* p_node) {
        const auto& self = p_node->Input(0).toTensor();
        const auto& other = p_node->Input(1).toTensor();
        p_node->Output(0) = self.expand(other.sizes());
      };
    });

REGISTER_NATIVE_OPERATOR_FUNCTOR(
    prim::isinstance,
    prim_isinstance,
    [](Node* n) -> SROperator {
      if (!n->matches(
              torch::schema("prim::isinstance(Any to_check) -> bool"))) {
        LogAndDumpSchema(n);
        return nullptr;
      }
      return [](ProcessedNode* p_node) {
        auto input_type = p_node->Input(0).type();

        auto* node = p_node->node();
        const std::vector<TypePtr>& candidates = node->tys(attr::types);
        for (const auto& candidate_type : candidates) {
          if (input_type->isSubtypeOf(*candidate_type)) {
            p_node->Output(0) = true;
            return;
          }
        }

        p_node->Output(0) = false;
      };
    });

REGISTER_NATIVE_OPERATOR_FUNCTOR(
    prim::TypeCheck,
    prim_TypeCheck,
    [](Node* n) -> SROperator {
      if (!sr_schema_check_kind(n, prim::TypeCheck)) {
        return nullptr;
      }
      return [](ProcessedNode* p_node) {
        auto* node = p_node->node();
        const size_t num_inputs = node->inputs().size();
        TORCH_INTERNAL_ASSERT(
            num_inputs && num_inputs + 1 == node->outputs().size());

        const auto& expected_types = node->tys(attr::types);

        for (size_t i = 0; i < num_inputs; i++) {
          p_node->Output(i) = p_node->Input(i);
        }

        for (size_t i = 0; i < num_inputs; i++) {
          auto& input_tensor = p_node->Input(i).toTensor();
          auto* expected_type = expected_types[i]->castRaw<TensorType>();
          if (input_tensor.defined() &&
              !expected_type->matchTensor(input_tensor)) {
            p_node->Output(num_inputs) = false;
            return;
          }
        }

        p_node->Output(num_inputs) = true;
      };
    });

// See [Borrowed IValue Outputs]
REGISTER_NATIVE_OPERATOR_FUNCTOR(
    static_runtime::VarTupleUnpack,
    static_runtime_VarTupleUnpack,
    [](Node* n) -> SROperator {
      if (!sr_schema_check(n, "static_runtime::VarTupleUnpack(...) -> ...")) {
        return nullptr;
      }
      return [](ProcessedNode* pnode) {
        size_t output_idx = 0;
        for (const auto idx : c10::irange(pnode->num_inputs())) {
          const auto& tuple = pnode->Input(idx);
          for (auto& elem : tuple.toTupleRef().elements()) {
            pnode->Output(output_idx) = createBorrowedIValue(elem);
            ++output_idx;
          }
        }
      };
    });

REGISTER_NATIVE_OPERATOR_FUNCTOR(
    aten::view,
    aten_view,
    [](Node* n) -> SROperator {
      if (!n->matches(torch::schema(
              "aten::view(Tensor(a) self, int[] size) -> (Tensor(a))"))) {
        LogAndDumpSchema(n);
        return nullptr;
      }
      return [](ProcessedNode* p_node) {
        const auto& input = p_node->Input(0).toTensor();
        const auto size = p_node->Input(1).toIntList();
        p_node->Output(0) = at::native::view(input, size.vec());
      };
    });

REGISTER_NATIVE_OPERATOR_FUNCTOR(
    aten::size,
    aten_size,
    [](Node* n) -> SROperator {
      if (n->matches(
              torch::schema("aten::size(Tensor self, int dim) -> int"))) {
        return [](ProcessedNode* p_node) {
          const auto& input = p_node->Input(0).toTensor();
          auto dim = p_node->Input(1).toInt();
          const auto ndim = input.dim();

          if (dim < 0 || dim >= ndim) {
            dim = c10::maybe_wrap_dim(dim, ndim);
          }
          p_node->Output(0) = input.sizes()[dim];
        };
      }
      if (n->matches(torch::schema("aten::size(Tensor self) -> int[]"))) {
        return [](ProcessedNode* p_node) {
          const auto& input = p_node->Input(0).toTensor();
          p_node->Output(0) = input.sizes();
        };
      }
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_NATIVE_OPERATOR_FUNCTOR(
    aten::squeeze,
    aten_squeeze,
    [](Node* n) -> SROperator {
      if (!n->matches(torch::schema(
              "aten::squeeze.dim(Tensor(a) self, int dim) -> Tensor(a)"))) {
        LogAndDumpSchema(n);
        return nullptr;
      }

      return [](ProcessedNode* p_node) {
        const auto& self = p_node->Input(0).toTensor();
        const auto dim = p_node->Input(1).toInt();
        p_node->Output(0) = at::native::squeeze(self, dim);
      };
    });

REGISTER_NATIVE_OPERATOR_FUNCTOR(aten::split, aten_split, [](Node* n) -> SROperator {
  if (n->matches(torch::schema(
          "aten::split(Tensor(a -> *) self, int split_size, int dim=0) -> Tensor(a)[]"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      const auto split_size = p_node->Input(1).toInt();
      const auto dim = p_node->Input(2).toInt();
      p_node->Output(0) = at::native::split(self, split_size, dim);
    };
  }

  if (n->matches(torch::schema(
          "aten::split(Tensor(a -> *) self, int[] split_sizes, int dim=0) -> (Tensor[])"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      const auto& split_sizes = p_node->Input(1).toIntList();
      const auto dim = p_node->Input(2).toInt();
      p_node->Output(0) =
          at::native::split_with_sizes(self, split_sizes.vec(), dim);
    };
  }

  LogAndDumpSchema(n);
  return nullptr;
});

REGISTER_NATIVE_OPERATOR_FUNCTOR(
    aten::split_with_sizes,
    aten_split_with_sizes,
    [](Node* n) -> SROperator {
      if (!n->matches(torch::schema(
              "aten::split_with_sizes(Tensor(a -> *) self, int[] split_sizes, int dim=0) -> Tensor(a)[]")) &&
          !n->matches(torch::schema(
              "aten::split_with_sizes(Tensor(a -> *) self, int[] split_sizes, int dim=0) -> (Tensor[])"))) {
        LogAndDumpSchema(n);
        return nullptr;
      }
      return [](ProcessedNode* p_node) {
        const auto& self = p_node->Input(0).toTensor();
        const auto& split_sizes = p_node->Input(1).toIntList();
        const auto dim = p_node->Input(2).toInt();
        p_node->Output(0) =
            at::native::split_with_sizes(self, split_sizes.vec(), dim);
      };
    });

REGISTER_NATIVE_OPERATOR_FUNCTOR(
    static_runtime::select_tensor,
    aten_select_tensor,
    [](Node* n) -> SROperator {
      if (!sr_schema_check(
              n,
              "static_runtime::select_tensor(Tensor(a) a, Tensor(b) b, bool use_b) -> Tensor(a|b)")) {
        return nullptr;
      }
      return [](ProcessedNode* p_node) {
        const auto did_copy = p_node->Input(2).toBool();
        DCHECK(p_node->Input(0).isTensor());
        DCHECK(!did_copy || p_node->Input(1).isTensor());
        const IValue& assignFrom =
            did_copy ? p_node->Input(1) : p_node->Input(0);
        // Create an IValue that borrows the input Tensor in order to
        // save a refcount increment here and decrement in
        // MemoryPlanner::deallocate. MemoryPlanner knows about this
        // and will safely clean it up by using the corresponding
        // destroyBorrow method.
        TORCH_DCHECK_NE(&assignFrom, &p_node->Output(0));
        // MemoryPlanner should have cleaned this up!
        DCHECK(p_node->Output(0).isNone());
        p_node->Output(0) =
            IValue(c10::MaybeOwnedTraits<at::TensorBase>::createBorrow(
                assignFrom.toTensor()));
      };
    });

REGISTER_NATIVE_OPERATOR_FUNCTOR(
    aten::mul,
    aten_mul,
    [](Node* n) -> SROperator {
      if (!n->matches(
              torch::schema("aten::mul.left_t(t[] l, int n) -> (t[])"))) {
        LogAndDumpSchema(n);
        return nullptr;
      }
      return [](ProcessedNode* pnode) {
        const auto& list = pnode->Input(0).toList();
        const auto n = pnode->Input(1).toInt();

        auto list_type = list.elementType();
        auto ret = c10::impl::GenericList(list_type);
        ret.reserve(list.size() * n);
        for (const auto i : c10::irange(n)) {
          (void)i;
          for (const auto& ival : list) {
            ret.push_back(ival);
          }
        }
        pnode->Output(0) = ret;
      };
    });

REGISTER_NATIVE_OPERATOR_FUNCTOR(
    aten::sub,
    aten_sub,
    [](Node* n) -> SROperator {
      if (!n->matches(torch::schema("aten::sub.int(int a, int b) -> (int)"))) {
        LogAndDumpSchema(n);
        return nullptr;
      }
      return [](ProcessedNode* pnode) {
        const auto a = pnode->Input(0).toInt();
        const auto b = pnode->Input(1).toInt();
        pnode->Output(0) = a - b;
      };
    });

REGISTER_NATIVE_OPERATOR_FUNCTOR(
    aten::add,
    aten_add,
    [](Node* n) -> SROperator {
      if (n->matches(torch::schema("aten::add.t(t[] a, t[] b) -> (t[])"))) {
        return [](ProcessedNode* pnode) {
          const auto& a = pnode->Input(0).toList();
          const auto& b = pnode->Input(1).toList();
          auto ret = a.copy();
          ret.append(b);
          pnode->Output(0) = ret;
        };
      }

      if (n->matches(torch::schema("aten::add.int(int a, int b) -> (int)"))) {
        return [](ProcessedNode* pnode) {
          const auto a = pnode->Input(0).toInt();
          const auto b = pnode->Input(1).toInt();
          pnode->Output(0) = a + b;
        };
      }

      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_NATIVE_OPERATOR_FUNCTOR(aten::tensor_split, aten_tensor_split, [](Node* n) -> SROperator {
  if (n->matches(torch::schema(
          "aten::tensor_split.indices(Tensor(a -> *) self, int[] indices, int dim=0) -> Tensor(a)[]"))) {
    return [](ProcessedNode* pnode) {
      const auto& a = pnode->Input(0).toTensor();
      const auto& b = pnode->Input(1).toIntVector();
      const auto c = pnode->Input(2).toInt();
      pnode->Output(0) = at::native::tensor_split(a, b, c);
    };
  }

  if (n->matches(torch::schema(
          "aten::tensor_split.sections(Tensor(a -> *) self, int sections, int dim=0) -> Tensor(a)[]"))) {
    return [](ProcessedNode* pnode) {
      const auto& a = pnode->Input(0).toTensor();
      const auto b = pnode->Input(1).toSymInt();
      const auto c = pnode->Input(2).toInt();
      pnode->Output(0) = at::native::tensor_split_sections_symint(a, b, c);
    };
  }

  if (n->matches(torch::schema(
          "aten::tensor_split.tensor_indices_or_sections(Tensor(a -> *) self, Tensor tensor_indices_or_sections, int dim=0) -> Tensor(a)[]"))) {
    return [](ProcessedNode* pnode) {
      const auto& a = pnode->Input(0).toTensor();
      const auto& b = pnode->Input(1).toTensor();
      const auto c = pnode->Input(2).toInt();
      pnode->Output(0) = at::native::tensor_split(a, b, c);
    };
  }
  LogAndDumpSchema(n);
  return nullptr;
});

REGISTER_NATIVE_OPERATOR_FUNCTOR(
    aten::Int,
    aten_Int,
    [](Node* n) -> SROperator {
      if (!n->matches(torch::schema("aten::Int(Tensor a) -> int"))) {
        LogAndDumpSchema(n);
        return nullptr;
      }
      return [](ProcessedNode* pnode) {
        const auto& input = pnode->Input(0).toTensor();
        pnode->Output(0) = at::native::item(input).toInt();
      };
    });

// See [Create owned refs for special values]
REGISTER_NATIVE_OPERATOR_FUNCTOR(
    static_runtime::create_owned_ref,
    static_runtime_create_owned_ref,
    [](Node* n) -> SROperator {
      if (!sr_schema_check(n, "static_runtime::create_owned_ref(...) -> ...")) {
        return nullptr;
      }
      return
          [](ProcessedNode* p_node) { p_node->Output(0) = p_node->Input(0); };
    });

namespace {
bool outputsEmpty(const Block* block) {
  return block->outputs().size() == 1 && block->outputs().at(0)->mustBeNone();
}

bool blockEmpty(const Block* block) {
  return block->nodes().begin() == block->nodes().end();
}

enum class BlockRunPlan : int8_t {
  kRunOnlyTrueBlock,
  kRunOnlyFalseBlock,
  kRunBothBlocks,
  kRunNeitherBlock,
};
} // namespace

REGISTER_NATIVE_OPERATOR_FUNCTOR(
    prim::If,
    prim_If,
    [](Node* node) -> SROperator {
      if (!sr_schema_check_kind(node, prim::If)) {
        return nullptr;
      }
      TORCH_DCHECK_EQ(node->blocks().size(), 2);
      const Block* true_block = node->blocks().at(0);
      const Block* false_block = node->blocks().at(1);

      const bool true_block_returns_empty = outputsEmpty(true_block);
      const bool false_block_returns_empty = outputsEmpty(false_block);

      BlockRunPlan block_run_plan = BlockRunPlan::kRunNeitherBlock;

      if (true_block_returns_empty && false_block_returns_empty) {
        const bool false_block_is_empty = blockEmpty(false_block);
        const bool true_block_is_empty = blockEmpty(true_block);

        if (false_block_is_empty && !true_block_is_empty) {
          block_run_plan = BlockRunPlan::kRunOnlyTrueBlock;
        } else if (!false_block_is_empty && true_block_is_empty) {
          block_run_plan = BlockRunPlan::kRunOnlyFalseBlock;
        } else if (false_block_is_empty && true_block_is_empty) {
          block_run_plan = BlockRunPlan::kRunNeitherBlock;
        } else {
          block_run_plan = BlockRunPlan::kRunBothBlocks;
        }
      } else {
        block_run_plan = BlockRunPlan::kRunBothBlocks;
      }

      switch (block_run_plan) {
        case BlockRunPlan::kRunBothBlocks:
          return [](ProcessedNode* p_node) {
            auto condition = p_node->Input(0).toBool();
            auto* metadata = p_node->metadata();
            DCHECK(metadata);
            auto& block_runners = metadata->block_runners();
            TORCH_DCHECK_EQ(block_runners.size(), 2);
            auto& runner = block_runners[!condition];

            auto output = runner({});
            // If we are returning a tuple, we are either returning
            // multiple unpacked values or all of the values wrapped
            // in a single tuple. The second condition handles the
            // the latter case.
            if (!output.isTuple() || p_node->num_outputs() == 1) {
              p_node->Output(0) = std::move(output);
              return;
            }
            auto& elems = output.toTupleRef().elements();
            TORCH_DCHECK_EQ(elems.size(), p_node->num_outputs());
            for (const auto i : c10::irange(elems.size())) {
              p_node->Output(i) = elems[i];
            }
          };
        case BlockRunPlan::kRunOnlyTrueBlock:
          return [](ProcessedNode* p_node) {
            auto condition = p_node->Input(0).toBool();
            auto* metadata = p_node->metadata();
            DCHECK(metadata);
            auto& block_runners = metadata->block_runners();
            TORCH_DCHECK_EQ(block_runners.size(), 2);
            if (condition) {
              auto output = block_runners.front()({});
              DCHECK(output.isNone());
            }
          };
        case BlockRunPlan::kRunOnlyFalseBlock:
          return [](ProcessedNode* p_node) {
            auto condition = p_node->Input(0).toBool();
            auto* metadata = p_node->metadata();
            DCHECK(metadata);
            auto& block_runners = metadata->block_runners();
            TORCH_DCHECK_EQ(block_runners.size(), 2);
            if (!condition) {
              auto output = block_runners.back()({});
              DCHECK(output.isNone());
            }
          };
        case BlockRunPlan::kRunNeitherBlock:
          return [](ProcessedNode*) {};
      }
      return [](ProcessedNode*) {};
    });

namespace {

std::vector<IValue> collectLoopSubBlockInputs(const ProcessedNode& p_node) {
  const auto num_inputs = p_node.num_inputs();
  TORCH_DCHECK_GE(num_inputs, 2);
  // The first two inputs to the loop node are the max trip count
  // and initial condition. We don't collect them here, since those
  // are not inputs for the sub-block.
  const auto num_args = num_inputs - 2;

  std::vector<IValue> result;
  result.reserve(num_args + 1);
  // First argument to the loop sub-block is always the loop counter, initially
  // zero.
  result.emplace_back(0);

  for (const auto i : c10::irange(num_args)) {
    result.push_back(p_node.Input(2 + i));
  }

  return result;
}

} // namespace

namespace {
/*
  ForkedSubgraphSRLauncher is responsible for the execution of
  forked subgraph on new instance of static runtime. Once the
  execution is completed, future is marked as complete to
  indicate aten::wait() to proceed
*/
class TORCH_API ForkedSubgraphSRLauncher {
 public:
  ForkedSubgraphSRLauncher(
      std::shared_ptr<StaticModule> smodule,
      std::vector<IValue> args,
      c10::intrusive_ptr<Future> future,
      TaskLauncher launcher)
      : smodule_(std::move(smodule)),
        args_(std::move(args)),
        future_(std::move(future)),
        launcher_(std::move(launcher)) {}

  void operator()() {
    try {
      StaticRuntime runtime(*smodule_);
      auto future_subgraph = runtime.runAsync(args_, {}, launcher_);
      future_subgraph->waitAndThrow();
      future_->markCompleted(future_subgraph->value());
    } catch (const std::exception& e) {
      future_->setErrorIfNeeded(
          std::make_exception_ptr(c10::ivalue::Future::FutureError(e.what())));
    }
  }

 private:
  std::shared_ptr<StaticModule> smodule_;
  std::vector<IValue> args_;
  c10::intrusive_ptr<Future> future_;
  torch::jit::TaskLauncher launcher_;
};

/*
  helper function to create a future on return type
  of the graph outputs. This function is utilized by
  prim::fork and aten::wait operations for async
  execution of subgraphs
*/
c10::intrusive_ptr<Future> createFutureTypeFromGraphOutput(
    std::shared_ptr<torch::jit::Graph> graph) {
  TypePtr return_type_;
  if (graph->outputs().size() == 1) {
    return_type_ = graph->outputs().at(0)->type();
  } else {
    return_type_ = TupleType::create(
        fmap(graph->outputs(), [](const Value* v) { return v->type(); }));
  }
  c10::intrusive_ptr<Future> future = c10::make_intrusive<Future>(return_type_);
  return future;
}
} // namespace

/*
  prim::fork forks the execution of a subgraph. It returns a future on which
  the corresponding aten::wait op waits until future is marked complete
  Current implementation creates a instance of StaticModule uses it to
  create StaticRuntime instances on the fly during runtime to handle the
  execution of forked subgraph. Async execution is handled by
  aten::ParallelThreadPoolNative threadpool.
*/
REGISTER_NATIVE_OPERATOR_FUNCTOR(
    prim::fork,
    prim_Fork,
    [](Node* node) -> SROperator {
      if (!sr_schema_check_kind(node, prim::fork)) {
        return nullptr;
      }
      auto forkedGraph = node->g(attr::Subgraph);
      Inline(*forkedGraph);
      auto sr_metadata = node->ival(getStaticRuntimeMetadataSymbol())
                             .toCustomClass<StaticRuntimeMetadata>();
      auto smodule =
          std::make_shared<StaticModule>(forkedGraph, sr_metadata->get_opts());

      return [forkedGraph = std::move(forkedGraph),
              smodule = std::move(smodule)](ProcessedNode* p_node) {
        std::vector<IValue> args;
        args.reserve(p_node->num_inputs());
        for (const auto i : c10::irange(p_node->num_inputs())) {
          args.push_back(p_node->Input(i));
        }

        c10::intrusive_ptr<Future> future =
            createFutureTypeFromGraphOutput(forkedGraph);
        p_node->Output(0) = future;

        auto* metadata = p_node->metadata();
        DCHECK(metadata);
        auto* launcher = metadata->launcher();
        DCHECK(launcher);
        ForkedSubgraphSRLauncher runtime_launcher(
            smodule, args, future, *launcher);
        (*launcher)(std::move(runtime_launcher));
      };
    });
/*
  aten::wait waits on the future (present in corresponding fork)
  to be executed. Once the execution is complete, the future is marked
  completed and wait execution continues.
*/
REGISTER_NATIVE_OPERATOR_FUNCTOR(
    aten::wait,
    aten_Wait,
    [](Node* n) -> SROperator {
      if (!sr_schema_check(n, "aten::wait(Future(t) self) -> t")) {
        return nullptr;
      }
      return [](ProcessedNode* p_node) {
        TORCH_INTERNAL_ASSERT(p_node->Input(0).isFuture());
        auto future = p_node->Input(0).toFuture();

        // blocking call: waiting for the future to be completed
        future->waitAndThrow();

        TORCH_INTERNAL_ASSERT(future->completed());
        TORCH_INTERNAL_ASSERT(!future->hasError());
        TORCH_INTERNAL_ASSERT(future->hasValue());

        if (!future->value().isTuple()) {
          p_node->Output(0) = future->value();
          return;
        }
        auto& elems = future->value().toTupleRef().elements();
        TORCH_DCHECK_EQ(elems.size(), p_node->num_outputs());
        for (const auto i : c10::irange(elems.size())) {
          p_node->Output(i) = elems[i];
        }
      };
    });

REGISTER_NATIVE_OPERATOR_FUNCTOR(
    prim::Loop,
    prim_Loop,
    [](Node* n) -> SROperator {
      if (!sr_schema_check_kind(n, prim::Loop)) {
        return nullptr;
      }
      return [](ProcessedNode* p_node) {
        const auto max_trip_count = p_node->Input(0).toInt();
        auto condition = p_node->Input(1).toBool();

        auto* metadata = p_node->metadata();
        DCHECK(metadata);
        auto& block_runners = metadata->block_runners();
        TORCH_DCHECK_EQ(block_runners.size(), 1);
        auto& runner = block_runners[0];

        auto args = collectLoopSubBlockInputs(*p_node);
        int64_t loop_count = 0;

        while (condition && loop_count < max_trip_count) {
          auto output = runner(args);

          if (output.isTuple()) {
            auto& elems = output.toTupleRef().elements();
            DCHECK(elems.size() == args.size());
            for (const auto i : c10::irange(1, args.size())) {
              args[i] = elems[i];
            }
            condition = elems[0].toBool();
          } else {
            condition = output.toBool();
          }
          args[0] = ++loop_count;
        }

        const auto num_outputs = p_node->num_outputs();
        TORCH_DCHECK_EQ(args.size(), num_outputs + 1);
        for (const auto i : c10::irange(num_outputs)) {
          p_node->Output(i) = std::move(args[i + 1]);
        }
      };
    });

REGISTER_NATIVE_OPERATOR_FUNCTOR(
    prim::CreateObject,
    prim_CreateObject,
    [](Node* node) -> SROperator {
      if (!sr_schema_check_kind(node, prim::CreateObject)) {
        return nullptr;
      }
      auto class_type = node->output()->type()->expect<ClassType>();
      return [class_type = std::move(class_type)](ProcessedNode* pnode) {
        pnode->Output(0) = c10::ivalue::Object::create(
            c10::StrongTypePtr(class_type->compilation_unit(), class_type),
            class_type->numAttributes());
      };
    });

REGISTER_NATIVE_OPERATOR_FUNCTOR(
    prim::TupleIndex,
    prim_TupleIndex,
    [](Node* n) -> SROperator {
      if (!sr_schema_check_kind(n, prim::TupleIndex)) {
        return nullptr;
      }
      return [](ProcessedNode* pnode) {
        const auto& elems = pnode->Input(0).toTupleRef().elements();
        using c10::ssize;
        const auto num_elems = ssize(elems);
        const auto idx = pnode->Input(1).toInt();
        const auto norm_idx = normalizeIndex(idx, num_elems);
        if (norm_idx < 0 || norm_idx >= num_elems) {
          // Use std::runtime_error instead of c10::Error to be consistent with
          // JIT
          throw std::out_of_range("Tuple index out of range");
        }
        pnode->Output(0) = elems[norm_idx];
      };
    });

REGISTER_NATIVE_OPERATOR_FUNCTOR(
    prim::RaiseException,
    prim_RaiseException,
    [](Node* n) -> SROperator {
      if (!sr_schema_check_kind(n, prim::RaiseException)) {
        return nullptr;
      }
      return [](ProcessedNode* pnode) {
        const auto& message = pnode->Input(0).toStringRef();
        throw std::runtime_error(message);
      };
    });

REGISTER_NATIVE_OPERATOR_FUNCTOR(
    prim::Uninitialized,
    prim_Uninitialized,
    [](Node* n) -> SROperator {
      if (!sr_schema_check_kind(n, prim::Uninitialized)) {
        return nullptr;
      }
      return [](ProcessedNode* pnode) {
        pnode->Output(0) = IValue::uninitialized();
      };
    });

REGISTER_NATIVE_OPERATOR_FUNCTOR(
    aten::format,
    aten_format,
    [](Node* n) -> SROperator {
      if (!sr_schema_check(n, "aten::format(str self, ...) -> str")) {
        return nullptr;
      }
      TORCH_CHECK(!n->inputs().empty());
      return [](ProcessedNode* pnode) {
        const auto num_inputs = pnode->num_inputs();
        auto stack = boxInputs(*pnode);
        format(stack, num_inputs);
        TORCH_DCHECK_EQ(stack.size(), 1);
        pnode->Output(0) = std::move(stack[0]);
      };
    });

REGISTER_NATIVE_OPERATOR_FUNCTOR(
    prim::device,
    prim_device,
    [](Node* n) -> SROperator {
      if (!sr_schema_check(n, "prim::device(Tensor a) -> Device")) {
        return nullptr;
      }
      return [](ProcessedNode* pnode) {
        const auto& input = pnode->Input(0).toTensor();
        pnode->Output(0) = input.device();
      };
    });

REGISTER_NATIVE_OPERATOR_FUNCTOR(
    prim::dtype,
    prim_dtype,
    [](Node* n) -> SROperator {
      if (!sr_schema_check_kind(n, prim::dtype)) {
        return nullptr;
      }
      return [](ProcessedNode* pnode) {
        const auto& input = pnode->Input(0).toTensor();
        pnode->Output(0) = static_cast<int64_t>(input.scalar_type());
      };
    });

REGISTER_NATIVE_OPERATOR_FUNCTOR(
    aten::dim,
    aten_dim,
    [](Node* n) -> SROperator {
      if (!sr_schema_check(n, "aten::dim(Tensor self) -> int")) {
        return nullptr;
      }
      return [](ProcessedNode* pnode) {
        const auto& input = pnode->Input(0).toTensor();
        pnode->Output(0) = input.dim();
      };
    });

REGISTER_NATIVE_OPERATOR_FUNCTOR(
    aten::__not__,
    aten_not,
    [](Node* n) -> SROperator {
      if (!sr_schema_check(n, "aten::__not__(bool self) -> bool")) {
        return nullptr;
      }
      return [](ProcessedNode* pnode) {
        auto input = pnode->Input(0).toBool();
        pnode->Output(0) = !input;
      };
    });

REGISTER_NATIVE_OPERATOR_FUNCTOR(
    aten::Bool,
    aten_Bool,
    [](Node* n) -> SROperator {
      if (n->matches(torch::schema("aten::Bool.Tensor(Tensor a) -> bool"))) {
        return [](ProcessedNode* pnode) {
          const auto& input = pnode->Input(0).toTensor();
          pnode->Output(0) = at::native::is_nonzero(input);
        };
      }
      if (n->matches(torch::schema("aten::Bool.int(int a) -> bool"))) {
        return [](ProcessedNode* pnode) {
          const auto input = pnode->Input(0).toInt();
          pnode->Output(0) = static_cast<bool>(input);
        };
      }
      if (n->matches(torch::schema("aten::Bool.float(float a) -> bool"))) {
        return [](ProcessedNode* pnode) {
          const auto input = pnode->Input(0).toDouble();
          pnode->Output(0) = static_cast<bool>(input);
        };
      }
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_NATIVE_OPERATOR_FUNCTOR(
    prim::is_cuda,
    prim_is_cuda,
    [](Node* n) -> SROperator {
      if (!sr_schema_check(n, "prim::is_cuda(Tensor a) -> bool")) {
        return nullptr;
      }
      return [](ProcessedNode* pnode) {
        const auto& input = pnode->Input(0).toTensor();
        pnode->Output(0) = input.is_cuda();
      };
    });

REGISTER_NATIVE_OPERATOR_FUNCTOR(
    prim::tolist,
    prim_tolist,
    [](Node* n) -> SROperator {
      if (!sr_schema_check_kind(n, prim::tolist)) {
        return nullptr;
      }
      return [](ProcessedNode* pnode) {
        const auto& input = pnode->Input(0).toTensor();
        const auto dim = pnode->Input(1).toInt();
        const auto elem_type = pnode->Input(2).toInt();
        std::vector<IValue> stack{input, dim, elem_type};
        toList(stack);
        TORCH_DCHECK_EQ(stack.size(), 1);
        pnode->Output(0) = std::move(stack[0]);
      };
    });

// See [Borrowed IValue Outputs]
REGISTER_NATIVE_OPERATOR_FUNCTOR(
    prim::IfThenElse,
    prim_IfThenElse,
    [](Node* n) -> SROperator {
      if (!sr_schema_check_kind(n, prim::IfThenElse)) {
        return nullptr;
      }
      return [](ProcessedNode* pnode) {
        const auto condition = pnode->Input(0).toBool();
        pnode->Output(0) = condition ? createBorrowedIValue(pnode->Input(1))
                                     : createBorrowedIValue(pnode->Input(2));
      };
    });

REGISTER_NATIVE_OPERATOR_FUNCTOR(
    aten::len,
    aten_len,
    [](Node* n) -> SROperator {
      if (n->matches(torch::schema("aten::len.t(t[] a) -> int")) ||
          n->matches(torch::schema("aten::len.any(Any[] a) -> int"))) {
        return [](ProcessedNode* pnode) {
          const auto list = pnode->Input(0).toListRef();
          const int64_t size = list.size();
          pnode->Output(0) = size;
        };
      }
      if (n->matches(torch::schema("aten::len.Tensor(Tensor t) -> int"))) {
        return [](ProcessedNode* pnode) {
          const auto& t = pnode->Input(0).toTensor();
          TORCH_CHECK(t.dim() > 0);
          pnode->Output(0) = t.sizes()[0];
        };
      }
      if (n->matches(torch::schema("aten::len.str(str s) -> int"))) {
        return [](ProcessedNode* pnode) {
          const auto& string = pnode->Input(0).toStringRef();
          pnode->Output(0) = static_cast<int64_t>(string.size());
        };
      }
      if (n->matches(
              torch::schema("aten::len.Dict_str(Dict(str, t) self) -> int")) ||
          n->matches(
              torch::schema("aten::len.Dict_int(Dict(int, t) self) -> int")) ||
          n->matches(torch::schema(
              "aten::len.Dict_bool(Dict(bool, t) self) -> int")) ||
          n->matches(torch::schema(
              "aten::len.Dict_float(Dict(float, t) self) -> int")) ||
          n->matches(torch::schema(
              "aten::len.Dict_complex(Dict(complex, t) self) -> int")) ||
          n->matches(torch::schema(
              "aten::len.Dict_Tensor(Dict(Tensor, t) self) -> int"))) {
        return [](ProcessedNode* pnode) {
          const auto& dict = pnode->Input(0).toGenericDict();
          pnode->Output(0) = static_cast<int64_t>(dict.size());
        };
      }
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_NATIVE_OPERATOR_FUNCTOR(
    aten::IntImplicit,
    aten_IntImplicit,
    [](Node* n) -> SROperator {
      if (!n->matches(torch::schema("aten::IntImplicit(Tensor a) -> int"))) {
        LogAndDumpSchema(n);
        return nullptr;
      }
      return [](ProcessedNode* pnode) {
        const auto& tensor = pnode->Input(0).toTensor();
        // JIT does a check for requires_grad, but we skip it here since SR is
        // inference only
        if (!tensor.sizes().empty()) {
          throw std::runtime_error(
              "Cannot convert a tensor of dimension > 0 to scalar");
        }
        if (!isIntegralType(tensor.scalar_type(), /*includeBool=*/false)) {
          std::stringstream ss;
          ss << "Cannot input a tensor of type " << tensor.scalar_type()
             << " as an integral argument";
          throw std::runtime_error(ss.str());
        }
        pnode->Output(0) = at::native::item(tensor).toInt();
      };
    });

REGISTER_NATIVE_OPERATOR_FUNCTOR(
    aten::select,
    aten_select,
    [](Node* n) -> SROperator {
      if (!n->matches(torch::schema(
              "aten::select(Tensor(a) self, int dim, int index) -> Tensor(a)"))) {
        LogAndDumpSchema(n);
        return nullptr;
      }
      return [](ProcessedNode* pnode) {
        const auto& self = pnode->Input(0).toTensor();
        const auto dim = pnode->Input(1).toInt();
        const auto index = pnode->Input(2).toInt();
        pnode->Output(0) = at::native::select(self, dim, index);
      };
    });

REGISTER_NATIVE_OPERATOR_FUNCTOR(
    aten::reshape_as,
    aten_reshape_as,
    [](Node* n) -> SROperator {
      if (!n->matches(torch::schema(
              "aten::reshape_as(Tensor(a) self, Tensor other) -> Tensor(a)"))) {
        LogAndDumpSchema(n);
        return nullptr;
      }
      return [](ProcessedNode* pnode) {
        const auto& self = pnode->Input(0).toTensor();
        const auto& other = pnode->Input(1).toTensor();
        pnode->Output(0) = at::native::reshape(self, other.sizes());
      };
    });

} // namespace jit
} // namespace torch
