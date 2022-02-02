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
#include <torch/csrc/jit/runtime/register_ops_utils.h>
#include <torch/csrc/jit/runtime/vararg_functions.h>

namespace {
constexpr auto createBorrowedIValue =
    c10::MaybeOwnedTraits<c10::IValue>::createBorrow;
} // namespace
namespace torch {
namespace jit {

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

REGISTER_NATIVE_OPERATOR_FUNCTOR(
    prim::TupleConstruct,
    prim_TupleConstruct,
    [](Node* n) -> SROperator {
      return [](ProcessedNode* p_node) {
        // prepare inputs
        std::vector<IValue> stack;
        const size_t size = p_node->num_inputs();
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

REGISTER_NATIVE_OPERATOR_FUNCTOR(
    prim::TupleUnpack,
    prim_TupleUnpack,
    [](Node* n) -> SROperator {
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
      return [](ProcessedNode* p_node) {
        // prepare inputs
        std::vector<IValue> stack;
        const size_t size = p_node->num_inputs();
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

// See [Borrowed IValue Outputs]
REGISTER_NATIVE_OPERATOR_FUNCTOR(
    static_runtime::dict_unpack,
    static_runtime_dict_unpack,
    [](Node*) -> SROperator {
      return [](ProcessedNode* p_node) {
        DCHECK(p_node->num_inputs() - 1 == p_node->outputs().size());
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

REGISTER_NATIVE_OPERATOR_FUNCTOR(
    aten::__getitem__,
    aten_getitem,
    [](Node* n) -> SROperator {
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
      return [](ProcessedNode* p_node) {
        // prepare inputs
        std::vector<IValue> stack;
        const size_t size = p_node->num_inputs();
        stack.reserve(size);
        for (const auto i : c10::irange(size)) {
          stack.emplace_back(p_node->Input(i));
        }
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
      return [](ProcessedNode* p_node) {
        // prepare inputs
        std::vector<IValue> stack;
        const size_t size = p_node->num_inputs();
        stack.reserve(size);
        for (const auto i : c10::irange(size)) {
          stack.emplace_back(p_node->Input(i));
        }
        // run op
        size_t num_outputs = p_node->outputs().size();
        listUnpack(stack, num_outputs);
        // put output back
        DCHECK_EQ(stack.size(), num_outputs);
        for (const auto i : c10::irange(num_outputs)) {
          p_node->Output(i) = std::move(stack[i]);
        }
      };
    });

REGISTER_NATIVE_OPERATOR_FUNCTOR(
    aten::append,
    aten_append,
    [](Node* n) -> SROperator {
      return [](ProcessedNode* p_node) {
        auto list = p_node->Input(0).toList();
        list.push_back(p_node->Input(1));
      };
    });

REGISTER_NATIVE_OPERATOR_FUNCTOR(
    prim::GetAttr,
    prim_GetAttr,
    [](Node* n) -> SROperator {
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
    const auto in2_i = p_node->Input(2).toInt();
    const auto in3_i = p_node->Input(3).toInt();
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
    [](Node*) -> SROperator {
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
      if (!n->matches(
              torch::schema("aten::size(Tensor self, int dim) -> int"))) {
        LogAndDumpSchema(n);
        return nullptr;
      }
      return [](ProcessedNode* p_node) {
        const auto& input = p_node->Input(0).toTensor();
        auto dim = p_node->Input(1).toInt();
        const auto ndim = input.dim();

        if (dim < 0 || dim >= ndim) {
          dim = c10::maybe_wrap_dim(dim, ndim);
        }
        p_node->Output(0) = input.sizes()[dim];
      };
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
      TORCH_CHECK(n->inputs().size() == 3);
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
        DCHECK_NE(&assignFrom, &p_node->Output(0));
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
      if (!n->matches(torch::schema("aten::add.t(t[] a, t[] b) -> (t[])"))) {
        LogAndDumpSchema(n);
        return nullptr;
      }
      return [](ProcessedNode* pnode) {
        const auto& a = pnode->Input(0).toList();
        const auto& b = pnode->Input(1).toList();
        auto ret = a.copy();
        ret.append(b);
        pnode->Output(0) = ret;
      };
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

} // namespace jit
} // namespace torch
