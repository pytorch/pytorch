// @lint-ignore-every CLANGTIDY HOWTOEVEN
// AUTO-GENERATED FROM: torchgen/static_runtime/gen_static_runtime_ops.py
#include <torch/csrc/jit/runtime/static/ops.h>

#include <ATen/CPUFunctions.h>
#include <ATen/InferSize.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Parallel.h>
#include <ATen/ScalarOps.h>
#include <ATen/TensorUtils.h>
#include <ATen/cpu/vec/functional.h>
#include <ATen/cpu/vec/vec.h>
#include <ATen/native/EmbeddingBag.h>
#include <ATen/native/Fill.h>
#include <ATen/native/IndexingUtils.h>
#include <ATen/native/NonSymbolicBC.h>
#include <ATen/native/Resize.h>
#include <ATen/native/SharedReduceOps.h>
#include <ATen/native/TensorAdvancedIndexing.h>
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
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/runtime/static/impl.h>
#include <torch/csrc/jit/runtime/static/te_wrapper.h>
#include <torch/csrc/jit/runtime/vararg_functions.h>
#include <torch/csrc/jit/tensorexpr/ir.h>
#include <torch/csrc/jit/tensorexpr/ir_simplifier.h>
#include <torch/csrc/jit/tensorexpr/llvm_codegen.h>
#include <torch/csrc/jit/tensorexpr/loopnest.h>

namespace torch {
namespace jit {

REGISTER_OPERATOR_FUNCTOR(
    aten::absolute,
    aten_absolute,
    [](Node* n) -> SROperator {
      if (n->matches(torch::schema("aten::absolute(Tensor self) -> Tensor"))) {
        return [](ProcessedNode* p_node) {
          const auto& self = p_node->Input(0).toTensor();
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::native::absolute(self);
            return;
          }
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          at::native::absolute_out(self, out);
        };
      }
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_OPERATOR_FUNCTOR(aten::angle, aten_angle, [](Node* n) -> SROperator {
  if (n->matches(torch::schema("aten::angle(Tensor self) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::native::angle(self);
        return;
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::native::angle_out(self, out);
    };
  }
  LogAndDumpSchema(n);
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(aten::sgn, aten_sgn, [](Node* n) -> SROperator {
  if (n->matches(torch::schema("aten::sgn(Tensor self) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::sgn(self);
        return;
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::cpu::sgn_out(out, self);
    };
  }
  LogAndDumpSchema(n);
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(aten::acos, aten_acos, [](Node* n) -> SROperator {
  if (n->matches(torch::schema("aten::acos(Tensor self) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::acos(self);
        return;
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::cpu::acos_out(out, self);
    };
  }
  LogAndDumpSchema(n);
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(aten::arccos, aten_arccos, [](Node* n) -> SROperator {
  if (n->matches(torch::schema("aten::arccos(Tensor self) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::native::arccos(self);
        return;
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::native::arccos_out(self, out);
    };
  }
  LogAndDumpSchema(n);
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(aten::_add_relu, aten__add_relu, [](Node* n) -> SROperator {
  if (n->matches(torch::schema(
          "aten::_add_relu.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      const auto& other = p_node->Input(1).toTensor();
      const auto alpha = p_node->Input(2).toScalar();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::native::add_relu(self, other, alpha);
        return;
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::native::add_relu_out(self, other, alpha, out);
    };
  }
  LogAndDumpSchema(n);
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(aten::addmv, aten_addmv, [](Node* n) -> SROperator {
  if (n->matches(torch::schema(
          "aten::addmv(Tensor self, Tensor mat, Tensor vec, *, Scalar beta=1, Scalar alpha=1) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      const auto& mat = p_node->Input(1).toTensor();
      const auto& vec = p_node->Input(2).toTensor();
      const auto beta = p_node->Input(3).toScalar();
      const auto alpha = p_node->Input(4).toScalar();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::addmv(self, mat, vec, beta, alpha);
        return;
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::cpu::addmv_out(out, self, mat, vec, beta, alpha);
    };
  }
  LogAndDumpSchema(n);
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(aten::addr, aten_addr, [](Node* n) -> SROperator {
  if (n->matches(torch::schema(
          "aten::addr(Tensor self, Tensor vec1, Tensor vec2, *, Scalar beta=1, Scalar alpha=1) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      const auto& vec1 = p_node->Input(1).toTensor();
      const auto& vec2 = p_node->Input(2).toTensor();
      const auto beta = p_node->Input(3).toScalar();
      const auto alpha = p_node->Input(4).toScalar();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::native::addr(self, vec1, vec2, beta, alpha);
        return;
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::native::addr_out(self, vec1, vec2, beta, alpha, out);
    };
  }
  LogAndDumpSchema(n);
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(
    aten::_test_functorch_fallback,
    aten__test_functorch_fallback,
    [](Node* n) -> SROperator {
      if (n->matches(torch::schema(
              "aten::_test_functorch_fallback(Tensor self, Tensor other) -> Tensor"))) {
        return [](ProcessedNode* p_node) {
          const auto& self = p_node->Input(0).toTensor();
          const auto& other = p_node->Input(1).toTensor();
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) =
                at::native::_test_functorch_fallback(self, other);
            return;
          }
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          at::native::_test_functorch_fallback_out(self, other, out);
        };
      }
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_OPERATOR_FUNCTOR(aten::argmax, aten_argmax, [](Node* n) -> SROperator {
  if (n->matches(torch::schema(
          "aten::argmax(Tensor self, int? dim=None, bool keepdim=False) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      const auto dim = p_node->Input(1).toOptional<int64_t>();
      const auto keepdim = p_node->Input(2).toBool();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::argmax(self, dim, keepdim);
        return;
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::cpu::argmax_out(out, self, dim, keepdim);
    };
  }
  LogAndDumpSchema(n);
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(aten::acosh, aten_acosh, [](Node* n) -> SROperator {
  if (n->matches(torch::schema("aten::acosh(Tensor self) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::acosh(self);
        return;
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::cpu::acosh_out(out, self);
    };
  }
  LogAndDumpSchema(n);
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(aten::asinh, aten_asinh, [](Node* n) -> SROperator {
  if (n->matches(torch::schema("aten::asinh(Tensor self) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::asinh(self);
        return;
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::cpu::asinh_out(out, self);
    };
  }
  LogAndDumpSchema(n);
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(
    aten::arcsinh,
    aten_arcsinh,
    [](Node* n) -> SROperator {
      if (n->matches(torch::schema("aten::arcsinh(Tensor self) -> Tensor"))) {
        return [](ProcessedNode* p_node) {
          const auto& self = p_node->Input(0).toTensor();
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::native::arcsinh(self);
            return;
          }
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          at::native::arcsinh_out(self, out);
        };
      }
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_OPERATOR_FUNCTOR(aten::atanh, aten_atanh, [](Node* n) -> SROperator {
  if (n->matches(torch::schema("aten::atanh(Tensor self) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::atanh(self);
        return;
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::cpu::atanh_out(out, self);
    };
  }
  LogAndDumpSchema(n);
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(
    aten::arctanh,
    aten_arctanh,
    [](Node* n) -> SROperator {
      if (n->matches(torch::schema("aten::arctanh(Tensor self) -> Tensor"))) {
        return [](ProcessedNode* p_node) {
          const auto& self = p_node->Input(0).toTensor();
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::native::arctanh(self);
            return;
          }
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          at::native::arctanh_out(self, out);
        };
      }
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_OPERATOR_FUNCTOR(aten::asin, aten_asin, [](Node* n) -> SROperator {
  if (n->matches(torch::schema("aten::asin(Tensor self) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::asin(self);
        return;
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::cpu::asin_out(out, self);
    };
  }
  LogAndDumpSchema(n);
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(aten::arcsin, aten_arcsin, [](Node* n) -> SROperator {
  if (n->matches(torch::schema("aten::arcsin(Tensor self) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::native::arcsin(self);
        return;
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::native::arcsin_out(self, out);
    };
  }
  LogAndDumpSchema(n);
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(aten::atan, aten_atan, [](Node* n) -> SROperator {
  if (n->matches(torch::schema("aten::atan(Tensor self) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::atan(self);
        return;
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::cpu::atan_out(out, self);
    };
  }
  LogAndDumpSchema(n);
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(aten::arctan, aten_arctan, [](Node* n) -> SROperator {
  if (n->matches(torch::schema("aten::arctan(Tensor self) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::native::arctan(self);
        return;
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::native::arctan_out(self, out);
    };
  }
  LogAndDumpSchema(n);
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(aten::baddbmm, aten_baddbmm, [](Node* n) -> SROperator {
  if (n->matches(torch::schema(
          "aten::baddbmm(Tensor self, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      const auto& batch1 = p_node->Input(1).toTensor();
      const auto& batch2 = p_node->Input(2).toTensor();
      const auto beta = p_node->Input(3).toScalar();
      const auto alpha = p_node->Input(4).toScalar();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::baddbmm(self, batch1, batch2, beta, alpha);
        return;
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::cpu::baddbmm_out(out, self, batch1, batch2, beta, alpha);
    };
  }
  LogAndDumpSchema(n);
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(
    aten::bitwise_not,
    aten_bitwise_not,
    [](Node* n) -> SROperator {
      if (n->matches(
              torch::schema("aten::bitwise_not(Tensor self) -> Tensor"))) {
        return [](ProcessedNode* p_node) {
          const auto& self = p_node->Input(0).toTensor();
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::cpu::bitwise_not(self);
            return;
          }
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          at::cpu::bitwise_not_out(out, self);
        };
      }
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_OPERATOR_FUNCTOR(
    aten::copysign,
    aten_copysign,
    [](Node* n) -> SROperator {
      if (n->matches(torch::schema(
              "aten::copysign.Tensor(Tensor self, Tensor other) -> Tensor"))) {
        return [](ProcessedNode* p_node) {
          const auto& self = p_node->Input(0).toTensor();
          const auto& other = p_node->Input(1).toTensor();
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::cpu::copysign(self, other);
            return;
          }
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          at::cpu::copysign_out(out, self, other);
        };
      }
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_OPERATOR_FUNCTOR(
    aten::logical_not,
    aten_logical_not,
    [](Node* n) -> SROperator {
      if (n->matches(
              torch::schema("aten::logical_not(Tensor self) -> Tensor"))) {
        return [](ProcessedNode* p_node) {
          const auto& self = p_node->Input(0).toTensor();
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::native::logical_not(self);
            return;
          }
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          at::native::logical_not_out(self, out);
        };
      }
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_OPERATOR_FUNCTOR(
    aten::logical_xor,
    aten_logical_xor,
    [](Node* n) -> SROperator {
      if (n->matches(torch::schema(
              "aten::logical_xor(Tensor self, Tensor other) -> Tensor"))) {
        return [](ProcessedNode* p_node) {
          const auto& self = p_node->Input(0).toTensor();
          const auto& other = p_node->Input(1).toTensor();
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::native::logical_xor(self, other);
            return;
          }
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          at::native::logical_xor_out(self, other, out);
        };
      }
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_OPERATOR_FUNCTOR(
    aten::logical_and,
    aten_logical_and,
    [](Node* n) -> SROperator {
      if (n->matches(torch::schema(
              "aten::logical_and(Tensor self, Tensor other) -> Tensor"))) {
        return [](ProcessedNode* p_node) {
          const auto& self = p_node->Input(0).toTensor();
          const auto& other = p_node->Input(1).toTensor();
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::native::logical_and(self, other);
            return;
          }
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          at::native::logical_and_out(self, other, out);
        };
      }
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_OPERATOR_FUNCTOR(
    aten::logical_or,
    aten_logical_or,
    [](Node* n) -> SROperator {
      if (n->matches(torch::schema(
              "aten::logical_or(Tensor self, Tensor other) -> Tensor"))) {
        return [](ProcessedNode* p_node) {
          const auto& self = p_node->Input(0).toTensor();
          const auto& other = p_node->Input(1).toTensor();
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::native::logical_or(self, other);
            return;
          }
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          at::native::logical_or_out(self, other, out);
        };
      }
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_OPERATOR_FUNCTOR(aten::ceil, aten_ceil, [](Node* n) -> SROperator {
  if (n->matches(torch::schema("aten::ceil(Tensor self) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::ceil(self);
        return;
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::cpu::ceil_out(out, self);
    };
  }
  LogAndDumpSchema(n);
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(
    aten::clamp_max,
    aten_clamp_max,
    [](Node* n) -> SROperator {
      if (n->matches(torch::schema(
              "aten::clamp_max(Tensor self, Scalar max) -> Tensor"))) {
        return [](ProcessedNode* p_node) {
          const auto& self = p_node->Input(0).toTensor();
          const auto max = p_node->Input(1).toScalar();
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::cpu::clamp_max(self, max);
            return;
          }
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          at::cpu::clamp_max_out(out, self, max);
        };
      }

      if (n->matches(torch::schema(
              "aten::clamp_max.Tensor(Tensor self, Tensor max) -> Tensor"))) {
        return [](ProcessedNode* p_node) {
          const auto& self = p_node->Input(0).toTensor();
          const auto& max = p_node->Input(1).toTensor();
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::cpu::clamp_max(self, max);
            return;
          }
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          at::cpu::clamp_max_out(out, self, max);
        };
      }
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_OPERATOR_FUNCTOR(aten::clip, aten_clip, [](Node* n) -> SROperator {
  if (n->matches(torch::schema(
          "aten::clip(Tensor self, Scalar? min=None, Scalar? max=None) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      const auto min = p_node->Input(1).toOptional<at::Scalar>();
      const auto max = p_node->Input(2).toOptional<at::Scalar>();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::native::clip(self, min, max);
        return;
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::native::clip_out(self, min, max, out);
    };
  }
  LogAndDumpSchema(n);
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(
    aten::complex,
    aten_complex,
    [](Node* n) -> SROperator {
      if (n->matches(torch::schema(
              "aten::complex(Tensor real, Tensor imag) -> Tensor"))) {
        return [](ProcessedNode* p_node) {
          const auto& real = p_node->Input(0).toTensor();
          const auto& imag = p_node->Input(1).toTensor();
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::native::complex(real, imag);
            return;
          }
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          at::native::complex_out(real, imag, out);
        };
      }
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_OPERATOR_FUNCTOR(aten::polar, aten_polar, [](Node* n) -> SROperator {
  if (n->matches(
          torch::schema("aten::polar(Tensor abs, Tensor angle) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& abs = p_node->Input(0).toTensor();
      const auto& angle = p_node->Input(1).toTensor();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::native::polar(abs, angle);
        return;
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::native::polar_out(abs, angle, out);
    };
  }
  LogAndDumpSchema(n);
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(aten::cos, aten_cos, [](Node* n) -> SROperator {
  if (n->matches(torch::schema("aten::cos(Tensor self) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::cos(self);
        return;
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::cpu::cos_out(out, self);
    };
  }
  LogAndDumpSchema(n);
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(aten::cosh, aten_cosh, [](Node* n) -> SROperator {
  if (n->matches(torch::schema("aten::cosh(Tensor self) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::cosh(self);
        return;
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::cpu::cosh_out(out, self);
    };
  }
  LogAndDumpSchema(n);
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(aten::cumprod, aten_cumprod, [](Node* n) -> SROperator {
  if (n->matches(torch::schema(
          "aten::cumprod(Tensor self, int dim, *, ScalarType? dtype=None) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      const auto dim = p_node->Input(1).toInt();
      const auto dtype = p_node->Input(2).toOptional<at::ScalarType>();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::cumprod(self, dim, dtype);
        return;
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::cpu::cumprod_out(out, self, dim, dtype);
    };
  }
  LogAndDumpSchema(n);
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(aten::diff, aten_diff, [](Node* n) -> SROperator {
  if (n->matches(torch::schema(
          "aten::diff(Tensor self, int n=1, int dim=-1, Tensor? prepend=None, Tensor? append=None) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      const auto n = p_node->Input(1).toInt();
      const auto dim = p_node->Input(2).toInt();
      const auto prepend = p_node->Input(3).toOptional<at::Tensor>();
      const auto append = p_node->Input(4).toOptional<at::Tensor>();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::native::diff(self, n, dim, prepend, append);
        return;
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::native::diff_out(self, n, dim, prepend, append, out);
    };
  }
  LogAndDumpSchema(n);
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(aten::divide, aten_divide, [](Node* n) -> SROperator {
  if (n->matches(torch::schema(
          "aten::divide.Tensor(Tensor self, Tensor other) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      const auto& other = p_node->Input(1).toTensor();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::native::divide(self, other);
        return;
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::native::divide_out(self, other, out);
    };
  }
  LogAndDumpSchema(n);
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(
    aten::true_divide,
    aten_true_divide,
    [](Node* n) -> SROperator {
      if (n->matches(torch::schema(
              "aten::true_divide.Tensor(Tensor self, Tensor other) -> Tensor"))) {
        return [](ProcessedNode* p_node) {
          const auto& self = p_node->Input(0).toTensor();
          const auto& other = p_node->Input(1).toTensor();
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::native::true_divide(self, other);
            return;
          }
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          at::native::true_divide_out(self, other, out);
        };
      }
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_OPERATOR_FUNCTOR(aten::dot, aten_dot, [](Node* n) -> SROperator {
  if (n->matches(
          torch::schema("aten::dot(Tensor self, Tensor tensor) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      const auto& tensor = p_node->Input(1).toTensor();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::native::dot(self, tensor);
        return;
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::native::dot_out(self, tensor, out);
    };
  }
  LogAndDumpSchema(n);
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(aten::vdot, aten_vdot, [](Node* n) -> SROperator {
  if (n->matches(
          torch::schema("aten::vdot(Tensor self, Tensor other) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      const auto& other = p_node->Input(1).toTensor();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::native::vdot(self, other);
        return;
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::native::vdot_out(self, other, out);
    };
  }
  LogAndDumpSchema(n);
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(aten::erf, aten_erf, [](Node* n) -> SROperator {
  if (n->matches(torch::schema("aten::erf(Tensor self) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::erf(self);
        return;
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::cpu::erf_out(out, self);
    };
  }
  LogAndDumpSchema(n);
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(aten::erfc, aten_erfc, [](Node* n) -> SROperator {
  if (n->matches(torch::schema("aten::erfc(Tensor self) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::erfc(self);
        return;
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::cpu::erfc_out(out, self);
    };
  }
  LogAndDumpSchema(n);
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(aten::exp, aten_exp, [](Node* n) -> SROperator {
  if (n->matches(torch::schema("aten::exp(Tensor self) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::exp(self);
        return;
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::cpu::exp_out(out, self);
    };
  }
  LogAndDumpSchema(n);
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(aten::exp2, aten_exp2, [](Node* n) -> SROperator {
  if (n->matches(torch::schema("aten::exp2(Tensor self) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::exp2(self);
        return;
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::cpu::exp2_out(out, self);
    };
  }
  LogAndDumpSchema(n);
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(aten::expm1, aten_expm1, [](Node* n) -> SROperator {
  if (n->matches(torch::schema("aten::expm1(Tensor self) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::expm1(self);
        return;
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::cpu::expm1_out(out, self);
    };
  }
  LogAndDumpSchema(n);
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(aten::floor, aten_floor, [](Node* n) -> SROperator {
  if (n->matches(torch::schema("aten::floor(Tensor self) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::floor(self);
        return;
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::cpu::floor_out(out, self);
    };
  }
  LogAndDumpSchema(n);
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(aten::frac, aten_frac, [](Node* n) -> SROperator {
  if (n->matches(torch::schema("aten::frac(Tensor self) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::frac(self);
        return;
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::cpu::frac_out(out, self);
    };
  }
  LogAndDumpSchema(n);
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(aten::gcd, aten_gcd, [](Node* n) -> SROperator {
  if (n->matches(
          torch::schema("aten::gcd(Tensor self, Tensor other) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      const auto& other = p_node->Input(1).toTensor();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::gcd(self, other);
        return;
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::cpu::gcd_out(out, self, other);
    };
  }
  LogAndDumpSchema(n);
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(aten::lcm, aten_lcm, [](Node* n) -> SROperator {
  if (n->matches(
          torch::schema("aten::lcm(Tensor self, Tensor other) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      const auto& other = p_node->Input(1).toTensor();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::lcm(self, other);
        return;
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::cpu::lcm_out(out, self, other);
    };
  }
  LogAndDumpSchema(n);
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(aten::index_copy, aten_index_copy, [](Node* n) -> SROperator {
  if (n->matches(torch::schema(
          "aten::index_copy(Tensor self, int dim, Tensor index, Tensor source) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      const auto dim = p_node->Input(1).toInt();
      const auto& index = p_node->Input(2).toTensor();
      const auto& source = p_node->Input(3).toTensor();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::index_copy(self, dim, index, source);
        return;
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::cpu::index_copy_out(out, self, dim, index, source);
    };
  }
  LogAndDumpSchema(n);
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(aten::isin, aten_isin, [](Node* n) -> SROperator {
  if (n->matches(torch::schema(
          "aten::isin.Tensor_Tensor(Tensor elements, Tensor test_elements, *, bool assume_unique=False, bool invert=False) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& elements = p_node->Input(0).toTensor();
      const auto& test_elements = p_node->Input(1).toTensor();
      const auto assume_unique = p_node->Input(2).toBool();
      const auto invert = p_node->Input(3).toBool();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) =
            at::cpu::isin(elements, test_elements, assume_unique, invert);
        return;
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::cpu::isin_out(out, elements, test_elements, assume_unique, invert);
    };
  }

  if (n->matches(torch::schema(
          "aten::isin.Tensor_Scalar(Tensor elements, Scalar test_element, *, bool assume_unique=False, bool invert=False) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& elements = p_node->Input(0).toTensor();
      const auto test_element = p_node->Input(1).toScalar();
      const auto assume_unique = p_node->Input(2).toBool();
      const auto invert = p_node->Input(3).toBool();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) =
            at::cpu::isin(elements, test_element, assume_unique, invert);
        return;
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::cpu::isin_out(out, elements, test_element, assume_unique, invert);
    };
  }

  if (n->matches(torch::schema(
          "aten::isin.Scalar_Tensor(Scalar element, Tensor test_elements, *, bool assume_unique=False, bool invert=False) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto element = p_node->Input(0).toScalar();
      const auto& test_elements = p_node->Input(1).toTensor();
      const auto assume_unique = p_node->Input(2).toBool();
      const auto invert = p_node->Input(3).toBool();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) =
            at::cpu::isin(element, test_elements, assume_unique, invert);
        return;
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::cpu::isin_out(out, element, test_elements, assume_unique, invert);
    };
  }
  LogAndDumpSchema(n);
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(aten::kron, aten_kron, [](Node* n) -> SROperator {
  if (n->matches(
          torch::schema("aten::kron(Tensor self, Tensor other) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      const auto& other = p_node->Input(1).toTensor();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::native::kron(self, other);
        return;
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::native::kron_out(self, other, out);
    };
  }
  LogAndDumpSchema(n);
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(aten::ldexp, aten_ldexp, [](Node* n) -> SROperator {
  if (n->matches(torch::schema(
          "aten::ldexp.Tensor(Tensor self, Tensor other) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      const auto& other = p_node->Input(1).toTensor();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::native::ldexp(self, other);
        return;
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::native::ldexp_out(self, other, out);
    };
  }
  LogAndDumpSchema(n);
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(aten::log10, aten_log10, [](Node* n) -> SROperator {
  if (n->matches(torch::schema("aten::log10(Tensor self) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::log10(self);
        return;
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::cpu::log10_out(out, self);
    };
  }
  LogAndDumpSchema(n);
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(aten::log1p, aten_log1p, [](Node* n) -> SROperator {
  if (n->matches(torch::schema("aten::log1p(Tensor self) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::log1p(self);
        return;
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::cpu::log1p_out(out, self);
    };
  }
  LogAndDumpSchema(n);
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(aten::log2, aten_log2, [](Node* n) -> SROperator {
  if (n->matches(torch::schema("aten::log2(Tensor self) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::log2(self);
        return;
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::cpu::log2_out(out, self);
    };
  }
  LogAndDumpSchema(n);
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(
    aten::logaddexp,
    aten_logaddexp,
    [](Node* n) -> SROperator {
      if (n->matches(torch::schema(
              "aten::logaddexp(Tensor self, Tensor other) -> Tensor"))) {
        return [](ProcessedNode* p_node) {
          const auto& self = p_node->Input(0).toTensor();
          const auto& other = p_node->Input(1).toTensor();
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::cpu::logaddexp(self, other);
            return;
          }
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          at::cpu::logaddexp_out(out, self, other);
        };
      }
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_OPERATOR_FUNCTOR(
    aten::logaddexp2,
    aten_logaddexp2,
    [](Node* n) -> SROperator {
      if (n->matches(torch::schema(
              "aten::logaddexp2(Tensor self, Tensor other) -> Tensor"))) {
        return [](ProcessedNode* p_node) {
          const auto& self = p_node->Input(0).toTensor();
          const auto& other = p_node->Input(1).toTensor();
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::cpu::logaddexp2(self, other);
            return;
          }
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          at::cpu::logaddexp2_out(out, self, other);
        };
      }
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_OPERATOR_FUNCTOR(aten::xlogy, aten_xlogy, [](Node* n) -> SROperator {
  if (n->matches(torch::schema(
          "aten::xlogy.Tensor(Tensor self, Tensor other) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      const auto& other = p_node->Input(1).toTensor();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::xlogy(self, other);
        return;
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::cpu::xlogy_out(out, self, other);
    };
  }
  LogAndDumpSchema(n);
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(
    aten::_log_softmax,
    aten__log_softmax,
    [](Node* n) -> SROperator {
      if (n->matches(torch::schema(
              "aten::_log_softmax(Tensor self, int dim, bool half_to_float) -> Tensor"))) {
        return [](ProcessedNode* p_node) {
          const auto& self = p_node->Input(0).toTensor();
          const auto dim = p_node->Input(1).toInt();
          const auto half_to_float = p_node->Input(2).toBool();
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::cpu::_log_softmax(self, dim, half_to_float);
            return;
          }
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          at::cpu::_log_softmax_out(out, self, dim, half_to_float);
        };
      }
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_OPERATOR_FUNCTOR(
    aten::_log_softmax_backward_data,
    aten__log_softmax_backward_data,
    [](Node* n) -> SROperator {
      if (n->matches(torch::schema(
              "aten::_log_softmax_backward_data(Tensor grad_output, Tensor output, int dim, ScalarType input_dtype) -> Tensor"))) {
        return [](ProcessedNode* p_node) {
          const auto& grad_output = p_node->Input(0).toTensor();
          const auto& output = p_node->Input(1).toTensor();
          const auto dim = p_node->Input(2).toInt();
          const auto input_dtype = p_node->Input(3).toScalarType();
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::cpu::_log_softmax_backward_data(
                grad_output, output, dim, input_dtype);
            return;
          }
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          at::cpu::_log_softmax_backward_data_out(
              out, grad_output, output, dim, input_dtype);
        };
      }
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_OPERATOR_FUNCTOR(
    aten::_logcumsumexp,
    aten__logcumsumexp,
    [](Node* n) -> SROperator {
      if (n->matches(torch::schema(
              "aten::_logcumsumexp(Tensor self, int dim) -> Tensor"))) {
        return [](ProcessedNode* p_node) {
          const auto& self = p_node->Input(0).toTensor();
          const auto dim = p_node->Input(1).toInt();
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::native::_logcumsumexp_cpu(self, dim);
            return;
          }
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          at::native::_logcumsumexp_out_cpu(self, dim, out);
        };
      }
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_OPERATOR_FUNCTOR(
    aten::logcumsumexp,
    aten_logcumsumexp,
    [](Node* n) -> SROperator {
      if (n->matches(torch::schema(
              "aten::logcumsumexp(Tensor self, int dim) -> Tensor"))) {
        return [](ProcessedNode* p_node) {
          const auto& self = p_node->Input(0).toTensor();
          const auto dim = p_node->Input(1).toInt();
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::native::logcumsumexp(self, dim);
            return;
          }
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          at::native::logcumsumexp_out(self, dim, out);
        };
      }
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_OPERATOR_FUNCTOR(
    aten::matrix_power,
    aten_matrix_power,
    [](Node* n) -> SROperator {
      if (n->matches(torch::schema(
              "aten::matrix_power(Tensor self, int n) -> Tensor"))) {
        return [](ProcessedNode* p_node) {
          const auto& self = p_node->Input(0).toTensor();
          const auto n = p_node->Input(1).toInt();
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::native::matrix_power(self, n);
            return;
          }
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          at::native::matrix_power_out(self, n, out);
        };
      }
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_OPERATOR_FUNCTOR(aten::mm, aten_mm, [](Node* n) -> SROperator {
  if (n->matches(
          torch::schema("aten::mm(Tensor self, Tensor mat2) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      const auto& mat2 = p_node->Input(1).toTensor();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::mm(self, mat2);
        return;
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::cpu::mm_out(out, self, mat2);
    };
  }
  LogAndDumpSchema(n);
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(
    aten::multiply,
    aten_multiply,
    [](Node* n) -> SROperator {
      if (n->matches(torch::schema(
              "aten::multiply.Tensor(Tensor self, Tensor other) -> Tensor"))) {
        return [](ProcessedNode* p_node) {
          const auto& self = p_node->Input(0).toTensor();
          const auto& other = p_node->Input(1).toTensor();
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::native::multiply(self, other);
            return;
          }
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          at::native::multiply_out(self, other, out);
        };
      }
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_OPERATOR_FUNCTOR(aten::mv, aten_mv, [](Node* n) -> SROperator {
  if (n->matches(
          torch::schema("aten::mv(Tensor self, Tensor vec) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      const auto& vec = p_node->Input(1).toTensor();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::native::mv(self, vec);
        return;
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::native::mv_out(self, vec, out);
    };
  }
  LogAndDumpSchema(n);
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(
    aten::mvlgamma,
    aten_mvlgamma,
    [](Node* n) -> SROperator {
      if (n->matches(
              torch::schema("aten::mvlgamma(Tensor self, int p) -> Tensor"))) {
        return [](ProcessedNode* p_node) {
          const auto& self = p_node->Input(0).toTensor();
          const auto p = p_node->Input(1).toInt();
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::native::mvlgamma(self, p);
            return;
          }
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          at::native::mvlgamma_out(self, p, out);
        };
      }
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_OPERATOR_FUNCTOR(
    aten::rad2deg,
    aten_rad2deg,
    [](Node* n) -> SROperator {
      if (n->matches(torch::schema("aten::rad2deg(Tensor self) -> Tensor"))) {
        return [](ProcessedNode* p_node) {
          const auto& self = p_node->Input(0).toTensor();
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::native::rad2deg(self);
            return;
          }
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          at::native::rad2deg_out(self, out);
        };
      }
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_OPERATOR_FUNCTOR(
    aten::deg2rad,
    aten_deg2rad,
    [](Node* n) -> SROperator {
      if (n->matches(torch::schema("aten::deg2rad(Tensor self) -> Tensor"))) {
        return [](ProcessedNode* p_node) {
          const auto& self = p_node->Input(0).toTensor();
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::native::deg2rad(self);
            return;
          }
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          at::native::deg2rad_out(self, out);
        };
      }
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_OPERATOR_FUNCTOR(
    aten::reciprocal,
    aten_reciprocal,
    [](Node* n) -> SROperator {
      if (n->matches(
              torch::schema("aten::reciprocal(Tensor self) -> Tensor"))) {
        return [](ProcessedNode* p_node) {
          const auto& self = p_node->Input(0).toTensor();
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::cpu::reciprocal(self);
            return;
          }
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          at::cpu::reciprocal_out(out, self);
        };
      }
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_OPERATOR_FUNCTOR(aten::neg, aten_neg, [](Node* n) -> SROperator {
  if (n->matches(torch::schema("aten::neg(Tensor self) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::neg(self);
        return;
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::cpu::neg_out(out, self);
    };
  }
  LogAndDumpSchema(n);
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(
    aten::negative,
    aten_negative,
    [](Node* n) -> SROperator {
      if (n->matches(torch::schema("aten::negative(Tensor self) -> Tensor"))) {
        return [](ProcessedNode* p_node) {
          const auto& self = p_node->Input(0).toTensor();
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::native::negative(self);
            return;
          }
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          at::native::negative_out(self, out);
        };
      }
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_OPERATOR_FUNCTOR(aten::round, aten_round, [](Node* n) -> SROperator {
  if (n->matches(torch::schema("aten::round(Tensor self) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::round(self);
        return;
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::cpu::round_out(out, self);
    };
  }

  if (n->matches(torch::schema(
          "aten::round.decimals(Tensor self, *, int decimals) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      const auto decimals = p_node->Input(1).toInt();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::round(self, decimals);
        return;
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::cpu::round_out(out, self, decimals);
    };
  }
  LogAndDumpSchema(n);
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(aten::gelu, aten_gelu, [](Node* n) -> SROperator {
  if (n->matches(torch::schema(
          "aten::gelu(Tensor self, *, str approximate='none') -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      const auto approximate = p_node->Input(1).toStringView();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::gelu(self, approximate);
        return;
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::cpu::gelu_out(out, self, approximate);
    };
  }
  LogAndDumpSchema(n);
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(
    aten::gelu_backward,
    aten_gelu_backward,
    [](Node* n) -> SROperator {
      if (n->matches(torch::schema(
              "aten::gelu_backward(Tensor grad_output, Tensor self, *, str approximate='none') -> Tensor"))) {
        return [](ProcessedNode* p_node) {
          const auto& grad_output = p_node->Input(0).toTensor();
          const auto& self = p_node->Input(1).toTensor();
          const auto approximate = p_node->Input(2).toStringView();
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) =
                at::cpu::gelu_backward(grad_output, self, approximate);
            return;
          }
          auto& grad_input = p_node->Output(0).toTensor();
          fastResizeToZero(grad_input);
          at::cpu::gelu_backward_out(
              grad_input, grad_output, self, approximate);
        };
      }
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_OPERATOR_FUNCTOR(
    aten::hardshrink,
    aten_hardshrink,
    [](Node* n) -> SROperator {
      if (n->matches(torch::schema(
              "aten::hardshrink(Tensor self, Scalar lambd=0.5) -> Tensor"))) {
        return [](ProcessedNode* p_node) {
          const auto& self = p_node->Input(0).toTensor();
          const auto lambd = p_node->Input(1).toScalar();
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::cpu::hardshrink(self, lambd);
            return;
          }
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          at::cpu::hardshrink_out(out, self, lambd);
        };
      }
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_OPERATOR_FUNCTOR(
    aten::hardshrink_backward,
    aten_hardshrink_backward,
    [](Node* n) -> SROperator {
      if (n->matches(torch::schema(
              "aten::hardshrink_backward(Tensor grad_out, Tensor self, Scalar lambd) -> Tensor"))) {
        return [](ProcessedNode* p_node) {
          const auto& grad_out = p_node->Input(0).toTensor();
          const auto& self = p_node->Input(1).toTensor();
          const auto lambd = p_node->Input(2).toScalar();
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) =
                at::cpu::hardshrink_backward(grad_out, self, lambd);
            return;
          }
          auto& grad_input = p_node->Output(0).toTensor();
          fastResizeToZero(grad_input);
          at::cpu::hardshrink_backward_out(grad_input, grad_out, self, lambd);
        };
      }
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_OPERATOR_FUNCTOR(aten::rsqrt, aten_rsqrt, [](Node* n) -> SROperator {
  if (n->matches(torch::schema("aten::rsqrt(Tensor self) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::rsqrt(self);
        return;
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::cpu::rsqrt_out(out, self);
    };
  }
  LogAndDumpSchema(n);
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(aten::silu, aten_silu, [](Node* n) -> SROperator {
  if (n->matches(torch::schema("aten::silu(Tensor self) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::silu(self);
        return;
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::cpu::silu_out(out, self);
    };
  }
  LogAndDumpSchema(n);
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(
    aten::silu_backward,
    aten_silu_backward,
    [](Node* n) -> SROperator {
      if (n->matches(torch::schema(
              "aten::silu_backward(Tensor grad_output, Tensor self) -> Tensor"))) {
        return [](ProcessedNode* p_node) {
          const auto& grad_output = p_node->Input(0).toTensor();
          const auto& self = p_node->Input(1).toTensor();
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::cpu::silu_backward(grad_output, self);
            return;
          }
          auto& grad_input = p_node->Output(0).toTensor();
          fastResizeToZero(grad_input);
          at::cpu::silu_backward_out(grad_input, grad_output, self);
        };
      }
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_OPERATOR_FUNCTOR(aten::mish, aten_mish, [](Node* n) -> SROperator {
  if (n->matches(torch::schema("aten::mish(Tensor self) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::mish(self);
        return;
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::cpu::mish_out(out, self);
    };
  }
  LogAndDumpSchema(n);
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(aten::sin, aten_sin, [](Node* n) -> SROperator {
  if (n->matches(torch::schema("aten::sin(Tensor self) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::sin(self);
        return;
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::cpu::sin_out(out, self);
    };
  }
  LogAndDumpSchema(n);
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(aten::sinc, aten_sinc, [](Node* n) -> SROperator {
  if (n->matches(torch::schema("aten::sinc(Tensor self) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::sinc(self);
        return;
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::cpu::sinc_out(out, self);
    };
  }
  LogAndDumpSchema(n);
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(aten::sinh, aten_sinh, [](Node* n) -> SROperator {
  if (n->matches(torch::schema("aten::sinh(Tensor self) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::sinh(self);
        return;
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::cpu::sinh_out(out, self);
    };
  }
  LogAndDumpSchema(n);
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(aten::_softmax, aten__softmax, [](Node* n) -> SROperator {
  if (n->matches(torch::schema(
          "aten::_softmax(Tensor self, int dim, bool half_to_float) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      const auto dim = p_node->Input(1).toInt();
      const auto half_to_float = p_node->Input(2).toBool();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::_softmax(self, dim, half_to_float);
        return;
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::cpu::_softmax_out(out, self, dim, half_to_float);
    };
  }
  LogAndDumpSchema(n);
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(
    aten::_softmax_backward_data,
    aten__softmax_backward_data,
    [](Node* n) -> SROperator {
      if (n->matches(torch::schema(
              "aten::_softmax_backward_data(Tensor grad_output, Tensor output, int dim, ScalarType input_dtype) -> Tensor"))) {
        return [](ProcessedNode* p_node) {
          const auto& grad_output = p_node->Input(0).toTensor();
          const auto& output = p_node->Input(1).toTensor();
          const auto dim = p_node->Input(2).toInt();
          const auto input_dtype = p_node->Input(3).toScalarType();
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::cpu::_softmax_backward_data(
                grad_output, output, dim, input_dtype);
            return;
          }
          auto& grad_input = p_node->Output(0).toTensor();
          fastResizeToZero(grad_input);
          at::cpu::_softmax_backward_data_out(
              grad_input, grad_output, output, dim, input_dtype);
        };
      }
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_OPERATOR_FUNCTOR(aten::sqrt, aten_sqrt, [](Node* n) -> SROperator {
  if (n->matches(torch::schema("aten::sqrt(Tensor self) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::sqrt(self);
        return;
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::cpu::sqrt_out(out, self);
    };
  }
  LogAndDumpSchema(n);
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(aten::square, aten_square, [](Node* n) -> SROperator {
  if (n->matches(torch::schema("aten::square(Tensor self) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::native::square(self);
        return;
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::native::square_out(self, out);
    };
  }
  LogAndDumpSchema(n);
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(aten::prod, aten_prod, [](Node* n) -> SROperator {
  if (n->matches(torch::schema(
          "aten::prod(Tensor self, *, ScalarType? dtype=None) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      const auto dtype = p_node->Input(1).toOptional<at::ScalarType>();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::native::prod(self, dtype);
        return;
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::native::prod_out(self, dtype, out);
    };
  }

  if (n->matches(torch::schema(
          "aten::prod.dim_int(Tensor self, int dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      const auto dim = p_node->Input(1).toInt();
      const auto keepdim = p_node->Input(2).toBool();
      const auto dtype = p_node->Input(3).toOptional<at::ScalarType>();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::prod(self, dim, keepdim, dtype);
        return;
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::cpu::prod_out(out, self, dim, keepdim, dtype);
    };
  }
  LogAndDumpSchema(n);
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(aten::tan, aten_tan, [](Node* n) -> SROperator {
  if (n->matches(torch::schema("aten::tan(Tensor self) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::tan(self);
        return;
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::cpu::tan_out(out, self);
    };
  }
  LogAndDumpSchema(n);
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(aten::threshold, aten_threshold, [](Node* n) -> SROperator {
  if (n->matches(torch::schema(
          "aten::threshold(Tensor self, Scalar threshold, Scalar value) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      const auto threshold = p_node->Input(1).toScalar();
      const auto value = p_node->Input(2).toScalar();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::threshold(self, threshold, value);
        return;
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::cpu::threshold_out(out, self, threshold, value);
    };
  }
  LogAndDumpSchema(n);
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(
    aten::threshold_backward,
    aten_threshold_backward,
    [](Node* n) -> SROperator {
      if (n->matches(torch::schema(
              "aten::threshold_backward(Tensor grad_output, Tensor self, Scalar threshold) -> Tensor"))) {
        return [](ProcessedNode* p_node) {
          const auto& grad_output = p_node->Input(0).toTensor();
          const auto& self = p_node->Input(1).toTensor();
          const auto threshold = p_node->Input(2).toScalar();
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) =
                at::cpu::threshold_backward(grad_output, self, threshold);
            return;
          }
          auto& grad_input = p_node->Output(0).toTensor();
          fastResizeToZero(grad_input);
          at::cpu::threshold_backward_out(
              grad_input, grad_output, self, threshold);
        };
      }
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_OPERATOR_FUNCTOR(aten::trunc, aten_trunc, [](Node* n) -> SROperator {
  if (n->matches(torch::schema("aten::trunc(Tensor self) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::trunc(self);
        return;
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::cpu::trunc_out(out, self);
    };
  }
  LogAndDumpSchema(n);
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(aten::fix, aten_fix, [](Node* n) -> SROperator {
  if (n->matches(torch::schema("aten::fix(Tensor self) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::native::fix(self);
        return;
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::native::fix_out(self, out);
    };
  }
  LogAndDumpSchema(n);
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(
    aten::nuclear_norm,
    aten_nuclear_norm,
    [](Node* n) -> SROperator {
      if (n->matches(torch::schema(
              "aten::nuclear_norm(Tensor self, bool keepdim=False) -> Tensor"))) {
        return [](ProcessedNode* p_node) {
          const auto& self = p_node->Input(0).toTensor();
          const auto keepdim = p_node->Input(1).toBool();
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::native::nuclear_norm(self, keepdim);
            return;
          }
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          at::native::nuclear_norm_out(self, keepdim, out);
        };
      }
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_OPERATOR_FUNCTOR(aten::subtract, aten_subtract, [](Node* n) -> SROperator {
  if (n->matches(torch::schema(
          "aten::subtract.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      const auto& other = p_node->Input(1).toTensor();
      const auto alpha = p_node->Input(2).toScalar();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::native::subtract(self, other, alpha);
        return;
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::native::subtract_out(self, other, alpha, out);
    };
  }
  LogAndDumpSchema(n);
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(
    aten::heaviside,
    aten_heaviside,
    [](Node* n) -> SROperator {
      if (n->matches(torch::schema(
              "aten::heaviside(Tensor self, Tensor values) -> Tensor"))) {
        return [](ProcessedNode* p_node) {
          const auto& self = p_node->Input(0).toTensor();
          const auto& values = p_node->Input(1).toTensor();
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::cpu::heaviside(self, values);
            return;
          }
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          at::cpu::heaviside_out(out, self, values);
        };
      }
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_OPERATOR_FUNCTOR(
    aten::_addmm_activation,
    aten__addmm_activation,
    [](Node* n) -> SROperator {
      if (n->matches(torch::schema(
              "aten::_addmm_activation(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1, bool use_gelu=False) -> Tensor"))) {
        return [](ProcessedNode* p_node) {
          const auto& self = p_node->Input(0).toTensor();
          const auto& mat1 = p_node->Input(1).toTensor();
          const auto& mat2 = p_node->Input(2).toTensor();
          const auto beta = p_node->Input(3).toScalar();
          const auto alpha = p_node->Input(4).toScalar();
          const auto use_gelu = p_node->Input(5).toBool();
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::cpu::_addmm_activation(
                self, mat1, mat2, beta, alpha, use_gelu);
            return;
          }
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          at::cpu::_addmm_activation_out(
              out, self, mat1, mat2, beta, alpha, use_gelu);
        };
      }
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_OPERATOR_FUNCTOR(aten::index_add, aten_index_add, [](Node* n) -> SROperator {
  if (n->matches(torch::schema(
          "aten::index_add(Tensor self, int dim, Tensor index, Tensor source, *, Scalar alpha=1) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      const auto dim = p_node->Input(1).toInt();
      const auto& index = p_node->Input(2).toTensor();
      const auto& source = p_node->Input(3).toTensor();
      const auto alpha = p_node->Input(4).toScalar();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::index_add(self, dim, index, source, alpha);
        return;
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::cpu::index_add_out(out, self, dim, index, source, alpha);
    };
  }
  LogAndDumpSchema(n);
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(aten::scatter, aten_scatter, [](Node* n) -> SROperator {
  if (n->matches(torch::schema(
          "aten::scatter.src(Tensor self, int dim, Tensor index, Tensor src) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      const auto dim = p_node->Input(1).toInt();
      const auto& index = p_node->Input(2).toTensor();
      const auto& src = p_node->Input(3).toTensor();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::scatter(self, dim, index, src);
        return;
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::cpu::scatter_out(out, self, dim, index, src);
    };
  }

  if (n->matches(torch::schema(
          "aten::scatter.value(Tensor self, int dim, Tensor index, Scalar value) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      const auto dim = p_node->Input(1).toInt();
      const auto& index = p_node->Input(2).toTensor();
      const auto value = p_node->Input(3).toScalar();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::scatter(self, dim, index, value);
        return;
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::cpu::scatter_out(out, self, dim, index, value);
    };
  }

  if (n->matches(torch::schema(
          "aten::scatter.reduce(Tensor self, int dim, Tensor index, Tensor src, *, str reduce) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      const auto dim = p_node->Input(1).toInt();
      const auto& index = p_node->Input(2).toTensor();
      const auto& src = p_node->Input(3).toTensor();
      const auto reduce = p_node->Input(4).toStringView();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::scatter(self, dim, index, src, reduce);
        return;
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::cpu::scatter_out(out, self, dim, index, src, reduce);
    };
  }

  if (n->matches(torch::schema(
          "aten::scatter.value_reduce(Tensor self, int dim, Tensor index, Scalar value, *, str reduce) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      const auto dim = p_node->Input(1).toInt();
      const auto& index = p_node->Input(2).toTensor();
      const auto value = p_node->Input(3).toScalar();
      const auto reduce = p_node->Input(4).toStringView();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::scatter(self, dim, index, value, reduce);
        return;
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::cpu::scatter_out(out, self, dim, index, value, reduce);
    };
  }
  LogAndDumpSchema(n);
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(aten::scatter_add, aten_scatter_add, [](Node* n) -> SROperator {
  if (n->matches(torch::schema(
          "aten::scatter_add(Tensor self, int dim, Tensor index, Tensor src) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      const auto dim = p_node->Input(1).toInt();
      const auto& index = p_node->Input(2).toTensor();
      const auto& src = p_node->Input(3).toTensor();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::scatter_add(self, dim, index, src);
        return;
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::cpu::scatter_add_out(out, self, dim, index, src);
    };
  }
  LogAndDumpSchema(n);
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(
    aten::scatter_reduce,
    aten_scatter_reduce,
    [](Node* n) -> SROperator {
      if (n->matches(torch::schema(
              "aten::scatter_reduce.two(Tensor self, int dim, Tensor index, Tensor src, str reduce, *, bool include_self=True) -> Tensor"))) {
        return [](ProcessedNode* p_node) {
          const auto& self = p_node->Input(0).toTensor();
          const auto dim = p_node->Input(1).toInt();
          const auto& index = p_node->Input(2).toTensor();
          const auto& src = p_node->Input(3).toTensor();
          const auto reduce = p_node->Input(4).toStringView();
          const auto include_self = p_node->Input(5).toBool();
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::cpu::scatter_reduce(
                self, dim, index, src, reduce, include_self);
            return;
          }
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          at::cpu::scatter_reduce_out(
              out, self, dim, index, src, reduce, include_self);
        };
      }
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_OPERATOR_FUNCTOR(aten::eq, aten_eq, [](Node* n) -> SROperator {
  if (n->matches(torch::schema(
          "aten::eq.Scalar(Tensor self, Scalar other) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      const auto other = p_node->Input(1).toScalar();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::eq(self, other);
        return;
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::cpu::eq_out(out, self, other);
    };
  }

  if (n->matches(torch::schema(
          "aten::eq.Tensor(Tensor self, Tensor other) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      const auto& other = p_node->Input(1).toTensor();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::eq(self, other);
        return;
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::cpu::eq_out(out, self, other);
    };
  }
  LogAndDumpSchema(n);
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(
    aten::bitwise_and,
    aten_bitwise_and,
    [](Node* n) -> SROperator {
      if (n->matches(torch::schema(
              "aten::bitwise_and.Tensor(Tensor self, Tensor other) -> Tensor"))) {
        return [](ProcessedNode* p_node) {
          const auto& self = p_node->Input(0).toTensor();
          const auto& other = p_node->Input(1).toTensor();
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::cpu::bitwise_and(self, other);
            return;
          }
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          at::cpu::bitwise_and_out(out, self, other);
        };
      }
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_OPERATOR_FUNCTOR(
    aten::bitwise_or,
    aten_bitwise_or,
    [](Node* n) -> SROperator {
      if (n->matches(torch::schema(
              "aten::bitwise_or.Tensor(Tensor self, Tensor other) -> Tensor"))) {
        return [](ProcessedNode* p_node) {
          const auto& self = p_node->Input(0).toTensor();
          const auto& other = p_node->Input(1).toTensor();
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::cpu::bitwise_or(self, other);
            return;
          }
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          at::cpu::bitwise_or_out(out, self, other);
        };
      }
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_OPERATOR_FUNCTOR(
    aten::bitwise_xor,
    aten_bitwise_xor,
    [](Node* n) -> SROperator {
      if (n->matches(torch::schema(
              "aten::bitwise_xor.Tensor(Tensor self, Tensor other) -> Tensor"))) {
        return [](ProcessedNode* p_node) {
          const auto& self = p_node->Input(0).toTensor();
          const auto& other = p_node->Input(1).toTensor();
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::cpu::bitwise_xor(self, other);
            return;
          }
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          at::cpu::bitwise_xor_out(out, self, other);
        };
      }
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_OPERATOR_FUNCTOR(
    aten::bitwise_left_shift,
    aten_bitwise_left_shift,
    [](Node* n) -> SROperator {
      if (n->matches(torch::schema(
              "aten::bitwise_left_shift.Tensor(Tensor self, Tensor other) -> Tensor"))) {
        return [](ProcessedNode* p_node) {
          const auto& self = p_node->Input(0).toTensor();
          const auto& other = p_node->Input(1).toTensor();
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::cpu::bitwise_left_shift(self, other);
            return;
          }
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          at::cpu::bitwise_left_shift_out(out, self, other);
        };
      }
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_OPERATOR_FUNCTOR(
    aten::bitwise_right_shift,
    aten_bitwise_right_shift,
    [](Node* n) -> SROperator {
      if (n->matches(torch::schema(
              "aten::bitwise_right_shift.Tensor(Tensor self, Tensor other) -> Tensor"))) {
        return [](ProcessedNode* p_node) {
          const auto& self = p_node->Input(0).toTensor();
          const auto& other = p_node->Input(1).toTensor();
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::cpu::bitwise_right_shift(self, other);
            return;
          }
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          at::cpu::bitwise_right_shift_out(out, self, other);
        };
      }
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_OPERATOR_FUNCTOR(aten::tril, aten_tril, [](Node* n) -> SROperator {
  if (n->matches(
          torch::schema("aten::tril(Tensor self, int diagonal=0) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      const auto diagonal = p_node->Input(1).toInt();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::tril(self, diagonal);
        return;
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::cpu::tril_out(out, self, diagonal);
    };
  }
  LogAndDumpSchema(n);
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(aten::triu, aten_triu, [](Node* n) -> SROperator {
  if (n->matches(
          torch::schema("aten::triu(Tensor self, int diagonal=0) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      const auto diagonal = p_node->Input(1).toInt();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::triu(self, diagonal);
        return;
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::cpu::triu_out(out, self, diagonal);
    };
  }
  LogAndDumpSchema(n);
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(
    aten::digamma,
    aten_digamma,
    [](Node* n) -> SROperator {
      if (n->matches(torch::schema("aten::digamma(Tensor self) -> Tensor"))) {
        return [](ProcessedNode* p_node) {
          const auto& self = p_node->Input(0).toTensor();
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::cpu::digamma(self);
            return;
          }
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          at::cpu::digamma_out(out, self);
        };
      }
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_OPERATOR_FUNCTOR(aten::lerp, aten_lerp, [](Node* n) -> SROperator {
  if (n->matches(torch::schema(
          "aten::lerp.Scalar(Tensor self, Tensor end, Scalar weight) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      const auto& end = p_node->Input(1).toTensor();
      const auto weight = p_node->Input(2).toScalar();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::lerp(self, end, weight);
        return;
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::cpu::lerp_out(out, self, end, weight);
    };
  }

  if (n->matches(torch::schema(
          "aten::lerp.Tensor(Tensor self, Tensor end, Tensor weight) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      const auto& end = p_node->Input(1).toTensor();
      const auto& weight = p_node->Input(2).toTensor();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::lerp(self, end, weight);
        return;
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::cpu::lerp_out(out, self, end, weight);
    };
  }
  LogAndDumpSchema(n);
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(aten::addbmm, aten_addbmm, [](Node* n) -> SROperator {
  if (n->matches(torch::schema(
          "aten::addbmm(Tensor self, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      const auto& batch1 = p_node->Input(1).toTensor();
      const auto& batch2 = p_node->Input(2).toTensor();
      const auto beta = p_node->Input(3).toScalar();
      const auto alpha = p_node->Input(4).toScalar();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) =
            at::native::addbmm(self, batch1, batch2, beta, alpha);
        return;
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::native::addbmm_out(self, batch1, batch2, beta, alpha, out);
    };
  }
  LogAndDumpSchema(n);
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(aten::diag, aten_diag, [](Node* n) -> SROperator {
  if (n->matches(
          torch::schema("aten::diag(Tensor self, int diagonal=0) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      const auto diagonal = p_node->Input(1).toInt();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::native::diag(self, diagonal);
        return;
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::native::diag_out(self, diagonal, out);
    };
  }
  LogAndDumpSchema(n);
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(aten::cross, aten_cross, [](Node* n) -> SROperator {
  if (n->matches(torch::schema(
          "aten::cross(Tensor self, Tensor other, int? dim=None) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      const auto& other = p_node->Input(1).toTensor();
      const auto dim = p_node->Input(2).toOptional<int64_t>();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::native::cross(self, other, dim);
        return;
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::native::cross_out(self, other, dim, out);
    };
  }
  LogAndDumpSchema(n);
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(aten::ne, aten_ne, [](Node* n) -> SROperator {
  if (n->matches(torch::schema(
          "aten::ne.Scalar(Tensor self, Scalar other) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      const auto other = p_node->Input(1).toScalar();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::ne(self, other);
        return;
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::cpu::ne_out(out, self, other);
    };
  }

  if (n->matches(torch::schema(
          "aten::ne.Tensor(Tensor self, Tensor other) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      const auto& other = p_node->Input(1).toTensor();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::ne(self, other);
        return;
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::cpu::ne_out(out, self, other);
    };
  }
  LogAndDumpSchema(n);
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(aten::ge, aten_ge, [](Node* n) -> SROperator {
  if (n->matches(torch::schema(
          "aten::ge.Scalar(Tensor self, Scalar other) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      const auto other = p_node->Input(1).toScalar();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::ge(self, other);
        return;
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::cpu::ge_out(out, self, other);
    };
  }

  if (n->matches(torch::schema(
          "aten::ge.Tensor(Tensor self, Tensor other) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      const auto& other = p_node->Input(1).toTensor();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::ge(self, other);
        return;
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::cpu::ge_out(out, self, other);
    };
  }
  LogAndDumpSchema(n);
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(aten::le, aten_le, [](Node* n) -> SROperator {
  if (n->matches(torch::schema(
          "aten::le.Scalar(Tensor self, Scalar other) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      const auto other = p_node->Input(1).toScalar();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::le(self, other);
        return;
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::cpu::le_out(out, self, other);
    };
  }

  if (n->matches(torch::schema(
          "aten::le.Tensor(Tensor self, Tensor other) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      const auto& other = p_node->Input(1).toTensor();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::le(self, other);
        return;
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::cpu::le_out(out, self, other);
    };
  }
  LogAndDumpSchema(n);
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(aten::gt, aten_gt, [](Node* n) -> SROperator {
  if (n->matches(torch::schema(
          "aten::gt.Scalar(Tensor self, Scalar other) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      const auto other = p_node->Input(1).toScalar();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::gt(self, other);
        return;
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::cpu::gt_out(out, self, other);
    };
  }

  if (n->matches(torch::schema(
          "aten::gt.Tensor(Tensor self, Tensor other) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      const auto& other = p_node->Input(1).toTensor();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::gt(self, other);
        return;
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::cpu::gt_out(out, self, other);
    };
  }
  LogAndDumpSchema(n);
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(aten::lt, aten_lt, [](Node* n) -> SROperator {
  if (n->matches(torch::schema(
          "aten::lt.Scalar(Tensor self, Scalar other) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      const auto other = p_node->Input(1).toScalar();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::lt(self, other);
        return;
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::cpu::lt_out(out, self, other);
    };
  }

  if (n->matches(torch::schema(
          "aten::lt.Tensor(Tensor self, Tensor other) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      const auto& other = p_node->Input(1).toTensor();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::lt(self, other);
        return;
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::cpu::lt_out(out, self, other);
    };
  }
  LogAndDumpSchema(n);
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(aten::take, aten_take, [](Node* n) -> SROperator {
  if (n->matches(
          torch::schema("aten::take(Tensor self, Tensor index) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      const auto& index = p_node->Input(1).toTensor();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::native::take(self, index);
        return;
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::native::take_out(self, index, out);
    };
  }
  LogAndDumpSchema(n);
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(
    aten::take_along_dim,
    aten_take_along_dim,
    [](Node* n) -> SROperator {
      if (n->matches(torch::schema(
              "aten::take_along_dim(Tensor self, Tensor indices, int? dim=None) -> Tensor"))) {
        return [](ProcessedNode* p_node) {
          const auto& self = p_node->Input(0).toTensor();
          const auto& indices = p_node->Input(1).toTensor();
          const auto dim = p_node->Input(2).toOptional<int64_t>();
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::native::take_along_dim(self, indices, dim);
            return;
          }
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          at::native::take_along_dim_out(self, indices, dim, out);
        };
      }
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_OPERATOR_FUNCTOR(
    aten::masked_select,
    aten_masked_select,
    [](Node* n) -> SROperator {
      if (n->matches(torch::schema(
              "aten::masked_select(Tensor self, Tensor mask) -> Tensor"))) {
        return [](ProcessedNode* p_node) {
          const auto& self = p_node->Input(0).toTensor();
          const auto& mask = p_node->Input(1).toTensor();
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::native::masked_select_cpu(self, mask);
            return;
          }
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          at::native::masked_select_out_cpu(self, mask, out);
        };
      }
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_OPERATOR_FUNCTOR(
    aten::nonzero_static,
    aten_nonzero_static,
    [](Node* n) -> SROperator {
      if (n->matches(torch::schema(
              "aten::nonzero_static(Tensor self, *, int size, int fill_value=-1) -> Tensor"))) {
        return [](ProcessedNode* p_node) {
          const auto& self = p_node->Input(0).toTensor();
          const auto size = p_node->Input(1).toInt();
          const auto fill_value = p_node->Input(2).toInt();
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) =
                at::native::nonzero_static_cpu(self, size, fill_value);
            return;
          }
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          at::native::nonzero_static_out_cpu(self, size, fill_value, out);
        };
      }
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_OPERATOR_FUNCTOR(aten::gather, aten_gather, [](Node* n) -> SROperator {
  if (n->matches(torch::schema(
          "aten::gather(Tensor self, int dim, Tensor index, *, bool sparse_grad=False) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      const auto dim = p_node->Input(1).toInt();
      const auto& index = p_node->Input(2).toTensor();
      const auto sparse_grad = p_node->Input(3).toBool();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::gather(self, dim, index, sparse_grad);
        return;
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::cpu::gather_out(out, self, dim, index, sparse_grad);
    };
  }
  LogAndDumpSchema(n);
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(aten::addcmul, aten_addcmul, [](Node* n) -> SROperator {
  if (n->matches(torch::schema(
          "aten::addcmul(Tensor self, Tensor tensor1, Tensor tensor2, *, Scalar value=1) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      const auto& tensor1 = p_node->Input(1).toTensor();
      const auto& tensor2 = p_node->Input(2).toTensor();
      const auto value = p_node->Input(3).toScalar();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::addcmul(self, tensor1, tensor2, value);
        return;
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::cpu::addcmul_out(out, self, tensor1, tensor2, value);
    };
  }
  LogAndDumpSchema(n);
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(aten::addcdiv, aten_addcdiv, [](Node* n) -> SROperator {
  if (n->matches(torch::schema(
          "aten::addcdiv(Tensor self, Tensor tensor1, Tensor tensor2, *, Scalar value=1) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      const auto& tensor1 = p_node->Input(1).toTensor();
      const auto& tensor2 = p_node->Input(2).toTensor();
      const auto value = p_node->Input(3).toScalar();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::addcdiv(self, tensor1, tensor2, value);
        return;
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::cpu::addcdiv_out(out, self, tensor1, tensor2, value);
    };
  }
  LogAndDumpSchema(n);
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(
    aten::linalg_solve_triangular,
    aten_linalg_solve_triangular,
    [](Node* n) -> SROperator {
      if (n->matches(torch::schema(
              "aten::linalg_solve_triangular(Tensor self, Tensor B, *, bool upper, bool left=True, bool unitriangular=False) -> Tensor"))) {
        return [](ProcessedNode* p_node) {
          const auto& self = p_node->Input(0).toTensor();
          const auto& B = p_node->Input(1).toTensor();
          const auto upper = p_node->Input(2).toBool();
          const auto left = p_node->Input(3).toBool();
          const auto unitriangular = p_node->Input(4).toBool();
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::native::linalg_solve_triangular(
                self, B, upper, left, unitriangular);
            return;
          }
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          at::native::linalg_solve_triangular_out(
              self, B, upper, left, unitriangular, out);
        };
      }
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_OPERATOR_FUNCTOR(
    aten::cholesky_solve,
    aten_cholesky_solve,
    [](Node* n) -> SROperator {
      if (n->matches(torch::schema(
              "aten::cholesky_solve(Tensor self, Tensor input2, bool upper=False) -> Tensor"))) {
        return [](ProcessedNode* p_node) {
          const auto& self = p_node->Input(0).toTensor();
          const auto& input2 = p_node->Input(1).toTensor();
          const auto upper = p_node->Input(2).toBool();
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::native::cholesky_solve(self, input2, upper);
            return;
          }
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          at::native::cholesky_solve_out(self, input2, upper, out);
        };
      }
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_OPERATOR_FUNCTOR(
    aten::cholesky_inverse,
    aten_cholesky_inverse,
    [](Node* n) -> SROperator {
      if (n->matches(torch::schema(
              "aten::cholesky_inverse(Tensor self, bool upper=False) -> Tensor"))) {
        return [](ProcessedNode* p_node) {
          const auto& self = p_node->Input(0).toTensor();
          const auto upper = p_node->Input(1).toBool();
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::native::cholesky_inverse(self, upper);
            return;
          }
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          at::native::cholesky_inverse_out(self, upper, out);
        };
      }
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_OPERATOR_FUNCTOR(aten::orgqr, aten_orgqr, [](Node* n) -> SROperator {
  if (n->matches(
          torch::schema("aten::orgqr(Tensor self, Tensor input2) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      const auto& input2 = p_node->Input(1).toTensor();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::native::orgqr(self, input2);
        return;
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::native::orgqr_out(self, input2, out);
    };
  }
  LogAndDumpSchema(n);
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(aten::ormqr, aten_ormqr, [](Node* n) -> SROperator {
  if (n->matches(torch::schema(
          "aten::ormqr(Tensor self, Tensor input2, Tensor input3, bool left=True, bool transpose=False) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      const auto& input2 = p_node->Input(1).toTensor();
      const auto& input3 = p_node->Input(2).toTensor();
      const auto left = p_node->Input(3).toBool();
      const auto transpose = p_node->Input(4).toBool();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) =
            at::native::ormqr(self, input2, input3, left, transpose);
        return;
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::native::ormqr_out(self, input2, input3, left, transpose, out);
    };
  }
  LogAndDumpSchema(n);
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(aten::lgamma, aten_lgamma, [](Node* n) -> SROperator {
  if (n->matches(torch::schema("aten::lgamma(Tensor self) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::lgamma(self);
        return;
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::cpu::lgamma_out(out, self);
    };
  }
  LogAndDumpSchema(n);
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(
    aten::polygamma,
    aten_polygamma,
    [](Node* n) -> SROperator {
      if (n->matches(
              torch::schema("aten::polygamma(int n, Tensor self) -> Tensor"))) {
        return [](ProcessedNode* p_node) {
          const auto n = p_node->Input(0).toInt();
          const auto& self = p_node->Input(1).toTensor();
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::cpu::polygamma(n, self);
            return;
          }
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          at::cpu::polygamma_out(out, n, self);
        };
      }
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_OPERATOR_FUNCTOR(aten::erfinv, aten_erfinv, [](Node* n) -> SROperator {
  if (n->matches(torch::schema("aten::erfinv(Tensor self) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::erfinv(self);
        return;
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::cpu::erfinv_out(out, self);
    };
  }
  LogAndDumpSchema(n);
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(aten::i0, aten_i0, [](Node* n) -> SROperator {
  if (n->matches(torch::schema("aten::i0(Tensor self) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::i0(self);
        return;
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::cpu::i0_out(out, self);
    };
  }
  LogAndDumpSchema(n);
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(
    aten::signbit,
    aten_signbit,
    [](Node* n) -> SROperator {
      if (n->matches(torch::schema("aten::signbit(Tensor self) -> Tensor"))) {
        return [](ProcessedNode* p_node) {
          const auto& self = p_node->Input(0).toTensor();
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::cpu::signbit(self);
            return;
          }
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          at::cpu::signbit_out(out, self);
        };
      }
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_OPERATOR_FUNCTOR(aten::atan2, aten_atan2, [](Node* n) -> SROperator {
  if (n->matches(
          torch::schema("aten::atan2(Tensor self, Tensor other) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      const auto& other = p_node->Input(1).toTensor();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::atan2(self, other);
        return;
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::cpu::atan2_out(out, self, other);
    };
  }
  LogAndDumpSchema(n);
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(
    aten::arctan2,
    aten_arctan2,
    [](Node* n) -> SROperator {
      if (n->matches(torch::schema(
              "aten::arctan2(Tensor self, Tensor other) -> Tensor"))) {
        return [](ProcessedNode* p_node) {
          const auto& self = p_node->Input(0).toTensor();
          const auto& other = p_node->Input(1).toTensor();
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::native::arctan2(self, other);
            return;
          }
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          at::native::arctan2_out(self, other, out);
        };
      }
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_OPERATOR_FUNCTOR(aten::histc, aten_histc, [](Node* n) -> SROperator {
  if (n->matches(torch::schema(
          "aten::histc(Tensor self, int bins=100, Scalar min=0, Scalar max=0) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      const auto bins = p_node->Input(1).toInt();
      const auto min = p_node->Input(2).toScalar();
      const auto max = p_node->Input(3).toScalar();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::native::histogram_histc(self, bins, min, max);
        return;
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::native::histogram_histc_out(self, bins, min, max, out);
    };
  }
  LogAndDumpSchema(n);
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(aten::hypot, aten_hypot, [](Node* n) -> SROperator {
  if (n->matches(
          torch::schema("aten::hypot(Tensor self, Tensor other) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      const auto& other = p_node->Input(1).toTensor();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::hypot(self, other);
        return;
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::cpu::hypot_out(out, self, other);
    };
  }
  LogAndDumpSchema(n);
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(aten::igamma, aten_igamma, [](Node* n) -> SROperator {
  if (n->matches(
          torch::schema("aten::igamma(Tensor self, Tensor other) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      const auto& other = p_node->Input(1).toTensor();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::igamma(self, other);
        return;
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::cpu::igamma_out(out, self, other);
    };
  }
  LogAndDumpSchema(n);
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(
    aten::igammac,
    aten_igammac,
    [](Node* n) -> SROperator {
      if (n->matches(torch::schema(
              "aten::igammac(Tensor self, Tensor other) -> Tensor"))) {
        return [](ProcessedNode* p_node) {
          const auto& self = p_node->Input(0).toTensor();
          const auto& other = p_node->Input(1).toTensor();
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::cpu::igammac(self, other);
            return;
          }
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          at::cpu::igammac_out(out, self, other);
        };
      }
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_OPERATOR_FUNCTOR(
    aten::nextafter,
    aten_nextafter,
    [](Node* n) -> SROperator {
      if (n->matches(torch::schema(
              "aten::nextafter(Tensor self, Tensor other) -> Tensor"))) {
        return [](ProcessedNode* p_node) {
          const auto& self = p_node->Input(0).toTensor();
          const auto& other = p_node->Input(1).toTensor();
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::cpu::nextafter(self, other);
            return;
          }
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          at::cpu::nextafter_out(out, self, other);
        };
      }
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_OPERATOR_FUNCTOR(aten::fmin, aten_fmin, [](Node* n) -> SROperator {
  if (n->matches(
          torch::schema("aten::fmin(Tensor self, Tensor other) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      const auto& other = p_node->Input(1).toTensor();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::fmin(self, other);
        return;
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::cpu::fmin_out(out, self, other);
    };
  }
  LogAndDumpSchema(n);
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(aten::fmax, aten_fmax, [](Node* n) -> SROperator {
  if (n->matches(
          torch::schema("aten::fmax(Tensor self, Tensor other) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      const auto& other = p_node->Input(1).toTensor();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::fmax(self, other);
        return;
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::cpu::fmax_out(out, self, other);
    };
  }
  LogAndDumpSchema(n);
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(
    aten::maximum,
    aten_maximum,
    [](Node* n) -> SROperator {
      if (n->matches(torch::schema(
              "aten::maximum(Tensor self, Tensor other) -> Tensor"))) {
        return [](ProcessedNode* p_node) {
          const auto& self = p_node->Input(0).toTensor();
          const auto& other = p_node->Input(1).toTensor();
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::cpu::maximum(self, other);
            return;
          }
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          at::cpu::maximum_out(out, self, other);
        };
      }
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_OPERATOR_FUNCTOR(
    aten::minimum,
    aten_minimum,
    [](Node* n) -> SROperator {
      if (n->matches(torch::schema(
              "aten::minimum(Tensor self, Tensor other) -> Tensor"))) {
        return [](ProcessedNode* p_node) {
          const auto& self = p_node->Input(0).toTensor();
          const auto& other = p_node->Input(1).toTensor();
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::cpu::minimum(self, other);
            return;
          }
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          at::cpu::minimum_out(out, self, other);
        };
      }
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_OPERATOR_FUNCTOR(aten::min, aten_min, [](Node* n) -> SROperator {
  if (n->matches(torch::schema(
          "aten::min.other(Tensor self, Tensor other) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      const auto& other = p_node->Input(1).toTensor();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::native::min(self, other);
        return;
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::native::min_out(self, other, out);
    };
  }
  LogAndDumpSchema(n);
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(aten::quantile, aten_quantile, [](Node* n) -> SROperator {
  if (n->matches(torch::schema(
          "aten::quantile(Tensor self, Tensor q, int? dim=None, bool keepdim=False, *, str interpolation='linear') -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      const auto& q = p_node->Input(1).toTensor();
      const auto dim = p_node->Input(2).toOptional<int64_t>();
      const auto keepdim = p_node->Input(3).toBool();
      const auto interpolation = p_node->Input(4).toStringView();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) =
            at::native::quantile(self, q, dim, keepdim, interpolation);
        return;
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::native::quantile_out(self, q, dim, keepdim, interpolation, out);
    };
  }
  LogAndDumpSchema(n);
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(aten::nanquantile, aten_nanquantile, [](Node* n) -> SROperator {
  if (n->matches(torch::schema(
          "aten::nanquantile(Tensor self, Tensor q, int? dim=None, bool keepdim=False, *, str interpolation='linear') -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      const auto& q = p_node->Input(1).toTensor();
      const auto dim = p_node->Input(2).toOptional<int64_t>();
      const auto keepdim = p_node->Input(3).toBool();
      const auto interpolation = p_node->Input(4).toStringView();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) =
            at::native::nanquantile(self, q, dim, keepdim, interpolation);
        return;
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::native::nanquantile_out(self, q, dim, keepdim, interpolation, out);
    };
  }
  LogAndDumpSchema(n);
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(aten::msort, aten_msort, [](Node* n) -> SROperator {
  if (n->matches(torch::schema("aten::msort(Tensor self) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::native::msort(self);
        return;
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::native::msort_out(self, out);
    };
  }
  LogAndDumpSchema(n);
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(aten::renorm, aten_renorm, [](Node* n) -> SROperator {
  if (n->matches(torch::schema(
          "aten::renorm(Tensor self, Scalar p, int dim, Scalar maxnorm) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      const auto p = p_node->Input(1).toScalar();
      const auto dim = p_node->Input(2).toInt();
      const auto maxnorm = p_node->Input(3).toScalar();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::renorm(self, p, dim, maxnorm);
        return;
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::cpu::renorm_out(out, self, p, dim, maxnorm);
    };
  }
  LogAndDumpSchema(n);
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(
    aten::_convert_indices_from_coo_to_csr,
    aten__convert_indices_from_coo_to_csr,
    [](Node* n) -> SROperator {
      if (n->matches(torch::schema(
              "aten::_convert_indices_from_coo_to_csr(Tensor self, int size, *, bool out_int32=False) -> Tensor"))) {
        return [](ProcessedNode* p_node) {
          const auto& self = p_node->Input(0).toTensor();
          const auto size = p_node->Input(1).toInt();
          const auto out_int32 = p_node->Input(2).toBool();
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::cpu::_convert_indices_from_coo_to_csr(
                self, size, out_int32);
            return;
          }
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          at::cpu::_convert_indices_from_coo_to_csr_out(
              out, self, size, out_int32);
        };
      }
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_OPERATOR_FUNCTOR(
    aten::_convert_indices_from_csr_to_coo,
    aten__convert_indices_from_csr_to_coo,
    [](Node* n) -> SROperator {
      if (n->matches(torch::schema(
              "aten::_convert_indices_from_csr_to_coo(Tensor crow_indices, Tensor col_indices, *, bool out_int32=False, bool transpose=False) -> Tensor"))) {
        return [](ProcessedNode* p_node) {
          const auto& crow_indices = p_node->Input(0).toTensor();
          const auto& col_indices = p_node->Input(1).toTensor();
          const auto out_int32 = p_node->Input(2).toBool();
          const auto transpose = p_node->Input(3).toBool();
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::cpu::_convert_indices_from_csr_to_coo(
                crow_indices, col_indices, out_int32, transpose);
            return;
          }
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          at::cpu::_convert_indices_from_csr_to_coo_out(
              out, crow_indices, col_indices, out_int32, transpose);
        };
      }
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_OPERATOR_FUNCTOR(aten::mse_loss, aten_mse_loss, [](Node* n) -> SROperator {
  if (n->matches(torch::schema(
          "aten::mse_loss(Tensor self, Tensor target, int reduction=Mean) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      const auto& target = p_node->Input(1).toTensor();
      const auto reduction = p_node->Input(2).toInt();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::mse_loss(self, target, reduction);
        return;
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::cpu::mse_loss_out(out, self, target, reduction);
    };
  }
  LogAndDumpSchema(n);
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(
    aten::multi_margin_loss,
    aten_multi_margin_loss,
    [](Node* n) -> SROperator {
      if (n->matches(torch::schema(
              "aten::multi_margin_loss(Tensor self, Tensor target, Scalar p=1, Scalar margin=1, Tensor? weight=None, int reduction=Mean) -> Tensor"))) {
        return [](ProcessedNode* p_node) {
          const auto& self = p_node->Input(0).toTensor();
          const auto& target = p_node->Input(1).toTensor();
          const auto p = p_node->Input(2).toScalar();
          const auto margin = p_node->Input(3).toScalar();
          const auto weight = p_node->Input(4).toOptional<at::Tensor>();
          const auto reduction = p_node->Input(5).toInt();
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::native::multi_margin_loss_cpu(
                self, target, p, margin, weight, reduction);
            return;
          }
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          at::native::multi_margin_loss_cpu_out(
              self, target, p, margin, weight, reduction, out);
        };
      }
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_OPERATOR_FUNCTOR(
    aten::multilabel_margin_loss,
    aten_multilabel_margin_loss,
    [](Node* n) -> SROperator {
      if (n->matches(torch::schema(
              "aten::multilabel_margin_loss(Tensor self, Tensor target, int reduction=Mean) -> Tensor"))) {
        return [](ProcessedNode* p_node) {
          const auto& self = p_node->Input(0).toTensor();
          const auto& target = p_node->Input(1).toTensor();
          const auto reduction = p_node->Input(2).toInt();
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) =
                at::native::multilabel_margin_loss(self, target, reduction);
            return;
          }
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          at::native::multilabel_margin_loss_out(self, target, reduction, out);
        };
      }
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_OPERATOR_FUNCTOR(
    aten::soft_margin_loss,
    aten_soft_margin_loss,
    [](Node* n) -> SROperator {
      if (n->matches(torch::schema(
              "aten::soft_margin_loss(Tensor self, Tensor target, int reduction=Mean) -> Tensor"))) {
        return [](ProcessedNode* p_node) {
          const auto& self = p_node->Input(0).toTensor();
          const auto& target = p_node->Input(1).toTensor();
          const auto reduction = p_node->Input(2).toInt();
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) =
                at::native::soft_margin_loss(self, target, reduction);
            return;
          }
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          at::native::soft_margin_loss_out(self, target, reduction, out);
        };
      }
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_OPERATOR_FUNCTOR(aten::elu, aten_elu, [](Node* n) -> SROperator {
  if (n->matches(torch::schema(
          "aten::elu(Tensor self, Scalar alpha=1, Scalar scale=1, Scalar input_scale=1) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      const auto alpha = p_node->Input(1).toScalar();
      const auto scale = p_node->Input(2).toScalar();
      const auto input_scale = p_node->Input(3).toScalar();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::elu(self, alpha, scale, input_scale);
        return;
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::cpu::elu_out(out, self, alpha, scale, input_scale);
    };
  }
  LogAndDumpSchema(n);
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(
    aten::elu_backward,
    aten_elu_backward,
    [](Node* n) -> SROperator {
      if (n->matches(torch::schema(
              "aten::elu_backward(Tensor grad_output, Scalar alpha, Scalar scale, Scalar input_scale, bool is_result, Tensor self_or_result) -> Tensor"))) {
        return [](ProcessedNode* p_node) {
          const auto& grad_output = p_node->Input(0).toTensor();
          const auto alpha = p_node->Input(1).toScalar();
          const auto scale = p_node->Input(2).toScalar();
          const auto input_scale = p_node->Input(3).toScalar();
          const auto is_result = p_node->Input(4).toBool();
          const auto& self_or_result = p_node->Input(5).toTensor();
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::cpu::elu_backward(
                grad_output,
                alpha,
                scale,
                input_scale,
                is_result,
                self_or_result);
            return;
          }
          auto& grad_input = p_node->Output(0).toTensor();
          fastResizeToZero(grad_input);
          at::cpu::elu_backward_out(
              grad_input,
              grad_output,
              alpha,
              scale,
              input_scale,
              is_result,
              self_or_result);
        };
      }
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_OPERATOR_FUNCTOR(aten::glu, aten_glu, [](Node* n) -> SROperator {
  if (n->matches(
          torch::schema("aten::glu(Tensor self, int dim=-1) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      const auto dim = p_node->Input(1).toInt();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::glu(self, dim);
        return;
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::cpu::glu_out(out, self, dim);
    };
  }
  LogAndDumpSchema(n);
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(
    aten::hardsigmoid,
    aten_hardsigmoid,
    [](Node* n) -> SROperator {
      if (n->matches(
              torch::schema("aten::hardsigmoid(Tensor self) -> Tensor"))) {
        return [](ProcessedNode* p_node) {
          const auto& self = p_node->Input(0).toTensor();
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::cpu::hardsigmoid(self);
            return;
          }
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          at::cpu::hardsigmoid_out(out, self);
        };
      }
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_OPERATOR_FUNCTOR(
    aten::hardsigmoid_backward,
    aten_hardsigmoid_backward,
    [](Node* n) -> SROperator {
      if (n->matches(torch::schema(
              "aten::hardsigmoid_backward(Tensor grad_output, Tensor self) -> Tensor"))) {
        return [](ProcessedNode* p_node) {
          const auto& grad_output = p_node->Input(0).toTensor();
          const auto& self = p_node->Input(1).toTensor();
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) =
                at::cpu::hardsigmoid_backward(grad_output, self);
            return;
          }
          auto& grad_input = p_node->Output(0).toTensor();
          fastResizeToZero(grad_input);
          at::cpu::hardsigmoid_backward_out(grad_input, grad_output, self);
        };
      }
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_OPERATOR_FUNCTOR(aten::hardtanh, aten_hardtanh, [](Node* n) -> SROperator {
  if (n->matches(torch::schema(
          "aten::hardtanh(Tensor self, Scalar min_val=-1, Scalar max_val=1) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      const auto min_val = p_node->Input(1).toScalar();
      const auto max_val = p_node->Input(2).toScalar();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::native::hardtanh(self, min_val, max_val);
        return;
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::native::hardtanh_out(self, min_val, max_val, out);
    };
  }
  LogAndDumpSchema(n);
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(
    aten::hardswish,
    aten_hardswish,
    [](Node* n) -> SROperator {
      if (n->matches(torch::schema("aten::hardswish(Tensor self) -> Tensor"))) {
        return [](ProcessedNode* p_node) {
          const auto& self = p_node->Input(0).toTensor();
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::native::hardswish(self);
            return;
          }
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          at::native::hardswish_out(self, out);
        };
      }
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_OPERATOR_FUNCTOR(
    aten::leaky_relu_backward,
    aten_leaky_relu_backward,
    [](Node* n) -> SROperator {
      if (n->matches(torch::schema(
              "aten::leaky_relu_backward(Tensor grad_output, Tensor self, Scalar negative_slope, bool self_is_result) -> Tensor"))) {
        return [](ProcessedNode* p_node) {
          const auto& grad_output = p_node->Input(0).toTensor();
          const auto& self = p_node->Input(1).toTensor();
          const auto negative_slope = p_node->Input(2).toScalar();
          const auto self_is_result = p_node->Input(3).toBool();
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::cpu::leaky_relu_backward(
                grad_output, self, negative_slope, self_is_result);
            return;
          }
          auto& grad_input = p_node->Output(0).toTensor();
          fastResizeToZero(grad_input);
          at::cpu::leaky_relu_backward_out(
              grad_input, grad_output, self, negative_slope, self_is_result);
        };
      }
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_OPERATOR_FUNCTOR(
    aten::log_sigmoid,
    aten_log_sigmoid,
    [](Node* n) -> SROperator {
      if (n->matches(
              torch::schema("aten::log_sigmoid(Tensor self) -> Tensor"))) {
        return [](ProcessedNode* p_node) {
          const auto& self = p_node->Input(0).toTensor();
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::native::log_sigmoid(self);
            return;
          }
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          at::native::log_sigmoid_out(self, out);
        };
      }
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_OPERATOR_FUNCTOR(aten::softplus, aten_softplus, [](Node* n) -> SROperator {
  if (n->matches(torch::schema(
          "aten::softplus(Tensor self, Scalar beta=1, Scalar threshold=20) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      const auto beta = p_node->Input(1).toScalar();
      const auto threshold = p_node->Input(2).toScalar();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::cpu::softplus(self, beta, threshold);
        return;
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::cpu::softplus_out(out, self, beta, threshold);
    };
  }
  LogAndDumpSchema(n);
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(
    aten::softplus_backward,
    aten_softplus_backward,
    [](Node* n) -> SROperator {
      if (n->matches(torch::schema(
              "aten::softplus_backward(Tensor grad_output, Tensor self, Scalar beta, Scalar threshold) -> Tensor"))) {
        return [](ProcessedNode* p_node) {
          const auto& grad_output = p_node->Input(0).toTensor();
          const auto& self = p_node->Input(1).toTensor();
          const auto beta = p_node->Input(2).toScalar();
          const auto threshold = p_node->Input(3).toScalar();
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) =
                at::cpu::softplus_backward(grad_output, self, beta, threshold);
            return;
          }
          auto& grad_input = p_node->Output(0).toTensor();
          fastResizeToZero(grad_input);
          at::cpu::softplus_backward_out(
              grad_input, grad_output, self, beta, threshold);
        };
      }
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_OPERATOR_FUNCTOR(
    aten::softshrink,
    aten_softshrink,
    [](Node* n) -> SROperator {
      if (n->matches(torch::schema(
              "aten::softshrink(Tensor self, Scalar lambd=0.5) -> Tensor"))) {
        return [](ProcessedNode* p_node) {
          const auto& self = p_node->Input(0).toTensor();
          const auto lambd = p_node->Input(1).toScalar();
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::cpu::softshrink(self, lambd);
            return;
          }
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          at::cpu::softshrink_out(out, self, lambd);
        };
      }
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_OPERATOR_FUNCTOR(
    aten::softshrink_backward,
    aten_softshrink_backward,
    [](Node* n) -> SROperator {
      if (n->matches(torch::schema(
              "aten::softshrink_backward(Tensor grad_output, Tensor self, Scalar lambd) -> Tensor"))) {
        return [](ProcessedNode* p_node) {
          const auto& grad_output = p_node->Input(0).toTensor();
          const auto& self = p_node->Input(1).toTensor();
          const auto lambd = p_node->Input(2).toScalar();
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) =
                at::cpu::softshrink_backward(grad_output, self, lambd);
            return;
          }
          auto& grad_input = p_node->Output(0).toTensor();
          fastResizeToZero(grad_input);
          at::cpu::softshrink_backward_out(
              grad_input, grad_output, self, lambd);
        };
      }
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_OPERATOR_FUNCTOR(
    aten::adaptive_max_pool2d_backward,
    aten_adaptive_max_pool2d_backward,
    [](Node* n) -> SROperator {
      if (n->matches(torch::schema(
              "aten::adaptive_max_pool2d_backward(Tensor grad_output, Tensor self, Tensor indices) -> Tensor"))) {
        return [](ProcessedNode* p_node) {
          const auto& grad_output = p_node->Input(0).toTensor();
          const auto& self = p_node->Input(1).toTensor();
          const auto& indices = p_node->Input(2).toTensor();
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::cpu::adaptive_max_pool2d_backward(
                grad_output, self, indices);
            return;
          }
          auto& grad_input = p_node->Output(0).toTensor();
          fastResizeToZero(grad_input);
          at::cpu::adaptive_max_pool2d_backward_out(
              grad_input, grad_output, self, indices);
        };
      }
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_OPERATOR_FUNCTOR(
    aten::adaptive_max_pool3d_backward,
    aten_adaptive_max_pool3d_backward,
    [](Node* n) -> SROperator {
      if (n->matches(torch::schema(
              "aten::adaptive_max_pool3d_backward(Tensor grad_output, Tensor self, Tensor indices) -> Tensor"))) {
        return [](ProcessedNode* p_node) {
          const auto& grad_output = p_node->Input(0).toTensor();
          const auto& self = p_node->Input(1).toTensor();
          const auto& indices = p_node->Input(2).toTensor();
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::cpu::adaptive_max_pool3d_backward(
                grad_output, self, indices);
            return;
          }
          auto& grad_input = p_node->Output(0).toTensor();
          fastResizeToZero(grad_input);
          at::cpu::adaptive_max_pool3d_backward_out(
              grad_input, grad_output, self, indices);
        };
      }
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_OPERATOR_FUNCTOR(
    aten::sigmoid_backward,
    aten_sigmoid_backward,
    [](Node* n) -> SROperator {
      if (n->matches(torch::schema(
              "aten::sigmoid_backward(Tensor grad_output, Tensor output) -> Tensor"))) {
        return [](ProcessedNode* p_node) {
          const auto& grad_output = p_node->Input(0).toTensor();
          const auto& output = p_node->Input(1).toTensor();
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::cpu::sigmoid_backward(grad_output, output);
            return;
          }
          auto& grad_input = p_node->Output(0).toTensor();
          fastResizeToZero(grad_input);
          at::cpu::sigmoid_backward_out(grad_input, grad_output, output);
        };
      }
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_OPERATOR_FUNCTOR(
    aten::tanh_backward,
    aten_tanh_backward,
    [](Node* n) -> SROperator {
      if (n->matches(torch::schema(
              "aten::tanh_backward(Tensor grad_output, Tensor output) -> Tensor"))) {
        return [](ProcessedNode* p_node) {
          const auto& grad_output = p_node->Input(0).toTensor();
          const auto& output = p_node->Input(1).toTensor();
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::cpu::tanh_backward(grad_output, output);
            return;
          }
          auto& grad_input = p_node->Output(0).toTensor();
          fastResizeToZero(grad_input);
          at::cpu::tanh_backward_out(grad_input, grad_output, output);
        };
      }
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_OPERATOR_FUNCTOR(
    aten::isposinf,
    aten_isposinf,
    [](Node* n) -> SROperator {
      if (n->matches(torch::schema("aten::isposinf(Tensor self) -> Tensor"))) {
        return [](ProcessedNode* p_node) {
          const auto& self = p_node->Input(0).toTensor();
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::cpu::isposinf(self);
            return;
          }
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          at::cpu::isposinf_out(out, self);
        };
      }
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_OPERATOR_FUNCTOR(
    aten::isneginf,
    aten_isneginf,
    [](Node* n) -> SROperator {
      if (n->matches(torch::schema("aten::isneginf(Tensor self) -> Tensor"))) {
        return [](ProcessedNode* p_node) {
          const auto& self = p_node->Input(0).toTensor();
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::cpu::isneginf(self);
            return;
          }
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          at::cpu::isneginf_out(out, self);
        };
      }
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_OPERATOR_FUNCTOR(
    aten::special_entr,
    aten_special_entr,
    [](Node* n) -> SROperator {
      if (n->matches(
              torch::schema("aten::special_entr(Tensor self) -> Tensor"))) {
        return [](ProcessedNode* p_node) {
          const auto& self = p_node->Input(0).toTensor();
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::cpu::special_entr(self);
            return;
          }
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          at::cpu::special_entr_out(out, self);
        };
      }
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_OPERATOR_FUNCTOR(
    aten::special_ndtri,
    aten_special_ndtri,
    [](Node* n) -> SROperator {
      if (n->matches(
              torch::schema("aten::special_ndtri(Tensor self) -> Tensor"))) {
        return [](ProcessedNode* p_node) {
          const auto& self = p_node->Input(0).toTensor();
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::cpu::special_ndtri(self);
            return;
          }
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          at::cpu::special_ndtri_out(out, self);
        };
      }
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_OPERATOR_FUNCTOR(
    aten::special_log_ndtr,
    aten_special_log_ndtr,
    [](Node* n) -> SROperator {
      if (n->matches(
              torch::schema("aten::special_log_ndtr(Tensor self) -> Tensor"))) {
        return [](ProcessedNode* p_node) {
          const auto& self = p_node->Input(0).toTensor();
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::cpu::special_log_ndtr(self);
            return;
          }
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          at::cpu::special_log_ndtr_out(out, self);
        };
      }
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_OPERATOR_FUNCTOR(
    aten::special_expm1,
    aten_special_expm1,
    [](Node* n) -> SROperator {
      if (n->matches(
              torch::schema("aten::special_expm1(Tensor self) -> Tensor"))) {
        return [](ProcessedNode* p_node) {
          const auto& self = p_node->Input(0).toTensor();
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::native::special_expm1(self);
            return;
          }
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          at::native::special_expm1_out(self, out);
        };
      }
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_OPERATOR_FUNCTOR(
    aten::special_exp2,
    aten_special_exp2,
    [](Node* n) -> SROperator {
      if (n->matches(
              torch::schema("aten::special_exp2(Tensor self) -> Tensor"))) {
        return [](ProcessedNode* p_node) {
          const auto& self = p_node->Input(0).toTensor();
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::native::special_exp2(self);
            return;
          }
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          at::native::special_exp2_out(self, out);
        };
      }
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_OPERATOR_FUNCTOR(
    aten::special_psi,
    aten_special_psi,
    [](Node* n) -> SROperator {
      if (n->matches(
              torch::schema("aten::special_psi(Tensor self) -> Tensor"))) {
        return [](ProcessedNode* p_node) {
          const auto& self = p_node->Input(0).toTensor();
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::native::special_psi(self);
            return;
          }
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          at::native::special_psi_out(self, out);
        };
      }
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_OPERATOR_FUNCTOR(
    aten::special_digamma,
    aten_special_digamma,
    [](Node* n) -> SROperator {
      if (n->matches(
              torch::schema("aten::special_digamma(Tensor self) -> Tensor"))) {
        return [](ProcessedNode* p_node) {
          const auto& self = p_node->Input(0).toTensor();
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::native::special_digamma(self);
            return;
          }
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          at::native::special_digamma_out(self, out);
        };
      }
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_OPERATOR_FUNCTOR(
    aten::special_gammaln,
    aten_special_gammaln,
    [](Node* n) -> SROperator {
      if (n->matches(
              torch::schema("aten::special_gammaln(Tensor self) -> Tensor"))) {
        return [](ProcessedNode* p_node) {
          const auto& self = p_node->Input(0).toTensor();
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::native::special_gammaln(self);
            return;
          }
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          at::native::special_gammaln_out(self, out);
        };
      }
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_OPERATOR_FUNCTOR(
    aten::special_erf,
    aten_special_erf,
    [](Node* n) -> SROperator {
      if (n->matches(
              torch::schema("aten::special_erf(Tensor self) -> Tensor"))) {
        return [](ProcessedNode* p_node) {
          const auto& self = p_node->Input(0).toTensor();
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::native::special_erf(self);
            return;
          }
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          at::native::special_erf_out(self, out);
        };
      }
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_OPERATOR_FUNCTOR(
    aten::special_erfc,
    aten_special_erfc,
    [](Node* n) -> SROperator {
      if (n->matches(
              torch::schema("aten::special_erfc(Tensor self) -> Tensor"))) {
        return [](ProcessedNode* p_node) {
          const auto& self = p_node->Input(0).toTensor();
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::native::special_erfc(self);
            return;
          }
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          at::native::special_erfc_out(self, out);
        };
      }
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_OPERATOR_FUNCTOR(
    aten::special_erfcx,
    aten_special_erfcx,
    [](Node* n) -> SROperator {
      if (n->matches(
              torch::schema("aten::special_erfcx(Tensor self) -> Tensor"))) {
        return [](ProcessedNode* p_node) {
          const auto& self = p_node->Input(0).toTensor();
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::cpu::special_erfcx(self);
            return;
          }
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          at::cpu::special_erfcx_out(out, self);
        };
      }
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_OPERATOR_FUNCTOR(
    aten::special_erfinv,
    aten_special_erfinv,
    [](Node* n) -> SROperator {
      if (n->matches(
              torch::schema("aten::special_erfinv(Tensor self) -> Tensor"))) {
        return [](ProcessedNode* p_node) {
          const auto& self = p_node->Input(0).toTensor();
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::native::special_erfinv(self);
            return;
          }
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          at::native::special_erfinv_out(self, out);
        };
      }
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_OPERATOR_FUNCTOR(
    aten::special_ndtr,
    aten_special_ndtr,
    [](Node* n) -> SROperator {
      if (n->matches(
              torch::schema("aten::special_ndtr(Tensor self) -> Tensor"))) {
        return [](ProcessedNode* p_node) {
          const auto& self = p_node->Input(0).toTensor();
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::native::special_ndtr(self);
            return;
          }
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          at::native::special_ndtr_out(self, out);
        };
      }
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_OPERATOR_FUNCTOR(
    aten::special_xlog1py,
    aten_special_xlog1py,
    [](Node* n) -> SROperator {
      if (n->matches(torch::schema(
              "aten::special_xlog1py(Tensor self, Tensor other) -> Tensor"))) {
        return [](ProcessedNode* p_node) {
          const auto& self = p_node->Input(0).toTensor();
          const auto& other = p_node->Input(1).toTensor();
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::cpu::special_xlog1py(self, other);
            return;
          }
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          at::cpu::special_xlog1py_out(out, self, other);
        };
      }
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_OPERATOR_FUNCTOR(
    aten::special_xlogy,
    aten_special_xlogy,
    [](Node* n) -> SROperator {
      if (n->matches(torch::schema(
              "aten::special_xlogy(Tensor self, Tensor other) -> Tensor"))) {
        return [](ProcessedNode* p_node) {
          const auto& self = p_node->Input(0).toTensor();
          const auto& other = p_node->Input(1).toTensor();
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::native::special_xlogy(self, other);
            return;
          }
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          at::native::special_xlogy_out(self, other, out);
        };
      }
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_OPERATOR_FUNCTOR(
    aten::special_zeta,
    aten_special_zeta,
    [](Node* n) -> SROperator {
      if (n->matches(torch::schema(
              "aten::special_zeta(Tensor self, Tensor other) -> Tensor"))) {
        return [](ProcessedNode* p_node) {
          const auto& self = p_node->Input(0).toTensor();
          const auto& other = p_node->Input(1).toTensor();
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::cpu::special_zeta(self, other);
            return;
          }
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          at::cpu::special_zeta_out(out, self, other);
        };
      }
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_OPERATOR_FUNCTOR(
    aten::special_i0,
    aten_special_i0,
    [](Node* n) -> SROperator {
      if (n->matches(
              torch::schema("aten::special_i0(Tensor self) -> Tensor"))) {
        return [](ProcessedNode* p_node) {
          const auto& self = p_node->Input(0).toTensor();
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::native::special_i0(self);
            return;
          }
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          at::native::special_i0_out(self, out);
        };
      }
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_OPERATOR_FUNCTOR(
    aten::special_i0e,
    aten_special_i0e,
    [](Node* n) -> SROperator {
      if (n->matches(
              torch::schema("aten::special_i0e(Tensor self) -> Tensor"))) {
        return [](ProcessedNode* p_node) {
          const auto& self = p_node->Input(0).toTensor();
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::cpu::special_i0e(self);
            return;
          }
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          at::cpu::special_i0e_out(out, self);
        };
      }
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_OPERATOR_FUNCTOR(
    aten::special_i1,
    aten_special_i1,
    [](Node* n) -> SROperator {
      if (n->matches(
              torch::schema("aten::special_i1(Tensor self) -> Tensor"))) {
        return [](ProcessedNode* p_node) {
          const auto& self = p_node->Input(0).toTensor();
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::cpu::special_i1(self);
            return;
          }
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          at::cpu::special_i1_out(out, self);
        };
      }
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_OPERATOR_FUNCTOR(
    aten::special_i1e,
    aten_special_i1e,
    [](Node* n) -> SROperator {
      if (n->matches(
              torch::schema("aten::special_i1e(Tensor self) -> Tensor"))) {
        return [](ProcessedNode* p_node) {
          const auto& self = p_node->Input(0).toTensor();
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::cpu::special_i1e(self);
            return;
          }
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          at::cpu::special_i1e_out(out, self);
        };
      }
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_OPERATOR_FUNCTOR(
    aten::special_polygamma,
    aten_special_polygamma,
    [](Node* n) -> SROperator {
      if (n->matches(torch::schema(
              "aten::special_polygamma(int n, Tensor self) -> Tensor"))) {
        return [](ProcessedNode* p_node) {
          const auto n = p_node->Input(0).toInt();
          const auto& self = p_node->Input(1).toTensor();
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::native::special_polygamma(n, self);
            return;
          }
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          at::native::special_polygamma_out(n, self, out);
        };
      }
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_OPERATOR_FUNCTOR(
    aten::special_expit,
    aten_special_expit,
    [](Node* n) -> SROperator {
      if (n->matches(
              torch::schema("aten::special_expit(Tensor self) -> Tensor"))) {
        return [](ProcessedNode* p_node) {
          const auto& self = p_node->Input(0).toTensor();
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::native::special_expit(self);
            return;
          }
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          at::native::special_expit_out(self, out);
        };
      }
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_OPERATOR_FUNCTOR(
    aten::special_sinc,
    aten_special_sinc,
    [](Node* n) -> SROperator {
      if (n->matches(
              torch::schema("aten::special_sinc(Tensor self) -> Tensor"))) {
        return [](ProcessedNode* p_node) {
          const auto& self = p_node->Input(0).toTensor();
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::native::special_sinc(self);
            return;
          }
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          at::native::special_sinc_out(self, out);
        };
      }
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_OPERATOR_FUNCTOR(
    aten::special_round,
    aten_special_round,
    [](Node* n) -> SROperator {
      if (n->matches(torch::schema(
              "aten::special_round(Tensor self, *, int decimals=0) -> Tensor"))) {
        return [](ProcessedNode* p_node) {
          const auto& self = p_node->Input(0).toTensor();
          const auto decimals = p_node->Input(1).toInt();
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::native::special_round(self, decimals);
            return;
          }
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          at::native::special_round_out(self, decimals, out);
        };
      }
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_OPERATOR_FUNCTOR(
    aten::special_log1p,
    aten_special_log1p,
    [](Node* n) -> SROperator {
      if (n->matches(
              torch::schema("aten::special_log1p(Tensor self) -> Tensor"))) {
        return [](ProcessedNode* p_node) {
          const auto& self = p_node->Input(0).toTensor();
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::native::special_log1p(self);
            return;
          }
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          at::native::special_log1p_out(self, out);
        };
      }
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_OPERATOR_FUNCTOR(
    aten::special_gammainc,
    aten_special_gammainc,
    [](Node* n) -> SROperator {
      if (n->matches(torch::schema(
              "aten::special_gammainc(Tensor self, Tensor other) -> Tensor"))) {
        return [](ProcessedNode* p_node) {
          const auto& self = p_node->Input(0).toTensor();
          const auto& other = p_node->Input(1).toTensor();
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::native::special_gammainc(self, other);
            return;
          }
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          at::native::special_gammainc_out(self, other, out);
        };
      }
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_OPERATOR_FUNCTOR(
    aten::special_gammaincc,
    aten_special_gammaincc,
    [](Node* n) -> SROperator {
      if (n->matches(torch::schema(
              "aten::special_gammaincc(Tensor self, Tensor other) -> Tensor"))) {
        return [](ProcessedNode* p_node) {
          const auto& self = p_node->Input(0).toTensor();
          const auto& other = p_node->Input(1).toTensor();
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::native::special_gammaincc(self, other);
            return;
          }
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          at::native::special_gammaincc_out(self, other, out);
        };
      }
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_OPERATOR_FUNCTOR(
    aten::special_multigammaln,
    aten_special_multigammaln,
    [](Node* n) -> SROperator {
      if (n->matches(torch::schema(
              "aten::special_multigammaln(Tensor self, int p) -> Tensor"))) {
        return [](ProcessedNode* p_node) {
          const auto& self = p_node->Input(0).toTensor();
          const auto p = p_node->Input(1).toInt();
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::native::special_multigammaln(self, p);
            return;
          }
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          at::native::special_multigammaln_out(self, p, out);
        };
      }
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_OPERATOR_FUNCTOR(
    aten::linalg_cross,
    aten_linalg_cross,
    [](Node* n) -> SROperator {
      if (n->matches(torch::schema(
              "aten::linalg_cross(Tensor self, Tensor other, *, int dim=-1) -> Tensor"))) {
        return [](ProcessedNode* p_node) {
          const auto& self = p_node->Input(0).toTensor();
          const auto& other = p_node->Input(1).toTensor();
          const auto dim = p_node->Input(2).toInt();
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::cpu::linalg_cross(self, other, dim);
            return;
          }
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          at::cpu::linalg_cross_out(out, self, other, dim);
        };
      }
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_OPERATOR_FUNCTOR(
    aten::linalg_det,
    aten_linalg_det,
    [](Node* n) -> SROperator {
      if (n->matches(torch::schema("aten::linalg_det(Tensor A) -> Tensor"))) {
        return [](ProcessedNode* p_node) {
          const auto& A = p_node->Input(0).toTensor();
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::native::linalg_det(A);
            return;
          }
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          at::native::linalg_det_out(A, out);
        };
      }
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_OPERATOR_FUNCTOR(
    aten::linalg_matmul,
    aten_linalg_matmul,
    [](Node* n) -> SROperator {
      if (n->matches(torch::schema(
              "aten::linalg_matmul(Tensor self, Tensor other) -> Tensor"))) {
        return [](ProcessedNode* p_node) {
          const auto& self = p_node->Input(0).toTensor();
          const auto& other = p_node->Input(1).toTensor();
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::native::linalg_matmul(self, other);
            return;
          }
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          at::native::linalg_matmul_out(self, other, out);
        };
      }
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_OPERATOR_FUNCTOR(
    aten::linalg_eigvals,
    aten_linalg_eigvals,
    [](Node* n) -> SROperator {
      if (n->matches(
              torch::schema("aten::linalg_eigvals(Tensor self) -> Tensor"))) {
        return [](ProcessedNode* p_node) {
          const auto& self = p_node->Input(0).toTensor();
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::native::linalg_eigvals(self);
            return;
          }
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          at::native::linalg_eigvals_out(self, out);
        };
      }
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_OPERATOR_FUNCTOR(
    aten::linalg_inv,
    aten_linalg_inv,
    [](Node* n) -> SROperator {
      if (n->matches(torch::schema("aten::linalg_inv(Tensor A) -> Tensor"))) {
        return [](ProcessedNode* p_node) {
          const auto& A = p_node->Input(0).toTensor();
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::native::linalg_inv(A);
            return;
          }
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          at::native::linalg_inv_out(A, out);
        };
      }
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_OPERATOR_FUNCTOR(
    aten::inverse,
    aten_inverse,
    [](Node* n) -> SROperator {
      if (n->matches(torch::schema("aten::inverse(Tensor self) -> Tensor"))) {
        return [](ProcessedNode* p_node) {
          const auto& self = p_node->Input(0).toTensor();
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::native::inverse(self);
            return;
          }
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          at::native::inverse_out(self, out);
        };
      }
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_OPERATOR_FUNCTOR(aten::inner, aten_inner, [](Node* n) -> SROperator {
  if (n->matches(
          torch::schema("aten::inner(Tensor self, Tensor other) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      const auto& other = p_node->Input(1).toTensor();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::native::inner(self, other);
        return;
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::native::inner_out(self, other, out);
    };
  }
  LogAndDumpSchema(n);
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(aten::outer, aten_outer, [](Node* n) -> SROperator {
  if (n->matches(
          torch::schema("aten::outer(Tensor self, Tensor vec2) -> Tensor"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      const auto& vec2 = p_node->Input(1).toTensor();
      if (p_node->Output(0).isNone()) {
        p_node->Output(0) = at::native::outer(self, vec2);
        return;
      }
      auto& out = p_node->Output(0).toTensor();
      fastResizeToZero(out);
      at::native::outer_out(self, vec2, out);
    };
  }
  LogAndDumpSchema(n);
  return nullptr;
});

REGISTER_OPERATOR_FUNCTOR(
    aten::linalg_cond,
    aten_linalg_cond,
    [](Node* n) -> SROperator {
      if (n->matches(torch::schema(
              "aten::linalg_cond(Tensor self, Scalar? p=None) -> Tensor"))) {
        return [](ProcessedNode* p_node) {
          const auto& self = p_node->Input(0).toTensor();
          const auto p = p_node->Input(1).toOptional<at::Scalar>();
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::native::linalg_cond(self, p);
            return;
          }
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          at::native::linalg_cond_out(self, p, out);
        };
      }
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_OPERATOR_FUNCTOR(
    aten::linalg_solve,
    aten_linalg_solve,
    [](Node* n) -> SROperator {
      if (n->matches(torch::schema(
              "aten::linalg_solve(Tensor A, Tensor B, *, bool left=True) -> Tensor"))) {
        return [](ProcessedNode* p_node) {
          const auto& A = p_node->Input(0).toTensor();
          const auto& B = p_node->Input(1).toTensor();
          const auto left = p_node->Input(2).toBool();
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::native::linalg_solve(A, B, left);
            return;
          }
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          at::native::linalg_solve_out(A, B, left, out);
        };
      }
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_OPERATOR_FUNCTOR(
    aten::linalg_tensorinv,
    aten_linalg_tensorinv,
    [](Node* n) -> SROperator {
      if (n->matches(torch::schema(
              "aten::linalg_tensorinv(Tensor self, int ind=2) -> Tensor"))) {
        return [](ProcessedNode* p_node) {
          const auto& self = p_node->Input(0).toTensor();
          const auto ind = p_node->Input(1).toInt();
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::native::linalg_tensorinv(self, ind);
            return;
          }
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          at::native::linalg_tensorinv_out(self, ind, out);
        };
      }
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_OPERATOR_FUNCTOR(
    aten::linalg_matrix_power,
    aten_linalg_matrix_power,
    [](Node* n) -> SROperator {
      if (n->matches(torch::schema(
              "aten::linalg_matrix_power(Tensor self, int n) -> Tensor"))) {
        return [](ProcessedNode* p_node) {
          const auto& self = p_node->Input(0).toTensor();
          const auto n = p_node->Input(1).toInt();
          if (p_node->Output(0).isNone()) {
            p_node->Output(0) = at::native::linalg_matrix_power(self, n);
            return;
          }
          auto& out = p_node->Output(0).toTensor();
          fastResizeToZero(out);
          at::native::linalg_matrix_power_out(self, n, out);
        };
      }
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_NATIVE_OPERATOR_FUNCTOR(
    aten::view_as_real,
    aten_view_as_real,
    [](Node* n) -> SROperator {
      if (n->matches(torch::schema(
              "aten::view_as_real(Tensor(a) self) -> Tensor(a)"))) {
        return [](ProcessedNode* p_node) {
          const auto& self = p_node->Input(0).toTensor();
          p_node->Output(0) = at::native::view_as_real(self);
        };
      }
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_NATIVE_OPERATOR_FUNCTOR(
    aten::view_as_complex,
    aten_view_as_complex,
    [](Node* n) -> SROperator {
      if (n->matches(torch::schema(
              "aten::view_as_complex(Tensor(a) self) -> Tensor(a)"))) {
        return [](ProcessedNode* p_node) {
          const auto& self = p_node->Input(0).toTensor();
          p_node->Output(0) = at::native::view_as_complex(self);
        };
      }
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_NATIVE_OPERATOR_FUNCTOR(
    aten::real,
    aten_real,
    [](Node* n) -> SROperator {
      if (n->matches(
              torch::schema("aten::real(Tensor(a) self) -> Tensor(a)"))) {
        return [](ProcessedNode* p_node) {
          const auto& self = p_node->Input(0).toTensor();
          p_node->Output(0) = at::native::real(self);
        };
      }
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_NATIVE_OPERATOR_FUNCTOR(
    aten::imag,
    aten_imag,
    [](Node* n) -> SROperator {
      if (n->matches(
              torch::schema("aten::imag(Tensor(a) self) -> Tensor(a)"))) {
        return [](ProcessedNode* p_node) {
          const auto& self = p_node->Input(0).toTensor();
          p_node->Output(0) = at::native::imag(self);
        };
      }
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_NATIVE_OPERATOR_FUNCTOR(
    aten::_conj,
    aten__conj,
    [](Node* n) -> SROperator {
      if (n->matches(
              torch::schema("aten::_conj(Tensor(a) self) -> Tensor(a)"))) {
        return [](ProcessedNode* p_node) {
          const auto& self = p_node->Input(0).toTensor();
          p_node->Output(0) = at::native::_conj(self);
        };
      }
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_NATIVE_OPERATOR_FUNCTOR(
    aten::conj,
    aten_conj,
    [](Node* n) -> SROperator {
      if (n->matches(
              torch::schema("aten::conj(Tensor(a) self) -> Tensor(a)"))) {
        return [](ProcessedNode* p_node) {
          const auto& self = p_node->Input(0).toTensor();
          p_node->Output(0) = at::native::conj(self);
        };
      }
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_NATIVE_OPERATOR_FUNCTOR(
    aten::resolve_conj,
    aten_resolve_conj,
    [](Node* n) -> SROperator {
      if (n->matches(torch::schema(
              "aten::resolve_conj(Tensor(a) self) -> Tensor(a)"))) {
        return [](ProcessedNode* p_node) {
          const auto& self = p_node->Input(0).toTensor();
          p_node->Output(0) = at::native::resolve_conj(self);
        };
      }
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_NATIVE_OPERATOR_FUNCTOR(
    aten::resolve_neg,
    aten_resolve_neg,
    [](Node* n) -> SROperator {
      if (n->matches(torch::schema(
              "aten::resolve_neg(Tensor(a) self) -> Tensor(a)"))) {
        return [](ProcessedNode* p_node) {
          const auto& self = p_node->Input(0).toTensor();
          p_node->Output(0) = at::native::resolve_neg(self);
        };
      }
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_NATIVE_OPERATOR_FUNCTOR(
    aten::_neg_view,
    aten__neg_view,
    [](Node* n) -> SROperator {
      if (n->matches(
              torch::schema("aten::_neg_view(Tensor(a) self) -> Tensor(a)"))) {
        return [](ProcessedNode* p_node) {
          const auto& self = p_node->Input(0).toTensor();
          p_node->Output(0) = at::native::_neg_view(self);
        };
      }
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_NATIVE_OPERATOR_FUNCTOR(aten::diagonal, aten_diagonal, [](Node* n) -> SROperator {
  if (n->matches(torch::schema(
          "aten::diagonal(Tensor(a) self, int offset=0, int dim1=0, int dim2=1) -> Tensor(a)"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      const auto offset = p_node->Input(1).toInt();
      const auto dim1 = p_node->Input(2).toInt();
      const auto dim2 = p_node->Input(3).toInt();
      p_node->Output(0) = at::native::diagonal(self, offset, dim1, dim2);
    };
  }
  LogAndDumpSchema(n);
  return nullptr;
});

REGISTER_NATIVE_OPERATOR_FUNCTOR(
    aten::linalg_diagonal,
    aten_linalg_diagonal,
    [](Node* n) -> SROperator {
      if (n->matches(torch::schema(
              "aten::linalg_diagonal(Tensor(a) A, *, int offset=0, int dim1=-2, int dim2=-1) -> Tensor(a)"))) {
        return [](ProcessedNode* p_node) {
          const auto& A = p_node->Input(0).toTensor();
          const auto offset = p_node->Input(1).toInt();
          const auto dim1 = p_node->Input(2).toInt();
          const auto dim2 = p_node->Input(3).toInt();
          p_node->Output(0) =
              at::native::linalg_diagonal(A, offset, dim1, dim2);
        };
      }
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_NATIVE_OPERATOR_FUNCTOR(aten::movedim, aten_movedim, [](Node* n) -> SROperator {
  if (n->matches(torch::schema(
          "aten::movedim.int(Tensor(a) self, int source, int destination) -> Tensor(a)"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      const auto source = p_node->Input(1).toInt();
      const auto destination = p_node->Input(2).toInt();
      p_node->Output(0) = at::native::movedim(self, source, destination);
    };
  }
  LogAndDumpSchema(n);
  return nullptr;
});

REGISTER_NATIVE_OPERATOR_FUNCTOR(aten::moveaxis, aten_moveaxis, [](Node* n) -> SROperator {
  if (n->matches(torch::schema(
          "aten::moveaxis.int(Tensor(a) self, int source, int destination) -> Tensor(a)"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      const auto source = p_node->Input(1).toInt();
      const auto destination = p_node->Input(2).toInt();
      p_node->Output(0) = at::native::moveaxis(self, source, destination);
    };
  }
  LogAndDumpSchema(n);
  return nullptr;
});

REGISTER_NATIVE_OPERATOR_FUNCTOR(
    aten::numpy_T,
    aten_numpy_T,
    [](Node* n) -> SROperator {
      if (n->matches(
              torch::schema("aten::numpy_T(Tensor(a) self) -> Tensor(a)"))) {
        return [](ProcessedNode* p_node) {
          const auto& self = p_node->Input(0).toTensor();
          p_node->Output(0) = at::native::numpy_T(self);
        };
      }
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_NATIVE_OPERATOR_FUNCTOR(
    aten::matrix_H,
    aten_matrix_H,
    [](Node* n) -> SROperator {
      if (n->matches(
              torch::schema("aten::matrix_H(Tensor(a) self) -> Tensor(a)"))) {
        return [](ProcessedNode* p_node) {
          const auto& self = p_node->Input(0).toTensor();
          p_node->Output(0) = at::native::matrix_H(self);
        };
      }
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_NATIVE_OPERATOR_FUNCTOR(aten::mT, aten_mT, [](Node* n) -> SROperator {
  if (n->matches(torch::schema("aten::mT(Tensor(a) self) -> Tensor(a)"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      p_node->Output(0) = at::native::mT(self);
    };
  }
  LogAndDumpSchema(n);
  return nullptr;
});

REGISTER_NATIVE_OPERATOR_FUNCTOR(aten::mH, aten_mH, [](Node* n) -> SROperator {
  if (n->matches(torch::schema("aten::mH(Tensor(a) self) -> Tensor(a)"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      p_node->Output(0) = at::native::mH(self);
    };
  }
  LogAndDumpSchema(n);
  return nullptr;
});

REGISTER_NATIVE_OPERATOR_FUNCTOR(
    aten::adjoint,
    aten_adjoint,
    [](Node* n) -> SROperator {
      if (n->matches(
              torch::schema("aten::adjoint(Tensor(a) self) -> Tensor(a)"))) {
        return [](ProcessedNode* p_node) {
          const auto& self = p_node->Input(0).toTensor();
          p_node->Output(0) = at::native::adjoint(self);
        };
      }
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_NATIVE_OPERATOR_FUNCTOR(
    aten::ravel,
    aten_ravel,
    [](Node* n) -> SROperator {
      if (n->matches(
              torch::schema("aten::ravel(Tensor(a) self) -> Tensor(a)"))) {
        return [](ProcessedNode* p_node) {
          const auto& self = p_node->Input(0).toTensor();
          p_node->Output(0) = at::native::ravel(self);
        };
      }
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_NATIVE_OPERATOR_FUNCTOR(aten::t, aten_t, [](Node* n) -> SROperator {
  if (n->matches(torch::schema("aten::t(Tensor(a) self) -> Tensor(a)"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      p_node->Output(0) = at::native::t(self);
    };
  }
  LogAndDumpSchema(n);
  return nullptr;
});

REGISTER_NATIVE_OPERATOR_FUNCTOR(
    aten::unsqueeze,
    aten_unsqueeze,
    [](Node* n) -> SROperator {
      if (n->matches(torch::schema(
              "aten::unsqueeze(Tensor(a) self, int dim) -> Tensor(a)"))) {
        return [](ProcessedNode* p_node) {
          const auto& self = p_node->Input(0).toTensor();
          const auto dim = p_node->Input(1).toInt();
          p_node->Output(0) = at::native::unsqueeze(self, dim);
        };
      }
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_NATIVE_OPERATOR_FUNCTOR(
    aten::view_as,
    aten_view_as,
    [](Node* n) -> SROperator {
      if (n->matches(torch::schema(
              "aten::view_as(Tensor(a) self, Tensor other) -> Tensor(a)"))) {
        return [](ProcessedNode* p_node) {
          const auto& self = p_node->Input(0).toTensor();
          const auto& other = p_node->Input(1).toTensor();
          p_node->Output(0) = at::native::view_as(self, other);
        };
      }
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_NATIVE_OPERATOR_FUNCTOR(
    aten::positive,
    aten_positive,
    [](Node* n) -> SROperator {
      if (n->matches(
              torch::schema("aten::positive(Tensor(a) self) -> Tensor(a)"))) {
        return [](ProcessedNode* p_node) {
          const auto& self = p_node->Input(0).toTensor();
          p_node->Output(0) = at::native::positive(self);
        };
      }
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_NATIVE_OPERATOR_FUNCTOR(
    aten::_autocast_to_reduced_precision,
    aten__autocast_to_reduced_precision,
    [](Node* n) -> SROperator {
      if (n->matches(torch::schema(
              "aten::_autocast_to_reduced_precision(Tensor(a) self, bool cuda_enabled, bool cpu_enabled, ScalarType cuda_dtype, ScalarType cpu_dtype) -> Tensor(a)"))) {
        return [](ProcessedNode* p_node) {
          const auto& self = p_node->Input(0).toTensor();
          const auto cuda_enabled = p_node->Input(1).toBool();
          const auto cpu_enabled = p_node->Input(2).toBool();
          const auto cuda_dtype = p_node->Input(3).toScalarType();
          const auto cpu_dtype = p_node->Input(4).toScalarType();
          p_node->Output(0) = at::native::_autocast_to_reduced_precision(
              self, cuda_enabled, cpu_enabled, cuda_dtype, cpu_dtype);
        };
      }
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_NATIVE_OPERATOR_FUNCTOR(
    aten::_autocast_to_full_precision,
    aten__autocast_to_full_precision,
    [](Node* n) -> SROperator {
      if (n->matches(torch::schema(
              "aten::_autocast_to_full_precision(Tensor(a) self, bool cuda_enabled, bool cpu_enabled) -> Tensor(a)"))) {
        return [](ProcessedNode* p_node) {
          const auto& self = p_node->Input(0).toTensor();
          const auto cuda_enabled = p_node->Input(1).toBool();
          const auto cpu_enabled = p_node->Input(2).toBool();
          p_node->Output(0) = at::native::_autocast_to_full_precision(
              self, cuda_enabled, cpu_enabled);
        };
      }
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_NATIVE_OPERATOR_FUNCTOR(
    aten::swapaxes,
    aten_swapaxes,
    [](Node* n) -> SROperator {
      if (n->matches(torch::schema(
              "aten::swapaxes(Tensor(a) self, int axis0, int axis1) -> Tensor(a)"))) {
        return [](ProcessedNode* p_node) {
          const auto& self = p_node->Input(0).toTensor();
          const auto axis0 = p_node->Input(1).toInt();
          const auto axis1 = p_node->Input(2).toInt();
          p_node->Output(0) = at::native::swapaxes(self, axis0, axis1);
        };
      }
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_NATIVE_OPERATOR_FUNCTOR(
    aten::swapdims,
    aten_swapdims,
    [](Node* n) -> SROperator {
      if (n->matches(torch::schema(
              "aten::swapdims(Tensor(a) self, int dim0, int dim1) -> Tensor(a)"))) {
        return [](ProcessedNode* p_node) {
          const auto& self = p_node->Input(0).toTensor();
          const auto dim0 = p_node->Input(1).toInt();
          const auto dim1 = p_node->Input(2).toInt();
          p_node->Output(0) = at::native::swapdims(self, dim0, dim1);
        };
      }
      LogAndDumpSchema(n);
      return nullptr;
    });

REGISTER_NATIVE_OPERATOR_FUNCTOR(aten::unfold, aten_unfold, [](Node* n) -> SROperator {
  if (n->matches(torch::schema(
          "aten::unfold(Tensor(a) self, int dimension, int size, int step) -> Tensor(a)"))) {
    return [](ProcessedNode* p_node) {
      const auto& self = p_node->Input(0).toTensor();
      const auto dimension = p_node->Input(1).toInt();
      const auto size = p_node->Input(2).toInt();
      const auto step = p_node->Input(3).toInt();
      p_node->Output(0) = at::native::unfold(self, dimension, size, step);
    };
  }
  LogAndDumpSchema(n);
  return nullptr;
});

REGISTER_NATIVE_OPERATOR_FUNCTOR(
    aten::alias,
    aten_alias,
    [](Node* n) -> SROperator {
      if (n->matches(
              torch::schema("aten::alias(Tensor(a) self) -> Tensor(a)"))) {
        return [](ProcessedNode* p_node) {
          const auto& self = p_node->Input(0).toTensor();
          p_node->Output(0) = at::native::alias(self);
        };
      }
      LogAndDumpSchema(n);
      return nullptr;
    });

} // namespace jit
} // namespace torch
