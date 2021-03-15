#include <pybind11/functional.h>
#include <pybind11/operators.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/jit/tensorexpr/codegen.h>
#ifdef USE_CUDA
#include <torch/csrc/jit/tensorexpr/cuda_codegen.h>
#endif
#include <torch/csrc/jit/tensorexpr/ir_printer.h>
#include <torch/csrc/jit/tensorexpr/ir_simplifier.h>
#include <torch/csrc/jit/tensorexpr/llvm_codegen.h>
#include <torch/csrc/jit/tensorexpr/loopnest.h>
#include <torch/csrc/jit/tensorexpr/reduction.h>

namespace torch {
namespace jit {
using namespace torch::jit::tensorexpr;

void initTensorExprBindings(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();

  // Tensor Expr Classes
  auto te = m.def_submodule("_te");
  py::class_<KernelScope>(te, "KernelScope").def(py::init<>());

  auto dtype_class = py::class_<Dtype>(te, "Dtype");

#define DTYPE_SINGLETON_ACCESSOR(ctype, name) \
  dtype_class.def_property_readonly_static(   \
      #name, [](py::object) { return k##name; }); // NOLINT
  AT_FORALL_SCALAR_TYPES_AND2(Bool, Half, DTYPE_SINGLETON_ACCESSOR)
#undef DTYPE_SINGLETON_ACCESSOR

  auto expr_handle_class =
      py::class_<ExprHandle>(te, "ExprHandle")
          .def(py::self + py::self)
          .def(py::self * py::self)
          .def(py::self - py::self)
          .def(py::self / py::self)
          .def(py::self % py::self)
          .def(py::self == py::self)
          .def(py::self != py::self)
          .def(py::self > py::self)
          .def(py::self >= py::self)
          .def(py::self < py::self)
          .def(py::self <= py::self)
          .def(py::self & py::self)
          .def(py::self | py::self)
          .def(py::self ^ py::self)
          .def(py::self << py::self)
          .def(py::self >> py::self)
          .def("sin", [](const ExprHandle& self) { return sin(self); })
          .def("cos", [](const ExprHandle& self) { return cos(self); })
          .def("tan", [](const ExprHandle& self) { return tan(self); })
          .def("asin", [](const ExprHandle& self) { return asin(self); })
          .def("acos", [](const ExprHandle& self) { return acos(self); })
          .def("atan", [](const ExprHandle& self) { return atan(self); })
          .def("sinh", [](const ExprHandle& self) { return sinh(self); })
          .def("cosh", [](const ExprHandle& self) { return cosh(self); })
          .def("tanh", [](const ExprHandle& self) { return tanh(self); })
          .def("sigmoid", [](const ExprHandle& self) { return sigmoid(self); })
          .def("exp", [](const ExprHandle& self) { return exp(self); })
          .def("expm1", [](const ExprHandle& self) { return expm1(self); })
          .def(
              "abs",
              [](const ExprHandle& self) { return tensorexpr::abs(self); })
          .def("log", [](const ExprHandle& self) { return log(self); })
          .def(
              "fast_log", [](const ExprHandle& self) { return fast_log(self); })
          .def("log2", [](const ExprHandle& self) { return log2(self); })
          .def("log10", [](const ExprHandle& self) { return log10(self); })
          .def("log1p", [](const ExprHandle& self) { return log1p(self); })
          .def("erf", [](const ExprHandle& self) { return erf(self); })
          .def("erfc", [](const ExprHandle& self) { return erfc(self); })
          .def(
              "sqrt",
              [](const ExprHandle& self) { return tensorexpr::sqrt(self); })
          .def("rsqrt", [](const ExprHandle& self) { return rsqrt(self); })
          .def("ceil", [](const ExprHandle& self) { return ceil(self); })
          .def("floor", [](const ExprHandle& self) { return floor(self); })
          .def("round", [](const ExprHandle& self) { return round(self); })
          .def("trunc", [](const ExprHandle& self) { return trunc(self); })
          .def("frac", [](const ExprHandle& self) { return frac(self); })
          .def("lgamma", [](const ExprHandle& self) { return lgamma(self); })
          .def("isnan", [](const ExprHandle& self) { return isnan(self); });
  te.def(
      "ifThenElse",
      [](const ExprHandle& c, const ExprHandle& t, const ExprHandle& f) {
        return ifThenElse(c, t, f);
      });
  te.def("atan2", [](const ExprHandle& v1, const ExprHandle& v2) {
    return atan2(v1, v2);
  });
  te.def("pow", [](const ExprHandle& v1, const ExprHandle& v2) {
    return pow(v1, v2);
  });
  te.def("fmod", [](const ExprHandle& v1, const ExprHandle& v2) {
    return fmod(v1, v2);
  });
  te.def("remainder", [](const ExprHandle& v1, const ExprHandle& v2) {
    return remainder(v1, v2);
  });

#define EXPRHANDLE_CTOR(ctype, name) \
  expr_handle_class.def_static(#ctype, [](ctype v) { return ExprHandle(v); });
  AT_FORALL_SCALAR_TYPES_AND2(Bool, Half, EXPRHANDLE_CTOR)
#undef EXPRHANDLE_CTOR

  py::class_<VarHandle, ExprHandle>(te, "VarHandle")
      .def(py::init<const std::string&, Dtype>());
  py::class_<BufHandle, ExprHandle>( // NOLINT
      te,
      "BufHandle")
      .def(
          py::init<const std::string&, const std::vector<ExprHandle>&, Dtype>())
      .def("load", [](BufHandle& self, const std::vector<ExprHandle>& v) {
        return Load::make(self, v);
      });

  py::class_<Placeholder>(te, "Placeholder")
      .def(py::init<
           const std::string&,
           const Dtype&,
           std::vector<ExprHandle>&>())
      .def(
          "load",
          [](Placeholder& self, const std::vector<ExprHandle>& v) {
            return self.load(v);
          })
      .def("buf", [](Placeholder& self) { return BufHandle(self.data()); });
  py::class_<Tensor, std::unique_ptr<Tensor, py::nodelete>>(te, "Tensor")
      .def(py::init(
          [](BufHandle& b, Stmt* s) { return new Tensor(b.node(), s); }))
      .def(
          "load",
          [](Tensor& self, const std::vector<ExprHandle>& v) {
            return self.call(v);
          })
      .def("buf", [](Tensor& self) { return BufHandle(self.buf()); })
      .def("stmt", &Tensor::stmt, py::return_value_policy::reference);
  py::class_<Cast>(te, "Cast").def_static("make", &Cast::make);

  py::class_<DimArg>(te, "DimArg")
      .def(py::init<const ExprHandle&>())
      .def(py::init<const ExprHandle&, const std::string&>());

  te.def(
      "Compute",
      [](const std::string& func_name,
         const std::vector<DimArg>& dim_args,
         py::function func) {
        if (dim_args.size() == 1) {
          return Compute(func_name, dim_args, [&func](const VarHandle& a) {
            return py::cast<ExprHandle>(func(a));
          });
        } else if (dim_args.size() == 2) {
          return Compute(
              func_name,
              dim_args,
              [&func](const VarHandle& a, const VarHandle& b) {
                return py::cast<ExprHandle>(func(a, b));
              });
        } else if (dim_args.size() == 3) {
          return Compute(
              func_name,
              dim_args,
              [&func](
                  const VarHandle& a, const VarHandle& b, const VarHandle& c) {
                return py::cast<ExprHandle>(func(a, b, c));
              });
        } else if (dim_args.size() == 4) {
          return Compute(
              func_name,
              dim_args,
              [&func](
                  const VarHandle& a,
                  const VarHandle& b,
                  const VarHandle& c,
                  const VarHandle& d) {
                return py::cast<ExprHandle>(func(a, b, c, d));
              });
        } else {
          throw std::runtime_error("Too many args");
        }
      },
      py::return_value_policy::reference);
  py::class_<Reducer>(te, "Reducer")
      .def(py::init<
           ExprHandle,
           std::function<ExprHandle(ExprHandle, ExprHandle)>>());

  py::class_<Sum, Reducer>(te, "Sum").def(py::init<>());
  py::class_<Maximum, Reducer>(te, "Maximum").def(py::init<Dtype>());
  te.def(
      "Reduce",
      [](const std::string& func_name,
         const std::vector<DimArg>& dim_args,
         const Reducer& reducer,
         Tensor* buffer,
         const std::vector<DimArg>& reduce_args) {
        return Reduce(func_name, dim_args, reducer, buffer, reduce_args);
      },
      py::return_value_policy::reference);
  te.def(
      "Reduce",
      [](const std::string& func_name,
         const std::vector<DimArg>& dim_args,
         const Reducer& reducer,
         const Placeholder& buffer,
         const std::vector<DimArg>& reduce_args) {
        return Reduce(func_name, dim_args, reducer, buffer, reduce_args);
      },
      py::return_value_policy::reference);

  te.def(
      "Reduce",
      [](const std::string& func_name,
         const std::vector<DimArg>& dim_args,
         const Reducer& reducer,
         const BufHandle& buffer,
         const std::vector<DimArg>& reduce_args) {
        return Reduce(func_name, dim_args, reducer, buffer, reduce_args);
      },
      py::return_value_policy::reference);

  py::class_<Stmt, std::unique_ptr<Stmt, py::nodelete>>(te, "Stmt")
      .def(py::init([](const std::vector<Stmt*>& stmts) {
        return tensorexpr::Block::make(stmts);
      }))
      .def("__str__", [](const Stmt& self) {
        std::stringstream ss;
        ss << self;
        return ss.str();
      });
  py::class_<For, Stmt, std::unique_ptr<For, py::nodelete>>(te, "For")
      .def(
          "index_var",
          [](const For& self) { return VarHandle(self.var()); },
          py::return_value_policy::reference)
      .def("body", &For::body, py::return_value_policy::reference);

  py::class_<
      tensorexpr::Block,
      Stmt,
      std::unique_ptr<tensorexpr::Block, py::nodelete>>(te, "Block")
      .def(
          "stmts",
          &tensorexpr::Block::stmts,
          py::return_value_policy::reference);
  py::class_<ExternalCall, Stmt, std::unique_ptr<ExternalCall, py::nodelete>>(
      te, "ExternalCall")
      .def(py::init(&ExternalCall::make), py::return_value_policy::reference);

  py::class_<LoopNest>(te, "LoopNest")
      .def(py::init<const std::vector<Tensor*>&>())
      .def(py::init([](Stmt* s, const std::vector<BufHandle>& bufs) {
        std::unordered_set<const Buf*> buf_nodes;
        for (const auto& buf : bufs) {
          buf_nodes.insert(buf.node());
        }
        return std::make_unique<LoopNest>(s, buf_nodes);
      }))
      .def("vectorize_inner_loops", &LoopNest::vectorizeInnerLoops)
      .def("prepare_for_codegen", &LoopNest::prepareForCodegen)
      .def(
          "get_loop_body_for",
          [](const LoopNest& self, Tensor* t) {
            return self.getLoopBodyFor(t);
          },
          py::return_value_policy::reference)
      .def(
          "get_loops_for",
          [](const LoopNest& self, Tensor* t) {
            return self.getLoopStmtsFor(t);
          },
          py::return_value_policy::reference)
      .def(
          "split_with_tail",
          [](const LoopNest& self, For* f, int factor) {
            For *outer = nullptr, *inner = nullptr, *tail = nullptr;
            self.splitWithTail(f, factor, &outer, &inner, &tail);
            return std::make_tuple(outer, inner, tail);
          },
          py::return_value_policy::reference)
      .def(
          "split_with_mask",
          [](const LoopNest& self, For* f, int factor) {
            For *outer = nullptr, *inner = nullptr;
            self.splitWithMask(f, factor, &outer, &inner);
            return std::make_tuple(outer, inner);
          },
          py::return_value_policy::reference)
      .def(
          "unroll",
          [](const LoopNest& self, For* f) {
            Stmt* unrolled = nullptr;
            self.unroll(f, &unrolled);
            return unrolled;
          },
          py::return_value_policy::reference)
      .def(
          "vectorize",
          [](const LoopNest& self, For* f) { self.vectorize(f); },
          py::return_value_policy::reference)
      .def(
          "compute_inline",
          [](LoopNest& self, Stmt* s) { self.computeInline(s); },
          py::return_value_policy::reference)
      .def(
          "compute_inline",
          [](LoopNest& self, const BufHandle& b) {
            self.computeInline(b.node());
          },
          py::return_value_policy::reference)
      .def(
          "rfactor",
          [](LoopNest& self, const Stmt& s, const VarHandle& v) {
            auto st = dynamic_cast<const Store*>(&s);
            if (!st) {
              return;
            }
            auto r = st->value();
            self.rfactor(r, v.node());
          },
          py::return_value_policy::reference)
      .def(
          "rfactor",
          [](LoopNest& self,
             const Stmt& s,
             const VarHandle& v,
             tensorexpr::Block& ins_point) {
            auto st = dynamic_cast<const Store*>(&s);
            if (!st) {
              return;
            }
            auto r = st->value();
            self.rfactor(r, v.node(), &ins_point);
          },
          py::return_value_policy::reference)
      .def(
          "flatten",
          [](const LoopNest& self, const std::vector<For*>& loops) {
            For* flattened = nullptr;
            self.flatten(loops, &flattened);
            return flattened;
          },
          py::return_value_policy::reference)
      .def(
          "reorder", &LoopNest::reorderAxis, py::return_value_policy::reference)
      .def("simplify", &LoopNest::simplify, py::return_value_policy::reference)
      .def(
          "set_GPU_block_index",
          &LoopNest::setGPUBlockIndex,
          py::return_value_policy::reference)
      .def(
          "set_GPU_thread_index",
          &LoopNest::setGPUThreadIndex,
          py::return_value_policy::reference)
      .def(
          "__str__",
          [](const LoopNest& self) {
            std::stringstream ss;
            ss << *self.root_stmt();
            return ss.str();
          })
      .def(
          "root_stmt",
          &LoopNest::root_stmt,
          py::return_value_policy::reference);

  te.def(
      "simplify",
      [](Stmt* stmt) { return IRSimplifier::simplify(stmt); },
      py::return_value_policy::reference);

  py::class_<CodeGen>(te, "CodeGen")
      .def(
          "call",
          [](CodeGen& self, const std::vector<at::Tensor>& values) {
            std::vector<CodeGen::CallArg> value_ptrs;
            value_ptrs.reserve(values.size());
            for (const auto& value : values) {
              value_ptrs.emplace_back(CodeGen::CallArg(value.data_ptr()));
            }
            self.call(value_ptrs);
          })
      .def(
          "getCodeText",
          [](CodeGen& self, const std::string& attr = "") {
            return self.getCodeText(attr);
          },
          py::arg("attr") = "");
  py::class_<SimpleIREvaluator, CodeGen>(te, "SimpleIREvaluator"); // NOLINT
#ifdef TORCH_ENABLE_LLVM
  py::class_<LLVMCodeGen, CodeGen>(te, "LLVMCodeGen"); // NOLINT
#endif

  py::class_<CodeGen::BufferArg>(te, "BufferArg")
      .def(py::init<const Placeholder&>())
      .def(py::init<Tensor*>())
      .def(py::init<const VarHandle&>())
      .def(py::init<const BufHandle&>());

  te.def(
      "construct_codegen",
      [](const std::string& name,
         Stmt* stmt,
         const std::vector<CodeGen::BufferArg>& args) {
        CodeGen* cg = nullptr;
        if (name == "llvm") {
#ifdef TORCH_ENABLE_LLVM
          cg = new LLVMCodeGen(stmt, args);
#else
          throw std::runtime_error("PyTorch not compiled with LLVM support!");
#endif
        } else if (name == "cuda") {
#ifdef USE_CUDA
          cg = new CudaCodeGen(stmt, args);
#else
          throw std::runtime_error("PyTorch not compiled with CUDA support!");
#endif
        } else {
          cg = new SimpleIREvaluator(stmt, args);
        }
        return cg;
      });
}
} // namespace jit
} // namespace torch
