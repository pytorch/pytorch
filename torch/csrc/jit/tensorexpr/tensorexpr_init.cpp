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

void initTensorExprBindings(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();

  // Tensor Expr Classes
  auto te = m.def_submodule("_te");
  py::class_<tensorexpr::KernelScope>(te, "KernelScope").def(py::init<>());

  auto dtype_class = py::class_<tensorexpr::Dtype>(te, "Dtype");

#define DTYPE_SINGLETON_ACCESSOR(ctype, name) \
  dtype_class.def_property_readonly_static(   \
      #name, [](py::object) { return tensorexpr::k##name; });
  AT_FORALL_SCALAR_TYPES_AND2(Bool, Half, DTYPE_SINGLETON_ACCESSOR)
#undef DTYPE_SINGLETON_ACCESSOR

  auto expr_handle_class =
      py::class_<tensorexpr::ExprHandle>(te, "ExprHandle")
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
          .def(
              "sin",
              [](const tensorexpr::ExprHandle& self) {
                return tensorexpr::sin(self);
              })
          .def(
              "cos",
              [](const tensorexpr::ExprHandle& self) {
                return tensorexpr::cos(self);
              })
          .def(
              "tan",
              [](const tensorexpr::ExprHandle& self) {
                return tensorexpr::tan(self);
              })
          .def(
              "asin",
              [](const tensorexpr::ExprHandle& self) {
                return tensorexpr::asin(self);
              })
          .def(
              "acos",
              [](const tensorexpr::ExprHandle& self) {
                return tensorexpr::acos(self);
              })
          .def(
              "atan",
              [](const tensorexpr::ExprHandle& self) {
                return tensorexpr::atan(self);
              })
          .def(
              "sinh",
              [](const tensorexpr::ExprHandle& self) {
                return tensorexpr::sinh(self);
              })
          .def(
              "cosh",
              [](const tensorexpr::ExprHandle& self) {
                return tensorexpr::cosh(self);
              })
          .def(
              "tanh",
              [](const tensorexpr::ExprHandle& self) {
                return tensorexpr::tanh(self);
              })
          .def(
              "sigmoid",
              [](const tensorexpr::ExprHandle& self) {
                return tensorexpr::sigmoid(self);
              })
          .def(
              "exp",
              [](const tensorexpr::ExprHandle& self) {
                return tensorexpr::exp(self);
              })
          .def(
              "expm1",
              [](const tensorexpr::ExprHandle& self) {
                return tensorexpr::expm1(self);
              })
          .def(
              "abs",
              [](const tensorexpr::ExprHandle& self) {
                return tensorexpr::abs(self);
              })
          .def(
              "log",
              [](const tensorexpr::ExprHandle& self) {
                return tensorexpr::log(self);
              })
          .def(
              "fast_log",
              [](const tensorexpr::ExprHandle& self) {
                return tensorexpr::fast_log(self);
              })
          .def(
              "log2",
              [](const tensorexpr::ExprHandle& self) {
                return tensorexpr::log2(self);
              })
          .def(
              "log10",
              [](const tensorexpr::ExprHandle& self) {
                return tensorexpr::log10(self);
              })
          .def(
              "log1p",
              [](const tensorexpr::ExprHandle& self) {
                return tensorexpr::log1p(self);
              })
          .def(
              "erf",
              [](const tensorexpr::ExprHandle& self) {
                return tensorexpr::erf(self);
              })
          .def(
              "erfc",
              [](const tensorexpr::ExprHandle& self) {
                return tensorexpr::erfc(self);
              })
          .def(
              "sqrt",
              [](const tensorexpr::ExprHandle& self) {
                return tensorexpr::sqrt(self);
              })
          .def(
              "rsqrt",
              [](const tensorexpr::ExprHandle& self) {
                return tensorexpr::rsqrt(self);
              })
          .def(
              "ceil",
              [](const tensorexpr::ExprHandle& self) {
                return tensorexpr::ceil(self);
              })
          .def(
              "floor",
              [](const tensorexpr::ExprHandle& self) {
                return tensorexpr::floor(self);
              })
          .def(
              "round",
              [](const tensorexpr::ExprHandle& self) {
                return tensorexpr::round(self);
              })
          .def(
              "trunc",
              [](const tensorexpr::ExprHandle& self) {
                return tensorexpr::trunc(self);
              })
          .def(
              "frac",
              [](const tensorexpr::ExprHandle& self) {
                return tensorexpr::frac(self);
              })
          .def(
              "lgamma",
              [](const tensorexpr::ExprHandle& self) {
                return tensorexpr::lgamma(self);
              })
          .def("isnan", [](const tensorexpr::ExprHandle& self) {
            return tensorexpr::isnan(self);
          });
  te.def(
      "ifThenElse",
      [](const tensorexpr::ExprHandle& c,
         const tensorexpr::ExprHandle& t,
         const tensorexpr::ExprHandle& f) {
        return tensorexpr::ifThenElse(c, t, f);
      });
  te.def(
      "atan2",
      [](const tensorexpr::ExprHandle& v1, const tensorexpr::ExprHandle& v2) {
        return tensorexpr::atan2(v1, v2);
      });
  te.def(
      "pow",
      [](const tensorexpr::ExprHandle& v1, const tensorexpr::ExprHandle& v2) {
        return tensorexpr::pow(v1, v2);
      });
  te.def(
      "fmod",
      [](const tensorexpr::ExprHandle& v1, const tensorexpr::ExprHandle& v2) {
        return tensorexpr::fmod(v1, v2);
      });
  te.def(
      "remainder",
      [](const tensorexpr::ExprHandle& v1, const tensorexpr::ExprHandle& v2) {
        return tensorexpr::remainder(v1, v2);
      });

#define EXPRHANDLE_CTOR(ctype, name) \
  expr_handle_class.def_static(      \
      #ctype, [](ctype v) { return tensorexpr::ExprHandle(v); });
  AT_FORALL_SCALAR_TYPES_AND2(Bool, Half, EXPRHANDLE_CTOR)
#undef EXPRHANDLE_CTOR

  py::class_<tensorexpr::VarHandle, tensorexpr::ExprHandle>(te, "VarHandle");
  py::class_<tensorexpr::BufHandle, tensorexpr::ExprHandle>(te, "BufHandle");

  py::class_<tensorexpr::Placeholder>(te, "Placeholder")
      .def(py::init<
           const std::string&,
           const tensorexpr::Dtype&,
           std::vector<tensorexpr::ExprHandle>&>())
      .def(
          "load",
          [](tensorexpr::Placeholder& self,
             const std::vector<tensorexpr::ExprHandle>& v) {
            return self.load(v);
          });
  py::class_<tensorexpr::Tensor>(te, "Tensor")
      .def(
          "load",
          [](tensorexpr::Tensor& self,
             const std::vector<tensorexpr::ExprHandle>& v) {
            return self.call(v);
          });
  py::class_<tensorexpr::Cast>(te, "Cast")
      .def_static("make", &tensorexpr::Cast::make);

  py::class_<tensorexpr::DimArg>(te, "DimArg")
      .def(py::init<const tensorexpr::ExprHandle&>())
      .def(py::init<const tensorexpr::ExprHandle&, const std::string&>());

  te.def(
      "Compute",
      [](const std::string& func_name,
         const std::vector<tensorexpr::DimArg>& dim_args,
         py::function func) {
        if (dim_args.size() == 1) {
          return tensorexpr::Compute(
              func_name, dim_args, [&func](const tensorexpr::VarHandle& a) {
                return py::cast<tensorexpr::ExprHandle>(func(a));
              });
        } else if (dim_args.size() == 2) {
          return tensorexpr::Compute(
              func_name,
              dim_args,
              [&func](
                  const tensorexpr::VarHandle& a,
                  const tensorexpr::VarHandle& b) {
                return py::cast<tensorexpr::ExprHandle>(func(a, b));
              });
        } else if (dim_args.size() == 3) {
          return tensorexpr::Compute(
              func_name,
              dim_args,
              [&func](
                  const tensorexpr::VarHandle& a,
                  const tensorexpr::VarHandle& b,
                  const tensorexpr::VarHandle& c) {
                return py::cast<tensorexpr::ExprHandle>(func(a, b, c));
              });
        } else if (dim_args.size() == 4) {
          return tensorexpr::Compute(
              func_name,
              dim_args,
              [&func](
                  const tensorexpr::VarHandle& a,
                  const tensorexpr::VarHandle& b,
                  const tensorexpr::VarHandle& c,
                  const tensorexpr::VarHandle& d) {
                return py::cast<tensorexpr::ExprHandle>(func(a, b, c, d));
              });
        } else {
          throw std::runtime_error("Too many args");
        }
      },
      py::return_value_policy::reference);
  py::class_<tensorexpr::Reducer>(te, "Reducer");

  te.def(
      "SumReduce",
      [](const std::string& func_name,
         const std::vector<tensorexpr::DimArg>& dim_args,
         tensorexpr::Tensor* buffer,
         const std::vector<tensorexpr::DimArg>& reduce_args) {
        return tensorexpr::Reduce(
            func_name, dim_args, tensorexpr::Sum(), buffer, reduce_args);
      },
      py::return_value_policy::reference);

  py::class_<tensorexpr::Stmt>(te, "Stmt")
      .def("__str__", [](const tensorexpr::Stmt& self) {
        std::stringstream ss;
        ss << self;
        return ss.str();
      });
  py::class_<tensorexpr::For, tensorexpr::Stmt>(te, "For")
      .def(
          "index_var",
          [](const tensorexpr::For& self) {
            return tensorexpr::VarHandle(self.var());
          },
          py::return_value_policy::reference)
      .def("body", &tensorexpr::For::body, py::return_value_policy::reference);

  py::class_<tensorexpr::Block, tensorexpr::Stmt>(te, "Block")
      .def(
          "stmts",
          &tensorexpr::Block::stmts,
          py::return_value_policy::reference);

  py::class_<tensorexpr::LoopNest>(te, "LoopNest")
      .def(py::init<const std::vector<tensorexpr::Tensor*>&>())
      .def("vectorize_inner_loops", &tensorexpr::LoopNest::vectorizeInnerLoops)
      .def("prepare_for_codegen", &tensorexpr::LoopNest::prepareForCodegen)
      .def(
          "get_loop_body_for",
          &tensorexpr::LoopNest::getLoopBodyFor,
          py::return_value_policy::reference)
      .def(
          "get_loops_for",
          [](const tensorexpr::LoopNest& self, tensorexpr::Tensor* t) {
            return self.getLoopStmtsFor(t);
          },
          py::return_value_policy::reference)
      .def(
          "split_with_tail",
          [](const tensorexpr::LoopNest& self, tensorexpr::For* f, int factor) {
            tensorexpr::For *outer = nullptr, *inner = nullptr, *tail = nullptr;
            self.splitWithTail(f, factor, &outer, &inner, &tail);
            return std::make_tuple(outer, inner, tail);
          },
          py::return_value_policy::reference)
      .def(
          "split_with_mask",
          [](const tensorexpr::LoopNest& self, tensorexpr::For* f, int factor) {
            tensorexpr::For *outer = nullptr, *inner = nullptr;
            self.splitWithMask(f, factor, &outer, &inner);
            return std::make_tuple(outer, inner);
          },
          py::return_value_policy::reference)
      .def(
          "unroll",
          [](const tensorexpr::LoopNest& self, tensorexpr::For* f) {
            tensorexpr::Stmt* unrolled = nullptr;
            self.unroll(f, &unrolled);
            return unrolled;
          },
          py::return_value_policy::reference)
      .def(
          "vectorize",
          [](const tensorexpr::LoopNest& self, tensorexpr::For* f) {
            self.vectorize(f);
          },
          py::return_value_policy::reference)
      .def(
          "compute_inline",
          [](tensorexpr::LoopNest& self, tensorexpr::Stmt* s) {
            self.computeInline(s);
          },
          py::return_value_policy::reference)
      .def(
          "compute_inline",
          [](tensorexpr::LoopNest& self, const tensorexpr::BufHandle& b) {
            self.computeInline(b.node());
          },
          py::return_value_policy::reference)
      .def(
          "rfactor",
          [](tensorexpr::LoopNest& self,
             const tensorexpr::Stmt& s,
             const tensorexpr::VarHandle& v) {
            auto st = dynamic_cast<const tensorexpr::Store*>(&s);
            if (!st) {
              return;
            }
            auto r = st->value();
            self.rfactor(r, v.node());
          },
          py::return_value_policy::reference)
      .def(
          "rfactor",
          [](tensorexpr::LoopNest& self,
             const tensorexpr::Stmt& s,
             const tensorexpr::VarHandle& v,
             tensorexpr::Block& ins_point) {
            auto st = dynamic_cast<const tensorexpr::Store*>(&s);
            if (!st) {
              return;
            }
            auto r = st->value();
            self.rfactor(r, v.node(), &ins_point);
          },
          py::return_value_policy::reference)
      .def(
          "flatten",
          [](const tensorexpr::LoopNest& self,
             const std::vector<tensorexpr::For*>& loops) {
            tensorexpr::For* flattened;
            self.flatten(loops, &flattened);
            return flattened;
          },
          py::return_value_policy::reference)
      .def(
          "reorder",
          &tensorexpr::LoopNest::reorderAxis,
          py::return_value_policy::reference)
      .def(
          "simplify",
          &tensorexpr::LoopNest::simplify,
          py::return_value_policy::reference)
      .def(
          "set_GPU_block_index",
          &tensorexpr::LoopNest::setGPUBlockIndex,
          py::return_value_policy::reference)
      .def(
          "set_GPU_thread_index",
          &tensorexpr::LoopNest::setGPUThreadIndex,
          py::return_value_policy::reference)
      .def(
          "__str__",
          [](const tensorexpr::LoopNest& self) {
            std::stringstream ss;
            ss << *self.root_stmt();
            return ss.str();
          })
      .def(
          "root_stmt",
          &tensorexpr::LoopNest::root_stmt,
          py::return_value_policy::reference);

  te.def(
      "simplify",
      [](tensorexpr::Stmt* stmt) {
        return tensorexpr::IRSimplifier::simplify(stmt);
      },
      py::return_value_policy::reference);

  py::class_<tensorexpr::CodeGen>(te, "CodeGen")
      .def(
          "call",
          [](tensorexpr::CodeGen& self, const std::vector<at::Tensor>& values) {
            std::vector<tensorexpr::CodeGen::CallArg> value_ptrs;
            for (const auto& value : values) {
              value_ptrs.emplace_back(
                  tensorexpr::CodeGen::CallArg(value.data_ptr()));
            }
            self.call(value_ptrs);
          });
  py::class_<tensorexpr::SimpleIREvaluator, tensorexpr::CodeGen>(
      te, "SimpleIREvaluator");
#ifdef TORCH_ENABLE_LLVM
  py::class_<tensorexpr::LLVMCodeGen, tensorexpr::CodeGen>(te, "LLVMCodeGen");
#endif

  py::class_<tensorexpr::CodeGen::BufferArg>(te, "BufferArg")
      .def(py::init<const tensorexpr::Placeholder&>())
      .def(py::init<tensorexpr::Tensor*>())
      .def(py::init<const tensorexpr::VarHandle&>());

  te.def(
      "construct_codegen",
      [](const std::string& name,
         tensorexpr::Stmt* stmt,
         const std::vector<tensorexpr::CodeGen::BufferArg>& args) {
        tensorexpr::CodeGen* cg = nullptr;
        if (name == "llvm") {
#ifdef TORCH_ENABLE_LLVM
          cg = new tensorexpr::LLVMCodeGen(stmt, args);
#else
          throw std::runtime_error("PyTorch not compiled with LLVM support!");
#endif
        } else if (name == "cuda") {
#ifdef USE_CUDA
          cg = new tensorexpr::CudaCodeGen(stmt, args);
#else
          throw std::runtime_error("PyTorch not compiled with CUDA support!");
#endif
        } else {
          cg = new tensorexpr::SimpleIREvaluator(stmt, args);
        }
        return cg;
      });
}
} // namespace jit
} // namespace torch
