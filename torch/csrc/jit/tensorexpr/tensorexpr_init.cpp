#include <pybind11/functional.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/jit/tensorexpr/codegen.h>
#ifdef USE_CUDA
#include <torch/csrc/jit/tensorexpr/cuda_codegen.h>
#endif
#include <torch/csrc/jit/tensorexpr/ir_printer.h>
#include <torch/csrc/jit/tensorexpr/ir_simplifier.h>
#include <torch/csrc/jit/tensorexpr/kernel.h>
#include <torch/csrc/jit/tensorexpr/llvm_codegen.h>
#include <torch/csrc/jit/tensorexpr/loopnest.h>
#include <torch/csrc/jit/tensorexpr/reduction.h>

namespace torch {
namespace jit {
using namespace torch::jit::tensorexpr;

ArgValue convertPyToArgValue(py::handle inp) {
  if (py::isinstance<Placeholder>(inp)) {
    return py::cast<Placeholder>(inp).handle();
  } else if (py::isinstance<BufHandle>(inp)) {
    return py::cast<BufHandle>(inp);
  } else if (py::isinstance<VarHandle>(inp)) {
    return py::cast<VarHandle>(inp);
  } else if (py::isinstance<py::bool_>(inp)) {
    return py::cast<bool>(inp);
  } else if (py::isinstance<py::float_>(inp)) {
    return py::cast<double>(inp);
  } else if (py::isinstance<py::int_>(inp)) {
    return py::cast<int64_t>(inp);
  } else if (py::isinstance<py::none>(inp)) {
    return ArgNone();
  } else if (py::isinstance<py::list>(inp)) {
    auto l = py::cast<py::list>(inp);
    if (l.size() == 0) {
      return std::vector<BufHandle>();
    } else if (py::isinstance<py::int_>(l[0])) {
      return py::cast<IntList>(inp);
    } else if (py::isinstance<BufHandle>(l[0])) {
      return py::cast<BufList>(inp);
    } else {
      throw std::runtime_error("vector conversion failed");
    }
  } else {
    throw std::runtime_error("nyi");
  }
}
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
      .def(
          "store",
          [](Placeholder& self,
             const std::vector<ExprHandle>& args,
             const ExprHandle& val) { return self.store(args, val); })
      .def(
          "data",
          [](Placeholder& self) { return BufHandle(self.data()); },
          py::return_value_policy::reference);
  py::class_<Tensor, std::unique_ptr<Tensor, py::nodelete>>(te, "Tensor")
      .def(py::init(
          [](BufHandle& b, Stmt* s) { return new Tensor(b.node(), s); }))
      .def(
          "load",
          [](Tensor& self, const std::vector<ExprHandle>& v) {
            return self.load(v);
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
  te.def(
      "Reduce",
      [](const std::string& func_name,
         const std::vector<DimArg>& dim_args,
         const Reducer& reducer,
         const std::function<ExprHandle(const std::vector<VarHandle>&)>&
             body_func,
         const std::vector<DimArg>& reduce_args) {
        return Reduce(func_name, dim_args, reducer, body_func, reduce_args);
      },
      py::return_value_policy::reference);
  te.def(
      "Reduce",
      [](const std::string& func_name,
         const std::vector<DimArg>& dim_args,
         const Reducer& reducer,
         const std::function<ExprHandle(const std::vector<VarHandle>&)>&
             init_func,
         const std::function<ExprHandle(const std::vector<VarHandle>&)>&
             body_func,
         const std::vector<DimArg>& reduce_args) {
        return Reduce(func_name, dim_args, reducer, body_func, reduce_args);
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
  py::class_<Store, Stmt, std::unique_ptr<Store, py::nodelete>>(te, "Store")
      .def_static(
          "make",
          [](const BufHandle& buf,
             std::vector<ExprHandle>& indicies,
             const ExprHandle& value) {
            return Store::make(buf, indicies, value);
          },
          py::return_value_policy::reference);

  py::class_<For, Stmt, std::unique_ptr<For, py::nodelete>>(te, "For")
      .def(
          "index_var",
          [](const For& self) { return VarHandle(self.var()); },
          py::return_value_policy::reference)
      .def("body", &For::body, py::return_value_policy::reference)
      .def("set_parallel", &For::set_parallel)
      .def_static(
          "make",
          [](const VarHandle& var,
             const ExprHandle& start,
             const ExprHandle& stop,
             Stmt* body) { return For::make(var, start, stop, body); },
          py::return_value_policy::reference);

  py::class_<Cond, Stmt, std::unique_ptr<Cond, py::nodelete>>(te, "Cond")
      .def_static(
          "make",
          [](const ExprHandle& condition, Stmt* true_stmt, Stmt* false_stmt) {
            return new Cond(condition.node(), true_stmt, false_stmt);
          },
          py::return_value_policy::reference)
      .def("true_stmt", &Cond::true_stmt, py::return_value_policy::reference)
      .def("false_stmt", &Cond::false_stmt, py::return_value_policy::reference);

  py::class_<
      tensorexpr::Block,
      Stmt,
      std::unique_ptr<tensorexpr::Block, py::nodelete>>(te, "Block")
      .def(py::init([](const std::vector<Stmt*>& stmts) {
        return tensorexpr::Block::make(stmts);
      }))
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
          "get_loop_body_for",
          [](const LoopNest& self, BufHandle* b) {
            return self.getLoopBodyFor(b->node());
          },
          py::return_value_policy::reference)
      .def(
          "get_loops_for",
          [](const LoopNest& self, Tensor* t) {
            return self.getLoopStmtsFor(t);
          },
          py::return_value_policy::reference)
      .def(
          "get_all_loopnests_for",
          [](const LoopNest& self, const BufHandle* b) {
            return self.getAllLoopNestsWritingToBuf(b->node());
          },
          py::return_value_policy::reference)
      .def(
          "get_enclosing_loopnest",
          [](const LoopNest& self, const Stmt* s) {
            return self.getEnclosingLoopNest(s);
          },
          py::return_value_policy::reference)
      .def(
          "get_innermost_loops_for",
          [](const LoopNest& self, const BufHandle* b) {
            return self.getAllInnermostLoopsWritingToBuf(b->node());
          },
          py::return_value_policy::reference)
      .def(
          "get_writes_for",
          [](const LoopNest& self, const BufHandle* b) {
            return self.getAllWritesToBuf(b->node());
          },
          py::return_value_policy::reference)
      .def(
          "get_parent_loop",
          [](const LoopNest& self, const Stmt* s) {
            return self.getParentLoop(s);
          },
          py::return_value_policy::reference)
      .def_static(
          "get_loop_stmts_in_loopnest",
          [](For* f, size_t num) {
            return LoopNest::getLoopStmtsInLoopNest(f, num);
          },
          py::return_value_policy::reference)
      .def(
          "split_with_tail",
          [](const LoopNest& self, For* f, int factor) {
            For *inner = nullptr, *tail = nullptr;
            self.splitWithTail(f, factor, &inner, &tail);
            return std::make_tuple(inner, tail);
          },
          py::return_value_policy::reference)
      .def(
          "split_with_mask",
          [](const LoopNest& self, For* f, int factor) {
            For* inner = nullptr;
            self.splitWithMask(f, factor, &inner);
            return inner;
          },
          py::return_value_policy::reference)
      .def(
          "slice_head",
          [](LoopNest& self, For* f, int factor) {
            For *head = nullptr, *tail = nullptr;
            self.sliceHead(f, factor, &head, &tail);
            return std::make_tuple(head, tail);
          },
          py::return_value_policy::reference)
      .def(
          "slice_tail",
          [](LoopNest& self, For* f, int factor) {
            For *head = nullptr, *tail = nullptr;
            self.sliceTail(f, factor, &head, &tail);
            return std::make_tuple(head, tail);
          },
          py::return_value_policy::reference)
      .def_static(
          "normalize",
          [](For* f) {
            LoopNest::normalize(f);
            return f;
          },
          py::return_value_policy::reference)
      .def_static(
          "distribute_loop",
          [](For* f) { return LoopNest::distributeLoop(f); },
          py::return_value_policy::reference)
      .def_static(
          "distribute_loop",
          [](For* f, const std::unordered_set<Stmt*>& pivots) {
            return LoopNest::distributeLoop(f, pivots);
          },
          py::return_value_policy::reference)
      .def_static(
          "distribute_loop_over_inner_loops",
          [](For* f) { return LoopNest::distributeLoopOverInnerLoops(f); },
          py::return_value_policy::reference)
      .def_static(
          "fuse_loops",
          [](const std::vector<For*>& loops) {
            // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
            For* fused_loop;
            LoopNest::fuseLoops(loops, &fused_loop);
            return fused_loop;
          },
          py::return_value_policy::reference)
      .def_static(
          "reorder",
          [](const std::vector<For*>& loops,
             const std::vector<size_t>& permutation) {
            return LoopNest::reorder(loops, permutation);
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
          "cache_accesses",
          [](LoopNest& self,
             const Buf* producer,
             const std::string& name,
             Stmt* consumer) {
            return self.cacheAccesses(producer, name, consumer);
          },
          py::return_value_policy::reference)
      .def(
          "compute_at",
          [](LoopNest& self, Stmt* s, For* at) { self.computeAt(s, at); })
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
          [](LoopNest& self, Stmt* s, For* target_for) {
            Buf* rfac_buf = nullptr;
            self.rfactor(s, target_for, &rfac_buf);
            return BufHandle(rfac_buf);
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
          "reorder_axis",
          &LoopNest::reorderAxis,
          py::return_value_policy::reference)
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
          "inline_intermediate_bufs",
          [](LoopNest& self, bool allow_duplicated_work) {
            self.inlineIntermediateBufs(allow_duplicated_work);
          })
      .def(
          "eliminate_dead_stores",
          [](LoopNest& self) { self.eliminateDeadStores(); })
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

  te.def(
      "lower",
      [](std::string op_str,
         py::list inputs,
         std::vector<ExprHandle> outputShape,
         Dtype outputType) {
        auto op = c10::Symbol::fromQualString(op_str);
        std::vector<ArgValue> argInputs;
        for (auto inp : inputs) {
          argInputs.push_back(convertPyToArgValue(inp));
        }
        return computeOperandValue(
            op, argInputs, outputShape, outputType.scalar_type());
      });

  using TSGraph = std::shared_ptr<Graph>;
  py::class_<TensorExprKernel>(te, "TensorExprKernel")
      .def(py::init<const TSGraph&>())
      .def(
          "run",
          [](TensorExprKernel& self, const py::tuple& inputs) {
            Stack stack;
            stack.reserve(inputs.size()); // captures?
            for (auto& obj : inputs) {
              stack.push_back(toTypeInferredIValue(obj));
            }
            auto g_inputs = self.graph()->inputs();
            for (size_t i = 0; i < inputs.size(); ++i) {
              if (stack[i].isTensor()) {
                g_inputs[i]->setType(stack[i].type());
              }
            }
            self.run(stack);
            return createPyObjectForStack(std::move(stack));
          })
      .def(
          "fallback",
          [](TensorExprKernel& self, const py::tuple& inputs) {
            Stack stack;
            stack.reserve(inputs.size()); // captures?
            for (auto& obj : inputs) {
              stack.push_back(toTypeInferredIValue(obj));
            }
            auto g_inputs = self.graph()->inputs();
            for (size_t i = 0; i < inputs.size(); ++i) {
              if (stack[i].isTensor()) {
                g_inputs[i]->setType(stack[i].type());
              }
            }
            self.fallback(stack);
            return createPyObjectForStack(std::move(stack));
          })
      .def(
          "get_codegen_stmt",
          [](TensorExprKernel& self) { return self.getCodeGenStmt(); },
          py::return_value_policy::reference)
      .def(
          "get_code_text",
          [](TensorExprKernel& self, const std::string& attr = "") {
            return self.getCodeText(attr);
          },
          py::arg("attr") = "");

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
          "get_code_text",
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
  te.def("annotate_input_shapes", &tensorexpr::annotateInputShapes);
  te.def("remove_unused_self_argument", &tensorexpr::removeUnusedSelfArgument);
}
} // namespace jit
} // namespace torch
