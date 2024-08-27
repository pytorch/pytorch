#include <pybind11/functional.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/jit/tensorexpr/codegen.h>
#include <torch/csrc/utils/pybind.h>
#ifdef USE_CUDA
#include <torch/csrc/jit/tensorexpr/cuda_codegen.h>
#endif
#include <torch/csrc/jit/tensorexpr/graph_opt.h>
#include <torch/csrc/jit/tensorexpr/ir_printer.h>
#include <torch/csrc/jit/tensorexpr/ir_simplifier.h>
#include <torch/csrc/jit/tensorexpr/kernel.h>
#include <torch/csrc/jit/tensorexpr/llvm_codegen.h>
#include <torch/csrc/jit/tensorexpr/loopnest.h>
#include <torch/csrc/jit/tensorexpr/lowerings.h>
#include <torch/csrc/jit/tensorexpr/reduction.h>

#include <utility>

template <>
struct pybind11::detail::type_caster<torch::jit::tensorexpr::ArgValue>
    : public type_caster_base<torch::jit::tensorexpr::ArgValue> {};

namespace torch::jit {
using namespace torch::jit::tensorexpr;

ArgValue convertPyToArgValue(py::handle inp) {
  if (py::isinstance<BufHandle>(inp)) {
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
    if (l.empty()) {
      return std::vector<BufHandle>();
    } else if (py::isinstance<py::int_>(l[0])) {
      return py::cast<IntList>(inp);
    } else if (py::isinstance<BufHandle>(l[0])) {
      return py::cast<BufList>(inp);
    } else {
      throw std::runtime_error("vector conversion failed");
    }
  } else {
    throw std::runtime_error("conversion not yet implemented");
  }
}

Dtype parsePythonDtype(py::handle obj) {
  if (THPDtype_Check(obj.ptr())) {
    return Dtype(reinterpret_cast<THPDtype*>(obj.ptr())->scalar_type);
  } else {
    throw std::runtime_error("expected a torch.dtype instance");
  }
}

void initTensorExprBindings(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();

  // Tensor Expr Classes
  auto te = m.def_submodule("_te");

  auto dtype_class =
      py::class_<Dtype>(te, "Dtype").def(py::init(&parsePythonDtype));
  py::implicitly_convertible<py::object, Dtype>();

#define DTYPE_SINGLETON_ACCESSOR(ctype, name) \
  dtype_class.def_property_readonly_static(   \
      #name, [](const py::object&) { return k##name; });
  AT_FORALL_SCALAR_TYPES_AND3(Bool, Half, BFloat16, DTYPE_SINGLETON_ACCESSOR)
#undef DTYPE_SINGLETON_ACCESSOR

  auto expr_handle_class =
      py::class_<ExprHandle>(te, "ExprHandle")
          .def(
              "__str__",
              [](const ExprHandle& self) {
                std::stringstream ss;
                ss << self;
                return ss.str();
              })
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
              "__pow__",
              [](const ExprHandle& self, const ExprHandle& other) {
                return pow(self, other);
              })
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
              "fast_tanh",
              [](const ExprHandle& self) { return fast_tanh(self); })
          .def(
              "fast_sigmoid",
              [](const ExprHandle& self) { return fast_sigmoid(self); })
          .def(
              "fast_log", [](const ExprHandle& self) { return fast_log(self); })
          .def("log_vml", [](const ExprHandle& self) { return log_vml(self); })
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
          .def("isnan", [](const ExprHandle& self) { return isnan(self); })
          .def(
              "cast",
              [](const ExprHandle& self, const Dtype& dt) {
                return Cast::make(dt, self);
              })
#define EXPRHANDLE_INIT(ctype, name) \
  .def(py::init([](ctype val) { return name##Imm::make(val); }))
              AT_FORALL_SCALAR_TYPES_AND3(Bool, Half, BFloat16, EXPRHANDLE_INIT)
#undef EXPRHANDLE_INIT
      ;

#define EXPRHANDLE_IMPL_CONV(ctype, name) \
  py::implicitly_convertible<ctype, ExprHandle>();
  AT_FORALL_SCALAR_TYPES_AND3(Bool, Half, BFloat16, EXPRHANDLE_IMPL_CONV)
#undef EXPRHANDLE_IMPL_CONV

  te.def(
      "ifThenElse",
      [](const ExprHandle& c, const ExprHandle& t, const ExprHandle& f) {
        return ifThenElse(c, t, f);
      });

  te.def("sin", [](const ExprHandle& v1) { return sin(v1); });
  te.def("cos", [](const ExprHandle& v1) { return cos(v1); });
  te.def("tan", [](const ExprHandle& v1) { return tan(v1); });
  te.def("asin", [](const ExprHandle& v1) { return asin(v1); });
  te.def("acos", [](const ExprHandle& v1) { return acos(v1); });
  te.def("atan", [](const ExprHandle& v1) { return atan(v1); });
  te.def("sinh", [](const ExprHandle& v1) { return sinh(v1); });
  te.def("cosh", [](const ExprHandle& v1) { return cosh(v1); });
  te.def("tanh", [](const ExprHandle& v1) { return tanh(v1); });
  te.def("sigmoid", [](const ExprHandle& v1) { return sigmoid(v1); });
  te.def("exp", [](const ExprHandle& v1) { return exp(v1); });
  te.def("expm1", [](const ExprHandle& v1) { return expm1(v1); });
  te.def("abs", [](const ExprHandle& v1) { return abs(v1); });
  te.def("log", [](const ExprHandle& v1) { return log(v1); });
  te.def("log2", [](const ExprHandle& v1) { return log2(v1); });
  te.def("log10", [](const ExprHandle& v1) { return log10(v1); });
  te.def("log1p", [](const ExprHandle& v1) { return log1p(v1); });
  te.def("erf", [](const ExprHandle& v1) { return erf(v1); });
  te.def("erfc", [](const ExprHandle& v1) { return erfc(v1); });
  te.def("sqrt", [](const ExprHandle& v1) { return sqrt(v1); });
  te.def("rsqrt", [](const ExprHandle& v1) { return rsqrt(v1); });
  te.def("ceil", [](const ExprHandle& v1) { return ceil(v1); });
  te.def("floor", [](const ExprHandle& v1) { return floor(v1); });
  te.def("round", [](const ExprHandle& v1) { return round(v1); });
  te.def("trunc", [](const ExprHandle& v1) { return trunc(v1); });
  te.def("frac", [](const ExprHandle& v1) { return frac(v1); });
  te.def("lgamma", [](const ExprHandle& v1) { return lgamma(v1); });
  te.def("isnan", [](const ExprHandle& v1) { return isnan(v1); });

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
  AT_FORALL_SCALAR_TYPES_AND3(Bool, Half, BFloat16, EXPRHANDLE_CTOR)
#undef EXPRHANDLE_CTOR

  py::class_<VarHandle, ExprHandle>(te, "VarHandle")
      .def(
          "__str__",
          [](const ExprHandle& self) {
            std::stringstream ss;
            ss << self;
            return ss.str();
          })
      .def(py::init<Dtype>())
      .def(py::init<const std::string&, Dtype>());
  py::class_<BufHandle, ExprHandle>(te, "BufHandle")
      .def(
          py::init<const std::string&, const std::vector<ExprHandle>&, Dtype>())
      .def(py::init<const std::vector<ExprHandle>&, Dtype>())
      .def(py::init<Dtype>())
      .def(
          "__hash__",
          [](const BufHandle& self) {
            return std::hash<BufPtr>()(self.node());
          })
      .def(
          "__eq__",
          [](const BufHandle& self, const BufHandle& other) {
            return self.node() == other.node();
          })
      .def(
          "load",
          [](BufHandle& self, const std::vector<ExprHandle>& v) {
            return Load::make(self, v);
          })
      .def(
          "load",
          [](BufHandle& self, const ExprHandle& v) {
            return Load::make(self, {v});
          })
      .def(
          "store",
          [](BufHandle& self,
             const std::vector<ExprHandle>& i,
             const ExprHandle& v) { return Store::make(self, i, v); })
      .def(
          "store",
          [](BufHandle& self, const ExprHandle& i, const ExprHandle& v) {
            return Store::make(self, {i}, v);
          });
  py::class_<Tensor>(te, "Tensor")
      .def(py::init([](const BufHandle& b, const StmtPtr& s) {
        return Tensor(b.node(), s);
      }))
      .def(
          "load",
          [](Tensor& self, const std::vector<ExprHandle>& v) {
            return self.load(v);
          })
      .def("buf", [](Tensor& self) { return BufHandle(self.buf()); })
      .def("stmt", &Tensor::stmt);
  py::class_<Cast, std::shared_ptr<Cast>>(te, "Cast")
      .def_static("make", &Cast::make)
      .def(
          "src_value",
          [](CastPtr& self) { return ExprHandle(self->src_value()); })
      .def("set_src_value", [](CastPtr& self, const ExprHandle& value) {
        self->set_src_value(value.node());
      });

  te.def(
      "Compute",
      [](const std::string& func_name,
         const std::vector<ExprHandle>& dim_args,
         const py::function& func) {
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

  te.def(
      "Compute2",
      [](const std::string& func_name,
         const std::vector<ExprHandle>& dim_args,
         const py::function& func) {
        return Compute(
            func_name, dim_args, [&func](const std::vector<VarHandle>& dims) {
              return py::cast<ExprHandle>(func(dims));
            });
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
         const std::vector<ExprHandle>& dim_args,
         const Reducer& reducer,
         const Tensor& buffer,
         const std::vector<ExprHandle>& reduce_args) {
        return Reduce(func_name, dim_args, reducer, buffer, reduce_args);
      },
      py::return_value_policy::reference);

  te.def(
      "Reduce",
      [](const std::string& func_name,
         const std::vector<ExprHandle>& dim_args,
         const Reducer& reducer,
         const BufHandle& buffer,
         const std::vector<ExprHandle>& reduce_args) {
        return Reduce(func_name, dim_args, reducer, buffer, reduce_args);
      },
      py::return_value_policy::reference);
  te.def(
      "Reduce",
      [](const std::string& func_name,
         const std::vector<ExprHandle>& dim_args,
         const Reducer& reducer,
         const std::function<ExprHandle(const std::vector<VarHandle>&)>&
             body_func,
         const std::vector<ExprHandle>& reduce_args) {
        return Reduce(func_name, dim_args, reducer, body_func, reduce_args);
      },
      py::return_value_policy::reference);
  te.def(
      "Reduce",
      [](const std::string& func_name,
         const std::vector<ExprHandle>& dim_args,
         const Reducer& reducer,
         const std::function<ExprHandle(const std::vector<VarHandle>&)>&
             init_func,
         const std::function<ExprHandle(const std::vector<VarHandle>&)>&
             body_func,
         const std::vector<ExprHandle>& reduce_args) {
        return Reduce(func_name, dim_args, reducer, body_func, reduce_args);
      },
      py::return_value_policy::reference);

  py::class_<Stmt, std::shared_ptr<Stmt>>(te, "Stmt")
      .def(py::init([](const std::vector<StmtPtr>& stmts) {
        return tensorexpr::Block::make(stmts);
      }))
      .def("__str__", [](Stmt& self) {
        std::stringstream ss;
        ss << self;
        return ss.str();
      });
  py::class_<Store, Stmt, std::shared_ptr<Store>>(te, "Store")
      .def_static(
          "make",
          [](const BufHandle& buf,
             std::vector<ExprHandle>& indices,
             const ExprHandle& value) {
            return Store::make(buf, indices, value);
          });

  py::class_<For, Stmt, std::shared_ptr<For>>(te, "For")
      .def("index_var", [](For& self) { return VarHandle(self.var()); })
      .def("body", &For::body)
      .def("set_parallel", &For::set_parallel)
      .def(
          "set_gpu_block_index",
          [](For& self, int block_index) {
            self.set_gpu_block_index(block_index);
          })
      .def(
          "set_gpu_thread_index",
          [](For& self, int thread_index) {
            self.set_gpu_thread_index(thread_index);
          })
      .def_static(
          "make",
          [](const VarHandle& var,
             const ExprHandle& start,
             const ExprHandle& stop,
             const StmtPtr& body) {
            return For::make(var, start, stop, body);
          });

  py::class_<Cond, Stmt, std::shared_ptr<Cond>>(te, "Cond")
      .def_static(
          "make",
          [](const ExprHandle& condition,
             const StmtPtr& true_stmt,
             const StmtPtr& false_stmt) {
            return Cond::make(condition, true_stmt, false_stmt);
          })
      .def("true_stmt", &Cond::true_stmt)
      .def("false_stmt", &Cond::false_stmt);

  py::class_<tensorexpr::Block, Stmt, std::shared_ptr<tensorexpr::Block>>(
      te, "Block")
      .def(py::init([](const std::vector<StmtPtr>& stmts) {
        return tensorexpr::Block::make(stmts);
      }))
      .def("stmts", &tensorexpr::Block::stmts);
  py::class_<ExternalCall, Stmt, std::shared_ptr<ExternalCall>>(
      te, "ExternalCall")
      .def(py::init(&ExternalCall::make));

  py::class_<LoopNest>(te, "LoopNest")
      .def(py::init<const std::vector<Tensor>&>())
      .def(py::init<const std::vector<Tensor>&, const std::vector<Tensor>&>())
      .def(py::init([](const StmtPtr& s, const std::vector<BufHandle>& bufs) {
        std::unordered_set<BufPtr> buf_nodes;
        buf_nodes.reserve(bufs.size());
        for (auto& buf : bufs) {
          buf_nodes.insert(buf.node());
        }
        return std::make_unique<LoopNest>(s, buf_nodes);
      }))
      .def("vectorize_inner_loops", &LoopNest::vectorizeInnerLoops)
      .def(
          "prepare_for_codegen",
          [](LoopNest& self) { return self.prepareForCodegen(); },
          py::return_value_policy::reference)
      .def(
          "get_loop_body_for",
          [](const LoopNest& self, const Tensor& t) {
            return self.getLoopBodyFor(t);
          },
          py::return_value_policy::reference)
      .def(
          "get_loop_body_for",
          [](const LoopNest& self, BufHandle& b) {
            return self.getLoopBodyFor(b.node());
          },
          py::return_value_policy::reference)
      .def(
          "get_loops_for",
          [](const LoopNest& self, const Tensor& t) {
            return self.getLoopStmtsFor(t);
          },
          py::return_value_policy::reference)
      .def(
          "get_all_loopnests_for",
          [](const LoopNest& self, const BufHandle& b) {
            return self.getAllLoopNestsWritingToBuf(b.node());
          },
          py::return_value_policy::reference)
      .def(
          "get_enclosing_loopnest",
          [](const LoopNest& self, const StmtPtr& s) {
            return self.getEnclosingLoopNest(s);
          },
          py::return_value_policy::reference)
      .def(
          "get_innermost_loops_for",
          [](const LoopNest& self, const BufHandle& b) {
            return self.getAllInnermostLoopsWritingToBuf(b.node());
          },
          py::return_value_policy::reference)
      .def(
          "get_writes_for",
          [](const LoopNest& self, const BufHandle& b) {
            return self.getAllWritesToBuf(b.node());
          },
          py::return_value_policy::reference)
      .def(
          "get_loop_at",
          [](const LoopNest& self,
             ForPtr root,
             const std::vector<int>& indices) {
            return self.getLoopAt(std::move(root), indices);
          },
          py::return_value_policy::reference)
      .def(
          "get_parent_loop",
          [](const LoopNest& self, const StmtPtr& s) {
            return self.getParentLoop(s);
          },
          py::return_value_policy::reference)
      .def_static(
          "get_loop_stmts_in_loopnest",
          [](const ForPtr& f, size_t num) {
            return LoopNest::getLoopStmtsInLoopNest(f, num);
          },
          py::return_value_policy::reference)
      .def(
          "split_with_tail",
          [](const ForPtr& f, int factor) {
            ForPtr inner = nullptr, tail = nullptr;
            LoopNest::splitWithTail(f, factor, &inner, &tail);
            return std::make_tuple(std::move(inner), std::move(tail));
          },
          py::return_value_policy::reference)
      .def(
          "split_with_mask",
          [](const ForPtr& f, int factor) {
            ForPtr inner = nullptr;
            LoopNest::splitWithMask(f, factor, &inner);
            return inner;
          },
          py::return_value_policy::reference)
      .def(
          "slice_head",
          [](const ForPtr& f, int factor) {
            ForPtr head = nullptr, tail = nullptr;
            LoopNest::sliceHead(f, factor, &head, &tail);
            return std::make_tuple(std::move(head), std::move(tail));
          },
          py::return_value_policy::reference)
      .def(
          "slice_tail",
          [](const ForPtr& f, int factor) {
            ForPtr head = nullptr, tail = nullptr;
            LoopNest::sliceTail(f, factor, &head, &tail);
            return std::make_tuple(std::move(head), std::move(tail));
          },
          py::return_value_policy::reference)
      .def_static(
          "normalize",
          [](const ForPtr& f) {
            LoopNest::normalize(f);
            return f;
          },
          py::return_value_policy::reference)
      .def(
          "tile",
          [](LoopNest& self,
             const ForPtr& x,
             const ForPtr& y,
             int x_factor,
             int y_factor) { return self.tile(x, y, x_factor, y_factor); },
          py::return_value_policy::reference)
      .def_static(
          "distribute_loop",
          [](const ForPtr& f) { return LoopNest::distributeLoop(f); },
          py::return_value_policy::reference)
      .def_static(
          "distribute_loop",
          [](const ForPtr& f, const std::unordered_set<StmtPtr>& pivots) {
            return LoopNest::distributeLoop(f, pivots);
          },
          py::return_value_policy::reference)
      .def_static(
          "distribute_loop_over_inner_loops",
          [](const ForPtr& f) {
            return LoopNest::distributeLoopOverInnerLoops(f);
          },
          py::return_value_policy::reference)
      .def_static(
          "unsafe_fuse_loops",
          [](const std::vector<ForPtr>& loops) {
            ForPtr fused_loop = nullptr;
            LoopNest::unsafeFuseLoops(loops, &fused_loop);
            return fused_loop;
          },
          py::return_value_policy::reference)
      .def_static(
          "fuse_loops",
          [](const std::vector<ForPtr>& loops) {
            ForPtr fused_loop = nullptr;
            LoopNest::fuseLoops(loops, &fused_loop);
            return fused_loop;
          },
          py::return_value_policy::reference)
      .def_static(
          "reorder",
          [](const std::vector<ForPtr>& loops,
             const std::vector<size_t>& permutation) {
            return LoopNest::reorder(loops, permutation);
          },
          py::return_value_policy::reference)
      .def(
          "fullUnroll",
          [](const ForPtr& f) {
            StmtPtr unrolled = nullptr;
            LoopNest::fullUnroll(f, &unrolled);
            return unrolled;
          },
          py::return_value_policy::reference)
      .def(
          "unroll",
          [](const ForPtr& f, int factor) {
            LoopNest::unroll(f, factor);
            return f;
          },
          py::return_value_policy::reference)
      .def(
          "vectorize",
          [](const ForPtr& f) { LoopNest::vectorize(f); },
          py::return_value_policy::reference)
      .def_static(
          "compress_buffer",
          [](BufHandle& buf, const StmtPtr& stmt) {
            return LoopNest::compressBuffer(buf.node(), stmt);
          },
          py::return_value_policy::reference)
      .def_static(
          "cache_accesses",
          [](const BufHandle& producer,
             const std::string& name,
             const StmtPtr& consumer) {
            std::pair<BufPtr, StmtPtr> ret =
                LoopNest::cacheAccesses(producer.node(), name, consumer);
            return std::make_pair(BufHandle(ret.first), ret.second);
          },
          py::return_value_policy::reference)
      .def_static(
          "compute_at",
          [](const StmtPtr& s, const ForPtr& at) {
            LoopNest::computeAt(s, at);
          })
      .def(
          "compute_inline",
          [](LoopNest& self, const StmtPtr& s) { self.computeInline(s); },
          py::return_value_policy::reference)
      .def(
          "compute_inline",
          [](LoopNest& self, const BufHandle& b) {
            self.computeInline(b.node());
          },
          py::return_value_policy::reference)
      .def(
          "rfactor",
          [](const StmtPtr& s, const ForPtr& target_for) {
            BufPtr rfac_buf = nullptr;
            LoopNest::rfactor(s, target_for, &rfac_buf);
            return BufHandle(rfac_buf);
          },
          py::return_value_policy::reference)
      .def(
          "flatten",
          [](LoopNest& self, const std::vector<ForPtr>& loops) {
            ForPtr flattened = nullptr;
            LoopNest::flatten(loops, &flattened);
            return flattened;
          },
          py::return_value_policy::reference)
      .def(
          "reorder_axis",
          &LoopNest::reorderAxis,
          py::return_value_policy::reference)
      .def("simplify", &LoopNest::simplify, py::return_value_policy::reference)
      .def_static("sanitize_names", &LoopNest::sanitizeNames)
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
      [](const StmtPtr& stmt) { return IRSimplifier::simplify(stmt); },
      py::return_value_policy::reference);

  te.def(
      "lower",
      [](const std::string& op_str,
         const py::list& inputs,
         const std::vector<ExprHandle>& outputShape,
         Dtype outputType) {
        auto op = c10::Symbol::fromQualString(op_str);
        std::vector<ArgValue> argInputs;
        for (auto inp : inputs) {
          argInputs.push_back(convertPyToArgValue(inp));
        }
        if (NNCLoweringFunction lowering =
                getStandardLoweringFor(op.toQualString())) {
          std::vector<ExprHandle> outputStrides =
              c10::fmap<ExprHandle>(make_channels_last_strides(outputShape));
          return lowering(
              argInputs,
              outputShape,
              outputStrides,
              outputType.scalar_type(),
              at::kCPU);
        }
        std::string msg = std::string("Unhandled node kind (in te.lower): ") +
            op.toQualString();
        throw malformed_input(msg);
      });

  py::class_<ArgValue>(te, "ArgValue")
      .def(py::init([](py::handle inp) {
        return std::make_unique<ArgValue>(convertPyToArgValue(inp));
      }))
      .def(
          "as_buf",
          [](const ArgValue& self) { return std::get<BufHandle>(self); })
      .def(
          "as_var",
          [](const ArgValue& self) { return std::get<VarHandle>(self); })
      .def(
          "as_float",
          [](const ArgValue& self) { return std::get<double>(self); })
      .def(
          "as_int",
          [](const ArgValue& self) { return std::get<int64_t>(self); })
      .def("as_bool", [](const ArgValue& self) { return std::get<bool>(self); })
      .def(
          "as_none",
          [](const ArgValue& self) { return std::get<ArgNone>(self); })
      .def(
          "as_buflist",
          [](const ArgValue& self) { return std::get<BufList>(self); })
      .def("as_intlist", [](const ArgValue& self) {
        return std::get<IntList>(self);
      });

  py::class_<c10::ScalarType> give_me_a_name(te, "ScalarType");

  using TSGraph = std::shared_ptr<Graph>;
  py::class_<TensorExprKernel>(te, "TensorExprKernel")
      .def(py::init<const TSGraph&>())
      .def(
          py::init(
              [](const TSGraph& g,
                 const std::unordered_map<std::string, NNCLoweringFunction>&
                     custom_lowerings_str,
                 std::vector<int64_t> symbolic_shape_inputs,
                 bool pre_alloc = false) {
                std::unordered_map<c10::Symbol, NNCLoweringFunction>
                    custom_lowerings;
                custom_lowerings.reserve(custom_lowerings_str.size());
                for (auto& kv : custom_lowerings_str) {
                  custom_lowerings[c10::Symbol::fromQualString(kv.first)] =
                      kv.second;
                }
                return std::make_unique<TensorExprKernel>(
                    g,
                    std::move(custom_lowerings),
                    std::move(symbolic_shape_inputs),
                    pre_alloc);
              }),
          py::arg("g"),
          py::arg("custom_lowerings_str"),
          py::arg("symbolic_shape_inputs") = std::vector<int64_t>(),
          py::arg("pre_alloc") = false)
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
          py::arg("attr") = "")
      .def("recompile", [](TensorExprKernel& self) { self.recompile(); });

  py::class_<CodeGen>(te, "CodeGen")
      .def(
          "call",
          [](CodeGen& self, const py::sequence& values) {
            std::vector<CodeGen::CallArg> value_ptrs;
            value_ptrs.reserve(py::len(values));
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
            for (const auto& value : values) {
              if (py::isinstance<py::int_>(value)) {
                value_ptrs.emplace_back(value.cast<int64_t>());
              } else {
                value_ptrs.emplace_back(value.cast<at::Tensor>().data_ptr());
              }
            }
#elif __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
            if (py::len(values) != self.buffer_args().size()) {
              throw malformed_input("bad args in CodeGen.call function");
            }
            for (size_t i = 0; i < py::len(values); i++) {
              const auto& value = values[i];
              const auto& bufArg = self.buffer_args()[i];
              if (py::isinstance<py::int_>(value)) {
                if (!bufArg.isVar()) {
                  throw malformed_input(
                      "Integer variable expected in CodeGen.call function");
                }
                switch (bufArg.dtype().scalar_type()) {
#define TYPE_CASE(Type, Name)                    \
  case ScalarType::Name: {                       \
    value_ptrs.emplace_back(value.cast<Type>()); \
    break;                                       \
  }
                  AT_FORALL_INT_TYPES(TYPE_CASE);
                  default:
                    throw unsupported_dtype();
                }
              } else {
                value_ptrs.emplace_back(value.cast<at::Tensor>().data_ptr());
              }
            }
#else
#error Unexpected or undefined __BYTE_ORDER__
#endif
            self.call(value_ptrs);
          })
      .def(
          "call_raw",
          [](CodeGen& self, const py::sequence& values) {
            std::vector<void*> value_ptrs;
            value_ptrs.reserve(py::len(values));
            for (const auto& value : values) {
              // Tensor.data_ptr() returns an int in python
              value_ptrs.emplace_back(
                  reinterpret_cast<void*>(value.cast<intptr_t>()));
            }
            self.call_raw(value_ptrs);
          })
      .def(
          "get_code_text",
          [](CodeGen& self, const std::string& attr = "") {
            return self.getCodeText(attr);
          },
          py::arg("attr") = "");
  // NOLINTNEXTLINE(bugprone-unused-raii)
  py::class_<SimpleIREvaluator, CodeGen>(te, "SimpleIREvaluator");
#ifdef TORCH_ENABLE_LLVM
  py::class_<LLVMCodeGen, CodeGen>(te, "LLVMCodeGen");
#endif

  py::class_<CodeGen::BufferArg>(te, "BufferArg")
      .def(py::init<Tensor>())
      .def(py::init<const VarHandle&>())
      .def(py::init<const BufHandle&>());

  py::implicitly_convertible<Tensor, CodeGen::BufferArg>();
  py::implicitly_convertible<VarHandle, CodeGen::BufferArg>();
  py::implicitly_convertible<BufHandle, CodeGen::BufferArg>();

  te.def(
      "construct_codegen",
      [](const std::string& name,
         const StmtPtr& stmt,
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
        } else if (name == "ir_eval") {
          cg = new SimpleIREvaluator(stmt, args);
        } else {
          throw std::runtime_error(
              "construct_codegen() expects 'llvm', 'cuda', or 'ir_eval'");
        }
        return cg;
      });
  te.def("annotate_input_shapes", &tensorexpr::annotateInputShapes);
  te.def("remove_unused_self_argument", &tensorexpr::removeUnusedSelfArgument);
  te.def("make_shapes_symbolic", &tensorexpr::makeShapesSymbolic);
  te.def("is_graph_compilable", &tensorexpr::isGraphCompilable);
  te.def("fixup_missing_shape_info", &tensorexpr::fixupMissingShapeInfo);
  te.def("remove_graph_output", &tensorexpr::removeGraphOutput);
  te.def(
      "replace_list_output_with_tuple",
      &tensorexpr::replaceListOutputWithTuple);
  te.def("trim_graph", &tensorexpr::trimGraph);
#ifdef TORCH_ENABLE_LLVM
  te.def("set_llvm_target_triple", [](const std::optional<std::string>& val) {
    tensorexpr::LLVMTargetTriple() = val;
  });
  te.def("set_llvm_target_cpu", [](const std::optional<std::string>& val) {
    tensorexpr::LLVMTargetCPU() = val;
  });
  te.def("set_llvm_target_attrs", [](const std::optional<std::string>& val) {
    tensorexpr::LLVMTargetAttrs() = val;
  });
  te.def("set_llvm_aot_workflow", [](bool val) {
    tensorexpr::LLVMAOTWorkflow() = val;
  });
#endif
}

} // namespace torch::jit
