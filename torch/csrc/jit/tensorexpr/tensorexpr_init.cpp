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
#include <array>
#include <cassert>
#include <map>
#include <mutex>

#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)
#define AT __FILE__ ":" TOSTRING(__LINE__)
#define AA(test) \
  if (!(test))   \
  throw std::runtime_error("assert failed " AT)

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
    throw std::runtime_error("conversion not yet implemented");
  }
}

template <int MAX_DIMS>
class SpecializationKey {
  enum DimFlags {
    SIZE_MISSING = 1 << 0, // leading dimension implicitly added
    SIZE_ONE = 1 << 1, // == 1
    SIZE_OTHER = 1 << 2, // > 1

    STRIDE_ZERO = 1 << 3, // == 0 (broadcast)
    STRIDE_ONE = 1 << 4, // == 1 (packed)
    STRIDE_CONTIGUOUS = 1 << 5, // stride[i+1] * sizes[i+1]
    STRIDE_TRANSPOSED_CONTIGUOUS = 1 << 6, // stride[i-1] * sizes[i-1]
    STRIDE_OTHER = 1 << 7,
  };
  static constexpr int MASK = (1 << 5) - 1;

  static inline uint16_t pack_flags(const at::Tensor& v) {
    // pack all the tensor properties into a uint16 for fast hash/compare
    static_assert(static_cast<int>(at::ScalarType::NumOptions) <= MASK);
    static_assert(static_cast<int>(at::Layout::NumOptions) <= MASK);
    static_assert(
        static_cast<int>(at::DeviceType::COMPILE_TIME_MAX_DEVICE_TYPES) <=
        MASK);

    at::ScalarType dtype = v.dtype().toScalarType();
    at::DeviceType device = v.device().type();
    at::Layout layout = v.layout();
    bool requires_grad = v.requires_grad();

    return static_cast<uint16_t>(dtype) + (static_cast<uint16_t>(device) << 5) +
        (static_cast<uint16_t>(layout) << 10) +
        (static_cast<uint16_t>(requires_grad) << 15);
  }

  template <typename T>
  inline void init_dimflags(const T& sizes, const T& strides, int64_t ndims) {
    // pack all the properties for each dimension into a uint8
    int out_idx = 0;
    for (int dim = ndims - 1; dim >= 0; --dim) {
      uint8_t flag = (sizes[dim] == 1 ? SIZE_ONE : SIZE_OTHER);
      if (strides[dim] == 0)
        flag |= STRIDE_ZERO;
      else if (strides[dim] == 1)
        flag |= STRIDE_ONE;
      else if (
          dim + 1 < sizes.size() &&
          strides[dim] == strides[dim + 1] * sizes[dim + 1])
        flag |= STRIDE_CONTIGUOUS;
      else if (dim > 0 && strides[dim] == strides[dim - 1] * sizes[dim - 1])
        flag |= STRIDE_TRANSPOSED_CONTIGUOUS;
      else
        flag |= STRIDE_OTHER;
      dimflags_[out_idx++] = flag;
    }
    while (out_idx < MAX_DIMS)
      dimflags_[out_idx++] = SIZE_MISSING | STRIDE_ZERO;
  }

 public:
  SpecializationKey(const at::Tensor& v, int8_t alias_group)
      : flags_(pack_flags(v)), alias_group_(alias_group) {
    init_dimflags(v.sizes(), v.strides(), v.ndimension());
  }

  int ndims() const {
    // this can be slow/O(MAX_DIMS) as it is only used in codegen
    int d = MAX_DIMS;
    while (d > 0 && (dimflags_[d - 1] & SIZE_MISSING) > 0)
      --d;
    return d;
  }

  at::ScalarType dtype() const {
    return static_cast<at::ScalarType>(flags_ & MASK);
  }
  at::DeviceType device() const {
    return static_cast<at::DeviceType>((flags_ >> 5) & MASK);
  }
  at::Layout layout() const {
    return static_cast<at::Layout>((flags_ >> 10) & MASK);
  }
  bool requires_grad() const {
    return static_cast<bool>(flags_ >> 15);
  }

  int cmp(const SpecializationKey<MAX_DIMS>& other) const {
    return memcmp(
        &flags_,
        &other.flags_,
        sizeof(flags_) + sizeof(alias_group_) + sizeof(dimflags_));
  }

  struct Less {
    template <typename T>
    size_t operator()(const T& left, const T& right) const {
      for (int i = 0; i < left.size(); ++i) {
        auto c = left[i].cmp(right[i]);
        if (c < 0)
          return true;
        if (c > 0)
          return false;
      }
      return false;
    }
  };

 protected:
  uint16_t flags_; // all the flags packed together
  int8_t alias_group_; // 0 = no aliasing
                       // >0 = same data, strides, and shapes with group
                       // <0 = overlapping storage madness
  uint8_t dimflags_[MAX_DIMS]; // stored in reverse order
} __attribute__((packed));

template <NARGS, MAX_DIMS>
class CompileCache3 {
  typedef SpecializationKey<MAX_DIMS> ArgKey;
  typedef std::array<ArgKey, NARGS> Key;
  typedef std::map<Key, CodeGen*> Map;

 public:
  CompileCache3(py::handle compile_fn) : compile_fn_(compile_fn) {}

  CodeGen* cached_compile(const Key& key) {
    std::lock_guard<std::mutex> guard(mutex_);
    auto item = cache.find(key);
    if (item != cache.end()) {
      return item->second;
    } else {
      cache[key] = compile_fn_();
    }
  }

  at::Tensor call(const st::array<at::Tensor*, NARGS>& args) {
    std::vector<void*> args;
    CodeGen* cg = cached_compile(key);
    cg->call_raw(args)

    int64_t n = a.sizes()[0];
    int64_t shapes[] = {n};
    int64_t strides[] = {1};
    at::Tensor out = at::empty_strided(shapes, strides);
    std::vector<void*> args = {a.data_ptr(), b.data_ptr(), out.data_ptr(), &n};
    self.call_raw(args);
  }

 private:
  std::mutex mutex_;
  Map cache_;
  py::object compile_fn_;
};

template <NARGS>
class CompileCache2 {
 public:
  CompileCache2(py::handle compile_fn) :
    cache2(compile_fn),
    cache4(compile_fn),
    cache8(compile_fn)
    {}

  at::Tensor call(const st::array<at::Tensor*, NARGS>& args) {
    // fan out and and specialize on number of dimension buckets
    int64_t ndims = 0;
    for (auto item : args) {
      ndims = std::max(item->ndimension(), ndims);
    }
    if (ndims <= 2)
      return cache2.call(args) if (ndims <= 4) return cache4
          .call(args) if (ndims <= 8) return cache8.call(
              args) throw sts::runtime_error("TODO: handle more dims")
  }

 private:
  CompileCache3<NARGS, 2> cache2;
  CompileCache3<NARGS, 4> cache4;
  CompileCache3<NARGS, 8> cache8;
};

class CompileCache : public KernelScopedObject {
 public:
  CompileCache(py::handle compile_fn) :
    cache1(compile_fn),
    cache2(compile_fn),
    cache3(compile_fn),
    cache4(compile_fn)
    {}

  at::Tensor call(py::args, py::kwargs) {
    // fan out an specialize on arg counts
    if (kwargs) {
      throw std::runtime_error("TODO: handle `out=` etc")
    }
    int nargs = py::len(args);
    if (nargs == 1)
      return cache1.call({args[0].cast<at::Tensor*>()});
    if (nargs == 2)
      return cache2.call({
          args[0].cast<at::Tensor*>(),
          args[1].cast<at::Tensor*>(),
      });
    if (nargs == 3)
      return cache3.call({
          args[0].cast<at::Tensor*>(),
          args[1].cast<at::Tensor*>(),
          args[2].cast<at::Tensor*>(),
      });
    if (nargs == 4)
      return cache4.call({
          args[0].cast<at::Tensor*>(),
          args[1].cast<at::Tensor*>(),
          args[2].cast<at::Tensor*>(),
          args[3].cast<at::Tensor*>(),
      });
    throw std::runtime_error("TODO: handle other arg counts")
  }

 private:
  CompileCache2<1> cache1;
  CompileCache2<2> cache2;
  CompileCache2<3> cache3;
  CompileCache2<4> cache4;
};

at::Tensor
call_jansel(CodeGen& self, const at::Tensor& a, const at::Tensor& b) {
  typedef SpecializationKey<2> K2;
  typedef std::array<K2, 2> A;

  static std::map<A, void*, K2::Less> cache;
  static std::mutex mutex;
  std::lock_guard<std::mutex> guard(mutex);

  A key = {K2(a, 0), K2(b, 0)};
  auto item = cache.find(key);
  if (item != cache.end()) {
    AA(item->second == nullptr);
  } else {
    TORCH_WARN("codegen");
    cache[key] = nullptr;
  }
  // std::mapstd::array<K2, 2> SpecializationKey<2> akey(a, 0);
  // SpecializationKey<2> bkey(b, 0);
  // AA(akey == bkey);
  // AA(akey.dtype() == a.dtype().toScalarType());
  // AA(akey.device() == a.device().type());
  // AA(akey.layout() == a.layout());
  // AA(akey.requires_grad() == a.requires_grad());
  // AA(bkey.dtype() == b.dtype().toScalarType());
  // AA(bkey.device() == b.device().type());
  // AA(bkey.layout() == b.layout());
  // AA(bkey.requires_grad() == b.requires_grad());

  int64_t n = a.sizes()[0];
  int64_t shapes[] = {n};
  int64_t strides[] = {1};
  at::Tensor out = at::empty_strided(shapes, strides);
  std::vector<void*> args = {a.data_ptr(), b.data_ptr(), out.data_ptr(), &n};
  self.call_raw(args);
  return out;
}

Dtype parsePythonDtype(py::handle obj) {
  if (py::isinstance(obj, py::module_::import("torch").attr("dtype"))) {
    return Dtype(reinterpret_cast<THPDtype*>(obj.ptr())->scalar_type);
  } else {
    throw std::runtime_error("expected a torch.dtype instance");
  }
}

void initTensorExprBindings(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();

  // Tensor Expr Classes
  auto te = m.def_submodule("_te");
  py::class_<KernelScope>(te, "KernelScope").def(py::init<>());

  auto dtype_class =
      py::class_<Dtype>(te, "Dtype").def(py::init(&parsePythonDtype));
  py::implicitly_convertible<py::object, Dtype>();

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
      .def(py::init<Dtype>())
      .def(py::init<const std::string&, Dtype>());
  py::class_<BufHandle, ExprHandle>( // NOLINT
      te,
      "BufHandle")
      .def(
          py::init<const std::string&, const std::vector<ExprHandle>&, Dtype>())
      .def(py::init<const std::vector<ExprHandle>&, Dtype>())
      .def(py::init<Dtype>())
      .def(
          "load",
          [](BufHandle& self, const std::vector<ExprHandle>& v) {
            return Load::make(self, v);
          })
      .def("load", [](BufHandle& self, const ExprHandle& v) {
        return Load::make(self, {v});
      });

  py::class_<Placeholder>(te, "Placeholder")
      .def(py::init<
           const std::string&,
           const Dtype&,
           const std::vector<ExprHandle>&>())
      .def(py::init<const std::vector<ExprHandle>&, const Dtype&>())
      .def(py::init<const std::vector<ExprHandle>&>())
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
  py::implicitly_convertible<ExprHandle, DimArg>();

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

  te.def(
      "Compute2",
      [](const std::string& func_name,
         const std::vector<DimArg>& dim_args,
         py::function func) {
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
          [](const LoopNest& self, BufHandle& b) {
            return self.getLoopBodyFor(b.node());
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
          [](const LoopNest& self, const BufHandle& b) {
            return self.getAllLoopNestsWritingToBuf(b.node());
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
          [](For* f, int factor) {
            For *inner = nullptr, *tail = nullptr;
            LoopNest::splitWithTail(f, factor, &inner, &tail);
            return std::make_tuple(inner, tail);
          },
          py::return_value_policy::reference)
      .def(
          "split_with_mask",
          [](For* f, int factor) {
            For* inner = nullptr;
            LoopNest::splitWithMask(f, factor, &inner);
            return inner;
          },
          py::return_value_policy::reference)
      .def(
          "slice_head",
          [](For* f, int factor) {
            For *head = nullptr, *tail = nullptr;
            LoopNest::sliceHead(f, factor, &head, &tail);
            return std::make_tuple(head, tail);
          },
          py::return_value_policy::reference)
      .def(
          "slice_tail",
          [](For* f, int factor) {
            For *head = nullptr, *tail = nullptr;
            LoopNest::sliceTail(f, factor, &head, &tail);
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
            For* fused_loop = nullptr;
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
          [](For* f) { LoopNest::vectorize(f); },
          py::return_value_policy::reference)
      .def_static(
          "compress_buffer",
          [](BufHandle& buf, Stmt* stmt) {
            return LoopNest::compressBuffer(buf.node(), stmt);
          },
          py::return_value_policy::reference)
      .def(
          "cache_accesses",
          [](const BufHandle& producer,
             const std::string& name,
             Stmt* consumer) {
            std::pair<const Buf*, Stmt*> ret =
                LoopNest::cacheAccesses(producer.node(), name, consumer);
            return std::make_pair(BufHandle(ret.first), ret.second);
          },
          py::return_value_policy::reference)
      .def("compute_at", [](Stmt* s, For* at) { LoopNest::computeAt(s, at); })
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
          [](Stmt* s, For* target_for) {
            Buf* rfac_buf = nullptr;
            LoopNest::rfactor(s, target_for, &rfac_buf);
            return BufHandle(rfac_buf);
          },
          py::return_value_policy::reference)
      .def(
          "flatten",
          [](const std::vector<For*>& loops) {
            For* flattened = nullptr;
            LoopNest::flatten(loops, &flattened);
            return flattened;
          },
          py::return_value_policy::reference)
      .def(
          "reorder_axis",
          &LoopNest::reorderAxis,
          py::return_value_policy::reference)
      .def("simplify", &LoopNest::simplify, py::return_value_policy::reference)
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
          [](CodeGen& self, const py::sequence& values) {
            std::vector<CodeGen::CallArg> value_ptrs;
            value_ptrs.reserve(py::len(values));
            for (const auto& value : values) {
              if (py::isinstance<py::int_>(value)) {
                value_ptrs.emplace_back(value.cast<int64_t>());
              } else {
                value_ptrs.emplace_back(value.cast<at::Tensor>().data_ptr());
              }
            }
            self.call(value_ptrs);
          })
      .def("call_jansel", &call_jansel)
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
  py::class_<SimpleIREvaluator, CodeGen>(te, "SimpleIREvaluator"); // NOLINT
#ifdef TORCH_ENABLE_LLVM
  py::class_<LLVMCodeGen, CodeGen>(te, "LLVMCodeGen"); // NOLINT
#endif

  py::class_<CodeGen::BufferArg>(te, "BufferArg")
      .def(py::init<const Placeholder&>())
      .def(py::init<Tensor*>())
      .def(py::init<const VarHandle&>())
      .def(py::init<const BufHandle&>());

  py::implicitly_convertible<Placeholder, CodeGen::BufferArg>();
  py::implicitly_convertible<Tensor*, CodeGen::BufferArg>();
  py::implicitly_convertible<VarHandle, CodeGen::BufferArg>();
  py::implicitly_convertible<BufHandle, CodeGen::BufferArg>();

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
}
} // namespace jit
} // namespace torch
