#include <pybind11/functional.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/jit/tensorexpr/codegen.h>
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
namespace {
using namespace torch::jit::tensorexpr;

template <int MAX_DIMS>
class SpecializationKey {
 protected:
  enum DimFlags {
    SIZE_MISSING = 1 << 0, // leading dimension implicitly added
    SIZE_ONE = 1 << 1, // == 1
    SIZE_OTHER = 1 << 2, // > 1

    STRIDE_ZERO = 1 << 3, // == 0 (broadcast)
    STRIDE_ONE = 1 << 4, // == 1 (packed)
    STRIDE_CONTIGUOUS = 1 << 5, // stride[i+1] * sizes[i+1]
    STRIDE_TRANSPOSED_CONTIGUOUS = 1 << 6, // stride[i-1] * sizes[i-1]
    STRIDE_AS_ARG = 1 << 7,
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
    while (out_idx < MAX_DIMS - ndims)
      dimflags_[out_idx++] = SIZE_MISSING | STRIDE_ZERO;

    for (int dim = 0; dim < ndims; ++dim) {
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
        flag |= STRIDE_AS_ARG;
      dimflags_[out_idx++] = flag;
    }
  }

 public:
  SpecializationKey() {}

  SpecializationKey(const at::Tensor& v, int8_t alias_group)
      : flags_(pack_flags(v)), alias_group_(alias_group) {
    init_dimflags(v.sizes(), v.strides(), v.ndimension());
  }

  int cmp(const SpecializationKey<MAX_DIMS>& other) const {
    return memcmp(
        &flags_,
        &other.flags_,
        sizeof(flags_) + sizeof(alias_group_) + sizeof(dimflags_));
  }

  std::vector<std::string> shape() const {
    std::vector<std::string> result;
    for (int i = 0; i < MAX_DIMS; ++i) {
      if ((dimflags_[i] & SIZE_MISSING) > 0)
        break;

      if ((dimflags_[i] & SIZE_ONE) > 0)
        result.push_back("one");
      else
        result.push_back("other");
    }
    return result;
  }
  std::vector<std::string> stride() const {
    std::vector<std::string> result;
    for (int i = 0; i < MAX_DIMS; ++i) {
      if ((dimflags_[i] & SIZE_MISSING) > 0)
        break;

      if ((dimflags_[i] & STRIDE_ZERO) > 0)
        result.push_back("zero");
      else if ((dimflags_[i] & STRIDE_ONE) > 0)
        result.push_back("one");
      else if ((dimflags_[i] & STRIDE_CONTIGUOUS) > 0)
        result.push_back("contiguous");
      else if ((dimflags_[i] & STRIDE_TRANSPOSED_CONTIGUOUS) > 0)
        result.push_back("transposed_contiguous");
      else if ((dimflags_[i] & STRIDE_AS_ARG) > 0)
        result.push_back("as_arg");
      else
        throw std::runtime_error("??");
    }
    return result;
  }

  py::object to_python(const at::Tensor& example) const {
    py::object ex = py::cast(example);
    py::object namedtuple =
        py::module_::import("collections").attr("namedtuple");
    py::object rtype = namedtuple(
        "SpecializationKey",
        "alias_group,dim,dtype,device,layout,requires_grad,shape,stride");
    return rtype(
        static_cast<int>(alias_group_),
        ex.attr("dim"),
        ex.attr("dtype"),
        ex.attr("device"),
        ex.attr("layout"),
        ex.attr("requires_grad"),
        shape(),
        stride());
  }

 private:
  uint16_t flags_; // dtype, layout, device, and requires_grad
  int8_t alias_group_; // 0 = no aliasing
                       // >0 = same data, strides, and shapes within group
                       // <0 = overlapping storage madness
  uint8_t dimflags_[MAX_DIMS];
} __attribute__((packed));

class CompileOptions {
 public:
  virtual ~CompileOptions() = default;
};

struct CompileOptionsProxy {
  py::object spec;
  CompileOptions* options;
};

struct CmpLess {
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

template <int NARGS, int MAX_DIMS>
class CompileCache3 {
 public:
  typedef SpecializationKey<MAX_DIMS> ArgKey;
  typedef std::array<ArgKey, NARGS> Key;
  typedef std::array<at::Tensor, NARGS> Args;
  typedef std::array<int8_t, NARGS> AliasGroups;

  class CompileResult {
   public:
    void set_codegen(py::object cg) {
      objects_.push_back(cg);
      cg_ = cg.cast<CodeGen*>();
    }
    at::Tensor call(const Args& args, std::vector<void*>& call_args) {
      cg_->call_raw(call_args);
      return args[2];
    }

   private:
    CodeGen* cg_;
    std::vector<py::object> objects_; // for ref count
  };
  typedef std::map<Key, CompileResult*, CmpLess> Map;

  class CompileOptionsImpl : public CompileOptions {
   public:
    CompileOptionsImpl(const Key& key, CompileResult& result)
        : key_(key), result_(result) {}

   private:
    const Key& key_;
    CompileResult& result_;
  };

  CompileResult* cached_compile(const Key& key, const Args& example) {
    std::lock_guard<std::mutex> guard(mutex_);
    auto item = cache_.find(key);
    if (item != cache_.end()) {
      return item->second;
    } else {
      auto cr = new CompileResult();
      CompileOptionsImpl opts(key, *cr);
      cr->set_codegen(compile_fn_(
          CompileOptionsProxy({key[0].to_python(example[0]), &opts})));
      cache_.emplace(std::make_pair(key, cr));
      return cr;
    }
  }

  int8_t aliasing_check(const at::Tensor& a, const at::Tensor& b) {
    if (a.is_alias_of(b)) {
      if (a.is_set_to(b)) {
        return 1;
      } else {
        // TODO: check for non-overlapping and return 0
        //       likely we could lift some logic from tensoriterator
        return -1;
      }
    } else {
      return 0;
    }
  }

  AliasGroups compute_alias_groups(const Args& args) {
    AliasGroups alias_groups;
    int8_t alias_group = 0;
    for (int i = 0; i < NARGS; ++i) {
      for (int j = i + 1; j < NARGS; ++j) {
        int8_t alias_type = aliasing_check(args[i], args[j]);
        if (alias_type != 0) {
          if (alias_groups[i] == 0)
            ++alias_group;
          alias_groups[i] = alias_group;
          alias_groups[j] = alias_group * alias_type;
        }
      }
    }
    return alias_groups;
  }

  Key compute_cache_key(const Args& args) {
    AliasGroups alias_groups = compute_alias_groups(args);
    Key key;
    for (int i = 0; i < NARGS; ++i) {
      key[i] = ArgKey(args[i], alias_groups[i]);
    }
    return key;
  }

  CompileCache3(const py::object& compile_fn) : compile_fn_(compile_fn) {}

  at::Tensor call(const Args& args) {
    std::vector<void*> call_args;
    call_args.reserve(NARGS + NARGS * MAX_DIMS);
    for (const auto& arg : args) {
      call_args.emplace_back(arg.data_ptr());
    }
    auto key = compute_cache_key(args);
    return cached_compile(key, args)->call(args, call_args);
    /*
    int64_t n = a.sizes()[0];
    int64_t shapes[] = {n};
    int64_t strides[] = {1};
    at::Tensor out = at::empty_strided(shapes, strides);
    std::vector<void*> args = {a.data_ptr(), b.data_ptr(), out.data_ptr(), &n};
    self.call_raw(args);
    */
  }

 public:
  std::mutex mutex_;
  Map cache_;
  py::object compile_fn_;
};

template <int NARGS>
class CompileCache2 {
 public:
  CompileCache2(const py::object& compile_fn)
      : cache2(compile_fn), cache4(compile_fn), cache8(compile_fn) {}

  at::Tensor call(const std::array<at::Tensor, NARGS>& args) {
    // fan out and and specialize on number of dimension buckets
    int64_t ndims = 0;
    for (const auto& item : args) {
      ndims = std::max(item.ndimension(), ndims);
    }
    if (ndims <= 2)
      return cache2.call(args);
    if (ndims <= 4)
      return cache4.call(args);
    if (ndims <= 8)
      return cache8.call(args);
    throw std::runtime_error("TODO: handle more dims");
  }

 private:
  CompileCache3<NARGS, 2> cache2;
  CompileCache3<NARGS, 4> cache4;
  CompileCache3<NARGS, 8> cache8;
};

class CompileCache {
 public:
  CompileCache(const py::object& compile_fn)
      : cache1(compile_fn),
        cache2(compile_fn),
        cache3(compile_fn),
        cache4(compile_fn) {}

  at::Tensor call(py::args args, py::kwargs kwargs) {
    // fan out an specialize on arg counts
    if (py::len(kwargs) > 0) {
      throw std::runtime_error("TODO: handle `out=` etc");
    }
    int nargs = py::len(args);
    if (nargs == 1)
      return cache1.call(std::array<at::Tensor, 1>{args[0].cast<at::Tensor>()});
    if (nargs == 2)
      return cache2.call(std::array<at::Tensor, 2>{
          args[0].cast<at::Tensor>(),
          args[1].cast<at::Tensor>(),
      });
    if (nargs == 3)
      return cache3.call(std::array<at::Tensor, 3>{
          args[0].cast<at::Tensor>(),
          args[1].cast<at::Tensor>(),
          args[2].cast<at::Tensor>(),
      });
    if (nargs == 4)
      return cache4.call(std::array<at::Tensor, 4>{
          args[0].cast<at::Tensor>(),
          args[1].cast<at::Tensor>(),
          args[2].cast<at::Tensor>(),
          args[3].cast<at::Tensor>(),
      });
    throw std::runtime_error("TODO: handle other arg counts");
  }

 private:
  CompileCache2<1> cache1;
  CompileCache2<2> cache2;
  CompileCache2<3> cache3;
  CompileCache2<4> cache4;
};

/*
class KeyProxy {
 public:
  virtual ~KeyProxy() = default;
  virtual int ndims() const = 0;
  virtual at::ScalarType dtype() const = 0;
  virtual at::DeviceType device() const = 0;
  virtual at::Layout layout() const = 0;
  virtual bool requires_grad() const = 0;
};

template <int MAX_DIMS>
class KeyProxyImpl : public KeyProxy, public SpecializationKey<MAX_DIMS> {
 public:
  KeyProxyImpl(const SpecializationKey<MAX_DIMS>& key)
      : SpecializationKey<MAX_DIMS>(key) {}

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
};

at::Tensor call_jansel(
    CodeGen& self,
    const at::Tensor& a,
    const at::Tensor& b) {
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
*/
} // namespace

void initTensorExprAuthoringBindings(PyObject* te_obj) {
  py::handle te(te_obj);

  py::class_<CompileCache>(te, "CompileCache")
      .def(py::init<py::object>())
      .def("__call__", &CompileCache::call);

  py::class_<CompileOptionsProxy>(te, "CompileOptions");
}
} // namespace jit
} // namespace torch
