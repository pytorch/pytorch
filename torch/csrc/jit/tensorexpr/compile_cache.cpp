#include <torch/csrc/autograd/custom_function.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/jit/tensorexpr/codegen.h>
#include <torch/csrc/jit/tensorexpr/compile_cache.h>
#include <torch/csrc/jit/tensorexpr/cuda_codegen.h>
#include <array>
#include <map>
#include <mutex>

namespace torch {
namespace jit {
namespace {
using namespace torch::jit::tensorexpr;

class CompileCache;
typedef torch::autograd::variable_list variable_list;
typedef std::tuple<int, CompileCache*> CompileCacheBackwards;

class CompiledAutoGradNode : public torch::autograd::Node {
 public:
  CompiledAutoGradNode() = default;

  variable_list apply(variable_list&& new_inputs) override;

  void release_variables() override {
    inputs_.clear();
  }

  void setup(
      std::vector<CompileCacheBackwards>& backwards,
      at::Tensor* args,
      size_t len) {
    inputs_.reserve(len);
    for (int i = 0; i < len; ++i) {
      inputs_.emplace_back(args[i].detach());
    }

    // node outputs
    backwards_functions_.reserve(backwards.size());
    torch::autograd::edge_list next_edges;
    for (auto& item : backwards) {
      backwards_functions_.emplace_back(item);
      next_edges.emplace_back(
          torch::autograd::impl::gradient_edge(args[std::get<0>(item)]));
    }
    set_next_edges(std::move(next_edges));
  }

 private:
  std::vector<CompileCacheBackwards> backwards_functions_;
  std::vector<at::Tensor> inputs_;
};

static py::object python_specialization_key() {
  // namedtuple() we map SpecializationKey to
  static py::object* rv = nullptr; // NOLINT
  if (rv == nullptr) {
    // create it lazily
    py::object namedtuple =
        py::module_::import("collections").attr("namedtuple");
    rv = new py::object();
    *rv = namedtuple(
        "SpecializationKey",
        "alias_group,ndim,dtype,device,layout,requires_grad,out,shape,stride");
  }
  return *rv;
}

struct LocalState {
  // TLS state that changes operators
  c10::impl::LocalDispatchKeySet dispatch_modifier;
  bool grad_mode_enabled;

  at::DispatchKeySet apply(at::DispatchKeySet ks) const {
    return (ks | dispatch_modifier.included_) - dispatch_modifier.excluded_;
  }

  LocalState()
      : dispatch_modifier(c10::impl::tls_local_dispatch_key_set()),
        grad_mode_enabled(at::GradMode::is_enabled()) {}
};

template <int MAX_DIMS>
struct SpecializationKey {
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

  static inline uint16_t pack_flags(
      const LocalState& state,
      const at::Tensor& v,
      bool is_out) {
    static_assert(
        static_cast<int>(at::ScalarType::NumOptions) < 64, "overflow possible");
    at::ScalarType dtype = v.dtype().toScalarType();
    bool requires_grad = state.grad_mode_enabled && v.requires_grad();
    return static_cast<uint8_t>(is_out) |
        (static_cast<uint8_t>(requires_grad) << 1) |
        (static_cast<uint8_t>(dtype) << 2);
  }

  template <typename T>
  inline void init_dimflags(const T& sizes, const T& strides, int64_t ndims) {
    // pack all the properties for each dimension into a uint8
    for (int dim = 0; dim < MAX_DIMS; ++dim) {
      if (dim < ndims) {
        uint8_t flag = (sizes[dim] == 1 ? SIZE_ONE : SIZE_OTHER);
        if (strides[dim] == 0) {
          flag |= STRIDE_ZERO;
        } else if (strides[dim] == 1) {
          flag |= STRIDE_ONE;
        } else if (
            dim + 1 < sizes.size() &&
            strides[dim] == strides[dim + 1] * sizes[dim + 1]) {
          flag |= STRIDE_CONTIGUOUS;
        } else if (
            dim > 0 && strides[dim] == strides[dim - 1] * sizes[dim - 1] &&
            (dimflags_[dim - 1] & STRIDE_CONTIGUOUS) == 0) {
          flag |= STRIDE_TRANSPOSED_CONTIGUOUS;
        } else {
          flag |= STRIDE_AS_ARG;
        }
        dimflags_[dim] = flag;
      } else {
        dimflags_[dim] = SIZE_MISSING | STRIDE_ZERO;
      }
    }
  }

 public:
  SpecializationKey() {} // NOLINT: intentionally not initialized

  // NOLINTNEXTLINE: intentionally not initializing dimflags_
  SpecializationKey(
      const LocalState& state,
      const at::Tensor& v,
      int8_t alias_group,
      bool is_out)
      : flags_(pack_flags(state, v, is_out)),
        alias_group_(alias_group),
        dispatch_key_(state.apply(v.key_set()).raw_repr()) {
    init_dimflags(v.sizes(), v.strides(), v.ndimension());
  }

  int cmp(const SpecializationKey<MAX_DIMS>& other) const {
    return memcmp(
        &flags_,
        &other.flags_,
        sizeof(flags_) + sizeof(alias_group_) + sizeof(dispatch_key_) +
            sizeof(dimflags_));
  }

  void clear() {
    memset(
        &flags_,
        0,
        sizeof(flags_) + sizeof(alias_group_) + sizeof(dispatch_key_) +
            sizeof(dimflags_));
  }

  DispatchKeySet dispatch_key() const {
    return DispatchKeySet(DispatchKeySet::RAW, dispatch_key_);
  }

  std::vector<std::string> shape() const {
    std::vector<std::string> result;
    for (int i = 0; i < MAX_DIMS; ++i) {
      if ((dimflags_[i] & SIZE_MISSING) > 0)
        break;

      if ((dimflags_[i] & SIZE_ONE) > 0)
        result.emplace_back("one");
      else
        result.emplace_back("other");
    }
    return result;
  }
  std::vector<std::string> stride() const {
    std::vector<std::string> result;
    for (int i = 0; i < MAX_DIMS; ++i) {
      if ((dimflags_[i] & SIZE_MISSING) > 0)
        break;

      if ((dimflags_[i] & STRIDE_ZERO) > 0)
        result.emplace_back("zero");
      else if ((dimflags_[i] & STRIDE_ONE) > 0)
        result.emplace_back("one");
      else if ((dimflags_[i] & STRIDE_CONTIGUOUS) > 0)
        result.emplace_back("contiguous");
      else if ((dimflags_[i] & STRIDE_TRANSPOSED_CONTIGUOUS) > 0)
        result.emplace_back("transposed_contiguous");
      else if ((dimflags_[i] & STRIDE_AS_ARG) > 0)
        result.emplace_back("as_arg");
      else
        throw std::runtime_error("??");
    }
    return result;
  }

  py::object to_python(const at::Tensor& example) const {
    py::object ex = py::cast(example);
    return python_specialization_key()(
        static_cast<int>(alias_group_),
        ex.attr("ndim"),
        ex.attr("dtype"),
        ex.attr("device"),
        ex.attr("layout"),
        ex.attr("requires_grad"),
        py::bool_(flags_ % 2), // out
        shape(),
        stride());
  }

 private:
  uint8_t flags_; // is_out, requires_grad, and dtype
  int8_t alias_group_; // 0 = no aliasing
                       // >0 = same data, strides, and shapes within group
                       // <0 = overlapping storage madness
  uint64_t dispatch_key_; // DispatchKeySet includes device/layout
  // NOLINTNEXTLINE: C-style arrays
  uint8_t dimflags_[MAX_DIMS];
} __attribute__((packed));

class CompileResultBase : public KernelScopedObject {
 public:
  ~CompileResultBase() override = default;
  virtual void set_code(const py::object& cg) = 0;
  virtual void set_shape_from(
      const std::vector<std::pair<int, int>>& indices) = 0;
  virtual void set_stride_args_from(
      const std::vector<std::pair<int, int>>& indices) = 0;
  virtual void add_allocated_output(
      int options_from,
      const std::vector<int>& storage_order) = 0;
  virtual void add_shape_check(
      const std::tuple<int, int, int, int>& indices) = 0;
  virtual void set_backwards(int index, py::object backward_compiler) = 0;
};

struct CompileResultProxy {
  CompileResultBase* res;
  explicit CompileResultProxy(CompileResultBase* r) : res(r) {}
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
class alignas(32) CompileCache3 {
 public:
  typedef std::array<SpecializationKey<MAX_DIMS>, NARGS> SpecializationKeys;
  typedef std::array<int8_t, NARGS> AliasGroups;

  class CompileResultImpl : public CompileResultBase {
   public:
    void set_code(const py::object& cg) override {
      objects_.emplace_back(cg);
      cg_ = cg.cast<CodeGen*>();
    }

    void set_shape_from(
        const std::vector<std::pair<int, int>>& indices) override {
      assert(indices.shape() <= MAX_DIMS);
      shape_from_ = indices;
    }

    void set_stride_args_from(
        const std::vector<std::pair<int, int>>& indices) override {
      assert(indices.shape() <= MAX_DIMS * NARGS);
      stride_args_from_ = indices;
    }

    void add_allocated_output(
        int options_from,
        const std::vector<int>& storage_order) override {
      if (allocated_outputs_.size() > 0) {
        throw std::runtime_error("TODO: support more than one output");
      }
      allocated_outputs_.emplace_back(
          std::make_pair(options_from, storage_order));
    }

    void add_shape_check(
        const std::tuple<int, int, int, int>& indices) override {
      shape_checks_.emplace_back(indices);
    }

    void set_backwards(int index, py::object backward_compiler) override {
      objects_.emplace_back(backward_compiler);
      backwards_functions_.emplace_back(
          std::make_tuple(index, backward_compiler.cast<CompileCache*>()));
    }

    at::Tensor call(at::Tensor* args) {
      for (const auto& ck : shape_checks_) {
        if (args[std::get<0>(ck)].size(std::get<1>(ck)) !=
            args[std::get<2>(ck)].size(std::get<3>(ck))) {
          // TODO(jansel): make this error message match aten
          throw std::runtime_error(
              "The size of tensor A must match the size of tensor B at non-singleton dimension X");
        }
      }

      // NOLINTNEXTLINE: C-style arrays
      void* call_args
          [NARGS + allocated_outputs_.size() + stride_args_from_.size() +
           shape_from_.size()];
      for (int i = 0; i < NARGS; ++i) {
        call_args[i] = args[i].data_ptr();
      }
      // we might insert the output pointer at call_args[NARGS] below

      int stride_args_offset = NARGS + allocated_outputs_.size();
      for (int i : c10::irange(stride_args_from_.size())) {
        auto& item = stride_args_from_[i];
        call_args[stride_args_offset + i] =
            // NOLINTNEXTLINE: const_cast
            const_cast<int64_t*>(&args[item.first].strides()[item.second]);
      }

      int shape_args_offset = stride_args_offset + stride_args_from_.size();
      size_t numel = 1;
      // NOLINTNEXTLINE: C-style arrays
      int64_t shapes[MAX_DIMS];
      int ndims = shape_from_.size();
      for (int i = 0; i < ndims; ++i) {
        shapes[i] = args[shape_from_[i].first].size(shape_from_[i].second);
        numel *= shapes[i];
        call_args[shape_args_offset + i] = &shapes[i];
      }

      at::Tensor output;
      if (allocated_outputs_.size() > 0) {
        int options_from = allocated_outputs_[0].first;
        auto& output_order = allocated_outputs_[0].second;
        // NOLINTNEXTLINE: C-style arrays
        int64_t strides[MAX_DIMS];
        int64_t next_stride = 1;
        for (int i : output_order) {
          strides[i] = next_stride;
          next_stride *= shapes[i];
        }
        output = at::empty_strided(
            c10::IntArrayRef(shapes, shapes + ndims),
            c10::IntArrayRef(strides, strides + ndims),
            args[options_from].options());
        call_args[NARGS] = output.data_ptr();
      } else {
        output = args[NARGS - 1];
      }

      cg_->call_fast(call_args, numel);

      if (backwards_functions_.size() > 0) {
        std::shared_ptr<CompiledAutoGradNode> node(
            new CompiledAutoGradNode(), torch::autograd::deleteNode);
        node->setup(
            backwards_functions_,
            args,
            NARGS - (allocated_outputs_.size() == 0));
        torch::autograd::create_gradient_edge(output, node);
      }

      return output;
    }

   private:
    CodeGen* cg_ = nullptr;
    std::vector<std::pair<int, int>> shape_from_;
    std::vector<std::pair<int, int>> stride_args_from_;
    std::vector<std::tuple<int, int, int, int>> shape_checks_;
    std::vector<std::pair<int, std::vector<int>>> allocated_outputs_;
    std::vector<CompileCacheBackwards> backwards_functions_;
    std::vector<py::object> objects_; // for ref counting
  };
  typedef std::map<SpecializationKeys, CompileResultImpl*, CmpLess> Cache;

  void check_dispatch_keys(const SpecializationKeys& key) {
    DispatchKeySet ks;
    for (auto& item : key) {
      ks = ks | item.dispatch_key();
    }
    constexpr DispatchKeySet supported = DispatchKeySet({
        DispatchKey::CPU,
        DispatchKey::CUDA,
        DispatchKey::AutogradCPU,
        DispatchKey::AutogradCUDA,
        DispatchKey::BackendSelect,
        DispatchKey::ADInplaceOrView,
    });
    ks = ks - supported;
    if (C10_LIKELY(ks.empty())) {
      return;
    }
    std::stringstream ss;
    ss << "DispatchKeys not yet supported:";
    for (DispatchKey k : ks) {
      ss << " " << k;
    }
    throw std::runtime_error(ss.str());
  }

  CompileResultImpl* compile(const SpecializationKeys& key, at::Tensor* args) {
    // handle a cache miss by creating a new specialized implementation
    check_dispatch_keys(key);
    KernelScope scope(&arena_);
    auto cr = new CompileResultImpl();
    py::gil_scoped_acquire guard;
    std::vector<py::object> spec;
    spec.reserve(key.size());
    for (int i = 0; i < key.size(); ++i) {
      spec.emplace_back(key[i].to_python(args[i]));
    }
    compile_fn_(spec, CompileResultProxy(cr));
    return cr;
  }

  CompileResultImpl* cached_compile(
      const SpecializationKeys& key,
      at::Tensor* args) {
    // TODO(jansel): optimization: make this lock-free in the cache-hit case
    std::lock_guard<std::mutex> guard(mutex_);
    auto item = cache_.find(key);
    if (C10_LIKELY(item != cache_.end())) {
      return item->second;
    } else { // cache miss
      auto cr = compile(key, args);
      cache_.emplace(std::make_pair(key, cr));
      return cr;
    }
  }

  int8_t aliasing_check(const at::Tensor& a, const at::Tensor& b) {
    // 0 means a/b don't alias
    // 1 means a/b alias and are the same
    // -1 means a/b have crazy aliasing overlaps
    if (a.is_alias_of(b)) {
      if (a.is_set_to(b)) {
        return 1;
      } else {
        // TODO(jansel): check for non-overlapping and return 0 in cases where
        // we can prove no aliasing. Possibly could take some logic from
        // tensoriterator.
        return -1;
      }
    } else {
      return 0;
    }
  }

  AliasGroups compute_alias_groups(at::Tensor* args) {
    AliasGroups alias_groups;
    int8_t current_id = 0;
    for (int i = 0; i < NARGS; ++i) {
      alias_groups[i] = 0;
    }
    for (int i = 0; i < NARGS; ++i) {
      if (alias_groups[i] == 0) {
        for (int j = i + 1; j < NARGS; ++j) {
          int8_t alias_type = aliasing_check(args[i], args[j]);
          if (alias_type != 0) {
            if (alias_groups[i] == 0)
              ++current_id;
            alias_groups[i] = current_id;
            alias_groups[j] = current_id * alias_type;
          }
        }
      }
    }
    return alias_groups;
  }

  SpecializationKeys compute_cache_key(at::Tensor* args, bool has_out) {
    LocalState state;
    AliasGroups alias_groups = compute_alias_groups(args);
    SpecializationKeys key;
    int i = 0;
    for (; i < NARGS - 1; ++i) {
      key[i] =
          SpecializationKey<MAX_DIMS>(state, args[i], alias_groups[i], false);
    }
    if (NARGS != 0) {
      key[i] =
          SpecializationKey<MAX_DIMS>(state, args[i], alias_groups[i], has_out);
    }
    return key;
  }

  CompileCache3(py::object compile_fn) : compile_fn_(std::move(compile_fn)) {}

  at::Tensor call(at::Tensor* args, bool has_out) {
    auto key = compute_cache_key(args, has_out);
    return cached_compile(key, args)->call(args);
  }

 public:
  std::mutex mutex_;
  Cache cache_;
  py::object compile_fn_;
  KernelArena arena_;
};

template <int NARGS>
class CompileCache2 {
 public:
  CompileCache2(const py::object& compile_fn)
      : cache2(compile_fn), cache4(compile_fn), cache8(compile_fn) {}

  at::Tensor call(at::Tensor* args, bool has_out) {
    // fan out and and specialize on number of dimension buckets
    int64_t ndims = 0;
    for (int i : c10::irange(NARGS)) {
      ndims = std::max(args[i].dim(), ndims);
    }
    if (ndims <= 2)
      return cache2.call(args, has_out);
    if (ndims <= 4)
      return cache4.call(args, has_out);
    if (ndims <= 8)
      return cache8.call(args, has_out);
    throw std::runtime_error("TODO: handle more dims");
  }

 private:
  CompileCache3<NARGS, 2> cache2;
  CompileCache3<NARGS, 4> cache4;
  CompileCache3<NARGS, 8> cache8;
};

class CompileCache {
 public:
  virtual ~CompileCache() = default;
  virtual PyObject* py_call(PyObject* args, PyObject* kwargs) = 0;
  virtual at::Tensor call(const std::vector<at::Tensor>& args) = 0;
  virtual const std::string& get_name() const = 0;
};

template <int NARGS>
class CompileCacheImpl : public CompileCache {
 public:
  CompileCacheImpl(
      std::string name,
      std::string module_name,
      const std::vector<std::string>& signatures,
      const py::object& compile_fn)
      : cache(compile_fn),
        cache_out(compile_fn),
        parser_(signatures),
        name_(std::move(name)),
        module_name_(std::move(module_name)) {
    if (signatures.size() != 1) {
      throw std::runtime_error("TODO: support overloaded signatures");
    }
  }

  const std::string& get_name() const override {
    return name_;
  }

  PyObject* py_call(PyObject* args, PyObject* kwargs) override {
    ParsedArgs<NARGS + 1> parsed_args;
    PythonArgs r = parser_.parse(args, kwargs, parsed_args);
    bool pre_sampled = false;
    if (C10_UNLIKELY(r.has_torch_function())) {
      py::object op = py::cast(static_cast<CompileCache*>(this));
      return handle_torch_function_no_python_arg_parser(
          r.signature.overloaded_args,
          args,
          kwargs,
          name_.c_str(),
          op.ptr(),
          module_name_.c_str());
    } else if (C10_UNLIKELY(
                   at::hasCallbacks() &&
                   at::shouldRunRecordFunction(&pre_sampled))) {
      throw std::runtime_error("TODO: implement record function");
    } else {
      at::Tensor tensor_args[NARGS + 1]; // NOLINT: c-style arrays
      for (int i = 0; i < NARGS + 1; ++i) {
        tensor_args[i] = r.tensor(i);
      }
      py::gil_scoped_release release;
      if (tensor_args[NARGS].defined()) {
        return THPVariable_Wrap(cache_out.call(tensor_args, true));
      } else {
        return THPVariable_Wrap(cache.call(tensor_args, false));
      }
    }
  }

  at::Tensor call(const std::vector<at::Tensor>& args) override {
    if (C10_UNLIKELY(args.size() != NARGS)) {
      throw std::runtime_error("wrong number of args");
    }
    return cache.call(const_cast<at::Tensor*>(args.data()), false); // NOLINT
  }

 private:
  CompileCache2<NARGS> cache;
  CompileCache2<NARGS + 1> cache_out; // out=... variant
  PythonArgParser parser_;
  std::string name_;
  std::string module_name_;
};

CompileCache* create_compile_cache(
    const std::string& name,
    const std::string& module_name,
    const std::vector<std::string>& sig,
    const py::object& compile_fn,
    int num_args) {
  switch (num_args) {
    case 1:
      return new CompileCacheImpl<1>(name, module_name, sig, compile_fn);
    case 2:
      return new CompileCacheImpl<2>(name, module_name, sig, compile_fn);
    case 3:
      return new CompileCacheImpl<3>(name, module_name, sig, compile_fn);
    case 4:
      return new CompileCacheImpl<4>(name, module_name, sig, compile_fn);
    case 5:
      return new CompileCacheImpl<5>(name, module_name, sig, compile_fn);
    case 6:
      return new CompileCacheImpl<6>(name, module_name, sig, compile_fn);
    case 7:
      return new CompileCacheImpl<7>(name, module_name, sig, compile_fn);
    case 8:
      return new CompileCacheImpl<8>(name, module_name, sig, compile_fn);
    default:
      throw std::runtime_error("TODO: support other arg counts");
  }
}

variable_list CompiledAutoGradNode::apply(variable_list&& new_inputs) {
  // TODO(jansel): we likely need to copy some error checking from eager to
  // here
  // TODO(jansel): possible optimization: horizontal fusion of each backwards
  // fn
  // TODO(jansel): possible optimization: reuse the forwards SpecializationKey
  // TODO(jansel): possible optimization: dont save all the inputs_
  // TODO(jansel): possible optimization: precompute in forwards
  variable_list args;
  args.reserve(inputs_.size() + 1);
  std::copy(inputs_.begin(), inputs_.end(), std::back_inserter(args));
  args.emplace_back(new_inputs[0]);

  variable_list result;
  result.reserve(backwards_functions_.size());
  for (auto& bk : backwards_functions_) {
    result.emplace_back(std::get<1>(bk)->call(args));
  }
  return result;
}

} // namespace

void initTensorExprCompileCacheBindings(PyObject* te_obj) {
  py::handle te(te_obj);

  py::class_<CompileCache>(te, "CompileCache")
      .def(py::init(&create_compile_cache))
      .def(
          "__call__", [](CompileCache& self, py::args args, py::kwargs kwargs) {
            return py::reinterpret_steal<py::object>(
                self.py_call(args.ptr(), kwargs.ptr()));
          });

  py::class_<CompileResultProxy>(te, "CompileResult")
      .def(
          "set_code",
          [](CompileResultProxy& self, const py::object& cg) {
            self.res->set_code(cg);
          })
      .def(
          "add_shape_check",
          [](CompileResultProxy& self,
             const std::tuple<int, int, int, int>& indices) {
            self.res->add_shape_check(indices);
          })
      .def(
          "set_shape_from",
          [](CompileResultProxy& self,
             const std::vector<std::pair<int, int>>& indices) {
            self.res->set_shape_from(indices);
          })
      .def(
          "set_stride_args_from",
          [](CompileResultProxy& self,
             const std::vector<std::pair<int, int>>& indices) {
            self.res->set_stride_args_from(indices);
          })
      .def(
          "add_allocated_output",
          [](CompileResultProxy& self,
             int options_from,
             const std::vector<int>& storage_order) {
            self.res->add_allocated_output(options_from, storage_order);
          })
      .def(
          "set_backwards",
          [](CompileResultProxy& self,
             int index,
             py::object backward_compiler) {
            self.res->set_backwards(index, backward_compiler);
          });
}
} // namespace jit
} // namespace torch
