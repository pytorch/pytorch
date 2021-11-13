// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

///
/// This design stemmed of from the PointwiseOperatorCompileCache with the
/// purpose of making it more generic for AOTAutograd. This is Compile Cache
/// allowing different types of hashing functions, and is agnostic of the
/// compiler.
///
#include <functorch/csrc/CompileCache.h>
#include <torch/csrc/autograd/custom_function.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/jit/tensorexpr/codegen.h>
#include <torch/csrc/utils/pybind.h>

using namespace torch::jit::tensorexpr;

namespace {

/// Record of thread-local state that changes operator behavior.
struct LocalState {
  c10::impl::LocalDispatchKeySet dispatchModifier;
  bool gradModeEnabled;

  at::DispatchKeySet apply(at::DispatchKeySet ks) const {
    return (ks | dispatchModifier.included_) - dispatchModifier.excluded_;
  }

  LocalState()
      : dispatchModifier(c10::impl::tls_local_dispatch_key_set()),
        gradModeEnabled(at::GradMode::is_enabled()) {}
};

/// Helper to pack tensor (dtype, requires grad) into an 8-bit key.
static uint8_t packFlags(const LocalState &state, const at::Tensor &v) {
  static_assert(static_cast<int>(at::ScalarType::NumOptions) < 128,
                "overflow possible");
  at::ScalarType dtype = v.dtype().toScalarType();
  bool requires_grad = state.gradModeEnabled && v.requires_grad();
  return static_cast<uint8_t>(requires_grad) |
         (static_cast<uint8_t>(dtype) << 1);
}

/// Per-tensor cache specialization key targetting dynamic shapes. Records
/// dtype, dispatch options, aliasing, and per-dim contiguity/broadcasting
/// information.
#pragma pack(push, 1)
struct DynamicArgSpecializationKey {
  /// Default constructor; does no initialization, use only for
  /// declarations, e.g., std::array.
  DynamicArgSpecializationKey() {} // NOLINT: intentionally not initialized

  /// Construct a specialization key from a given TLS state and
  /// Tensor.
  // NOLINTNEXTLINE: intentionally not initializing dimflags_
  DynamicArgSpecializationKey(const LocalState &state, const at::Tensor &v,
                              int8_t aliasGroup)
      : flags_(packFlags(state, v)), aliasGroup_(aliasGroup),
        dispatchKey_(state.apply(v.key_set()).raw_repr()),
        nDims_(v.ndimension()) {
    initDimflags(v.sizes(), v.strides());
  }

  // TODO (anijain) - Code seems expensive for each comparison. Revisit if cache
  // latency is bad.
  bool operator<(const DynamicArgSpecializationKey &other) const {
    auto this_tie = std::tie(flags_, aliasGroup_, dispatchKey_, nDims_);
    auto other_tie = std::tie(other.flags_, other.aliasGroup_,
                              other.dispatchKey_, other.nDims_);
    if (this_tie != other_tie) {
      return this_tie < other_tie;
    }

    for (int dim = 0; dim < nDims_; dim++) {
      if (dimflags_[dim] != other.dimflags_[dim]) {
        return dimflags_[dim] < other.dimflags_[dim];
      }
    }
    return false;
  }

  /// Get the dispatch key for this specialization.
  at::DispatchKeySet dispatchKey() const {
    return at::DispatchKeySet(at::DispatchKeySet::RAW, dispatchKey_);
  }

private:
  /// Flag bits indicating tensor shape properties like contiguity and
  /// broadcasting that are relevant for codegen.
  enum DimFlags {
    /// A leading dimension implicitly added by broadcasting.
    SIZE_MISSING = 1 << 0,

    /// Size == 1.
    SIZE_ONE = 1 << 1,

    /// Size > 1.
    SIZE_OTHER = 1 << 2,

    /// Stride == 0; broadcasting.
    STRIDE_ZERO = 1 << 3,

    /// Stride == 1; packed contiguously in memory.
    STRIDE_ONE = 1 << 4,

    /// Stride = Stride[i + 1] * Size[i + 1].
    /// Used to collapse dimensions.
    STRIDE_CONTIGUOUS = 1 << 5,

    /// Stride = Stride[i - 1] * Size[i - 1].
    /// Used to collapse dimensions in the other direction.
    STRIDE_TRANSPOSED_CONTIGUOUS = 1 << 6, // stride[i-1] * sizes[i-1]

    /// Stride must be provided as an argument.
    STRIDE_AS_ARG = 1 << 7,
  };

  /// Initialize the shape flags for each dimension.
  void initDimflags(c10::IntArrayRef sizes, c10::IntArrayRef strides) {
    // Pack all the properties for each dimension into a uint8.
    dimflags_.reserve(nDims_);
    for (int64_t dim = 0; dim < nDims_; ++dim) {
      uint8_t flag =
          (sizes[dim] == 0 ? SIZE_MISSING
                           : (sizes[dim] == 1 ? SIZE_ONE : SIZE_OTHER));
      if (strides[dim] == 0) {
        flag |= STRIDE_ZERO;
      } else if (strides[dim] == 1) {
        flag |= STRIDE_ONE;
      } else if (dim + 1 < (int64_t)sizes.size() &&
                 strides[dim] == strides[dim + 1] * sizes[dim + 1]) {
        flag |= STRIDE_CONTIGUOUS;
      } else if (dim > 0 && strides[dim] == strides[dim - 1] * sizes[dim - 1] &&
                 (dimflags_[dim - 1] & STRIDE_CONTIGUOUS) == 0) {
        flag |= STRIDE_TRANSPOSED_CONTIGUOUS;
      } else {
        flag |= STRIDE_AS_ARG;
      }
      dimflags_.push_back(flag);
    }
  }

private:
  /// Packed field with requires_grad and dtype.
  uint8_t flags_;

  /// Bits indicating whether there is aliasing in this group.
  /// 0 = no aliasing
  /// >0 = same data, strides, and shapes within group
  /// <0 = overlapping storage madness
  int8_t aliasGroup_;

  /// DispatchKeySet includes device/layout.
  uint64_t dispatchKey_;

  /// Saving the number of dimensions
  int nDims_;

  /// Per-dimension shape flags.
  // NOLINTNEXTLINE: C-style arrays
  std::vector<uint8_t> dimflags_;
};
#pragma pack(pop)

#pragma pack(push, 1)
/// Per-tensor cache specialization key targetting static shapes. Recordsdtype,
/// dispatch options, aliasing, and full shapes and strides.
struct StaticArgSpecializationKey {
  /// Default constructor; does no initialization, use only for
  /// declarations, e.g., std::array.
  StaticArgSpecializationKey() {} // NOLINT: intentionally not initialized

  StaticArgSpecializationKey(const LocalState &state, const at::Tensor &v,
                             int8_t aliasGroup)
      : flags_(packFlags(state, v)), aliasGroup_(aliasGroup),
        dispatchKey_(state.apply(v.key_set()).raw_repr()),
        nDims_(v.ndimension()) {
    for (int dim = 0; dim < nDims_; dim++) {
      shapes_.push_back(v.sizes()[dim]);
      strides_.push_back(v.strides()[dim]);
    }
  }

  // TODO (anijain) - Code seems expensive for each comparison. Revisit if cache
  // latency is bad.
  bool operator<(const StaticArgSpecializationKey &other) const {
    auto this_tie = std::tie(flags_, aliasGroup_, dispatchKey_, nDims_);
    auto other_tie = std::tie(other.flags_, other.aliasGroup_,
                              other.dispatchKey_, other.nDims_);
    if (this_tie != other_tie) {
      return this_tie < other_tie;
    }

    for (int dim = 0; dim < nDims_; dim++) {
      auto this_tie = std::tie(shapes_[dim], strides_[dim]);
      auto other_tie = std::tie(other.shapes_[dim], other.strides_[dim]);
      if (this_tie != other_tie) {
        return this_tie < other_tie;
      }
    }
    return false;
  }

private:
  /// Packed field with requires_grad and dtype.
  uint8_t flags_;

  /// Bits indicating whether there is aliasing in this group.
  /// 0 = no aliasing
  /// >0 = same data, strides, and shapes within group
  /// <0 = overlapping storage madness
  int8_t aliasGroup_;

  /// DispatchKeySet includes device/layout.
  uint64_t dispatchKey_;

  /// Saving the number of dimensions
  int nDims_;

  /// Record all tensor shapes.
  std::vector<uint64_t> shapes_;

  /// Record all tensor strides.
  std::vector<uint64_t> strides_;
};
#pragma pack(pop)

/// This is the base class for recording Arg or Tensor propoerties. To create a
/// new Compile cache, we can inherit from this base class and record the
/// properties we are interested in.
struct ArgCompileCacheBase {
  /// Destructor.
  virtual ~ArgCompileCacheBase() = default;

  /// Check if a key (computed from args and kwargs) is present in the cache.
  virtual py::object at(PyObject *args, PyObject *kwargs) = 0;

  /// Inserts a new compiled_function for given args.
  virtual void insert(const py::object &compileFn, PyObject *args,
                      PyObject *kwargs) = 0;

  /// Get name of kernel.
  virtual const std::string &getName() const = 0;

  /// Get the size of the cache. Helps in counting the number of recompilations.
  virtual const int64_t size() const = 0;

  /// Clear the cache.
  virtual void clear() = 0;
};

/// ArgCompileCache is a templated class allowing plugging of different types of
/// Hasher/Specialization Keys.
template <int NUM_IN, class SpeciaizationKey>
struct ArgsCompileCache : public ArgCompileCacheBase {
public:
  constexpr static int NUM_ARGS = NUM_IN;

  /// Array of keys used for specializing kernels in this cache.
  using SpecializationKeys = std::array<SpeciaizationKey, NUM_ARGS>;

  /// Array defining groups of aliased tensors.
  using AliasGroups = std::array<int8_t, NUM_ARGS>;

  /// Cache type mapping specialization keys to compiled kernels.
  using Cache = std::map<SpecializationKeys, py::object>;

  /// Construct a kernel cache for a kernel with given name,
  /// module_name, and signatures, using a given compilation function.
  ArgsCompileCache(std::string name, std::string moduleName,
                   const std::vector<std::string> &signatures)
      : parser_(signatures), name_(std::move(name)),
        moduleName_(std::move(moduleName_)) {
    if (signatures.size() != 1) {
      throw std::runtime_error("TODO: support overloaded signatures");
    }
  }

  /// Returns name of kernel.
  const std::string &getName() const { return name_; }

  /// Compute aliasing relationships between tensors a and b.
  /// 0 means a/b don't alias.
  /// 1 means a/b alias and are the same.
  /// -1 means a/b have crazy aliasing overlaps.
  int8_t computeAliasing(const at::Tensor &a, const at::Tensor &b) {
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

  /// Compute aliasing groups: group of tensors that alias each other.
  AliasGroups computeAliasGroups(at::Tensor *args) {
    AliasGroups aliasGroups;
    int8_t currentId = 0;
    for (int i = 0; i < NUM_ARGS; ++i) {
      aliasGroups[i] = 0;
    }
    for (int i = 0; i < NUM_ARGS; ++i) {
      if (aliasGroups[i] == 0) {
        for (int j = i + 1; j < NUM_ARGS; ++j) {
          int8_t alias_type = computeAliasing(args[i], args[j]);
          if (alias_type != 0) {
            if (aliasGroups[i] == 0)
              ++currentId;
            aliasGroups[i] = currentId;
            aliasGroups[j] = currentId * alias_type;
          }
        }
      }
    }
    return aliasGroups;
  }

  /// Compute the set of specialization keys based on the inputs to
  /// the kernel.
  SpecializationKeys computeCacheKey(at::Tensor *args) {
    LocalState state;
    AliasGroups aliasGroups = computeAliasGroups(args);
    SpecializationKeys key;
    for (int i = 0; i < NUM_ARGS; ++i) {
      key[i] = SpeciaizationKey(state, args[i], aliasGroups[i]);
    }
    return key;
  }

  /// Check if the function has already been compiled.
  py::object at(PyObject *args, PyObject *kwargs) {
    torch::ParsedArgs<NUM_ARGS> parsed_args;
    torch::PythonArgs r = parser_.parse(args, kwargs, parsed_args);
    at::Tensor tensorArgs[NUM_ARGS]; // NOLINT: c-style arrays
    for (int i = 0; i < NUM_ARGS; ++i) {
      tensorArgs[i] = r.tensor(i);
    }

    LocalState state;
    SpecializationKeys key = computeCacheKey(tensorArgs);
    auto item = cache_.find(key); // protected by GIL

    if (C10_LIKELY(item != cache_.end())) {
      return cache_.at(key);
    }
    return py::none();
  }

  /// Insert a new compiled functions for new tensor properties.
  void insert(const py::object &compileFn, PyObject *args, PyObject *kwargs) {
    torch::ParsedArgs<NUM_ARGS> parsed_args;
    torch::PythonArgs r = parser_.parse(args, kwargs, parsed_args);
    at::Tensor tensorArgs[NUM_ARGS]; // NOLINT: c-style arrays
    for (int i = 0; i < NUM_ARGS; ++i) {
      tensorArgs[i] = r.tensor(i);
    }

    LocalState state;
    SpecializationKeys key = computeCacheKey(tensorArgs);
    cache_.emplace(key, compileFn);
  }

  const int64_t size() const { return cache_.size(); }

  /// Clear the cache.
  void clear() {
    cache_.clear();
  }

private:
  /// Parser for kernel args.
  torch::PythonArgParser parser_;

  /// Name of kernel.
  std::string name_;

  /// Module name of kernel.
  std::string moduleName_;

  /// Compilation cache holding key and the compiled function.
  Cache cache_;
};

/// This is the top CompileCache wrapper. In addition to Tensor/Arg properties,
/// we can add function id etc in this cache. Essentially, this act as a 2
/// levels of cachce. First one caches on function id and function signature and
/// the second one caches on arg/tensor properties.
struct CompileCache {
  CompileCache() = default;
  ~CompileCache() = default;

  struct FunctionKey {
    int64_t id_;
    int numArgs_;

    FunctionKey(const int64_t id, const int numArgs) {
      id_ = id;
      numArgs_ = numArgs;
    }

    bool operator==(const FunctionKey &other) const {
      return id_ == other.id_ && numArgs_ == other.numArgs_;
    }
  };

  struct FunctionKeyHash {
    int64_t operator()(const FunctionKey &node) const {
      return std::hash<int64_t>()(node.id_) ^
             std::hash<int64_t>()(node.numArgs_);
    }
  };

  /// Check if a key is present at two levels of caches.
  py::object at(int64_t id, int numArgs, PyObject *args, PyObject *kwargs) {
    auto key = FunctionKey(id, numArgs);
    if (functions_.find(key) != functions_.end()) {
      return functions_[key]->at(args, kwargs);
    }
    return py::none();
  }

  template <class HasherType>
  ArgCompileCacheBase *getNewCache(const int64_t id,
                                   const std::vector<std::string> &sig,
                                   int numArgs) {
    const std::string name = "fn_" + std::to_string(id);
    const std::string moduleName = "module_" + std::to_string(id);
    switch (numArgs) {
    case 1:
      return new ArgsCompileCache<1, HasherType>(name, moduleName, sig);
    case 2:
      return new ArgsCompileCache<2, HasherType>(name, moduleName, sig);
    case 3:
      return new ArgsCompileCache<3, HasherType>(name, moduleName, sig);
    case 4:
      return new ArgsCompileCache<4, HasherType>(name, moduleName, sig);
    case 5:
      return new ArgsCompileCache<5, HasherType>(name, moduleName, sig);
    case 6:
      return new ArgsCompileCache<6, HasherType>(name, moduleName, sig);
    case 7:
      return new ArgsCompileCache<7, HasherType>(name, moduleName, sig);
    case 8:
      return new ArgsCompileCache<8, HasherType>(name, moduleName, sig);
    default:
      throw std::runtime_error("TODO: support other arg counts");
    }
  }

  /// Insert a compiled function. First, we check if this function has been seen
  /// before. If not, we instantiate a new arg/tensor level cache and keep track
  /// of it.
  void insert(int64_t id, const std::vector<std::string> &signatures,
              int numArgs, const std::string hasherType,
              const py::object &compileFn, PyObject *args, PyObject *kwargs) {
    auto key = FunctionKey(id, numArgs);
    if (functions_.find(key) == functions_.end()) {
      if (hasherType == "StaticShapeHasher") {
        functions_[key] =
            getNewCache<StaticArgSpecializationKey>(id, signatures, numArgs);
      } else if (hasherType == "DynamicShapeHasher") {
        functions_[key] =
            getNewCache<DynamicArgSpecializationKey>(id, signatures, numArgs);
      } else {
        throw std::runtime_error("Unsupported Hasher key");
      }
    }
    functions_[key]->insert(compileFn, args, kwargs);
  }

  /// Sum up all the size of all compilations.
  int size() {
    int num_of_recompilations = 0;
    for (auto c : functions_) {
      num_of_recompilations += functions_[c.first]->size();
    }
    return num_of_recompilations;
  }

  void clear() {
    for (auto c : functions_) {
      functions_[c.first]->clear();
    }
    functions_.clear();
  }

private:
  /// Compilation cache based on function properties.
  std::unordered_map<FunctionKey, ArgCompileCacheBase *, FunctionKeyHash>
      functions_;
};

static CompileCache *createCompileCache() { return new CompileCache(); }

} // namespace

namespace at {
namespace functorch {

// TODO(anijain) - Add static compilation cache
void initCompileCacheBindings(PyObject *module) {
  py::handle te(module);
  py::class_<CompileCache>(te, "CompileCache")
      .def(py::init(&createCompileCache))
      .def("at",
           [](CompileCache &self, int64_t id, int numArgs, py::args args,
              py::kwargs kwargs) {
             return self.at(id, numArgs, args.ptr(), kwargs.ptr());
           })
      .def("insert",
           [](CompileCache &self, int64_t id,
              const std::vector<std::string> &signatures, int numArgs,
              const std::string hasherType, const py::object &compileFn,
              py::args args, py::kwargs kwargs) {
             self.insert(id, signatures, numArgs, hasherType, compileFn,
                         args.ptr(), kwargs.ptr());
           })
      .def("clear", [](CompileCache &self) { self.clear(); })
      .def("size", [](CompileCache &self) { return self.size(); });
}

} // namespace functorch
} // namespace at
