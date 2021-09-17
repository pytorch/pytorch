///
/// This file implements a pretty complicated cache hierarchy to
/// support fast lookup of cached compiled kernels based on input
/// tensor properties.
///
/// To compile correct and efficient kernels, we need to know several
/// details about the tensors involved (see SpecializationKey): the
/// dtype, number of dimensions, contiguity and broadcasting in each
/// dimension, etc.  We use these keys to look up the proper kernel,
/// and compile a new version if we see a novel key.
///
/// To make the cache hit path as fast as possible, we create
/// specialized caches based on the number of arguments (input and
/// output) to the kernel, as well as the max number of tensor
/// dimensions supported by the kernel.  All of this specialization is
/// done with a bunch of template metaprogramming.
///
/// CompileCache
///   --is a---> InOutSpecializedCache<NUM_IN, NUM_OUT>
///   --has a--> ArgCountSpecializedCache<ArgCounts>
///   --has a--> ArgAndDimSpecializedCache<ArgCounts, MAX_DIMS>
///   --has a--> map<SpecializationKey<MAX_DIMS>,
///                  CompileResult<ArgCounts, MAX_DIMS>>
///
/// With this structure, a SpecializationKey and CompileResult know
/// exactly how many arguments and dimensions they have to deal with,
/// so a bunch of checks can be done statically against fixed-size
/// arrays rather than traversing dynamically allocated structures.
/// This saves precious cycles in figuring out which kernel to
/// launch!.
///
#include <torch/csrc/autograd/custom_function.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/jit/tensorexpr/codegen.h>

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
static uint8_t packFlags(const LocalState& state, const at::Tensor& v) {
  static_assert(
      static_cast<int>(at::ScalarType::NumOptions) < 128, "overflow possible");
  at::ScalarType dtype = v.dtype().toScalarType();
  bool requires_grad = state.gradModeEnabled && v.requires_grad();
  return static_cast<uint8_t>(requires_grad) |
      (static_cast<uint8_t>(dtype) << 1);
}

/// Per-tensor cache specialization key, templated on the number of
/// tensor dims.  Records dtype, dispatch options, aliasing, and
/// per-dim contiguity/broadcasting information.
#pragma pack(push, 1)
template <int MAX_DIMS>
struct SpecializationKey {
  /// Default constructor; does no initialization, use only for
  /// declarations, e.g., std::array.
  SpecializationKey() {} // NOLINT: intentionally not initialized

  /// Construct a specialization key from a given TLS state and
  /// Tensor.
  // NOLINTNEXTLINE: intentionally not initializing dimflags_
  SpecializationKey(
      const LocalState& state,
      const at::Tensor& v,
      int8_t aliasGroup)
      : flags_(packFlags(state, v)),
        aliasGroup_(aliasGroup),
        dispatchKey_(state.apply(v.key_set()).raw_repr()) {
    initDimflags(v.sizes(), v.strides(), v.ndimension());
  }

  /// Compare key to other and return 0 if equal, <0 if key is less
  /// than other, or >0 if key is greater than other.
  bool operator<(const SpecializationKey<MAX_DIMS>& other) const {
    static_assert(
        sizeof(SpecializationKey<MAX_DIMS>) == 10 + MAX_DIMS,
        "struct is not packed, memcmp requires no padding");
    return memcmp(this, &other, sizeof(SpecializationKey<MAX_DIMS>)) < 0;
  }

  /// Get the dispatch key for this specialization.
  at::DispatchKeySet dispatchKey() const {
    return at::DispatchKeySet(at::DispatchKeySet::RAW, dispatchKey_);
  }

  /// Return vector of strings describing key sizes.
  std::vector<std::string> shape() const {
    std::vector<std::string> result;
    for (int i = 0; i < MAX_DIMS; ++i) {
      if ((dimflags_[i] & SIZE_MISSING) > 0) {
        break;
      }

      if ((dimflags_[i] & SIZE_ONE) > 0) {
        result.emplace_back("one");
      } else {
        result.emplace_back("other");
      }
    }
    return result;
  }

  /// Return vector of strings describing key strides.
  std::vector<std::string> stride() const {
    std::vector<std::string> result;
    for (int i = 0; i < MAX_DIMS; ++i) {
      if ((dimflags_[i] & SIZE_MISSING) > 0) {
        break;
      }

      if ((dimflags_[i] & STRIDE_ZERO) > 0) {
        result.emplace_back("zero");
      } else if ((dimflags_[i] & STRIDE_ONE) > 0) {
        result.emplace_back("one");
      } else if ((dimflags_[i] & STRIDE_CONTIGUOUS) > 0) {
        result.emplace_back("contiguous");
      } else if ((dimflags_[i] & STRIDE_TRANSPOSED_CONTIGUOUS) > 0) {
        result.emplace_back("transposed_contiguous");
      } else if ((dimflags_[i] & STRIDE_AS_ARG) > 0) {
        result.emplace_back("as_arg");
      } else {
        TORCH_INTERNAL_ASSERT(false, "unknown stride properties");
      }
    }
    return result;
  }

  /// Convert this specialization key to a python namedtuple.
  py::object toPython(const at::Tensor& example, bool is_out) const {
    // Create the python specialization key type (a namedtuple) lazily.
    static py::object keyType = [] {
      // create it lazily
      py::object namedtuple =
          py::module_::import("collections").attr("namedtuple");
      return namedtuple(
          "SpecializationKey",
          "alias_group,ndim,dtype,device,layout,requires_grad,out,shape,stride");
    }();
    py::object ex = py::cast(example);
    return keyType(
        static_cast<int>(aliasGroup_),
        ex.attr("ndim"),
        ex.attr("dtype"),
        ex.attr("device"),
        ex.attr("layout"),
        ex.attr("requires_grad"),
        py::bool_(is_out),
        shape(),
        stride());
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
  void initDimflags(
      c10::IntArrayRef sizes,
      c10::IntArrayRef strides,
      int64_t ndims) {
    // Pack all the properties for each dimension into a uint8.
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

  /// Per-dimension shape flags.
  // NOLINTNEXTLINE: C-style arrays
  uint8_t dimflags_[MAX_DIMS];
};
#pragma pack(pop)

} // namespace
