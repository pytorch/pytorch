#pragma once

#include <c10/core/DispatchKey.h>
#include <c10/util/Exception.h>
#include <c10/util/Metaprogramming.h>
#include <c10/util/llvmMathExtras.h>
#include <ostream>

namespace c10 {

// A representation of a set of DispatchKeys.  A tensor may have multiple
// tensor type ids, e.g., a Variable tensor can also be a CPU tensor; the
// DispatchKeySet specifies what type ids apply.  The internal representation is
// as a 64-bit bit set (this means only 64 tensor type ids are supported).
//
// Note that DispatchKeys are ordered; thus, we can ask questions like "what is
// the highest priority DispatchKey in the set"?  (The set itself is not
// ordered; two sets with the same ids will always have the ids ordered in the
// same way.)
//
// At the moment, there are no nontrivial uses of this set; tensors are always
// singletons.  In the near future, this set will represent variable? + tensor
// type id.  In the far future, it will be requires grad? + profiling? +
// tracing? + lazy? + tensor type id.
//
// (The difference between variable and requires grad, is that
// there are currently three states a tensor can be:
//  1. Not a variable
//  2. Variable with requires_grad=False
//  3. Variable with requires_grad=True
// Eventually, we want to kill state (1), and only dispatch to autograd
// handling code if one of the inputs requires grad.)
//
// An undefined tensor is one with an empty tensor type set.
class DispatchKeySet final {
 public:
  enum Full { FULL };
  enum FullAfter { FULL_AFTER };
  enum Raw { RAW };

  // NB: default constructor representation as zero is MANDATORY as
  // use of DispatchKeySet in TLS requires this.
  constexpr DispatchKeySet() : repr_(0) {}
  constexpr DispatchKeySet(Full)
      : repr_(std::numeric_limits<decltype(repr_)>::max()) {}
  constexpr DispatchKeySet(FullAfter, DispatchKey t)
      // LSB after t are OK, but not t itself.
      : repr_((1ULL << (static_cast<uint8_t>(t) - 1)) - 1) {}
  // Public version of DispatchKeySet(uint64_t) API; external users
  // must be explicit when they do this!
  constexpr DispatchKeySet(Raw, uint64_t x) : repr_(x) {}
  explicit constexpr DispatchKeySet(DispatchKey t)
      : repr_(
            t == DispatchKey::Undefined
                ? 0
                : 1ULL << (static_cast<uint8_t>(t) - 1)) {}
  explicit constexpr DispatchKeySet(std::initializer_list<DispatchKey> ks)
      : repr_(0) {
    for (auto k : ks) {
      repr_ |= DispatchKeySet(k).repr_;
    }
  }
  // Test if a DispatchKey is in the set
  bool inline has(DispatchKey t) const {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(t != DispatchKey::Undefined);
    return static_cast<bool>(repr_ & DispatchKeySet(t).repr_);
  }
  // Test if DispatchKeySet is a superset of ks.
  bool isSupersetOf(DispatchKeySet ks) const {
    return (repr_ & ks.repr_) == ks.repr_;
  }
  // Perform set union
  constexpr DispatchKeySet operator|(DispatchKeySet other) const {
    return DispatchKeySet(repr_ | other.repr_);
  }
  // Perform set intersection
  DispatchKeySet operator&(DispatchKeySet other) const {
    return DispatchKeySet(repr_ & other.repr_);
  }
  // Compute the set difference self - other
  DispatchKeySet operator-(DispatchKeySet other) const {
    return DispatchKeySet(repr_ & ~other.repr_);
  }
  // Compute self ^ other
  DispatchKeySet operator^(DispatchKeySet other) const {
    return DispatchKeySet(repr_ ^ other.repr_);
  }
  // Perform set equality
  bool operator==(DispatchKeySet other) const {
    return repr_ == other.repr_;
  }
  // Add a DispatchKey to the DispatchKey set.  Does NOT mutate,
  // returns the extended DispatchKeySet!
  C10_NODISCARD DispatchKeySet add(DispatchKey t) const {
    return *this | DispatchKeySet(t);
  }
  // Remove a DispatchKey from the DispatchKey set.  This is
  // generally not an operation you should be doing (it's
  // used to implement operator<<)
  C10_NODISCARD DispatchKeySet remove(DispatchKey t) const {
    return DispatchKeySet(repr_ & ~DispatchKeySet(t).repr_);
  }
  // Is the set empty?  (AKA undefined tensor)
  bool empty() const {
    return repr_ == 0;
  }
  uint64_t raw_repr() {
    return repr_;
  }
  // Return the type id in this set with the highest priority (i.e.,
  // is the largest in the DispatchKey enum).  Intuitively, this
  // type id is the one that should handle dispatch (assuming there
  // aren't any further exclusions or inclusions).
  DispatchKey highestPriorityTypeId() const {
    // TODO: If I put Undefined as entry 64 and then adjust the
    // singleton constructor to shift from the right, we can get rid of the
    // subtraction here.  It's modestly more complicated to get right so I
    // didn't do it for now.
    return static_cast<DispatchKey>(64 - llvm::countLeadingZeros(repr_));
  }

  DispatchKey highestPriorityBackendTypeId() const {
    return (*this &
            ((1ULL << static_cast<uint8_t>(DispatchKey::EndOfBackendKeys)) - 1))
        .highestPriorityTypeId();
  }

 private:
  constexpr DispatchKeySet(uint64_t repr) : repr_(repr) {}
  uint64_t repr_ = 0;

 public:
  // STL iterator for DispatchKeySet. Iterates through all DispatchKeys in the
  // set. The iterator is only invalidated by the destruction of the underlying
  // DispatchKeySet as the iterator stores a pointer to the raw representation
  // of the DispatchKeySet.
  class iterator {
   public:
    using self_type = iterator;
    using iterator_category = std::input_iterator_tag;
    using value_type = DispatchKey;
    using difference_type = ptrdiff_t;

    explicit iterator(const uint64_t* data_ptr, uint8_t i = 0)
        : data_ptr_(data_ptr), i_(i) {
      // Go to the first key in the set
      ++(*this);
    }

    self_type& operator++() {
      TORCH_INTERNAL_ASSERT(
          i_ <= static_cast<uint8_t>(DispatchKey::NumDispatchKeys));

      // Create a masked version of the set representation to ignore previous
      // keys that we've iterated through.
      uint64_t masked_data = llvm::maskTrailingZeros<uint64_t>(i_) & *data_ptr_;
      uint64_t firstKeyIndex = llvm::findFirstSet(masked_data);

      // If there are no keys, set to end iterator value
      if (firstKeyIndex == std::numeric_limits<uint64_t>::max() ||
          i_ == static_cast<uint8_t>(DispatchKey::NumDispatchKeys)) {
        i_ = static_cast<uint8_t>(DispatchKey::NumDispatchKeys);
        return *this;
      }

      i_ = static_cast<uint8_t>(firstKeyIndex) + 1;
      return *this;
    }

    self_type operator++(int) {
      self_type previous_iterator = *this;
      ++(*this);
      return previous_iterator;
    }

    bool operator==(const self_type& rhs) const {
      return i_ == rhs.i_;
    }
    bool operator!=(const self_type& rhs) const {
      return i_ != rhs.i_;
    }
    DispatchKey operator*() const {
      return static_cast<DispatchKey>(i_);
    }

   private:
    const uint64_t* data_ptr_;
    uint8_t i_;
  };

 public:
  // Returns iterator to the first key in the set. If no keys are in the
  // set, then will return the end iterator.
  iterator begin() const {
    return iterator(&repr_);
  }

  // We do not need to iterate beyond NumDispatchKeys so we will treat this as
  // the end iterator. NumDispatchKeys will always be strictly less than 64.
  iterator end() const {
    return iterator(&repr_, static_cast<uint8_t>(DispatchKey::NumDispatchKeys));
  }
};

C10_API std::string toString(DispatchKeySet);
C10_API std::ostream& operator<<(std::ostream&, DispatchKeySet);

// autograd_dispatch_keyset should include all runtime autograd keys.
// Alias key DispatchKey::Autograd maps to autograd_dispatch_keyset.
// NB: keys in this set also get associated with CompositeImplicitAutograd
constexpr DispatchKeySet autograd_dispatch_keyset = DispatchKeySet({
    DispatchKey::AutogradCPU,
    DispatchKey::AutogradCUDA,
    DispatchKey::AutogradXLA,
    DispatchKey::AutogradLazy,
    DispatchKey::AutogradNestedTensor,
    DispatchKey::AutogradMLC,
    DispatchKey::AutogradHPU,
    DispatchKey::AutogradXPU,
    DispatchKey::AutogradPrivateUse1,
    DispatchKey::AutogradPrivateUse2,
    DispatchKey::AutogradPrivateUse3,
    DispatchKey::AutogradOther,
});

constexpr DispatchKeySet autocast_dispatch_keyset = DispatchKeySet({
    DispatchKey::AutocastCPU,
    DispatchKey::AutocastCUDA,
});

// See Note [TLS Initialization]
constexpr DispatchKeySet default_included_set = DispatchKeySet({
    DispatchKey::BackendSelect,
    DispatchKey::ADInplaceOrView,
});

constexpr DispatchKeySet default_excluded_set = DispatchKeySet({
    DispatchKey::AutocastCPU,
    DispatchKey::AutocastCUDA,
});

constexpr DispatchKeySet autograd_dispatch_keyset_with_ADInplaceOrView =
    autograd_dispatch_keyset | DispatchKeySet(DispatchKey::ADInplaceOrView);

// backend dispatch keys that map to DispatchKey::AutogradOther
// NB: keys in this set also get associated with CompositeImplicitAutograd
constexpr DispatchKeySet autogradother_backends = DispatchKeySet(
    {DispatchKey::HIP,
     DispatchKey::VE,
     DispatchKey::FPGA,
     DispatchKey::ORT,
     DispatchKey::Vulkan,
     DispatchKey::Metal,
     DispatchKey::QuantizedCPU,
     DispatchKey::QuantizedCUDA,
     DispatchKey::CustomRNGKeyId,
     DispatchKey::MkldnnCPU,
     DispatchKey::SparseCPU,
     DispatchKey::SparseCUDA,
     DispatchKey::SparseHIP,
     DispatchKey::SparseVE,
     DispatchKey::SparseCsrCPU,
     DispatchKey::SparseCsrCUDA,
     DispatchKey::Meta});

// The set of dispatch keys that come after autograd
// n.b. this relies on the fact that AutogradOther is currently the lowest
// Autograd key
constexpr DispatchKeySet after_autograd_keyset =
    DispatchKeySet(DispatchKeySet::FULL_AFTER, c10::DispatchKey::AutogradOther);

// The set of dispatch keys that come after ADInplaceOrView
constexpr DispatchKeySet after_ADInplaceOrView_keyset = DispatchKeySet(
    DispatchKeySet::FULL_AFTER,
    c10::DispatchKey::ADInplaceOrView);

// true if t is a backend dispatch key
C10_API bool isBackendDispatchKey(DispatchKey t);

// Resolve alias dispatch key to DispatchKeySet if applicable
C10_API DispatchKeySet getRuntimeDispatchKeySet(DispatchKey t);

// Returns a DispatchKeySet of all backend keys mapped to Autograd dispatch key
// t, DispatchKeySet is empty if t is not alias of DispatchKey::Autograd.
C10_API DispatchKeySet getBackendKeySetFromAutograd(DispatchKey t);

// Returns a DispatchKeySet of autograd related keys mapped to backend.
C10_API DispatchKeySet getAutogradRelatedKeySetFromBackend(DispatchKey t);

// Returns a DispatchKeySet of autocast related keys mapped to backend.
C10_API DispatchKeySet getAutocastRelatedKeySetFromBackend(DispatchKey t);

// This API exists because we have a use case for checking
// getRuntimeDispatchKeySet(alias).has(DispatchKey::Undefined)
// in OperatorEntry.cpp but we disallow it in has() API.
C10_API bool isIncludedInAlias(DispatchKey k, DispatchKey alias);

// Historically, every tensor only had a single DispatchKey, and it was always
// something like CPU, and there wasn't any of this business where TLS
// could cause the DispatchKey of a tensor to change.  But we still have some
// legacy code that is still using DispatchKey for things like instanceof
// checks; if at all possible, refactor the code to stop using DispatchKey in
// those cases.
static inline DispatchKey legacyExtractDispatchKey(DispatchKeySet s) {
  // NB: If you add any extra keys that can be stored in TensorImpl on
  // top of existing "backend" keys like CPU/CUDA, you need to add it
  // here.  At the moment, autograd keys and ADInplaceOrView key need this
  // treatment;
  return (s - autograd_dispatch_keyset_with_ADInplaceOrView -
          autocast_dispatch_keyset)
      .highestPriorityTypeId();
}

template <class T>
using is_not_DispatchKeySet = guts::negation<std::is_same<DispatchKeySet, T>>;

// Given a function type, constructs a function_traits type that drops the first
// parameter type if the first parameter is of type DispatchKeySet. NB:
// DispatchKeySet is currently explicitly hidden from JIT (mainly to avoid
// pushing unnecessary arguments on the stack - see Note [ Plumbing Keys Through
// the Dispatcher] for details). If at any point in the future we need to expose
// this type to JIT, revisit the usage of this type alias.
template <class FuncType>
using remove_DispatchKeySet_arg_from_func = guts::make_function_traits_t<
    typename guts::infer_function_traits_t<FuncType>::return_type,
    typename std::conditional_t<
        std::is_same<
            DispatchKeySet,
            typename guts::typelist::head_with_default_t<
                void,
                typename guts::infer_function_traits_t<
                    FuncType>::parameter_types>>::value,
        guts::typelist::drop_if_nonempty_t<
            typename guts::infer_function_traits_t<FuncType>::parameter_types,
            1>,
        typename guts::infer_function_traits_t<FuncType>::parameter_types>>;
} // namespace c10
