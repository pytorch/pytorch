#pragma once

#include <c10/core/DispatchKey.h>
#include <c10/util/Exception.h>
#include <c10/util/Metaprogramming.h>
#include <c10/util/llvmMathExtras.h>
#include <ostream>
#include <array>

namespace c10 {
C10_API uint64_t keyset_ctr(DispatchKey);

// A representation of a set of DispatchKeys.  A tensor may have multiple
// tensor type ids, e.g., a Variable tensor can also be a CPU tensor; the
// DispatchKeySet specifies what type ids apply.  The internal representation is
// as a 64-bit bit set (this means only 64 tensor type ids are supported).
//
// TODO: modernize this description.
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
class C10_API DispatchKeySet final {
 public:
  enum Full { FULL };
  enum FullAfter { FULL_AFTER };
  enum Raw { RAW };

  // NB: default constructor representation as zero is MANDATORY as
  // use of DispatchKeySet in TLS requires this.
  constexpr DispatchKeySet() : repr_(0) {}

  constexpr DispatchKeySet(Full)
      //: repr_(std::numeric_limits<decltype(repr_)>::max()) {}
      : repr_((1ULL << (static_cast<uint8_t>(DispatchKey::EndOfFunctionalityKeys) - 1)) - 1) {}

  constexpr DispatchKeySet(FullAfter, DispatchKey t)
      // LSB after t are OK, but not t itself.
      : repr_((1ULL << (static_cast<uint8_t>(t) - 1)) - 1) {
    // "functionalities" have a notion of ordering (e.g. Autograd > Sparse > Quantized > Dense).
    // But backends don't really have an ordering.
    // Therefore, we're enforcing that FullAfter can only be used on "functionality" keys.
    TORCH_INTERNAL_ASSERT(t > DispatchKey::EndOfBackendKeys && t <= DispatchKey::EndOfFunctionalityKeys);
  }

  // Public version of DispatchKeySet(uint64_t) API; external users
  // must be explicit when they do this!
  constexpr DispatchKeySet(Raw, uint64_t x) : repr_(x) {}

  // This is difficult to make constexpr: toFunctionalityKey and toBackendKey will need to be constexpr too
  explicit DispatchKeySet(DispatchKey k) {
      repr_ = keyset_ctr(k);
  }

  explicit constexpr DispatchKeySet(std::initializer_list<DispatchKey> ks)
      : repr_(0) {
    for (auto k : ks) {
      repr_ |= DispatchKeySet(k).repr_;
    }
  }
  // Test if a DispatchKey is in the set
  bool inline has(DispatchKey t) const {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(t != DispatchKey::Undefined);
    auto ks = DispatchKeySet(t);
    return static_cast<bool>((repr_ & ks.repr_) == ks.repr_);
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
  // Compute the set difference self - other,
  // but ONLY for the functionality keys.
  // Any backend bits set on self will remain unchanged.
  DispatchKeySet operator-(DispatchKeySet other) const {
    return DispatchKeySet(repr_ & (full_backend_mask | ~other.repr_));
  }
  // Perform a right bitshift
  DispatchKeySet operator>>(uint8_t shift) const {
    return DispatchKeySet(repr_ >> shift);
  }

  // Compute self ^ other
  // TODO: check where this is used
  constexpr DispatchKeySet operator^(DispatchKeySet other) const {
    return DispatchKeySet(repr_ ^ other.repr_);
  }
  bool operator==(DispatchKeySet other) const {
    return repr_ == other.repr_;
  }
  bool operator!=(DispatchKeySet other) const {
    return repr_ != other.repr_;
  }
  // Add a DispatchKey to the DispatchKey set.  Does NOT mutate,
  // returns the extended DispatchKeySet!
  C10_NODISCARD DispatchKeySet add(DispatchKey t) const {
    return *this | DispatchKeySet(t);
  }
  // TODO: remove
  //C10_NODISCARD DispatchKeySet remove(DispatchKey t) const {
    //return DispatchKeySet(repr_ & ~DispatchKeySet(t).repr_);
  //}

  // Remove a DispatchKey from the DispatchKey set.
  // Only functionality bits are allowed to be removed from a keyset.
  // This is generally not an operation you should be doing
  // (it's used to implement operator<<)
  C10_NODISCARD DispatchKeySet removeFunctionalityKey(DispatchKey t) const {
    // For now, we're only allowing removal of "functionality bits" from the keyset,
    // which is specifically needed by the fallthrough key calculation logic.
    // Why is removing backend bits problematic? Consider this example:
    //
    // DispatchKeySet([DispatchKey.CPU, DispatchKey.AutogradCUDA, DispatchKey.CUDA]).remove(DispatchKey.AutogradCUDA)
    // DispatchKeySet([DispatchKey.CPU, DispatchKey.AutogradCUDA]).remove(DispatchKey.AutogradCUDA)
    //
    // What do we want to happen?
    // Technically, we'd like it to be true that after removal,
    // the first keyset still has the CUDA dispatch key while the second doesn't.
    // Unfortunately there's no way to represent that, because the two keysets are represented the same way internally:
    // functionality bits: Autograd, Dense
    // backend bits: CPU, CUDA
    //
    // Instead, removeFunctionalityKey(DispatchKey.AutogradCPU) will only remove the "Autograd" bit from the bitset.
    TORCH_INTERNAL_ASSERT(t > DispatchKey::EndOfBackendKeys);
    TORCH_INTERNAL_ASSERT(!isAliasDispatchKey(t));
    auto functionality_k = isPerBackendFunctionalityKey(t) ? t : toFunctionalityKey(t);
    return DispatchKeySet(repr_ & ~DispatchKeySet(functionality_k).repr_);
  }
  // Is the set empty?  (AKA undefined tensor)
  bool empty() const {
    return repr_ == 0;
  }
  uint64_t raw_repr() {
    return repr_;
  }

  DispatchKey highestFunctionalityKey() const;
  DispatchKey highestBackendKey() const;

  // returns the DispatchKey of highest priority in the set.
  DispatchKey highestPriorityTypeId() const {
    auto functionality_k = highestFunctionalityKey();
    if (isPerBackendFunctionalityKey(functionality_k)) {
      return toRuntimePerBackendFunctionalityKey(functionality_k, highestBackendKey());
    }
    return functionality_k;
  }

  // TODO remove
  //DispatchKey highestPriorityTypeId() const {
    //return static_cast<DispatchKey>(64 - llvm::countLeadingZeros(repr_));
  //}

  // TODO remove
  //DispatchKey highestPriorityBackendTypeId() const {
    //return (*this &
            //((1ULL << static_cast<uint8_t>(DispatchKey::EndOfBackendKeys)) - 1))
        //.highestPriorityTypeId();
  //}
  // Returns the index of the most-significant bit in the keyset.
  // This is used to as part of the calculation into the operator table to get:
  // - the highest "functionality" bit in the keyset.
  // - the highest "backend" bit in the keyset.
  uint8_t indexOfHighestBit() const {
    return 64 - llvm::countLeadingZeros(repr_);
  }

  // returns the index in the operator table of highest priority key in the the keyset
  // Note that we could in theory implement this using highestPriorityTypeId(),
  // but this code is very hotpath and we can do it faster without it.
  uint64_t getDispatchTableIndexForDispatchKeySet() const;

 private:
  constexpr DispatchKeySet(uint64_t repr) : repr_(repr) {}
  uint64_t repr_ = 0;

 public:
  // STL iterator for DispatchKeySet. Iterates through all DispatchKeys in the
  // set. The iterator is only invalidated by the destruction of the underlying
  // DispatchKeySet as the iterator stores a pointer to the raw representation
  // of the DispatchKeySet.
  // Note: We only iterate through *functionality* keys. The only purpose that
  // the backend bits serve during iteration is to decide which runtime key
  // to return when we hit a per-backend functionality key.
  // For example, if the next functionality key to iterate over is Autograd,
  // and the backend bits in the keyset correspond to [DispatchKey::CPUBit, DispatchKey::CUDABit],
  // then the next key we return will be DispatchKey::AutogradCUDA
  // (because CUDA has higher precedence than CPU among backend keys).
  class iterator {
   public:
    using self_type = iterator;
    using iterator_category = std::input_iterator_tag;
    using value_type = DispatchKey;
    using difference_type = ptrdiff_t;

    // functionality_idx_ will iterate through all functionality bits.
    // backend_idx_ will iterate through all backend bits.
    explicit iterator(
            const uint64_t* data_ptr,
            uint8_t functionality_mask = static_cast<uint8_t>(DispatchKey::EndOfBackendKeys) + 1,
            uint8_t backend_mask = 0)
        : data_ptr_(data_ptr),
          functionality_mask_(functionality_mask),
          backend_mask_(backend_mask),
          // These are in an invalid state at construction time, and set by the first increment call
          functionality_idx_(static_cast<uint8_t>(DispatchKey::EndOfFunctionalityKeys)),
          backend_idx_(static_cast<uint8_t>(DispatchKey::EndOfFunctionalityKeys)) {
      // Go to the first key in the set
      ++(*this);
    }

    C10_API self_type& operator++();

    self_type operator++(int) {
      self_type previous_iterator = *this;
      ++(*this);
      return previous_iterator;
    }

    bool operator==(const self_type& rhs) const {
      return functionality_mask_ == rhs.functionality_mask_ &&
             functionality_idx_ == rhs.functionality_idx_ &&
             backend_mask_ == rhs.backend_mask_ &&
             backend_idx_ == rhs.backend_idx_;
    }
    bool operator!=(const self_type& rhs) const {
      return functionality_mask_ != rhs.functionality_mask_ ||
             functionality_idx_ != rhs.functionality_idx_ ||
             backend_mask_ != rhs.backend_mask_ ||
             backend_idx_ != rhs.backend_idx_;
    }
    DispatchKey operator*() const {
      auto functionality_key = static_cast<DispatchKey>(functionality_idx_);
      if (isPerBackendFunctionalityKey(functionality_key)) {
         auto next_key = toRuntimePerBackendFunctionalityKey(functionality_key, static_cast<DispatchKey>(backend_idx_));
         // We expect all of the Dense, Sparse, Quantized, and Autograd keys to be ordered the same way
         // with respect to their backends
         TORCH_INTERNAL_ASSERT(toBackendKey(next_key) == static_cast<DispatchKey>(backend_idx_),
           "Tried to map functionality key ", toString(functionality_key), " and backend key ",
           toString(static_cast<DispatchKey>(backend_idx_)), " to a runtime key, but ended up with ", toString(next_key),
           ". This can happen if the order of the backend dispatch keys in DispatchKey.h isn't consistent.",
           " Please double check that enum for inconsistencies.");
         return next_key;
      } else {
        return functionality_key;
      }
    }

   private:
    const uint64_t* data_ptr_;
    uint8_t functionality_mask_;
    uint8_t backend_mask_;
    uint8_t functionality_idx_;
    uint8_t backend_idx_;
  };

 public:
  // Returns iterator to the first key in the set. If no keys are in the
  // set, then will return the end iterator.
  iterator begin() const {
    return iterator(&repr_);
  }

  // We do not need to iterate beyond EndOfFunctionalityKeys so we will treat this as
  // the end iterator.
  iterator end() const {
    return iterator(&repr_, static_cast<uint8_t>(DispatchKey::EndOfFunctionalityKeys));
  }
};

C10_API std::string toString(DispatchKeySet);
C10_API std::ostream& operator<<(std::ostream&, DispatchKeySet);

extern C10_API DispatchKeySet autograd_dispatch_keyset;
extern C10_API DispatchKeySet autocast_dispatch_keyset;
extern C10_API DispatchKeySet default_included_set;
extern C10_API DispatchKeySet default_excluded_set;
extern C10_API DispatchKeySet autograd_dispatch_keyset_with_ADInplaceOrView;
extern C10_API DispatchKeySet autogradother_backends;
extern C10_API DispatchKeySet after_autograd_keyset;
extern C10_API DispatchKeySet after_ADInplaceOrView_keyset;
extern C10_API DispatchKeySet after_func_keyset;

constexpr DispatchKeySet backend_bitset_mask = DispatchKeySet(
    DispatchKeySet::RAW,
    (1ULL << (static_cast<uint8_t>(DispatchKey::EndOfBackendKeys) + 1)) - 1);

struct OpTableOffsetAndMask {
  uint16_t offset;
  uint16_t backend_mask;
};

static_assert(
    static_cast<uint8_t>(DispatchKey::EndOfBackendKeys) < 16,
    "Right now we expect the number of backends not to exceed 16. In the (unlikely) event"
    " that this changes, the size of OpTableOffsetAndMask::backend_mask needs to be increased too.");

// true if t is a backend dispatch key
C10_API bool isBackendDispatchKey(DispatchKey t);

// Resolve alias dispatch key to DispatchKeySet if applicable
C10_API DispatchKeySet getRuntimeDispatchKeySet(DispatchKey t);

// Resolve alias dispatch key to DispatchKeySet if applicable,
// and chek if k is a part of that set
C10_API bool runtimeDispatchKeySetHas(DispatchKey t, DispatchKey k);

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

C10_API uint64_t getDispatchTableIndexForDispatchKeySet(DispatchKeySet ks);

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
  auto ks = s.removeFunctionalityKey(DispatchKey::AutogradFunctionality)
             .removeFunctionalityKey(DispatchKey::ADInplaceOrView)
             .removeFunctionalityKey(DispatchKey::AutocastCPU)
             .removeFunctionalityKey(DispatchKey::AutocastCUDA);
  return ks.highestPriorityTypeId();
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
