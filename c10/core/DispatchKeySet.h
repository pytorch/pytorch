#pragma once
#include <c10/core/DispatchKey.h>
#include <c10/util/Exception.h>
#include <c10/util/Metaprogramming.h>
#include <c10/util/llvmMathExtras.h>
#include <ostream>
#include <array>

namespace c10 {

struct FunctionalityOffsetAndMask {
    // empty constructor shouldn't be used; only needed to initialize
    // the array before populating it.
    FunctionalityOffsetAndMask() {}
    FunctionalityOffsetAndMask(uint16_t offset, uint16_t mask) :
        offset(offset), mask(mask) {}
    // This needs to big enough to cover the size of the operator table.
    uint16_t offset;
    // This mask needs to be big enough to mask all of the backend bits.
    // We probably don't ever want to have more than 16 backend bits, so uint16_t should be enough.
    uint16_t mask;
};

C10_API std::array<FunctionalityOffsetAndMask, num_functionality_keys>
initializeFunctionalityOffsetsAndMasks();

C10_ALWAYS_INLINE static std::array<FunctionalityOffsetAndMask, num_functionality_keys>&
offsetsAndMasks() {
    static auto offsets_and_masks_ = initializeFunctionalityOffsetsAndMasks();
    return offsets_and_masks_;
}

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
// Note [DispatchKeySet Internal Representation]
// Internally, dispatch keys are packed into 64-bit DispatchKeySet objects
// that get passed around at runtime.
// However, there isn't necessarily a 1-to-1 mapping between bits in the keyset
// and individual dispatch keys.
//
// First: why do we have this distinction, and why not map every dispatch key directly to a bit?
// This is mostly because we have several types of functionalities
// that different backends would like to customize.
// For example, we have:
// - "Dense":     CPU, CUDA, XLA, ... (~12 keys)
// - "Sparse":    SparseCPU, SparseCUDA, ...
// - "Quantized": QuantizedCPU, QuantizedCUDA, QuantizedXLA, ...
// - "Autograd":  AutogradCPU, AutogradCUDA, Autograd XLA, ...
// The problem is that total number of keys grows quadratically with [# backends] x [# functionalities],
// making it very difficult to map each key directly to a bit in a bitset without dramatically increasing
// the size of the bitset over time.
//
// Dispatch keys can be divided into 3 broad categories.
//
//
// (1) "Building block" keys
//    (a) Backend-bit keys (e.g. CPUBit, CUDABIt)
//    (b) (per-backend) functionality-bit keys (e.g. AutogradFunctionality, Sparse, Dense)
//
// Building block keys always correspond to individual bits in a DispatchKeySet.
// They can also be combined in a DispatchKeySet to form actual runtime keys.
// e.g.
//     auto dense_cpu_ks = DispatchKeySet({DispatchKey::CPUBit, DispatchKey::Dense});
//     // The keyset has the runtime dense-cpu key.
//     dense_cpu_ks.has(DispatchKey::CPU);
//     // And it contains the building block keys too.
//     dense_cpu_ks.has(DispatchKey::CPUBit);
//     dense_cpu_ks.has(DispatchKey::Dense);
//
// Not every backend and not every functionality counts as a "building block key".
// This is mostly to give us more levers to pull in the design space.
// Backend keys and functionality keys that count as "building blocks" will contribute
// to a full cross product of functionality that can be overriden.
//
// For example, right now we have at least 12 "backend" building blocks (CPU, CUDA, XLA, ...)
// and at least 4 "functionality" building blocks (Dense, Sparse, Quantized, AutogradFunctionality, ...).
// These keys together allow every dispatcher operator to be customized in up to 12*4 different ways.
// Each of those requires a slot in the operator table of every dispatcher operator.
// Not every piece of functionality necessarily needs to be customizeable per-backend,
// and not every backend necessarily needs to be able to customize every type of functionality.
//
//
// (2) "Runtime" keys
//    (a) Per-backend functionality "instances" (AutogradCUDA, SparseCUDA, CUDA)
//    (b) backend-agnostic functionalities (e.g. FuncTorchBatched)
//    (c) non-customizeable backends (e.g. FPGA)
//
// Every runtime key corresponds directly to a slot in an operator's runtime dispatch table,
// and you can directly register kernels to a runtime dispatch key.
//
// For per-backend functionalities like "Dense" or "AutogradFunctionality",
// you can think of the corresponding runtime dispatch keys as "instances" of that functionality, per backend.
// E.g. "CPU", "CUDA", "XLA", etc. are all runtime instances of the "Dense" building block key.

// (2a) and (2b) are represented identically:
// - backend-agnostic functionalities (e.g. FuncTorchBatched) are NOT customizeable per backend.
//   In order to do so, we'd need to promote it to a per-backend functionality "building block" key.
// - non-customizeable backends (e.g. FPGA) can NOT customize existing functionality like Sparse, Autograd, etc.
//   In order to do so, we'd need to promote it to a backend "building block" key.
//
// In both cases, these keys directly correspond to runtime slots in the operator table.
//
//
// (3) "Alias" keys
// See Note [Alias Dispatch Keys]

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
      : repr_((1ULL << (num_backends + num_functionality_keys - 1)) - 1) {}

  constexpr DispatchKeySet(FullAfter, DispatchKey t)
      // LSB after t are OK, but not t itself.
      // "functionalities" have a notion of ordering (e.g. Autograd > Sparse > Quantized > Dense).
      // But backends don't really have an ordering.
      // Therefore, we're enforcing that FullAfter can only be used on "functionality" keys.
      : repr_((1ULL << (num_backends + static_cast<uint8_t>(toFunctionalityKey(t)) - 1)) - 1) {
  }

  // Public version of DispatchKeySet(uint64_t) API; external users
  // must be explicit when they do this!
  constexpr DispatchKeySet(Raw, uint64_t x) : repr_(x) {}

  constexpr explicit DispatchKeySet(BackendBit k) {
    if (k == BackendBit::InvalidBit) {
      repr_ = 0;
    } else {
      repr_ = 1ULL << (static_cast<uint8_t>(k) - 1);
    }
  }

  constexpr explicit DispatchKeySet(DispatchKey k) {
    if (k == DispatchKey::Undefined) {
      // Case 1: handle Undefined specifically
      repr_ = 0;
    } else if (k <= DispatchKey::EndOfFunctionalityKeys) {
      // Case 2: handle "functionality-only" keys
      // These keys have a functionality bit set, but no backend bits
      // These can technically be either:
      // - valid runtime keys (e.g. DispatchKey::AutogradOther, DispatchKey::FuncTorchBatched, etc)
      // - "building block" keys that aren't actual runtime keys (e.g. DispatchKey::Dense or Sparse)
      uint64_t functionality_val = 1ULL << (num_backends + static_cast<uint8_t>(k) - 1);
      repr_ = functionality_val;
    } else if (k <= DispatchKey::EndOfAutogradBackends) {
      // Case 3: "runtime" keys that have a functionality bit AND a backend bit.
      // First compute which bit to flip for the functionality.
      auto functionality_k = toFunctionalityKey(k);
      // The - 1 is because Undefined is technically a "functionality" that doesn't show up in the bitset.
      // So e.g. Dense is technically the second functionality, but the lowest functionality bit.
      uint64_t functionality_val = 1ULL << (num_backends + static_cast<uint8_t>(functionality_k) - 1);

      // then compute which bit to flip for the backend
      // Case 4a: handle the runtime instances of "per-backend functionality" keys
      // For example, given DispatchKey::CPU, we should set:
      // - the Dense functionality bit
      // - the CPUBit backend bit
      // first compute which bit to flip for the backend
      auto backend_k = toBackendBit(k);
      uint64_t backend_val = 1ULL << (static_cast<uint8_t>(backend_k) - 1);
      repr_ = functionality_val + backend_val;
    } else {
      // At this point, we should have covered every case except for alias keys.
      // Technically it would be possible to add alias dispatch keys to a DispatchKeySet,
      // but the semantics are a little confusing and this currently isn't needed anywhere.
      repr_ = 0;
    }

  }

  explicit constexpr DispatchKeySet(std::initializer_list<DispatchKey> ks)
      : repr_(0) {
    for (auto k : ks) {
      repr_ |= DispatchKeySet(k).repr_;
    }
  }

  explicit constexpr DispatchKeySet(std::initializer_list<BackendBit> ks)
      : repr_(0) {
    for (auto k : ks) {
      repr_ |= DispatchKeySet(k).repr_;
    }
  }

  // Test if a DispatchKey is in the set
  inline bool has(DispatchKey t) const {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(t != DispatchKey::Undefined);
    return has_all(DispatchKeySet(t));
  }
  constexpr bool has(BackendBit t) const {
    return has_all(DispatchKeySet(t));
  }

  // Test if a DispatchKey is in the set
  // Given a DispatchKeySet of functionality keys and (potentially) backend keys,
  // tests if all of them are in the current set.
  constexpr bool has_all(DispatchKeySet ks) const {
    return static_cast<bool>((repr_ & ks.repr_) == ks.repr_);
  }

  // Given a DispatchKeySet of functionality keys and (potentially) backend keys,
  // tests if any of them are in the current set.
  // This could technically be pretty easily implemented using has().
  // It is strictly a perf optimization though.
  // There are many places in the code base
  // where we want to test for multiple functionality keys together.
  // HOWEVER, runtime per-backend functionality keys aren't allowed to be used with this function,
  // because you can end up with weird results.
  // e.g. DispatchKeySet(DispatchKey::AutogradCPU).has_any(DispatchKeySet(DispatchKey::CPU)) would return true.
  inline bool has_any(DispatchKeySet ks) const {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      // Either there are no backend bits in the input keyset
      ((ks.repr_ & full_backend_mask) == 0) ||
      // or there are no per-backend-functionality bits
      ((ks & DispatchKeySet({
          DispatchKey::Dense,
          DispatchKey::Quantized,
          DispatchKey::Sparse,
          DispatchKey::AutogradFunctionality,
        }).repr_) == 0));
    return static_cast<bool>((repr_ & ks.repr_) != 0);
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
  constexpr DispatchKeySet operator&(DispatchKeySet other) const {
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
  C10_NODISCARD DispatchKeySet add(DispatchKeySet ks) const {
    return *this | ks;
  }

  // Remove a DispatchKey from the DispatchKey set.
  // Only functionality bits are allowed to be removed from a keyset.
  // This is generally not an operation you should be doing
  // (it's used to implement operator<<)
  constexpr DispatchKeySet removeFunctionalityKey(DispatchKey t) const {
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
    return DispatchKeySet(repr_ & ~(DispatchKeySet(t).repr_ & ~full_backend_mask));
  }
  constexpr DispatchKeySet removeFunctionalityKeys(DispatchKeySet ks) const {
    return DispatchKeySet(repr_ & ~(ks.repr_ & ~full_backend_mask));
  }
  // Is the set empty?  (AKA undefined tensor)
  bool empty() const {
    return repr_ == 0;
  }
  uint64_t raw_repr() {
    return repr_;
  }

  DispatchKey highestFunctionalityKey() const {
    auto functionality_idx = indexOfHighestBit();
    // This means that none of the functionality bits were set.
    if (functionality_idx < num_backends) return DispatchKey::Undefined;
    // The first num_backend bits in the keyset don't correspond to real dispatch keys.
    return static_cast<DispatchKey>(functionality_idx - num_backends);
  }

  // This is similar like toBackendBit(DispatchKey), but less restrictive.
  // toBackendBit() errors out if the key that it was passed has no backend bits,
  // which is useful for error checking.
  // We need a version of that here that can also handle "fake" backends like FPGA,
  // because they need to map to the AutogradOther key. For those backends,
  // we return BackendBit::InvalidBit.
  BackendBit highestBackendKey() const {
    // mask to mask out functionality bits
    auto backend_idx = DispatchKeySet(repr_ & full_backend_mask).indexOfHighestBit();
    // all zeros across the backend bits means that no backend bits are set.
    if (backend_idx == 0) return BackendBit::InvalidBit;
    return static_cast<BackendBit>(backend_idx);
  }

  // returns the DispatchKey of highest priority in the set.
  DispatchKey highestPriorityTypeId() const {
    auto functionality_k = highestFunctionalityKey();
    if (isPerBackendFunctionalityKey(functionality_k)) {
      return toRuntimePerBackendFunctionalityKey(functionality_k, highestBackendKey());
    }
    return functionality_k;
  }

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
  uint64_t getDispatchTableIndexForDispatchKeySet() const {
    auto functionality_idx =
        DispatchKeySet(repr_ >> num_backends)
           .indexOfHighestBit();
    auto offset_and_mask = offsetsAndMasks()[functionality_idx];
    // Mask the functionality bits out first, then right-shift by 1.
    // right-shifting by 1 because everything is zero-indexed.
    // E.g. 000001 (CPU) should give us an offset of 0, 000010 (CUDA) should give us an offset of 1, etc.
    auto backend_idx = DispatchKeySet((repr_ & offset_and_mask.mask) >> 1).indexOfHighestBit();
    return offset_and_mask.offset + backend_idx;
  }

 private:
  constexpr DispatchKeySet(uint64_t repr) : repr_(repr) {}
  uint64_t repr_ = 0;

 public:
  // STL iterator for DispatchKeySet. Iterates through all runtime DispatchKeys in
  // the set. The iterator is only invalidated by the destruction of the underlying
  // DispatchKeySet as the iterator stores a pointer to the raw representation
  // of the DispatchKeySet.
  // Note: When we encounter a per-backend functionality (e.g. Dense or Sparse),
  // we will iterate through EVERY backend in the keyset, for that functionality.
  // For example, if the next functionality key to iterate over is Autograd,
  // and the backend bits in the keyset correspond to [BackendBit::CPUBit, DispatchKey::CUDABit],
  // then the next two keys we return will be DispatchKey::AutogradCPU, DispatchKey::AutogradCUDA
  // (CPU first because it has lower precedence than CUDA in DispatchKey.h).
  class iterator {
   public:
    using self_type = iterator;
    using iterator_category = std::input_iterator_tag;
    using value_type = DispatchKey;
    using difference_type = ptrdiff_t;
    // final mask value should mask out the entire keyset
    static const uint8_t end_iter_mask_val = num_backends + num_functionality_keys;
    // final key value should be the last DispatchKey
    static const uint8_t end_iter_key_val = num_functionality_keys;

    // functionality_idx_ will iterate through all functionality bits.
    // backend_idx_ will iterate through all backend bits.
    explicit iterator(
            const uint64_t* data_ptr,
            uint8_t functionality_mask = num_backends,
            uint8_t backend_mask = 0)
        : data_ptr_(data_ptr),
          functionality_mask_(functionality_mask),
          backend_mask_(backend_mask),
          // These are in an invalid state at construction time, and set by the first increment call
          functionality_idx_(end_iter_key_val),
          backend_idx_(end_iter_key_val) {
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
         auto next_key = toRuntimePerBackendFunctionalityKey(functionality_key, static_cast<BackendBit>(backend_idx_));
         // We expect all of the Dense, Sparse, Quantized, and Autograd keys to be ordered the same way
         // with respect to their backends
         TORCH_INTERNAL_ASSERT(toBackendBit(next_key) == static_cast<BackendBit>(backend_idx_),
           "Tried to map functionality key ", toString(functionality_key), " and backend bit ",
           toString(static_cast<BackendBit>(backend_idx_)), " to a runtime key, but ended up with ", toString(next_key),
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
    return iterator(&repr_, iterator::end_iter_mask_val);
  }
};

C10_API std::string toString(DispatchKeySet);
C10_API std::ostream& operator<<(std::ostream&, DispatchKeySet);

// autograd_dispatch_keyset should include all runtime autograd keys.
// Alias key DispatchKey::Autograd maps to autograd_dispatch_keyset.
// NB: keys in this set also get associated with CompositeImplicitAutograd
constexpr DispatchKeySet autograd_dispatch_keyset = DispatchKeySet({
        DispatchKey::AutogradFunctionality,
        DispatchKey::AutogradOther,
    }) | DispatchKeySet(DispatchKeySet::RAW, full_backend_mask);

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
        // TODO: delete commented code before landing.
        // HIP and VE now have their own backend bits, which means that
        // they can now have their own Autograd keys.
        // Technically, HIP will now redispatch to its own custom AutogradHIP slot
        // in the runtime table.
        //{DispatchKey::HIP,
        //DispatchKey::VE,
        {DispatchKey::FPGA,
        DispatchKey::ORT,
        DispatchKey::Vulkan,
        DispatchKey::Metal,
        DispatchKey::SparseCsrCPU,
        DispatchKey::SparseCsrCUDA,
        DispatchKey::CustomRNGKeyId,
        DispatchKey::MkldnnCPU,
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

// The set of dispatch keys that come after Functionalize
constexpr DispatchKeySet after_func_keyset =
        DispatchKeySet(DispatchKeySet::FULL_AFTER, c10::DispatchKey::Functionalize)
            .removeFunctionalityKey(
                // NOTE: we also need to remove ADInplaceOrView from the keyset when
                // redispatching after the func kernels. This is because we're not
                // calling the same op; we originally called an inplace op, and now
                // we aren't. The original key calculation figured out which keys
                // were Fallthrough based on the inplace op. That means that it did
                // not include the ADInPlaceOrView kernel as a fallthrough key.
                // However, we WANT the ADInPlaceOrView kernel to be ignored now
                // that we're calling an out-of-place op. Re-invoking
                // Dispatcher::call would re-run the Fallthrough key calculation and
                // get us that, But at::redispatch is more performant. We can get
                // away with it by explicitly removing the key here.
                c10::DispatchKey::ADInplaceOrView);

// backend_dispatch_keyset should include all runtime backend keys.
// Alias key DispatchKey::CompositeExplicitAutograd maps to
// backend_dispatch_keyset NestedTensor has been explicitly removed due to
// incompatibility with some kernels, such as structured kernels, that use the
// DefaultBackend key.
constexpr DispatchKeySet backend_dispatch_keyset = autogradother_backends |
        DispatchKeySet(DispatchKeySet::RAW, full_backend_mask) |
        DispatchKeySet({
            DispatchKey::Dense,
            DispatchKey::Sparse,
            DispatchKey::Quantized,
        });

// math_dispatch_keyset contains all keys in backend_dispatch_keyset and
// autograd_dispatch_keyset Alias key DispatchKey::CompositeImplicitAutograd
// maps to math_dispatch_keyset.
constexpr DispatchKeySet math_dispatch_keyset =
        backend_dispatch_keyset | autograd_dispatch_keyset;

constexpr DispatchKeySet backend_bitset_mask = DispatchKeySet(
    DispatchKeySet::RAW, (1ULL << num_backends) - 1);

constexpr auto inplace_or_view_ks = DispatchKeySet(DispatchKey::ADInplaceOrView);
constexpr auto autograd_cpu_ks = DispatchKeySet(DispatchKey::AutogradCPU);
constexpr auto autograd_xpu_ks = DispatchKeySet(DispatchKey::AutogradXPU);
constexpr auto autograd_cuda_ks = DispatchKeySet(DispatchKey::AutogradCUDA);
constexpr auto autograd_xla_ks = DispatchKeySet(DispatchKey::AutogradXLA);
constexpr auto autograd_lazy_ks = DispatchKeySet(DispatchKey::AutogradLazy);
constexpr auto autograd_mlc_ks = DispatchKeySet(DispatchKey::AutogradMLC);
constexpr auto autograd_hpu_ks = DispatchKeySet(DispatchKey::AutogradHPU);
constexpr auto autograd_nestedtensor_ks = DispatchKeySet(DispatchKey::AutogradNestedTensor);
constexpr auto autograd_privateuse1_ks = DispatchKeySet(DispatchKey::AutogradPrivateUse1);
constexpr auto autograd_privateuse2_ks = DispatchKeySet(DispatchKey::AutogradPrivateUse2);
constexpr auto autograd_privateuse3_ks = DispatchKeySet(DispatchKey::AutogradPrivateUse3);
constexpr auto autograd_other_ks = DispatchKeySet(DispatchKey::AutogradOther);

struct OpTableOffsetAndMask {
  uint16_t offset;
  uint16_t backend_mask;
};

static_assert(
    num_backends <= 16,
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
// for a given backend key, use the associated autograd key.
// for non-backend keys, use AutogradOther as a default.
// Note: it's convenient and fast to return a default here rather than (say)
// returning an optional<DispatchKey>, or throwing. But it makes callers
// responsible for either a) enforcing the invariant that only backend keys
// be passed as arguments, or b) interpreting our return value carefully.
inline DispatchKeySet getAutogradRelatedKeySetFromBackend(BackendBit t) {
  switch (t) {
    case BackendBit::CPUBit:
      return inplace_or_view_ks | autograd_cpu_ks;
    case BackendBit::XPUBit:
      return inplace_or_view_ks | autograd_xpu_ks;
    case BackendBit::CUDABit:
      return inplace_or_view_ks | autograd_cuda_ks;
    case BackendBit::XLABit:
      return inplace_or_view_ks | autograd_xla_ks;
    case BackendBit::LazyBit:
      return inplace_or_view_ks | autograd_lazy_ks;
    case BackendBit::MLCBit:
      return inplace_or_view_ks | autograd_mlc_ks;
    case BackendBit::HPUBit:
      return inplace_or_view_ks | autograd_hpu_ks;
    case BackendBit::NestedTensorBit:
      return inplace_or_view_ks | autograd_nestedtensor_ks;
    case BackendBit::PrivateUse1Bit:
      return inplace_or_view_ks | autograd_privateuse1_ks;
    case BackendBit::PrivateUse2Bit:
      return inplace_or_view_ks | autograd_privateuse2_ks;
    case BackendBit::PrivateUse3Bit:
      return inplace_or_view_ks | autograd_privateuse3_ks;
    default:
      return inplace_or_view_ks | autograd_other_ks;
  }
}

// Returns a DispatchKeySet of autocast related keys mapped to backend.
inline DispatchKeySet getAutocastRelatedKeySetFromBackend(BackendBit t) {
  constexpr auto autocast_cpu_ks = DispatchKeySet(DispatchKey::AutocastCPU);
  constexpr auto autocast_cuda_ks = DispatchKeySet(DispatchKey::AutocastCUDA);
  switch (t) {
    case BackendBit::CPUBit:
      return autocast_cpu_ks;
    case BackendBit::CUDABit:
    case BackendBit::XLABit:
      return autocast_cuda_ks;
    default:
      return DispatchKeySet();
  }
}

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
