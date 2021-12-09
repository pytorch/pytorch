#include <c10/core/DispatchKeySet.h>

namespace c10 {

// autograd_dispatch_keyset should include all runtime autograd keys.
// Alias key DispatchKey::Autograd maps to autograd_dispatch_keyset.
// NB: keys in this set also get associated with CompositeImplicitAutograd
DispatchKeySet autograd_dispatch_keyset = DispatchKeySet({
    DispatchKey::AutogradFunctionality,
    DispatchKey::AutogradOther,
}) | DispatchKeySet(DispatchKeySet::RAW, full_backend_mask);

DispatchKeySet autocast_dispatch_keyset = DispatchKeySet({
    DispatchKey::AutocastCPU,
    DispatchKey::AutocastCUDA,
});

// See Note [TLS Initialization]
DispatchKeySet default_included_set = DispatchKeySet({
    DispatchKey::BackendSelect,
    DispatchKey::ADInplaceOrView,
});

DispatchKeySet default_excluded_set = DispatchKeySet({
    DispatchKey::AutocastCPU,
    DispatchKey::AutocastCUDA,
});

DispatchKeySet autograd_dispatch_keyset_with_ADInplaceOrView =
    autograd_dispatch_keyset | DispatchKeySet(DispatchKey::ADInplaceOrView);

// backend dispatch keys that map to DispatchKey::AutogradOther
// NB: keys in this set also get associated with CompositeImplicitAutograd
DispatchKeySet autogradother_backends = DispatchKeySet(
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
     DispatchKey::CustomRNGKeyId,
     DispatchKey::MkldnnCPU,
     DispatchKey::Meta});

// The set of dispatch keys that come after autograd
// n.b. this relies on the fact that AutogradOther is currently the lowest
// Autograd key
DispatchKeySet after_autograd_keyset =
    DispatchKeySet(DispatchKeySet::FULL_AFTER, c10::DispatchKey::AutogradOther);

// The set of dispatch keys that come after ADInplaceOrView
DispatchKeySet after_ADInplaceOrView_keyset = DispatchKeySet(
    DispatchKeySet::FULL_AFTER,
    c10::DispatchKey::ADInplaceOrView);

// The set of dispatch keys that come after Functionalize
DispatchKeySet after_func_keyset =
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
DispatchKeySet backend_dispatch_keyset = autogradother_backends |
    DispatchKeySet(DispatchKeySet::RAW, full_backend_mask) |
    DispatchKeySet({
        DispatchKey::Dense,
        DispatchKey::Sparse,
        DispatchKey::Quantized,
    });

bool isBackendDispatchKey(DispatchKey t) {
  return t != DispatchKey::Undefined
      // See Note [No Alias Keys in DispatchKeySet]
      && !isAliasDispatchKey(t) && backend_dispatch_keyset.has(t);
}

// math_dispatch_keyset contains all keys in backend_dispatch_keyset and
// autograd_dispatch_keyset Alias key DispatchKey::CompositeImplicitAutograd
// maps to math_dispatch_keyset.
DispatchKeySet math_dispatch_keyset =
    backend_dispatch_keyset | autograd_dispatch_keyset;

DispatchKeySet getRuntimeDispatchKeySet(DispatchKey t) {
  TORCH_INTERNAL_ASSERT(t != DispatchKey::Undefined);
  switch (t) {
    case DispatchKey::Autograd:
      return autograd_dispatch_keyset;
    case DispatchKey::CompositeImplicitAutograd:
      return math_dispatch_keyset;
    case DispatchKey::CompositeExplicitAutograd:
      return backend_dispatch_keyset;
    default:
      return DispatchKeySet(t);
  }
}

bool runtimeDispatchKeySetHas(DispatchKey t, DispatchKey k) {
  TORCH_INTERNAL_ASSERT(t != DispatchKey::Undefined);
  switch (t) {
    case DispatchKey::Autograd:
      return autograd_dispatch_keyset.has(k);
    case DispatchKey::CompositeImplicitAutograd:
      return math_dispatch_keyset.has(k);
    case DispatchKey::CompositeExplicitAutograd:
      return backend_dispatch_keyset.has(k);
    default:
      return t == k;
  }
}

// for a given autograd key, return the (guaranteed nonempty) set of associated
// backend keys. for a non-autograd key, return the empty keyset.
DispatchKeySet getBackendKeySetFromAutograd(DispatchKey t) {
  switch (t) {
    case DispatchKey::AutogradCPU:
      return DispatchKeySet(DispatchKey::CPU);
    case DispatchKey::AutogradCUDA:
      return DispatchKeySet(DispatchKey::CUDA);
    case DispatchKey::AutogradXLA:
      return DispatchKeySet(DispatchKey::XLA);
    case DispatchKey::AutogradLazy:
      return DispatchKeySet(DispatchKey::Lazy);
    case DispatchKey::AutogradMLC:
      return DispatchKeySet(DispatchKey::MLC);
    case DispatchKey::AutogradHPU:
      return DispatchKeySet(DispatchKey::HPU);
    case DispatchKey::AutogradNestedTensor:
      return DispatchKeySet(DispatchKey::NestedTensor);
    case DispatchKey::AutogradXPU:
      return DispatchKeySet(DispatchKey::XPU);
    case DispatchKey::AutogradPrivateUse1:
      return DispatchKeySet(DispatchKey::PrivateUse1);
    case DispatchKey::AutogradPrivateUse2:
      return DispatchKeySet(DispatchKey::PrivateUse2);
    case DispatchKey::AutogradPrivateUse3:
      return DispatchKeySet(DispatchKey::PrivateUse3);
    case DispatchKey::AutogradOther:
      return autogradother_backends;
    default:
      return DispatchKeySet();
  }
}

DispatchKeySet getAutocastRelatedKeySetFromBackend(DispatchKey t) {
  TORCH_INTERNAL_ASSERT(t <= DispatchKey::EndOfBackendKeys || t == DispatchKey::Undefined);
  switch (t) {
    case DispatchKey::CPUBit:
      return DispatchKeySet(DispatchKey::AutocastCPU);
    case DispatchKey::CUDABit:
    case DispatchKey::XLA:
      return DispatchKeySet(DispatchKey::AutocastCUDA);
    default:
      return DispatchKeySet();
  }
}

DispatchKeySet getAutogradRelatedKeySetFromBackend(DispatchKey t) {
  return DispatchKeySet(
      {DispatchKey::ADInplaceOrView, getAutogradKeyFromBackend(t)});
}

bool isIncludedInAlias(DispatchKey k, DispatchKey alias) {
  return k != DispatchKey::Undefined && runtimeDispatchKeySetHas(alias, k);
}

std::string toString(DispatchKeySet ts) {
  std::stringstream ss;
  ss << ts;
  return ss.str();
}

std::ostream& operator<<(std::ostream& os, DispatchKeySet ts) {
  if (ts.empty()) {
    os << "DispatchKeySet()";
    return os;
  }
  os << "DispatchKeySet(";
  bool first = true;
  for (auto k : ts) {
    if (!first) {
      os << ", ";
    }
    os << k;
    first = false;
  }
  os << ")";
  return os;
}

C10_API uint64_t keyset_ctr(DispatchKey k) {
    uint64_t repr_ = 0;
    // Technically it would be possible to add alias dispatch keys to a DispatchKeySet,
    // but the semantics are a little confusing and this currently isn't needed anywhere.
    TORCH_INTERNAL_ASSERT(!isAliasDispatchKey(k));

    if (k == DispatchKey::Undefined) {
      // Case 1: handle Undefined specifically
      repr_ = 0;
    } else if (k <= DispatchKey::EndOfBackendKeys) {
      // Case 2: handle "backend-only" keys
      // These keys (e.g. DispatchKey::CPUBit) have a backend bit set, but no functionality bits.
      uint64_t backend_val = 1ULL << static_cast<uint8_t>(k);
      repr_ = backend_val;
    } else if (isPerBackendFunctionalityKey(k)) {
      // Case 3: handle "per-backend-functionality" keys
      // E.g. for DispatchKey::Dense, or DispatchKey::Sparse,
      // we'll only set the functionality bit and not set any backend bits.
      // The - 1 is because Undefined is technically a "functionality" that doesn't show up in the bitset.
      // So e.g. Dense is technically the second functionality, but the lowest functionality bit.
      uint64_t functionality_val = 1ULL << (static_cast<uint8_t>(k) - 1);
      repr_ = functionality_val;
    } else {
      // Case 4: "runtime" keys that have a functionality bit.
      // First compute which bit to flip for the functionality.
      auto functionality_k = toFunctionalityKey(k);
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(functionality_k > DispatchKey::EndOfBackendKeys);
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(functionality_k <= DispatchKey::EndOfFunctionalityKeys);
      // The - 1 is because Undefined is technically a "functionality" that doesn't show up in the bitset.
      // So e.g. Dense is technically the second functionality, but the lowest functionality bit.
      uint64_t functionality_val = 1ULL << (static_cast<uint8_t>(functionality_k) - 1);

      // then compute which bit to flip for the backend
      if (k > DispatchKey::EndOfFunctionalityKeys) {
        // Case 4a: handle the runtime instances of "per-backend functionality" keys
        // For example, given DispatchKey::CPU, we should set:
        // - the Dense functionality bit
        // - the CPUBit backend bit
        // first compute which bit to flip for the backend
        auto backend_k = toBackendKey(k);
        uint64_t backend_val = 1ULL << static_cast<uint8_t>(backend_k);
        repr_ = functionality_val + backend_val;
      } else {
        // Case 4b handle the runtime functionality keys that are not per-backend.
        // In this case, the functionality key is not per-backend, so we don't set any backend bits.
        // e.g. DispatchKey::FuncTorchBatched.
        repr_ = functionality_val;
      }
    }
    return repr_;
}
DispatchKeySet::iterator& DispatchKeySet::iterator::operator++() {
      TORCH_INTERNAL_ASSERT(
          functionality_mask_ > static_cast<uint8_t>(DispatchKey::EndOfBackendKeys));
      TORCH_INTERNAL_ASSERT(
          functionality_mask_ <= static_cast<uint8_t>(DispatchKey::EndOfFunctionalityKeys));
      TORCH_INTERNAL_ASSERT(
          backend_mask_ <= static_cast<uint8_t>(DispatchKey::EndOfBackendKeys));

      // Create a masked version of the set representation to ignore previous
      // keys that we've iterated through.
      uint64_t masked_functionality_bits = llvm::maskTrailingZeros<uint64_t>(functionality_mask_) & *data_ptr_;
      uint64_t masked_backend_bits = llvm::maskTrailingZeros<uint64_t>(backend_mask_) & full_backend_mask & *data_ptr_;

      uint64_t first_functionality_idx = llvm::findFirstSet(masked_functionality_bits);
      uint64_t first_backend_idx = llvm::findFirstSet(masked_backend_bits);

      // If there are no keys, set to end iterator value
      if (first_functionality_idx == std::numeric_limits<uint64_t>::max() ||
          functionality_mask_ == static_cast<uint8_t>(DispatchKey::EndOfFunctionalityKeys)) {
        // Set up state to be the same as end()
        functionality_mask_ = static_cast<uint8_t>(DispatchKey::EndOfFunctionalityKeys);
        functionality_idx_ = static_cast<uint8_t>(DispatchKey::EndOfFunctionalityKeys);
        backend_mask_ = 0;
        backend_idx_ = static_cast<uint8_t>(DispatchKey::EndOfFunctionalityKeys);
        return *this;
      }

      // If the current functionality bit is a per-backend bit, we need special handling
      if (isPerBackendFunctionalityKey(static_cast<DispatchKey>(first_functionality_idx + 1))) {
        // case 1: if the current backend is undefined, then there is no valid backend instance
        // of this functionality key so we can skip it.
        if (first_backend_idx == std::numeric_limits<uint64_t>::max()) {
          // increment the functionality mask so we skip the current functionality bit on the next increment.
          functionality_mask_ = static_cast<uint8_t>(first_functionality_idx) + 1;
          ++(*this);
          return *this;
        }

        // Otherwise, at this point we know what the current backend and functionality bits are.
        // (The +1 for the functionality idx but not the backend idx is because of the Undefined key)
        functionality_idx_ = static_cast<uint8_t>(first_functionality_idx) + 1;
        backend_idx_ = static_cast<uint8_t>(first_backend_idx);

        // Next, we need to set up the masks for the next increment.
        uint64_t next_backend_bits = llvm::maskTrailingZeros<uint64_t>(first_backend_idx + 1) & full_backend_mask & *data_ptr_;
        uint64_t next_backend_idx = llvm::findFirstSet(next_backend_bits);
        if (next_backend_idx == std::numeric_limits<uint64_t>::max()) {
          // case 2: the current backend is valid, but there is not another backend in the keyset.
          // In this case, we need to bump the functionality mask and reset the backend mask for the next increment
          functionality_mask_ = static_cast<uint8_t>(first_functionality_idx) + 1;
          backend_mask_ = 0;
        } else {
          // case 3: we have another backend to iterate over. We want to iterate over the same functionality bit
          // next time, but a different backend bit.
          backend_mask_ = static_cast<uint8_t>(first_backend_idx) + 1;
        }
      } else {
          // Functionality bits that aren't per backend are simpler to handle. We can ignore the backend bits.
          TORCH_INTERNAL_ASSERT(backend_mask_ == 0);
          functionality_idx_ = static_cast<uint8_t>(first_functionality_idx) + 1;
          functionality_mask_ = static_cast<uint8_t>(first_functionality_idx) + 1;
      }
      return *this;
    }

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

static std::array<FunctionalityOffsetAndMask, num_functionality_keys>
initializeFunctionalityOffsetsAndMasks() {
    std::array<FunctionalityOffsetAndMask, num_functionality_keys>
    offsets_and_masks_;
    // manual set the first entry, which corresponds to Undefined.
    offsets_and_masks_[0] = FunctionalityOffsetAndMask(0, 0);
    // loop through every functionality key (aside from Undefined).
    for (uint8_t k_idx = static_cast<uint8_t>(DispatchKey::EndOfBackendKeys) + 2;
            k_idx < static_cast<uint8_t>(DispatchKey::EndOfFunctionalityKeys); ++k_idx) {
        // functionality_idx should be Dense -> 1, ...
        auto functionality_idx = k_idx - static_cast<uint8_t>(DispatchKey::EndOfBackendKeys) - 1;
        auto prev_offset_and_mask = offsets_and_masks_[functionality_idx - 1];
        auto k = static_cast<DispatchKey>(k_idx);


        // If the previous functionality was not per-backend, then we can just increment the previous offset.
        // Otherwise, the next offset = previous_offset + num_backends.
        auto next_offset = prev_offset_and_mask.offset +
            (prev_offset_and_mask.mask == 0 ? 1 : (static_cast<uint8_t>(DispatchKey::EndOfBackendKeys) + 1));
        // the mask is used in the runtime index calculation to find the offset of the backend.
        // For non-per-backend functionalities, this offset should always be 0.
        // Otherwise, we need to get the index of the backend (which we can do using a backend mask).
        auto next_mask = isPerBackendFunctionalityKey(k) ? full_backend_mask : 0;
        offsets_and_masks_[functionality_idx] = FunctionalityOffsetAndMask(next_offset, next_mask);
    }
    // Sanity check that the computed offset index of the last functionality key is correct.
    // This assumes that the highest priority functionality key is not per backend.
    TORCH_INTERNAL_ASSERT(offsets_and_masks_[num_functionality_keys - 1].offset == (num_runtime_entries - 1),
        "num_runtime_entries: ", num_runtime_entries,
        "last_offset: ", offsets_and_masks_[num_functionality_keys - 1].offset);
    return offsets_and_masks_;
}

static std::array<FunctionalityOffsetAndMask, num_functionality_keys>
offsets_and_masks_ = initializeFunctionalityOffsetsAndMasks();

DispatchKey DispatchKeySet::highestFunctionalityKey() const {
    auto functionality_idx = indexOfHighestBit();
    // This means that none of the functionality bits were set.
    if (functionality_idx <= static_cast<uint8_t>(DispatchKey::Undefined)) return DispatchKey::Undefined;
    // Add 1 to deal with undefined being a dispatch key below all functionality keys.
    return static_cast<DispatchKey>(functionality_idx);
}
// This is effectively like toBackendKey(DispatchKey), but less restrictive.
// toBackendKey() errors out if the key that it was passed has no backend bits,
// which is useful for error checking.
// We need a version of that here that can also handle "fake" backends like FPGA,
// because they need to map to the AutogradOther key.
DispatchKey DispatchKeySet::highestBackendKey() const {
    // mask to mask out functionality bits
    auto backend_idx = DispatchKeySet(repr_ & full_backend_mask).indexOfHighestBit();
    // all zeros across the backend bits means that no backend bits are set.
    if (backend_idx == 0) return DispatchKey::Undefined;
    // Subtract 1 because backend_idx=1 --> the CPU key, which has value 0 in the enum.
    return static_cast<DispatchKey>(backend_idx - 1);
}

uint64_t DispatchKeySet::getDispatchTableIndexForDispatchKeySet() const {
    auto functionality_idx =
        DispatchKeySet(repr_ >> (static_cast<uint8_t>(DispatchKey::EndOfBackendKeys) + 1))
           .indexOfHighestBit();
    auto offset_and_mask = offsets_and_masks_[functionality_idx];
    // Mask the functionality bits out first, then right-shift by 1.
    // right-shifting by 1 because everything is zero-indexed.
    // E.g. 000001 (CPU) should give us an offset of 0, 000010 (CUDA) should give us an offset of 1, etc.
    auto backend_idx = DispatchKeySet((repr_ & offset_and_mask.mask) >> 1).indexOfHighestBit();
    return offset_and_mask.offset + backend_idx;
}


} // namespace c10
