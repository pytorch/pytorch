#include <c10/core/DispatchKeySet.h>
#include <c10/util/irange.h>

namespace c10 {

bool isBackendDispatchKey(DispatchKey t) {
  return t != DispatchKey::Undefined
      // See Note [No Alias Keys in DispatchKeySet]
      && !isAliasDispatchKey(t) && backend_dispatch_keyset.has(t);
}

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

DispatchKeySet::iterator& DispatchKeySet::iterator::operator++() {
      TORCH_INTERNAL_ASSERT(functionality_mask_ >= num_backends);
      TORCH_INTERNAL_ASSERT(functionality_mask_ <= iterator::end_iter_mask_val);
      TORCH_INTERNAL_ASSERT(backend_mask_ <= num_backends);

      // Create a masked version of the set representation to ignore previous
      // keys that we've iterated through.
      uint64_t masked_functionality_bits = llvm::maskTrailingZeros<uint64_t>(functionality_mask_) & *data_ptr_;
      uint64_t masked_backend_bits = llvm::maskTrailingZeros<uint64_t>(backend_mask_) & full_backend_mask & *data_ptr_;

      uint64_t first_functionality_idx = llvm::findFirstSet(masked_functionality_bits);
      uint64_t first_backend_idx = llvm::findFirstSet(masked_backend_bits);

      // If there are no keys, set to end iterator value
      if (first_functionality_idx == std::numeric_limits<uint64_t>::max() ||
          functionality_mask_ == iterator::end_iter_mask_val) {
        // Set up state to be the same as end()
        functionality_mask_ = iterator::end_iter_mask_val;
        functionality_idx_ = iterator::end_iter_key_val;
        backend_mask_ = 0;
        backend_idx_ = iterator::end_iter_key_val;
        return *this;
      }

      // The +1 is because of DispatchKey::Undefined and BackendBit::InvalidBit
      auto new_functionality_mask = first_functionality_idx + 1;
      auto new_backend_idx = first_backend_idx + 1;
      // and the -num_backends is because the first <num_backends> bits in the keyste are not Dispatch Keys.
      auto new_functionality_idx = new_functionality_mask - num_backends;

      // If the current functionality bit is a per-backend bit, we need special handling
      if (isPerBackendFunctionalityKey(static_cast<DispatchKey>(new_functionality_idx))) {
        // case 1: if the current backend is undefined, then there is no valid backend instance
        // of this functionality key so we can skip it.
        if (first_backend_idx == std::numeric_limits<uint64_t>::max()) {
          // increment the functionality mask so we skip the current functionality bit on the next increment.
          functionality_mask_ = new_functionality_mask;
          ++(*this);
          return *this;
        }

        // Otherwise, at this point we know what the current backend and functionality bits are.
        functionality_idx_ = new_functionality_idx;
        backend_idx_ = new_backend_idx;

        // Next, we need to set up the masks for the next increment.
        uint64_t next_backend_bits = llvm::maskTrailingZeros<uint64_t>(first_backend_idx + 1) & full_backend_mask & *data_ptr_;
        uint64_t next_backend_idx = llvm::findFirstSet(next_backend_bits);
        if (next_backend_idx == std::numeric_limits<uint64_t>::max()) {
          // case 2: the current backend is valid, but there is not another backend in the keyset.
          // In this case, we need to bump the functionality mask and reset the backend mask for the next increment
          functionality_mask_ = new_functionality_mask;
          backend_mask_ = 0;
        } else {
          // case 3: we have another backend to iterate over. We want to iterate over the same functionality bit
          // next time, but a different backend bit.
          backend_mask_ = first_backend_idx + 1;
        }
      } else {
          // Functionality bits that aren't per backend are simpler to handle. We can ignore the backend bits.
          TORCH_INTERNAL_ASSERT(backend_mask_ == 0);
          functionality_idx_ = new_functionality_idx;
          functionality_mask_ = new_functionality_mask;
      }
      return *this;
    }

std::array<FunctionalityOffsetAndMask, num_functionality_keys>
initializeFunctionalityOffsetsAndMasks() {
    std::array<FunctionalityOffsetAndMask, num_functionality_keys>
    offsets_and_masks;
    // manual set the first entry, which corresponds to Undefined.
    offsets_and_masks[0] = FunctionalityOffsetAndMask(0, 0);
    // loop through every functionality key (aside from Undefined).
    for (const auto functionality_idx : c10::irange(1, num_functionality_keys)) {
        // functionality_idx should be Dense -> 1, ...
        auto prev_offset_and_mask = offsets_and_masks[functionality_idx - 1];
        auto k = static_cast<DispatchKey>(functionality_idx);

#if defined(C10_MOBILE_TRIM_DISPATCH_KEYS)
// [Note: Trimmed Mobile Dispatch Keys]
        uint16_t mask = 0;
        uint16_t offset = -1;
        switch (k) {
          case DispatchKey::Undefined:
            offset = 0;
          case DispatchKey::CPU:
            offset = 1;
          case DispatchKey::QuantizedCPU:
            offset = 2;
          case DispatchKey::SparseCPU:
            offset = 3;
          case DispatchKey::BackendSelect:
            offset = 4;
          case DispatchKey::ADInplaceOrView:
            offset = 5;
          case DispatchKey::AutogradOther:
            offset = 6;
          case DispatchKey::AutogradCPU:
            offset = 7;
          default:
            offset = -1;
        }
        offsets_and_masks[functionality_idx] = FunctionalityOffsetAndMask(offset, 0);
#else
        // If the previous functionality was not per-backend, then we can just increment the previous offset.
        // Otherwise, the next offset = previous_offset + num_backends.
        auto next_offset = prev_offset_and_mask.offset + (prev_offset_and_mask.mask == 0 ? 1 : num_backends);
        // the mask is used in the runtime index calculation to find the offset of the backend.
        // For non-per-backend functionalities, this offset should always be 0.
        // Otherwise, we need to get the index of the backend (which we can do using a backend mask).
        auto next_mask = isPerBackendFunctionalityKey(k) ? full_backend_mask : 0;
        offsets_and_masks[functionality_idx] = FunctionalityOffsetAndMask(next_offset, next_mask);
    }
    // Sanity check that the computed offset index of the last functionality key is correct.
    // This assumes that the highest priority functionality key is not per backend.
    TORCH_INTERNAL_ASSERT(offsets_and_masks[num_functionality_keys - 1].offset == (num_runtime_entries - 1),
        "num_runtime_entries: ", num_runtime_entries,
        "last_offset: ", offsets_and_masks[num_functionality_keys - 1].offset);
#endif
    return offsets_and_masks;
}

} // namespace c10
