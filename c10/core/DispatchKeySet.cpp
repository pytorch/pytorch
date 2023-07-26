#include <c10/core/DispatchKeySet.h>
#include <c10/util/irange.h>

namespace c10 {

// backend_dispatch_keyset includes all dispatch keys that map to backends.
// Alias key DispatchKey::CompositeExplicitAutograd maps to
// backend_dispatch_keyset
constexpr DispatchKeySet backend_dispatch_keyset =
    autogradother_backends | DispatchKeySet(DispatchKey::Dense);

// See Note [CompositeExplicitAutogradNonFunctional Key]
// We have several types of decompositions in aten, that each have their own
// alias key. You should register your decomposition to the
// `CompositeExplicitAutogradNonFunctional key` if: (1) It's an out-of-place op
// (2) It decomposes into one more mutation ops
// (3) It has a derivative formula
//     (In theory we could also have a separate key for
//     "CompositeImplicitAutogradNonFunctional", but there isn't much of a use
//     case for it currently).
// This key is important for "functional" backends like LazyTensor / XLA.
// If you're a backend that only expects to deal with "functional ops",
// then you don't want to decompose a functional op into an op that causes
// aliasing. You should just directly write a kernel for that functional op
// instead!
constexpr DispatchKeySet non_functional_backend_dispatch_keyset =
    backend_dispatch_keyset
        // XLA and LazyTensor are currently the only 2 backends in core
        // that use functionalization pass in eager mode.
        .remove(DispatchKey::Sparse)
        .remove_backend(BackendComponent::XLABit)
        .remove_backend(BackendComponent::LazyBit);

bool isBackendDispatchKey(DispatchKey t) {
  return t != DispatchKey::Undefined
      // See Note [No Alias Keys in DispatchKeySet]
      && !isAliasDispatchKey(t)
      // Note [NestedTensor Not Included in Backend Keys]
      // NestedTensor has been explicitly removed from the "backend keyset" due
      // to incompatibility with some kernels, so we don't want it to be
      // included in CompositeExplicitAutograd kernels.
      && t != DispatchKey::NestedTensor && backend_dispatch_keyset.has(t);
}

// math_dispatch_keyset contains all keys in backend_dispatch_keyset and
// autograd_dispatch_keyset Alias key DispatchKey::CompositeImplicitAutograd
// maps to [math_dispatch_keyset x full_backend_mask]
constexpr DispatchKeySet math_dispatch_keyset = backend_dispatch_keyset |
    autograd_dispatch_keyset |
    // See Note [NestedTensor Not Included in Backend Keys]
    // The caveat to that note is that nested_tensor is a special case
    // where we would like to support composite implicit kernels but not
    // explicit kernels therefore we manually add the key to the
    // math_dispatch_keyset
    DispatchKeySet{DispatchKey::NestedTensor} |
    // Functionalize should always re-use CompositeImplicit decomps.
    DispatchKeySet{DispatchKey::Functionalize};

constexpr DispatchKeySet nested_dispatch_keyset =
    DispatchKeySet(
        {DispatchKey::AutogradNestedTensor, DispatchKey::NestedTensor}) |
    DispatchKeySet(DispatchKeySet::RAW, full_backend_mask);

DispatchKeySet getRuntimeDispatchKeySet(DispatchKey t) {
  TORCH_INTERNAL_ASSERT(t != DispatchKey::Undefined);
  switch (t) {
    case DispatchKey::Autograd:
      // See Note [autograd_dispatch_keyset Does Not Include Backend Bits]
      // That's why we OR it with a mask of the backend bits here.
      // getRuntimeDispatchKeySet() expects to return a keyset of runtime
      // dispatch keys, like AutogradCPU, but that requires having backend bits.
      return autograd_dispatch_keyset |
          DispatchKeySet(DispatchKeySet::RAW, full_backend_mask);
    case DispatchKey::CompositeImplicitAutograd:
      return math_dispatch_keyset;
    case DispatchKey::CompositeImplicitAutogradNestedTensor:
      return nested_dispatch_keyset;
    case DispatchKey::CompositeExplicitAutograd:
      return backend_dispatch_keyset;
    case DispatchKey::CompositeExplicitAutogradNonFunctional:
      return non_functional_backend_dispatch_keyset;
    default:
      return DispatchKeySet(t);
  }
}

bool runtimeDispatchKeySetHas(DispatchKey t, DispatchKey k) {
  TORCH_INTERNAL_ASSERT(t != DispatchKey::Undefined);
  switch (t) {
    case DispatchKey::Autograd:
      return autograd_dispatch_keyset.has(toFunctionalityKey(k));
    case DispatchKey::CompositeImplicitAutograd:
      // See Note [NestedTensor Not Included in Backend Keys]
      return math_dispatch_keyset.has(k);
    case DispatchKey::CompositeImplicitAutogradNestedTensor:
      // See Note [NestedTensor Not Included in Backend Keys]
      return nested_dispatch_keyset.has(k);
    case DispatchKey::CompositeExplicitAutograd:
      // See Note [NestedTensor Not Included in Backend Keys]
      return k != DispatchKey::NestedTensor && backend_dispatch_keyset.has(k);
    case DispatchKey::CompositeExplicitAutogradNonFunctional:
      // See Note [NestedTensor Not Included in Backend Keys]
      return k != DispatchKey::NestedTensor &&
          non_functional_backend_dispatch_keyset.has(k);
    case DispatchKey::FuncTorchBatchedDecomposition:
      return functorch_batched_ks.has(k);
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
    case DispatchKey::AutogradMeta:
      return DispatchKeySet(DispatchKey::Meta);
    case DispatchKey::AutogradMPS:
      return DispatchKeySet(DispatchKey::MPS);
    case DispatchKey::AutogradHPU:
      return DispatchKeySet(DispatchKey::HPU);
    case DispatchKey::AutogradIPU:
      return DispatchKeySet(DispatchKey::IPU);
    case DispatchKey::AutogradXPU:
      return DispatchKeySet(DispatchKey::XPU);
    case DispatchKey::AutogradPrivateUse1:
      return DispatchKeySet(DispatchKey::PrivateUse1);
    case DispatchKey::AutogradPrivateUse2:
      return DispatchKeySet(DispatchKey::PrivateUse2);
    case DispatchKey::AutogradPrivateUse3:
      return DispatchKeySet(DispatchKey::PrivateUse3);
    case DispatchKey::AutogradNestedTensor:
      return DispatchKeySet(DispatchKey::NestedTensor) |
          DispatchKeySet(DispatchKeySet::RAW, full_backend_mask);
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
  TORCH_INTERNAL_ASSERT(next_functionality_ <= iterator::end_iter_mask_val);
  TORCH_INTERNAL_ASSERT(next_backend_ <= num_backends, next_backend_);

  // Create a masked version of the set representation to ignore previous
  // keys that we've iterated through.
  uint64_t masked_functionality_bits =
      llvm::maskTrailingZeros<uint64_t>(next_functionality_) & *data_ptr_;
  uint64_t masked_backend_bits =
      llvm::maskTrailingZeros<uint64_t>(next_backend_) & full_backend_mask &
      *data_ptr_;

  uint64_t first_functionality_idx =
      llvm::findFirstSet(masked_functionality_bits);
  uint64_t first_backendcomponent_idx = llvm::findFirstSet(masked_backend_bits);

  // If there are no keys, set to end iterator value
  if (first_functionality_idx == std::numeric_limits<uint64_t>::max() ||
      next_functionality_ == iterator::end_iter_mask_val) {
    // Set up state to be the same as end()
    next_functionality_ = iterator::end_iter_mask_val;
    current_dispatchkey_idx_ = iterator::end_iter_key_val;
    next_backend_ = 0;
    current_backendcomponent_idx_ = iterator::end_iter_key_val;
    return *this;
  }

  // The +1 is because of DispatchKey::Undefined and
  // BackendComponent::InvalidBit
  auto new_next_functionality = first_functionality_idx + 1;
  auto new_backendcomponent_idx = first_backendcomponent_idx + 1;
  // and the -num_backends is because the first <num_backends> bits in the
  // keyset are not Dispatch Keys.
  auto next_dispatchkey_idx = new_next_functionality - num_backends;

  // If the current functionality bit is a per-backend bit, we need special
  // handling
  if (isPerBackendFunctionalityKey(
          static_cast<DispatchKey>(next_dispatchkey_idx))) {
    // case 1: if the current backend is undefined, then there is no valid
    // backend instance of this functionality key so we can skip it.
    if (first_backendcomponent_idx == std::numeric_limits<uint64_t>::max()) {
      // increment the functionality mask so we skip the current functionality
      // bit on the next increment.
      next_functionality_ = new_next_functionality;
      ++(*this);
      return *this;
    }

    // Otherwise, at this point we know what the current backend and
    // functionality bits are.
    current_dispatchkey_idx_ = next_dispatchkey_idx;
    current_backendcomponent_idx_ = new_backendcomponent_idx;

    // Next, we need to set up the masks for the next increment.
    uint64_t next_backendcomponent_bits =
        llvm::maskTrailingZeros<uint64_t>(first_backendcomponent_idx + 1) &
        full_backend_mask & *data_ptr_;
    uint64_t next_backendcomponent_idx =
        llvm::findFirstSet(next_backendcomponent_bits);
    if (next_backendcomponent_idx == std::numeric_limits<uint64_t>::max()) {
      // case 2: the current backend is valid, but there is not another backend
      // in the keyset. In this case, we need to bump the functionality mask and
      // reset the backend mask for the next increment
      next_functionality_ = new_next_functionality;
      next_backend_ = 0;
    } else {
      // case 3: we have another backend to iterate over. We want to iterate
      // over the same functionality bit next time, but a different backend bit.
      next_backend_ = first_backendcomponent_idx + 1;
    }
  } else {
    // Functionality bits that aren't per backend are simpler to handle. We can
    // ignore the backend bits.
    TORCH_INTERNAL_ASSERT(next_backend_ == 0);
    current_dispatchkey_idx_ = next_dispatchkey_idx;
    next_functionality_ = new_next_functionality;
  }
  return *this;
}

std::array<FunctionalityOffsetAndMask, num_functionality_keys>
initializeFunctionalityOffsetsAndMasks() {
  std::array<FunctionalityOffsetAndMask, num_functionality_keys>
      offsets_and_masks;
  // manually set the first entry, which corresponds to Undefined.
  offsets_and_masks[0] = FunctionalityOffsetAndMask(0, 0);
  // loop through every functionality key (aside from Undefined).
  for (const auto functionality_idx : c10::irange(1, num_functionality_keys)) {
    // functionality_idx should be Dense -> 1, ...
    auto prev_offset_and_mask = offsets_and_masks[functionality_idx - 1];
    auto k = static_cast<DispatchKey>(functionality_idx);

    // If the previous functionality was not per-backend, then we can just
    // increment the previous offset. Otherwise, the next offset =
    // previous_offset + num_backends.
    auto next_offset = prev_offset_and_mask.offset +
        (prev_offset_and_mask.mask == 0 ? 1 : num_backends);
    // the mask is used in the runtime index calculation to find the offset of
    // the backend. For non-per-backend functionalities, this offset should
    // always be 0. Otherwise, we need to get the index of the backend (which we
    // can do using a backend mask).
    auto next_mask = isPerBackendFunctionalityKey(k) ? full_backend_mask : 0;
    offsets_and_masks[functionality_idx] =
        FunctionalityOffsetAndMask(next_offset, next_mask);
  }
  // Sanity check that the computed offset index of the last functionality key
  // is correct. This assumes that the highest priority functionality key is not
  // per backend.
  TORCH_INTERNAL_ASSERT(
      offsets_and_masks[num_functionality_keys - 1].offset ==
          (num_runtime_entries - 1),
      "num_runtime_entries: ",
      num_runtime_entries,
      "last_offset: ",
      offsets_and_masks[num_functionality_keys - 1].offset);
  return offsets_and_masks;
}

} // namespace c10
