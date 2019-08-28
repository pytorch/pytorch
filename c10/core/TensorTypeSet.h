#pragma once

#include <c10/core/TensorTypeId.h>
#include <c10/util/llvmMathExtras.h>
#include <ostream>

namespace c10 {

// A representation of a set of TensorTypeIds.  A tensor may
// have multiple tensor type ids, e.g., a Variable tensor
// can also be a CPU tensor; the TensorTypeSet specifies what
// type ids apply.  The internal representation is as a 64-bit
// bit set (this means only 64 tensor type ids are supported).
//
// At the moment, there are no nontrivial uses of this set; tensors
// are always singletons.  In the near future, this set will represent
// variable? + tensor type id.  In the far future, it will be
// variable? + profiling? + tracing? + lazy? + tensor type id.
//
// An undefined tensor is one with an empty tensor type set.
class TensorTypeSet {
public:
  TensorTypeSet() {}
  explicit TensorTypeSet(TensorTypeId t)
    : repr_(t == TensorTypeId::UndefinedTensorId
              ? 0
              : 1ULL << (static_cast<uint8_t>(t) - 1)) {}
  // Test if a TensorTypeId is in the set
  bool has(TensorTypeId t) const {
    return static_cast<bool>(repr_ & TensorTypeSet(t).repr_);
  }
  // Perform set union
  TensorTypeSet operator|(TensorTypeSet other) const {
    return TensorTypeSet(repr_ | other.repr_);
  }
  // Perform set equality
  bool operator==(TensorTypeSet other) const {
    return repr_ == other.repr_;
  }
  // Add a TensorTypeId to the TensorTypeId set.  Does NOT mutate,
  // returns the extended TensorTypeSet!
  C10_NODISCARD TensorTypeSet add(TensorTypeId t) const {
    return *this | TensorTypeSet(t);
  }
  // Remove a TensorTypeId from the TensorTypeId set.  This is
  // generally not an operation you should be doing (it's
  // used to implement operator<<)
  C10_NODISCARD TensorTypeSet remove(TensorTypeId t) const {
    return TensorTypeSet(repr_ & ~TensorTypeSet(t).repr_);
  }
  // Is the set empty?  (AKA undefined tensor)
  bool empty() const {
    return repr_ == 0;
  }
  // Return the "first" type id in this set; i.e., the one that
  // should handle dispatch for this operator
  TensorTypeId firstTypeId() const {
    // TODO: If I put UndefinedTensorId as entry 64 and then adjust the
    // singleton constructor to shift from the right, we can get rid of the
    // subtraction here.  It's modestly more complicated to get right so I
    // didn't do it for now.
    return static_cast<TensorTypeId>(64 - llvm::countTrailingZeros(repr_));
  }
private:
  TensorTypeSet(uint64_t repr) : repr_(repr) {}
  uint64_t repr_ = 0;
};

std::string toString(TensorTypeSet);
std::ostream& operator<<(std::ostream&, TensorTypeSet);

}
