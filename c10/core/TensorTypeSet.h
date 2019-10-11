#pragma once

#include <c10/core/TensorTypeId.h>
#include <c10/util/llvmMathExtras.h>
#include <c10/util/Exception.h>
#include <ostream>

namespace c10 {

// A representation of a set of TensorTypeIds.  A tensor may have multiple
// tensor type ids, e.g., a Variable tensor can also be a CPU tensor; the
// TensorTypeSet specifies what type ids apply.  The internal representation is
// as a 64-bit bit set (this means only 64 tensor type ids are supported).
//
// Note that TensorTypeIds are ordered; thus, we can ask questions like "what is
// the highest priority TensorTypeId in the set"?  (The set itself is not
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
class TensorTypeSet final {
public:
  enum Full { FULL };
  enum Raw { RAW };

  // NB: default constructor representation as zero is MANDATORY as
  // use of TensorTypeSet in TLS requires this.
  TensorTypeSet()
    : repr_(0) {}
  TensorTypeSet(Full)
    : repr_(-1) {}
  // Public version of TensorTypeSet(uint64_t) API; external users
  // must be explicit when they do this!
  TensorTypeSet(Raw, uint64_t x)
    : repr_(x) {}
  explicit TensorTypeSet(TensorTypeId t)
    : repr_(t == TensorTypeId::UndefinedTensorId
              ? 0
              : 1ULL << (static_cast<uint8_t>(t) - 1)) {}
  // Test if a TensorTypeId is in the set
  bool has(TensorTypeId t) const {
    TORCH_INTERNAL_ASSERT(t != TensorTypeId::UndefinedTensorId);
    return static_cast<bool>(repr_ & TensorTypeSet(t).repr_);
  }
  // Perform set union
  TensorTypeSet operator|(TensorTypeSet other) const {
    return TensorTypeSet(repr_ | other.repr_);
  }
  // Perform set intersection
  TensorTypeSet operator&(TensorTypeSet other) const {
    return TensorTypeSet(repr_ & other.repr_);
  }
  // Compute the set difference self - other
  TensorTypeSet operator-(TensorTypeSet other) const {
    return TensorTypeSet(repr_ & ~other.repr_);
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
  uint64_t raw_repr() { return repr_; }
  // Return the type id in this set with the highest priority (i.e.,
  // is the largest in the TensorTypeId enum).  Intuitively, this
  // type id is the one that should handle dispatch (assuming there
  // aren't any further exclusions or inclusions).
  TensorTypeId highestPriorityTypeId() const {
    // TODO: If I put UndefinedTensorId as entry 64 and then adjust the
    // singleton constructor to shift from the right, we can get rid of the
    // subtraction here.  It's modestly more complicated to get right so I
    // didn't do it for now.
    return static_cast<TensorTypeId>(64 - llvm::countLeadingZeros(repr_));
  }
private:
  TensorTypeSet(uint64_t repr) : repr_(repr) {}
  uint64_t repr_ = 0;
};

C10_API std::string toString(TensorTypeSet);
C10_API std::ostream& operator<<(std::ostream&, TensorTypeSet);

// Historically, every tensor only had a single TensorTypeId, and it was
// always something like CPUTensorId and not something weird like VariableId.
// For the forseeable future, it will still be possible to extract /that/
// TensorTypeId, and that's what this function does.  It should be used
// for legacy code that is still using TensorTypeId for things like instanceof
// checks; if at all possible, refactor the code to stop using TensorTypeId
// in those cases.
//
// What's the difference between 'legacyExtractTypeId(s) == id'
// and 's.has(id)'?  legacyExtractTypeId will NEVER return VariableTensorId;
// but s.has(VariableTensorId) will evaluate to true if s has VariableTensorId.
// For non-VariableTensorId equality tests, they are indistinguishable.
//
// NB: If you add other non-VariableTensorId other keys to this set, you'll
// have to adjust this some more (sorry.)
static inline TensorTypeId legacyExtractTypeId(TensorTypeSet s) {
  return s.remove(TensorTypeId::VariableTensorId).highestPriorityTypeId();
}

}
