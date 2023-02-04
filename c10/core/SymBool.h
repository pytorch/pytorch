#pragma once

#include <c10/core/SymNodeImpl.h>
#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>
#include <c10/util/intrusive_ptr.h>

namespace c10 {

class C10_API SymBool {
 public:
  /*implicit*/ SymBool(bool b) : data_(b){};
  SymBool(SymNode ptr) : data_(false), ptr_(std::move(ptr)) {
    TORCH_CHECK(ptr_->is_bool());
  };
  SymBool() : data_(false) {}

  SymNodeImpl* toSymNodeImplUnowned() const {
    return ptr_.get();
  }

  SymNodeImpl* release() && {
    return std::move(ptr_).release();
  }

  SymNode toSymNodeImpl() const;

  bool expect_bool() const {
    TORCH_CHECK(!is_symbolic());
    return data_;
  }

  SymBool sym_and(const SymBool&) const;
  SymBool sym_or(const SymBool&) const;
  SymBool sym_not() const;

  SymBool operator&(const SymBool& other) const {
    return sym_and(other);
  }
  SymBool operator|(const SymBool& other) const {
    return sym_or(other);
  }
  SymBool operator~() const {
    return sym_not();
  }

  // Insert a guard for the bool to be its concrete value, and then return
  // that value.  Note that C++ comparison operations default to returning
  // bool, so it's not so common to have to call this
  bool guard_bool(const char* file, int64_t line) const;

  C10_ALWAYS_INLINE bool is_symbolic() const {
    return ptr_;
  }

  bool as_bool_unchecked() const {
    return data_;
  }

 private:
  // TODO: optimize to union
  bool data_;
  SymNode ptr_;
};

C10_API std::ostream& operator<<(std::ostream& os, const SymBool& s);
} // namespace c10
