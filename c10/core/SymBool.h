#pragma once

#include <c10/core/SymNodeImpl.h>
#include <c10/macros/Export.h>
#include <c10/util/Exception.h>
#include <c10/util/intrusive_ptr.h>
#include <cstdint>
#include <optional>
#include <ostream>
#include <utility>

namespace c10 {

class C10_API SymBool {
 public:
  /*implicit*/ SymBool(bool b) : data_(b) {}
  SymBool(SymNode ptr) : data_(false), ptr_(std::move(ptr)) {
    TORCH_CHECK(ptr_->is_bool());
  }
  SymBool() : data_(false) {}

  SymNodeImpl* toSymNodeImplUnowned() const {
    return ptr_.get();
  }

  SymNodeImpl* release() && {
    return std::move(ptr_).release();
  }

  // Only valid if is_heap_allocated()
  SymNode toSymNodeImpl() const;

  // Guaranteed to return a SymNode, wrapping using base if necessary
  SymNode wrap_node(const SymNode& base) const;

  bool expect_bool() const {
    std::optional<bool> c = maybe_as_bool();
    TORCH_CHECK(c.has_value());
    return *c;
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
  bool expect_true(const char* file, int64_t line) const;
  bool guard_size_oblivious(const char* file, int64_t line) const;

  bool has_hint() const;

  bool as_bool_unchecked() const {
    return data_;
  }

  std::optional<bool> maybe_as_bool() const {
    if (!is_heap_allocated()) {
      return data_;
    }
    return toSymNodeImplUnowned()->constant_bool();
  }

  bool is_heap_allocated() const {
    return ptr_;
  }

 private:
  // TODO: optimize to union
  bool data_;
  SymNode ptr_;
};

C10_API std::ostream& operator<<(std::ostream& os, const SymBool& s);

#define TORCH_SYM_CHECK(cond, ...) \
  TORCH_CHECK((cond).expect_true(__FILE__, __LINE__), __VA_ARGS__)
#define TORCH_SYM_INTERNAL_ASSERT(cond, ...) \
  TORCH_INTERNAL_ASSERT((cond).expect_true(__FILE__, __LINE__), __VA_ARGS__)

inline bool guard_size_oblivious(
    bool b,
    const char* file [[maybe_unused]],
    int64_t line [[maybe_unused]]) {
  return b;
}

inline bool guard_size_oblivious(
    const c10::SymBool& b,
    const char* file,
    int64_t line) {
  return b.guard_size_oblivious(file, line);
}

#define TORCH_GUARD_SIZE_OBLIVIOUS(cond) \
  c10::guard_size_oblivious((cond), __FILE__, __LINE__)

} // namespace c10
