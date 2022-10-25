#pragma once

#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>
#include <c10/util/intrusive_ptr.h>
#include <memory>
#include <mutex>
#include <vector>

namespace c10 {

class SymIntNodeImpl;
using SymIntNode = c10::intrusive_ptr<SymIntNodeImpl>;

class SymFloat;
class SymFloatNodeImpl;
using SymFloatNode = c10::intrusive_ptr<SymFloatNodeImpl>;

class C10_API SymFloatNodeImpl : public c10::intrusive_ptr_target {
 public:
  c10::SymFloat toSymFloat();
  virtual ~SymFloatNodeImpl(){};

  template <typename T>
  c10::intrusive_ptr<T> dyn_cast() const {
    return c10::intrusive_ptr<T>::reclaim_copy(dynamic_cast<T*>(this));
  }

  virtual SymFloatNode wrap(double num) {
    TORCH_CHECK(false, "NYI");
  };
  virtual SymIntNode wrap(int64_t num) {
    TORCH_CHECK(false, "NYI");
  };
  virtual SymFloatNode add(const SymFloatNode& other) {
    TORCH_CHECK(false, "NYI");
  }
  virtual SymFloatNode add(const SymIntNode& other) {
    TORCH_CHECK(false, "NYI");
  }
  virtual SymFloatNode sub(const SymFloatNode& other) {
    TORCH_CHECK(false, "NYI");
  }
  virtual SymFloatNode sub(const SymIntNode& other) {
    TORCH_CHECK(false, "NYI");
  }
  virtual SymFloatNode mul(const SymFloatNode& other) {
    TORCH_CHECK(false, "NYI");
  }
  virtual SymFloatNode mul(const SymIntNode& other) {
    TORCH_CHECK(false, "NYI");
  }
  virtual SymFloatNode truediv(const SymFloatNode& other) {
    TORCH_CHECK(false, "NYI");
  }
  virtual SymFloatNode truediv(const SymIntNode& other) {
    TORCH_CHECK(false, "NYI");
  }
  virtual SymIntNode floordiv(const SymFloatNode& other) {
    TORCH_CHECK(false, "NYI");
  };
  virtual SymIntNode floordiv(const SymIntNode& other) {
    TORCH_CHECK(false, "NYI");
  };
  virtual SymFloatNode pow(const SymFloatNode& other) {
    TORCH_CHECK(false, "NYI");
  }
  virtual SymFloatNode pow(const SymIntNode& other) {
    TORCH_CHECK(false, "NYI");
  }
  virtual SymFloatNode eq(const SymFloatNode& other) {
    TORCH_CHECK(false, "NYI");
  };
  virtual SymFloatNode eq(const SymIntNode& other) {
    TORCH_CHECK(false, "NYI");
  };
  virtual SymFloatNode ne(const SymFloatNode& other) {
    TORCH_CHECK(false, "NYI");
  };
  virtual SymFloatNode ne(const SymIntNode& other) {
    TORCH_CHECK(false, "NYI");
  };
  virtual SymFloatNode gt(const SymFloatNode& other) {
    TORCH_CHECK(false, "NYI");
  };
  virtual SymFloatNode gt(const SymIntNode& other) {
    TORCH_CHECK(false, "NYI");
  };
  virtual SymFloatNode lt(const SymFloatNode& other) {
    TORCH_CHECK(false, "NYI");
  };
  virtual SymFloatNode lt(const SymIntNode& other) {
    TORCH_CHECK(false, "NYI");
  };
  virtual SymFloatNode le(const SymFloatNode& other) {
    TORCH_CHECK(false, "NYI");
  };
  virtual SymFloatNode le(const SymIntNode& other) {
    TORCH_CHECK(false, "NYI");
  };
  virtual SymFloatNode ge(const SymFloatNode& other) {
    TORCH_CHECK(false, "NYI");
  };
  virtual SymFloatNode ge(const SymIntNode& other) {
    TORCH_CHECK(false, "NYI");
  };
  virtual SymFloatNode min(const SymFloatNode& other) {
    TORCH_CHECK(false, "NYI");
  };
  virtual SymFloatNode min(const SymIntNode& other) {
    TORCH_CHECK(false, "NYI");
  };
  virtual SymFloatNode max(const SymFloatNode& other) {
    TORCH_CHECK(false, "NYI");
  };
  virtual SymFloatNode max(const SymIntNode& other) {
    TORCH_CHECK(false, "NYI");
  };
  virtual SymFloatNode mod(const SymFloatNode& other) {
    TORCH_CHECK(false, "NYI");
  };
  virtual SymFloatNode mod(const SymIntNode& other) {
    TORCH_CHECK(false, "NYI");
  };
  virtual SymIntNode ceil();
  virtual SymIntNode floor();
  virtual SymFloatNode neg() {
    TORCH_CHECK(false, "NYI");
  };
  virtual SymIntNode sym_int() {
    TORCH_CHECK(false, "NYI");
  };
  virtual bool bool_() {
    TORCH_CHECK(false, "NYI");
  };
  virtual std::string str() {
    TORCH_CHECK(false, "NYI");
  };
  std::ostream& operator<<(std::ostream& os) {
    os << str();
    return os;
  };
  virtual double guard_float(const char* file, int64_t line) {
    TORCH_CHECK(false, "NYI");
  };
};

} // namespace c10
