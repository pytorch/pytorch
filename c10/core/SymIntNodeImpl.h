#pragma once

#include <c10/core/SymInt.h>
#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>
#include <c10/util/intrusive_ptr.h>
#include <memory>
#include <mutex>
#include <vector>

namespace c10 {

class SymIntNodeImpl;
using SymIntNode = c10::intrusive_ptr<SymIntNodeImpl>;

class C10_API SymIntNodeImpl : public c10::intrusive_ptr_target {
 public:
  c10::SymInt toSymInt();
  virtual ~SymIntNodeImpl(){};

  template <typename T>
  c10::intrusive_ptr<T> dyn_cast() const {
    return c10::intrusive_ptr<T>::reclaim_copy(dynamic_cast<T*>(this));
  }

  // these could be pure virtual when we implement LTC versions
  virtual SymIntNode add(const SymIntNode& other) {
    TORCH_CHECK(false, "NYI");
  };
  virtual SymIntNode sub(const SymIntNode& other) {
    TORCH_CHECK(false, "NYI");
  };
  virtual SymIntNode mul(const SymIntNode& other) {
    TORCH_CHECK(false, "NYI");
  };
  virtual SymIntNode div(const SymIntNode& other) {
    TORCH_CHECK(false, "NYI");
  };
  virtual SymIntNode mod(const SymIntNode& other) {
    TORCH_CHECK(false, "NYI");
  };
  virtual SymIntNode eq(const SymIntNode& other) {
    TORCH_CHECK(false, "NYI");
  };
  virtual SymIntNode ne(const SymIntNode& other) {
    TORCH_CHECK(false, "NYI");
  };
  virtual SymIntNode gt(const SymIntNode& other) {
    TORCH_CHECK(false, "NYI");
  };
  virtual SymIntNode lt(const SymIntNode& other) {
    TORCH_CHECK(false, "NYI");
  };
  virtual SymIntNode le(const SymIntNode& other) {
    TORCH_CHECK(false, "NYI");
  };
  virtual SymIntNode ge(const SymIntNode& other) {
    TORCH_CHECK(false, "NYI");
  };
  virtual SymIntNode wrap(int64_t num) {
    TORCH_CHECK(false, "NYI");
  };
  virtual bool bool_() {
    TORCH_CHECK(false, "NYI");
  };
  virtual int64_t int_() {
    TORCH_CHECK(false, "NYI");
  }
  virtual std::string str() {
    TORCH_CHECK(false, "NYI");
  };
  std::ostream& operator<<(std::ostream& os) {
    os << str();
    return os;
  };
};

class C10_API SymIntTable {
 public:
  uint64_t addNode(SymIntNode sin);
  SymIntNode getNode(size_t index);

 private:
  std::vector<SymIntNode> nodes_;
  std::mutex mutex_;
};

C10_API SymIntTable& getSymIntTable();

} // namespace c10
