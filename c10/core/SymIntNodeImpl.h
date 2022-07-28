#pragma once

#include <c10/core/SymInt.h>
#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>
#include <memory>
#include <mutex>
#include <vector>

namespace c10 {

class SymIntNodeImpl;
using SymIntNode = std::shared_ptr<SymIntNodeImpl>;

class C10_API SymIntNodeImpl
    : public std::enable_shared_from_this<SymIntNodeImpl> {
 public:
  c10::SymInt toSymInt();
  virtual ~SymIntNodeImpl(){};
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
