#pragma once

#include <c10/core/SymInt.h>
#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>
#include <mutex>
#include <vector>

namespace c10 {

class C10_API SymIntNodeImpl
    : public std::enable_shared_from_this<SymIntNodeImpl> {
 public:
  c10::SymInt toSymInt();
  virtual ~SymIntNodeImpl(){};
  // these could be pure virtual when we implement LTC versions
  virtual std::shared_ptr<SymIntNodeImpl> add(
      const std::shared_ptr<SymIntNodeImpl>& other) {
    TORCH_CHECK(false, "NYI");
  };
  virtual std::shared_ptr<SymIntNodeImpl> sub(
      const std::shared_ptr<SymIntNodeImpl>& other) {
    TORCH_CHECK(false, "NYI");
  };
  virtual std::shared_ptr<SymIntNodeImpl> mul(
      const std::shared_ptr<SymIntNodeImpl>& other) {
    TORCH_CHECK(false, "NYI");
  };
  virtual std::shared_ptr<SymIntNodeImpl> div(
      const std::shared_ptr<SymIntNodeImpl>& other) {
    TORCH_CHECK(false, "NYI");
  };
  virtual std::shared_ptr<SymIntNodeImpl> mod(
      const std::shared_ptr<SymIntNodeImpl>& other) {
    TORCH_CHECK(false, "NYI");
  };
  virtual std::shared_ptr<SymIntNodeImpl> eq(
      const std::shared_ptr<SymIntNodeImpl>& other) {
    TORCH_CHECK(false, "NYI");
  };
  virtual std::shared_ptr<SymIntNodeImpl> ne(
      const std::shared_ptr<SymIntNodeImpl>& other) {
    TORCH_CHECK(false, "NYI");
  };
  virtual std::shared_ptr<SymIntNodeImpl> gt(
      const std::shared_ptr<SymIntNodeImpl>& other) {
    TORCH_CHECK(false, "NYI");
  };
  virtual std::shared_ptr<SymIntNodeImpl> lt(
      const std::shared_ptr<SymIntNodeImpl>& other) {
    TORCH_CHECK(false, "NYI");
  };
  virtual std::shared_ptr<SymIntNodeImpl> le(
      const std::shared_ptr<SymIntNodeImpl>& other) {
    TORCH_CHECK(false, "NYI");
  };
  virtual std::shared_ptr<SymIntNodeImpl> ge(
      const std::shared_ptr<SymIntNodeImpl>& other) {
    TORCH_CHECK(false, "NYI");
  };
  virtual std::shared_ptr<SymIntNodeImpl> wrap(int64_t num) {
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
  uint64_t addNode(std::shared_ptr<SymIntNodeImpl> sin);
  std::shared_ptr<SymIntNodeImpl> getNode(size_t index);

 private:
  std::vector<std::shared_ptr<SymIntNodeImpl>> nodes_;
  std::mutex mutex_;
};

C10_API SymIntTable& getSymIntTable();

} // namespace c10
