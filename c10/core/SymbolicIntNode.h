#pragma once

#include <c10/core/SymInt.h>
#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>
#include <mutex>
#include <vector>

namespace c10 {

class C10_API SymbolicIntNode
    : public std::enable_shared_from_this<SymbolicIntNode> {
 public:
  c10::SymInt toSymInt();
  virtual ~SymbolicIntNode(){};
  // these could be pure virtual when we implement LTC versions
  virtual std::shared_ptr<SymbolicIntNode> add(
      const std::shared_ptr<SymbolicIntNode>& other) {
    TORCH_CHECK(false, "NYI");
  };
  virtual std::shared_ptr<SymbolicIntNode> sub(
      const std::shared_ptr<SymbolicIntNode>& other) {
    TORCH_CHECK(false, "NYI");
  };
  virtual std::shared_ptr<SymbolicIntNode> mul(
      const std::shared_ptr<SymbolicIntNode>& other) {
    TORCH_CHECK(false, "NYI");
  };
  virtual std::shared_ptr<SymbolicIntNode> div(
      const std::shared_ptr<SymbolicIntNode>& other) {
    TORCH_CHECK(false, "NYI");
  };
  virtual std::shared_ptr<SymbolicIntNode> mod(
      const std::shared_ptr<SymbolicIntNode>& other) {
    TORCH_CHECK(false, "NYI");
  };
  virtual std::shared_ptr<SymbolicIntNode> eq(
      const std::shared_ptr<SymbolicIntNode>& other) {
    TORCH_CHECK(false, "NYI");
  };
  virtual std::shared_ptr<SymbolicIntNode> gt(
      const std::shared_ptr<SymbolicIntNode>& other) {
    TORCH_CHECK(false, "NYI");
  };
  virtual std::shared_ptr<SymbolicIntNode> lt(
      const std::shared_ptr<SymbolicIntNode>& other) {
    TORCH_CHECK(false, "NYI");
  };
  virtual std::shared_ptr<SymbolicIntNode> wrap(int64_t num) {
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
  uint64_t addNode(std::shared_ptr<SymbolicIntNode> sin);
  std::shared_ptr<SymbolicIntNode> getNode(size_t index);

 private:
  std::vector<std::shared_ptr<SymbolicIntNode>> nodes_;
  std::mutex mutex_;
};

C10_API SymIntTable& getSymIntTable();

} // namespace c10
