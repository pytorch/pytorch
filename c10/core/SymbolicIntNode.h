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
  virtual SymbolicIntNode* add(SymbolicIntNode* other) { TORCH_CHECK(false, "NYI"); };
  virtual SymbolicIntNode* sub(SymbolicIntNode* other) { TORCH_CHECK(false, "NYI"); };
  virtual SymbolicIntNode* mul(SymbolicIntNode* other) { TORCH_CHECK(false, "NYI"); };
  virtual SymbolicIntNode* div(SymbolicIntNode* other) { TORCH_CHECK(false, "NYI"); };
  virtual SymbolicIntNode* mod(SymbolicIntNode* other) { TORCH_CHECK(false, "NYI"); };
  virtual SymbolicIntNode* eq(SymbolicIntNode* other) { TORCH_CHECK(false, "NYI"); };
  virtual SymbolicIntNode* gt(SymbolicIntNode* other) { TORCH_CHECK(false, "NYI"); };
  virtual SymbolicIntNode* lt(SymbolicIntNode* other) { TORCH_CHECK(false, "NYI"); };
  virtual SymbolicIntNode* wrap(int64_t num) { TORCH_CHECK(false, "NYI"); };
  virtual bool bool_() { TORCH_CHECK(false, "NYI"); };
  virtual std::string str() { TORCH_CHECK(false, "NYI"); };
  virtual std::ostream& operator<<(std::ostream& os) {
    os << str();
    return os;
  };
};

class C10_API SymIntTable {
 public:
  uint64_t addNode(std::shared_ptr<SymbolicIntNode> sin);
  std::shared_ptr<SymbolicIntNode> getNode(size_t index);
  void clear();

 private:
  std::vector<std::shared_ptr<SymbolicIntNode>> nodes_;
  std::mutex mutex_;
};

C10_API SymIntTable& getSymIntTable();

} // namespace c10
