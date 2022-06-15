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
  virtual std::ostream& operator<<(std::ostream& os) {
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
