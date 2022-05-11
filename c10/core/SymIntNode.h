#pragma once

#include <c10/core/SymInt.h>
#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>
#include <mutex>
#include <vector>

namespace c10 {

class C10_API SymIntNode : public std::enable_shared_from_this<SymIntNode> {
 public:
  c10::SymInt toSymInt();
  virtual ~SymIntNode(){};
  virtual std::ostream& operator<<(std::ostream& os) {
    return os;
  };
};

class C10_API SymIntTable {
 public:
  int64_t addNode(std::shared_ptr<SymIntNode> sin);
  std::shared_ptr<SymIntNode> getNode(size_t index);

 private:
  std::vector<std::shared_ptr<SymIntNode>> nodes_;
  std::mutex mutex_;
};

C10_API SymIntTable& getSymIntTable();

} // namespace c10
