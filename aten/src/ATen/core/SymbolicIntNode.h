#pragma once

#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>
#include <vector>
#include <ATen/core/SymInt.h>
#include <mutex>

namespace c10 {

class TORCH_API SymbolicIntNode: public std::enable_shared_from_this<SymbolicIntNode>  {
    public:
        c10::SymInt toSymInt();
        virtual ~SymbolicIntNode() {};
        virtual std::ostream& operator<<(std::ostream& os) { return os; };
};

class TORCH_API SymIntTable {

public:

    int64_t addNode(std::shared_ptr<SymbolicIntNode> sin);
    std::shared_ptr<SymbolicIntNode> getNode(size_t index);

private:
    std::vector<std::shared_ptr<SymbolicIntNode>> nodes_;
    std::mutex mutex_;
};

TORCH_API SymIntTable& getSymIntTable();

}
