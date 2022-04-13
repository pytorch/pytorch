#include <ATen/core/SymbolicIntNode.h>


namespace c10 {

    size_t SymIntTable::addNode(SymbolicIntNode* sin) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto index = nodes_.size();
        nodes_.push_back(sin);
        return index;
    }
    SymbolicIntNode* SymIntTable::getNode(size_t index) {
        TORCH_CHECK(index < nodes_.size());
        std::lock_guard<std::mutex> lock(mutex_);
        return nodes_[index];
    }

    c10::SymInt SymbolicIntNode::toSymInt() {
        //TODO: memoize this
        auto& sit = getSymIntTable();
        auto data = sit.addNode(this) | SYM_TAG_MASK;
        return c10::SymInt(data);
    }


    SymIntTable& getSymIntTable() {
        static SymIntTable sit;
        return sit;
    }
}




