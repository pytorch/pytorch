
#include <ATen/core/SymInt.h>
#include <ATen/core/SymbolicIntNode.h>

namespace c10 {

    bool SymInt::is_symbolic() const {
        return SymbolicIntNode::is_symbolic(data_);
    }

    c10::SymbolicIntNode* SymInt::toSymbolicIntNode() {
        auto& st = getSymIntTable();
        TORCH_CHECK(is_symbolic());
        return st.getNode(SymbolicIntNode::SYM_TAG_MASK ^ data_);
    }

}
