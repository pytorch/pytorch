#pragma once

#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>
#include <vector>
#include <ATen/core/SymInt.h>
#include <mutex>

namespace c10 {

class TORCH_API SymbolicIntNode {
    public:
        c10::SymInt toSymInt();
        static bool is_symbolic(int64_t data) {
            return SYM_TAG_MASK & data;
        }

        const static int64_t SYM_TAG_MASK = 1LL << 63;

        virtual ~SymbolicIntNode() {};

        virtual SymbolicIntNode* add(SymbolicIntNode* other) = 0;
    private:
        int64_t data_;
        
};

class TORCH_API SymIntTable {

public:
    
    size_t addNode(SymbolicIntNode* sin);
    SymbolicIntNode* getNode(size_t index);

    ~SymIntTable() {
        for (auto sit: nodes_) {
            free(sit);
        }
    }

private:
    std::vector<SymbolicIntNode*> nodes_;
    std::mutex mutex_;
};

TORCH_API SymIntTable& getSymIntTable();

//TORCH_API std::ostream& operator<<(std::ostream& os, SymbolicIntNode s);
}


