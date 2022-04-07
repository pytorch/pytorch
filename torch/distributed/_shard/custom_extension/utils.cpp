#include <ATen.h>

namespace XLA {
    bool is_sharded(at::Tensor obj) {
        // maybe to access python object field and see if it's sharded
        // or register a new dispatch key for sharded tensor?
        return PyObject_HasAttr(obj, "local_shards")
    }

    at::Tensor add(at::Tensor lhs, at::Tensor rhs) {
        if (is_sharded(lhs) || is_sharded(rhs)) {
            XLA::SPMDPartition(lhs, rhs, ...)
        }
        // regular XLA::add op, create HLO graph?

    }
}
