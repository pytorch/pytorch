#pragma once

#include <c10/util/Logging.h>

#include <ostream>

namespace lazy_tensors {

class Status {
 public:
  Status(const char*) {}

  static Status OK() {
    static const Status ok = Status(nullptr);
    return ok;
  }

  bool ok() const { return true; }
};

inline std::ostream& operator<<(std::ostream& os,
                                const Status& primitive_type) {
  LOG(FATAL) << "Not implemented yet.";
}

}  // namespace lazy_tensors
