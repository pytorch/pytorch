#include <c10/core/TensorTypeSet.h>

namespace c10 {

std::string toString(TensorTypeSet ts) {
  std::stringstream ss;
  ss << ts;
  return ss.str();
}

std::ostream& operator<<(std::ostream& os, TensorTypeSet ts) {
  if (ts.empty()) {
    os << "TensorTypeSet()";
    return os;
  }
  os << "TensorTypeSet(";
  TensorTypeId tid;
  bool first = true;
  while ((tid = ts.highestPriorityTypeId()) != TensorTypeId::UndefinedTensorId) {
    if (!first) {
      os << ", ";
    }
    os << tid;
    ts = ts.remove(tid);
    first = false;
  }
  os << ")";
  return os;
}

}
