#include "lazy_tensors/str_join.h"

namespace lazy_tensors {


void ToString(std::string name, const c10::Scalar& val, std::ostream& ss){
  ss << ", (TODO implement ToString for ScalarType)";
}

void ToString(std::string name, const c10::string_view& val, std::ostream& ss) {
  ss << val;
}

}  // namespace lazy_tensors
