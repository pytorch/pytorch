#include "lazy_tensors/str_join.h"

namespace lazy_tensors {


void ToString(std::string name, const c10::Scalar& val, std::ostream& ss){
  ss << ", (TODO implement ToString for ScalarType)";
}


}  // namespace lazy_tensors
