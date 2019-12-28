#include "types.h"

namespace torch {
namespace jit {
namespace fuser {

  std::ostream& operator<<(std::ostream& os, const DType& dtype){

  switch(dtype.ctype()){
    case(CType::kInt32):
    os<<"int32";
    break;
    case(CType::kFloat32):
    os<<"float32";
    break;
    case(CType::kStatement):
    break;
    case(CType::kNull):
    break;
    }
    return os;
  }


bool is_scalar(const CType& type){
  if(type<CType::kStatement)
    return true;
  return false;
}

CType promote(const CType& t1, const CType& t2){
  assert(
    (t1 < CType::kStatement && t2 < CType::kStatement) ||
    (t1 > CType::kStatement && t2 > CType::kStatement)
  );
  return(t1 < t2 ? t1 : t2);
}

bool is_scalar(const DType& type){
  return is_scalar(type.ctype());
}

DType promote(const DType& t1, const DType& t2){
  assert(t1.lanes() == t2.lanes());
  return DType(promote(t1.ctype(), t2.ctype()), t1.lanes());
}


} // namespace fuser
} // namespace jit
} // namespace torch
