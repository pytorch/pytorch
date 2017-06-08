#pragma once


#include "TensorLib/Type.h"

namespace tlib {

static inline std::ostream& operator<<(std::ostream & out, IntList list) {
  int i = 0;
  out << "[";
  for(auto e : list) {
    if (i++ > 0)
      out << ", ";
    out << e;
  }
  out << "]";
  return out;
}

}
