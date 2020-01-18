#pragma once

#include<string>
#include<vector>
#include <typeinfo>
#include <cxxabi.h>
#include <cassert>

namespace Fuser {

#define STRINGIFY(x) x
#include "data_struct.h"
#undef STRINGIFY

// TODO: totally not safe!
template<typename T>
std::string getTypeName() {
  const char *name = typeid(T).name();
  int status;
  char *undecorated_name = abi::__cxa_demangle(name, 0, 0, &status);
  assert(status==0);
  return undecorated_name;
}

#define STRINGIFY(x) #x
static auto typeinfo = 
#include"data_struct.h"
;
#undef STRINGIFY

std::string saxpy_codegen(std::string name);

}
