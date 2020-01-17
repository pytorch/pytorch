#pragma once

#include<string>
#include<vector>
#include <typeinfo>
#include <cxxabi.h>
#include <cassert>

namespace Fuser {

template<typename T>
struct IO_struct {
  std::vector<int> shapes;
  std::vector<int> strides;
  T* data;
};

// TODO: totally not safe!
template<typename T>
std::string getTypeName() {
  const char *name = typeid(T).name();
  int status;
  char *undecorated_name = abi::__cxa_demangle(name, 0, 0, &status);
  assert(status==0);
  return undecorated_name;
}

std::string saxpy_codegen(std::string name);

}
