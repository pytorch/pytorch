#pragma once

#include <c10/util/string_view.h>
#include <c10/util/StringUtil.h>
#include <c10/util/Optional.h>
#include <c10/core/Scalar.h>

#include <algorithm>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

namespace lazy_tensors {

// TODO - is is_scalar the right thing here?
// I mainly wanted to make sure it wasn't ambiguous for vector<T>
template <typename T, typename std::enable_if<
                          std::is_scalar<T>::value>::type* = nullptr>
void ToString(std::string name, T val, std::ostream& ss){
  ss << std::string(", ") << name << std::string("=(") << val << std::string(")");
}
void ToString(std::string name, const c10::Scalar& val, std::ostream& ss);

template <typename T>
void ToString(std::string name, std::vector<T> val, std::ostream& ss){
  ss << std::string(", ") << name << std::string("=(") << c10::Join(", ", val) << std::string(")");
}

template <typename T>
void ToString(std::string name, c10::optional<T> val, std::ostream& ss){
  if (val.has_value()){
    ToString(name, val.value(), ss);
  } else {
    ss << std::string(", ") << name << std::string("=(nullopt)");
  }
}

}  // namespace lazy_tensors
