#pragma once

#include <cuda.h>
#include <c10/util/flat_hash_map.h>

// there is no name for BasicTypes that sit within Arrays. This is a
// mistake on my part.
struct BasicType {
  std::string type_name;
  size_t offset = 0;
  size_t size;
  bool is_pointer;
};

struct ArrayType;

// StructType has no copy constructor because of the presence of unique_ptr
struct StructType {
  std::string type_name;
  // std::unique_ptr<std::vector<std::variant<BasicType, StructType, ArrayType>>> members;
  std::vector<std::pair<std::string, std::variant<BasicType, StructType, ArrayType>>> members;
  size_t offset = 0;
  // we need a size field because structs can have tail padding.
  size_t size;
};

struct ArrayType {
  std::string type_name;
  // dwarf debug information will never allow an array to be nested
  // within an array. Instead, you will just have multiple
  // "subranges".
  size_t offset = 0;
  std::variant<BasicType, StructType> element_type;
  // std::unique_ptr<std::variant<BasicType, StructType, ArrayType>> element_type;
  size_t num_elements;
};

// in this case, the first element of each pair in members refers to
// the name of the argument. "type_name" has no meaning. "size" also
// has no meaning.
using ArgumentInformation = StructType;

inline auto conversion_visitor = [](auto&& value) {
  using T = std::decay_t<decltype(value)>;
  return std::variant<BasicType, StructType, ArrayType>{
    std::in_place_type<T>,
    std::forward<decltype(value)>(value) };
};

ArgumentInformation
getArgumentInformation(const char* linkageName, const std::string& elfPath);

ArgumentInformation
getArgumentInformation(const char* linkageName, void *buffer, size_t buffer_size);

std::unordered_map<std::string, ArgumentInformation>
get_argument_information(const std::vector<std::string> &function_names);

ArgumentInformation
get_argument_information(CUfunction func);

bool is_equal(void *arg1, void *arg2, std::variant<BasicType, StructType, ArrayType> info);

void prettyPrintArgumentInfo(const ArgumentInformation& args);
