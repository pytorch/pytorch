#pragma once
#include <ATen/Backend.h>
${extension_backend_headers}

namespace at {

namespace detail {

std::unordered_map<std::string, std::set<std::string>> api_name_to_schemas = {
  ${api_name_to_schemas_map_entries}
};

inline void check_schema_existence(const std::string & schema) {
  auto index = schema.find_first_of("(");
  if (index == std::string::npos) {
    AT_ERROR("Error parsing operator name from schema: ", schema);
  }
  auto api_name = schema.substr(0, index);

  auto it1 = api_name_to_schemas.find(api_name);
  if (it1 == api_name_to_schemas.end()) {
    AT_ERROR("Attempting to register nonexistent operator name: ", api_name);
  }

  auto it2 = it1->second.find(schema);
  if (it2 == it1->second.end()) {
    std::stringstream ss;
    for (auto & s : it1->second) {
      ss << s << "\n";
    }
    AT_ERROR("Attempting to register nonexistent schema: ", api_name,
        "\nValid schemas for given op:\n", ss.str());
  }
}

template <typename FnPtr>
inline void check_function_pointer_type(const std::string & schema);

${check_function_pointer_type_specializations}

} // namespace detail

template <typename FnPtr>
inline void register_extension_backend_op(
    Backend backend,
    const char * schema,
    FnPtr fn) {
      std::string schema_str(schema);
      detail::check_schema_existence(schema_str);
      detail::check_function_pointer_type<FnPtr>(schema_str);
  switch (backend) {
        ${extension_backend_register_switches}
        default:
          AT_ERROR("Invalid extension backend: ", toString(backend));
    }
}

} // namespace at
