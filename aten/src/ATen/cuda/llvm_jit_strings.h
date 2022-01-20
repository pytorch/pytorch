#pragma once

#include <string>

namespace at {
namespace cuda {

const std::string &get_traits_string();
const std::string &get_cmath_string();
const std::string &get_complex_string();

}} // namespace at
