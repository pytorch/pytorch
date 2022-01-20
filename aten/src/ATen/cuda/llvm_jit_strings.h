#pragma once

#include <string>

namespace at {
namespace cuda {

const std::string &get_traits_definition();
const std::string &get_cmath_definition();
const std::string &get_complex_definition();

}} // namespace at
