#pragma once

#include <ATen/core/ivalue.h>
#include <unordered_map>

namespace c10 {
static std::unordered_map<std::string, c10::StrongTypePtr> tmap;
}