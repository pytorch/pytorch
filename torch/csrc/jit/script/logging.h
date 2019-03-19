#pragma once

#include <string>
#include <unordered_map>

namespace torch { namespace jit { namespace logging {

void bumpCounter(std::string counter, float val);
std::unordered_map<std::string, float> getCounters();

}}}