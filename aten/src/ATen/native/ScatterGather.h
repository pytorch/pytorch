#pragma once
#include <vector>

namespace {

inline void ensure_nonempty(std::vector<int64_t> &vec) {
  if(vec.size() == 0) {
    vec.push_back(1);
  }
}

}  // namespace
