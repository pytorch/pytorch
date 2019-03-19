#include "torch/csrc/jit/script/logging.h"

#include <mutex>
#include <unordered_map>

namespace torch { namespace jit { namespace logging {

std::mutex m;
std::unordered_map<std::string, float> counters;

void bumpCounter(std::string counter, float val) {
  std::unique_lock<std::mutex> lk(m);
  counters[counter] += val;
}

std::unordered_map<std::string, float> getCounters() {
  std::unique_lock<std::mutex> lk(m);
  return counters;
}

}}}