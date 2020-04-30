#include <ATen/FunctionSequenceNumber.h>

namespace at {

namespace {
thread_local uint64_t sequence_nr_ = 0;
}

/* static */
uint64_t FunctionSequenceNumber::peek() {
  return sequence_nr_;
}

/* static */
uint64_t FunctionSequenceNumber::get_and_increment() {
  return sequence_nr_++;
}

} // namespace at
