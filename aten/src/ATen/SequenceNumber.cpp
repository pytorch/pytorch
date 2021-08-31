#include <ATen/SequenceNumber.h>

namespace at {
namespace sequence_number {

namespace {
thread_local uint64_t sequence_nr_ = 0;
} // namespace

uint64_t peek() {
  return sequence_nr_;
}

uint64_t get_and_increment() {
  return sequence_nr_++;
}

} // namespace sequence_number
} // namespace at
