#include "ProcessGroup.hpp"

namespace c10d {

ProcessGroup::Work::~Work() {
}

ProcessGroup::ProcessGroup(int rank, int size)
    : rank_(rank),
      size_(size) {
}

ProcessGroup::~ProcessGroup() {
}

} // namespace c10d
