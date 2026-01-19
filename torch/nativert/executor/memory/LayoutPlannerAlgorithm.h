#pragma once

#include <cstddef>
#include <vector>

namespace torch::nativert {

// represents the inclusive lifetime of a tensor
// i.e., the buffer used by tensor x with lifetime [m, n]
// can only be safely used during intervals 0 --> m-1 and n+1 --> ...
//
// e.g.,
//
// g(x):           0
//  a = op_a(x)    1
//  b = op_b(a)    2
//  c = op_c(a)    3
//  return (b, c)  4
//
// gives:
//
//  lifetime(x) = 0 --> 1
//  lifetime(a) = 1 --> 3
//  lifetime(b) = 2 --> 4
//  lifetime(c) = 3 --> 4
//
// assuming no aliasing...
// however, if b aliases a we'd get
//
//  lifetime(x) = 0 --> 1
//  lifetime(a) = 1 --> *4* (max{l_end(a), l_end(b)})
//  lifetime(b) = 2 --> 4
//  lifetime(c) = 3 --> 4

struct AllocationLifetime {
  AllocationLifetime() = default;
  AllocationLifetime(size_t s, size_t e) : start(s), end(e) {}

  // two lifetime intervals are considered not overlapping
  // if their lifetimes are exclusive.
  // e.g.,
  //  l(a) = 0 --> 3
  //  overlaps with
  //  l(b) = 3 --> 5
  // since both tensors can exist at t = 3
  //
  // however, if l(b) = 4 --> 5
  // l(a) and l(b) do not overlap.
  bool not_overlapping_with(const AllocationLifetime& other) const {
    return this->end < other.start || this->start > other.end;
  }

  bool operator==(const AllocationLifetime& other) const {
    return this->start == other.start && this->end == other.end;
  }

  size_t start{0};
  size_t end{0};
};

struct AllocationSpec {
  AllocationLifetime lifetime;
  size_t size{0};

  bool not_overlapping_with(const AllocationSpec& other) const {
    return this->lifetime.not_overlapping_with(other.lifetime);
  }
};

struct Allocation {
  size_t size{0};
  size_t offset{0};
};

struct LayoutPlan {
  size_t total_size{0};
  // in practice, each allocation has an associated
  // allocation spec
  //
  // for example, given:
  //
  // allocation_specs = [s1, s2, s3]
  // plan = algorithm(allocation_specs)
  //
  // plan.allocations will be [a1, a2, a3]
  //                            ^   ^   ^
  // mapping back to          [s1, s2, s3]
  std::vector<Allocation> allocations;
};

// a layout planner algorithm is provided a vector of
// allocation specs, and returns a plan containing
// a vector of allocations (i.e., offset & size)
// whose order MUST correspond to that of the input
//
// specifically, provided:
// auto plan = algorithm(allocation_specs);
//
// allocation_specs.size() == plan.allocations.size()
//
// AND
//
// allocation_specs[0] --> plan.allocations[0]
// ...
// allocation_specs[i] --> plan.allocations[i]
// ...
// allocation_specs[allocation_specs.size() - 1] -->
// plan.allocations[plan.allocations.size() - 1]
using LayoutPlannerAlgorithm =
    LayoutPlan(const std::vector<AllocationSpec>& allocation_specs);

} // namespace torch::nativert
