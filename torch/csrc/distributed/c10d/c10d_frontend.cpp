
#include <torch/csrc/distributed/c10d/c10d_frontend.h>

namespace c10d {

const std::string& DistributedC10d::backend() const {
  return backend_;
}

void DistributedC10d::set_backend(std::string const& backend_name) {
  backend_ = backend_name;
}

const std::unordered_map<std::shared_ptr<ProcessGroup>, std::vector<int64_t>>&
DistributedC10d::pg_group_ranks() {
  return pg_group_ranks_;
}

void DistributedC10d::set_pg_group_ranks(
    std::unordered_map<
        std::shared_ptr<ProcessGroup>,
        std::vector<int64_t>> const& new_ranks) {
  pg_group_ranks_ = new_ranks;
}

const std::string& DistributedC10d::default_pg_init_method() const {
  return default_pg_init_method_;
}

void DistributedC10d::set_default_pg_init_method(
    std::string const& init_method) {
  default_pg_init_method_ = init_method;
}

} // namespace c10d
