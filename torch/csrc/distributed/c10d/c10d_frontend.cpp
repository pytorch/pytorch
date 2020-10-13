#include "c10d_frontend.h"
#include <c10/util/Exception.h>
#include <c10d/PrefixStore.hpp>
#include <c10d/Utils.hpp>
#include <stdexcept>
#include <utility>

#ifdef USE_C10D_GLOO
#include <c10d/ProcessGroupGloo.hpp>
#endif

#ifdef USE_C10D_NCCL
#include <c10d/ProcessGroupNCCL.hpp>
#endif

#ifdef USE_C10D_MPI
#include <c10d/ProcessGroupMPI.hpp>
#endif

namespace c10d {

const std::string& DistributedC10d::backend() const {
  return backend_;
}
void DistributedC10d::backend(std::string const& backend_name) {
  backend_ = backend_name;
}

const std::unordered_map<std::shared_ptr<ProcessGroup>, std::vector<int64_t>>&
DistributedC10d::pg_group_ranks() {
  return pg_group_ranks_;
}
void DistributedC10d::pg_group_ranks(std::unordered_map<
                                     std::shared_ptr<ProcessGroup>,
                                     std::vector<int64_t>> const& new_ranks) {
  pg_group_ranks_ = new_ranks;
}

const std::string& DistributedC10d::default_pg_init_method() const {
  return default_pg_init_method_;
}
void DistributedC10d::default_pg_init_method(std::string const& init_method) {
  default_pg_init_method_ = init_method;
}

std::shared_ptr<ProcessGroup> DistributedC10d::newProcessGroupHelper(
    const int64_t world_size,
    const int64_t rank,
    const std::vector<int64_t>& group_ranks,
    const std::string& backend_str,
    const std::shared_ptr<Store>& store,
    c10::optional<std::string> group_name,
    std::chrono::milliseconds timeout) {
  if (!group_name.has_value()) {
    group_name = std::to_string(group_count_);
    ++group_count_;
  }

  auto it = std::find_if(
      pg_names_.begin(),
      pg_names_.end(),
      [&](const std::pair<std::shared_ptr<ProcessGroup>, std::string>&
              pg_name) { return pg_name.second == *group_name; });

  if (it == pg_names_.end()) {
    throw std::runtime_error(
        "The specified group name has already been "
        "created, please use a different group name");
  }

  bool is_default_group = pg_group_ranks_.size() == 0;

  std::shared_ptr<ProcessGroup> pg = nullptr;

  std::string backend = Backend::get(backend_str);
  if (backend == "mpi") {
#ifdef USE_C10D_MPI
    pg = ProcessGruopMPI::createProcessGroupMPI(group_ranks);
#else
    throw std::runtime_error(
        "Distributed package doesn't have MPI built in."
        " MPI is only included if you build PyTorch from"
        " source on a host that has MPI installed.");
#endif
  } else {
    if (!is_default_group) {
      int64_t global_rank = default_pg_->getRank();
      if (std::find(group_ranks.begin(), group_ranks.end(), global_rank) ==
          group_ranks.end()) {
        return nullptr;
      }
    }

    auto prefix_store = std::make_shared<PrefixStore>(*group_name, store);

    if (backend == "gloo") {
#ifdef USE_C10D_GLOO
      auto options = ProcessGroupGloo::Options();

      // Use interfaces listed in "GLOO_SOCKET_IFNAME", if set.
      char* ifnameEnv = getenv(GLOO_SOCKET_IFNAME_ENV);
      if (ifnameEnv) {
        for (const auto& iface : split(',', ifnameEnv)) {
          options.devices.push_back(
              ::c10d::ProcessGroupGloo::createDeviceForInterface(iface));
        }
      } else {
        // If no hostname is specified, this function looks up
        // the machine's hostname and returns a device instance
        // associated with the address that the hostname resolves to.
        options.devices.push_back(
            ::c10d::ProcessGroupGloo::createDefaultDevice());
      }

      options.timeout = timeout;
      options.threads = options.devices.size() * 2;

      pg = std::make_shared<ProcessGroupGloo>(
          prefix_store, rank, world_size, options);
#endif
    } else if (backend == "nccl") {
#ifdef USE_C10D_NCCL
      auto options = ProcessGroupNCCL::Options();

      options.isHighPriorityStream = false;
      options.opTimeout = timeout;
      pg = std::make_shared<ProcessGroupNCCL>(
          prefix_store, rank, world_size, options);
#endif
    } else {
      // TODO: discuss to figure out how to extend this to third party backends?
      pg = nullptr;
      return pg;
    }
  }

  // register to process group map
  pg_map_[pg] = std::make_pair(backend, store);
  pg_names_[pg] = *group_name;
  return pg;
}

void DistributedC10d::checkDefaultPg() const {
  TORCH_CHECK(default_pg_, "Default process group is not initialized");
}

} // namespace c10d