#include <c10d_frontend.h>
#include "c10/util/Exception.h"

namespace c10d {

    
void DistributedC10d::initProcessGroup(
    const std::string &backend,
    const std::string &init_method,
    const std::chrono::milliseconds &timeout,
    int64_t world_size,
    int64_t rank,
    std::shared_ptr<Store> store,
    const std::string &group_name) {

    TORCH_CHECK(default_pg_ != nullptr,
                "trying to initialize the default process group "
                           "twice!");

    TORCH_CHECK((store == nullptr) || (init_method.empty()),
                "Cannot specify both init_method and store.");

    if (store != nullptr) {
        TORCH_CHECK(world_size > 0, "world_size must be positive if using store");
        TORCH_CHECK(rank >= 0, "rank must be non-negative if using store");
    } else if (init_method.empty()) {
        // TODO: to fill
    }

    // backend initialization
    std::string backend_str = Backend::get(backend);


    if (backend_str == "MPI") {
        if (world_size != -1 || rank != -1) {
            TORCH_WARN(
                "For MPI backend, world_size (", world_size,
                ") and rank (", rank, ") are ignored since"
                " they are assigned by the MPI runtime.");
        }

        // default_pg_ = 
    } else {
        // backward compatible API
        if (store == nullptr) {

        }
        // default_pg_ =

    }

    default_pg_init_method_ = init_method;

}


 void DistributedC10d::newProcessGroupHelper(const int64_t work_size,
                                             const int64_t rank,
                                             const std::unordered_map<std::shared_ptr<ProcessGroup>, std::vector<int64_t>>& pg_group_ranks_,
                                             const std::string& backend,
                                             std::shared_ptr<Store> store,
                                             c10::optional<std::string> group_name,
                                             std::chrono::milliseconds timeout) {

 }

void DistributedC10d::checkDefaultPg() const {
    TORCH_CHECK(default_pg_, "Default process group is not initialized");
}



}