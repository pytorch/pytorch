#pragma once

#include <torch/lib/c10d/ProcessGroup.hpp>
#include <torch/lib/c10d/Store.hpp>
#include <ATen/ATen.h>
#include <c10/util/Optional.h>

#include <string>
#include <unordered_map>
#include <memory>
#include <chrono>

namespace c10d {

class Backend {
    public:
     // Maps to Backend.__new__ in Python.
     static std::string get(std::string);

     // TODO: How to support registering third_party backend?
     static void registerBackend();

    private:
     // TODO: Should this be an enum list instead since this set doesn't
     // change at all.
     std::unordered_set<std::string> registered_backends_;
};

class DistributedC10d{
    public:
     void initProcessGroup(
         const std::string& backend,
         const std::string& init_method,
         const std::chrono::milliseconds& timeout,
         int64_t world_size,
         int64_t rank,
         std::shared_ptr<Store> store,
         const std::string& group_name);

     void destroyProcessGroup(std::shared_ptr<ProcessGroup> group);
     int64_t getRank(std::shared_ptr<ProcessGroup> group);
     int64_t getWorldSize(std::shared_ptr<ProcessGroup> group);

     ProcessGroup::Work isend(at::Tensor tensor, int64_t dst, std::shared_ptr<ProcessGroup> group, c10::optional<int64_t> tag);
     ProcessGroup::Work irecv(at::Tensor tensor, int64_t src, std::shared_ptr<ProcessGroup> group, c10::optional<int64_t> tag);

    private:
     DistributedC10d(){};

     bool rankNotInGroup(std::shared_ptr<ProcessGroup> group) const;
     int64_t getGroupRank(
         std::shared_ptr<ProcessGroup> group,
         const int64_t rank) const;
     int64_t getGlobalRank(
         std::shared_ptr<ProcessGroup> group,
         const int64_t global_rank) const;
     void checkDefaultPg() const;
     int64_t getGroupSize(std::shared_ptr<ProcessGroup> group) const;
     int64_t getBackend(std::shared_ptr<ProcessGroup> group);

     std::string backend_;
     // TODO: Ask Alex what kind of equality we need. It determine whether we
     // need to use ProcessGroup or ProcesGroup* as key.
     std::unordered_map<
         std::shared_ptr<ProcessGroup>,
         std::pair<std::shared_ptr<Backend>, std::shared_ptr<Store>>>
         pg_map_;

     // Note, this is different mapping relationship than original Python
     // implementation.
     std::unordered_map<std::shared_ptr<ProcessGroup>, std::string> pg_names_;

     // Value is global_rank:group_rank mapping.
     std::unordered_map<std::shared_ptr<ProcessGroup>, std::vector<int64_t>>
         pg_group_ranks_;

     std::shared_ptr<ProcessGroup> default_pg_;

     // Default value should be "env://"
     std::string default_pg_init_method_;

     int64_t group_count_;
};


} // namespace c10d
