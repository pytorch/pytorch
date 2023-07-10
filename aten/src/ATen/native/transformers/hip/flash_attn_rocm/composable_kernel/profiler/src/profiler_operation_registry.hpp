// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include <functional>
#include <iostream>
#include <iterator>
#include <map>
#include <optional>
#include <string_view>
#include <utility>

class ProfilerOperationRegistry final
{
    ProfilerOperationRegistry()  = default;
    ~ProfilerOperationRegistry() = default;

    public:
    using Operation = std::function<int(int, char*[])>;

    private:
    struct Entry final
    {
        explicit Entry(std::string_view description, Operation operation) noexcept
            : description_(description), operation_(std::move(operation))
        {
        }

        std::string_view description_;
        Operation operation_;
    };

    std::map<std::string_view, Entry> entries_;

    friend std::ostream& operator<<(std::ostream& stream, const ProfilerOperationRegistry& registry)
    {
        stream << "{\n";
        for(auto& [name, entry] : registry.entries_)
        {
            stream << "\t" << name << ": " << entry.description_ << "\n";
        }
        stream << "}";

        return stream;
    }

    public:
    static ProfilerOperationRegistry& GetInstance()
    {
        static ProfilerOperationRegistry registry;
        return registry;
    }

    std::optional<Operation> Get(std::string_view name) const
    {
        const auto found = entries_.find(name);
        if(found == end(entries_))
        {
            return std::nullopt;
        }

        return (found->second).operation_;
    }

    bool Add(std::string_view name, std::string_view description, Operation operation)
    {
        return entries_
            .emplace(std::piecewise_construct,
                     std::forward_as_tuple(name),
                     std::forward_as_tuple(description, std::move(operation)))
            .second;
    }
};

#define PP_CONCAT(x, y) PP_CONCAT_IMPL(x, y)
#define PP_CONCAT_IMPL(x, y) x##y

#define REGISTER_PROFILER_OPERATION(name, description, operation)              \
    static const bool PP_CONCAT(operation_registration_result_, __COUNTER__) = \
        ::ProfilerOperationRegistry::GetInstance().Add(name, description, operation)
