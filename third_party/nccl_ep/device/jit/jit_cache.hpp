/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 * See LICENSE.txt for more license information.
 */

#pragma once

#include <filesystem>
#include <mutex>
#include <string>
#include <unordered_set>

namespace nccl_ep {
namespace jit {

class JitCache {
public:
    static JitCache& instance();

    std::filesystem::path root_dir() const;
    bool has_failed(const std::string& key);
    void mark_failed(const std::string& key);
    void warn_once(const std::string& key, const std::string& message);

private:
    JitCache() = default;

    std::mutex mutex_;
    std::unordered_set<std::string> failed_keys_;
    std::unordered_set<std::string> warned_keys_;
};

} // namespace jit
} // namespace nccl_ep
