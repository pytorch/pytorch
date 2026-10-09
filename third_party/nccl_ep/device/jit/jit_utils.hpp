/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 * See LICENSE.txt for more license information.
 */

#pragma once

#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

namespace nccl_ep {
namespace jit {

std::string env_value(const char* name);
bool env_flag_enabled(const char* name, bool default_value = false);
void jit_log(std::string_view message);

class ScopedFileLock {
public:
    ScopedFileLock() = default;
    ScopedFileLock(const ScopedFileLock&) = delete;
    ScopedFileLock& operator=(const ScopedFileLock&) = delete;
    ~ScopedFileLock();

    bool lock(const std::filesystem::path& path, std::string* error);

private:
    int fd_ = -1;
};

std::vector<std::string> split_env_flags(const char* flags);
std::string json_escape(std::string_view text);
std::string read_file_or_empty(const std::filesystem::path& path);
bool write_file_atomic(const std::filesystem::path& path, const std::string& data, bool binary);
std::string fnv1a_digest(const std::vector<std::string>& parts);

} // namespace jit
} // namespace nccl_ep
