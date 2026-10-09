/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 * See LICENSE.txt for more license information.
 */

#include "device/jit/jit_utils.hpp"

#include <cctype>
#include <cerrno>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <fcntl.h>
#include <iterator>
#include <sstream>
#include <sys/file.h>
#include <thread>
#include <unistd.h>

namespace nccl_ep {
namespace jit {
std::string env_value(const char* name) {
    const char* value = std::getenv(name);
    return (value != nullptr && value[0] != '\0') ? value : std::string();
}

bool env_flag_enabled(const char* name, bool default_value) {
    std::string value = env_value(name);
    if (value.empty()) return default_value;
    return value[0] != '0' &&
           value != "false" &&
           value != "FALSE" &&
           value != "off" &&
           value != "OFF";
}

void jit_log(std::string_view message) {
    if (!env_flag_enabled("NCCL_EP_JIT_LOG")) return;
    std::fprintf(
        stderr,
        "[nccl_ep jit] pid=%ld %.*s\n",
        static_cast<long>(getpid()),
        static_cast<int>(message.size()),
        message.data());
}

ScopedFileLock::~ScopedFileLock() {
    if (fd_ >= 0) {
        flock(fd_, LOCK_UN);
        close(fd_);
    }
}

bool ScopedFileLock::lock(const std::filesystem::path& path, std::string* error) {
    std::error_code ec;
    std::filesystem::create_directories(path.parent_path(), ec);
    if (ec) {
        if (error != nullptr) *error = "mkdir " + path.parent_path().string() + ": " + ec.message();
        return false;
    }

    fd_ = open(path.c_str(), O_CREAT | O_RDWR | O_CLOEXEC, 0666);
    if (fd_ < 0) {
        if (error != nullptr) *error = "open " + path.string() + ": " + std::strerror(errno);
        return false;
    }
    if (flock(fd_, LOCK_EX) != 0) {
        if (error != nullptr) *error = "flock " + path.string() + ": " + std::strerror(errno);
        close(fd_);
        fd_ = -1;
        return false;
    }
    return true;
}

std::vector<std::string> split_env_flags(const char* flags) {
    std::vector<std::string> out;
    if (flags == nullptr || flags[0] == '\0') return out;

    std::string current;
    char quote = '\0';
    bool escape = false;
    for (const char* p = flags; *p != '\0'; ++p) {
        const char c = *p;
        if (escape) {
            current.push_back(c);
            escape = false;
            continue;
        }
        if (c == '\\') {
            escape = true;
            continue;
        }
        if (quote != '\0') {
            if (c == quote) quote = '\0';
            else current.push_back(c);
            continue;
        }
        if (c == '\'' || c == '"') {
            quote = c;
            continue;
        }
        if (std::isspace(static_cast<unsigned char>(c))) {
            if (!current.empty()) {
                out.push_back(current);
                current.clear();
            }
            continue;
        }
        current.push_back(c);
    }
    if (escape) current.push_back('\\');
    if (!current.empty()) out.push_back(current);
    return out;
}

std::string json_escape(std::string_view text) {
    std::ostringstream out;
    for (char c : text) {
        switch (c) {
        case '\\': out << "\\\\"; break;
        case '"': out << "\\\""; break;
        case '\n': out << "\\n"; break;
        case '\r': out << "\\r"; break;
        case '\t': out << "\\t"; break;
        default: out << c; break;
        }
    }
    return out.str();
}

std::string read_file_or_empty(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return {};
    return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

bool write_file_atomic(const std::filesystem::path& path, const std::string& data, bool binary) {
    std::error_code ec;
    std::filesystem::create_directories(path.parent_path(), ec);
    if (ec) {
        jit_log("cache write mkdir failed for " + path.parent_path().string() + ": " + ec.message());
        return false;
    }

    std::ostringstream tmp_name;
    tmp_name << path.string()
             << ".tmp." << static_cast<long>(getpid())
             << "." << std::hash<std::thread::id>{}(std::this_thread::get_id());
    const std::filesystem::path tmp_path = tmp_name.str();
    const auto mode = binary ? (std::ios::binary | std::ios::trunc) : std::ios::trunc;
    {
        std::ofstream out(tmp_path, mode);
        if (!out) {
            jit_log("cache write open failed for " + tmp_path.string());
            return false;
        }
        out.write(data.data(), static_cast<std::streamsize>(data.size()));
        if (!out) {
            jit_log("cache write failed for " + tmp_path.string());
            return false;
        }
    }

    std::filesystem::rename(tmp_path, path, ec);
    if (!ec) return true;

    jit_log("cache rename failed for " + tmp_path.string() + " -> " + path.string() + ": " + ec.message());
    std::filesystem::remove(tmp_path, ec);
    return false;
}

std::string fnv1a_digest(const std::vector<std::string>& parts) {
    uint64_t hash = 1469598103934665603ull;
    for (const std::string& part : parts) {
        for (unsigned char c : part) {
            hash ^= static_cast<uint64_t>(c);
            hash *= 1099511628211ull;
        }
        hash ^= 0xffu;
        hash *= 1099511628211ull;
    }

    std::ostringstream out;
    out << std::hex << hash;
    return out.str();
}

} // namespace jit
} // namespace nccl_ep
