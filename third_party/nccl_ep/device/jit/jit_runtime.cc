/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 * See LICENSE.txt for more license information.
 */

#include "device/jit/jit_runtime.hpp"

#include "device/jit/jit_cache.hpp"
#include "device/jit/jit_utils.hpp"
#include "device/jit/nvcc_compiler.hpp"

#include <cuda.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <mutex>
#include <shared_mutex>
#include <sstream>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <vector>
#include <unistd.h>

#ifndef NCCL_EP_JIT_SOURCE_DIR
#define NCCL_EP_JIT_SOURCE_DIR "/usr/include/nccl_ep"
#endif

#ifndef NCCL_EP_JIT_BUILD_INCLUDE_DIR
#define NCCL_EP_JIT_BUILD_INCLUDE_DIR "/usr/include"
#endif

#ifndef NCCL_EP_JIT_CUDA_INCLUDE_DIR
#define NCCL_EP_JIT_CUDA_INCLUDE_DIR ""
#endif

namespace nccl_ep {
namespace jit {
namespace {

struct LoadedKernelModule {
    CUmodule module = nullptr;
    CUfunction function = nullptr;
};

struct FastCacheKey {
    const void* identity = nullptr;
    std::uint64_t runtime_key = 0;
    CUcontext context = nullptr;
    int device = -1;
};

struct FastCacheKeyHash {
    size_t operator()(const FastCacheKey& key) const {
        size_t seed = std::hash<const void*>{}(key.identity);
        seed ^= std::hash<std::uint64_t>{}(key.runtime_key) +
                0x9e3779b97f4a7c15ull + (seed << 6) + (seed >> 2);
        seed ^= std::hash<const void*>{}(static_cast<const void*>(key.context)) +
                0x9e3779b97f4a7c15ull + (seed << 6) + (seed >> 2);
        seed ^= std::hash<int>{}(key.device) + 0x9e3779b97f4a7c15ull + (seed << 6) + (seed >> 2);
        return seed;
    }
};

struct FastCacheKeyEqual {
    bool operator()(const FastCacheKey& lhs, const FastCacheKey& rhs) const {
        return lhs.identity == rhs.identity &&
               lhs.runtime_key == rhs.runtime_key &&
               lhs.context == rhs.context &&
               lhs.device == rhs.device;
    }
};

std::mutex g_module_mutex;
std::mutex g_compile_mutex;
std::unordered_map<std::string, LoadedKernelModule> g_modules;

// Fast in-process cache: avoids all file I/O and fingerprint computation on
// repeat launches of the same variant in the same CUDA context.
struct FastCacheEntry {
    CUfunction function = nullptr;
    int configured_smem_bytes = 0;
};

// Small per-thread cache keeps repeated launches off the shared map.
constexpr int kTlCacheSlots = 4;
struct ThreadCache {
    struct Slot {
        bool valid = false;
        FastCacheKey key{};
        FastCacheEntry entry{};
    };
    Slot slots[kTlCacheSlots];
    int next_evict = 0;
};

std::shared_mutex g_fast_mutex;
std::unordered_map<FastCacheKey, FastCacheEntry, FastCacheKeyHash, FastCacheKeyEqual> g_fast_cache;
thread_local ThreadCache g_thread_cache;

const JitCompiler& compiler_backend() {
    static const NvccCompiler compiler;
    return compiler;
}

void log_jit_event(const JitKernelVariant& variant, const std::string& message) {
    (void)variant;
    jit_log(message);
}

void warn_once_jit_event(
    JitCache& cache,
    const std::string& key,
    const JitKernelVariant& variant,
    const std::string& message) {
    (void)variant;
    cache.warn_once(key, message);
}

double elapsed_sec(std::chrono::steady_clock::time_point begin) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
}

std::filesystem::path configured_source_dir() {
    std::string path = env_value("NCCL_EP_JIT_SOURCE_DIR");
    if (!path.empty()) return std::filesystem::path(path);

    std::string nccl_home = env_value("NCCL_HOME");
    if (!nccl_home.empty()) {
        return std::filesystem::path(nccl_home) / "include" / "nccl_ep";
    }

    return std::filesystem::path(NCCL_EP_JIT_SOURCE_DIR);
}

std::filesystem::path configured_build_include_dir() {
    std::string path = env_value("NCCL_EP_JIT_BUILD_INCLUDE_DIR");
    if (!path.empty()) return std::filesystem::path(path);

    std::string nccl_home = env_value("NCCL_HOME");
    if (!nccl_home.empty()) {
        return std::filesystem::path(nccl_home) / "include";
    }

    return std::filesystem::path(NCCL_EP_JIT_BUILD_INCLUDE_DIR);
}

std::filesystem::path configured_cuda_include_dir() {
    std::string path = env_value("NCCL_EP_JIT_CUDA_INCLUDE_DIR");
    if (!path.empty()) return std::filesystem::path(path);

    std::string cuda_home = env_value("CUDA_HOME");
    if (cuda_home.empty()) cuda_home = env_value("CUDA_PATH");
    if (!cuda_home.empty()) {
        return std::filesystem::path(cuda_home) / "include";
    }

    return std::filesystem::path(NCCL_EP_JIT_CUDA_INCLUDE_DIR);
}

bool is_header_file(const std::filesystem::path& path) {
    const std::string ext = path.extension().string();
    return ext == ".h" || ext == ".hh" || ext == ".hpp" ||
           ext == ".cuh" || ext == ".inc" || ext == ".inl";
}

void append_file_fingerprint(
    std::vector<std::string>* parts,
    const std::filesystem::path& path) {
    if (parts == nullptr) return;

    std::error_code ec;
    const bool exists = std::filesystem::exists(path, ec);
    parts->push_back(path.lexically_normal().string());
    if (ec || !exists) {
        parts->push_back(ec ? ("missing:" + ec.message()) : "missing");
        return;
    }
    parts->push_back(read_file_or_empty(path));
}

void append_header_tree_fingerprint(
    std::vector<std::string>* parts,
    const std::filesystem::path& root) {
    if (parts == nullptr) return;

    std::error_code ec;
    if (!std::filesystem::exists(root, ec)) {
        parts->push_back(root.lexically_normal().string());
        parts->push_back(ec ? ("missing:" + ec.message()) : "missing");
        return;
    }
    if (!std::filesystem::is_directory(root, ec)) {
        append_file_fingerprint(parts, root);
        return;
    }

    std::vector<std::filesystem::path> headers;
    const auto options = std::filesystem::directory_options::skip_permission_denied;
    std::filesystem::recursive_directory_iterator it(root, options, ec);
    const std::filesystem::recursive_directory_iterator end;
    while (!ec && it != end) {
        const std::filesystem::path path = it->path();
        std::error_code file_ec;
        if (it->is_regular_file(file_ec) && !file_ec && is_header_file(path)) {
            headers.push_back(path);
        }
        it.increment(ec);
    }
    std::sort(headers.begin(), headers.end());

    parts->push_back(root.lexically_normal().string());
    if (ec) parts->push_back("walk_error:" + ec.message());
    for (const std::filesystem::path& path : headers) {
        parts->push_back(path.lexically_relative(root).lexically_normal().string());
        parts->push_back(read_file_or_empty(path));
    }
}

// Per-SM compiler environment. Header hashing is paid once per process/SM,
// outside the repeated launch path.
struct SmEnvInfo {
    std::filesystem::path source_dir;
    std::filesystem::path build_include_dir;
    std::filesystem::path cuda_include_dir;
    std::string compiler_id;
    std::vector<std::string> options;
    std::string env_hash;  // hash(compiler_id + headers + options)
};

const SmEnvInfo& get_sm_env_info(int sm) {
    static std::mutex env_mutex;
    static std::unordered_map<int, SmEnvInfo> env_cache;
    std::lock_guard<std::mutex> lock(env_mutex);

    auto it = env_cache.find(sm);
    if (it != env_cache.end()) return it->second;

    SmEnvInfo info;
    info.source_dir = configured_source_dir();
    info.build_include_dir = configured_build_include_dir();
    info.cuda_include_dir = configured_cuda_include_dir();

    const JitCompiler& compiler = compiler_backend();
    info.compiler_id = compiler.compiler_id();
    JitCompileConfig compile_config;
    compile_config.sm = sm;
    compile_config.source_dir = info.source_dir;
    compile_config.build_include_dir = info.build_include_dir;
    compile_config.cuda_include_dir = info.cuda_include_dir;
    info.options = compiler.compile_options(compile_config);

    std::vector<std::string> env_parts = {
        info.compiler_id,
    };
    append_header_tree_fingerprint(&env_parts, info.source_dir / "device");
    append_header_tree_fingerprint(&env_parts, info.build_include_dir);
    env_parts.insert(env_parts.end(), info.options.begin(), info.options.end());
    info.env_hash = fnv1a_digest(env_parts);

    auto [inserted_it, ok] = env_cache.emplace(sm, std::move(info));
    (void)ok;
    return inserted_it->second;
}

std::string kernel_family(const JitKernelVariant& variant) {
    return variant.kernel_family.empty() ? "jit_kernel" : std::string(variant.kernel_family);
}

std::string kernel_log_prefix(const JitKernelVariant& variant) {
    return kernel_family(variant) + " JIT";
}

// Per-variant fingerprint. No file I/O here; env_hash covers header inputs.
std::string source_fingerprint(
    const JitKernelVariant& variant,
    int sm,
    const std::string& env_hash) {
    const std::vector<std::string> parts = {
        kernel_family(variant),
        std::string(variant.variant_name),
        std::string(variant.entry_name),
        std::string(variant.source),
        "sm=" + std::to_string(sm),
        "runtime_key=" + std::to_string(variant.runtime_key),
        env_hash,
    };
    return fnv1a_digest(parts);
}

std::string metadata_json(
    const JitKernelVariant& variant,
    const std::string& key,
    int sm,
    const std::vector<std::string>& options,
    const std::string& compiler_id,
    double jit_overhead_sec,
    double compiler_sec) {
    std::ostringstream out;
    out << "{\n"
        << "  \"key\": \"" << json_escape(key) << "\",\n"
        << "  \"kernel_family\": \"" << json_escape(kernel_family(variant)) << "\",\n"
        << "  \"variant\": \"" << json_escape(variant.variant_name) << "\",\n"
        << "  \"entry\": \"" << json_escape(variant.entry_name) << "\",\n"
        << "  \"sm\": " << sm << ",\n"
        << "  \"blocks\": " << variant.num_blocks << ",\n"
        << "  \"block_dim\": " << variant.block_dim << ",\n"
        << "  \"dynamic_smem_bytes\": " << variant.dynamic_smem_bytes << ",\n"
        << "  \"jit_overhead_sec\": " << jit_overhead_sec << ",\n"
        << "  \"nvcc_process_sec\": " << compiler_sec << ",\n"
        << "  \"compiler\": \"" << json_escape(compiler_id) << "\",\n"
        << "  \"options\": [";
    for (size_t i = 0; i < options.size(); ++i) {
        if (i != 0) out << ", ";
        out << "\"" << json_escape(options[i]) << "\"";
    }
    out << "]\n}\n";
    return out.str();
}

std::filesystem::path temporary_cubin_path(const std::filesystem::path& cubin_path) {
    std::ostringstream suffix;
    suffix << ".tmp." << static_cast<long>(getpid()) << "."
           << std::hash<std::thread::id>{}(std::this_thread::get_id());
    return cubin_path.parent_path() / (cubin_path.filename().string() + suffix.str());
}

std::string cu_error_string(CUresult result) {
    const char* name = nullptr;
    const char* message = nullptr;
    cuGetErrorName(result, &name);
    cuGetErrorString(result, &message);
    std::ostringstream out;
    out << (name == nullptr ? "CUDA_ERROR_UNKNOWN" : name);
    if (message != nullptr) out << ": " << message;
    return out.str();
}

std::string context_key(CUcontext context) {
    std::ostringstream out;
    out << static_cast<const void*>(context);
    return out.str();
}

void log_memory_cache_event(
    const FastCacheKey& fast_key,
    const JitKernelVariant& variant,
    const char* state,
    const char* scope) {
    log_jit_event(
        variant,
        kernel_log_prefix(variant) + " memory_cache=" + state +
        " scope=" + scope +
        " variant=" + std::string(variant.variant_name) +
        " device=" + std::to_string(fast_key.device));
}

JitKernelStatus load_function(
    const std::string& key,
    const std::string& cubin,
    const std::string& entry_name,
    CUfunction* function,
    std::string* error) {
    {
        std::lock_guard<std::mutex> lock(g_module_mutex);
        const auto it = g_modules.find(key);
        if (it != g_modules.end()) {
            *function = it->second.function;
            return JitKernelStatus::kLaunched;
        }
    }

    // Load outside the mutex so unrelated threads are not blocked by CUDA linking.
    CUmodule module = nullptr;
    CUresult rc = cuInit(0);
    if (rc == CUDA_SUCCESS) rc = cuModuleLoadDataEx(&module, cubin.data(), 0, nullptr, nullptr);
    if (rc != CUDA_SUCCESS) {
        if (error != nullptr) *error = cu_error_string(rc);
        return JitKernelStatus::kLoadFailed;
    }

    rc = cuModuleGetFunction(function, module, entry_name.c_str());
    if (rc != CUDA_SUCCESS) {
        if (error != nullptr) *error = cu_error_string(rc);
        cuModuleUnload(module);
        return JitKernelStatus::kLoadFailed;
    }

    std::lock_guard<std::mutex> lock(g_module_mutex);
    auto [it, inserted] = g_modules.emplace(key, LoadedKernelModule{module, *function});
    if (!inserted) {
        cuModuleUnload(module);
        *function = it->second.function;
    }
    return JitKernelStatus::kLaunched;
}

JitKernelStatus configure_smem(
    CUfunction function,
    int dynamic_smem_bytes,
    std::string* error) {
    CUresult rc = cuFuncSetAttribute(
        function,
        CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
        dynamic_smem_bytes);
    if (rc != CUDA_SUCCESS) {
        if (error != nullptr) *error = cu_error_string(rc);
        return JitKernelStatus::kAttributeFailed;
    }
    return JitKernelStatus::kLaunched;
}

void update_thread_fast_cache(
    const FastCacheKey& fast_key,
    CUfunction function,
    int configured_smem_bytes) {
    if (fast_key.identity == nullptr) return;

    // Update an existing slot if the key is already present (preserves max smem).
    for (int i = 0; i < kTlCacheSlots; ++i) {
        ThreadCache::Slot& slot = g_thread_cache.slots[i];
        if (slot.valid && FastCacheKeyEqual{}(slot.key, fast_key)) {
            if (slot.entry.configured_smem_bytes > configured_smem_bytes) {
                configured_smem_bytes = slot.entry.configured_smem_bytes;
            }
            slot.entry = FastCacheEntry{function, configured_smem_bytes};
            return;
        }
    }

    // Insert into the next round-robin eviction slot.
    ThreadCache::Slot& slot = g_thread_cache.slots[g_thread_cache.next_evict];
    slot.valid = true;
    slot.key = fast_key;
    slot.entry = FastCacheEntry{function, configured_smem_bytes};
    g_thread_cache.next_evict = (g_thread_cache.next_evict + 1) % kTlCacheSlots;
}

void erase_thread_fast_cache(const FastCacheKey& fast_key) {
    for (int i = 0; i < kTlCacheSlots; ++i) {
        ThreadCache::Slot& slot = g_thread_cache.slots[i];
        if (slot.valid && FastCacheKeyEqual{}(slot.key, fast_key)) {
            slot = ThreadCache::Slot{};
        }
    }
}

void update_fast_cache(const FastCacheKey& fast_key, CUfunction function, int configured_smem_bytes) {
    if (fast_key.identity == nullptr) return;

    {
        std::unique_lock<std::shared_mutex> lock(g_fast_mutex);
        FastCacheEntry& entry = g_fast_cache[fast_key];
        entry.function = function;
        if (configured_smem_bytes > entry.configured_smem_bytes) {
            entry.configured_smem_bytes = configured_smem_bytes;
        }
        configured_smem_bytes = entry.configured_smem_bytes;
    }
    update_thread_fast_cache(fast_key, function, configured_smem_bytes);
}

void erase_fast_cache(const FastCacheKey& fast_key) {
    {
        std::unique_lock<std::shared_mutex> lock(g_fast_mutex);
        g_fast_cache.erase(fast_key);
    }
    erase_thread_fast_cache(fast_key);
}

CUresult launch_function(
    CUfunction function,
    const JitKernelVariant& variant,
    void* kernel_param,
    std::size_t kernel_param_size,
    cudaStream_t stream) {
    // A zero size means kernel_param points to one fixed-size kernel argument.
    // Non-zero size means kernel_param points to a byte buffer with all kernel arguments.
    if (kernel_param_size == 0) {
        void* kernel_args[] = {kernel_param};
        return cuLaunchKernel(
            function,
            variant.num_blocks, 1, 1,
            variant.block_dim, 1, 1,
            static_cast<unsigned int>(variant.dynamic_smem_bytes),
            reinterpret_cast<CUstream>(stream),
            kernel_args,
            nullptr);
    }

    // This buffer already has the arguments laid out in the order the JIT kernel
    // expects, so pass CUDA the buffer pointer and its size.
    void* extra[] = {
        CU_LAUNCH_PARAM_BUFFER_POINTER, kernel_param,
        CU_LAUNCH_PARAM_BUFFER_SIZE, &kernel_param_size,
        CU_LAUNCH_PARAM_END
    };
    return cuLaunchKernel(
        function,
        variant.num_blocks, 1, 1,
        variant.block_dim, 1, 1,
        static_cast<unsigned int>(variant.dynamic_smem_bytes),
        reinterpret_cast<CUstream>(stream),
        nullptr,
        extra);
}

JitKernelStatus try_fast_launch(
    const FastCacheKey& fast_key,
    const JitKernelVariant& variant,
    void* kernel_param,
    std::size_t kernel_param_size,
    cudaStream_t stream,
    std::string* error) {
    if (fast_key.identity == nullptr) {
        log_memory_cache_event(fast_key, variant, "skip", "no_identity");
        return JitKernelStatus::kDisabled;
    }

    FastCacheEntry entry;
    bool tl_hit = false;
    for (int i = 0; i < kTlCacheSlots; ++i) {
        const ThreadCache::Slot& slot = g_thread_cache.slots[i];
        if (slot.valid && FastCacheKeyEqual{}(slot.key, fast_key)) {
            entry = slot.entry;
            tl_hit = true;
            break;
        }
    }
    if (!tl_hit) {
        {
            std::shared_lock<std::shared_mutex> lock(g_fast_mutex);
            const auto it = g_fast_cache.find(fast_key);
            if (it == g_fast_cache.end()) {
                log_memory_cache_event(fast_key, variant, "miss", "process");
                return JitKernelStatus::kDisabled;
            }
            entry = it->second;
        }
        update_thread_fast_cache(fast_key, entry.function, entry.configured_smem_bytes);
        log_memory_cache_event(fast_key, variant, "hit", "process");
    } else {
        log_memory_cache_event(fast_key, variant, "hit", "thread");
    }

    if (variant.dynamic_smem_bytes > entry.configured_smem_bytes) {
        const JitKernelStatus status = configure_smem(entry.function, variant.dynamic_smem_bytes, error);
        if (status != JitKernelStatus::kLaunched) {
            erase_fast_cache(fast_key);
            return JitKernelStatus::kDisabled;
        }
        update_fast_cache(fast_key, entry.function, variant.dynamic_smem_bytes);
    }

    CUresult rc = launch_function(entry.function, variant, kernel_param, kernel_param_size, stream);
    if (rc != CUDA_SUCCESS) {
        if (error != nullptr) *error = cu_error_string(rc);
        return JitKernelStatus::kLaunchFailed;
    }
    return JitKernelStatus::kLaunched;
}

struct CachePaths {
    std::filesystem::path variant_dir;
    std::filesystem::path cubin_path;
    std::filesystem::path metadata_path;
    std::filesystem::path log_path;
    std::filesystem::path source_path;
    std::filesystem::path failed_path;
    std::filesystem::path tmp_cubin_path;
    std::filesystem::path lock_path;
};

CachePaths make_cache_paths(const JitCache& cache, const std::string& key) {
    CachePaths paths;
    paths.variant_dir = cache.root_dir() / key;
    paths.cubin_path = paths.variant_dir / "kernel.cubin";
    paths.metadata_path = paths.variant_dir / "metadata.json";
    paths.log_path = paths.variant_dir / "compile.log";
    paths.source_path = paths.variant_dir / "kernel.cu";
    paths.failed_path = paths.variant_dir / "compile.failed";
    paths.tmp_cubin_path = temporary_cubin_path(paths.cubin_path);
    paths.lock_path = paths.variant_dir / "compile.lock";
    return paths;
}

JitKernelStatus ensure_cubin(
    const JitKernelVariant& variant,
    const SmEnvInfo& env_info,
    const CachePaths& paths,
    const std::string& key,
    const std::string& log_prefix,
    int sm,
    std::chrono::steady_clock::time_point jit_begin,
    std::string* cubin,
    std::string* error) {
    JitCache& cache = JitCache::instance();
    if (cubin == nullptr) return JitKernelStatus::kCompileFailed;

    *cubin = read_file_or_empty(paths.cubin_path);
    if (cubin->empty()) {
        const std::string cached_failure = read_file_or_empty(paths.failed_path);
        if (!cached_failure.empty()) {
            cache.mark_failed(key);
            if (error != nullptr) *error = cached_failure;
            log_jit_event(
                variant,
                log_prefix + " disk_cache=failed_marker key=" + key +
                    " variant=" + std::string(variant.variant_name) +
                    " marker=" + paths.failed_path.string() +
                    " jit_overhead_sec=" + std::to_string(elapsed_sec(jit_begin)));
            return JitKernelStatus::kCompileFailed;
        }
    }
    if (!cubin->empty()) {
        log_jit_event(
            variant,
            log_prefix + " disk_cache=hit key=" + key +
                " variant=" + std::string(variant.variant_name) +
                " cubin=" + paths.cubin_path.string() +
                " jit_overhead_sec=" + std::to_string(elapsed_sec(jit_begin)));
        return JitKernelStatus::kLaunched;
    }

    log_jit_event(
        variant,
        log_prefix + " disk_cache=miss key=" + key +
            " variant=" + std::string(variant.variant_name) +
            " cubin=" + paths.cubin_path.string());

    ScopedFileLock file_lock;
    std::string lock_error;
    log_jit_event(variant, log_prefix + " compile_lock=waiting key=" + key + " path=" + paths.lock_path.string());
    if (!file_lock.lock(paths.lock_path, &lock_error)) {
        cache.mark_failed(key);
        if (error != nullptr) *error = lock_error;
        warn_once_jit_event(
            cache,
            key,
            variant,
            log_prefix + " compile_lock=failed key=" + key +
                " variant=" + std::string(variant.variant_name) +
                "; using static path");
        return JitKernelStatus::kCompileFailed;
    }
    log_jit_event(variant, log_prefix + " compile_lock=acquired key=" + key + " path=" + paths.lock_path.string());

    std::lock_guard<std::mutex> compile_lock(g_compile_mutex);
    *cubin = read_file_or_empty(paths.cubin_path);
    if (!cubin->empty()) {
        log_jit_event(
            variant,
            log_prefix + " disk_cache=hit_after_lock key=" + key +
                " variant=" + std::string(variant.variant_name) +
                " cubin=" + paths.cubin_path.string() +
                " jit_overhead_sec=" + std::to_string(elapsed_sec(jit_begin)));
        return JitKernelStatus::kLaunched;
    }

    const std::string cached_failure = read_file_or_empty(paths.failed_path);
    if (!cached_failure.empty()) {
        cache.mark_failed(key);
        if (error != nullptr) *error = cached_failure;
        log_jit_event(
            variant,
            log_prefix + " disk_cache=failed_marker_after_lock key=" + key +
                " variant=" + std::string(variant.variant_name) +
                " marker=" + paths.failed_path.string() +
                " jit_overhead_sec=" + std::to_string(elapsed_sec(jit_begin)));
        return JitKernelStatus::kCompileFailed;
    }

    JitCompileInput input;
    input.source = std::string(variant.source);
    input.options = env_info.options;
    input.source_path = paths.source_path;
    input.cubin_path = paths.tmp_cubin_path;
    input.log_enabled = env_flag_enabled("NCCL_EP_JIT_LOG");

    JitCompileOutput compile_output;
    const JitCompiler& compiler = compiler_backend();
    if (!compiler.compile_to_cubin(input, &compile_output)) {
        cache.mark_failed(key);
        std::error_code remove_ec;
        std::filesystem::remove(paths.tmp_cubin_path, remove_ec);
        const double jit_overhead_sec = elapsed_sec(jit_begin);
        compile_output.log += "jit_overhead_sec=" + std::to_string(jit_overhead_sec) + "\n";
        if (!write_file_atomic(paths.log_path, compile_output.log, false)) {
            log_jit_event(variant, log_prefix + " artifact=compile_log write=failed path=" + paths.log_path.string());
        }
        if (!write_file_atomic(paths.failed_path, compile_output.log, false)) {
            log_jit_event(variant, log_prefix + " artifact=failed_marker write=failed path=" + paths.failed_path.string());
        }
        warn_once_jit_event(
            cache,
            key,
            variant,
            log_prefix + " compile=failed key=" + key +
                " variant=" + std::string(variant.variant_name) +
                " log=" + paths.log_path.string() +
                " jit_overhead_sec=" + std::to_string(jit_overhead_sec) +
                " nvcc_process_sec=" + std::to_string(compile_output.compiler_sec) +
                "; using static path");
        if (error != nullptr) *error = compile_output.log;
        return JitKernelStatus::kCompileFailed;
    }

    *cubin = compile_output.cubin;
    std::error_code remove_ec;
    std::filesystem::remove(paths.tmp_cubin_path, remove_ec);
    if (write_file_atomic(paths.cubin_path, *cubin, true)) {
        log_jit_event(
            variant,
            log_prefix + " disk_cache=write artifact=cubin key=" + key +
                " path=" + paths.cubin_path.string() +
                " bytes=" + std::to_string(cubin->size()));
    } else {
        log_jit_event(variant, log_prefix + " artifact=cubin write=failed path=" + paths.cubin_path.string());
    }
    const double jit_overhead_sec = elapsed_sec(jit_begin);
    compile_output.log += "jit_overhead_sec=" + std::to_string(jit_overhead_sec) + "\n";
    if (!write_file_atomic(paths.log_path, compile_output.log, false)) {
        log_jit_event(variant, log_prefix + " artifact=compile_log write=failed path=" + paths.log_path.string());
    }
    std::filesystem::remove(paths.failed_path, remove_ec);
    if (!write_file_atomic(
            paths.metadata_path,
            metadata_json(variant, key, sm, env_info.options, env_info.compiler_id, jit_overhead_sec, compile_output.compiler_sec),
            false)) {
        log_jit_event(variant, log_prefix + " artifact=metadata write=failed path=" + paths.metadata_path.string());
    }
    log_jit_event(
        variant,
        log_prefix + " compile=succeeded key=" + key +
            " variant=" + std::string(variant.variant_name) +
            " cubin=" + paths.cubin_path.string() +
            " jit_overhead_sec=" + std::to_string(jit_overhead_sec) +
            " nvcc_process_sec=" + std::to_string(compile_output.compiler_sec));
    return JitKernelStatus::kLaunched;
}

JitKernelStatus load_and_launch_kernel(
    const JitKernelVariant& variant,
    const FastCacheKey& fast_key,
    const CachePaths& paths,
    const std::string& key,
    const std::string& cubin,
    const std::string& log_prefix,
    void* kernel_param,
    std::size_t kernel_param_size,
    cudaStream_t stream,
    std::string* error) {
    JitCache& cache = JitCache::instance();
    const std::string module_key =
        key + ":ctx=" + context_key(fast_key.context) + ":device=" + std::to_string(fast_key.device);

    CUfunction function = nullptr;
    JitKernelStatus status = load_function(module_key, cubin, std::string(variant.entry_name), &function, error);
    if (status != JitKernelStatus::kLaunched) {
        cache.mark_failed(key);
        warn_once_jit_event(
            cache,
            key,
            variant,
            log_prefix + " module_load=failed key=" + key +
                " variant=" + std::string(variant.variant_name) +
                "; using static path");
        return status;
    }

    status = configure_smem(function, variant.dynamic_smem_bytes, error);
    if (status != JitKernelStatus::kLaunched) {
        cache.mark_failed(key);
        warn_once_jit_event(
            cache,
            key,
            variant,
            log_prefix + " smem_attribute=failed key=" + key +
                " variant=" + std::string(variant.variant_name) +
                "; using static path");
        return status;
    }

    update_fast_cache(fast_key, function, variant.dynamic_smem_bytes);

    CUresult rc = launch_function(function, variant, kernel_param, kernel_param_size, stream);
    if (rc != CUDA_SUCCESS) {
        if (error != nullptr) *error = cu_error_string(rc);
        warn_once_jit_event(
            cache,
            key,
            variant,
            log_prefix + " launch=failed key=" + key +
                " variant=" + std::string(variant.variant_name));
        return JitKernelStatus::kLaunchFailed;
    }

    log_jit_event(
        variant,
        log_prefix + " launch=succeeded key=" + key +
            " variant=" + std::string(variant.variant_name) +
            " module_cache=ready cubin=" + paths.cubin_path.string());
    return JitKernelStatus::kLaunched;
}

} // namespace

const char* jit_kernel_status_name(JitKernelStatus status) {
    switch (status) {
    case JitKernelStatus::kLaunched: return "launched";
    case JitKernelStatus::kDisabled: return "disabled";
    case JitKernelStatus::kUnsupportedDevice: return "unsupported device";
    case JitKernelStatus::kCompileFailed: return "compile failed";
    case JitKernelStatus::kLoadFailed: return "load failed";
    case JitKernelStatus::kAttributeFailed: return "attribute failed";
    case JitKernelStatus::kLaunchFailed: return "launch failed";
    }
    return "unknown";
}

JitKernelStatus launch_jit_kernel(
    const JitKernelVariant& variant,
    void* kernel_param,
    std::size_t kernel_param_size,
    cudaStream_t stream,
    std::string* error) {
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) {
        if (error != nullptr) *error = "cudaGetDevice failed";
        return JitKernelStatus::kUnsupportedDevice;
    }

    CUcontext context = nullptr;
    CUresult rc = cuCtxGetCurrent(&context);
    if (rc == CUDA_SUCCESS && context == nullptr) {
        cudaFree(nullptr);
        rc = cuCtxGetCurrent(&context);
    }
    if (rc != CUDA_SUCCESS || context == nullptr) {
        if (error != nullptr) *error = (rc == CUDA_SUCCESS) ? "no current CUDA context" : cu_error_string(rc);
        return JitKernelStatus::kUnsupportedDevice;
    }

    const FastCacheKey fast_key{variant.identity, variant.runtime_key, context, device};
    const JitKernelStatus fast_status =
        try_fast_launch(fast_key, variant, kernel_param, kernel_param_size, stream, error);
    if (fast_status == JitKernelStatus::kLaunched ||
        fast_status == JitKernelStatus::kLaunchFailed) {
        return fast_status;
    }

    const auto jit_begin = std::chrono::steady_clock::now();
    const std::string log_prefix = kernel_log_prefix(variant);

    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, device) != cudaSuccess) {
        if (error != nullptr) *error = "cudaGetDeviceProperties failed";
        return JitKernelStatus::kUnsupportedDevice;
    }

    const int sm = prop.major * 10 + prop.minor;
    if (sm < variant.min_sm) {
        if (error != nullptr) {
            *error = log_prefix + " requires sm_" + std::to_string(variant.min_sm) + " or newer";
        }
        return JitKernelStatus::kUnsupportedDevice;
    }

    const SmEnvInfo& env_info = get_sm_env_info(sm);
    const std::string key = source_fingerprint(variant, sm, env_info.env_hash);
    JitCache& cache = JitCache::instance();
    if (cache.has_failed(key)) return JitKernelStatus::kCompileFailed;
    const CachePaths paths = make_cache_paths(cache, key);

    log_jit_event(
        variant,
        log_prefix + " request variant=" + std::string(variant.variant_name) +
        " key=" + key +
        " device=" + std::to_string(device) +
        " sm=" + std::to_string(sm) +
        " cache_root=" + cache.root_dir().string() +
        " source_dir=" + env_info.source_dir.string());

    std::string cubin;
    JitKernelStatus status =
        ensure_cubin(variant, env_info, paths, key, log_prefix, sm, jit_begin, &cubin, error);
    if (status != JitKernelStatus::kLaunched) return status;

    return load_and_launch_kernel(
        variant, fast_key, paths, key, cubin, log_prefix, kernel_param, kernel_param_size, stream, error);
}

// This overload is for kernels that don't need to pass kernel arguments.
JitKernelStatus launch_jit_kernel(
    const JitKernelVariant& variant,
    void* kernel_param,
    cudaStream_t stream,
    std::string* error) {
    return launch_jit_kernel(variant, kernel_param, 0, stream, error);
}

} // namespace jit
} // namespace nccl_ep
