/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 * See LICENSE.txt for more license information.
 */

#include "device/jit/nvcc_compiler.hpp"

#include "device/jit/jit_utils.hpp"

#include <chrono>
#include <cerrno>
#include <cstring>
#include <filesystem>
#include <sstream>
#include <string>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

namespace nccl_ep {
namespace jit {
namespace {

struct ProcessResult {
    int exit_code = -1;
    std::string output;
};

std::string resolve_nvcc() {
    std::string nvcc = env_value("NCCL_EP_JIT_NVCC");
    if (!nvcc.empty()) return nvcc;

    nvcc = env_value("NVCC");
    if (!nvcc.empty()) return nvcc;

    std::string cuda_home = env_value("CUDA_HOME");
    if (cuda_home.empty()) cuda_home = env_value("CUDA_PATH");
    if (!cuda_home.empty()) {
        const std::filesystem::path cuda_nvcc = std::filesystem::path(cuda_home) / "bin" / "nvcc";
        std::error_code ec;
        if (std::filesystem::exists(cuda_nvcc, ec)) return cuda_nvcc.string();
    }

    return "nvcc";
}

std::vector<std::string> include_options(const JitCompileConfig& config) {
    std::vector<std::string> options;
    const auto add_include = [&options](const std::filesystem::path& path) {
        if (!path.empty()) options.push_back("-I" + path.string());
    };

    // Include nccl_ep kernel headers and common.hpp
    add_include(config.source_dir);
    add_include(config.source_dir / "device");

    // Include all NCCL device headers
    if (!config.build_include_dir.empty()) {
        add_include(config.build_include_dir);
        add_include(config.build_include_dir / "nccl_device");
    }

    if (!config.cuda_include_dir.empty()) {
        add_include(config.cuda_include_dir);
        add_include(config.cuda_include_dir / "cccl");
    }

    return options;
}

std::string command_to_string(const std::vector<std::string>& argv) {
    std::ostringstream out;
    for (size_t i = 0; i < argv.size(); ++i) {
        if (i != 0) out << ' ';
        const bool needs_quotes = argv[i].find_first_of(" \t\n\"'\\") != std::string::npos;
        if (!needs_quotes) {
            out << argv[i];
            continue;
        }

        out << '\'';
        for (char c : argv[i]) {
            if (c == '\'') out << "'\\''";
            else out << c;
        }
        out << '\'';
    }
    return out.str();
}

bool run_process(const std::vector<std::string>& argv, ProcessResult* result) {
    if (result == nullptr) return false;
    result->exit_code = -1;
    result->output.clear();
    if (argv.empty()) {
        result->output = "empty command";
        return false;
    }

    int pipe_fd[2] = {-1, -1};
    if (pipe(pipe_fd) != 0) {
        result->output = std::string("pipe failed: ") + std::strerror(errno);
        return false;
    }

    std::vector<char*> raw_argv;
    raw_argv.reserve(argv.size() + 1);
    for (const std::string& arg : argv) raw_argv.push_back(const_cast<char*>(arg.c_str()));
    raw_argv.push_back(nullptr);

    const pid_t pid = fork();
    if (pid < 0) {
        const int saved_errno = errno;
        close(pipe_fd[0]);
        close(pipe_fd[1]);
        result->output = std::string("fork failed: ") + std::strerror(saved_errno);
        return false;
    }

    if (pid == 0) {
        close(pipe_fd[0]);
        dup2(pipe_fd[1], STDOUT_FILENO);
        dup2(pipe_fd[1], STDERR_FILENO);
        close(pipe_fd[1]);
        execvp(raw_argv[0], raw_argv.data());
        _exit(127);
    }

    close(pipe_fd[1]);
    char buffer[4096];
    while (true) {
        const ssize_t bytes_read = read(pipe_fd[0], buffer, sizeof(buffer));
        if (bytes_read > 0) {
            result->output.append(buffer, static_cast<size_t>(bytes_read));
            continue;
        }
        if (bytes_read == 0) break;
        if (errno == EINTR) continue;
        result->output += std::string("\nread failed: ") + std::strerror(errno);
        break;
    }
    close(pipe_fd[0]);

    int status = 0;
    while (waitpid(pid, &status, 0) < 0) {
        if (errno == EINTR) continue;
        result->output += std::string("\nwaitpid failed: ") + std::strerror(errno);
        return false;
    }

    if (WIFEXITED(status)) {
        result->exit_code = WEXITSTATUS(status);
    } else if (WIFSIGNALED(status)) {
        result->exit_code = 128 + WTERMSIG(status);
    }
    return result->exit_code == 0;
}

void compiler_log(const JitCompileInput& input, const std::string& message) {
    if (!input.log_enabled) return;
    jit_log(message);
}

} // namespace

std::string NvccCompiler::compiler_id() const {
    static const std::string id = [] {
        const std::string nvcc = resolve_nvcc();
        ProcessResult result;
        const std::vector<std::string> argv = {nvcc, "--version"};
        if (run_process(argv, &result)) {
            return "nvcc=" + nvcc + "\n" + result.output;
        }

        std::ostringstream out;
        out << "nvcc=" << nvcc << "\nversion unavailable";
        if (result.exit_code >= 0) out << " exit_code=" << result.exit_code;
        if (!result.output.empty()) out << "\n" << result.output;
        return out.str();
    }();
    return id;
}

std::vector<std::string> NvccCompiler::compile_options(const JitCompileConfig& config) const {
    std::vector<std::string> options;
    options.push_back("--std=c++17");
    options.push_back("--extended-lambda");
    options.push_back("--expt-relaxed-constexpr");

    const std::string arch_flags = env_value("NVCC_ARCH_FLAGS");
    std::vector<std::string> arch_options = split_env_flags(arch_flags.c_str());
    if (arch_options.empty()) {
        options.push_back("-arch=sm_" + std::to_string(config.sm));
    } else {
        options.insert(options.end(), arch_options.begin(), arch_options.end());
    }
#ifdef HYBRIDEP_ENABLE_WARP_TIMING
    options.push_back("-DHYBRIDEP_ENABLE_WARP_TIMING=1");
#endif
#ifdef NDEBUG
    options.push_back("-DNDEBUG=1");
#endif

    const std::vector<std::string> includes = include_options(config);
    options.insert(options.end(), includes.begin(), includes.end());

    const std::string extra_flags = env_value("NVCC_EXTRA_FLAGS");
    std::vector<std::string> extra_options = split_env_flags(extra_flags.c_str());
    options.insert(options.end(), extra_options.begin(), extra_options.end());
    return options;
}

bool NvccCompiler::compile_to_cubin(
    const JitCompileInput& input,
    JitCompileOutput* output) const {
    if (output == nullptr) return false;
    output->cubin.clear();
    output->log.clear();
    output->compiler_sec = 0.0;

    if (input.source_path.empty() || input.cubin_path.empty()) {
        output->log = "NVCC compile failed: source_path and cubin_path must be set";
        return false;
    }

    if (!write_file_atomic(input.source_path, input.source, false)) {
        output->log = "NVCC compile failed: could not write source " + input.source_path.string();
        return false;
    }

    std::error_code ec;
    std::filesystem::create_directories(input.cubin_path.parent_path(), ec);
    if (ec) {
        output->log = "NVCC compile failed: mkdir " + input.cubin_path.parent_path().string() + ": " + ec.message();
        return false;
    }

    std::filesystem::remove(input.cubin_path, ec);

    std::vector<std::string> argv;
    argv.reserve(input.options.size() + 5);
    argv.push_back(resolve_nvcc());
    argv.push_back("--cubin");
    argv.push_back(input.source_path.string());
    argv.push_back("-o");
    argv.push_back(input.cubin_path.string());
    argv.insert(argv.end(), input.options.begin(), input.options.end());

    const std::string command = command_to_string(argv);
    compiler_log(input, "compiler=nvcc action=compile_begin command=" + command);

    ProcessResult result;
    const auto compile_begin = std::chrono::steady_clock::now();
    const bool ok = run_process(argv, &result);
    const auto compile_end = std::chrono::steady_clock::now();
    output->compiler_sec = std::chrono::duration<double>(compile_end - compile_begin).count();

    std::ostringstream log;
    log << "$ " << command << "\n";
    if (!result.output.empty()) log << result.output;
    if (result.output.empty() || result.output.back() != '\n') log << "\n";
    log << "exit_code=" << result.exit_code << "\n";
    log << "nvcc_process_sec=" << output->compiler_sec << "\n";
    output->log = log.str();

    if (!ok) {
        compiler_log(
            input,
            "compiler=nvcc action=compile_failed exit_code=" + std::to_string(result.exit_code) +
                " nvcc_process_sec=" + std::to_string(output->compiler_sec));
        return false;
    }

    output->cubin = read_file_or_empty(input.cubin_path);
    if (output->cubin.empty()) {
        output->log += "NVCC compile failed: cubin was not produced at " + input.cubin_path.string() + "\n";
        compiler_log(input, "compiler=nvcc action=compile_failed reason=missing_cubin path=" + input.cubin_path.string());
        return false;
    }

    compiler_log(
        input,
        "compiler=nvcc action=compile_succeeded cubin_bytes=" + std::to_string(output->cubin.size()) +
            " nvcc_process_sec=" + std::to_string(output->compiler_sec));
    return true;
}

} // namespace jit
} // namespace nccl_ep
