/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 * See LICENSE.txt for more license information.
 */

#pragma once

#include "device/jit/jit_compiler.hpp"

namespace nccl_ep {
namespace jit {

class NvccCompiler final : public JitCompiler {
public:
    std::string compiler_id() const override;
    std::vector<std::string> compile_options(const JitCompileConfig& config) const override;
    bool compile_to_cubin(const JitCompileInput& input, JitCompileOutput* output) const override;
};

} // namespace jit
} // namespace nccl_ep
