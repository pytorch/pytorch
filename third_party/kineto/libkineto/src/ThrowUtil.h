/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

// Kineto is a standalone library and cannot depend on c10's error-reporting
// macros (TORCH_CHECK, C10_THROW_ERROR), so it raises std exceptions directly.
// KINETO_THROW centralizes those throws: it forwards the caller's exception
// type and constructor arguments to a single raw throw. Routing every throw
// through this macro keeps the one raw throw statement in this audited header,
// which is why only this file carries the RAWTHROW linter's allow annotation;
// call sites read as KINETO_THROW(...) and never contain a bare throw.
//
// ExceptionType is the exception to raise (e.g. std::invalid_argument); the
// remaining arguments are forwarded to its constructor, so any existing message
// expression -- a string literal, a std::string, a concatenation, or a
// fmt::format(...) call -- carries over unchanged and the thrown type is
// preserved. Callers keep whatever headers those arguments require.
#define KINETO_THROW(ExceptionType, ...) \
  throw ExceptionType(__VA_ARGS__) // @allow-raw-throw
