// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#ifndef __has_include
#  define OPENTELEMETRY_HAS_INCLUDE(x) 0
#else
#  define OPENTELEMETRY_HAS_INCLUDE(x) __has_include(x)
#endif

#if !defined(__GLIBCXX__) || OPENTELEMETRY_HAS_INCLUDE(<codecvt>)  // >= libstdc++-5
#  define OPENTELEMETRY_TRIVIALITY_TYPE_TRAITS
#endif
