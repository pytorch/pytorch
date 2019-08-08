/*
 *  Copyright (c) 2018-present, Facebook, Inc.
 *
 *  This source code is licensed under the MIT license found in the LICENSE
 *  file in the root directory of this source tree.
 *
 */
/**
 * @file ReferenceAllocators.h
 *
 * Reference allocators are used to create and delete various classes of JNI references (local,
 * global, and weak global).
 */

#pragma once

#include "Common.h"

namespace facebook { namespace jni {

/// Allocator that handles local references
class LocalReferenceAllocator {
 public:
  jobject newReference(jobject original) const;
  void deleteReference(jobject reference) const noexcept;
  bool verifyReference(jobject reference) const noexcept;
};

/// Allocator that handles global references
class GlobalReferenceAllocator {
 public:
  jobject newReference(jobject original) const;
  void deleteReference(jobject reference) const noexcept;
  bool verifyReference(jobject reference) const noexcept;
};

/// Allocator that handles weak global references
class WeakGlobalReferenceAllocator {
 public:
  jobject newReference(jobject original) const;
  void deleteReference(jobject reference) const noexcept;
  bool verifyReference(jobject reference) const noexcept;
};

/**
 * @return Helper based on GetObjectRefType.  Since this isn't defined
 * on all versions of Java or Android, if the type can't be
 * determined, this returns true.  If reference is nullptr, returns
 * true.
 */
bool isObjectRefType(jobject reference, jobjectRefType refType);

}}

#include "ReferenceAllocators-inl.h"
