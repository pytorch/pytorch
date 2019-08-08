/*
 *  Copyright (c) 2018-present, Facebook, Inc.
 *
 *  This source code is licensed under the MIT license found in the LICENSE
 *  file in the root directory of this source tree.
 *
 */
#pragma once

#include <cassert>
#include <new>
#include <atomic>

#include "Environment.h"

namespace facebook {
namespace jni {

/// @cond INTERNAL
namespace internal {

// Statistics mostly provided for test (only updated if FBJNI_DEBUG_REFS is defined)
struct ReferenceStats {
  std::atomic_uint locals_created, globals_created, weaks_created,
                   locals_deleted, globals_deleted, weaks_deleted;

  void reset() noexcept;
};

extern ReferenceStats g_reference_stats;
}
/// @endcond


// LocalReferenceAllocator /////////////////////////////////////////////////////////////////////////

inline jobject LocalReferenceAllocator::newReference(jobject original) const {
  internal::dbglog("Local new: %p", original);
  #ifdef FBJNI_DEBUG_REFS
    ++internal::g_reference_stats.locals_created;
  #endif
  auto ref = Environment::current()->NewLocalRef(original);
  FACEBOOK_JNI_THROW_PENDING_EXCEPTION();
  return ref;
}

inline void LocalReferenceAllocator::deleteReference(jobject reference) const noexcept {
  internal::dbglog("Local release: %p", reference);

  if (reference) {
    #ifdef FBJNI_DEBUG_REFS
      ++internal::g_reference_stats.locals_deleted;
    #endif
    assert(verifyReference(reference));
    Environment::current()->DeleteLocalRef(reference);
  }
}

inline bool LocalReferenceAllocator::verifyReference(jobject reference) const noexcept {
  return isObjectRefType(reference, JNILocalRefType);
}


// GlobalReferenceAllocator ////////////////////////////////////////////////////////////////////////

inline jobject GlobalReferenceAllocator::newReference(jobject original) const {
  internal::dbglog("Global new: %p", original);
  #ifdef FBJNI_DEBUG_REFS
    ++internal::g_reference_stats.globals_created;
  #endif
  auto ref = Environment::current()->NewGlobalRef(original);
  FACEBOOK_JNI_THROW_PENDING_EXCEPTION();
  return ref;
}

inline void GlobalReferenceAllocator::deleteReference(jobject reference) const noexcept {
  internal::dbglog("Global release: %p", reference);

  if (reference) {
    #ifdef FBJNI_DEBUG_REFS
      ++internal::g_reference_stats.globals_deleted;
    #endif
    assert(verifyReference(reference));
    Environment::current()->DeleteGlobalRef(reference);
  }
}

inline bool GlobalReferenceAllocator::verifyReference(jobject reference) const noexcept {
  return isObjectRefType(reference, JNIGlobalRefType);
}


// WeakGlobalReferenceAllocator ////////////////////////////////////////////////////////////////////

inline jobject WeakGlobalReferenceAllocator::newReference(jobject original) const {
  internal::dbglog("Weak global new: %p", original);
  #ifdef FBJNI_DEBUG_REFS
    ++internal::g_reference_stats.weaks_created;
  #endif
  auto ref = Environment::current()->NewWeakGlobalRef(original);
  FACEBOOK_JNI_THROW_PENDING_EXCEPTION();
  return ref;
}

inline void WeakGlobalReferenceAllocator::deleteReference(jobject reference) const noexcept {
  internal::dbglog("Weak Global release: %p", reference);

  if (reference) {
    #ifdef FBJNI_DEBUG_REFS
      ++internal::g_reference_stats.weaks_deleted;
    #endif
    assert(verifyReference(reference));
    Environment::current()->DeleteWeakGlobalRef(reference);
  }
}

inline bool WeakGlobalReferenceAllocator::verifyReference(jobject reference) const noexcept {
  return isObjectRefType(reference, JNIWeakGlobalRefType);
}

}}
