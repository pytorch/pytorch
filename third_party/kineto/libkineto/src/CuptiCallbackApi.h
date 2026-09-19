/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cupti.h>
#include <array>
#include <atomic>
#include <list>
#include <mutex>
#include <set>
#include <shared_mutex>

// TODO(T90238193)
// @lint-ignore-every CLANGTIDY facebook-hte-RelativeInclude
#include "CuptiCallbackApiMock.h"

namespace KINETO_NAMESPACE {

using namespace libkineto;

/* CuptiCallbackApi : Provides an abstraction over CUPTI callback
 *  interface. This enables various callback functions to be registered
 *  with this class. The class registers a global callback handler that
 *  redirects to the respective callbacks.
 *
 *  Note: one design choice we made is to only support simple function pointers
 *  in order to speed up the implementation for fast path.
 */

using CuptiCallbackFn = void (*)(
    CUpti_CallbackDomain domain,
    CUpti_CallbackId cbid,
    const CUpti_CallbackData* cbInfo);

class CuptiCallbackApi {
 public:
  /* Global list of supported callback ids
   *  use the class namespace to avoid confusing with CUPTI enums*/
  enum class CuptiCallBackID : int {
    CUDA_LAUNCH_KERNEL = 0,
    // can possibly support more callback ids per domain
    //
    __RUNTIME_CB_DOMAIN_START = CUDA_LAUNCH_KERNEL,
    CUDA_LAUNCH_KERNEL_EXC, // Used in H100

    // Callbacks under Resource CB domain
    RESOURCE_CONTEXT_CREATED,
    RESOURCE_CONTEXT_DESTROYED,

    __RUNTIME_CB_DOMAIN_END = RESOURCE_CONTEXT_CREATED,
    __RESOURCE_CB_DOMAIN_START = RESOURCE_CONTEXT_CREATED,

    __RESOURCE_CB_DOMAIN_END = RESOURCE_CONTEXT_DESTROYED + 1,
  };

  CuptiCallbackApi() = default;
  CuptiCallbackApi(const CuptiCallbackApi&) = delete;
  CuptiCallbackApi& operator=(const CuptiCallbackApi&) = delete;

  static CuptiCallbackApi& singleton();

  void initCallbackApi();

  bool initSuccess() const {
    return initSuccess_;
  }

  CUptiResult getCuptiStatus() const {
    return lastCuptiStatus_;
  }

  CUpti_SubscriberHandle getCuptiSubscriber() const {
    return subscriber_;
  }

  bool registerCallback(
      CUpti_CallbackDomain domain,
      CuptiCallBackID cbid,
      CuptiCallbackFn cbfn);

  // returns false if callback was not found
  bool deleteCallback(
      CUpti_CallbackDomain domain,
      CuptiCallBackID cbid,
      CuptiCallbackFn cbfn);

  // Cupti Callback may be enable for domain and cbid pairs, or domains alone.
  bool enableCallback(CUpti_CallbackDomain domain, CUpti_CallbackId cbid);
  bool disableCallback(CUpti_CallbackDomain domain, CUpti_CallbackId cbid);
  bool enableCallbackDomain(CUpti_CallbackDomain domain);
  bool disableCallbackDomain(CUpti_CallbackDomain domain);
  // Provide this API for when cuptiFinalize is executed, to allow the process
  // to re-enabled all previously running callback subscriptions.
  bool reenableCallbacks();

  // Please do not use this method. This has to be exposed as public
  // so it is accessible from the callback handler
  void __callback_switchboard(
      CUpti_CallbackDomain domain,
      CUpti_CallbackId cbid,
      const CUpti_CallbackData* cbInfo);

 private:
  // For callback table design overview see the .cpp file
  using CallbackList = std::list<CuptiCallbackFn>;

  // Computing the slot of the callback id within its domain's table.
  static constexpr size_t domainIndex(
      CuptiCallBackID cbid,
      CuptiCallBackID domainStart) {
    return static_cast<size_t>(cbid) - static_cast<size_t>(domainStart);
  }

  // level 2 tables sizes are known at compile time. Computed inline rather than
  // via domainIndex(): an inline member function isn't yet usable in a
  // constexpr member initializer within the same class definition.
  constexpr static size_t RUNTIME_CB_DOMAIN_SIZE =
      static_cast<size_t>(CuptiCallBackID::__RUNTIME_CB_DOMAIN_END) -
      static_cast<size_t>(CuptiCallBackID::__RUNTIME_CB_DOMAIN_START);

  constexpr static size_t RESOURCE_CB_DOMAIN_SIZE =
      static_cast<size_t>(CuptiCallBackID::__RESOURCE_CB_DOMAIN_END) -
      static_cast<size_t>(CuptiCallBackID::__RESOURCE_CB_DOMAIN_START);

  // level 1 table is a struct
  struct CallbackTable {
    std::array<CallbackList, RUNTIME_CB_DOMAIN_SIZE> runtime;
    std::array<CallbackList, RESOURCE_CB_DOMAIN_SIZE> resource;

    CallbackList* lookup(CUpti_CallbackDomain domain, CuptiCallBackID cbid);
  };

  CallbackTable callbacks_;
  bool initSuccess_ = false;
  // Record a list of enabled callbacks, so that after teardown, we can
  // re-enable the callbacks that were turned off to clean cupti context. As an
  // implementation detail, cbid == 0xffffffff means enable the domain.
  std::set<std::pair<CUpti_CallbackDomain, CUpti_CallbackId>> enabledCallbacks_;

  // Reader Writer lock types
  using ReaderWriterLock = std::shared_timed_mutex;
  using ReaderLockGuard = std::shared_lock<ReaderWriterLock>;
  using WriteLockGuard = std::unique_lock<ReaderWriterLock>;
  ReaderWriterLock callbackLock_;
  CUptiResult lastCuptiStatus_;
  CUpti_SubscriberHandle subscriber_{nullptr};
};

} // namespace KINETO_NAMESPACE
