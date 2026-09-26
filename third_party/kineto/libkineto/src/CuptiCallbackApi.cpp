/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// TODO(T90238193)
// @lint-ignore-every CLANGTIDY facebook-hte-RelativeInclude
#include "CuptiCallbackApi.h"
#include "CuptiActivityApi.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <mutex>

#include "DeviceUtil.h"
#include "Logger.h"

namespace KINETO_NAMESPACE {

// limit on number of handles per callback type
constexpr size_t MAX_CB_FNS_PER_CB = 8;

// Use this value in enabledCallbacks_ set, when all cbids in a domain
// is enabled, not a specific cbid.
constexpr uint32_t MAX_CUPTI_CALLBACK_ID_ALL = 0xffffffff;

/* Callback Table :
 *  Overall goal of the design is to optimize the lookup of function
 *  pointers. The table is structured at two levels and the leaf
 *  elements in the table are std::list to enable fast access/inserts/deletes
 *
 *   <callback domain0> |
 *                     -> cb id 0 -> std::list of callbacks
 *                     ...
 *                     -> cb id n -> std::list of callbacks
 *   <callback domain1> |
 *                    ...
 *  CallbackTable is the finaly table type above
 *  See type declrartions in header file.
 */

/* callback_switchboard : is the global callback handler we register
 *  with CUPTI. The goal is to make it as efficient as possible
 *  to re-direct to the registered callback(s).
 *
 *  Few things to care about :
 *   a) use if/then switches rather than map/hash structures
 *   b) avoid dynamic memory allocations
 *   c) be aware of locking overheads
 */
static void CUPTIAPI callback_switchboard(
    void* /* unused */,
    CUpti_CallbackDomain domain,
    CUpti_CallbackId cbid,
    const CUpti_CallbackData* cbInfo) {
  // __callback_switchboard acquires a reader lock
  // on the callback list
  CuptiCallbackApi::singleton().__callback_switchboard(domain, cbid, cbInfo);
}

void CuptiCallbackApi::__callback_switchboard(
    CUpti_CallbackDomain domain,
    CUpti_CallbackId cbid,
    const CUpti_CallbackData* cbInfo) {
  LOG(INFO) << "Callback: domain = " << domain << ", cbid = " << cbid;
  CallbackList* cblist = nullptr;

  switch (domain) {
    // add the fastest path for kernel launch callbacks
    // as these are the most frequent ones
    case CUPTI_CB_DOMAIN_RUNTIME_API:
      switch (cbid) {
        case CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000:
          cblist = &callbacks_.runtime[domainIndex(
              CuptiCallBackID::CUDA_LAUNCH_KERNEL,
              CuptiCallBackID::__RUNTIME_CB_DOMAIN_START)];
          break;
#if defined(CUDA_VERSION) && (CUDA_VERSION >= 11080)
        case CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernelExC_v11060:
          cblist = &callbacks_.runtime[domainIndex(
              CuptiCallBackID::CUDA_LAUNCH_KERNEL_EXC,
              CuptiCallBackID::__RUNTIME_CB_DOMAIN_START)];
          break;
#endif
        default:
          break;
      }
      // This is required to teardown cupti after profiling to prevent QPS
      // slowdown.
      if (CuptiActivityApi::singleton().teardownCupti_) {
        if (cbInfo->callbackSite == CUPTI_API_EXIT) {
          LOG(INFO) << "  Calling cuptiFinalize in exit callsite";
          // Teardown CUPTI calling cuptiFinalize()
          CUPTI_CALL(cuptiUnsubscribe(subscriber_));
          CUPTI_CALL(cuptiFinalize());
          initSuccess_ = false;
          subscriber_ = nullptr;
          CuptiActivityApi::singleton().teardownCupti_ = 0;
          CuptiActivityApi::singleton().finalizeCond_.notify_all();
          return;
        }
      }
      break;

    case CUPTI_CB_DOMAIN_RESOURCE:
      switch (cbid) {
        case CUPTI_CBID_RESOURCE_CONTEXT_CREATED:
          cblist = &callbacks_.resource[domainIndex(
              CuptiCallBackID::RESOURCE_CONTEXT_CREATED,
              CuptiCallBackID::__RESOURCE_CB_DOMAIN_START)];
          break;
        case CUPTI_CBID_RESOURCE_CONTEXT_DESTROY_STARTING:
          cblist = &callbacks_.resource[domainIndex(
              CuptiCallBackID::RESOURCE_CONTEXT_DESTROYED,
              CuptiCallBackID::__RESOURCE_CB_DOMAIN_START)];
          break;
        default:
          break;
      }
      break;

    default:
      return;
  }

  // ignore callbacks that are not handled
  if (cblist == nullptr) {
    return;
  }

  // make a copy of the callback list so we avoid holding lock
  // in common case this should be just one func pointer copy
  std::array<CuptiCallbackFn, MAX_CB_FNS_PER_CB> callbacks;
  size_t num_cbs = 0;
  {
    ReaderLockGuard rl(callbackLock_);
    size_t i = 0;
    for (auto it = cblist->begin();
         it != cblist->end() && i < MAX_CB_FNS_PER_CB;
         it++, i++) {
      callbacks[i] = *it;
    }
    num_cbs = i;
  }

  for (size_t i = 0; i < num_cbs; i++) {
    auto fn = callbacks[i];
    fn(domain, cbid, cbInfo);
  }
}

CuptiCallbackApi& CuptiCallbackApi::singleton() {
  static auto* instance = new CuptiCallbackApi();
  return *instance;
}

void CuptiCallbackApi::initCallbackApi() {
  lastCuptiStatus_ = CUPTI_ERROR_UNKNOWN;
  lastCuptiStatus_ = CUPTI_CALL_NOWARN(cuptiSubscribe(
      &subscriber_,
      reinterpret_cast<CUpti_CallbackFunc>(callback_switchboard),
      nullptr));

  // TODO: Remove temporarily to work around static initialization order issue
  // betweent this and GLOG.
  // if (lastCuptiStatus_ != CUPTI_SUCCESS) {
  //   LOG(INFO) << "Failed cuptiSubscribe, status: " << lastCuptiStatus_;
  // }

  initSuccess_ = (lastCuptiStatus_ == CUPTI_SUCCESS);
}

CuptiCallbackApi::CallbackList* CuptiCallbackApi::CallbackTable::lookup(
    CUpti_CallbackDomain domain,
    CuptiCallBackID cbid) {
  size_t idx;

  switch (domain) {
    case CUPTI_CB_DOMAIN_RESOURCE:
      assert(cbid >= CuptiCallBackID::__RESOURCE_CB_DOMAIN_START);
      assert(cbid < CuptiCallBackID::__RESOURCE_CB_DOMAIN_END);
      idx = domainIndex(cbid, CuptiCallBackID::__RESOURCE_CB_DOMAIN_START);
      return &resource.at(idx);

    case CUPTI_CB_DOMAIN_RUNTIME_API:
      assert(cbid >= CuptiCallBackID::__RUNTIME_CB_DOMAIN_START);
      assert(cbid < CuptiCallBackID::__RUNTIME_CB_DOMAIN_END);
      idx = domainIndex(cbid, CuptiCallBackID::__RUNTIME_CB_DOMAIN_START);
      return &runtime.at(idx);

    default:
      LOG(WARNING) << " Unsupported callback domain : " << domain;
      return nullptr;
  }
}

bool CuptiCallbackApi::registerCallback(
    CUpti_CallbackDomain domain,
    CuptiCallBackID cbid,
    CuptiCallbackFn cbfn) {
  CallbackList* cblist = callbacks_.lookup(domain, cbid);

  if (!cblist) {
    LOG(WARNING) << "Could not register callback -- domain = " << domain
                 << " callback id = " << static_cast<int>(cbid);
    return false;
  }

  // avoid duplicates
  auto it = std::ranges::find(*cblist, cbfn);
  if (it != cblist->end()) {
    LOG(WARNING) << "Adding duplicate callback -- domain = " << domain
                 << " callback id = " << static_cast<int>(cbid);
    return true;
  }

  if (cblist->size() == MAX_CB_FNS_PER_CB) {
    LOG(WARNING) << "Already registered max callback -- domain = " << domain
                 << " callback id = " << static_cast<int>(cbid);
  }

  WriteLockGuard wl(callbackLock_);
  cblist->push_back(cbfn);
  return true;
}

bool CuptiCallbackApi::deleteCallback(
    CUpti_CallbackDomain domain,
    CuptiCallBackID cbid,
    CuptiCallbackFn cbfn) {
  CallbackList* cblist = callbacks_.lookup(domain, cbid);
  if (!cblist) {
    LOG(WARNING) << "Attempting to remove unsupported callback -- domain = "
                 << domain << " callback id = " << static_cast<int>(cbid);
    return false;
  }

  // Locks are not required here as
  //  https://en.cppreference.com/w/cpp/container/list/erase
  //  "References and iterators to the erased elements are invalidated.
  //   Other references and iterators are not affected."
  auto it = std::ranges::find(*cblist, cbfn);
  if (it == cblist->end()) {
    LOG(WARNING) << "Could not find callback to remove -- domain = " << domain
                 << " callback id = " << static_cast<int>(cbid);
    return false;
  }

  WriteLockGuard wl(callbackLock_);
  cblist->erase(it);
  return true;
}

bool CuptiCallbackApi::enableCallback(
    CUpti_CallbackDomain domain,
    CUpti_CallbackId cbid) {
  if (initSuccess_) {
    lastCuptiStatus_ =
        CUPTI_CALL_NOWARN(cuptiEnableCallback(1, subscriber_, domain, cbid));
    enabledCallbacks_.insert({domain, cbid});
    return (lastCuptiStatus_ == CUPTI_SUCCESS);
  }
  return false;
}

bool CuptiCallbackApi::disableCallback(
    CUpti_CallbackDomain domain,
    CUpti_CallbackId cbid) {
  enabledCallbacks_.erase({domain, cbid});
  if (initSuccess_) {
    lastCuptiStatus_ =
        CUPTI_CALL_NOWARN(cuptiEnableCallback(0, subscriber_, domain, cbid));
    return (lastCuptiStatus_ == CUPTI_SUCCESS);
  }
  return false;
}

bool CuptiCallbackApi::enableCallbackDomain(CUpti_CallbackDomain domain) {
  if (initSuccess_) {
    lastCuptiStatus_ =
        CUPTI_CALL_NOWARN(cuptiEnableDomain(1, subscriber_, domain));
    enabledCallbacks_.insert({domain, MAX_CUPTI_CALLBACK_ID_ALL});
    return (lastCuptiStatus_ == CUPTI_SUCCESS);
  }
  return false;
}

bool CuptiCallbackApi::disableCallbackDomain(CUpti_CallbackDomain domain) {
  enabledCallbacks_.erase({domain, MAX_CUPTI_CALLBACK_ID_ALL});
  if (initSuccess_) {
    lastCuptiStatus_ =
        CUPTI_CALL_NOWARN(cuptiEnableDomain(0, subscriber_, domain));
    return (lastCuptiStatus_ == CUPTI_SUCCESS);
  }
  return false;
}

bool CuptiCallbackApi::reenableCallbacks() {
  if (initSuccess_) {
    for (auto& cbpair : enabledCallbacks_) {
      if (static_cast<uint32_t>(cbpair.second) == MAX_CUPTI_CALLBACK_ID_ALL) {
        lastCuptiStatus_ =
            CUPTI_CALL_NOWARN(cuptiEnableDomain(1, subscriber_, cbpair.first));
      } else {
        lastCuptiStatus_ = CUPTI_CALL_NOWARN(
            cuptiEnableCallback(1, subscriber_, cbpair.first, cbpair.second));
      }
    }
    return (lastCuptiStatus_ == CUPTI_SUCCESS);
  }
  return false;
}

} // namespace KINETO_NAMESPACE
