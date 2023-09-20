/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <stdlib.h>
#include <cstdint>

#pragma once

namespace torch {
namespace executor {

/// Represents an allocator id returned by track_allocator.
typedef uint32_t AllocatorID;
/// Represents the chain id that will be passed in by the user during
/// event logging.
typedef int32_t ChainID;
/// Represents the debug handle that is generally associated with each
/// op executed in the runtime.
typedef uint32_t DebugHandle;

/// Default id's for chain id and debug handle.
constexpr ChainID kUnsetChainId = -1;
constexpr DebugHandle kUnsetDebugHandle = 0;

/// Different types of delegate debug identifiers that are supported currently.
enum class DelegateDebugIdType {
  /// Default value, indicates that it's not a delegate event.
  kNone,
  /// Indicates a delegate event logged using an integer delegate debug
  /// identifier.
  kInt,
  /// Indicates a delegate event logged using a string delegate debug
  /// identifier i.e. the delegate debug id is a pointer to a string table
  /// managed by the class implementing EventTracer functionality.
  kStr
};

/**
 * This is the struct which should be returned when a profiling event is
 * started. This is used to uniquely identify that profiling event and will be
 * required to be passed into the end_profiling call to signal that the event
 * identified by this struct has completed.
 **/
struct EventTracerEntry {
  /// An event id to uniquely identify this event that was generated during a
  /// call to start the tracking of an event.
  int64_t event_id;
  /// The chain to which this event belongs to.
  ChainID chain_id;
  /// The debug handle corresponding to this event.
  DebugHandle debug_handle;
  /// The time at which this event was started to be tracked.
  uint64_t start_time;
  /// When delegate_event_id_type != DelegateDebugIdType::kNone it indicates
  /// that event_id represents a delegate event. If delegate_event_id_type is:
  /// 1) kInt then event_id contains an integer delegate debug id.
  /// 2) kStr then event_id contains a string table index into a string table
  /// maintained by the class implementing EventTracer functionality that will
  /// give us the string identifier of this delegate event. For more details
  /// refer to the DelegateMappingBuilder library present in
  /// executorch/exir/backend/utils.py.
  DelegateDebugIdType delegate_event_id_type;
};
/**
 * EventTracer is a class that users can inherit and implement to
 * log/serialize/stream etc. the profiling and debugging events that are
 * generated at runtime for a model. An example of this is the ETDump
 * implementation in the SDK codebase that serializes these events to a
 * flatbuffer.
 */
class EventTracer {
 public:
  /**
   * Start a new event block (can consist of profiling and/or debugging events.)
   * identified by this name. A block is conceptually a set of events that we
   * want to group together. e.g. all the events that occur during the call to
   * execute() (i.e. model inference) could be categorized as a block.
   *
   * @param[in] name A human readable identifier for the event block. Users
   * calling this interface do not need to keep the memory pointed to by this
   * pointer around. The string must be copied over into internal memory during
   * this call.
   */
  virtual void create_event_block(const char* name) = 0;

  /**
   * Start the profiling of the event identified by name and debug_handle.
   * The user can pass in a chain_id and debug_handle to this call, or leave
   * them empty (default values) which would then result in the chain_id and
   * debug handle stored within (set by set_chain_debug_handle) this class to be
   * used.
   * @param[in] name Human readable name for the profiling event. Users calling
   * this interface do not need to keep the memory pointed to by this pointer
   * around. The string must be copied over into internal memory during this
   * call.
   * @param[in] chain_id The id of the chain to which this event belongs to. If
   * kUnsetChainId is passed in the chain_id and kUnsetDebugHandle for
   * debug_handle then the values stored in the class internally for these
   * properties will be used.
   * @param[in] debug_handle Debug handle generated ahead-of-time during model
   * compilation.
   *
   * @return Returns an instance of EventTracerEntry which should be passed back
   * into the end_profiling() call.
   */
  virtual EventTracerEntry start_profiling(
      const char* name,
      ChainID chain_id = kUnsetChainId,
      DebugHandle debug_handle = kUnsetDebugHandle) = 0;

  /**
   * Start the profiling of a delegate event. Similar to start_profiling it will
   * return an instance of EventTracerEntry that contains the details of this
   * event.
   *
   * @param[in] name Human readable name for the delegate event. This name has
   * to be the same name that was passed in during the Debug delegate mapping
   * generation in the export/ahead-of-time process. If indices and not names
   * are used by this delegate to identify ops executed in the backend then
   * nullptr can be passed in. Users calling this interface do not need to keep
   * the memory pointed to by this pointer around. The string must be copied
   * over into internal memory during this call.
   * @param[in] delegate_debug_index The id of the delegate event. If string
   * based names are used by this delegate to identify ops executed in the
   * backend then kUnsetDebugHandle should be passed in here.
   */
  virtual EventTracerEntry start_profiling_delegate(
      const char* name,
      DebugHandle delegate_debug_index) = 0;

  /**
   * Signal the end of the delegate profiling event contained in
   * event_tracer_entry. Users also have the option to log some some free-from
   * string based metadata along with this.
   *
   * @param[in] event_tracer_entry The EventTracerEntry returned by a call to
   * start_profiling_delegate().
   * @param[in] metadata Optional free-form metadata associated with the
   * delegate event. This should be a null terminated ASCII string. Users
   * calling this interface do not need to keep the memory pointed to by this
   * pointer around. The string must be copied over into internal memory during
   * this call.
   */
  virtual void end_profiling_delegate(
      EventTracerEntry event_tracer_entry,
      const char* metadata = nullptr) = 0;

  /**
   * Some delegates get access to the profiling details only after the complete
   * graph has been executed. This interface is to support such use cases. It
   * can be called in a loop etc. to log any number of profiling events that are
   * part of this delegate.
   *
   * @param[in] name Human readable name for the delegate event. This name has
   * to be the same name that was passed in during the Debug delegate mapping
   * generation in the export/ahead-of-time process. If indices and not names
   * are used by this delegate to identify ops executed in the backend then
   * nullptr can be passed in. Users calling this interface do not need to keep
   * the memory pointed to by this pointer around. The string must be copied
   * over into internal memory during this call.
   * @param[in] delegate_debug_index The id of the delegate event. If string
   * based names are used by this delegate to identify ops executed in the
   * backend then kUnsetDebugHandle should be passed in here.
   * @param[in] start_time The timestamp when the delegate event started.
   * @param[in] end_time The timestamp when the delegate event finished.
   * @param[in] metadata Optional data relevant to the execution that the user
   * wants to log along with this event. Pointer to metadata doesn't need to be
   * valid after the call to this function. This should be a null terminated
   * ASCII string.
   */
  virtual void log_profiling_delegate(
      const char* name,
      DebugHandle delegate_debug_index,
      uint64_t start_time,
      uint64_t end_time,
      const char* metadata = nullptr) = 0;

  /**
   * End the profiling of the event identified by prof_entry
   *
   * @param[in] prof_entry Value returned by a call to start_profiling
   */
  virtual void end_profiling(EventTracerEntry prof_entry) = 0;

  /**
   * Track this allocation done via a MemoryAllocator which had profiling
   * enabled on it.
   *
   * @param[in] id Allocator id generated by a call to track_allocator.
   * @param[in] size The size of the allocation done, in bytes.
   */
  virtual void track_allocation(AllocatorID id, size_t size) = 0;

  /**
   * Generate an allocator id for this memory allocator that will be used in the
   * future to identify all the allocations done by this allocator.
   *
   * @param[in] name Human readable name for the allocator. Users calling
   * this interface do not need to keep the memory pointed to by this pointer
   * around. The string should be copied over into internal memory during this
   * call.
   *
   * @return Identifier to uniquely identify this allocator.
   */
  virtual AllocatorID track_allocator(const char* name) = 0;

  /**
   * Helper function to set the chain id ands debug handle. Users have two
   * options, the first is that they can directly pass in the chain id and debug
   * handle to start_profiling or they can explicitly set them through this
   * helper before calling start_profiling.
   *
   * The reason this helper exists is to
   * solve a specific problem. We want to do profiling logging inside the
   * codegen layer which calls the kernels. The problem though is that the
   * codegen layer doesn't have access to these ids when calling
   * start_profiling.
   *
   * Users should ideally use these within a RAII scope interface to make sure
   * that these values are unset after the end_profiling call. If non-default
   * values are passed into the start_profiling call they will always be given
   * precedence over the values set by this interface.
   *
   * So what we do is call this helper in method.cpp before
   * we hit the codegen layer and in the codegen layer we do a start_profiling
   * call without passing in a chain_id or debug_handle. This ensures that the
   * values set via this helper are the ones associated with that call.
   *
   * @param[in] chain_id Chain id of the current instruction being exectuted.
   * @param[in] debug_handle Debug handle of the current instruction being
   * executed. In this context debug handle and instruction id are the same
   * thing.
   */
  void set_chain_debug_handle(ChainID chain_id, DebugHandle debug_handle) {
    chain_id_ = chain_id;
    debug_handle_ = debug_handle;
  }

  ChainID get_current_chain_id() {
    return chain_id_;
  }

  DebugHandle get_current_debug_handle() {
    return debug_handle_;
  }

  virtual ~EventTracer() {}

 protected:
  ChainID chain_id_ = kUnsetChainId;
  DebugHandle debug_handle_ = kUnsetDebugHandle;
};

} // namespace executor
} // namespace torch
