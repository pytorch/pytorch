/*
    Copyright (c) 2005-2018 Intel Corporation

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.




*/

// Header guard and namespace names follow rml conventions.

#ifndef __RML_rml_base_H
#define __RML_rml_base_H

#include <cstddef>

#if _WIN32||_WIN64
#include <windows.h>
#endif /* _WIN32||_WIN64 */

#ifdef RML_PURE_VIRTUAL_HANDLER
#define RML_PURE(T) {RML_PURE_VIRTUAL_HANDLER(); return (T)0;}
#else
#define RML_PURE(T) = 0;
#endif

namespace rml {

//! Base class for denying assignment and copy constructor.
class no_copy {
    void operator=( no_copy& );
    no_copy( no_copy& );
public:
    no_copy() {}
};

class server;

class versioned_object {
public:
    //! A version number
    typedef unsigned version_type;
    
    //! Get version of this object
    /** The version number is incremented when a incompatible change is introduced.
        The version number is invariant for the lifetime of the object. */
    virtual version_type version() const RML_PURE(version_type)
};

//! Represents a client's job for an execution context.
/** A job object is constructed by the client.
    Not derived from versioned_object because version is same as for client. */
class job {
    friend class server;

    //! Word for use by server
    /** Typically the server uses it to speed up internal lookup.
        Clients must not modify the word. */
    void* scratch_ptr;
};

//! Information that client provides to server when asking for a server.
/** The instance must endure at least until acknowledge_close_connection is called. */
class client: public versioned_object {
public:
    //! Typedef for convenience of derived classes in other namespaces.
    typedef ::rml::job job;

    //! Index of a job in a job pool
    typedef unsigned size_type;

    //! Maximum number of threads that client can exploit profitably if nothing else is running on the machine.  
    /** The returned value should remain invariant for the lifetime of the connection.  [idempotent] */
    virtual size_type max_job_count() const RML_PURE(size_type)

    //! Minimum stack size for each job.  0 means to use default stack size. [idempotent]
    virtual std::size_t min_stack_size() const RML_PURE(std::size_t)

    //! Server calls this routine when it needs client to create a job object.
    virtual job* create_one_job() RML_PURE(job*)

    //! Acknowledge that all jobs have been cleaned up.
    /** Called by server in response to request_close_connection
        after cleanup(job) has been called for each job. */
    virtual void acknowledge_close_connection() RML_PURE(void)

    enum policy_type {turnaround,throughput};

    //! Inform server of desired policy. [idempotent]
    virtual policy_type policy() const RML_PURE(policy_type)

    //! Inform client that server is done with *this.   
    /** Client should destroy the job.
        Not necessarily called by execution context represented by *this.
        Never called while any other thread is working on the job. */
    virtual void cleanup( job& ) RML_PURE(void)

    // In general, we should not add new virtual methods, because that would 
    // break derived classes.  Think about reserving some vtable slots.  
};

// Information that server provides to client.
// Virtual functions are routines provided by the server for the client to call. 
class server: public versioned_object {
public:
    //! Typedef for convenience of derived classes.
    typedef ::rml::job job;

#if _WIN32||_WIN64
    typedef void* execution_resource_t;
#endif

    //! Request that connection to server be closed.
    /** Causes each job associated with the client to have its cleanup method called,
        possibly by a thread different than the thread that created the job. 
        This method can return before all cleanup methods return. 
        Actions that have to wait after all cleanup methods return should be part of 
        client::acknowledge_close_connection. 
        Pass true as exiting if request_close_connection() is called because exit() is
        called. In that case, it is the client's responsibility to make sure all threads
        are terminated. In all other cases, pass false.  */
    virtual void request_close_connection( bool exiting = false ) = 0;

    //! Called by client thread when it reaches a point where it cannot make progress until other threads do.  
    virtual void yield() = 0;

    //! Called by client to indicate a change in the number of non-RML threads that are running.
    /** This is a performance hint to the RML to adjust how many threads it should let run 
        concurrently.  The delta is the change in the number of non-RML threads that are running.
        For example, a value of 1 means the client has started running another thread, and a value 
        of -1 indicates that the client has blocked or terminated one of its threads. */
    virtual void independent_thread_number_changed( int delta ) = 0;

    //! Default level of concurrency for which RML strives when there are no non-RML threads running.
    /** Normally, the value is the hardware concurrency minus one. 
        The "minus one" accounts for the thread created by main(). */
    virtual unsigned default_concurrency() const = 0;

protected:
    static void*& scratch_ptr( job& j ) {return j.scratch_ptr;}
};

class factory {
public:
    //! status results
    enum status_type {
        st_success=0,
        st_connection_exists,
        st_not_found,
        st_incompatible
    };

    //! Scratch pointer for use by RML.
    void* scratch_ptr;

protected:
    //! Pointer to routine that waits for server to indicate when client can close itself.
    status_type (*my_wait_to_close_routine)( factory& );

public:
    //! Library handle for use by RML.
#if _WIN32||_WIN64
    HMODULE library_handle;
#else
    void* library_handle;
#endif /* _WIN32||_WIN64 */ 

    //! Special marker to keep dll from being unloaded prematurely
    static const std::size_t c_dont_unload = 1;
};

//! Typedef for callback functions to print server info
typedef void (*server_info_callback_t)( void* arg, const char* server_info );

} // namespace rml

#endif /* __RML_rml_base_H */
