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

/* The API to enable interoperability between Intel(R) Cilk(TM) Plus and
   Intel(R) Threading Building Blocks. */

#ifndef CILK_TBB_INTEROP_H
#define CILK_TBB_INTEROP_H

#ifndef _WIN32
#ifdef IN_CILK_RUNTIME
#define CILK_EXPORT __attribute__((visibility("protected")))
#else
#define CILK_EXPORT /* nothing */
#endif
#else
#ifdef IN_CILK_RUNTIME
#define CILK_EXPORT __declspec(dllexport)
#else
#define CILK_EXPORT __declspec(dllimport)
#endif  // IN_CILK_RUNTIME
#endif // _WIN32

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

/* A return code.  0 indicates success */
typedef int __cilk_tbb_retcode;

enum __cilk_tbb_stack_op {
    CILK_TBB_STACK_ORPHAN, // disconnecting stack from a thread
    CILK_TBB_STACK_ADOPT,  // reconnecting orphaned stack to a trhead
    CILK_TBB_STACK_RELEASE // releasing stack
};

typedef __cilk_tbb_retcode (*__cilk_tbb_pfn_stack_op)(enum __cilk_tbb_stack_op, void* data);

typedef __cilk_tbb_retcode (*__cilk_tbb_pfn_unwatch_stacks)(void *data);

/* Each thunk structure has two pointers: "routine" and "data".
   The caller of the thunk invokes *routine, passing "data" as the void* parameter. */

/* Thunk invoked by Intel Cilk Plus runtime (cilkrts) when it changes the relationship
   between a stack and a thread. It does not matter what stack the thunk runs on.
   The thread (not fiber) on which the thunk runs is important.

   CILK_TBB_STACK_ORPHAN
      The thunk must be invoked on the thread disconnecting itself from the stack.
      Must "happen before" the stack is adopted elsewhere.
   CILK_TBB_STACK_ADOPT
      The thunk must be invoked on the thread adopting the stack.
   CILK_TBB_STACK_RELEASE
      The thunk must be invoked on the thread doing the releasing,
      Must "happen before" the stack is used elsewhere.

   When a non-empty stack is transferred between threads, the first thread must orphan it
   and the second thread must adopt it.

   An empty stack can be transferred similarly, or simply released by the first thread.

   Here is a summary of the actions as transitions on a state machine.

                       watch                                    ORPHAN
                       -->-->                                   -->--
                      /      \                                 /     \
   (freed empty stack)       (TBB sees stack running on thread)      (stack in limbo)
                |     \     /                                  \     /     |
                |      --<--                                    --<--      |
                ^      RELEASE or                              ADOPT       V
                 \     unwatch                                            /
                  \                                                      /
                   --------------------------<---------------------------
                                          RELEASE
*/
struct __cilk_tbb_stack_op_thunk {
    __cilk_tbb_pfn_stack_op routine;
    void* data;                 /* Set by TBB */
};

/* Thunk invoked by TBB when it is no longer interested in watching the stack bound to the current thread. */
struct __cilk_tbb_unwatch_thunk {
    __cilk_tbb_pfn_unwatch_stacks routine;
    void* data;
};

/* Defined by cilkrts, called by TBB.
   Requests that cilkrts invoke __cilk_tbb_stack_op_thunk when it orphans a stack.
   cilkrts sets *u to a thunk that TBB should call when it is no longer interested in watching the stack. */
CILK_EXPORT
__cilk_tbb_retcode __cilkrts_watch_stack(struct __cilk_tbb_unwatch_thunk* u,
                                         struct __cilk_tbb_stack_op_thunk o);

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif  // CILK_TBB_INTEROP_H
