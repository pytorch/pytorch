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

#include "tbb/tbb_config.h"
#if __TBB_TSX_AVAILABLE
#include "tbb/spin_rw_mutex.h"
#include "tbb/tbb_machine.h"
#include "itt_notify.h"
#include "governor.h"
#include "tbb/atomic.h"

// __TBB_RW_MUTEX_DELAY_TEST shifts the point where flags aborting speculation are
// added to the read-set of the operation.  If 1, will add the test just before
// the transaction is ended; this technique is called lazy subscription.
// CAUTION: due to proven issues of lazy subscription, use of __TBB_RW_MUTEX_DELAY_TEST is discouraged!
#ifndef __TBB_RW_MUTEX_DELAY_TEST
    #define __TBB_RW_MUTEX_DELAY_TEST 0
#endif

#if defined(_MSC_VER) && defined(_Wp64)
    // Workaround for overzealous compiler warnings in /Wp64 mode
    #pragma warning (disable: 4244)
#endif

namespace tbb {

namespace interface8 {
namespace internal {

// abort code for mutexes that detect a conflict with another thread.
// value is hexadecimal
enum {
    speculation_transaction_aborted = 0x01,
    speculation_can_retry           = 0x02,
    speculation_memadd_conflict     = 0x04,
    speculation_buffer_overflow     = 0x08,
    speculation_breakpoint_hit      = 0x10,
    speculation_nested_abort        = 0x20,
    speculation_xabort_mask         = 0xFF000000,
    speculation_xabort_shift        = 24,
    speculation_retry               = speculation_transaction_aborted
                                      | speculation_can_retry
                                      | speculation_memadd_conflict
};

// maximum number of times to retry
// TODO: experiment on retry values.
static const int retry_threshold_read = 10;
static const int retry_threshold_write = 10;

//! Release speculative mutex
void x86_rtm_rw_mutex::internal_release(x86_rtm_rw_mutex::scoped_lock& s) {
    switch(s.transaction_state) {
    case RTM_transacting_writer:
    case RTM_transacting_reader:
        {
            __TBB_ASSERT(__TBB_machine_is_in_transaction(), "transaction_state && not speculating");
#if __TBB_RW_MUTEX_DELAY_TEST
            if(s.transaction_state == RTM_transacting_reader) {
                if(this->w_flag) __TBB_machine_transaction_conflict_abort();
            } else {
                if(this->state) __TBB_machine_transaction_conflict_abort();
            }
#endif
            __TBB_machine_end_transaction();
            s.my_scoped_lock.internal_set_mutex(NULL);
        }
        break;
    case RTM_real_reader:
        __TBB_ASSERT(!this->w_flag, "w_flag set but read lock acquired");
        s.my_scoped_lock.release();
        break;
    case RTM_real_writer:
        __TBB_ASSERT(this->w_flag, "w_flag unset but write lock acquired");
        this->w_flag = false;
        s.my_scoped_lock.release();
        break;
    case RTM_not_in_mutex:
        __TBB_ASSERT(false, "RTM_not_in_mutex, but in release");
    default:
        __TBB_ASSERT(false, "invalid transaction_state");
    }
    s.transaction_state = RTM_not_in_mutex;
}

//! Acquire write lock on the given mutex.
void x86_rtm_rw_mutex::internal_acquire_writer(x86_rtm_rw_mutex::scoped_lock& s, bool only_speculate)
{
    __TBB_ASSERT(s.transaction_state == RTM_not_in_mutex, "scoped_lock already in transaction");
    if(tbb::internal::governor::speculation_enabled()) {
        int num_retries = 0;
        unsigned int abort_code;
        do {
            tbb::internal::atomic_backoff backoff;
            if(this->state) {
                if(only_speculate) return;
                do {
                    backoff.pause();  // test the spin_rw_mutex (real readers or writers)
                } while(this->state);
            }
            // _xbegin returns -1 on success or the abort code, so capture it
            if(( abort_code = __TBB_machine_begin_transaction()) == ~(unsigned int)(0) )
            {
                // started speculation
#if !__TBB_RW_MUTEX_DELAY_TEST
                if(this->state) {  // add spin_rw_mutex to read-set.
                    // reader or writer grabbed the lock, so abort.
                    __TBB_machine_transaction_conflict_abort();
                }
#endif
                s.transaction_state = RTM_transacting_writer;
                s.my_scoped_lock.internal_set_mutex(this);  // need mutex for release()
                return;  // successfully started speculation
            }
            ++num_retries;
        } while( (abort_code & speculation_retry) != 0 && (num_retries < retry_threshold_write) );
    }

    if(only_speculate) return;              // should apply a real try_lock...
    s.my_scoped_lock.acquire(*this, true);  // kill transactional writers
    __TBB_ASSERT(!w_flag, "After acquire for write, w_flag already true");
    w_flag = true;                          // kill transactional readers
    s.transaction_state = RTM_real_writer;
    return;
}

//! Acquire read lock on given mutex.
//  only_speculate : true if we are doing a try_acquire.  If true and we fail to speculate, don't
//     really acquire the lock, return and do a try_acquire on the contained spin_rw_mutex.  If
//     the lock is already held by a writer, just return.
void x86_rtm_rw_mutex::internal_acquire_reader(x86_rtm_rw_mutex::scoped_lock& s, bool only_speculate) {
    __TBB_ASSERT(s.transaction_state == RTM_not_in_mutex, "scoped_lock already in transaction");
    if(tbb::internal::governor::speculation_enabled()) {
        int num_retries = 0;
        unsigned int abort_code;
        do {
            tbb::internal::atomic_backoff backoff;
            // if in try_acquire, and lock is held as writer, don't attempt to speculate.
            if(w_flag) {
                if(only_speculate) return;
                do {
                    backoff.pause();  // test the spin_rw_mutex (real readers or writers)
                } while(w_flag);
            }
            // _xbegin returns -1 on success or the abort code, so capture it
            if((abort_code = __TBB_machine_begin_transaction()) == ~(unsigned int)(0) )
            {
                // started speculation
#if !__TBB_RW_MUTEX_DELAY_TEST
                if(w_flag) {  // add w_flag to read-set.
                    __TBB_machine_transaction_conflict_abort();  // writer grabbed the lock, so abort.
                }
#endif
                s.transaction_state = RTM_transacting_reader;
                s.my_scoped_lock.internal_set_mutex(this);  // need mutex for release()
                return;  // successfully started speculation
            }
            // fallback path
            // retry only if there is any hope of getting into a transaction soon
            // Retry in the following cases (from Section 8.3.5 of Intel(R)
            // Architecture Instruction Set Extensions Programming Reference):
            // 1. abort caused by XABORT instruction (bit 0 of EAX register is set)
            // 2. the transaction may succeed on a retry (bit 1 of EAX register is set)
            // 3. if another logical processor conflicted with a memory address
            //    that was part of the transaction that aborted (bit 2 of EAX register is set)
            // That is, retry if (abort_code & 0x7) is non-zero
            ++num_retries;
        } while( (abort_code & speculation_retry) != 0 && (num_retries < retry_threshold_read) );
    }

    if(only_speculate) return;
    s.my_scoped_lock.acquire( *this, false );
    s.transaction_state = RTM_real_reader;
}

//! Upgrade reader to become a writer.
/** Returns whether the upgrade happened without releasing and re-acquiring the lock */
bool x86_rtm_rw_mutex::internal_upgrade(x86_rtm_rw_mutex::scoped_lock& s)
{
    switch(s.transaction_state) {
    case RTM_real_reader: {
            s.transaction_state = RTM_real_writer;
            bool no_release = s.my_scoped_lock.upgrade_to_writer();
            __TBB_ASSERT(!w_flag, "After upgrade_to_writer, w_flag already true");
            w_flag = true;
            return no_release;
        }
    case RTM_transacting_reader:
#if !__TBB_RW_MUTEX_DELAY_TEST
        if(this->state) {  // add spin_rw_mutex to read-set.
            // Real reader or writer holds the lock; so commit the read and re-acquire for write.
            internal_release(s);
            internal_acquire_writer(s);
            return false;
        } else
#endif
        {
            s.transaction_state = RTM_transacting_writer;
            return true;
        }
    default:
        __TBB_ASSERT(false, "Invalid state for upgrade");
        return false;
    }
}

//! Downgrade writer to a reader.
bool x86_rtm_rw_mutex::internal_downgrade(x86_rtm_rw_mutex::scoped_lock& s) {
    switch(s.transaction_state) {
    case RTM_real_writer:
        s.transaction_state = RTM_real_reader;
        __TBB_ASSERT(w_flag, "Before downgrade_to_reader w_flag not true");
        w_flag = false;
        return s.my_scoped_lock.downgrade_to_reader();
    case RTM_transacting_writer:
#if __TBB_RW_MUTEX_DELAY_TEST
        if(this->state) {  // a reader or writer has acquired mutex for real.
            __TBB_machine_transaction_conflict_abort();
        }
#endif
        s.transaction_state = RTM_transacting_reader;
        return true;
    default:
        __TBB_ASSERT(false, "Invalid state for downgrade");
        return false;
    }
}

//! Try to acquire write lock on the given mutex.
//  There may be reader(s) which acquired the spin_rw_mutex, as well as possibly
//  transactional reader(s).  If this is the case, the acquire will fail, and assigning
//  w_flag will kill the transactors.  So we only assign w_flag if we have successfully
//  acquired the lock.
bool x86_rtm_rw_mutex::internal_try_acquire_writer(x86_rtm_rw_mutex::scoped_lock& s)
{
    internal_acquire_writer(s, /*only_speculate=*/true);
    if(s.transaction_state == RTM_transacting_writer) {
        return true;
    }
    __TBB_ASSERT(s.transaction_state == RTM_not_in_mutex, "Trying to acquire writer which is already allocated");
    // transacting write acquire failed.  try_acquire the real mutex
    bool result = s.my_scoped_lock.try_acquire(*this, true);
    if(result) {
        // only shoot down readers if we're not transacting ourselves
        __TBB_ASSERT(!w_flag, "After try_acquire_writer, w_flag already true");
        w_flag = true;
        s.transaction_state = RTM_real_writer;
    }
    return result;
}

void x86_rtm_rw_mutex::internal_construct() {
    ITT_SYNC_CREATE(this, _T("tbb::x86_rtm_rw_mutex"), _T(""));
}

} // namespace internal
} // namespace interface8
} // namespace tbb

#endif /* __TBB_TSX_AVAILABLE */
