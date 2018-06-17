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

/** Before making any changes in the implementation, please emulate algorithmic changes
    with SPIN tool using <TBB directory>/tools/spin_models/ReaderWriterMutex.pml.
    There could be some code looking as "can be restructured" but its structure does matter! */

#include "tbb/queuing_rw_mutex.h"
#include "tbb/tbb_machine.h"
#include "tbb/tbb_stddef.h"
#include "tbb/tbb_machine.h"
#include "itt_notify.h"


namespace tbb {

using namespace internal;

//! Flag bits in a state_t that specify information about a locking request.
enum state_t_flags {
    STATE_NONE                   = 0,
    STATE_WRITER                 = 1<<0,
    STATE_READER                 = 1<<1,
    STATE_READER_UNBLOCKNEXT     = 1<<2,
    STATE_ACTIVEREADER           = 1<<3,
    STATE_UPGRADE_REQUESTED      = 1<<4,
    STATE_UPGRADE_WAITING        = 1<<5,
    STATE_UPGRADE_LOSER          = 1<<6,
    STATE_COMBINED_WAITINGREADER = STATE_READER | STATE_READER_UNBLOCKNEXT,
    STATE_COMBINED_READER        = STATE_COMBINED_WAITINGREADER | STATE_ACTIVEREADER,
    STATE_COMBINED_UPGRADING     = STATE_UPGRADE_WAITING | STATE_UPGRADE_LOSER
};

const unsigned char RELEASED = 0;
const unsigned char ACQUIRED = 1;

inline bool queuing_rw_mutex::scoped_lock::try_acquire_internal_lock()
{
    return as_atomic(my_internal_lock).compare_and_swap<tbb::acquire>(ACQUIRED,RELEASED) == RELEASED;
}

inline void queuing_rw_mutex::scoped_lock::acquire_internal_lock()
{
    // Usually, we would use the test-test-and-set idiom here, with exponential backoff.
    // But so far, experiments indicate there is no value in doing so here.
    while( !try_acquire_internal_lock() ) {
        __TBB_Pause(1);
    }
}

inline void queuing_rw_mutex::scoped_lock::release_internal_lock()
{
    __TBB_store_with_release(my_internal_lock,RELEASED);
}

inline void queuing_rw_mutex::scoped_lock::wait_for_release_of_internal_lock()
{
    spin_wait_until_eq(my_internal_lock, RELEASED);
}

inline void queuing_rw_mutex::scoped_lock::unblock_or_wait_on_internal_lock( uintptr_t flag ) {
    if( flag )
        wait_for_release_of_internal_lock();
    else
        release_internal_lock();
}

#if defined(_MSC_VER) && !defined(__INTEL_COMPILER)
    // Workaround for overzealous compiler warnings
    #pragma warning (push)
    #pragma warning (disable: 4311 4312)
#endif

//! A view of a T* with additional functionality for twiddling low-order bits.
template<typename T>
class tricky_atomic_pointer: no_copy {
public:
    typedef typename atomic_selector<sizeof(T*)>::word word;

    template<memory_semantics M>
    static T* fetch_and_add( T* volatile * location, word addend ) {
        return reinterpret_cast<T*>( atomic_traits<sizeof(T*),M>::fetch_and_add(location, addend) );
    }
    template<memory_semantics M>
    static T* fetch_and_store( T* volatile * location, T* value ) {
        return reinterpret_cast<T*>( atomic_traits<sizeof(T*),M>::fetch_and_store(location, reinterpret_cast<word>(value)) );
    }
    template<memory_semantics M>
    static T* compare_and_swap( T* volatile * location, T* value, T* comparand ) {
        return reinterpret_cast<T*>(
                 atomic_traits<sizeof(T*),M>::compare_and_swap(location, reinterpret_cast<word>(value),
                                                              reinterpret_cast<word>(comparand))
               );
    }

    T* & ref;
    tricky_atomic_pointer( T*& original ) : ref(original) {};
    tricky_atomic_pointer( T* volatile & original ) : ref(original) {};
    T* operator&( word operand2 ) const {
        return reinterpret_cast<T*>( reinterpret_cast<word>(ref) & operand2 );
    }
    T* operator|( word operand2 ) const {
        return reinterpret_cast<T*>( reinterpret_cast<word>(ref) | operand2 );
    }
};

typedef tricky_atomic_pointer<queuing_rw_mutex::scoped_lock> tricky_pointer;

#if defined(_MSC_VER) && !defined(__INTEL_COMPILER)
    // Workaround for overzealous compiler warnings
    #pragma warning (pop)
#endif

//! Mask for low order bit of a pointer.
static const tricky_pointer::word FLAG = 0x1;

inline
uintptr_t get_flag( queuing_rw_mutex::scoped_lock* ptr ) {
    return uintptr_t(ptr) & FLAG;
}

//------------------------------------------------------------------------
// Methods of queuing_rw_mutex::scoped_lock
//------------------------------------------------------------------------

//! A method to acquire queuing_rw_mutex lock
void queuing_rw_mutex::scoped_lock::acquire( queuing_rw_mutex& m, bool write )
{
    __TBB_ASSERT( !my_mutex, "scoped_lock is already holding a mutex");

    // Must set all fields before the fetch_and_store, because once the
    // fetch_and_store executes, *this becomes accessible to other threads.
    my_mutex = &m;
    __TBB_store_relaxed(my_prev , (scoped_lock*)0);
    __TBB_store_relaxed(my_next , (scoped_lock*)0);
    __TBB_store_relaxed(my_going, 0);
    my_state = state_t(write ? STATE_WRITER : STATE_READER);
    my_internal_lock = RELEASED;

    queuing_rw_mutex::scoped_lock* pred = m.q_tail.fetch_and_store<tbb::release>(this);

    if( write ) {       // Acquiring for write

        if( pred ) {
            ITT_NOTIFY(sync_prepare, my_mutex);
            pred = tricky_pointer(pred) & ~FLAG;
            __TBB_ASSERT( !( uintptr_t(pred) & FLAG ), "use of corrupted pointer!" );
#if TBB_USE_ASSERT
            __TBB_control_consistency_helper(); // on "m.q_tail"
            __TBB_ASSERT( !__TBB_load_relaxed(pred->my_next), "the predecessor has another successor!");
#endif
           __TBB_store_with_release(pred->my_next,this);
            spin_wait_until_eq(my_going, 1);
        }

    } else {            // Acquiring for read
#if DO_ITT_NOTIFY
        bool sync_prepare_done = false;
#endif
        if( pred ) {
            unsigned short pred_state;
            __TBB_ASSERT( !__TBB_load_relaxed(my_prev), "the predecessor is already set" );
            if( uintptr_t(pred) & FLAG ) {
                /* this is only possible if pred is an upgrading reader and it signals us to wait */
                pred_state = STATE_UPGRADE_WAITING;
                pred = tricky_pointer(pred) & ~FLAG;
            } else {
                // Load pred->my_state now, because once pred->my_next becomes
                // non-NULL, we must assume that *pred might be destroyed.
                pred_state = pred->my_state.compare_and_swap<tbb::acquire>(STATE_READER_UNBLOCKNEXT, STATE_READER);
            }
            __TBB_store_relaxed(my_prev, pred);
            __TBB_ASSERT( !( uintptr_t(pred) & FLAG ), "use of corrupted pointer!" );
#if TBB_USE_ASSERT
            __TBB_control_consistency_helper(); // on "m.q_tail"
            __TBB_ASSERT( !__TBB_load_relaxed(pred->my_next), "the predecessor has another successor!");
#endif
           __TBB_store_with_release(pred->my_next,this);
            if( pred_state != STATE_ACTIVEREADER ) {
#if DO_ITT_NOTIFY
                sync_prepare_done = true;
                ITT_NOTIFY(sync_prepare, my_mutex);
#endif
                spin_wait_until_eq(my_going, 1);
            }
        }

        // The protected state must have been acquired here before it can be further released to any other reader(s):
        unsigned short old_state = my_state.compare_and_swap<tbb::acquire>(STATE_ACTIVEREADER, STATE_READER);
        if( old_state!=STATE_READER ) {
#if DO_ITT_NOTIFY
            if( !sync_prepare_done )
                ITT_NOTIFY(sync_prepare, my_mutex);
#endif
            // Failed to become active reader -> need to unblock the next waiting reader first
            __TBB_ASSERT( my_state==STATE_READER_UNBLOCKNEXT, "unexpected state" );
            spin_wait_while_eq(my_next, (scoped_lock*)NULL);
            /* my_state should be changed before unblocking the next otherwise it might finish
               and another thread can get our old state and left blocked */
            my_state = STATE_ACTIVEREADER;
           __TBB_store_with_release(my_next->my_going,1);
        }
    }

    ITT_NOTIFY(sync_acquired, my_mutex);

    // Force acquire so that user's critical section receives correct values
    // from processor that was previously in the user's critical section.
    __TBB_load_with_acquire(my_going);
}

//! A method to acquire queuing_rw_mutex if it is free
bool queuing_rw_mutex::scoped_lock::try_acquire( queuing_rw_mutex& m, bool write )
{
    __TBB_ASSERT( !my_mutex, "scoped_lock is already holding a mutex");

    if( load<relaxed>(m.q_tail) )
        return false; // Someone already took the lock

    // Must set all fields before the fetch_and_store, because once the
    // fetch_and_store executes, *this becomes accessible to other threads.
    __TBB_store_relaxed(my_prev, (scoped_lock*)0);
    __TBB_store_relaxed(my_next, (scoped_lock*)0);
    __TBB_store_relaxed(my_going, 0); // TODO: remove dead assignment?
    my_state = state_t(write ? STATE_WRITER : STATE_ACTIVEREADER);
    my_internal_lock = RELEASED;

    // The CAS must have release semantics, because we are
    // "sending" the fields initialized above to other processors.
    if( m.q_tail.compare_and_swap<tbb::release>(this, NULL) )
        return false; // Someone already took the lock
    // Force acquire so that user's critical section receives correct values
    // from processor that was previously in the user's critical section.
    __TBB_load_with_acquire(my_going);
    my_mutex = &m;
    ITT_NOTIFY(sync_acquired, my_mutex);
    return true;
}

//! A method to release queuing_rw_mutex lock
void queuing_rw_mutex::scoped_lock::release( )
{
    __TBB_ASSERT(my_mutex!=NULL, "no lock acquired");

    ITT_NOTIFY(sync_releasing, my_mutex);

    if( my_state == STATE_WRITER ) { // Acquired for write

        // The logic below is the same as "writerUnlock", but elides
        // "return" from the middle of the routine.
        // In the statement below, acquire semantics of reading my_next is required
        // so that following operations with fields of my_next are safe.
        scoped_lock* n = __TBB_load_with_acquire(my_next);
        if( !n ) {
            if( this == my_mutex->q_tail.compare_and_swap<tbb::release>(NULL, this) ) {
                // this was the only item in the queue, and the queue is now empty.
                goto done;
            }
            spin_wait_while_eq( my_next, (scoped_lock*)NULL );
            n = __TBB_load_with_acquire(my_next);
        }
        __TBB_store_relaxed(n->my_going, 2); // protect next queue node from being destroyed too early
        if( n->my_state==STATE_UPGRADE_WAITING ) {
            // the next waiting for upgrade means this writer was upgraded before.
            acquire_internal_lock();
            queuing_rw_mutex::scoped_lock* tmp = tricky_pointer::fetch_and_store<tbb::release>(&(n->my_prev), NULL);
            n->my_state = STATE_UPGRADE_LOSER;
            __TBB_store_with_release(n->my_going,1);
            unblock_or_wait_on_internal_lock(get_flag(tmp));
        } else {
            __TBB_ASSERT( my_state & (STATE_COMBINED_WAITINGREADER | STATE_WRITER), "unexpected state" );
            __TBB_ASSERT( !( uintptr_t(__TBB_load_relaxed(n->my_prev)) & FLAG ), "use of corrupted pointer!" );
            __TBB_store_relaxed(n->my_prev, (scoped_lock*)0);
            __TBB_store_with_release(n->my_going,1);
        }

    } else { // Acquired for read

        queuing_rw_mutex::scoped_lock *tmp = NULL;
retry:
        // Addition to the original paper: Mark my_prev as in use
        queuing_rw_mutex::scoped_lock *pred = tricky_pointer::fetch_and_add<tbb::acquire>(&my_prev, FLAG);

        if( pred ) {
            if( !(pred->try_acquire_internal_lock()) )
            {
                // Failed to acquire the lock on pred. The predecessor either unlinks or upgrades.
                // In the second case, it could or could not know my "in use" flag - need to check
                tmp = tricky_pointer::compare_and_swap<tbb::release>(&my_prev, pred, tricky_pointer(pred) | FLAG );
                if( !(uintptr_t(tmp) & FLAG) ) {
                    // Wait for the predecessor to change my_prev (e.g. during unlink)
                    spin_wait_while_eq( my_prev, tricky_pointer(pred)|FLAG );
                    // Now owner of pred is waiting for _us_ to release its lock
                    pred->release_internal_lock();
                }
                // else the "in use" flag is back -> the predecessor didn't get it and will release itself; nothing to do

                tmp = NULL;
                goto retry;
            }
            __TBB_ASSERT(pred && pred->my_internal_lock==ACQUIRED, "predecessor's lock is not acquired");
            __TBB_store_relaxed(my_prev, pred);
            acquire_internal_lock();

            __TBB_store_with_release(pred->my_next,static_cast<scoped_lock *>(NULL));

            if( !__TBB_load_relaxed(my_next) && this != my_mutex->q_tail.compare_and_swap<tbb::release>(pred, this) ) {
                spin_wait_while_eq( my_next, (void*)NULL );
            }
            __TBB_ASSERT( !get_flag(__TBB_load_relaxed(my_next)), "use of corrupted pointer" );

            // ensure acquire semantics of reading 'my_next'
            if( scoped_lock *const l_next = __TBB_load_with_acquire(my_next) ) { // I->next != nil, TODO: rename to n after clearing up and adapting the n in the comment two lines below
                // Equivalent to I->next->prev = I->prev but protected against (prev[n]&FLAG)!=0
                tmp = tricky_pointer::fetch_and_store<tbb::release>(&(l_next->my_prev), pred);
                // I->prev->next = I->next;
                __TBB_ASSERT(__TBB_load_relaxed(my_prev)==pred, NULL);
                __TBB_store_with_release(pred->my_next, my_next);
            }
            // Safe to release in the order opposite to acquiring which makes the code simpler
            pred->release_internal_lock();

        } else { // No predecessor when we looked
            acquire_internal_lock();  // "exclusiveLock(&I->EL)"
            scoped_lock* n = __TBB_load_with_acquire(my_next);
            if( !n ) {
                if( this != my_mutex->q_tail.compare_and_swap<tbb::release>(NULL, this) ) {
                    spin_wait_while_eq( my_next, (scoped_lock*)NULL );
                    n = __TBB_load_relaxed(my_next);
                } else {
                    goto unlock_self;
                }
            }
            __TBB_store_relaxed(n->my_going, 2); // protect next queue node from being destroyed too early
            tmp = tricky_pointer::fetch_and_store<tbb::release>(&(n->my_prev), NULL);
            __TBB_store_with_release(n->my_going,1);
        }
unlock_self:
        unblock_or_wait_on_internal_lock(get_flag(tmp));
    }
done:
    spin_wait_while_eq( my_going, 2 );

    initialize();
}

bool queuing_rw_mutex::scoped_lock::downgrade_to_reader()
{
    __TBB_ASSERT( my_state==STATE_WRITER, "no sense to downgrade a reader" );

    ITT_NOTIFY(sync_releasing, my_mutex);
    my_state = STATE_READER;
    if( ! __TBB_load_relaxed(my_next) ) {
        // the following load of q_tail must not be reordered with setting STATE_READER above
        if( this==my_mutex->q_tail.load<full_fence>() ) {
            unsigned short old_state = my_state.compare_and_swap<tbb::release>(STATE_ACTIVEREADER, STATE_READER);
            if( old_state==STATE_READER )
                return true; // Downgrade completed
        }
        /* wait for the next to register */
        spin_wait_while_eq( my_next, (void*)NULL );
    }
    scoped_lock *const n = __TBB_load_with_acquire(my_next);
    __TBB_ASSERT( n, "still no successor at this point!" );
    if( n->my_state & STATE_COMBINED_WAITINGREADER )
        __TBB_store_with_release(n->my_going,1);
    else if( n->my_state==STATE_UPGRADE_WAITING )
        // the next waiting for upgrade means this writer was upgraded before.
        n->my_state = STATE_UPGRADE_LOSER;
    my_state = STATE_ACTIVEREADER;
    return true;
}

bool queuing_rw_mutex::scoped_lock::upgrade_to_writer()
{
    __TBB_ASSERT( my_state==STATE_ACTIVEREADER, "only active reader can be upgraded" );

    queuing_rw_mutex::scoped_lock * tmp;
    queuing_rw_mutex::scoped_lock * me = this;

    ITT_NOTIFY(sync_releasing, my_mutex);
    my_state = STATE_UPGRADE_REQUESTED;
requested:
    __TBB_ASSERT( !(uintptr_t(__TBB_load_relaxed(my_next)) & FLAG), "use of corrupted pointer!" );
    acquire_internal_lock();
    if( this != my_mutex->q_tail.compare_and_swap<tbb::release>(tricky_pointer(me)|FLAG, this) ) {
        spin_wait_while_eq( my_next, (void*)NULL );
        queuing_rw_mutex::scoped_lock * n;
        n = tricky_pointer::fetch_and_add<tbb::acquire>(&my_next, FLAG);
        unsigned short n_state = n->my_state;
        /* the next reader can be blocked by our state. the best thing to do is to unblock it */
        if( n_state & STATE_COMBINED_WAITINGREADER )
            __TBB_store_with_release(n->my_going,1);
        tmp = tricky_pointer::fetch_and_store<tbb::release>(&(n->my_prev), this);
        unblock_or_wait_on_internal_lock(get_flag(tmp));
        if( n_state & (STATE_COMBINED_READER | STATE_UPGRADE_REQUESTED) ) {
            // save n|FLAG for simplicity of following comparisons
            tmp = tricky_pointer(n)|FLAG;
            for( atomic_backoff b; __TBB_load_relaxed(my_next)==tmp; b.pause() ) {
                if( my_state & STATE_COMBINED_UPGRADING ) {
                    if( __TBB_load_with_acquire(my_next)==tmp )
                        __TBB_store_relaxed(my_next, n);
                    goto waiting;
                }
            }
            __TBB_ASSERT(__TBB_load_relaxed(my_next) != (tricky_pointer(n)|FLAG), NULL);
            goto requested;
        } else {
            __TBB_ASSERT( n_state & (STATE_WRITER | STATE_UPGRADE_WAITING), "unexpected state");
            __TBB_ASSERT( (tricky_pointer(n)|FLAG) == __TBB_load_relaxed(my_next), NULL);
            __TBB_store_relaxed(my_next, n);
        }
    } else {
        /* We are in the tail; whoever comes next is blocked by q_tail&FLAG */
        release_internal_lock();
    } // if( this != my_mutex->q_tail... )
    my_state.compare_and_swap<tbb::acquire>(STATE_UPGRADE_WAITING, STATE_UPGRADE_REQUESTED);

waiting:
    __TBB_ASSERT( !( intptr_t(__TBB_load_relaxed(my_next)) & FLAG ), "use of corrupted pointer!" );
    __TBB_ASSERT( my_state & STATE_COMBINED_UPGRADING, "wrong state at upgrade waiting_retry" );
    __TBB_ASSERT( me==this, NULL );
    ITT_NOTIFY(sync_prepare, my_mutex);
    /* if no one was blocked by the "corrupted" q_tail, turn it back */
    my_mutex->q_tail.compare_and_swap<tbb::release>( this, tricky_pointer(me)|FLAG );
    queuing_rw_mutex::scoped_lock * pred;
    pred = tricky_pointer::fetch_and_add<tbb::acquire>(&my_prev, FLAG);
    if( pred ) {
        bool success = pred->try_acquire_internal_lock();
        pred->my_state.compare_and_swap<tbb::release>(STATE_UPGRADE_WAITING, STATE_UPGRADE_REQUESTED);
        if( !success ) {
            tmp = tricky_pointer::compare_and_swap<tbb::release>(&my_prev, pred, tricky_pointer(pred)|FLAG );
            if( uintptr_t(tmp) & FLAG ) {
                spin_wait_while_eq(my_prev, pred);
                pred = __TBB_load_relaxed(my_prev);
            } else {
                spin_wait_while_eq( my_prev, tricky_pointer(pred)|FLAG );
                pred->release_internal_lock();
            }
        } else {
            __TBB_store_relaxed(my_prev, pred);
            pred->release_internal_lock();
            spin_wait_while_eq(my_prev, pred);
            pred = __TBB_load_relaxed(my_prev);
        }
        if( pred )
            goto waiting;
    } else {
        // restore the corrupted my_prev field for possible further use (e.g. if downgrade back to reader)
        __TBB_store_relaxed(my_prev, pred);
    }
    __TBB_ASSERT( !pred && !__TBB_load_relaxed(my_prev), NULL );

    // additional lifetime issue prevention checks
    // wait for the successor to finish working with my fields
    wait_for_release_of_internal_lock();
    // now wait for the predecessor to finish working with my fields
    spin_wait_while_eq( my_going, 2 );

    // Acquire critical section indirectly from previous owner or directly from predecessor (TODO: not clear).
    __TBB_control_consistency_helper(); // on either "my_mutex->q_tail" or "my_going" (TODO: not clear)

    bool result = ( my_state != STATE_UPGRADE_LOSER );
    my_state = STATE_WRITER;
    __TBB_store_relaxed(my_going, 1);

    ITT_NOTIFY(sync_acquired, my_mutex);
    return result;
}

void queuing_rw_mutex::internal_construct() {
    ITT_SYNC_CREATE(this, _T("tbb::queuing_rw_mutex"), _T(""));
}

} // namespace tbb
