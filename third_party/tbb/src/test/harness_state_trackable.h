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

// Declarations for a class that can track operations applied to its objects.
// This header is an optional part of the test harness.

#ifndef tbb_test_harness_state_trackable_H
#define tbb_test_harness_state_trackable_H

#include <cstddef>
#include <map>
#include <tbb/atomic.h>

#include "harness_assert.h"

namespace Harness{
    struct StateTrackableBase {
        enum StateValue {
            ZeroInitialized     = 0,
            DefaultInitialized  = 0xDEFAUL,
            DirectInitialized   = 0xD1111,
            CopyInitialized     = 0xC0314,
            MoveInitialized     = 0xAAAAA,
            Assigned            = 0x11AED,
            MoveAssigned        = 0x22AED,
            MovedFrom           = 0xFFFFF,
            Destroyed           = 0xDEADF00,
            Unspecified         = 0xEEEEE
        };

        class State {
        public:
            State() __TBB_NOEXCEPT(true) : state(Unspecified) {
                assignNewState(Unspecified);
            }
            State(const State& s) : state(Unspecified) {
                assignNewState(s.state);
            }
            State(StateValue s) __TBB_NOEXCEPT(true) : state(Unspecified) {
                assignNewState(s);
            };
            State& operator=(StateValue s) __TBB_NOEXCEPT(true) {
                assignNewState(s);
                return *this;
            }
            operator StateValue() const __TBB_NOEXCEPT(true) { return state; }
        private:
            void assignNewState(StateValue s) __TBB_NOEXCEPT(true);
            StateValue state;
        };
    };

    struct StateTrackableCounters {
        static void reset() {
            counters[StateTrackableBase::ZeroInitialized] = counters[StateTrackableBase::DefaultInitialized] =
                counters[StateTrackableBase::DirectInitialized] = counters[StateTrackableBase::CopyInitialized] =
                counters[StateTrackableBase::MoveInitialized] = counters[StateTrackableBase::Assigned] =
                counters[StateTrackableBase::MoveAssigned] = counters[StateTrackableBase::MovedFrom] =
                counters[StateTrackableBase::Destroyed] = counters[StateTrackableBase::Unspecified] = 0;
        }

        static bool initialize() {
            reset();
            return true;
        }

        typedef std::map<StateTrackableBase::StateValue, tbb::atomic<std::size_t> > counters_t;
        static counters_t counters;
    };

    StateTrackableCounters::counters_t StateTrackableCounters::counters;
    static const bool stateTrackableBaseStateInitialized = StateTrackableCounters::initialize();

    void StateTrackableBase::State::assignNewState(StateValue s) __TBB_NOEXCEPT(true) {
        ASSERT(stateTrackableBaseStateInitialized, "State trackable counters are not initialized");
        ASSERT(s == StateTrackableBase::Unspecified ||
            StateTrackableCounters::counters.find(s) != StateTrackableCounters::counters.end(), "The current state value is unknown");
        ASSERT(state == StateTrackableBase::Unspecified ||
            StateTrackableCounters::counters.find(state) != StateTrackableCounters::counters.end(), "The new state value is unknown");
        state = s;
        ++StateTrackableCounters::counters[state];
    }

    template<bool allow_zero_initialized_state = false>
    struct StateTrackable: StateTrackableBase {
        static const bool is_zero_initialized_state_allowed = allow_zero_initialized_state;
        State state;

        bool is_valid() const {
            return state == DefaultInitialized || state == DirectInitialized || state == CopyInitialized
                || state == MoveInitialized || state == Assigned || state == MoveAssigned || state == MovedFrom
                || (allow_zero_initialized_state && state == ZeroInitialized)
                ;
        }

        StateTrackable (intptr_t)       __TBB_NOEXCEPT(true) : state (DirectInitialized){}
        StateTrackable ()               __TBB_NOEXCEPT(true) : state (DefaultInitialized){}
        StateTrackable (const StateTrackable & src) __TBB_NOEXCEPT(true) {
            ASSERT( src.is_valid(), "bad source for copy" );
            state = CopyInitialized;
        }
    #if __TBB_CPP11_RVALUE_REF_PRESENT
        StateTrackable (StateTrackable && src) __TBB_NOEXCEPT(true) {
            ASSERT( src.is_valid(), "bad source for move?" );
            state = MoveInitialized;
            src.state = MovedFrom;
        }
        StateTrackable & operator=(StateTrackable && src) __TBB_NOEXCEPT(true) {
            ASSERT( src.is_valid(), "bad source for assignment" );
            ASSERT( is_valid(), "assigning to invalid instance?" );

            src.state = MovedFrom;
            state = MoveAssigned;
            return *this;
        }
    #endif
        StateTrackable & operator=(const StateTrackable & src) __TBB_NOEXCEPT(true) {
            ASSERT( src.is_valid(), "bad source for assignment?" );
            ASSERT( is_valid(), "assigning to invalid instance?" );

            state = Assigned;
            return *this;
        }
        ~StateTrackable () __TBB_NOEXCEPT(true) {
            ASSERT( is_valid(), "Calling destructor on invalid instance? (twice destructor call?)" );
            state = Destroyed;
        }
    };
} // Harness
#endif // tbb_test_harness_state_trackable_H
