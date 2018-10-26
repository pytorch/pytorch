#include <ATen/core/intrusive_ptr.h>

// vtable anchor
c10::intrusive_ptr_target::~intrusive_ptr_target() {
// Disable -Wterminate and -Wexceptions so we're allowed to use assertions
// (i.e. throw exceptions) in a destructor.
// We also have to disable -Wunknown-warning-option and -Wpragmas, because
// some other compilers don't know about -Wterminate or -Wexceptions and
// will show a warning about unknown warning options otherwise.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpragmas"
#pragma GCC diagnostic ignored "-Wunknown-warning-option"
#pragma GCC diagnostic ignored "-Wterminate"
#pragma GCC diagnostic ignored "-Wexceptions"
    AT_ASSERTM(
        refcount_.load() == 0,
        "Tried to destruct an intrusive_ptr_target that still has intrusive_ptr to it");
    AT_ASSERTM(
        weakcount_.load() == 0,
        "Tried to destruct an intrusive_ptr_target that still has weak_intrusive_ptr to it");
#pragma GCC diagnostic pop
}