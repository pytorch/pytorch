#include "OpenRegGuard.h"

namespace c10::openreg {

// LITERALINCLUDE START: OPENREG GUARD REGISTRATION
C10_REGISTER_GUARD_IMPL(PrivateUse1, OpenRegGuardImpl);
// LITERALINCLUDE END: OPENREG GUARD REGISTRATION

} // namespace c10::openreg
