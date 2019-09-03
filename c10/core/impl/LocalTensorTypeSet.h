#include <c10/core/TensorTypeSet.h>

// TLS management for TensorTypeSet
//
// This manages thread-local TensorTypeSet of VALID keys which.  Keys which are
// not in this set, even if they appear in a list of potential valid keys on a
// tensor, are not considered for dispatch.  This is used to, for example, turn
// off autograd after we have handled autograd for a top-level element.  (We
// store this bitmask, and not its inverted one, to save an instruction).

namespace c10 {
namespace impl {

C10_API bool tls_variable_is_enabled();
C10_API void tls_variable_set_enabled(bool enabled);
C10_API TensorTypeSet tls_valid_tensor_type_set();

}} // namespace c10::impl
